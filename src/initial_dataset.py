#!/usr/bin/env python3
import argparse
import datetime as dt
import os
from typing import Any, Optional

import pandas as pd
import requests

from src.clients.openmeteo import OpenMeteoClient
from src.clients.opensky import OpenSkyClient
from src.utils.helpers import ensure_dir, retry_request, utc_day_bounds
from src.utils.weather import hourly_payload_to_df, weather_snapshots

# -----------------------------
# Optional .env loading (must happen before reading env vars below)
# -----------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # It's fine if python-dotenv isn't installed; env vars can still be set externally.
    pass

# -----------------------------
# CONFIG (credentials/settings)
# -----------------------------
# Put these in your .env (recommended). The script will read them automatically.

# OpenSky API base (REST)
OPENSKY_API_BASE = os.getenv("OPENSKY_API_BASE", "https://opensky-network.org/api")

# Auth mode:
# - OAuth2 client credentials (preferred): set OPENSKY_CLIENT_ID + OPENSKY_CLIENT_SECRET + OPENSKY_TOKEN_URL
# - OR legacy basic auth (if your account supports it): set OPENSKY_USERNAME + OPENSKY_PASSWORD
OPENSKY_CLIENT_ID = os.getenv("OPENSKY_CLIENT_ID", "")
OPENSKY_CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET", "")
DEFAULT_OPENSKY_TOKEN_URL = "https://opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
_token_url_env = os.getenv("OPENSKY_TOKEN_URL", "").strip()
if _token_url_env.lower().startswith(("http://", "https://")):
    OPENSKY_TOKEN_URL = _token_url_env
else:
    if _token_url_env:
        print("[WARN] OPENSKY_TOKEN_URL is missing a valid scheme. Using the default token endpoint.")
    OPENSKY_TOKEN_URL = DEFAULT_OPENSKY_TOKEN_URL

OPENSKY_ACCESS_TOKEN = os.getenv("OPENSKY_TOKEN", "").strip()

OPENSKY_USERNAME = os.getenv("OPENSKY_USERNAME", "")
OPENSKY_PASSWORD = os.getenv("OPENSKY_PASSWORD", "")

# OurAirports metadata source
OURAIRPORTS_CSV_URL = os.getenv("OURAIRPORTS_CSV_URL", "https://ourairports.com/data/airports.csv")

# Open-Meteo Archive API
OPENMETEO_ARCHIVE_BASE = os.getenv("OPENMETEO_ARCHIVE_BASE", "https://archive-api.open-meteo.com/v1/archive")

# Output/cache
CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "30"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
RETRY_BACKOFF_BASE_SEC = float(os.getenv("RETRY_BACKOFF_BASE_SEC", "1.5"))

# Expected-duration baseline parameters
ROLLING_MEDIAN_WINDOW = int(os.getenv("ROLLING_MEDIAN_WINDOW", "30"))  # previous N flights per route

# Weather variables (hourly)
WEATHER_HOURLY_VARS = os.getenv(
    "WEATHER_HOURLY_VARS",
    "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,visibility,pressure_msl,"
    "wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code",
)

# Start with top 5 Swedish airports (ICAO)
DEFAULT_SWEDEN_TOP_5_ICAO = [
    "ESSA",  # Stockholm Arlanda
    "ESSB",  # Stockholm Bromma
    # "ESGG",  # Göteborg Landvetter
    # "ESMS",  # Malmö
    # "ESNU",  # Umeå
]

# -----------------------------
# Optional .env loading
# -----------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    # It's fine if python-dotenv isn't installed; env vars can still be set externally.
    pass


def load_airports_metadata() -> pd.DataFrame:
    """Load global airports metadata and normalize ICAO codes."""

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    local_csv = os.path.join(root_dir, "airports.csv")
    if os.path.exists(local_csv):
        df = pd.read_csv(local_csv)
    else:
        print(f"[INFO] Downloading OurAirports airports.csv from {OURAIRPORTS_CSV_URL}")
        resp = retry_request(
            requests.Session(),
            "GET",
            OURAIRPORTS_CSV_URL,
            timeout=REQUEST_TIMEOUT_SEC,
            max_retries=MAX_RETRIES,
            retry_backoff_base=RETRY_BACKOFF_BASE_SEC,
        )
        from io import StringIO

        df = pd.read_csv(StringIO(resp.text))

    keep_cols = [
        "ident",
        "iso_country",
        "iata_code",
        "name",
        "municipality",
        "type",
        "scheduled_service",
        "latitude_deg",
        "longitude_deg",
        "elevation_ft",
    ]
    df = df[keep_cols]
    df.rename(columns={"ident": "icao"}, inplace=True)
    df = df[df["icao"].notna()].copy()
    df["icao"] = df["icao"].astype(str).str.upper()
    return df


def load_airports_sweden(airports_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Filter airports to Sweden only."""

    source = airports_df if airports_df is not None else load_airports_metadata()
    df_se = source[source["iso_country"] == "SE"].copy()
    return df_se


def pick_airports(df_se: pd.DataFrame, airport_icaos: Optional[list[str]], include_all_sweden: bool) -> pd.DataFrame:
    if include_all_sweden:
        df = df_se.copy()
        # Prefer those with scheduled service and large/medium
        df = df[df["scheduled_service"] == "yes"]
        df = df[df["type"].isin(["large_airport", "medium_airport"])]
        return df.reset_index(drop=True)

    df = df_se[df_se["icao"].isin(airport_icaos)].copy()
    missing = sorted(set(airport_icaos) - set(df["icao"].tolist()))
    if missing:
        print(f"[WARN] These ICAOs were not found in OurAirports Sweden list (check spelling): {missing}")
    return df.reset_index(drop=True)


# -----------------------------
# Expected duration baseline
# -----------------------------
def compute_expected_durations(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each route (origin,dest) ordered by actual_takeoff_time_utc, compute expected_flight_time_sec
    as rolling median of the *previous* `window` real_flight_time_sec values.

    If insufficient history, fallback to the route median (computed on available history, if any),
    otherwise fallback to global median.
    """
    df = df.sort_values("actual_takeoff_time_utc").reset_index(drop=True)
    global_median = float(df["real_flight_time_sec"].median()) if len(df) else float("nan")

    expected = [None] * len(df)

    # group indices by route
    grouped = df.groupby(["origin_airport_icao", "destination_airport_icao"], sort=False).indices
    for route, idxs in grouped.items():
        idxs_sorted = sorted(idxs, key=lambda i: df.loc[i, "actual_takeoff_time_utc"])
        durations: list[float] = []
        route_all = [float(df.loc[i, "real_flight_time_sec"]) for i in idxs_sorted if not pd.isna(df.loc[i, "real_flight_time_sec"])]
        route_median = float(pd.Series(route_all).median()) if route_all else global_median

        for j, i in enumerate(idxs_sorted):
            if len(durations) == 0:
                expected[i] = route_median
            else:
                # rolling median of previous up to window values
                prev = durations[-window:]
                expected[i] = float(pd.Series(prev).median())

            # update durations with current duration if present
            cur = df.loc[i, "real_flight_time_sec"]
            if not pd.isna(cur):
                durations.append(float(cur))

    df["expected_flight_time_sec"] = expected
    return df


# -----------------------------
# Main dataset builder
# -----------------------------
def build_dataset(
    airports_df: pd.DataFrame,
    airports_meta_df: pd.DataFrame,
    days: int,
    out_csv: str,
    include_arrivals_backfill: bool = True,
) -> None:
    ensure_dir(CACHE_DIR)

    opensky = OpenSkyClient(
        api_base=OPENSKY_API_BASE,
        client_id=OPENSKY_CLIENT_ID,
        client_secret=OPENSKY_CLIENT_SECRET,
        token_url=OPENSKY_TOKEN_URL,
        username=OPENSKY_USERNAME,
        password=OPENSKY_PASSWORD,
        request_timeout=REQUEST_TIMEOUT_SEC,
        max_retries=MAX_RETRIES,
        retry_backoff_base=RETRY_BACKOFF_BASE_SEC,
        access_token=OPENSKY_ACCESS_TOKEN,
    )
    openmeteo = OpenMeteoClient(
        base_url=OPENMETEO_ARCHIVE_BASE,
        hourly_vars=WEATHER_HOURLY_VARS,
        cache_dir=CACHE_DIR,
        request_timeout=REQUEST_TIMEOUT_SEC,
        max_retries=MAX_RETRIES,
        retry_backoff_base=RETRY_BACKOFF_BASE_SEC,
    )

    # Choose date range: yesterday back `days`
    today_utc = dt.datetime.now(dt.timezone.utc).date()
    end_day = today_utc - dt.timedelta(days=1)  # yesterday
    start_day = end_day - dt.timedelta(days=days - 1)

    all_flights: list[dict[str, Any]] = []

    print(f"[INFO] Ingesting flights from {start_day.isoformat()} to {end_day.isoformat()} (UTC days).")
    for _, ap in airports_df.iterrows():
        origin_icao = str(ap["icao"]).upper()
        for n in range(days):
            d = start_day + dt.timedelta(days=n)
            begin_ts, end_ts = utc_day_bounds(d)

            print(f"[INFO] OpenSky departures: {origin_icao} {d.isoformat()}")
            deps = opensky.get_departures(origin_icao, begin_ts, end_ts)

            for f in deps:
                # normalize minimal schema
                all_flights.append(
                    {
                        "icao24": f.get("icao24"),
                        "callsign": (f.get("callsign") or "").strip() if f.get("callsign") else None,
                        "firstSeen": f.get("firstSeen"),
                        "lastSeen": f.get("lastSeen"),
                        "origin_airport_icao": f.get("estDepartureAirport"),
                        "destination_airport_icao": f.get("estArrivalAirport"),
                        "origin_query_airport_icao": origin_icao,
                        "day_utc": d.isoformat(),
                    }
                )

            # Optional: backfill arrivals (useful if you want destination-based data too)
            if include_arrivals_backfill:
                print(f"[INFO] OpenSky arrivals (optional backfill): {origin_icao} {d.isoformat()}")
                arrs = opensky.get_arrivals(origin_icao, begin_ts, end_ts)
                for f in arrs:
                    all_flights.append(
                        {
                            "icao24": f.get("icao24"),
                            "callsign": (f.get("callsign") or "").strip() if f.get("callsign") else None,
                            "firstSeen": f.get("firstSeen"),
                            "lastSeen": f.get("lastSeen"),
                            "origin_airport_icao": f.get("estDepartureAirport"),
                            "destination_airport_icao": f.get("estArrivalAirport"),
                            "arrival_query_airport_icao": origin_icao,
                            "day_utc": d.isoformat(),
                        }
                    )

    if not all_flights:
        raise RuntimeError("No flights returned. Check OpenSky credentials, date range, and airport ICAOs.")

    df = pd.DataFrame(all_flights).drop_duplicates()

    # Clean & validate
    selected_set = set(airports_df["icao"].astype(str).str.upper().tolist())
    df = df[
        df["origin_airport_icao"].isin(selected_set)
        | df["destination_airport_icao"].isin(selected_set)
    ].copy()

    # Basic required fields
    df = df[df["destination_airport_icao"].notna()].copy()
    df = df[df["firstSeen"].notna() & df["lastSeen"].notna()].copy()
    df["firstSeen"] = df["firstSeen"].astype(int)
    df["lastSeen"] = df["lastSeen"].astype(int)
    df = df[df["lastSeen"] > df["firstSeen"]].copy()

    df["actual_takeoff_time_utc"] = df["firstSeen"]
    df["landing_time_utc"] = df["lastSeen"]
    df["real_flight_time_sec"] = df["lastSeen"] - df["firstSeen"]

    # Sanity bounds
    df = df[(df["real_flight_time_sec"] > 10 * 60) & (df["real_flight_time_sec"] < 10 * 3600)].copy()

    # "Expected takeoff time" (OpenSky doesn't provide schedules)
    df["expected_takeoff_time_utc"] = df["actual_takeoff_time_utc"]

    # Compute expected flight time baseline
    df = compute_expected_durations(df, window=ROLLING_MEDIAN_WINDOW)

    # Enrich with airport metadata (origin + destination lat/lon)
    ap_meta = airports_meta_df[["icao", "name", "latitude_deg", "longitude_deg"]].copy()
    ap_meta.columns = ["icao", "airport_name", "lat", "lon"]

    df = df.merge(
        ap_meta.add_prefix("origin_"),
        left_on="origin_airport_icao",
        right_on="origin_icao",
        how="left",
    ).drop(columns=["origin_icao"], errors="ignore")

    df = df.merge(
        ap_meta.add_prefix("dest_"),
        left_on="destination_airport_icao",
        right_on="dest_icao",
        how="left",
    ).drop(columns=["dest_icao"], errors="ignore")

    # Some destinations may be outside Sweden (no meta). We'll still fetch weather by coordinates only if known.
    # For non-Sweden destinations, you can:
    # - either skip weather enrichment
    # - or load global airport metadata (easy extension: download airports.csv and keep all ICAOs)
    #
    # Here: we will attempt weather only where lat/lon exist; otherwise leave nulls.

    # Weather enrichment with caching:
    # We need hourly data for:
    # - origin airport on takeoff day (and maybe previous day for t-30 near midnight)
    # - destination airport on landing day (and maybe previous day)
    #
    # We'll fetch day by day based on each event timestamp.
    def ts_to_date_iso(ts: int) -> str:
        return dt.datetime.fromtimestamp(int(ts), tz=dt.timezone.utc).date().isoformat()

    # Build a weather cache map in-memory to avoid re-reading JSON a lot
    hourly_df_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_hourly_df(icao: str, lat: float, lon: float, date_iso_str: str) -> pd.DataFrame:
        key = (icao, date_iso_str)
        if key in hourly_df_cache:
            return hourly_df_cache[key]
        payload = openmeteo.ensure_weather_for_day(icao, float(lat), float(lon), date_iso_str)
        hdf = hourly_payload_to_df(payload)
        hourly_df_cache[key] = hdf
        return hdf

    # Enrich each flight row
    enriched_rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        origin_icao = row.get("origin_airport_icao")
        dest_icao = row.get("destination_airport_icao")

        takeoff_ts = int(row["actual_takeoff_time_utc"])
        landing_ts = int(row["landing_time_utc"])

        takeoff_day = ts_to_date_iso(takeoff_ts)
        landing_day = ts_to_date_iso(landing_ts)

        # For -30 minutes near midnight, fetch previous day as well and combine if needed.
        takeoff_prev_day = (dt.date.fromisoformat(takeoff_day) - dt.timedelta(days=1)).isoformat()
        landing_prev_day = (dt.date.fromisoformat(landing_day) - dt.timedelta(days=1)).isoformat()

        out = dict(row)

        # Origin weather (requires origin lat/lon)
        if pd.notna(row.get("origin_lat")) and pd.notna(row.get("origin_lon")):
            o_lat = float(row["origin_lat"])
            o_lon = float(row["origin_lon"])

            # merge hourly data for prev day + day (so interpolation has coverage)
            h0 = get_hourly_df(origin_icao, o_lat, o_lon, takeoff_prev_day)
            h1 = get_hourly_df(origin_icao, o_lat, o_lon, takeoff_day)
            oh = pd.concat([h0, h1], ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")

            out.update(weather_snapshots(oh, takeoff_ts, prefix="wx_origin"))
        else:
            # create empty keys if you prefer fixed schema; leaving missing is also fine.
            pass

        # Destination weather (only if we have destination coords)
        if pd.notna(row.get("dest_lat")) and pd.notna(row.get("dest_lon")):
            d_lat = float(row["dest_lat"])
            d_lon = float(row["dest_lon"])

            h0 = get_hourly_df(dest_icao, d_lat, d_lon, landing_prev_day)
            h1 = get_hourly_df(dest_icao, d_lat, d_lon, landing_day)
            dh = pd.concat([h0, h1], ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")

            out.update(weather_snapshots(dh, landing_ts, prefix="wx_dest"))
        else:
            pass

        enriched_rows.append(out)

    out_df = pd.DataFrame(enriched_rows)

    # Reorder columns (core first)
    core_cols = [
        "icao24",
        "callsign",
        "origin_airport_icao",
        "destination_airport_icao",
        "expected_takeoff_time_utc",
        "actual_takeoff_time_utc",
        "expected_flight_time_sec",
        "real_flight_time_sec",
        "landing_time_utc",
        "day_utc",
        "origin_airport_name",
        "origin_lat",
        "origin_lon",
        "dest_airport_name",
        "dest_lat",
        "dest_lon",
    ]
    remaining = [c for c in out_df.columns if c not in core_cols]
    out_df = out_df[core_cols + remaining]

    ensure_dir(os.path.dirname(out_csv) or ".")
    out_df.to_csv(out_csv, index=False)
    print(f"[DONE] Wrote {len(out_df):,} rows to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sweden flights + weather dataset (CSV).")
    parser.add_argument("--days", type=int, default=2, help="Number of past UTC days to ingest (ending yesterday).")
    parser.add_argument("--out", type=str, default="sweden_flights.csv", help="Output CSV path.")
    parser.add_argument(
        "--airports",
        type=str,
        default=",".join(DEFAULT_SWEDEN_TOP_5_ICAO),
        help="Comma-separated ICAO codes to include (default: top 10).",
    )
    parser.add_argument(
        "--all-sweden",
        action="store_true",
        help="Use all Sweden airports with scheduled_service=yes and type large/medium from OurAirports.",
    )
    parser.add_argument(
        "--no-arrivals-backfill",
        action="store_true",
        help="Disable optional /flights/arrival calls (fewer API calls).",
    )
    args = parser.parse_args()

    # Basic auth sanity message
    if OPENSKY_ACCESS_TOKEN:
        print("[INFO] Using OpenSky bearer token from OPENSKY_TOKEN.")
    elif OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET and OPENSKY_TOKEN_URL:
        print("[INFO] Using OpenSky OAuth2 (client credentials).")
    elif OPENSKY_USERNAME and OPENSKY_PASSWORD:
        print("[INFO] Using OpenSky Basic Auth (username/password).")
    else:
        print("[WARN] No OpenSky credentials found in env. You may hit anonymous limits or fail if auth is required.")

    airports_meta_df = load_airports_metadata()
    airports_df_se = load_airports_sweden(airports_meta_df)
    airport_list = [a.strip().upper() for a in args.airports.split(",")] if args.airports else None
    airports_df = pick_airports(airports_df_se, airport_list, include_all_sweden=bool(args.all_sweden))

    if airports_df.empty:
        raise RuntimeError("Airport selection is empty. Check --airports values or --all-sweden filters.")

    print(f"[INFO] Using {len(airports_df)} airport(s): {airports_df['icao'].tolist()}")

    build_dataset(
        airports_df=airports_df,
        airports_meta_df=airports_meta_df,
        days=int(args.days),
        out_csv=args.out,
        include_arrivals_backfill=not bool(args.no_arrivals_backfill),
    )


if __name__ == "__main__":
    main()
