#!/usr/bin/env python3
"""Batch delay prediction for upcoming flights using the latest trained model."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from src.clients.openmeteo import OpenMeteoClient
from src.utils.weather import hourly_payload_to_df, weather_snapshots

DEFAULT_SCENARIOS = [
    {
        "flight_id": "CDG-ARN-001",
        "origin": {"icao": "LFPG", "lat": 49.0097, "lon": 2.5479},
        "destination": {"icao": "ESSA", "lat": 59.6498, "lon": 17.9238},
        "takeoff_iso": "2025-01-15T08:00:00+00:00",
        "landing_iso": "2025-01-15T10:30:00+00:00",
    }
]

DELAY_BUCKETS = {
    "no delay expected": 300,   # <=5 minutes
    "5-15 minutes": 900,        # 5-15 minutes
}


@dataclass
class FlightScenario:
    flight_id: str
    origin_icao: str
    origin_lat: float
    origin_lon: float
    dest_icao: str
    dest_lat: float
    dest_lon: float
    takeoff_ts: int
    landing_ts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models", help="Directory containing the saved model joblib files.")
    parser.add_argument("--latest-file", default="lastest_model.txt", help="Path to the file with the last model name.")
    parser.add_argument("--scenarios", help="JSON file describing a list of flight scenarios.")
    parser.add_argument("--cache-dir", default="cache", help="Directory for weather cache files.")
    parser.add_argument("--openmeteo-url", default="https://archive-api.open-meteo.com/v1/archive", help="Open-Meteo archive endpoint.")
    parser.add_argument(
        "--hourly-vars",
        default="temperature_2m,relative_humidity_2m,precipitation,cloud_cover,visibility,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code",
        help="Comma-separated Open-Meteo hourly variables.",
    )
    return parser.parse_args()


def load_scenarios(path: Optional[str]) -> List[FlightScenario]:
    payload = DEFAULT_SCENARIOS if not path else json.loads(open(path, "r", encoding="utf-8").read())
    scenarios: List[FlightScenario] = []
    for entry in payload:
        origin = entry["origin"]
        dest = entry["destination"]
        takeoff_ts = iso_to_timestamp(entry["takeoff_iso"])
        landing_ts = iso_to_timestamp(entry["landing_iso"])
        scenarios.append(
            FlightScenario(
                flight_id=entry.get("flight_id", f"{origin['icao']}-{dest['icao']}-{takeoff_ts}"),
                origin_icao=origin["icao"].upper(),
                origin_lat=float(origin["lat"]),
                origin_lon=float(origin["lon"]),
                dest_icao=dest["icao"].upper(),
                dest_lat=float(dest["lat"]),
                dest_lon=float(dest["lon"]),
                takeoff_ts=takeoff_ts,
                landing_ts=landing_ts,
            )
        )
    return scenarios


def iso_to_timestamp(value: str) -> int:
    stamp = dt.datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return int(stamp.timestamp())


def ts_to_iso_day(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date().isoformat()


def load_model(model_dir: str, latest_file: str):
    if not os.path.exists(latest_file):
        raise FileNotFoundError(f"Latest model file not found: {latest_file}")
    model_name = open(latest_file, "r", encoding="utf-8").read().strip()
    if not model_name:
        raise RuntimeError("latest_model.txt is empty. Train and save a model first.")
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    pipeline = joblib.load(model_path)
    print(f"[MODEL] Loaded {model_name} from {model_path}")
    return pipeline, model_name


def ensure_feature_columns(pipeline, features: Dict[str, Any]) -> Dict[str, Any]:
    feature_names = []
    if hasattr(pipeline, "feature_names_in_"):
        feature_names = list(pipeline.feature_names_in_)
    elif hasattr(pipeline.named_steps["imputer"], "feature_names_in_"):
        feature_names = list(pipeline.named_steps["imputer"].feature_names_in_)
    if not feature_names:
        feature_names = [k for k in features.keys() if k.startswith("wx_")]
    return {name: features.get(name, math.nan) for name in feature_names}


def fetch_weather_snapshots(client: OpenMeteoClient, icao: str, lat: float, lon: float, event_ts: int) -> Dict[str, Any]:
    day = ts_to_iso_day(event_ts)
    prev_day = (dt.date.fromisoformat(day) - dt.timedelta(days=1)).isoformat()
    payload_prev = client.ensure_weather_for_day(icao, lat, lon, prev_day)
    payload_day = client.ensure_weather_for_day(icao, lat, lon, day)
    hourly_prev = hourly_payload_to_df(payload_prev)
    hourly_day = hourly_payload_to_df(payload_day)
    hourly = (
        pd.concat([hourly_prev, hourly_day], ignore_index=True)
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return weather_snapshots(hourly, event_ts, prefix="wx")


def build_feature_row(client: OpenMeteoClient, scenario: FlightScenario) -> Dict[str, Any]:
    origin_features = fetch_weather_snapshots(client, scenario.origin_icao, scenario.origin_lat, scenario.origin_lon, scenario.takeoff_ts)
    dest_features = fetch_weather_snapshots(client, scenario.dest_icao, scenario.dest_lat, scenario.dest_lon, scenario.landing_ts)
    renamed_origin = {f"wx_origin{key[2:]}": value for key, value in origin_features.items()}
    renamed_dest = {f"wx_dest{key[2:]}": value for key, value in dest_features.items()}
    row = {**renamed_origin, **renamed_dest}
    return row


def categorize_delay(seconds: float) -> str:
    if seconds <= 300:
        return "no delay expected"
    if seconds <= 900:
        return "5-15 minutes"
    return "over 15 minutes"


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)

    openmeteo_client = OpenMeteoClient(
        base_url=args.openmeteo_url,
        hourly_vars=args.hourly_vars,
        cache_dir=args.cache_dir,
    )

    pipeline, model_name = load_model(args.model_dir, args.latest_file)

    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        features = build_feature_row(openmeteo_client, scenario)
        rows.append(features)

    df_features = pd.DataFrame(rows)
    enriched_rows = []
    for scenario, feature_map in zip(scenarios, df_features.to_dict(orient="records")):
        vector = ensure_feature_columns(pipeline, feature_map)
        enriched_rows.append((scenario, vector))

    predictions: List[float] = []
    for scenario, vec in enriched_rows:
        df_vec = pd.DataFrame([vec])
        pred = float(pipeline.predict(df_vec)[0])
        predictions.append(pred)
        bucket = categorize_delay(pred)
        print(
            f"Flight {scenario.flight_id} ({scenario.origin_icao}->{scenario.dest_icao}): {pred:0.1f} sec => {bucket}"
        )


if __name__ == "__main__":
    main()
