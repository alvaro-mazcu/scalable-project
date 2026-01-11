#!/usr/bin/env python3
"""Daily training pipeline for arrival delay prediction (time encoding 2 + XGBoost)."""
import argparse
import datetime as dt
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

import hopsworks

load_dotenv()

API_KEY = os.getenv("EDGE_API_KEY", "")
DESTINATION = os.getenv("DESTINATION_IATA", "CPH").upper()
MODEL_NAME = "daily_departure_time_enc2_xgb"

# DEFAULT_AIRPORTS = {
#     "LHR": {"lat": 51.4700, "lon": -0.4543},
#     "FRA": {"lat": 50.0333, "lon": 8.5705},
#     "AMS": {"lat": 52.3086, "lon": 4.7639},
#     "CPH": {"lat": 55.6179, "lon": 12.6560},
# }

DEFAULT_AIRPORTS = {
    "LHR": {"lat": 51.4700, "lon": -0.4543},  # London Heathrow, UK
    "FRA": {"lat": 50.0333, "lon": 8.5705},   # Frankfurt, Germany
    "AMS": {"lat": 52.3086, "lon": 4.7639},   # Amsterdam Schiphol, Netherlands
    "CPH": {"lat": 55.6179, "lon": 12.6560},  # Copenhagen, Denmark
    "CDG": {"lat": 49.0097, "lon": 2.5479},   # Paris Charles de Gaulle, France
    "IST": {"lat": 41.2753, "lon": 28.7519},  # Istanbul, Turkey
    "MAD": {"lat": 40.4839, "lon": -3.5680},  # Madrid Adolfo Suárez-Barajas, Spain
    "BCN": {"lat": 41.2974, "lon": 2.0833},   # Barcelona–El Prat, Spain
    "FCO": {"lat": 41.8003, "lon": 12.2389},  # Rome Fiumicino, Italy
    "MUC": {"lat": 48.3537, "lon": 11.7750},  # Munich, Germany
}


def load_airports() -> dict[str, dict[str, float]]:
    raw = os.getenv("AIRPORTS_JSON", "").strip()
    if not raw:
        return DEFAULT_AIRPORTS
    try:
        payload = json.loads(raw)
        return {k.upper(): {"lat": v["lat"], "lon": v["lon"]} for k, v in payload.items()}
    except Exception as exc:
        print(f"[WARN] Failed to parse AIRPORTS_JSON: {exc}. Using defaults.")
        return DEFAULT_AIRPORTS


AIRPORTS = load_airports()
ROWS_TO_KEEP = [airport.lower() for airport in AIRPORTS.keys()]


def call_edge(date_from: str, date_to: str | None = None, airport: str = DESTINATION, kind: str = "departure"):
    url = "https://aviation-edge.com/v2/public/flightsHistory"
    params = {
        "key": API_KEY,
        "code": airport,
        "type": kind,
        "date_from": date_from,
        "date_to": date_to or date_from,
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    flights = response.json()
    if not flights or "error" in flights:
        return []
    return flights


def create_dataframe_flights(flights_json: list[dict]) -> pd.DataFrame:
    data = []
    for f in flights_json:
        dep = f.get("departure", {})
        arr = f.get("arrival", {})
        airline = f.get("airline", {})
        flight = f.get("flight", {})
        data.append(
            {
                "flight_iata": flight.get("iataNumber"),
                "airline": airline.get("name"),
                "dep_airport": dep.get("iataCode"),
                "dep_time_sched": dep.get("scheduledTime"),
                # "dep_time_actual": dep.get("actualTime"),
                "dep_delay": dep.get("delay") or 0,
                "arr_airport": arr.get("iataCode"),
                "arr_time_sched": arr.get("scheduledTime"),
                # "arr_time_actual": arr.get("actualTime"),
                "arr_delay": arr.get("delay") or 0,
            }
        )

    df = pd.DataFrame(data)
    for col in ["dep_time_sched", "arr_time_sched"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fetch_flights_range(start_date: str, end_date: str, airport: str = DESTINATION, kind: str = "departure") -> pd.DataFrame:
    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d")
    all_records: list[dict] = []
    current = start_dt

    while current <= end_dt:
        chunk_end = min(current + dt.timedelta(days=15), end_dt)
        from_str = current.strftime("%Y-%m-%d")
        to_str = chunk_end.strftime("%Y-%m-%d")
        try:
            chunk = call_edge(date_from=from_str, date_to=to_str, airport=airport, kind=kind)
            if isinstance(chunk, list):
                all_records.extend(chunk)
        except Exception as exc:
            print(f"[WARN] Flight fetch failed for {from_str} to {to_str}: {exc}")
        current = chunk_end + dt.timedelta(days=1)
        time.sleep(0.2)

    if not all_records:
        return pd.DataFrame()

    df = create_dataframe_flights(all_records)
    df = df.drop_duplicates(subset=["flight_iata", "dep_time_sched"])
    return df


def fetch_weather(flights_df: pd.DataFrame) -> pd.DataFrame:
    start_date = flights_df["dep_time_sched"].min().strftime("%Y-%m-%d")
    end_date = flights_df["dep_time_sched"].max().strftime("%Y-%m-%d")

    weather_frames: list[pd.DataFrame] = []
    for airport, coords in AIRPORTS.items():
        lat, lon = coords["lat"], coords["lon"]
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,pressure_msl,relative_humidity_2m,cloudcover,wind_direction_10m,weather_code",
            "timezone": "UTC",
        }
        try:
            res = requests.get("https://archive-api.open-meteo.com/v1/archive", params=weather_params, timeout=60)
            res.raise_for_status()
            w_data = res.json().get("hourly", {})
            temp_df = pd.DataFrame(w_data)
            temp_df["airport_iata"] = airport.lower()
            temp_df.rename(columns={"time": "weather_timestamp"}, inplace=True)
            weather_frames.append(temp_df)
        except Exception as exc:
            print(f"[WARN] Weather fetch failed for {airport}: {exc}")
        time.sleep(0.2)

    if not weather_frames:
        return pd.DataFrame()

    weather_df = pd.concat(weather_frames, ignore_index=True)
    weather_df["weather_timestamp"] = pd.to_datetime(weather_df["weather_timestamp"])
    return weather_df


def merge_flights_weather(flights_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    flights_df = flights_df.copy()
    flights_df["dep_airport"] = flights_df["dep_airport"].str.lower()
    flights_df["arr_airport"] = flights_df["arr_airport"].str.lower()

    departure_weather = weather_df[weather_df["airport_iata"].isin(ROWS_TO_KEEP)].copy()
    arrival_weather = weather_df[weather_df["airport_iata"] == DESTINATION.lower()].copy()

    flights_df["weather_timestamp_deo"] = flights_df["dep_time_sched"].dt.floor("H")
    flights_df["arr_time_hour"] = flights_df["arr_time_sched"].dt.floor("H")

    departure_weather = departure_weather.rename(columns={"airport_iata": "dep_airport"})
    arrival_weather = arrival_weather.rename(columns={"airport_iata": "arr_airport"})

    flights_df["weather_timestamp_deo"] = pd.to_datetime(flights_df["weather_timestamp_deo"], utc=True)
    flights_df["arr_time_hour"] = pd.to_datetime(flights_df["arr_time_hour"], utc=True)
    departure_weather["weather_timestamp"] = pd.to_datetime(departure_weather["weather_timestamp"], utc=True)
    arrival_weather["weather_timestamp"] = pd.to_datetime(arrival_weather["weather_timestamp"], utc=True)

    merged_df = pd.merge(
        flights_df,
        departure_weather,
        left_on=["dep_airport", "weather_timestamp_deo"],
        right_on=["dep_airport", "weather_timestamp"],
        how="left",
    )

    merged_df = pd.merge(
        merged_df,
        arrival_weather,
        left_on=["arr_airport", "arr_time_hour"],
        right_on=["arr_airport", "weather_timestamp"],
        how="left",
        suffixes=("_dep", "_arr"),
    )

    return merged_df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Uses time encoding and one-hot encoding for the airport.
    TODO: Improve docstring!
    """
    df = df.copy()
    for col in ["dep_time_sched", "arr_time_sched", "weather_timestamp_deo", "weather_timestamp_arr"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df.sort_values("dep_time_sched")
    df = df.drop_duplicates(subset=["dep_airport", "dep_time_sched", "arr_airport", "arr_time_sched"])

    drop_cols = ["flight_iata", "airline"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df["weather_timestamp_deo"] = pd.to_datetime(df["weather_timestamp_deo"], errors="coerce", utc=True)
    df["weather_timestamp_arr"] = pd.to_datetime(df["weather_timestamp_arr"], errors="coerce", utc=True)

    df["wind_dir_sin_dep"] = np.sin(2 * np.pi * df["wind_direction_10m_dep"] / 360)
    df["wind_dir_cos_dep"] = np.cos(2 * np.pi * df["wind_direction_10m_dep"] / 360)
    df["wind_dir_sin_arr"] = np.sin(2 * np.pi * df["wind_direction_10m_arr"] / 360)
    df["wind_dir_cos_arr"] = np.cos(2 * np.pi * df["wind_direction_10m_arr"] / 360)

    departure_cols = [
        "temperature_2m_dep",
        "precipitation_dep",
        "wind_speed_10m_dep",
        "wind_gusts_10m_dep",
        "pressure_msl_dep",
        "relative_humidity_2m_dep",
        "cloudcover_dep",
        "weather_code_dep",
        "wind_dir_sin_dep",
        "wind_dir_cos_dep",
    ]
    arrival_cols = [
        "temperature_2m_arr",
        "precipitation_arr",
        "wind_speed_10m_arr",
        "wind_gusts_10m_arr",
        "pressure_msl_arr",
        "relative_humidity_2m_arr",
        "cloudcover_arr",
        "weather_code_arr",
        "wind_dir_sin_arr",
        "wind_dir_cos_arr",
    ]
    all_cols = departure_cols # + arrival_cols

    df["dep_airport"] = df["dep_airport"].str.lower()

    dep_airport_cols = [f"dep_airport_{code.lower()}" for code in AIRPORTS.keys() if code.lower() != DESTINATION.lower()]

    df["dep_delay"] = np.clip(df["dep_delay"], 0, 180)

    X = df.drop(["dep_delay", "arr_delay"], axis=1)
    X_enc = pd.get_dummies(X, columns=["dep_airport"])
    for col in dep_airport_cols:
        if col not in X_enc:
            X_enc[col] = 0

    X_enc = X_enc[dep_airport_cols + [c for c in all_cols if c in X_enc.columns]]

    X_enc["dep_hour_sin"] = np.sin(2 * np.pi * df["weather_timestamp_deo"].dt.hour / 24)
    X_enc["dep_hour_cos"] = np.cos(2 * np.pi * df["weather_timestamp_deo"].dt.hour / 24)
    X_enc["arr_hour_sin"] = np.sin(2 * np.pi * df["weather_timestamp_arr"].dt.hour / 24)
    X_enc["arr_hour_cos"] = np.cos(2 * np.pi * df["weather_timestamp_arr"].dt.hour / 24)

    y = np.log1p(df["dep_delay"].fillna(0))
    return X_enc, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[xgb.XGBRegressor, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R^2: {r2:.4f}")

    return model, r2


def save_model(model: xgb.XGBRegressor, feature_cols: list[str], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    path = os.path.join(out_dir, f"{MODEL_NAME}_{stamp}.joblib")
    joblib.dump({"model": model, "features": feature_cols}, path)
    return path


def register_model(mr, model_path: str):
    model = mr.python.create_model(
        name=MODEL_NAME,
        description="Daily departure delay model (time encoding 2).",
    )
    return model.save(model_path)


def fetch_historical_dataset(path: str, start_date: str, end_date: str, refresh: bool) -> pd.DataFrame:
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path)
        for col in ["dep_time_sched", "arr_time_sched", "weather_timestamp_deo", "weather_timestamp_arr"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        return df

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flights_df = fetch_flights_range(start_date, end_date, airport=DESTINATION, kind="departure")
    if flights_df.empty:
        return pd.DataFrame()
    weather_df = fetch_weather(flights_df)
    merged_df = merge_flights_weather(flights_df, weather_df)
    merged_df.to_csv(path, index=False)
    return merged_df


def fetch_yesterday_dataset(path_2_save: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    yesterday = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=4)).strftime("%Y-%m-%d")
    frames: list[pd.DataFrame] = []
    for airport in DEFAULT_AIRPORTS.keys():
        df = fetch_flights_range(yesterday, yesterday, airport=airport, kind="departure")
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    flights_df = pd.concat(frames, ignore_index=True)
    weather_df = fetch_weather(flights_df)
    merged_df = merge_flights_weather(flights_df, weather_df)
    os.makedirs(os.path.dirname(path_2_save) or ".", exist_ok=True)
    merged_df.to_csv(path_2_save, index=False)
    return merged_df, weather_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-csv", default="data/historical_flights_weather.csv")
    parser.add_argument("--yesterday-csv", default="data/yesterday_flights_weather.csv")
    parser.add_argument("--model-dir", default="models")
    # parser.add_argument("--historical-start", default=os.getenv("HIST_START_DATE", "2025-01-01"))
    # parser.add_argument("--historical-end", default=os.getenv("HIST_END_DATE", (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).strftime("%Y-%m-%d")))
    # parser.add_argument("--refresh-historical", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("EDGE_API_KEY is required to fetch flight data.")
    
    project = hopsworks.login(
        engine="python",
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )

    fs = project.get_feature_store()
    flights_fg = fs.get_feature_group(name="european_flights_fg", version=2)
    mr = project.get_model_registry()

    # historical_df = fetch_historical_dataset(
    #     path=args.historical_csv,
    #     start_date=args.historical_start,
    #     end_date=args.historical_end,
    #     refresh=args.refresh_historical,
    # )

    historical_df = flights_fg.read()

    if historical_df.empty:
        raise RuntimeError("Historical dataset is empty. Check API key, dates, and airport configuration.")

    weather_fg = fs.get_feature_group(name="european_flights_weather_fg", version=1)
    historical_weather = weather_fg.read()
    historical_train = merge_flights_weather(historical_df, historical_weather)

    yesterday_df, yesterday_weather = fetch_yesterday_dataset(path_2_save=args.yesterday_csv)
    if yesterday_df.empty:
        print("[WARN] Yesterday dataset is empty; training with historical data only.")
        train_df = historical_train.copy()
    else:
        train_df = pd.concat([historical_train, yesterday_df], ignore_index=True)

    X, y = prepare_training_data(train_df)
    model, r2 = train_model(X, y)
    print(f"[METRIC] R2: {r2:.4f}")
    model_path = save_model(model, list(X.columns), args.model_dir)
    print(f"[DONE] Model saved to {model_path}")
    registered_model = register_model(mr, model_path)
    print(f"[DONE] Registered model {registered_model.name} v{registered_model.version}")

    if not yesterday_df.empty:
        key_cols = ["flight_iata", "dep_airport", "dep_time_sched"]
        flight_cols = [col for col in historical_df.columns if col in yesterday_df.columns]
        yesterday_flights = yesterday_df[flight_cols].copy()

        for df in (historical_df, yesterday_flights):
            if "dep_time_sched" in df.columns:
                df["dep_time_sched"] = pd.to_datetime(df["dep_time_sched"], errors="coerce", utc=True)
            if "dep_airport" in df.columns:
                df["dep_airport"] = df["dep_airport"].astype(str).str.upper()
            if "flight_iata" in df.columns:
                df["flight_iata"] = df["flight_iata"].astype(str).str.upper()

        existing = historical_df[key_cols].dropna()
        new_rows = yesterday_flights.merge(existing, on=key_cols, how="left", indicator=True)
        new_rows = new_rows[new_rows["_merge"] == "left_only"].drop(columns=["_merge"])
        if not new_rows.empty:
            flights_fg.insert(new_rows, write_options={"wait_for_job": False})
            print(f"[DONE] Inserted {len(new_rows)} new flight(s) into european_flights_fg")
        else:
            print("[INFO] No new yesterday flights to insert.")

        weather_key_cols = ["airport_iata", "weather_timestamp"]
        if not yesterday_weather.empty:
            weather_df = yesterday_weather.dropna(subset=weather_key_cols).drop_duplicates(subset=weather_key_cols).copy()
            weather_df["airport_iata"] = weather_df["airport_iata"].astype(str).str.lower()

            if "weather_timestamp" in historical_weather.columns:
                if pd.api.types.is_datetime64_any_dtype(historical_weather["weather_timestamp"]):
                    if historical_weather["weather_timestamp"].dt.tz is None:
                        weather_df["weather_timestamp"] = pd.to_datetime(weather_df["weather_timestamp"], errors="coerce")
                    else:
                        weather_df["weather_timestamp"] = pd.to_datetime(weather_df["weather_timestamp"], errors="coerce", utc=True)

            existing_weather = historical_weather[weather_key_cols].dropna()
            new_weather = weather_df.merge(existing_weather, on=weather_key_cols, how="left", indicator=True)
            new_weather = new_weather[new_weather["_merge"] == "left_only"].drop(columns=["_merge"])

            if not new_weather.empty:
                weather_fg.insert(new_weather, write_options={"wait_for_job": False})
                print(f"[DONE] Inserted {len(new_weather)} new weather row(s) into european_flights_weather_fg")
            else:
                print("[INFO] No new yesterday weather to insert.")

    # new_hist = train_df.drop_duplicates(subset=["dep_airport", "dep_time_sched", "arr_time_sched"])
    # os.makedirs(os.path.dirname(args.historical_csv) or ".", exist_ok=True)
    # new_hist.to_csv(args.historical_csv, index=False)
    # print(f"[DONE] Updated historical dataset at {args.historical_csv}")


if __name__ == "__main__":
    main()
