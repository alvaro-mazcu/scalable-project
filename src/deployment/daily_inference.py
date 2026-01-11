from dotenv import load_dotenv
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import json
import hopsworks
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import joblib
import glob
import tempfile
import xgboost as xgb

load_dotenv()
API_KEY = os.getenv("EDGE_API_KEY")
MODEL_NAME = "daily_departure_time_enc2_xgb"

DEFAULT_AIRPORTS = {
    "LHR": {"lat": 51.4700, "lon": -0.4543},
    "FRA": {"lat": 50.0333, "lon": 8.5705},
    "AMS": {"lat": 52.3086, "lon": 4.7639},
    "CPH": {"lat": 55.6179, "lon": 12.6560},
    "CDG": {"lat": 49.0097, "lon": 2.5479},
    "IST": {"lat": 41.2753, "lon": 28.7519},
    "MAD": {"lat": 40.4839, "lon": -3.5680},
    "BCN": {"lat": 41.2974, "lon": 2.0833},
    "FCO": {"lat": 41.8003, "lon": 12.2389},
    "MUC": {"lat": 48.3537, "lon": 11.7750},
}


def load_airports():
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


def call_edge(date_from: str, date_to: str = None, airport: str = "CPH", kind: str = "departure"):
    url = "https://aviation-edge.com/v2/public/timetable"
    params = {
        "key": API_KEY,
        "iataCode": airport,
        "type": kind,
        # "date_from": date_from,
        # "date_to": date_to,
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
    except Exception as exc:
        print(f"[WARN] Timetable API failed for {airport}: {exc}")
        return []

    flights = response.json()

    if not flights or "error" in flights:
        print("No data found or API error:", flights.get("error", "Unknown error") if isinstance(flights, dict) else "Unknown error")
        return []

    return flights


def create_dataframe_flights(flights_json: dict):
    data = []
    for f in flights_json:
        dep = f.get("departure", {})
        arr = f.get("arrival", {})
        airline = f.get("airline", {})
        flight = f.get("flight", {})

        data.append({
            "flight_iata": flight.get("iataNumber"),
            "airline": airline.get("name"),
            "dep_airport": dep.get("iataCode"),
            "dep_time_sched": dep.get("scheduledTime"),
            "dep_delay": dep.get("delay") or 0,
            "arr_airport": arr.get("iataCode"),
            "arr_time_sched": arr.get("scheduledTime"),
            "arr_delay": arr.get("delay") or 0,
        })

    df = pd.DataFrame(data)
    time_cols = ["dep_time_sched", "arr_time_sched"]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df.dropna(subset=time_cols).reset_index(drop=True)
    return df


def get_departures_window(airport_code: str, window_hours: int = 72) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    window_end = now + timedelta(hours=window_hours)
    date_from = now.strftime("%Y-%m-%d")
    date_to = window_end.strftime("%Y-%m-%d")

    flights_json = call_edge(date_from=date_from, date_to=date_to, airport=airport_code, kind="departure")
    if not flights_json:
        return pd.DataFrame()

    df = create_dataframe_flights(flights_json)
    df = df[(df["dep_time_sched"] >= now) & (df["dep_time_sched"] <= window_end)].copy()
    df["dep_airport"] = df["dep_airport"].str.upper()
    df["arr_airport"] = df["arr_airport"].str.upper()
    return df


def fetch_weather_window(start_date: str, end_date: str) -> pd.DataFrame:
    weather_frames = []

    for airport in AIRPORTS.keys():
        lat, lon = AIRPORTS[airport]["lat"], AIRPORTS[airport]["lon"]
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,pressure_msl,relative_humidity_2m,cloudcover,wind_direction_10m,weather_code",
            "timezone": "UTC",
        }

        time.sleep(0.2)

        try:
            res = requests.get("https://api.open-meteo.com/v1/forecast", params=weather_params, timeout=60)
            res.raise_for_status()
            w_data = res.json().get("hourly", {})
            temp_df = pd.DataFrame(w_data)
            temp_df["airport_iata"] = airport.lower()
            temp_df.rename(columns={"time": "weather_timestamp"}, inplace=True)
            weather_frames.append(temp_df)
        except Exception as exc:
            print(f"Error fetching weather for {airport}: {exc}")

    if not weather_frames:
        return pd.DataFrame()

    weather_df = pd.concat(weather_frames, ignore_index=True)
    weather_df["weather_timestamp"] = pd.to_datetime(weather_df["weather_timestamp"], utc=True)
    return weather_df


def merge_flights_weather(flights_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    flights_df = flights_df.copy()
    flights_df["dep_airport"] = flights_df["dep_airport"].str.lower()
    flights_df["arr_airport"] = flights_df["arr_airport"].str.lower()

    departure_weather = weather_df[weather_df["airport_iata"].isin(ROWS_TO_KEEP)].copy()
    arrival_weather = weather_df[weather_df["airport_iata"].isin(ROWS_TO_KEEP)].copy()

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

    # merged_df = pd.merge(
    #     merged_df,
    #     arrival_weather,
    #     left_on=["arr_airport", "arr_time_hour"],
    #     right_on=["arr_airport", "weather_timestamp"],
    #     how="left",
    #     suffixes=("_dep", "_arr"),
    # )

    merged_df = merged_df.drop_duplicates(subset=["dep_airport", "dep_time_sched", "arr_airport", "arr_time_sched"])

    return merged_df


def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df["dep_time_sched"] = pd.to_datetime(df.get("dep_time_sched"), errors="coerce", utc=True)
    df["arr_time_sched"] = pd.to_datetime(df.get("arr_time_sched"), errors="coerce", utc=True)
    df["weather_timestamp_deo"] = pd.to_datetime(df.get("weather_timestamp_deo"), errors="coerce", utc=True)
    df["weather_timestamp_arr"] = pd.to_datetime(
        df["weather_timestamp_arr"] if "weather_timestamp_arr" in df.columns else df["arr_time_sched"],
        errors="coerce",
        utc=True,
    )

    def _add_wind_features(prefix: str):
        col = f"wind_direction_10m_{prefix}"
        sin_col = f"wind_dir_sin_{prefix}"
        cos_col = f"wind_dir_cos_{prefix}"
        if col in df:
            radians = np.deg2rad(df[col].fillna(0))
            df[sin_col] = np.sin(radians)
            df[cos_col] = np.cos(radians)
        else:
            df[sin_col] = 0.0
            df[cos_col] = 0.0

    _add_wind_features("dep")
    _add_wind_features("arr")

    X = df.copy()
    X["dep_airport"] = X["dep_airport"].str.lower()

    X_enc = pd.get_dummies(X, columns=["dep_airport"])
    for col in feature_cols:
        if col not in X_enc:
            X_enc[col] = 0

    dep_hours = X["weather_timestamp_deo"].dt.hour.fillna(0)
    arr_hours = X["weather_timestamp_arr"].dt.hour.fillna(0)
    X_enc["dep_hour_sin"] = np.sin(2 * np.pi * dep_hours / 24)
    X_enc["dep_hour_cos"] = np.cos(2 * np.pi * dep_hours / 24)
    X_enc["arr_hour_sin"] = np.sin(2 * np.pi * arr_hours / 24)
    X_enc["arr_hour_cos"] = np.cos(2 * np.pi * arr_hours / 24)

    for col in feature_cols:
        if col not in X_enc:
            X_enc[col] = 0

    return X_enc[feature_cols]


def load_latest_model(project, model_name: str, model_dir: str = "models") -> tuple:
    mr = project.get_model_registry()
    models = mr.get_models(model_name)
    if not models:
        raise RuntimeError(f"No model named {model_name} found in the registry.")

    latest = max(models, key=lambda item: item.version or 0)
    os.makedirs(model_dir, exist_ok=True)
    download_dir = tempfile.mkdtemp(prefix=f"{model_name}_v{latest.version}_", dir=model_dir)
    download_path = latest.download(local_path=download_dir)
    candidates = sorted(glob.glob(os.path.join(download_path, "*.joblib")))
    if not candidates:
        raise RuntimeError(f"No joblib artifacts found in {download_path}")
    model_path = max(candidates, key=os.path.getmtime)
    payload = joblib.load(model_path)
    return payload["model"], payload["features"], model_path


def upload_predictions_hopsworks(df: pd.DataFrame, project=None):
    if project is None:
        project = hopsworks.login(
            engine="python",
            project=os.getenv("HOPSWORKS_PROJECT"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="daily_inference_predictions_fg",
        version=1,
        description="Daily inference: flights, weather, and delay predictions",
        primary_key=["flight_iata", "dep_airport", "dep_time_sched"],
        event_time="dep_time_sched",
    )

    required_cols = ["flight_iata", "dep_airport", "dep_time_sched"]
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    df = coerce_df_to_fg_schema(df, fg)
    fg.insert(df, write_options={"wait_for_job": False})


def coerce_df_to_fg_schema(df: pd.DataFrame, fg) -> pd.DataFrame:
    df = df.copy()
    schema = fg.schema or []
    if schema:
        schema_names = [feature.name for feature in schema]
        df = df[[col for col in df.columns if col in schema_names]]

    type_map = {feature.name: (feature.type or "").lower() for feature in schema}
    timestamp_cols: list[str] = []
    for name, ftype in type_map.items():
        if name not in df.columns:
            continue
        if ftype in {"int", "bigint", "smallint", "tinyint", "integer", "long"}:
            df[name] = pd.to_numeric(df[name], errors="coerce").astype("Int64")
        elif ftype in {"double", "float", "decimal"}:
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif "timestamp" in ftype or ftype in {"date", "time"}:
            df[name] = pd.to_datetime(df[name], errors="coerce", utc=True)
            timestamp_cols.append(name)
        elif ftype in {"string", "varchar"}:
            df[name] = df[name].astype(str)
        elif ftype == "boolean":
            df[name] = df[name].astype("boolean")

    if timestamp_cols:
        df = df.dropna(subset=timestamp_cols)
        for col in timestamp_cols:
            df[col] = df[col].apply(lambda value: value.to_pydatetime() if pd.notna(value) else None)

    string_cols = {"flight_iata", "airline", "dep_airport", "arr_airport", "model_path"}
    for col in df.columns:
        if col in string_cols:
            df[col] = df[col].astype(str)
            continue
        if df[col].dtype == object and ("time" in col or "timestamp" in col):
            series = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = series.apply(lambda value: value.to_pydatetime() if pd.notna(value) else None)
            continue
        if df[col].dtype == object:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() == df[col].notna().sum():
                df[col] = numeric
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].apply(lambda value: value.to_pydatetime() if pd.notna(value) else None)

    df = df.replace({pd.NaT: None})
    return df


def main():
    if not API_KEY:
        raise RuntimeError("EDGE_API_KEY is required to fetch flight data.")

    project = hopsworks.login(
        engine="python",
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    model, feature_cols, model_path = load_latest_model(project, MODEL_NAME)
    print(f"Loaded model: {model_path}")

    all_flights = []
    for airport in AIRPORTS.keys():
        print(f"Fetching departures for {airport}")
        try:
            df = get_departures_window(airport)
        except Exception as exc:
            print(f"[WARN] Skipping {airport} due to fetch error: {exc}")
            continue
        if df.empty:
            continue
        all_flights.append(df)

    if not all_flights:
        print("No departures found for the current window.")
        return

    flights_df = pd.concat(all_flights, ignore_index=True)
    now = datetime.now(timezone.utc)
    start_date = now.strftime("%Y-%m-%d")
    end_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    weather_df = fetch_weather_window(start_date, end_date)
    if weather_df.empty:
        print("Weather data could not be fetched.")
        return

    merged_df = merge_flights_weather(flights_df, weather_df)
    feature_df = prepare_features(merged_df, feature_cols)

    preds_log = model.predict(feature_df)
    preds = np.expm1(preds_log)
    merged_df["predicted_dep_delay"] = preds
    merged_df["model_path"] = model_path
    merged_df["inference_timestamp"] = datetime.now(timezone.utc)

    # print(merged_df[["flight_iata", "airline", "dep_airport", "dep_time_sched", "dep_delay", "predicted_arr_delay"]].head(25))
    print("R2: ", r2_score(merged_df["dep_delay"], np.clip(preds, 0, None)))
    mae = mean_absolute_error(merged_df["dep_delay"], np.clip(preds, 0, None))
    rmse = root_mean_squared_error(merged_df["dep_delay"], np.clip(preds, 0, None))
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    upload_predictions_hopsworks(merged_df, project=project)
    print("Daily inference completed and uploaded to Hopsworks.")


if __name__ == "__main__":
    main()
