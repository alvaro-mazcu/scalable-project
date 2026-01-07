from dotenv import load_dotenv
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import json
import hopsworks
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()
API_KEY = os.getenv("EDGE_API_KEY")

# dates and destination
START_DATE = "2025-01-08"
END_DATE = "2026-01-02"
DESTINATION = "CPH"

# origins
AIRPORTS = {
    "LHR": {"lat": 51.4700, "lon": -0.4543},
    "FRA": {"lat": 50.0333, "lon": 8.5705},
    "AMS": {"lat": 52.3086, "lon": 4.7639},
    "CPH": {"lat": 55.6179, "lon": 12.6560}
}

ROWS_TO_KEEP = [
        airport.lower() for airport in AIRPORTS.keys()
    ]

RESULTS = []

def call_edge(date_from: str, date_to: str = None, airport: str = "CPH", kind: str = "arrival"):

    url = "https://aviation-edge.com/v2/public/flightsHistory"
    params = {
        "key": API_KEY,
        "code": airport,
        "type": kind,
        "date_from": date_from,
        "date_to": date_to,
    }

    try:

        response = requests.get(url, params=params)
        response.raise_for_status()
        flights = response.json()


        if not flights or "error" in flights:
            print("No data found or API error:", flights.get("error", "Unknown error"))
            return

        return flights

    except requests.exceptions.RequestException as e:

        print(f"An error occurred: {e}") 

def get_all_flights(airport: str = "CPH", kind: str = "arrival"):

    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    total_days = (end_dt - start_dt).days
    num_chunks = (total_days // 15) + 1
    
    all_flight_records = []
    current_start = start_dt

    pbar = tqdm(total=num_chunks, desc="Fetching Yearly Data")

    while current_start < end_dt:

        current_end = min(current_start + timedelta(days=15), end_dt)
        
        from_str = current_start.strftime("%Y-%m-%d")
        to_str = current_end.strftime("%Y-%m-%d")
        
        chunk_data = call_edge(date_from=from_str, date_to=to_str, airport=airport, kind=kind)
        
        if chunk_data and isinstance(chunk_data, list):
            all_flight_records.extend(chunk_data)
        
        current_start = current_end + timedelta(days=1)
        pbar.update(1)
        
        time.sleep(0.2)

    pbar.close()

    if not all_flight_records:
        print("No data was retrieved.")
        return pd.DataFrame()

    df = create_dataframe_flights(all_flight_records)
    df = df.drop_duplicates(subset=['flight_iata', 'dep_time_sched'])

    return df

def create_dataframe_flights(flights_json: dict):

    data = []
    for f in flights_json:

        dep = f.get('departure', {})
        arr = f.get('arrival', {})
        airline = f.get('airline', {})
        flight = f.get('flight', {})

        data.append({
            'flight_iata': flight.get('iataNumber'),
            'airline': airline.get('name'),
            'dep_airport': dep.get('iataCode'),
            'dep_time_sched': dep.get('scheduledTime'),
            'dep_time_actual': dep.get('actualTime'),
            'dep_delay': dep.get('delay') or 0,
            'arr_airport': arr.get('iataCode'),
            'arr_time_sched': arr.get('scheduledTime'),
            'arr_time_actual': arr.get('actualTime'),
            'arr_delay': arr.get('delay') or 0
        })
    
    df = pd.DataFrame(data)

    time_cols = [
        'dep_time_sched', 'dep_time_actual', 
        'arr_time_sched', 'arr_time_actual'
    ]
    
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

def upload_hopsworks(flights_df: pd.DataFrame, airport: str = "CPH", kind: str = "arrival"):

    project = hopsworks.login(
        engine="python",
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()

    flights_fg = fs.get_or_create_feature_group(
        name=f"flights_{airport}_{kind}_fg",
        version=1,
        description="Flight data with weather features",
        primary_key=["flight_iata", "dep_time_sched"],
    )

    flights_fg.insert(flights_df, write_options={"wait_for_job": False})

def get_flights_hopsworks():

    project = hopsworks.login(
        engine="python",
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )
    fs = project.get_feature_store()

    flights_fg = fs.get_feature_group("flights_fg", version=1)

    flights_df = flights_fg.read()

    return flights_df.loc[flights_df['dep_airport'].isin(ROWS_TO_KEEP)]

def get_weather(flights_df):

    start_date = flights_df['dep_time_sched'].min().strftime('%Y-%m-%d')
    end_date = flights_df['dep_time_sched'].max().strftime('%Y-%m-%d')
    
    weather_frames = []
    
    for airport in AIRPORTS.keys():

        lat, lon = AIRPORTS[airport]['lat'], AIRPORTS[airport]['lon']
        
        if lat is None or lon is None:
            continue

        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        weather_params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,pressure_msl,relative_humidity_2m,cloudcover,wind_direction_10m,weather_code",
            "timezone": "UTC"
        }

        time.sleep(0.5)
        
        try:
            res = requests.get(weather_url, params=weather_params)
            res.raise_for_status()
            w_data = res.json().get('hourly', {})
            
            temp_df = pd.DataFrame(w_data)
            temp_df['airport_iata'] = airport.lower()
            temp_df.rename(columns={'time': 'weather_timestamp'}, inplace=True)
            weather_frames.append(temp_df)
        except Exception as e:
            print(f"Error fetching weather for {airport}: {e}")

    final_weather_df = pd.concat(weather_frames, ignore_index=True)
    final_weather_df['weather_timestamp'] = pd.to_datetime(final_weather_df['weather_timestamp'])

    final_weather_df.to_parquet('weather_data.parquet', index=False)
    
    return final_weather_df

def merge_flights_weather(flights_df, weather_df):

    departure_weather = weather_df[weather_df['airport_iata'].isin(ROWS_TO_KEEP)].copy()
    arrival_weather = weather_df[weather_df['airport_iata'] == 'cph'].copy()

    flights_df.loc[:, 'weather_timestamp_deo'] = flights_df['dep_time_sched'].dt.floor('H')
    flights_df.loc[:, 'arr_time_hour'] = flights_df['arr_time_sched'].dt.floor('H')

    departure_weather = departure_weather.rename(columns={'airport_iata': 'dep_airport'})
    arrival_weather = arrival_weather.rename(columns={'airport_iata': 'arr_airport'})

    flights_df.loc[:, 'weather_timestamp_deo'] = pd.to_datetime(flights_df['weather_timestamp_deo'], utc=True)
    departure_weather.loc[:, 'weather_timestamp'] = pd.to_datetime(departure_weather['weather_timestamp'], utc=True)
    arrival_weather.loc[:, 'weather_timestamp'] = pd.to_datetime(arrival_weather['weather_timestamp'], utc=True)

    merged_df = pd.merge(
        flights_df,
        departure_weather,
        left_on=['dep_airport', 'weather_timestamp_deo'],
        right_on=['dep_airport', 'weather_timestamp'],
        how='left'
    )

    merged_df = pd.merge(
        merged_df,
        arrival_weather,
        left_on=['arr_airport', 'arr_time_hour'],
        right_on=['arr_airport', 'weather_timestamp'],
        how='left',
        suffixes=('_dep', '_arr')
    )

    return merged_df

def get_operational_bin(hour):
    if 6 <= hour < 10:
        return 'morning'
    elif 10 <= hour < 16:
        return 'midday'
    elif 16 <= hour < 21:
        return 'evening'
    else:
        return 'night'

def train_regressor(model, X_train, y_train, X_test, y_test, name="", target="", feat_set=""):

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R^2: {r2:.4f}")

    RESULTS.append({
            'Model Name': name,
            'Target': target,
            'Feature Set': feat_set,
            'MAE': mae,
            'R2': r2,
            'Model': model,
            'Features': list(X_train.columns)
        })

    return model

def train_and_eval_models(split_strategy: str = "random"):

    flights_df = get_flights_hopsworks()
    weather_df = pd.read_parquet('weather_data.parquet') 
    merged_df = merge_flights_weather(flights_df, weather_df)

    unique_keys = ['dep_airport', 'dep_time_sched', 'arr_time_sched']
    merged_df = merged_df.drop_duplicates(subset=unique_keys)

    merged_df = merged_df.sort_values('dep_time_sched')
    print(merged_df[['dep_airport', 'dep_time_sched', 'arr_time_sched', 'flight_iata', 'airline']][:50])

    merged_df = merged_df.sort_values('dep_time_sched')
    drop_cols = ['dep_time_actual', 'arr_time_actual', 'flight_iata', 'airline']
    merged_df = merged_df.drop(columns=drop_cols)

    # df_dep_only = merged_df[merged_df['dep_delay'] > 5].copy()
    # df_arr_only = merged_df[merged_df['arr_delay'] > 5].copy()

    # print(f"Original flights: {len(merged_df)}")
    # print(f"Departure Delays (>5m): {len(df_dep_only)}")
    # print(f"Arrival Delays (>5m): {len(df_arr_only)}")

    # merged_df = df_dep_only

    # extreme_mask = (
    #     (merged_df['wind_gusts_10m_dep'] > 15) | 
    #     (merged_df['precipitation_dep'] > 2.5) |
    #     (merged_df['wind_speed_10m_dep'] > 10)
    #     # (merged_df['wind_gusts_10m_arr'] > 15) | 
    #     # (merged_df['precipitation_arr'] > 2.5)
    # )

    # df_extreme = merged_df[extreme_mask].copy()

    # print(f"Total flights: {len(merged_df)}")
    # print(f"Flights during extreme weather: {len(df_extreme)}")

    # merged_df = df_extreme

    merged_df['wind_dir_sin_dep'] = np.sin(2 * np.pi * merged_df['wind_direction_10m_dep'] / 360)
    merged_df['wind_dir_cos_dep'] = np.cos(2 * np.pi * merged_df['wind_direction_10m_dep'] / 360)
    merged_df['wind_dir_sin_arr'] = np.sin(2 * np.pi * merged_df['wind_direction_10m_arr'] / 360)
    merged_df['wind_dir_cos_arr'] = np.cos(2 * np.pi * merged_df['wind_direction_10m_arr'] / 360)

    # data preparation
    X = merged_df.drop(['dep_delay', 'arr_delay'], axis=1)
    y = merged_df[['dep_delay', 'arr_delay']]

    if split_strategy == "time_series":
        split_idx = int(len(merged_df) * 0.95)
        X_train = merged_df.iloc[:split_idx]
        X_test = merged_df.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

    elif split_strategy == "random":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    else:
        raise ValueError("Invalid split strategy. Choose 'random' or 'time_series'.")
    
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    y_train = y_train_log
    y_test = y_test_log

    # select features and prepare training and testing sets
    departure_cols = [
        'temperature_2m_dep', 'precipitation_dep', 'wind_speed_10m_dep', 'wind_gusts_10m_dep', 'pressure_msl_dep', 
        'relative_humidity_2m_dep', 'cloudcover_dep', 'weather_code_dep', 'wind_dir_sin_dep', 'wind_dir_cos_dep'
    ]
    arrival_cols = [
        'temperature_2m_arr', 'precipitation_arr', 'wind_speed_10m_arr', 'wind_gusts_10m_arr', 'pressure_msl_arr', 
        'relative_humidity_2m_arr', 'cloudcover_arr', 'weather_code_arr', 'wind_dir_sin_arr', 'wind_dir_cos_arr'
    ]
    all_cols = departure_cols + arrival_cols

    # training models with departure weather or arrival weather features only
    departure_X_train = X_train[departure_cols]
    departure_X_test = X_test[departure_cols]

    arrival_X_train = X_train[arrival_cols]
    arrival_X_test = X_test[arrival_cols]

    departure_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    departure_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("Training Random Forest Regressor for Departure Delays:")
    train_regressor(departure_rf_model, departure_X_train, y_train['dep_delay'], departure_X_test, y_test['dep_delay'], name="Random Forest", target="Departure", feat_set="Departure Weather Only")
    print("\nTraining XGBoost Regressor for Departure Delays:")
    train_regressor(departure_xgb_model, departure_X_train, y_train['dep_delay'], departure_X_test, y_test['dep_delay'], name="XGBoost", target="Departure", feat_set="Departure Weather Only")

    arrival_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    arrival_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Arrival Delays:")
    train_regressor(arrival_rf_model, arrival_X_train, y_train['arr_delay'], arrival_X_test, y_test['arr_delay'], name="Random Forest", target="Arrival", feat_set="Arrival Weather Only")
    print("\nTraining XGBoost Regressor for Arrival Delays:")
    train_regressor(arrival_xgb_model, arrival_X_train, y_train['arr_delay'], arrival_X_test, y_test['arr_delay'], name="XGBoost", target="Arrival", feat_set="Arrival Weather Only")

    # training models with all weather features
    all_X_train = X_train[all_cols]
    all_X_test = X_test[all_cols]

    all_dep_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    all_dep_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Departure Delays (All Features):")
    train_regressor(all_dep_rf_model, all_X_train, y_train['dep_delay'], all_X_test, y_test['dep_delay'], name="Random Forest", target="Departure", feat_set="All Weather Features")
    print("\nTraining XGBoost Regressor for Departure Delays (All Features):")
    train_regressor(all_dep_xgb_model, all_X_train, y_train['dep_delay'], all_X_test, y_test['dep_delay'], name="XGBoost", target="Departure", feat_set="All Weather Features")

    all_arr_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    all_arr_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Arrival Delays (All Features):")
    train_regressor(all_arr_rf_model, all_X_train, y_train['arr_delay'], all_X_test, y_test['arr_delay'], name="Random Forest", target="Arrival", feat_set="All Weather Features")
    print("\nTraining XGBoost Regressor for Arrival Delays (All Features):")
    train_regressor(all_arr_xgb_model, all_X_train, y_train['arr_delay'], all_X_test, y_test['arr_delay'], name="XGBoost", target="Arrival", feat_set="All Weather Features")

    # train models with categorical encoding for airports
    with_categorical_encoding = ['dep_airport_lhr', 'dep_airport_fra', 'dep_airport_ams'] + all_cols

    X_train_enc = pd.get_dummies(X_train, columns=['dep_airport'])[with_categorical_encoding]
    X_test_enc = pd.get_dummies(X_test, columns=['dep_airport'])[with_categorical_encoding]

    enc_dep_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    enc_dep_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Departure Delays (With Categorical Encoding):")
    train_regressor(enc_dep_rf_model, X_train_enc, y_train['dep_delay'], X_test_enc, y_test['dep_delay'], name="Random Forest", target="Departure", feat_set="With Categorical Encoding")
    print("\nTraining XGBoost Regressor for Departure Delays (With Categorical Encoding):")
    train_regressor(enc_dep_xgb_model, X_train_enc, y_train['dep_delay'], X_test_enc, y_test['dep_delay'], name="XGBoost", target="Departure", feat_set="With Categorical Encoding")

    enc_arr_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    enc_arr_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Arrival Delays (With Categorical Encoding):")
    train_regressor(enc_arr_rf_model, X_train_enc, y_train['arr_delay'], X_test_enc, y_test['arr_delay'], name="Random Forest", target="Arrival", feat_set="With Categorical Encoding")
    print("\nTraining XGBoost Regressor for Arrival Delays (With Categorical Encoding):")
    train_regressor(enc_arr_xgb_model, X_train_enc, y_train['arr_delay'], X_test_enc, y_test['arr_delay'], name="XGBoost", target="Arrival", feat_set="With Categorical Encoding")

    # train models with categorical encoding for time features
    X_train['dep_operational_bin'] = X_train['weather_timestamp_dep'].dt.hour.apply(get_operational_bin)
    X_test['dep_operational_bin'] = X_test['weather_timestamp_dep'].dt.hour.apply(get_operational_bin)
    X_train['arr_operational_bin'] = X_train['weather_timestamp_arr'].dt.hour.apply(get_operational_bin)
    X_test['arr_operational_bin'] = X_test['weather_timestamp_arr'].dt.hour.apply(get_operational_bin)

    with_time_encoding = (
        ['dep_operational_bin' + f'_{time}' for time in ['morning', 'midday', 'evening', 'night']]
        + ['arr_operational_bin' + f'_{time}' for time in ['morning', 'midday', 'evening', 'night']]
        + all_cols
    )

    X_train_time_enc = pd.get_dummies(X_train, columns=['arr_operational_bin', 'dep_operational_bin'])[with_time_encoding]
    X_test_time_enc = pd.get_dummies(X_test, columns=['arr_operational_bin', 'dep_operational_bin'])[with_time_encoding]

    time_enc_dep_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    time_enc_dep_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Departure Delays (With Time Encoding):")
    train_regressor(time_enc_dep_rf_model, X_train_time_enc, y_train['dep_delay'], X_test_time_enc, y_test['dep_delay'], name="Random Forest", target="Departure", feat_set="With Time Encoding")
    print("\nTraining XGBoost Regressor for Departure Delays (With Time Encoding):")
    train_regressor(time_enc_dep_xgb_model, X_train_time_enc, y_train['dep_delay'], X_test_time_enc, y_test['dep_delay'], name="XGBoost", target="Departure", feat_set="With Time Encoding")

    time_enc_arr_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    time_enc_arr_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Arrival Delays (With Time Encoding):")
    train_regressor(time_enc_arr_rf_model, X_train_time_enc, y_train['arr_delay'], X_test_time_enc, y_test['arr_delay'], name="Random Forest", target="Arrival", feat_set="With Time Encoding")
    print("\nTraining XGBoost Regressor for Arrival Delays (With Time Encoding):")
    train_regressor(time_enc_arr_xgb_model, X_train_time_enc, y_train['arr_delay'], X_test_time_enc, y_test['arr_delay'], name="XGBoost", target="Arrival", feat_set="With Time Encoding")

    # time encoding using sine and cosine transformations
    X_train_enc['dep_hour_sin'] = np.sin(2 * np.pi * X_train['weather_timestamp_deo'].dt.hour / 24)
    X_train_enc['dep_hour_cos'] = np.cos(2 * np.pi * X_train['weather_timestamp_deo'].dt.hour / 24)
    X_train_enc['arr_hour_sin'] = np.sin(2 * np.pi * X_train['weather_timestamp_arr'].dt.hour / 24)
    X_train_enc['arr_hour_cos'] = np.cos(2 * np.pi * X_train['weather_timestamp_arr'].dt.hour / 24)
    X_test_enc['dep_hour_sin'] = np.sin(2 * np.pi * X_test['weather_timestamp_deo'].dt.hour / 24)
    X_test_enc['dep_hour_cos'] = np.cos(2 * np.pi * X_test['weather_timestamp_deo'].dt.hour / 24)
    X_test_enc['arr_hour_sin'] = np.sin(2 * np.pi * X_test['weather_timestamp_arr'].dt.hour / 24)
    X_test_enc['arr_hour_cos'] = np.cos(2 * np.pi * X_test['weather_timestamp_arr'].dt.hour / 24)

    X_train_time_enc2 = X_train_enc # [all_cols + ['dep_hour_sin', 'dep_hour_cos', 'arr_hour_sin', 'arr_hour_cos']]
    X_test_time_enc2 = X_test_enc # [all_cols + ['dep_hour_sin', 'dep_hour_cos', 'arr_hour_sin', 'arr_hour_cos']]

    time_enc2_dep_rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    time_enc2_dep_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)  

    print("\nTraining Random Forest Regressor for Departure Delays (With Time Encoding 2):")
    train_regressor(time_enc2_dep_rf_model, X_train_time_enc2, y_train['dep_delay'], X_test_time_enc2, y_test['dep_delay'], name="Random Forest", target="Departure", feat_set="With Time Encoding 2")

    print("\nTraining XGBoost Regressor for Departure Delays (With Time Encoding 2):")
    train_regressor(time_enc2_dep_xgb_model, X_train_time_enc2, y_train['dep_delay'], X_test_time_enc2, y_test['dep_delay'], name="XGBoost", target="Departure", feat_set="With Time Encoding 2")

    time_enc2_arr_rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    time_enc2_arr_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("\nTraining Random Forest Regressor for Arrival Delays (With Time Encoding 2):")
    train_regressor(time_enc2_arr_rf_model, X_train_time_enc2, y_train['arr_delay'], X_test_time_enc2, y_test['arr_delay'], name="Random Forest", target="Arrival", feat_set="With Time Encoding 2")

    print("\nTraining XGBoost Regressor for Arrival Delays (With Time Encoding 2):")
    train_regressor(time_enc2_arr_xgb_model, X_train_time_enc2, y_train['arr_delay'], X_test_time_enc2, y_test['arr_delay'], name="XGBoost", target="Arrival", feat_set="With Time Encoding 2")

def plot_tournament_results():

    df_res = pd.DataFrame(RESULTS)

    fig, axes = plt.subplots(2, 2, figsize=(30, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    df_res['label'] = df_res['Model Name'] + " (" + df_res['Feature Set'] + ")"
    for target, ax in zip(['Departure', 'Arrival'], axes[0]):
        subset = df_res[df_res['Target'] == target].sort_values('MAE')
        ax.barh(subset['label'], subset['MAE'], color='skyblue')
        ax.set_title(f'Mean Absolute Error: {target}')
        ax.set_xlabel('Minutes')

    for target, ax in zip(['Departure', 'Arrival'], axes[1]):
        subset = df_res[df_res['Target'] == target].sort_values('R2')
        ax.barh(subset['label'], subset['R2'], color='salmon')
        ax.set_title(f'$R^2$ Score: {target}')
        ax.set_xlabel('Score')

    plt.savefig('model_tournament_metrics.png')

    best_model_info = df_res.iloc[df_res['R2'].idxmax()]
    plt.figure(figsize=(10, 6))
    
    if hasattr(best_model_info['Model'], 'feature_importances_'):
        importances = best_model_info['Model'].feature_importances_
        feat_names = best_model_info['Features']
        indices = np.argsort(importances)[-10:] # Top 10
        
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
        plt.title(f"Top 10 Features: {best_model_info['Model Name']} ({best_model_info['Feature Set']})")
        plt.tight_layout()
        plt.savefig('top_feature_importance.png')

def plot_delay_progression(df, airport_code="FRA"):

    subset = df[df['dep_airport'].str.upper() == airport_code.upper()].copy()
    
    if subset.empty:
        print(f"No data found for {airport_code}")
        return

    subset['hour'] = subset['dep_time_sched'].dt.hour
    
    hourly_stats = subset.groupby('hour')['dep_delay'].agg(['mean', 'median', 'std']).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(hourly_stats['hour'], hourly_stats['mean'], marker='o', linestyle='-', 
             linewidth=2, label='Mean Delay', color='#1f77b4')
    plt.fill_between(hourly_stats['hour'], 
                     hourly_stats['mean'] - (hourly_stats['std'] * 0.2),
                     hourly_stats['mean'] + (hourly_stats['std'] * 0.2), 
                     alpha=0.2, color='#1f77b4', label='Delay Volatility')

    plt.title(f'Delay Progression Throughout the Day at {airport_code}', fontsize=14)
    plt.xlabel('Hour of Day (24h format)', fontsize=12)
    plt.ylabel('Average Departure Delay (Minutes)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(f'{airport_code}_daily_progression.png')
    plt.show()

def prepare_fra_sequential_data(df):

    df = df.sort_values('dep_time_sched').copy()

    unique_keys = ['dep_airport', 'dep_time_sched', 'arr_time_sched', 'arr_airport']
    df = df.drop_duplicates(subset=unique_keys)

    df['hour'] = df['dep_time_sched'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['prev_10_avg_delay'] = df['dep_delay'].rolling(window=10, closed='left').mean()
    df = df.dropna(subset=['prev_10_avg_delay'])
    
    return df

def train_fra_sequence_tournament(fra_df):

    df_ready = prepare_fra_sequential_data(fra_df)
    
    features = ['hour_sin', 'hour_cos', 'prev_10_avg_delay']
    X = df_ready[features]
    y = np.log1p(df_ready['dep_delay']) 

    split_idx = int(len(df_ready) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)

    print("--- Training Sequential FRA Model (Log Transformed) ---")

    print(len(df_ready))
    
    print("\nRandom Forest (Sequential + Time):")
    train_regressor(rf_model, X_train, y_train, X_test, y_test, 
                    name="Random Forest", target="FRA Dep Delay", feat_set="Time + Prev 10")

    print("\nXGBoost (Sequential + Time):")
    train_regressor(xgb_model, X_train, y_train, X_test, y_test, 
                    name="XGBoost", target="FRA Dep Delay", feat_set="Time + Prev 10")

def prepare_fra_sequential_weather_data(fra_df, weather_df):
    # 1. Clean and Sort
    fra_df = fra_df.sort_values('dep_time_sched').copy()

    unique_keys = ['dep_airport', 'dep_time_sched', 'arr_time_sched', 'arr_airport']
    fra_df = fra_df.drop_duplicates(subset=unique_keys)
    
    # 2. Merge with Weather (FRA specific)
    # We floor to the hour to match the weather timestamp
    fra_df['dep_time_hour'] = fra_df['dep_time_sched'].dt.floor('H').dt.tz_localize(None)
    weather_df['weather_timestamp'] = weather_df['weather_timestamp'].dt.tz_localize(None)
    
    fra_weather = weather_df[weather_df['airport_iata'] == 'fra'].copy()
    
    df = pd.merge(
        fra_df, 
        fra_weather, 
        left_on='dep_time_hour', 
        right_on='weather_timestamp', 
        how='left'
    )

    # 3. Cyclical Time Encoding (Hour)
    df['hour'] = df['dep_time_sched'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 4. Cyclical Wind Encoding
    # 
    df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_direction_10m'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_direction_10m'] / 360)

    # 5. Previous 10 Flights Delay (Operational Momentum)
    # 
    df['prev_10_avg_delay'] = df['dep_delay'].rolling(window=10, closed='left').mean()
    
    # 6. Final Cleanup
    df = df.dropna(subset=['prev_10_avg_delay', 'temperature_2m'])
    
    return df

def train_fra_weather_sequence_tournament(fra_df, weather_df):
    df_ready = prepare_fra_sequential_weather_data(fra_df, weather_df)
    
    # Define expanded feature set
    weather_features = [
        'temperature_2m', 'precipitation', 'wind_speed_10m', 
        'wind_gusts_10m', 'cloudcover', 'wind_dir_sin', 'wind_dir_cos'
    ]
    sequential_features = ['hour_sin', 'hour_cos', 'prev_10_avg_delay']
    
    X = df_ready[weather_features + sequential_features]
    y = np.log1p(df_ready['dep_delay']) 

    # Temporal Split (80/20)
    split_idx = int(len(df_ready) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Models
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)

    print(f"--- Training RF FRA Model with Weather + Sequence ({len(df_ready)} flights) ---")
    
    train_regressor(rf_model, X_train, y_train, X_test, y_test, 
                    name="Random Forest", target="FRA Dep Delay", feat_set="Weather + Time + Prev 10")
    
    print(f"--- Training XGBoost FRA Model with Weather + Sequence ({len(df_ready)} flights) ---")

    train_regressor(xgb_model, X_train, y_train, X_test, y_test, 
                    name="XGBoost", target="FRA Dep Delay", feat_set="Weather + Time + Prev 10")

if __name__ == '__main__':

    # flights_df = get_flights_hopsworks()
    # get_weather(flights_df)
    train_and_eval_models(split_strategy="random")
    plot_tournament_results()

    fra_df = get_all_flights(airport="FRA", kind="departure")
    # # fra_df = fra_df.sort_values('dep_time_sched')
    # # print(fra_df[['dep_airport', 'arr_airport', 'dep_time_sched', 'arr_time_sched', 'flight_iata', 'airline', 'dep_delay']][:50])

    weather_df = get_weather(fra_df)
    train_fra_sequence_tournament(fra_df)
    train_fra_weather_sequence_tournament(fra_df, weather_df)
    
    if not fra_df.empty:
        # Plot the progression
        plot_delay_progression(fra_df, airport_code="FRA")