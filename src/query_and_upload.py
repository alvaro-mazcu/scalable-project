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
START_DATE = "2025-01-11"
END_DATE = "2026-01-07"

# origins
AIRPORTS = {
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

def call_edge(date_from: str, date_to: str = None, airport: str = "CPH", kind: str = "departure"):

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

def get_all_flights(airport: str = "CPH", kind: str = "departure") -> pd.DataFrame:

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
    unique_keys = ['dep_airport', 'dep_time_sched', 'arr_time_sched', 'arr_airport']
    df = df.drop_duplicates(subset=unique_keys).reset_index(drop=True)

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
            # 'dep_time_actual': dep.get('actualTime'),
            'dep_delay': dep.get('delay') or 0,
            'arr_airport': arr.get('iataCode'),
            'arr_time_sched': arr.get('scheduledTime'),
            # 'arr_time_actual': arr.get('actualTime'),
            'arr_delay': arr.get('delay') or 0
        })
    
    df = pd.DataFrame(data)

    time_cols = [
        'dep_time_sched', # 'dep_time_actual', 
        'arr_time_sched', # 'arr_time_actual'
    ]
    
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

        # bad = df[col].isna().any(axis=1)
        # df = df.loc[~bad].copy()

    df = df.dropna(subset=time_cols).reset_index(drop=True)

    return df

def get_every_airport_data_and_upload(kind: str = "departure", project=None):

    for airport_code in AIRPORTS.keys():
        print(f"Fetching data for airport: {airport_code}")
        airport_df = get_all_flights(airport=airport_code, kind=kind)
        upload_flights_hopsworks(airport_df, project=project)

def upload_flights_hopsworks(flights_df: pd.DataFrame, project=None):

    fs = project.get_feature_store()

    flights_fg = fs.get_or_create_feature_group(
        name=f"european_flights_fg",
        version=2,
        description="Flight data with weather features",
        primary_key=["flight_iata", "dep_airport"],
        event_time="dep_time_sched",
    )

    flights_fg.insert(flights_df, write_options={"wait_for_job": False})

def get_weather(start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    
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

def upload_weather_hopsworks(weather_df: pd.DataFrame, project=None):

    fs = project.get_feature_store()

    weather_fg = fs.get_or_create_feature_group(
        name="european_flights_weather_fg",
        version=1,
        description="Weather data for European airports",
        primary_key=["airport_iata", "weather_timestamp"],
    )

    weather_fg.insert(weather_df, write_options={"wait_for_job": False})

def main():
    
    project = hopsworks.login(
        engine="python",
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )

    print("Fetching departure data...")
    get_every_airport_data_and_upload(kind="departure", project=project)

    # print("Fetching weather data...")
    # weather_df = get_weather()
    # print(f"Total weather records fetched: {len(weather_df)}")

    # print("Uploading weather data to Hopsworks...")
    # upload_weather_hopsworks(weather_df, project=project)
    # print("Weather data upload complete.")

if __name__ == "__main__":
    main()
