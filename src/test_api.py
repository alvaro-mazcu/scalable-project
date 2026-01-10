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
END_DATE = "2025-02-07"

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

        print(len(chunk_data) if chunk_data else "0", f"records fetched from {from_str} to {to_str}")
        
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
        'dep_time_sched', #'dep_time_actual', 
        'arr_time_sched', #'arr_time_actual'
    ]
    
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

        # bad = df[col].isna().any(axis=1)
        # df = df.loc[~bad].copy()

    # Rows where ANY column is NaN
    df_with_nas = df[df.isna().any(axis=1)]

    print(f"Total rows with at least one missing value: {len(df_with_nas)}")

    # Print the first 10 rows that have missing data to inspect them
    print(df_with_nas.head(10))

    df = df.dropna(subset=time_cols).reset_index(drop=True)

    return df

if __name__ == "__main__":

    df = get_all_flights(airport="CPH", kind="departure")

    print(len(df), "total records fetched.")
    print(df['dep_time_sched'][0])