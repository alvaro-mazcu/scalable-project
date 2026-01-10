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

FUTURE_DATE = "2026-01-11"

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

def get_schedule(origin, kind="departure"):
    
    url = "https://aviation-edge.com/v2/public/timetable"
    params = {
        "key": API_KEY,
        "iataCode": origin,
        "type": kind,
        # "date": FUTURE_DATE,
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

def create_dataframe_schedule(flights):

    records = []
    for flight in flights:
        record = {
            "flight_iata": flight.get("flight", {}).get("iataNumber"),
            "airline": flight.get("airline", {}).get("name"),
            "flight_number": flight.get("flight", {}).get("number"),
            "dep_airport": flight.get("departure", {}).get("iataCode"),
            "dep_sched_time": flight.get("departure", {}).get("scheduledTime"),
            "arr_airport": flight.get("arrival", {}).get("iataCode"),
            "arr_sched_time": flight.get("arrival", {}).get("scheduledTime"),
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":

    fra_flights = get_schedule("FRA", kind="departure")
    df_fra = create_dataframe_schedule(fra_flights)
    print(df_fra.head())
