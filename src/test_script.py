import pandas as pd
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# --- CONFIGURATION ---
AV_EDGE_KEY = os.getenv("EDGE_API_KEY")
START_DATE = "2025-12-06"
END_DATE = "2026-01-01"
DESTINATION = "CPH"

# Origins we want to track
AIRPORTS = {
    "LHR": {"lat": 51.4700, "lon": -0.4543},
    "FRA": {"lat": 50.0333, "lon": 8.5705},
    "AMS": {"lat": 52.3086, "lon": 4.7639}
}

def get_all_cph_arrivals():
    """Fetch ALL historical arrivals at CPH within the date range."""
    url = "https://aviation-edge.com/v2/public/flightsHistory"
    params = {
        "key": AV_EDGE_KEY,
        "code": DESTINATION, # Fetching for CPH
        "type": "arrival",
        "date_from": START_DATE,
        "date_to": END_DATE,
    }

    print(f"Fetching all arrivals for {DESTINATION}...")
    res = requests.get(url, params=params)

    if res.status_code == 200:
        return res.json()
    else:
        print(f"Error fetching data: {res.status_code}")
        return []

def get_weather(lat, lon, date):
    """Fetch historical weather via Open-Meteo."""
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={date}&end_date={date}&hourly=temperature_2m,precipitation,wind_speed_10m"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json().get('hourly', {})
    return {}

# 1. FETCH ALL DATA ONCE
all_flights = get_all_cph_arrivals()

print(all_flights)

# 2. FILTER AND MERGE WITH WEATHER
all_data = []

# Loop through the flights we just downloaded
for f in all_flights:
    origin_code = f.get('departure', {}).get('iataCode', '').upper()
    
    # Filter: Only proceed if the origin is in our AIRPORTS list
    if origin_code in AIRPORTS:
        coords = AIRPORTS[origin_code]
        
        # Extract timing
        dep_time_str = f['departure']['scheduledTime']
        dep_dt = datetime.fromisoformat(dep_time_str.replace('Z', ''))
        date_str = dep_dt.strftime('%Y-%m-%d')
        hour = dep_dt.hour
        
        # Fetch weather for the specific origin at that time
        weather = get_weather(coords['lat'], coords['lon'], date_str)
        
        if weather and 'temperature_2m' in weather:
            all_data.append({
                'origin': origin_code,
                'hour': hour,
                'day_of_week': dep_dt.weekday(),
                'temp': weather['temperature_2m'][hour],
                'precip': weather['precipitation'][hour],
                'wind': weather['wind_speed_10m'][hour],
                'dep_delay': f['departure'].get('delay', 0) or 0,
                'arr_delay': f['arrival'].get('delay', 0) or 0
            })

# Convert to DataFrame
df = pd.DataFrame(all_data)

# 3. DATA PREPARATION & TRAINING (The rest remains the same)
if not df.empty:
    df = pd.get_dummies(df, columns=['origin'])
    X = df.drop(['dep_delay', 'arr_delay'], axis=1)
    y = df['dep_delay']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Model Training Complete.")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, preds):.2f} minutes")
else:
    print("No data found for the specified origins.")