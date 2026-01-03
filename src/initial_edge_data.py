from dotenv import load_dotenv
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import json

load_dotenv()
API_KEY = os.getenv("EDGE_API_KEY")

def get_historical_flights(date_from: str, date_to: str = None):

    url = "https://aviation-edge.com/v2/public/flightsHistory"
    params = {
        "key": API_KEY,
        "code": "CPH",
        "type": "arrival",
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

def get_full_year_data(start_date_str: str, end_date_str: str):

    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    total_days = (end_dt - start_dt).days
    num_chunks = (total_days // 30) + 1
    
    all_flight_records = []
    current_start = start_dt

    # Initialize the progress bar
    pbar = tqdm(total=num_chunks, desc="Fetching Yearly Data")

    while current_start < end_dt:
        # Define the chunk (max 30 days)
        current_end = min(current_start + timedelta(days=30), end_dt)
        
        # Convert back to strings for your existing function
        from_str = current_start.strftime("%Y-%m-%d")
        to_str = current_end.strftime("%Y-%m-%d")
        
        # Call your existing function
        chunk_data = get_historical_flights(date_from=from_str, date_to=to_str)
        
        if chunk_data and isinstance(chunk_data, list):
            all_flight_records.extend(chunk_data)
        
        # Increment to the next chunk (start the next day after current_end)
        current_start = current_end + timedelta(days=1)
        pbar.update(1)
        
        # Small sleep to prevent API rate limiting
        time.sleep(0.2)

    pbar.close()

    # Process into DataFrame
    if not all_flight_records:
        print("No data was retrieved.")
        return pd.DataFrame()

    df = create_dataframe_flights(all_flight_records)

    # DEDUPLICATION: Use flight number + scheduled departure to identify unique flights
    before_count = len(df)
    df = df.drop_duplicates(subset=['flight_iata', 'dep_time_sched'])
    after_count = len(df)

    print(f"Ingestion complete. Removed {before_count - after_count} duplicate rows.")
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

def get_coordinates_airport(iata_code: str) -> tuple:
    url = "https://aviation-edge.com/v2/public/airportDatabase"
    params = {
        "key": API_KEY,
        "codeIataAirport": iata_code
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        airports = response.json()
        if not airports or "error" in airports:
            print(f"No data found for airport {iata_code} or API error:", airports.get("error", "Unknown error"))
            return None, None
        
        airport = airports[0]
        lat = airport.get('latitudeAirport')
        lon = airport.get('longitudeAirport')
        return lat, lon

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None
    
CACHE_FILE = "airport_coords_cache.json"

def get_coordinates_with_cache(iata_code: str):
    # 1. Load existing cache if it exists
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)

    # 2. Return from cache if available
    if iata_code in cache:
        return cache[iata_code]['lat'], cache[iata_code]['lon']

    # 3. If not in cache, call Aviation Edge (Your existing logic)
    print(f"IATA {iata_code} not in cache. Calling Aviation Edge API...")
    lat, lon = get_coordinates_airport(iata_code) # Calling your existing function
    
    if lat and lon:
        # 4. Save to cache for next time
        cache[iata_code] = {'lat': lat, 'lon': lon}
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
            
    return lat, lon
    
def get_weather_bulk(flights_df):
    start_date = flights_df['dep_time_sched'].min().strftime('%Y-%m-%d')
    end_date = flights_df['dep_time_sched'].max().strftime('%Y-%m-%d')
    
    unique_airports = flights_df['dep_airport'].unique()
    
    weather_frames = []
    coord_records = []
    
    for iata in unique_airports:
        # Use our new cached function
        lat, lon = get_coordinates_with_cache(iata)
        
        if lat is None or lon is None:
            continue
            
        # Store for the coordinates dataframe
        coord_records.append({'airport_iata': iata, 'latitude': lat, 'longitude': lon})

        # Fetch Weather (Same logic as before)
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        weather_params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date, "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,visibility",
            "timezone": "UTC"
        }

        time.sleep(0.5)  # To respect API rate limits
        
        try:
            res = requests.get(weather_url, params=weather_params)
            res.raise_for_status()
            w_data = res.json().get('hourly', {})
            
            temp_df = pd.DataFrame(w_data)
            temp_df['airport_iata'] = iata
            temp_df.rename(columns={'time': 'weather_timestamp'}, inplace=True)
            weather_frames.append(temp_df)
        except Exception as e:
            print(f"Error fetching weather for {iata}: {e}")

    final_weather_df = pd.concat(weather_frames, ignore_index=True)
    final_weather_df['weather_timestamp'] = pd.to_datetime(final_weather_df['weather_timestamp'])
    
    final_coords_df = pd.DataFrame(coord_records)
    
    return final_weather_df, final_coords_df