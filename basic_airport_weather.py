import pandas as pd
import numpy as np
from datetime import datetime
# Use the lowercase 'hourly' and 'Point' or 'Station'
from meteostat import hourly, Point, Station

# 1. Define Airports using their official Meteostat Station IDs
# CPH: 06180, LHR: 03772, ARN: 02460, OSL: 01492, AMS: 06240, FRA: 10637
airports_config = {
    'CPH': '06180',
    'LHR': '03772',
    'ARN': '02460',
    'OSL': '01492',
    'AMS': '06240',
    'FRA': '10637'
}

weather_dict = {}
# Try a safe historical range (2024 instead of future 2026 for testing)
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 5)

for code, station_id in airports_config.items():
    print(f"Downloading weather for {code} (ID: {station_id})...")
    
    # Use the Station ID directly
    data = hourly(station_id, start, end)
    df = data.fetch()
    
    # 2. FIX: Check if data is empty before processing
    if df is None or df.empty:
        print(f"  Warning: No data found for {code}. Creating empty fallback.")
        weather_dict[code] = pd.DataFrame(columns=['temp', 'wspd'])
    else:
        # Interpolate to fill small gaps (missing hours)
        weather_dict[code] = df[['temp', 'wspd']].interpolate(method='linear').ffill().bfill()

# 3. Dummy Flight Data for Testing
flight_df = pd.DataFrame({
    'flight_no': ['SK402', 'BA814'],
    'origin': ['OSL', 'LHR'],
    'sched_arr': pd.to_datetime(['2024-01-02 10:30', '2024-01-02 14:15'])
})

# 4. Corrected Lookup Logic
def get_weather_features(row):
    time = row['sched_arr']
    origin = row['origin']
    
    # Get CPH Weather (Destination)
    cph_df = weather_dict.get('CPH')
    # Use asof merge logic or simple reindexing
    try:
        cph_row = cph_df.iloc[cph_df.index.get_indexer([time], method='pad')[0]]
        cph_temp, cph_wind = cph_row['temp'], cph_row['wspd']
    except (IndexError, KeyError):
        cph_temp, cph_wind = np.nan, np.nan

    # Get Origin Weather
    origin_df = weather_dict.get(origin)
    try:
        if origin_df is not None and not origin_df.empty:
            orig_row = origin_df.iloc[origin_df.index.get_indexer([time], method='pad')[0]]
            orig_temp, orig_wind = orig_row['temp'], orig_row['wspd']
        else:
            orig_temp, orig_wind = np.nan, np.nan
    except:
        orig_temp, orig_wind = np.nan, np.nan

    return pd.Series([cph_temp, cph_wind, orig_temp, orig_wind], 
                     index=['cph_temp', 'cph_wind', 'origin_temp', 'origin_wind'])

# Merge features back to flight data
final_df = pd.concat([flight_df, flight_df.apply(get_weather_features, axis=1)], axis=1)
print("\nFinal Model Input Data:")
print(final_df)