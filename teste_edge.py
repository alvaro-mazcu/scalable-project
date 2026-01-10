import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
API_KEY = os.getenv("EDGE_API_KEY")

def get_historical_flights():
    # Aviation Edge Historical API Endpoint
    url = "https://aviation-edge.com/v2/public/flightsHistory"
    
    # Define parameters based on your requirements
    params = {
        "key": API_KEY,
        "code": "CPH",            # Copenhagen Airport IATA
        "type": "arrival",        # Looking for arrivals
        "date_from": "2025-12-30", # Target date
        "airline_iata": "LH"      # Lufthansa IATA code
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Check for HTTP errors
        
        flights = response.json()
        
        if not flights or "error" in flights:
            print("No data found or API error:", flights.get("error", "Unknown error"))
            return

        print(f"--- Lufthansa Arrivals in CPH on 2026-01-01 ---")
        for flight in flights:
            flight_num = flight.get('flight', {}).get('iataNumber', 'N/A')
            status = flight.get('status', 'N/A')
            arr_time = flight.get('arrival', {}).get('scheduledTime', 'N/A')
            
            print(f"Flight: {flight_num} | Status: {status} | Scheduled Arrival: {arr_time}")

        print(flight)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_historical_flights()