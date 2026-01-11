#!/usr/bin/env python3
"""Streamlit dashboard for monitoring departure delays."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import textwrap
from typing import Dict, List, Tuple

import hopsworks
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(".env")

DEFAULT_AIRPORTS: Dict[str, Dict[str, float]] = {
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

WEATHER_CODE_DESC = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Freezing drizzle",
    57: "Freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Freezing rain",
    67: "Freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm w/ hail",
    99: "Severe thunderstorm",
}

WEATHER_CODE_EMOJI = {
    0: "☀️",
    1: "🌤️",
    2: "⛅",
    3: "☁️",
    45: "🌫️",
    48: "🌫️",
    51: "🌦️",
    53: "🌧️",
    55: "🌧️",
    56: "🌧️",
    57: "🌧️",
    61: "🌧️",
    63: "🌧️",
    65: "🌧️",
    66: "🌨️",
    67: "🌨️",
    71: "❄️",
    73: "❄️",
    75: "❄️",
    80: "🌧️",
    81: "🌧️",
    82: "🌧️",
    95: "⛈️",
    96: "⛈️",
    99: "⛈️",
}


def load_airports() -> Dict[str, Dict[str, float]]:
    raw = os.getenv("AIRPORTS_JSON", "").strip()
    if not raw:
        return DEFAULT_AIRPORTS
    try:
        payload = json.loads(raw)
        return {k.upper(): {"lat": v["lat"], "lon": v["lon"]} for k, v in payload.items()}
    except Exception as exc:
        st.warning(f"Failed to parse AIRPORTS_JSON: {exc}. Using defaults.")
        return DEFAULT_AIRPORTS


AIRPORTS = load_airports()


def set_page_config() -> None:
    st.set_page_config(page_title="Departure Delay Monitor", layout="wide", page_icon="✈️")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        * { font-family: "Share Tech Mono", "DIN Alternate", "Segoe UI", system-ui, sans-serif; }
        .weather-grid { display: flex; gap: 0.75rem; flex-wrap: nowrap; overflow-x: auto; padding-bottom: 0.25rem; }
        .weather-card { background: #0f172a; color: #e2e8f0; padding: 0.75rem 0.9rem; border-radius: 10px; min-width: 180px; box-shadow: 0 6px 16px rgba(15, 23, 42, 0.25); }
        .weather-card h4 { margin: 0 0 0.35rem 0; font-size: 0.95rem; }
        .weather-card p { margin: 0.1rem 0; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_hopsworks_project():
    return hopsworks.login(
        engine="python",
        project=os.getenv("HOPSWORKS_PROJECT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    )


@st.cache_data(ttl=300)
def load_predictions(_project, airport_code: str) -> pd.DataFrame:
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name="daily_inference_predictions_fg", version=2)
    if fg is None:
        return pd.DataFrame()
    df = fg.read()
    if df.empty:
        return df
    df["dep_airport"] = df["dep_airport"].astype(str).str.upper()
    df = df[df["dep_airport"] == airport_code].copy()
    df["dep_time_sched"] = pd.to_datetime(df["dep_time_sched"], errors="coerce", utc=True)
    df["arr_time_sched"] = pd.to_datetime(df["arr_time_sched"], errors="coerce", utc=True)
    df = df.sort_values("dep_time_sched", ascending=False)

    return df


@st.cache_data(ttl=600)
def fetch_weather(lat: float, lon: float, hours: int = 24) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,weather_code",
        "timezone": "UTC",
        "forecast_days": 2,
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly", {})
    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    now = dt.datetime.now(dt.timezone.utc)
    df = df[df["timestamp"] >= now].head(hours)
    return df


def jitter_location(code: str, base_lat: float, base_lon: float) -> Tuple[float, float]:
    digest = hashlib.md5(code.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    angle = math.radians(seed % 360)
    radius_m = 250 + (seed % 750)
    delta_lat = radius_m / 111_320
    delta_lon = radius_m / (111_320 * max(math.cos(math.radians(base_lat)), 0.3))
    return base_lat + delta_lat * math.sin(angle), base_lon + delta_lon * math.cos(angle)


def global_location(code: str) -> Tuple[float, float]:
    digest = hashlib.md5(code.encode("utf-8")).hexdigest()
    seed = int(digest[:12], 16)
    lat = (seed % 18000) / 100.0 - 90.0
    lon = ((seed // 18000) % 36000) / 100.0 - 180.0
    return lat, lon


def wind_direction(deg: float | None) -> str:
    if deg is None or pd.isna(deg):
        return "?"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((deg + 22.5) % 360 // 45)
    arrows = ["⬆️", "↗️", "➡️", "↘️", "⬇️", "↙️", "⬅️", "↖️"]
    return f"{dirs[ix]} {arrows[ix]}"


def build_map(df: pd.DataFrame, airport_code: str, coords: Dict[str, float]) -> go.Figure:
    base_lat = coords["lat"]
    base_lon = coords["lon"]

    points = []
    for ix, row in df.iterrows():
        code = str(row.get("flight_iata", "")) or f"{row.get('dep_time_sched', '')}" or str(ix)
        lat, lon = global_location(code)
        points.append((lat, lon))

    st.write(f"Map points requested: {len(points)}")

    if points:
        df = df.copy()
        df["map_lat"] = pd.to_numeric([p[0] for p in points], errors="coerce")
        df["map_lon"] = pd.to_numeric([p[1] for p in points], errors="coerce")
        df = df.dropna(subset=["map_lat", "map_lon"])
        st.write(f"Map points after dropna: {len(df)}")
        if not df.empty:
            st.write(
                f"Map lat range: {df['map_lat'].min():.2f} to {df['map_lat'].max():.2f}, "
                f"lon range: {df['map_lon'].min():.2f} to {df['map_lon'].max():.2f}"
            )

    hover_cols = [
        "flight_iata",
        "airline",
        "dep_airport",
        "arr_airport",
        "dep_time_sched",
        "dep_delay",
        "predicted_dep_delay",
        "temperature_2m_dep",
        "wind_speed_10m_dep",
        "precipitation_dep",
        "weather_code_dep",
    ]
    for col in hover_cols:
        if col not in df.columns:
            df[col] = None

    df = df.fillna("n/a")
    df["hover_text"] = (
        "Flight: " + df["flight_iata"].astype(str)
        + "<br>Airline: " + df["airline"].astype(str)
        + "<br>Route: " + df["dep_airport"].astype(str) + " → " + df["arr_airport"].astype(str)
        + "<br>Scheduled: " + df["dep_time_sched"].astype(str)
        + "<br>Actual delay: " + df["dep_delay"].astype(str) + " min"
        + "<br>Predicted delay: " + df["predicted_dep_delay"].astype(str) + " min"
        + "<br>Temp: " + df["temperature_2m_dep"].astype(str) + " °C"
        + "<br>Wind: " + df["wind_speed_10m_dep"].astype(str) + " m/s"
        + "<br>Precip: " + df["precipitation_dep"].astype(str) + " mm"
        + "<br>Weather code: " + df["weather_code_dep"].astype(str)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=[base_lat],
            lon=[base_lon],
            mode="markers",
            marker={"size": 16, "color": "#2563eb"},
            name=f"{airport_code} airport",
        )
    )

    if not df.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=df["map_lat"].astype(float).tolist(),
                lon=df["map_lon"].astype(float).tolist(),
                mode="markers",
                marker={"size": 14, "color": "#f97316", "opacity": 0.95},
                text=df["hover_text"],
                hovertemplate="%{text}<extra></extra>",
                name="Departures",
            )
        )

    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {"lat": 0, "lon": 0},
            "zoom": 1.4,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        height=520,
        showlegend=False,
    )
    return fig


def color_predicted(val: float) -> str:
    if pd.isna(val):
        return ""
    if val < 5:
        color = "#1b7f5c"
    elif val < 15:
        color = "#d97706"
    else:
        color = "#b91c1c"
    return f"background-color: {color}; color: #f8fafc;"


def main() -> None:
    set_page_config()

    st.title("Departure Delay Monitor")
    airport_codes = list(AIRPORTS.keys())
    selected_airport = st.selectbox("Select departure airport", airport_codes, index=0)
    coords = AIRPORTS[selected_airport]

    st.subheader("Weather report")
    weather_df = fetch_weather(coords["lat"], coords["lon"], hours=24)
    if weather_df.empty:
        st.info("Weather data is not available.")
    else:
        cards = []
        for _, row in weather_df.iterrows():
            stamp = row["timestamp"].strftime("%H:%M UTC")
            code = int(row.get("weather_code")) if pd.notna(row.get("weather_code")) else None
            desc = WEATHER_CODE_DESC.get(code, "Unknown")
            emoji = WEATHER_CODE_EMOJI.get(code, "🌍")
            wind = wind_direction(row.get("wind_direction_10m"))
            cards.append(
                textwrap.dedent(
                    f"""
                    <div class="weather-card">
                      <h4>{stamp}</h4>
                      <p>{emoji} {desc}</p>
                      <p>Temp: {row.get('temperature_2m', 'n/a')} °C</p>
                      <p>Wind: {row.get('wind_speed_10m', 'n/a')} m/s {wind}</p>
                      <p>Precip: {row.get('precipitation', 'n/a')} mm</p>
                    </div>
                    """
                ).strip()
            )
        st.markdown('<div class="weather-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    st.subheader("Departure activity map")
    project = get_hopsworks_project()
    predictions = load_predictions(project, selected_airport)

    now = dt.datetime.now(dt.timezone.utc)
    if not predictions.empty:
        window_end = now + dt.timedelta(hours=24)
        window_df = predictions[
            (predictions["dep_time_sched"] >= now) & (predictions["dep_time_sched"] <= window_end)
        ].copy()
        if window_df.empty:
            window_df = predictions.head(200).copy()
    else:
        window_df = predictions

    display_cols = [
        "flight_iata",
        "airline",
        "dep_airport",
        "arr_airport",
        "dep_time_sched",
        "predicted_dep_delay",
    ]
    for col in display_cols:
        if col not in predictions.columns:
            predictions[col] = None

    display_df = predictions.copy()
    display_df["arr_airport"] = display_df["arr_airport"].astype(str).str.upper()
    display_df = display_df.sort_values("dep_time_sched", ascending=True).reset_index(drop=True)
    styled = display_df[display_cols].style.map(color_predicted, subset=["predicted_dep_delay"])

    fig = build_map(display_df, selected_airport, coords)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Departure predictions")
    if predictions.empty:
        st.info("No predictions available in daily_inference_predictions_fg.")
        return

    st.dataframe(styled, use_container_width=True, height=420)


if __name__ == "__main__":
    main()
