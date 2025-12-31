#!/usr/bin/env python3
"""Streamlit dashboard for monitoring inbound flights to Stockholm Arlanda."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.clients.openmeteo import OpenMeteoClient
from src.predict_live_delay import (
    FlightScenario,
    build_feature_row,
    categorize_delay,
    ensure_feature_columns,
    load_model,
    load_scenarios,
)
from src.utils.weather import hourly_payload_to_df, interpolate_weather_at

DEST_ICAO = "ESSA"
DEST_COORDS = (59.6498, 17.9238)
FORECAST_OFFSETS_MIN = [0, 30, 60, 120, 180]
COLOR_MAP = {
    "no delay expected": "#2e8b57",
    "5-15 minutes": "#ff8c00",
    "over 15 minutes": "#cc2936",
}
WEATHER_CODE_DESC = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
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


def weather_code_emoji(code: int) -> str:
    table = {
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
    return table.get(code, "🌍")


def wind_direction(deg: float | None) -> str:
    if deg is None or math.isnan(deg):
        return "?"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((deg + 22.5) % 360 // 45)
    arrows = ["⬆️", "↗️", "➡️", "↘️", "⬇️", "↙️", "⬅️", "↖️"]
    return f"{dirs[ix]} {arrows[ix]}"


def heading_arrow(lat_start: float, lon_start: float, lat_end: float, lon_end: float) -> str:
    dx = lon_end - lon_start
    dy = lat_end - lat_start
    angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    directions = ["→", "↗️", "↑", "↖️", "←", "↙️", "↓", "↘️"]
    idx = int((angle + 22.5) // 45) % 8
    return directions[idx]


def set_page_config() -> None:
    st.set_page_config(
        page_title="Arlanda Inbound Flights",
        layout="wide",
        page_icon="✈️",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        * { font-family: "Share Tech Mono", "DIN Alternate", "Segoe UI", system-ui, sans-serif; }
        .css-18e3th9, .css-1d391kg { background-color: #f8fbff; }
        .timeline-row { display: flex; gap: 1rem; overflow-x: auto; padding: 0.5rem 0; }
        .timeline-item { min-width: 110px; display: flex; flex-direction: column; align-items: center; color: #0d1b2a; }
        .timeline-dot { width: 6px; height: 6px; background: #0d1b2a; border-radius: 50%; margin-bottom: 0.25rem; }
        .timeline-emoji { font-size: 1.5rem; }
        .timeline-time { font-size: 0.9rem; font-weight: bold; }
        .timeline-meta { font-size: 0.8rem; opacity: 0.8; text-align: center; }
        .warning-card { border-left: 5px solid #cc2936; padding: 0.75rem 0.9rem; background: #1c1f26; border-radius: 8px; color: #f5f7fa; box-shadow: 0 4px 16px rgba(0,0,0,0.25); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_scenarios_cached(path: str | None) -> List[FlightScenario]:
    return load_scenarios(path)


@st.cache_resource
def load_pipeline(model_dir: str, latest_file: str):
    return load_model(model_dir, latest_file)


@st.cache_data
def gather_arlanda_weather(_client: OpenMeteoClient) -> Dict[int, Dict[str, float]]:
    now = dt.datetime.now(dt.timezone.utc)
    timestamps = [int((now + dt.timedelta(minutes=offset)).timestamp()) for offset in FORECAST_OFFSETS_MIN]
    unique_dates = set()
    for ts in timestamps:
        date = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).date()
        unique_dates.add(date)
        unique_dates.add(date - dt.timedelta(days=1))
    frames: List[pd.DataFrame] = []
    for date in sorted(unique_dates):
        iso = date.isoformat()
        payload = _client.ensure_weather_for_day(DEST_ICAO, DEST_COORDS[0], DEST_COORDS[1], iso)
        frames.append(hourly_payload_to_df(payload))
    hourly = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    result: Dict[int, Dict[str, float]] = {}
    for ts in timestamps:
        result[ts] = interpolate_weather_at(hourly, ts)
    return result


@st.cache_data
def fetch_hourly_forecast(lat: float, lon: float, hourly_vars: str, hours: int = 15) -> List[Dict[str, float]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "timezone": "UTC",
        "forecast_days": 2,
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    out: List[Dict[str, float]] = []
    now = dt.datetime.now(dt.timezone.utc)
    for i, t in enumerate(times):
        stamp = dt.datetime.fromisoformat(t.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        else:
            stamp = stamp.astimezone(dt.timezone.utc)
        if stamp < now:
            continue
        record = {"time": stamp}
        for var, series in hourly.items():
            if var == "time":
                continue
            if i < len(series):
                record[var] = series[i]
        out.append(record)
        if len(out) >= hours:
            break
    return out


def format_weather(ts: int, wx: Dict[str, float]) -> str:
    stamp = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%H:%M UTC")
    temp = wx.get("temperature_2m")
    precip = wx.get("precipitation")
    wind = wx.get("wind_speed_10m")
    code = int(wx.get("weather_code", -1)) if wx.get("weather_code") is not None else None
    desc = WEATHER_CODE_DESC.get(code, "Unknown") if code is not None else "Unknown"
    parts = [f"{stamp}: {desc}"]
    if temp is not None:
        parts.append(f"Temp {temp:.1f}°C")
    if wind is not None:
        parts.append(f"Wind {wind:.1f} m/s")
    if precip is not None:
        parts.append(f"Precip {precip:.1f} mm")
    return " | ".join(parts)


def summarize_weather(feature_map: Dict[str, float]) -> str:
    temp = feature_map.get("wx_dest_t0_temperature_2m")
    wind = feature_map.get("wx_dest_t0_wind_speed_10m")
    code = feature_map.get("wx_dest_t0_weather_code")
    desc = WEATHER_CODE_DESC.get(int(code), "Unknown") if code is not None else "Unknown"
    parts = [desc]
    if temp is not None:
        parts.append(f"{temp:.1f}°C")
    if wind is not None:
        parts.append(f"{wind:.1f} m/s")
    return ", ".join(parts)


def build_predictions(
    scenarios: List[FlightScenario],
    client: OpenMeteoClient,
    pipeline,
) -> List[Tuple[FlightScenario, float, str, str]]:
    rows: List[Dict[str, float]] = []
    for scenario in scenarios:
        features = build_feature_row(client, scenario)
        rows.append(features)

    df_features = pd.DataFrame(rows)
    outputs: List[Tuple[FlightScenario, float, str, str]] = []
    for scenario, feature_map in zip(scenarios, df_features.to_dict(orient="records")):
        vector = ensure_feature_columns(pipeline, feature_map)
        df_vec = pd.DataFrame([vector])
        pred = float(pipeline.predict(df_vec)[0])
        bucket = categorize_delay(pred)
        landing_weather = summarize_weather(feature_map)
        outputs.append((scenario, pred, bucket, landing_weather))
    return outputs


def make_map(predictions: List[Tuple[FlightScenario, float, str, str]]):
    fig = go.Figure()
    mapbox_token = os.getenv("MAPBOX_TOKEN", "")
    map_style = "satellite-streets" if mapbox_token else "open-street-map"

    origins_lat: List[float] = []
    origins_lon: List[float] = []
    origins_text: List[str] = []
    plane_lat: List[float] = []
    plane_lon: List[float] = []
    plane_text: List[str] = []
    plane_hover: List[str] = []

    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()

    for scenario, pred, bucket, landing_weather in predictions:
        color = COLOR_MAP.get(bucket, "gray")
        eta = dt.datetime.fromtimestamp(scenario.landing_ts, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        hover = (
            f"Flight: {scenario.flight_id}<br>"
            f"Route: {scenario.origin_icao} → {scenario.dest_icao}<br>"
            f"ETA: {eta}<br>"
            f"Predicted delay: {pred/60:.1f} min ({bucket})<br>"
            f"Landing wx: {landing_weather}"
        )
        fig.add_trace(
            go.Scattermapbox(
                lon=[scenario.origin_lon, DEST_COORDS[1]],
                lat=[scenario.origin_lat, DEST_COORDS[0]],
                mode="lines",
                line=dict(width=3, color=color),
                hoverinfo="skip",
                name=bucket,
                opacity=0.4,
            )
        )
        origins_lat.append(scenario.origin_lat)
        origins_lon.append(scenario.origin_lon)
        origins_text.append(f"{scenario.origin_icao}")

        # plane marker placement
        if now_ts <= scenario.takeoff_ts:
            prog = 0.0
            plane_emoji = "🛫"
        elif now_ts >= scenario.landing_ts:
            prog = 1.0
            plane_emoji = "✈️"
        else:
            prog = (now_ts - scenario.takeoff_ts) / max(1, scenario.landing_ts - scenario.takeoff_ts)
            plane_emoji = "✈️"

        plane_latitude = scenario.origin_lat + prog * (DEST_COORDS[0] - scenario.origin_lat)
        plane_longitude = scenario.origin_lon + prog * (DEST_COORDS[1] - scenario.origin_lon)
        arrow = heading_arrow(plane_latitude, plane_longitude, DEST_COORDS[0], DEST_COORDS[1])
        plane_lat.append(plane_latitude)
        plane_lon.append(plane_longitude)
        plane_text.append(f"{plane_emoji}{arrow}")
        plane_hover.append(hover)

    # Destination marker
    fig.add_trace(
        go.Scattermapbox(
            lon=[DEST_COORDS[1]],
            lat=[DEST_COORDS[0]],
            mode="markers",
            marker=dict(size=28, color="#0d6efd", symbol="circle"),
            hoverinfo="text",
            text=["ESSA (Arlanda)"],
            name="ESSA",
        )
    )

    if origins_lat:
        fig.add_trace(
            go.Scattermapbox(
                lon=origins_lon,
                lat=origins_lat,
                mode="markers",
                marker=dict(size=6, color="#050505"),
                hoverinfo="text",
                text=origins_text,
                name="Origins",
            )
        )

    if plane_lat:
        fig.add_trace(
            go.Scattermapbox(
                lon=plane_lon,
                lat=plane_lat,
                mode="markers",
                marker=dict(size=16, color="#f8d54c", symbol="circle"),
                hoverinfo="skip",
                name="Aircraft marker",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattermapbox(
                lon=plane_lon,
                lat=plane_lat,
                mode="text",
                text=plane_text,
                textfont=dict(size=26, color="#1a1a1a"),
                textposition="middle center",
                hoverinfo="text",
                hovertext=plane_hover,
                name="Aircraft",
            )
        )

    fig.update_layout(
        mapbox=dict(
            style=map_style,
            accesstoken=mapbox_token or None,
            center=dict(lat=55, lon=15),
            zoom=2.7,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        paper_bgcolor="#0b0f16",
        plot_bgcolor="#0b0f16",
        showlegend=False,
    )
    return fig


def main() -> None:
    set_page_config()
    st.title("Stockholm Arlanda | Inbound Delay Outlook")
    st.caption("Live weather + model predictions for arriving flights. Powered by Open-Meteo, OpenSky, and your trained model.")

    st.sidebar.header("Settings")
    scenarios_file = st.sidebar.text_input("Scenarios JSON", "data/arlanda_scenarios.json")
    model_dir = st.sidebar.text_input("Models directory", "models")
    latest_file = st.sidebar.text_input("Latest model file", "lastest_model.txt")
    cache_dir = st.sidebar.text_input("Weather cache dir", "cache")
    openmeteo_url = st.sidebar.text_input("Open-Meteo URL", "https://archive-api.open-meteo.com/v1/archive")
    hourly_vars = st.sidebar.text_input(
        "Hourly weather vars",
        "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,visibility,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code",
    )

    try:
        scenarios = load_scenarios_cached(scenarios_file)
        scenarios = [s for s in scenarios if s.dest_icao.upper() == DEST_ICAO]
        if not scenarios:
            st.error("No scenarios targeting ESSA were provided.")
            return
    except Exception as exc:
        st.error(f"Could not load scenarios: {exc}")
        return

    try:
        pipeline, model_name = load_pipeline(model_dir, latest_file)
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    st.sidebar.success(f"Model loaded: {model_name}")

    client = OpenMeteoClient(
        base_url=openmeteo_url,
        hourly_vars=hourly_vars,
        cache_dir=cache_dir,
    )

    with st.spinner("Scoring flights..."):
        predictions = build_predictions(scenarios, client, pipeline)
    if not predictions:
        st.error("No predictions available. Check scenarios or model.")
        return

    avg_delay = sum(pred for _, pred, _, _ in predictions) / len(predictions)

    with st.spinner("Fetching Arlanda weather..."):
        weather_map = gather_arlanda_weather(client)
    weather_lines = [format_weather(ts, wx) for ts, wx in sorted(weather_map.items()) if wx]

    # Top forecast section
    st.subheader("Arlanda hourly weather (next 15h)")
    forecast = fetch_hourly_forecast(DEST_COORDS[0], DEST_COORDS[1], hourly_vars, hours=15)
    cards_html = []
    for entry in forecast[:15]:
        code = int(entry.get("weather_code", -1)) if entry.get("weather_code") is not None else -1
        emoji = weather_code_emoji(code)
        time_str = entry["time"].strftime("%H:%M UTC")
        temp = float(entry.get("temperature_2m") or 0.0)
        wind = float(entry.get("wind_speed_10m") or 0.0)
        wind_dir = wind_direction(entry.get("wind_direction_10m")) if "wind_direction_10m" in entry else "?"
        precip = float(entry.get("precipitation") or 0.0)
        cards_html.append(
            f"""
            <div class="timeline-item">
              <div class="timeline-emoji">{emoji}</div>
              <div class="timeline-time">{time_str}</div>
              <div class="timeline-meta">
                <strong>Temp</strong> {temp:.1f}°C<br/>
                <strong>Wind</strong> {wind:.1f} m/s ({wind_dir})<br/>
                <strong>Precip</strong> {precip:.1f} mm
              </div>
            </div>
            """
        )
    forecast_html = f"""
    <style>
    .timeline-wrapper {{
        display: flex;
        flex-direction: row;
        gap: 1rem;
        overflow-x: auto;
        padding-bottom: 0.5rem;
    }}
    .timeline-wrapper::-webkit-scrollbar {{
        height: 6px;
    }}
    .timeline-wrapper::-webkit-scrollbar-thumb {{
        background: #0d1b2a;
        border-radius: 4px;
    }}
    .timeline-item {{
        min-width: 150px;
        display: flex;
        flex-direction: column;
        align-items: center;
        color: #0d1b2a;
        background: #ffffff;
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
        font-family: "Segoe UI", system-ui, sans-serif;
    }}
    .timeline-emoji {{
        font-size: 1.8rem;
        margin: 0.2rem 0;
    }}
    .timeline-time {{
        font-size: 1rem;
        font-weight: bold;
    }}
    .timeline-meta {{
        font-size: 0.85rem;
        text-align: center;
    }}
    </style>
    <div class="timeline-wrapper">{''.join(cards_html)}</div>
    """
    st.components.v1.html(forecast_html, height=240, scrolling=False)

    # Warning section
    st.subheader("Attention: highest predicted delays")
    worst = sorted(predictions, key=lambda x: x[1], reverse=True)[:4]
    cols = st.columns(len(worst) or 1)
    for col, (scenario, pred, bucket, landing_weather) in zip(cols, worst):
        with col:
            st.markdown(
                f"""
                <div class="warning-card">
                  <strong>{scenario.flight_id}</strong><br/>{scenario.origin_icao} → {scenario.dest_icao}<br/>
                  Status: Forecasted arrival<br/>
                  Predicted delay: <strong>{pred/60:.1f} min</strong> ({bucket})<br/>
                  Expected landing: {dt.datetime.fromtimestamp(scenario.landing_ts, tz=dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}<br/>
                  Landing wx: {landing_weather}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Inbound map (hover for details)")
    map_predictions = [
        p
        for p in predictions
        if -25 <= p[0].origin_lon <= 45 and 30 <= p[0].origin_lat <= 75
    ][:12]
    if not map_predictions:
        st.warning("No flights within the Europe viewport; showing all instead.")
        map_predictions = predictions[:12]
    if not os.getenv("MAPBOX_TOKEN"):
        st.info("Set the MAPBOX_TOKEN environment variable to unlock the satellite basemap. Falling back to OpenStreetMap tiles.")
    fig = make_map(map_predictions)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
