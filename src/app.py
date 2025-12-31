#!/usr/bin/env python3
"""Streamlit dashboard for monitoring inbound flights to Stockholm Arlanda."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from typing import Dict, List, Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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


def set_page_config() -> None:
    st.set_page_config(
        page_title="Arlanda Inbound Flights",
        layout="wide",
        page_icon="✈️",
    )
    st.markdown(
        """
        <style>
        * { font-family: "DIN Alternate", "Segoe UI", system-ui, sans-serif; }
        .css-18e3th9, .css-1d391kg { background-color: #f8fbff; }
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


def build_predictions(
    scenarios: List[FlightScenario],
    client: OpenMeteoClient,
    pipeline,
) -> List[Tuple[FlightScenario, float, str]]:
    rows: List[Dict[str, float]] = []
    for scenario in scenarios:
        features = build_feature_row(client, scenario)
        rows.append(features)

    df_features = pd.DataFrame(rows)
    outputs: List[Tuple[FlightScenario, float, str]] = []
    for scenario, feature_map in zip(scenarios, df_features.to_dict(orient="records")):
        vector = ensure_feature_columns(pipeline, feature_map)
        df_vec = pd.DataFrame([vector])
        pred = float(pipeline.predict(df_vec)[0])
        bucket = categorize_delay(pred)
        outputs.append((scenario, pred, bucket))
    return outputs


def make_map(predictions: List[Tuple[FlightScenario, float, str]], avg_delay: float, weather_lines: List[str]):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection=proj)
    ax.set_extent([-25, 45, 35, 75], crs=proj)
    ax.add_feature(cfeature.OCEAN, facecolor="#d1ecff")
    ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--", alpha=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title("Predicted delays for inbound flights to Stockholm Arlanda", fontsize=12, fontweight="bold")

    dest_lat, dest_lon = DEST_COORDS
    ax.scatter(dest_lon, dest_lat, color="#004b91", s=80, marker="*", label="ESSA (Arlanda)", transform=proj)

    handles = {}
    for scenario, _, bucket in predictions:
        color = COLOR_MAP.get(bucket, "gray")
        ax.plot(
            [scenario.origin_lon, dest_lon],
            [scenario.origin_lat, dest_lat],
            color=color,
            linewidth=1.5,
            alpha=0.9,
            transform=ccrs.Geodetic(),
        )
        handles[bucket] = color

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=label)
        for label, color in COLOR_MAP.items()
        if label in handles
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="*", color="w", label="ESSA", markerfacecolor="#004b91", markersize=12))
    ax.legend(handles=legend_handles, loc="lower left")

    avg_min = avg_delay / 60.0
    ax.text(0.02, 0.97, f"Average predicted delay: {avg_min:.1f} min", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax.text(
        0.02,
        0.83,
        "Current & future weather at Arlanda:\n" + "\n".join(weather_lines),
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
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
    avg_delay = sum(pred for _, pred, _ in predictions) / len(predictions)

    with st.spinner("Fetching Arlanda weather..."):
        weather_map = gather_arlanda_weather(client)
    weather_lines = [format_weather(ts, wx) for ts, wx in sorted(weather_map.items()) if wx]

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = make_map(predictions, avg_delay, weather_lines)
        st.pyplot(fig)
    with col2:
        st.subheader("Flights")
        for scenario, pred, bucket in predictions:
            st.markdown(
                f"**{scenario.flight_id}** {scenario.origin_icao} → {scenario.dest_icao}  \n"
                f"Predicted delay: **{pred/60:.1f} min** ({bucket})"
            )
        st.divider()
        st.subheader("Arlanda outlook")
        for line in weather_lines:
            st.write(line)


if __name__ == "__main__":
    main()
