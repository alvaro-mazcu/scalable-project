#!/usr/bin/env python3
"""Visualise predicted delays for flights arriving at Stockholm Arlanda."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

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
FORECAST_OFFSETS_MIN = [0, 30, 60, 120, 180]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="data/arlanda_scenarios.json", help="JSON file with upcoming flight scenarios.")
    parser.add_argument("--model-dir", default="models", help="Directory where joblib models are stored.")
    parser.add_argument("--latest-file", default="lastest_model.txt", help="File containing the last trained model name.")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory for weather payloads.")
    parser.add_argument(
        "--openmeteo-url",
        default="https://archive-api.open-meteo.com/v1/archive",
        help="Open-Meteo archive endpoint.",
    )
    parser.add_argument(
        "--hourly-vars",
        default="temperature_2m,relative_humidity_2m,precipitation,cloud_cover,visibility,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m,weather_code",
        help="Comma-separated hourly weather variables to request.",
    )
    parser.add_argument("--save", help="Optional path to save the figure instead of showing it interactively.")
    return parser.parse_args()


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


def gather_arlanda_weather(client: OpenMeteoClient) -> Dict[int, Dict[str, float]]:
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
        payload = client.ensure_weather_for_day(DEST_ICAO, DEST_COORDS[0], DEST_COORDS[1], iso)
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
    return (
        f"{stamp}: {desc}, "
        f"Temp {temp:.1f}°C, Wind {wind:.1f} m/s, Precip {precip:.1f} mm"
        if temp is not None and wind is not None and precip is not None
        else f"{stamp}: {desc}"
    )


def plot_routes(data: List[Tuple[FlightScenario, float, str]], avg_delay: float, weather_lines: List[str], save_path: str | None) -> None:
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=proj)
    ax.set_extent([-25, 45, 35, 75], crs=proj)
    ax.add_feature(cfeature.OCEAN, facecolor="#d1ecff")
    ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--", alpha=0.6)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title("Predicted Delays for Incoming Flights to Stockholm Arlanda", fontsize=14, fontweight="bold")

    handles = {}
    dest_lat, dest_lon = DEST_COORDS
    ax.scatter(dest_lon, dest_lat, color="#004b91", s=80, marker="*", label="ESSA (Arlanda)", transform=proj)

    for scenario, _, bucket in data:
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
    ax.text(
        0.02,
        0.97,
        f"Average predicted delay: {avg_min:.1f} min",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    weather_text = "Current & future weather at Arlanda:\n" + "\n".join(weather_lines)
    ax.text(
        0.02,
        0.83,
        weather_text,
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[PLOT] Saved to {save_path}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    scenarios = load_scenarios(args.scenarios)
    scenarios = [s for s in scenarios if s.dest_icao.upper() == DEST_ICAO]
    if not scenarios:
        raise RuntimeError("No scenarios targeting ESSA were provided.")

    client = OpenMeteoClient(
        base_url=args.openmeteo_url,
        hourly_vars=args.hourly_vars,
        cache_dir=args.cache_dir,
    )
    pipeline, model_name = load_model(args.model_dir, args.latest_file)
    print(f"[INFO] Using model {model_name}")

    predictions = build_predictions(scenarios, client, pipeline)
    avg_delay = sum(pred for _, pred, _ in predictions) / len(predictions)

    weather_map = gather_arlanda_weather(client)
    weather_lines = [
        format_weather(ts, wx)
        for ts, wx in sorted(weather_map.items())
        if wx
    ]

    plot_routes(predictions, avg_delay, weather_lines, args.save)


if __name__ == "__main__":
    main()
