import os
from typing import Any

import pandas as pd


def weather_cache_key(cache_dir: str, icao: str, date_iso: str) -> str:
    safe = icao.upper().strip()
    return os.path.join(cache_dir, "weather", f"{safe}_{date_iso}.json")


def hourly_payload_to_df(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["time"] = df["time"].astype(int)
    return df.sort_values("time").reset_index(drop=True)


def interpolate_weather_at(df_hourly: pd.DataFrame, target_ts: int) -> dict[str, Any]:
    if df_hourly.empty:
        return {}

    if target_ts in set(df_hourly["time"].tolist()):
        row = df_hourly[df_hourly["time"] == target_ts].iloc[0].to_dict()
        row.pop("time", None)
        return row

    times = df_hourly["time"].values
    if target_ts < times[0] or target_ts > times[-1]:
        idx = 0 if target_ts < times[0] else (len(times) - 1)
        row = df_hourly.iloc[idx].to_dict()
        row.pop("time", None)
        return row

    pos = int(pd.Series(times).searchsorted(target_ts, side="left"))
    lo_idx = max(0, pos - 1)
    hi_idx = min(len(times) - 1, pos)

    t0 = int(df_hourly.iloc[lo_idx]["time"])
    t1 = int(df_hourly.iloc[hi_idx]["time"])
    if t1 == t0:
        row = df_hourly.iloc[lo_idx].to_dict()
        row.pop("time", None)
        return row

    alpha = (target_ts - t0) / (t1 - t0)

    out: dict[str, Any] = {}
    for col in df_hourly.columns:
        if col == "time":
            continue
        v0 = df_hourly.iloc[lo_idx][col]
        v1 = df_hourly.iloc[hi_idx][col]

        if pd.isna(v0) and pd.isna(v1):
            out[col] = None
            continue
        if col == "weather_code":
            out[col] = int(v0) if alpha < 0.5 else int(v1)
            continue

        if pd.isna(v0):
            out[col] = float(v1) if v1 is not None else None
        elif pd.isna(v1):
            out[col] = float(v0) if v0 is not None else None
        else:
            try:
                out[col] = (1 - alpha) * float(v0) + alpha * float(v1)
            except Exception:
                out[col] = float(v0)
    return out


def weather_snapshots(df_hourly: pd.DataFrame, event_ts: int, prefix: str) -> dict[str, Any]:
    stamps = {
        "t0": event_ts,
        "tminus15": event_ts - 15 * 60,
        "tminus30": event_ts - 30 * 60,
    }
    out: dict[str, Any] = {}
    for label, ts in stamps.items():
        wx = interpolate_weather_at(df_hourly, ts)
        for key, value in wx.items():
            out[f"{prefix}_{label}_{key}"] = value
    return out


__all__ = [
    "hourly_payload_to_df",
    "interpolate_weather_at",
    "weather_cache_key",
    "weather_snapshots",
]
