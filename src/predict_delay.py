#!/usr/bin/env python3
"""Simple regression model that predicts flight time deviations using weather features."""

from __future__ import annotations

import argparse
import math
from typing import List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

TARGET_COLUMN = "flight_time_diff_sec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="sweden_flights.csv", help="Path to the enriched flights CSV created by initial_dataset.py")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of samples to use for evaluation (0-1).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the train/test split.")
    parser.add_argument(
        "--model",
        choices=["linear", "random_forest", "xgboost"],
        default="linear",
        help="Regression model to train (default: linear).",
    )
    return parser.parse_args()


def select_weather_columns(df: pd.DataFrame) -> List[str]:
    weather_cols = [c for c in df.columns if c.startswith("wx_")]
    usable = [c for c in weather_cols if df[c].notna().any()]
    if not usable:
        raise RuntimeError("No weather-derived columns found in the dataset. Run initial_dataset.py with weather enrichment enabled.")
    return usable


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise RuntimeError(f"Required column '{TARGET_COLUMN}' not present. Regenerate the dataset with the latest initial_dataset.py.")
    df = df[df[TARGET_COLUMN].notna()].copy()
    if df.empty:
        raise RuntimeError("No rows with a valid flight_time_diff_sec target were found.")
    return df


def build_pipeline(model_kind: str, random_state: int) -> Pipeline:
    if model_kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
        )
    elif model_kind == "xgboost":
        model = HistGradientBoostingRegressor(
            max_depth=7,
            learning_rate=0.05,
            max_iter=400,
            random_state=random_state,
            max_leaf_nodes=None,
        )
    else:
        model = LinearRegression()

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("model", model),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    weather_cols: List[str],
    test_size: float,
    random_state: int,
    model_kind: str,
) -> None:
    X = df[weather_cols].astype(float)
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    pipeline = build_pipeline(model_kind, random_state)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"[MODEL] Trained {model_kind.replace('_', ' ').title()} on weather features only")
    print(f"Samples: train={len(X_train)}, test={len(X_test)}, features={len(weather_cols)}")
    print(f"MAE:  {mae:0.2f} sec")
    print(f"RMSE: {rmse:0.2f} sec")
    print(f"R^2:  {r2:0.3f}")



def main() -> None:
    args = parse_args()
    df = load_dataset(args.csv)
    weather_cols = select_weather_columns(df)
    train_and_evaluate(
        df,
        weather_cols,
        test_size=args.test_size,
        random_state=args.random_state,
        model_kind=args.model,
    )


if __name__ == "__main__":
    main()
