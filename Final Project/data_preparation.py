import json
from typing import Tuple, List

import numpy as np
import pandas as pd

from data_understanding import load_and_preprocess_data


def _ensure_polyline_list(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure POLYLINE column is a list of [lon, lat] pairs for every row."""
    if df.empty:
        return df
    if isinstance(df["POLYLINE"].iloc[0], str):
        df = df.copy()
        df["POLYLINE"] = df["POLYLINE"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(r * c)


def _path_length_km(poly: List[List[float]]) -> float:
    if not poly or len(poly) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(poly)):
        lon1, lat1 = poly[i - 1]
        lon2, lat2 = poly[i]
        dist += _haversine_km(lon1, lat1, lon2, lat2)
    return dist


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, dow, week, month, weekend from TIMESTAMP (seconds)."""
    df = df.copy()
    dt = pd.to_datetime(df["TIMESTAMP"], unit="s")
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["day_of_week"].isin([5, 6])).astype(int)
    return df


def build_training_table(train: pd.DataFrame, cut_len: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build destination training table by simulating partial trajectories.

    - Use first `cut_len` points as inputs (prefix) if trajectory is long enough.
    - Targets are the final destination coordinates from the full trajectory.
    - Include simple prefix-based geometric features and metadata signals.
    """
    train = _ensure_polyline_list(train)
    train = add_time_features(train)

    rows: List[dict] = []
    targets: List[List[float]] = []

    for _, row in train.iterrows():
        poly = row["POLYLINE"]
        if not poly or len(poly) <= cut_len:
            continue

        prefix = poly[:cut_len]
        dest_lon, dest_lat = poly[-1]
        start_lon, start_lat = prefix[0]
        last_lon, last_lat = prefix[-1]

        # prefix stats
        prefix_points = len(prefix)
        # According to dataset, sampling is ~15 seconds
        prefix_duration_s = max(0, (prefix_points - 1) * 15)
        prefix_length_km = _path_length_km(prefix)

        feature = {
            "TRIP_ID": row["TRIP_ID"],
            "CALL_TYPE": row.get("CALL_TYPE"),
            "ORIGIN_CALL": row.get("ORIGIN_CALL"),
            "ORIGIN_STAND": row.get("ORIGIN_STAND"),
            "TAXI_ID": row.get("TAXI_ID"),
            "hour": row.get("hour"),
            "day_of_week": row.get("day_of_week"),
            "weekofyear": row.get("weekofyear"),
            "month": row.get("month"),
            "is_weekend": row.get("is_weekend"),
            "start_lon": start_lon,
            "start_lat": start_lat,
            "last_lon": last_lon,
            "last_lat": last_lat,
            "prefix_points": prefix_points,
            "prefix_duration_s": prefix_duration_s,
            "prefix_length_km": prefix_length_km,
        }

        rows.append(feature)
        targets.append([dest_lon, dest_lat])

    X = pd.DataFrame(rows)
    y = pd.DataFrame(targets, columns=["dest_lon", "dest_lat"])
    return X, y


def save_training_tables(X: pd.DataFrame, y: pd.DataFrame, base_path: str = ".") -> None:
    X.to_csv(f"{base_path}/prep_X.csv", index=False)
    y.to_csv(f"{base_path}/prep_y.csv", index=False)


def build_test_table(test: pd.DataFrame, cut_len: int = 10) -> pd.DataFrame:
    """
    Build feature table for test by using available partial trajectories.

    - Use first `cut_len` points if possible; otherwise use as many points as available.
    - Mirrors training features: time features + prefix geometry + metadata.
    """
    test = _ensure_polyline_list(test)
    test = add_time_features(test)

    rows: List[dict] = []

    for _, row in test.iterrows():
        poly = row["POLYLINE"]
        if not poly:
            # skip empty trajectories (no points)
            continue

        prefix = poly[: cut_len] if len(poly) > cut_len else poly
        start_lon, start_lat = prefix[0]
        last_lon, last_lat = prefix[-1]

        prefix_points = len(prefix)
        prefix_duration_s = max(0, (prefix_points - 1) * 15)
        prefix_length_km = _path_length_km(prefix)

        feature = {
            "TRIP_ID": row.get("TRIP_ID"),
            "CALL_TYPE": row.get("CALL_TYPE"),
            "ORIGIN_CALL": row.get("ORIGIN_CALL"),
            "ORIGIN_STAND": row.get("ORIGIN_STAND"),
            "TAXI_ID": row.get("TAXI_ID"),
            "hour": row.get("hour"),
            "day_of_week": row.get("day_of_week"),
            "weekofyear": row.get("weekofyear"),
            "month": row.get("month"),
            "is_weekend": row.get("is_weekend"),
            "start_lon": start_lon,
            "start_lat": start_lat,
            "last_lon": last_lon,
            "last_lat": last_lat,
            "prefix_points": prefix_points,
            "prefix_duration_s": prefix_duration_s,
            "prefix_length_km": prefix_length_km,
        }

        rows.append(feature)

    X_test = pd.DataFrame(rows)
    return X_test


def main(cut_len: int = 10) -> None:
    train, test, sample, location = load_and_preprocess_data()
    X, y = build_training_table(train, cut_len=cut_len)
    save_training_tables(X, y)
    print(f"Prepared features: X={X.shape}, y={y.shape}; saved to prep_X.csv and prep_y.csv")


if __name__ == "__main__":
    main(cut_len=10)
