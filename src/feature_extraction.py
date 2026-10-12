"""
feature_extraction.py — Keystroke Dynamics and Circadian Feature Engineering

Extracts temporal, statistical, and circadian features from raw keystroke telemetry
to capture fine-grained motor degradation in Parkinson's Disease.

Key Innovation:
    - Circadian HoldTime Fluctuation: models Levodopa-induced 'On/Off' medication cycles
      by measuring the divergence between morning and evening typing behaviors.
    - HoldTime Asymmetry: measures left vs. right hand timing discrepancies to capture
      unilateral motor impairment.

Author: Fayez Ahmad Protik
"""

import os
import glob
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def compute_statistical_moments(values: List[float], prefix: str) -> Dict[str, float]:
    """Compute mean, standard deviation, skewness, and kurtosis for a signal.

    Args:
        values: List of numeric values (e.g. hold times or transition latencies).
        prefix: Column prefix for feature naming.

    Returns:
        Dictionary mapping feature names to their statistical moments.
    """
    if not values or len(values) < 2:
        return {
            f"{prefix}_Mean": 0.0,
            f"{prefix}_Std": 0.0,
            f"{prefix}_Skew": 0.0,
            f"{prefix}_Kurt": 0.0,
        }

    arr = np.array(values, dtype=np.float64)
    return {
        f"{prefix}_Mean": float(np.mean(arr)),
        f"{prefix}_Std": float(np.std(arr)),
        f"{prefix}_Skew": float(skew(arr)),
        f"{prefix}_Kurt": float(kurtosis(arr)),
    }


def extract_user_keystroke_features(
    user_files: List[str],
    max_hold_time_ms: float = 2000.0,
    min_keystrokes: int = 50,
) -> Optional[Dict[str, float]]:
    """Parse raw TSV keystroke logs for a single user and engineer summary features.

    Args:
        user_files: List of file paths containing raw keystrokes for the user.
        max_hold_time_ms: Threshold to filter out non-typing key rests (default 2000ms).
        min_keystrokes: Minimum required keystrokes to retain user in analysis.

    Returns:
        Dictionary of aggregated features or None if user has insufficient data.
    """
    hold_times_l = []
    hold_times_r = []
    latencies = defaultdict(list)
    morning_hold_times = []
    evening_hold_times = []
    total_keystrokes = 0

    col_names = ["UserID", "Date", "Time", "Hand", "HoldTime", "Direction", "Latency1", "Latency2"]

    for file_path in user_files:
        try:
            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names, on_bad_lines="skip")
            df["HoldTime"] = pd.to_numeric(df["HoldTime"], errors="coerce")
            df["Latency1"] = pd.to_numeric(df["Latency1"], errors="coerce")
            df = df.dropna(subset=["HoldTime"])
            df = df[df["HoldTime"] <= max_hold_time_ms]

            total_keystrokes += len(df)

            # Left and right hand hold times
            hold_times_l.extend(df.loc[df["Hand"] == "L", "HoldTime"].tolist())
            hold_times_r.extend(df.loc[df["Hand"] == "R", "HoldTime"].tolist())

            # Transition latencies by direction (LL, RR, LR, RL)
            for direction in ["LL", "RR", "LR", "RL"]:
                subset = df.loc[df["Direction"] == direction, "Latency1"].dropna()
                subset = subset[(subset > 0) & (subset <= 2000)]
                latencies[direction].extend(subset.tolist())

            # Circadian feature extraction (Morning: 06:00-11:59, Evening: 18:00-23:59)
            if "Time" in df.columns:
                df["Hour"] = df["Time"].astype(str).str.split(":").str[0].astype(float)
                morning_ht = df.loc[(df["Hour"] >= 6) & (df["Hour"] < 12), "HoldTime"].tolist()
                evening_ht = df.loc[(df["Hour"] >= 18) & (df["Hour"] <= 23), "HoldTime"].tolist()
                morning_hold_times.extend(morning_ht)
                evening_hold_times.extend(evening_ht)

        except Exception:
            continue

    if total_keystrokes < min_keystrokes:
        return None

    features = {"Total_Keystrokes": total_keystrokes}
    features.update(compute_statistical_moments(hold_times_l, "HoldTime_L"))
    features.update(compute_statistical_moments(hold_times_r, "HoldTime_R"))

    for direction in ["LL", "RR", "LR", "RL"]:
        features.update(compute_statistical_moments(latencies[direction], f"Latency_{direction}"))

    # Novel feature: Hand HoldTime Asymmetry
    mean_l = features["HoldTime_L_Mean"]
    mean_r = features["HoldTime_R_Mean"]
    features["HoldTime_Asymmetry"] = abs(mean_l - mean_r)

    # Novel feature: Circadian HoldTime Fluctuation
    m_mean = np.mean(morning_hold_times) if morning_hold_times else mean_l
    e_mean = np.mean(evening_hold_times) if evening_hold_times else mean_r
    features["Circadian_HoldTime_Diff"] = abs(float(m_mean) - float(e_mean))

    return features
