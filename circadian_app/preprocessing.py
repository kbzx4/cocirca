"""Signal preprocessing: accelerometer activity extraction (ENMO / MAD / Vector Magnitude) and circadian epoch binning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .io_loader import RawSensorDataset

ActivityMethod = Literal["enmo", "mad", "vector_magnitude", "raw"]


@dataclass
class BinnedCircadianSeries:
    """Uniformly binned circadian activity & light time series."""

    df: pd.DataFrame  # columns: timestamp, elapsed_hours, day_index, clock_hour, activity, lux, log_lux, is_missing
    epoch_minutes: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    n_days: float
    activity_method: str
    has_activity: bool
    has_light: bool


MIN_EPOCH_SECONDS = 1
MAX_EPOCH_SECONDS = 6 * 3600


def epoch_seconds(epoch_minutes: float) -> int:
    """Convert a bin size in minutes to whole seconds, clamped to the supported 1 s – 6 h range."""
    sec = int(round(float(epoch_minutes) * 60.0))
    return int(min(MAX_EPOCH_SECONDS, max(MIN_EPOCH_SECONDS, sec)))


def format_bin_size(epoch_minutes: float) -> str:
    """Human-readable bin size, e.g. '1 s', '30 s', '10 min', '1 h 30 min', '6 h'."""
    sec = epoch_seconds(epoch_minutes)
    if sec < 60:
        return f"{sec} s"
    if sec < 3600:
        m, s = divmod(sec, 60)
        return f"{m} min" if s == 0 else f"{m} min {s} s"
    h, rem = divmod(sec, 3600)
    m = rem // 60
    return f"{h} h" if m == 0 else f"{h} h {m} min"


def compute_sample_activity(
    df: pd.DataFrame,
    method: ActivityMethod = "enmo",
) -> pd.Series:
    """Compute instantaneous activity magnitude from 3-axis accelerometer or precomputed activity column."""
    if "activity_raw" in df.columns and (method == "raw" or not all(c in df.columns for c in ("ax", "ay", "az"))):
        return pd.to_numeric(df["activity_raw"], errors="coerce").clip(lower=0.0)

    if not all(c in df.columns for c in ("ax", "ay", "az")):
        return pd.Series(np.nan, index=df.index)

    ax = pd.to_numeric(df["ax"], errors="coerce")
    ay = pd.to_numeric(df["ay"], errors="coerce")
    az = pd.to_numeric(df["az"], errors="coerce")
    vm = np.sqrt(ax**2 + ay**2 + az**2)

    # Detect whether sensor already excludes gravity (Linear Acceleration) or is in g vs m/s^2
    is_linear = bool(df["is_linear_accel"].dropna().iloc[0]) if "is_linear_accel" in df.columns and df["is_linear_accel"].notna().any() else False
    if is_linear or method == "vector_magnitude":
        return vm

    from .formats import accel_g_units

    med_vm = float(vm.dropna().median()) if vm.notna().any() else 9.80665
    gravity = accel_g_units(med_vm)[0]

    if method == "enmo":
        # Euclidean Norm Minus One (standard actigraphy metric, scaled to milli-g or m/s^2)
        enmo = np.maximum(0.0, vm - gravity)
        # Convert to milli-g if in g or m/s^2 so values have intuitive circadian scale
        return enmo * (1000.0 / gravity)
    elif method == "mad":
        # Mean Amplitude Deviation within local rolling window or absolute deviation from gravity
        return np.abs(vm - gravity) * (1000.0 / gravity)
    return vm


def preprocess_to_epochs(
    dataset: RawSensorDataset,
    epoch_minutes: float = 10,
    activity_method: ActivityMethod = "enmo",
    interpolate_short_gaps_min: int = 60,
    start_at_midnight: bool = False,
) -> BinnedCircadianSeries:
    """Resample raw sensor observations into uniform circadian epochs (default 10 min, range 1 s – 6 h)."""
    epoch_minutes = epoch_seconds(epoch_minutes) / 60.0
    raw = dataset.df.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    has_act = dataset.has_activity
    has_light = dataset.has_light

    if has_act:
        raw["_act"] = compute_sample_activity(raw, method=activity_method)
    else:
        raw["_act"] = np.nan

    if has_light and "lux" in raw.columns:
        raw["_lux"] = pd.to_numeric(raw["lux"], errors="coerce").clip(lower=0.0)
    else:
        raw["_lux"] = np.nan

    raw = raw.set_index("timestamp")
    freq = f"{epoch_seconds(epoch_minutes)}s"

    agg_dict = {}
    if has_act:
        # For ENMO/MAD/raw activity, mean intensity per epoch; if raw looks sparse/event-like, mean preserves rate
        agg_dict["_act"] = "mean"
    if has_light:
        agg_dict["_lux"] = "mean"

    binned = raw.resample(freq).agg(agg_dict)

    if start_at_midnight and not binned.empty:
        t0 = binned.index.min().normalize()
        t1 = binned.index.max()
        full_idx = pd.date_range(t0, t1, freq=freq)
        binned = binned.reindex(full_idx)

    binned = binned.rename(columns={"_act": "activity", "_lux": "lux"})
    if "activity" not in binned.columns:
        binned["activity"] = np.nan
    if "lux" not in binned.columns:
        binned["lux"] = np.nan

    binned["is_missing_activity"] = binned["activity"].isna()
    binned["is_missing_light"] = binned["lux"].isna()

    # Interpolate short missing gaps (e.g. <= 60 min) for continuous actograms/FFT while keeping gap flags
    max_gap_bins = max(0, int(interpolate_short_gaps_min // epoch_minutes))
    if max_gap_bins > 0:
        if has_act:
            binned["activity"] = binned["activity"].interpolate(method="time", limit=max_gap_bins)
        if has_light:
            binned["lux"] = binned["lux"].interpolate(method="time", limit=max_gap_bins)

    binned["log_lux"] = np.log10(binned["lux"].clip(lower=0.0) + 1.0)

    binned = binned.reset_index().rename(columns={"index": "timestamp"})
    t_start = pd.Timestamp(binned["timestamp"].iloc[0])
    t_midnight = t_start.normalize()
    # `timestamp` is the bin start; elapsed_hours / clock_hour are the bin CENTRE, since each value is the
    # mean over [start, start + epoch). Using the start would bias every fitted phase early by half a bin
    # (5 min at 10-min bins, but 3 h at 6-h bins).
    half_bin_h = epoch_minutes / 120.0
    binned["elapsed_hours"] = (binned["timestamp"] - t_midnight).dt.total_seconds() / 3600.0 + half_bin_h
    binned["day_index"] = ((binned["timestamp"].dt.normalize() - t_midnight).dt.days + 1).astype(int)
    binned["date"] = binned["timestamp"].dt.strftime("%Y-%m-%d")
    binned["clock_hour"] = (
        binned["timestamp"].dt.hour
        + binned["timestamp"].dt.minute / 60.0
        + binned["timestamp"].dt.second / 3600.0
        + half_bin_h
    )

    total_hours = (binned["timestamp"].iloc[-1] - binned["timestamp"].iloc[0]).total_seconds() / 3600.0 + epoch_minutes / 60.0

    return BinnedCircadianSeries(
        df=binned,
        epoch_minutes=epoch_minutes,
        start_time=t_start,
        end_time=pd.Timestamp(binned["timestamp"].iloc[-1]),
        n_days=float(total_hours / 24.0),
        activity_method=activity_method,
        has_activity=has_act,
        has_light=has_light,
    )
