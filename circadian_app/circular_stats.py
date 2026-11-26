"""Circular statistics recorder (mean direction, R-bar, kappa, Rayleigh test, Watson-Williams test) and Acrophase Shift analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .cosinor import format_hours_hhmm


def hours_to_radians(hours: np.ndarray, period_hours: float = 24.0) -> np.ndarray:
    """Convert clock hours [0, 24) to circular angles in radians [0, 2*pi)."""
    return (2.0 * np.pi * (np.asarray(hours, dtype=float) % period_hours)) / period_hours


def radians_to_hours(rad: float, period_hours: float = 24.0) -> float:
    """Convert angle in radians to hours in [0, period_hours)."""
    return float((rad % (2.0 * np.pi)) / (2.0 * np.pi) * period_hours)


def wrap_phase_shift_hours(delta_hours: np.ndarray | float, period_hours: float = 24.0) -> np.ndarray | float:
    """Wrap phase difference into (-period/2, +period/2]. Positive = phase delay, negative = phase advance."""
    half = period_hours / 2.0
    wrapped = ((np.asarray(delta_hours, dtype=float) + half) % period_hours) - half
    # Map exact -half to +half for canonical (-12, +12]
    wrapped = np.where(np.isclose(wrapped, -half), half, wrapped)
    return float(wrapped) if np.ndim(delta_hours) == 0 else wrapped


def estimate_von_mises_kappa(r_bar: float, n: int) -> float:
    """Estimate von Mises concentration parameter kappa using Best & Fisher (1981) approximation."""
    r = float(np.clip(r_bar, 0.0, 0.999999))
    if r < 0.53:
        kappa = 2.0 * r + r**3 + (5.0 * r**5) / 6.0
    elif r < 0.85:
        kappa = -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    else:
        kappa = 1.0 / (r**3 - 4.0 * r**2 + 3.0 * r)
    if n <= 15 and n > 0:
        if kappa < 2.0:
            kappa = max(0.0, kappa - 2.0 / (n * max(kappa, 1e-6)))
        else:
            kappa = ((n - 1) ** 3 * kappa) / (n**3 + n)
    return float(max(0.0, kappa))


@dataclass
class CircularSummary:
    """Circular statistics summary for a collection of daily phase angles."""

    label: str
    n: int
    period_hours: float
    mean_angle_rad: float
    mean_angle_deg: float
    mean_hour: float
    mean_hhmm: str
    mean_resultant_length: float  # R-bar in [0, 1]
    circular_variance: float  # 1 - R-bar
    circular_std_rad: float
    circular_std_deg: float
    circular_std_hours: float
    kappa: float
    rayleigh_z: float
    rayleigh_p: float
    v_test_u: Optional[float] = None
    v_test_p: Optional[float] = None


@dataclass
class TwoSampleCircularTest:
    """Watson-Williams two-sample circular test comparing baseline vs. post-shift phases."""

    n1: int
    n2: int
    mean1_hour: float
    mean1_hhmm: str
    mean2_hour: float
    mean2_hhmm: str
    r_bar1: float
    r_bar2: float
    phase_shift_hours: float
    phase_shift_deg: float
    shift_direction: str
    f_stat: float
    p_value: float


@dataclass
class AcrophaseShiftReport:
    """Comprehensive acrophase shift summary: day-to-day drift + pre/post comparison."""

    daily_shift_df: pd.DataFrame
    split_day: int
    pre_post_acrophase: Optional[TwoSampleCircularTest]
    pre_post_onset: Optional[TwoSampleCircularTest]
    pre_post_offset: Optional[TwoSampleCircularTest]
    mean_daily_shift_hours: float
    cumulative_shift_hours: float
    drift_slope_hours_per_day: float
    implied_tau_from_drift_hours: float
    activity_light_phase_angle_hours: Optional[float]


def compute_circular_summary(
    hours: np.ndarray,
    label: str = "Phase",
    weights: Optional[np.ndarray] = None,
    period_hours: float = 24.0,
    expected_hour: Optional[float] = None,
) -> Optional[CircularSummary]:
    """Compute circular mean, resultant length R-bar, circular SD, kappa, Rayleigh test, and optional V-test."""
    arr = np.asarray(hours, dtype=float)
    mask = np.isfinite(arr)
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        mask = mask & np.isfinite(w_arr) & (w_arr > 0)
        w = w_arr[mask]
    else:
        w = None

    arr = arr[mask]
    n = len(arr)
    if n == 0:
        return None

    theta = hours_to_radians(arr, period_hours=period_hours)
    if w is None:
        w_norm = np.ones(n, dtype=float)
    else:
        w_norm = w / np.mean(w)

    C = float(np.sum(w_norm * np.cos(theta)))
    S = float(np.sum(w_norm * np.sin(theta)))
    W_sum = float(np.sum(w_norm))
    R = float(np.hypot(C, S))
    r_bar = float(np.clip(R / W_sum, 0.0, 1.0)) if W_sum > 0 else 0.0

    mean_rad = float(np.arctan2(S, C) % (2.0 * np.pi))
    mean_deg = float(np.degrees(mean_rad))
    mean_hour = radians_to_hours(mean_rad, period_hours=period_hours) % 24.0

    circ_var = float(1.0 - r_bar)
    circ_std_rad = float(np.sqrt(max(0.0, -2.0 * np.log(max(r_bar, 1e-12))))) if r_bar > 1e-12 else float(np.pi)
    circ_std_deg = float(np.degrees(circ_std_rad))
    circ_std_hours = float(circ_std_rad / (2.0 * np.pi) * period_hours)
    kappa = estimate_von_mises_kappa(r_bar, n)

    # Rayleigh test of uniformity
    z_stat = float(n * (r_bar**2))
    if n >= 5:
        R_n = n * r_bar
        p_ray = float(
            np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n**2 - R_n**2)) - (1.0 + 2.0 * n))
        )
    else:
        p_ray = float(np.exp(-z_stat))
    p_ray = float(np.clip(p_ray, 0.0, 1.0))

    v_u, v_p = None, None
    if expected_hour is not None and np.isfinite(expected_hour):
        exp_rad = float(hours_to_radians(expected_hour, period_hours=period_hours))
        v_stat = float(R * np.cos(mean_rad - exp_rad))
        v_u = float(np.sqrt(2.0 / n) * v_stat)
        v_p = float(stats.norm.sf(v_u))

    return CircularSummary(
        label=label,
        n=n,
        period_hours=period_hours,
        mean_angle_rad=mean_rad,
        mean_angle_deg=mean_deg,
        mean_hour=mean_hour,
        mean_hhmm=format_hours_hhmm(mean_hour),
        mean_resultant_length=r_bar,
        circular_variance=circ_var,
        circular_std_rad=circ_std_rad,
        circular_std_deg=circ_std_deg,
        circular_std_hours=circ_std_hours,
        kappa=kappa,
        rayleigh_z=z_stat,
        rayleigh_p=p_ray,
        v_test_u=v_u,
        v_test_p=v_p,
    )


def watson_williams_test(
    hours1: np.ndarray,
    hours2: np.ndarray,
    period_hours: float = 24.0,
) -> Optional[TwoSampleCircularTest]:
    """Watson-Williams circular two-sample F-test comparing mean phases of two groups."""
    h1 = np.asarray(hours1, dtype=float)
    h2 = np.asarray(hours2, dtype=float)
    h1 = h1[np.isfinite(h1)]
    h2 = h2[np.isfinite(h2)]
    n1, n2 = len(h1), len(h2)
    if n1 < 2 or n2 < 2:
        return None

    th1 = hours_to_radians(h1, period_hours)
    th2 = hours_to_radians(h2, period_hours)

    C1, S1 = float(np.sum(np.cos(th1))), float(np.sum(np.sin(th1)))
    C2, S2 = float(np.sum(np.cos(th2))), float(np.sum(np.sin(th2)))
    R1, R2 = float(np.hypot(C1, S1)), float(np.hypot(C2, S2))
    R_total = float(np.hypot(C1 + C2, S1 + S2))

    N = n1 + n2
    r_bar1 = R1 / n1
    r_bar2 = R2 / n2
    r_bar_w = (R1 + R2) / N

    mean1_h = radians_to_hours(float(np.arctan2(S1, C1)), period_hours) % 24.0
    mean2_h = radians_to_hours(float(np.arctan2(S2, C2)), period_hours) % 24.0
    shift_h = float(wrap_phase_shift_hours(mean2_h - mean1_h, period_hours))
    shift_deg = float((shift_h / period_hours) * 360.0)

    if abs(shift_h) < 0.05:
        direction = "Aligned (No Shift)"
    elif shift_h > 0:
        direction = "Phase Delay"
    else:
        direction = "Phase Advance"

    kappa = estimate_von_mises_kappa(r_bar_w, N)
    corr = 1.0 + 3.0 / (8.0 * max(kappa, 0.1))
    denom = N - (R1 + R2)
    if denom <= 1e-12:
        f_stat = 0.0 if abs(shift_h) < 1e-6 else 1e6
        p_val = 1.0 if abs(shift_h) < 1e-6 else 0.0
    else:
        f_stat = float(max(0.0, corr * ((N - 2) * (R1 + R2 - R_total)) / denom))
        p_val = float(stats.f.sf(f_stat, 1, N - 2))

    return TwoSampleCircularTest(
        n1=n1,
        n2=n2,
        mean1_hour=mean1_h,
        mean1_hhmm=format_hours_hhmm(mean1_h),
        mean2_hour=mean2_h,
        mean2_hhmm=format_hours_hhmm(mean2_h),
        r_bar1=r_bar1,
        r_bar2=r_bar2,
        phase_shift_hours=shift_h,
        phase_shift_deg=shift_deg,
        shift_direction=direction,
        f_stat=f_stat,
        p_value=p_val,
    )


def analyze_acrophase_shift(
    daily_df: pd.DataFrame,
    split_day: Optional[int] = None,
    period_hours: float = 24.0,
    light_acrophase_hours: Optional[float] = None,
    activity_acrophase_hours: Optional[float] = None,
) -> Optional[AcrophaseShiftReport]:
    """Compute day-to-day acrophase shifts, unwrapped drift slope, and pre/post-split comparison."""
    if daily_df is None or len(daily_df) < 2 or "acrophase_hours" not in daily_df.columns:
        return None

    df = daily_df.sort_values("day_index").reset_index(drop=True).copy()
    acro = df["acrophase_hours"].to_numpy(dtype=float)
    days = df["day_index"].to_numpy(dtype=float)

    # Sequential day-to-day shift
    daily_shifts = np.full(len(df), np.nan, dtype=float)
    for i in range(1, len(df)):
        if np.isfinite(acro[i]) and np.isfinite(acro[i - 1]):
            daily_shifts[i] = float(wrap_phase_shift_hours(acro[i] - acro[i - 1], period_hours))

    # Unwrapped phase relative to Day 1
    valid_mask = np.isfinite(acro)
    unwrapped = np.full(len(df), np.nan, dtype=float)
    if valid_mask.any():
        rad_valid = hours_to_radians(acro[valid_mask], period_hours)
        unwrapped_rad = np.unwrap(rad_valid)
        unwrapped_h = (unwrapped_rad / (2.0 * np.pi)) * period_hours
        unwrapped[valid_mask] = unwrapped_h - unwrapped_h[0]

    df["acrophase_shift_hours"] = daily_shifts
    df["acrophase_shift_deg"] = (daily_shifts / period_hours) * 360.0
    df["cumulative_shift_hours"] = unwrapped

    # Drift slope (hours/day) via linear regression on unwrapped phases
    if valid_mask.sum() >= 2:
        slope, _, _, _, _ = stats.linregress(days[valid_mask], unwrapped[valid_mask])
        drift_slope = float(slope)
        cum_shift = float(unwrapped[valid_mask][-1])
    else:
        drift_slope = 0.0
        cum_shift = 0.0

    valid_shifts = daily_shifts[np.isfinite(daily_shifts)]
    mean_daily_shift = float(np.mean(valid_shifts)) if len(valid_shifts) > 0 else 0.0
    implied_tau = 24.0 + drift_slope

    # Pre vs Post split day
    if split_day is None:
        # Default: midpoint day
        split_day = int(np.median(days))
        if split_day <= int(days[0]):
            split_day = int(days[0]) + max(1, len(days) // 2)

    pre = df[df["day_index"] < split_day]
    post = df[df["day_index"] >= split_day]

    pre_post_acro = (
        watson_williams_test(pre["acrophase_hours"].to_numpy(), post["acrophase_hours"].to_numpy(), period_hours)
        if len(pre) >= 2 and len(post) >= 2
        else None
    )

    pre_post_onset = None
    if "onset_hours" in df.columns and len(pre) >= 2 and len(post) >= 2:
        pre_post_onset = watson_williams_test(pre["onset_hours"].to_numpy(), post["onset_hours"].to_numpy(), period_hours)

    pre_post_offset = None
    if "offset_hours" in df.columns and len(pre) >= 2 and len(post) >= 2:
        pre_post_offset = watson_williams_test(
            pre["offset_hours"].to_numpy(), post["offset_hours"].to_numpy(), period_hours
        )

    phase_angle = None
    if (
        light_acrophase_hours is not None
        and activity_acrophase_hours is not None
        and np.isfinite(light_acrophase_hours)
        and np.isfinite(activity_acrophase_hours)
    ):
        phase_angle = float(wrap_phase_shift_hours(activity_acrophase_hours - light_acrophase_hours, period_hours))

    return AcrophaseShiftReport(
        daily_shift_df=df,
        split_day=int(split_day),
        pre_post_acrophase=pre_post_acro,
        pre_post_onset=pre_post_onset,
        pre_post_offset=pre_post_offset,
        mean_daily_shift_hours=mean_daily_shift,
        cumulative_shift_hours=cum_shift,
        drift_slope_hours_per_day=drift_slope,
        implied_tau_from_drift_hours=implied_tau,
        activity_light_phase_angle_hours=phase_angle,
    )
