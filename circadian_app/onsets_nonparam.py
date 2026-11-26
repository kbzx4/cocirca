"""Daily onset/offset detection, phase angle of entrainment, and non-parametric circadian variables (IS, IV, L5, M10, RA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy import signal

from .cosinor import CosinorResult, evaluate_cosinor_curve, format_hours_hhmm

OnsetMethod = Literal["template", "threshold", "cosinor"]


@dataclass
class NonParametricResult:
    """Van Someren non-parametric circadian rhythm metrics."""

    interdaily_stability_is: float
    intradaily_variability_iv: float
    m10_value: float
    m10_onset_hours: float
    m10_onset_hhmm: str
    m10_midpoint_hours: float
    m10_midpoint_hhmm: str
    l5_value: float
    l5_onset_hours: float
    l5_onset_hhmm: str
    l5_midpoint_hours: float
    l5_midpoint_hhmm: str
    relative_amplitude_ra: float
    hourly_profile: np.ndarray  # length 24


def compute_non_parametric_metrics(
    binned_df: pd.DataFrame,
    value_col: str = "activity",
    epoch_minutes: Optional[float] = None,
) -> Optional[NonParametricResult]:
    """Compute IS, IV, M10, L5, and RA from hourly-aggregated circadian data.

    Bins finer than 1 h are averaged into hours. Bins coarser than 1 h are held constant across
    every hour they span, so a 6 h bin contributes to all six of its hours, not only the first.
    """
    min_valid = 24 if epoch_minutes is None or epoch_minutes <= 60 else max(4, int(np.ceil(24 * 60 / epoch_minutes)))
    if value_col not in binned_df.columns or binned_df[value_col].notna().sum() < min_valid:
        return None

    tmp = binned_df[["timestamp", value_col]].copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"])
    hourly = tmp.set_index("timestamp")[value_col].resample("1h").mean()
    if epoch_minutes is not None and epoch_minutes > 60:
        hourly = hourly.ffill(limit=int(np.ceil(epoch_minutes / 60.0)) - 1)
    hourly = hourly.dropna()
    N = len(hourly)
    if N < 24:
        return None

    x = hourly.to_numpy(dtype=float)
    x_bar = float(np.mean(x))
    denom = float(np.sum((x - x_bar) ** 2))
    if denom <= 1e-15:
        return None

    # Hourly 24-bin mean profile
    hours_of_day = hourly.index.hour.to_numpy()
    profile = np.zeros(24, dtype=float)
    counts = np.zeros(24, dtype=float)
    np.add.at(profile, hours_of_day, x)
    np.add.at(counts, hours_of_day, 1.0)
    valid_h = counts > 0
    profile[valid_h] /= counts[valid_h]
    profile[~valid_h] = x_bar

    # Interdaily Stability (IS)
    p = 24
    is_num = N * float(np.sum((profile - x_bar) ** 2))
    is_val = float(np.clip(is_num / (p * denom), 0.0, 1.0))

    # Intradaily Variability (IV)
    diffs = np.diff(x)
    iv_num = N * float(np.sum(diffs**2))
    iv_val = float(iv_num / ((N - 1) * denom))

    # Circular 10-hour max (M10) and 5-hour min (L5) on 24h profile
    ext = np.concatenate([profile, profile])
    m10_means = np.array([np.mean(ext[h : h + 10]) for h in range(24)])
    l5_means = np.array([np.mean(ext[h : h + 5]) for h in range(24)])

    m10_start = int(np.argmax(m10_means))
    m10_val = float(m10_means[m10_start])
    m10_mid = (m10_start + 5.0) % 24.0

    l5_start = int(np.argmin(l5_means))
    l5_val = float(l5_means[l5_start])
    l5_mid = (l5_start + 2.5) % 24.0

    ra = float((m10_val - l5_val) / (m10_val + l5_val)) if (m10_val + l5_val) > 1e-15 else 0.0

    return NonParametricResult(
        interdaily_stability_is=is_val,
        intradaily_variability_iv=iv_val,
        m10_value=m10_val,
        m10_onset_hours=float(m10_start),
        m10_onset_hhmm=format_hours_hhmm(float(m10_start)),
        m10_midpoint_hours=float(m10_mid),
        m10_midpoint_hhmm=format_hours_hhmm(float(m10_mid)),
        l5_value=l5_val,
        l5_onset_hours=float(l5_start),
        l5_onset_hhmm=format_hours_hhmm(float(l5_start)),
        l5_midpoint_hours=float(l5_mid),
        l5_midpoint_hhmm=format_hours_hhmm(float(l5_mid)),
        relative_amplitude_ra=ra,
        hourly_profile=profile,
    )


def detect_daily_onsets_offsets(
    binned_df: pd.DataFrame,
    value_col: str = "activity",
    method: OnsetMethod = "template",
    epoch_minutes: float = 10,
    cosinor_result: Optional[CosinorResult] = None,
    template_window_hours: float = 3.0,
    threshold_quantile: float = 0.35,
    min_bout_minutes: int = 60,
) -> pd.DataFrame:
    """Detect daily phase onsets, offsets, active duration (alpha), and midpoint.

    Handles active bouts crossing midnight cleanly by scanning the continuous time series
    and pairing each day's onset with the following offset.
    """
    if value_col not in binned_df.columns or binned_df[value_col].notna().sum() < 12:
        return pd.DataFrame(
            columns=[
                "day_index",
                "date",
                "onset_hours",
                "onset_hhmm",
                "offset_hours",
                "offset_hhmm",
                "active_duration_hours",
                "rest_duration_hours",
                "midpoint_hours",
                "midpoint_hhmm",
            ]
        )

    df = binned_df.copy()
    t_h = df["elapsed_hours"].to_numpy(dtype=float)
    y_raw = pd.Series(df[value_col].to_numpy(dtype=float)).interpolate(limit_direction="both").to_numpy()

    smooth_bins = max(1, int(round(60.0 / epoch_minutes)))
    y_smooth = pd.Series(y_raw).rolling(window=smooth_bins, center=True, min_periods=1).mean().to_numpy()

    onset_indices: List[int] = []
    offset_indices: List[int] = []
    min_dist_bins = max(4, int(round((14.0 * 60.0) / epoch_minutes)))

    if method == "cosinor" and cosinor_result is not None:
        curve = evaluate_cosinor_curve(cosinor_result, t_h)
        above = curve >= cosinor_result.mesor
        cross_up = np.where((~above[:-1]) & above[1:])[0] + 1
        cross_dn = np.where(above[:-1] & (~above[1:]))[0] + 1
        onset_indices = cross_up.tolist()
        offset_indices = cross_dn.tolist()

    elif method == "threshold":
        q_low = float(np.nanpercentile(y_smooth, 15))
        q_high = float(np.nanpercentile(y_smooth, 85))
        thresh = q_low + threshold_quantile * (q_high - q_low)
        sustained = max(1, int(round(min_bout_minutes / epoch_minutes)))
        state = y_smooth >= thresh
        # all_on[i] / all_off[i]: state[i : i + sustained] is entirely active / entirely rest (O(N) via cumsum)
        cs = np.concatenate([[0], np.cumsum(state.astype(np.int64))])
        win_sum = cs[sustained:] - cs[:-sustained]
        all_on = win_sum == sustained
        all_off = win_sum == 0
        # Debounce short state flips
        in_active = bool(state[0])
        i = 0
        while i < len(state) - sustained:
            if not in_active and all_on[i]:
                if not onset_indices or (i - onset_indices[-1]) >= min_dist_bins:
                    onset_indices.append(i)
                in_active = True
                i += sustained
                continue
            elif in_active and all_off[i]:
                if not offset_indices or (i - offset_indices[-1]) >= min_dist_bins:
                    offset_indices.append(i)
                in_active = False
                i += sustained
                continue
            i += 1

    else:
        # Edge-detection template score: post_window_mean - pre_window_mean
        w_bins = max(2, int(round((template_window_hours * 60.0) / epoch_minutes)))
        score = np.zeros(len(y_smooth), dtype=float)
        csum = np.concatenate([[0.0], np.cumsum(y_smooth)])
        ii = np.arange(w_bins, len(y_smooth) - w_bins)
        if ii.size:
            pre_mean = (csum[ii] - csum[ii - w_bins]) / w_bins
            post_mean = (csum[ii + w_bins] - csum[ii]) / w_bins
            score[ii] = post_mean - pre_mean

        prom = max(1e-9, float(np.std(score) * 0.35))
        pk_on, _ = signal.find_peaks(score, distance=min_dist_bins, prominence=prom)
        pk_off, _ = signal.find_peaks(-score, distance=min_dist_bins, prominence=prom)
        onset_indices = pk_on.tolist()
        offset_indices = pk_off.tolist()

    # Pair each onset with the next valid offset (3h to 21h after onset) and assign to onset's day_index
    rows: List[Dict[str, object]] = []
    used_days = set()
    off_arr = np.array(offset_indices, dtype=int)

    for on_idx in onset_indices:
        day_i = int(df["day_index"].iloc[on_idx])
        if day_i in used_days:
            continue
        t_on_elapsed = float(t_h[on_idx])
        t_on_clock = float(df["clock_hour"].iloc[on_idx])

        # Find first offset 3h..21h after this onset
        candidates = off_arr[(off_arr > on_idx)]
        matched_off = None
        for off_idx in candidates:
            dt_h = float(t_h[off_idx] - t_on_elapsed)
            if 3.0 <= dt_h <= 21.5:
                matched_off = int(off_idx)
                break

        if matched_off is not None:
            t_off_clock = float(df["clock_hour"].iloc[matched_off])
            alpha_h = float(t_h[matched_off] - t_on_elapsed)
            rho_h = max(0.0, 24.0 - alpha_h)
            mid_h = float((t_on_clock + 0.5 * alpha_h) % 24.0)
        else:
            t_off_clock = np.nan
            alpha_h = np.nan
            rho_h = np.nan
            mid_h = np.nan

        used_days.add(day_i)
        rows.append(
            {
                "day_index": day_i,
                "date": str(df["date"].iloc[on_idx]),
                "onset_hours": t_on_clock,
                "onset_hhmm": format_hours_hhmm(t_on_clock),
                "offset_hours": t_off_clock,
                "offset_hhmm": format_hours_hhmm(t_off_clock) if np.isfinite(t_off_clock) else "--:--",
                "active_duration_hours": alpha_h,
                "rest_duration_hours": rho_h,
                "midpoint_hours": mid_h,
                "midpoint_hhmm": format_hours_hhmm(mid_h) if np.isfinite(mid_h) else "--:--",
            }
        )

    return pd.DataFrame(rows)
