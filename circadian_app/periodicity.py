"""Periodicity estimation: Lomb-Scargle periodogram, Sokolove-Bushell Chi-Square periodogram, FFT, and Cosinor tau sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal, stats

from .rhythm_bands import r2_period_sweep


@dataclass
class PeriodogramResult:
    """Combined periodicity analysis across Lomb-Scargle, Chi-Square, FFT, and Cosinor sweep."""

    periods_hours: np.ndarray
    lomb_scargle_power: np.ndarray
    chi2_qp: np.ndarray
    chi2_crit_95: np.ndarray
    cosinor_r2: np.ndarray
    fft_periods_hours: np.ndarray
    fft_power: np.ndarray
    # Best circadian period estimates (within [min_circadian_h, max_circadian_h])
    best_tau_lomb_scargle: float
    best_ls_power: float
    best_tau_chi2: float
    best_chi2_qp: float
    best_chi2_p_value: float
    best_tau_cosinor: float
    best_cosinor_r2: float
    consensus_tau_hours: float


def compute_chi_square_periodogram(
    values: np.ndarray,
    epoch_minutes: int,
    candidate_periods_hours: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sokolove & Bushell (1978) Chi-Square Periodogram on uniformly binned epoch values.

    Interpolates fractional period lengths onto discrete bins so arbitrary `candidate_periods_hours`
    are evaluated smoothly.
    """
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 24:
        return np.zeros_like(candidate_periods_hours), np.ones_like(candidate_periods_hours)

    # Fill isolated NaNs with series mean for folding stability
    y_clean = y.copy()
    mean_y = float(np.nanmean(y_clean))
    y_clean[~valid] = mean_y
    N = len(y_clean)
    var_total = float(np.sum((y_clean - mean_y) ** 2))
    if var_total <= 1e-15:
        return np.zeros_like(candidate_periods_hours), np.ones_like(candidate_periods_hours)

    t_epochs_h = np.arange(N, dtype=float) * (epoch_minutes / 60.0)
    P_arr = np.clip(np.rint((candidate_periods_hours * 60.0) / epoch_minutes).astype(int), 4, N // 2)
    crit_vals = stats.chi2.ppf(1.0 - alpha, np.maximum(1, P_arr - 1))
    qp_vals = np.zeros(len(candidate_periods_hours), dtype=float)

    for i, (tau_h, P) in enumerate(zip(candidate_periods_hours, P_arr)):
        P_int = int(P)
        if (N // P_int) < 2:
            continue

        phase_bins = np.floor(((t_epochs_h % tau_h) / tau_h) * P_int).astype(int) % P_int
        bin_sums = np.bincount(phase_bins, weights=y_clean, minlength=P_int)
        bin_counts = np.bincount(phase_bins, minlength=P_int)
        nz = bin_counts > 0
        bin_means = np.full(P_int, mean_y, dtype=float)
        bin_means[nz] = bin_sums[nz] / bin_counts[nz]

        ss_between = float(np.sum(bin_counts * (bin_means - mean_y) ** 2))
        qp_vals[i] = (N * ss_between) / var_total

    return qp_vals, crit_vals


def analyze_periodicity(
    t_hours: np.ndarray,
    values: np.ndarray,
    epoch_minutes: int = 10,
    min_period_hours: float = 6.0,
    max_period_hours: float = 36.0,
    step_hours: float = 0.1,
    circadian_bounds: tuple[float, float] = (18.0, 30.0),
) -> PeriodogramResult:
    """Run Lomb-Scargle, Sokolove-Bushell Chi-Square, FFT, and Cosinor tau-sweep periodograms."""
    t = np.asarray(t_hours, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t_valid = t[mask]
    y_valid = y[mask]

    if len(y_valid) < 24:
        raise ValueError("Need at least 24 valid epochs for periodicity analysis.")

    periods = np.arange(min_period_hours, max_period_hours + step_hours * 0.5, step_hours)
    ang_freqs = 2.0 * np.pi / periods

    # 1. Lomb-Scargle Periodogram (handles missing timestamps naturally; floating_mean avoids precenter deprecation)
    ls_power = signal.lombscargle(t_valid, y_valid - np.mean(y_valid), ang_freqs, floating_mean=True, normalize=True)

    # 2. Chi-Square Periodogram on binned series
    chi2_qp, chi2_crit = compute_chi_square_periodogram(y, epoch_minutes=epoch_minutes, candidate_periods_hours=periods)

    # 3. Cosinor R^2 sweep (vectorized)
    cos_r2 = r2_period_sweep(t_valid, y_valid, periods)

    # 4. FFT Power Spectrum on evenly spaced detrended series
    y_fill = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=float)
    y_detrend = y_fill - np.mean(y_fill)
    n_fft = max(len(y_detrend), 2048)
    dt_h = epoch_minutes / 60.0
    fft_vals = np.fft.rfft(y_detrend, n=n_fft)
    fft_freqs = np.fft.rfftfreq(n_fft, d=dt_h)  # cycles per hour
    valid_f = (fft_freqs > 1.0 / max_period_hours) & (fft_freqs < 1.0 / max(1.0, min_period_hours))
    fft_periods = (1.0 / fft_freqs[valid_f])[::-1]
    fft_power = (np.abs(fft_vals[valid_f]) ** 2)[::-1]
    if fft_power.max() > 0:
        fft_power = fft_power / fft_power.max()

    # Best circadian estimates inside circadian_bounds
    c_low, c_high = circadian_bounds
    c_mask = (periods >= c_low) & (periods <= c_high)
    if not c_mask.any():
        c_mask = np.ones_like(periods, dtype=bool)

    c_periods = periods[c_mask]
    ls_c = ls_power[c_mask]
    chi_c = chi2_qp[c_mask]
    chi_crit_c = chi2_crit[c_mask]
    cos_c = cos_r2[c_mask]

    best_ls_idx = int(np.argmax(ls_c))
    best_tau_ls = float(c_periods[best_ls_idx])
    best_ls_val = float(ls_c[best_ls_idx])

    # For chi-square, maximize excess over critical threshold (Qp - crit)
    excess = chi_c - chi_crit_c
    best_chi_idx = int(np.argmax(excess))
    best_tau_chi2 = float(c_periods[best_chi_idx])
    best_chi2_val = float(chi_c[best_chi_idx])
    p_df = max(1, int(round((best_tau_chi2 * 60.0) / epoch_minutes)) - 1)
    best_chi2_p = float(stats.chi2.sf(best_chi2_val, p_df))

    best_cos_idx = int(np.argmax(cos_c))
    best_tau_cos = float(c_periods[best_cos_idx])
    best_cos_val = float(cos_c[best_cos_idx])

    # Consensus tau: median of the three estimators (robust to single-method harmonic artifacts)
    consensus_tau = float(np.median([best_tau_ls, best_tau_chi2, best_tau_cos]))

    return PeriodogramResult(
        periods_hours=periods,
        lomb_scargle_power=ls_power,
        chi2_qp=chi2_qp,
        chi2_crit_95=chi2_crit,
        cosinor_r2=cos_r2,
        fft_periods_hours=fft_periods,
        fft_power=fft_power,
        best_tau_lomb_scargle=best_tau_ls,
        best_ls_power=best_ls_val,
        best_tau_chi2=best_tau_chi2,
        best_chi2_qp=best_chi2_val,
        best_chi2_p_value=best_chi2_p,
        best_tau_cosinor=best_tau_cos,
        best_cosinor_r2=best_cos_val,
        consensus_tau_hours=consensus_tau,
    )
