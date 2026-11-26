"""Single-component cosinor, multi-harmonic circadian cosinor, daily serial section, and population-mean cosinor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def format_hours_hhmm(hours: float) -> str:
    """Format an hour-of-day value (wrapped into [0, 24)) as HH:MM."""
    if hours is None or not np.isfinite(hours):
        return "--:--"
    h_mod = float(hours) % 24.0
    total_min = int(round(h_mod * 60.0)) % (24 * 60)
    hh = total_min // 60
    mm = total_min % 60
    return f"{hh:02d}:{mm:02d}"


@dataclass
class HarmonicComponent:
    """Fitted parameters and hypothesis test for one harmonic component k."""

    harmonic_index: int  # 1 = fundamental, 2 = 2nd harmonic (12h), etc.
    period_hours: float
    beta: float
    gamma: float
    amplitude: float
    amplitude_se: float
    amplitude_ci_low: float
    amplitude_ci_high: float
    acrophase_rad: float  # [-pi, pi] in cos(w*t + phi) convention
    acrophase_deg_neg: float  # (-360, 0] standard chronobiology convention
    acrophase_hours: float  # [0, period_hours) first peak time after t=0
    acrophase_clock_hours: float  # [0, 24) mod 24
    acrophase_hhmm: str
    acrophase_se_hours: float
    f_stat: float
    p_value: float
    partial_r2: float
    percent_rhythm: float
    amplitude_ratio_to_h1: float


@dataclass
class CosinorResult:
    """Complete result for single-component or multi-harmonic cosinor regression."""

    period_hours: float
    n_harmonics: int
    n_samples: int
    mesor: float
    mesor_se: float
    mesor_ci_low: float
    mesor_ci_high: float
    # Fundamental (k=1) convenience attributes
    amplitude: float
    amplitude_se: float
    amplitude_ci_low: float
    amplitude_ci_high: float
    acrophase_rad: float
    acrophase_deg_neg: float
    acrophase_hours: float
    acrophase_hhmm: str
    bathyphase_hours: float
    bathyphase_hhmm: str
    # Composite multi-harmonic waveform properties (over one period)
    orthophase_hours: float
    orthophase_hhmm: str
    orthophase_value: float
    composite_bathyphase_hours: float
    composite_bathyphase_hhmm: str
    composite_bathyphase_value: float
    composite_amplitude: float
    # Goodness of fit & overall test
    r2: float
    adj_r2: float
    percent_rhythm: float
    rmse: float
    f_stat: float
    p_value: float
    harmonics: List[HarmonicComponent] = field(default_factory=list)
    fitted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None


@dataclass
class PopulationCosinorResult:
    """Population-mean cosinor across multiple cycles/days (Hotelling/F zero-amplitude test)."""

    n_cycles: int
    period_hours: float
    mesor_mean: float
    mesor_se: float
    beta_mean: float
    gamma_mean: float
    amplitude: float
    acrophase_hours: float
    acrophase_hhmm: str
    acrophase_deg_neg: float
    f_stat: float
    p_value: float


def fit_cosinor(
    t_hours: np.ndarray,
    values: np.ndarray,
    period_hours: float = 24.0,
    n_harmonics: int = 1,
    alpha: float = 0.05,
) -> CosinorResult:
    """Fit single-component or multi-harmonic cosinor model to (t_hours, values).

    Model: y(t) = M + sum_{k=1..H} [beta_k * cos(2*pi*k*t/tau) + gamma_k * sin(2*pi*k*t/tau)]
    """
    t = np.asarray(t_hours, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    n_harmonics = max(1, int(n_harmonics))
    n = len(y)
    p = 1 + 2 * n_harmonics
    if n <= p + 1:
        raise ValueError(f"Insufficient valid data points ({n}) for {n_harmonics}-harmonic cosinor (need > {p + 1}).")

    omega = 2.0 * np.pi / float(period_hours)
    cols = [np.ones(n, dtype=float)]
    for k in range(1, n_harmonics + 1):
        cols.append(np.cos(k * omega * t))
        cols.append(np.sin(k * omega * t))
    X = np.column_stack(cols)

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coef
    resid = y - y_hat

    df_model = 2 * n_harmonics
    df_resid = n - p
    sst = float(np.sum((y - np.mean(y)) ** 2))
    ssr = float(np.sum(resid**2))
    ssm = max(0.0, sst - ssr)

    r2 = (ssm / sst) if sst > 1e-15 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 else r2
    sigma2 = ssr / df_resid if df_resid > 0 else 0.0
    rmse = float(np.sqrt(ssr / n))

    if sigma2 > 1e-15 and df_model > 0:
        f_stat = (ssm / df_model) / sigma2
        p_val = float(stats.f.sf(f_stat, df_model, df_resid))
    else:
        f_stat = 0.0
        p_val = 1.0

    # Covariance matrix of OLS estimates
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov = sigma2 * XtX_inv
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_resid))

    mesor = float(coef[0])
    mesor_se = float(np.sqrt(max(0.0, cov[0, 0])))
    mesor_ci_low = mesor - t_crit * mesor_se
    mesor_ci_high = mesor + t_crit * mesor_se

    harmonics: List[HarmonicComponent] = []
    amp1 = None

    for k in range(1, n_harmonics + 1):
        idx_b = 1 + 2 * (k - 1)
        idx_g = idx_b + 1
        bk = float(coef[idx_b])
        gk = float(coef[idx_g])
        amp = float(np.hypot(bk, gk))
        if k == 1:
            amp1 = amp

        # Delta-method standard errors for amplitude & acrophase
        vb = float(max(0.0, cov[idx_b, idx_b]))
        vg = float(max(0.0, cov[idx_g, idx_g]))
        cbg = float(cov[idx_b, idx_g])

        if amp > 1e-12:
            amp_var = (bk**2 * vb + gk**2 * vg + 2.0 * bk * gk * cbg) / (amp**2)
            amp_se = float(np.sqrt(max(0.0, amp_var)))
            phi_var = (gk**2 * vb + bk**2 * vg - 2.0 * bk * gk * cbg) / (amp**4)
            phi_se_rad = float(np.sqrt(max(0.0, phi_var)))
        else:
            amp_se = float(np.sqrt(0.5 * (vb + vg)))
            phi_se_rad = np.pi

        phi_rad = float(np.arctan2(-gk, bk))
        k_period = float(period_hours) / k
        k_omega = k * omega
        # Peak occurs when k_omega * t + phi_rad == 0 (mod 2*pi)
        peak_h = float((-phi_rad / k_omega) % k_period)
        peak_clock = float(peak_h % 24.0)
        deg_neg = -((peak_h / k_period) * 360.0) % -360.0
        if deg_neg > 0:
            deg_neg -= 360.0
        phi_se_hours = float(phi_se_rad / k_omega)

        # Partial F-test for harmonic k using 2x2 parameter covariance
        sub_cov = cov[idx_b : idx_g + 1, idx_b : idx_g + 1]
        theta_k = np.array([bk, gk])
        try:
            sub_inv = np.linalg.pinv(sub_cov)
            f_k = float(0.5 * (theta_k @ sub_inv @ theta_k)) if sigma2 > 1e-15 else 0.0
            p_k = float(stats.f.sf(f_k, 2, df_resid))
        except Exception:
            f_k, p_k = 0.0, 1.0

        # Contribution to variance (orthogonal for complete cycles, approximate via partial R^2 in general)
        keep_cols = [c for c in range(p) if c not in (idx_b, idx_g)]
        X_red = X[:, keep_cols]
        c_red, _, _, _ = np.linalg.lstsq(X_red, y, rcond=None)
        ssr_red = float(np.sum((y - X_red @ c_red) ** 2))
        partial_r2 = max(0.0, (ssr_red - ssr) / sst) if sst > 1e-15 else 0.0

        harmonics.append(
            HarmonicComponent(
                harmonic_index=k,
                period_hours=k_period,
                beta=bk,
                gamma=gk,
                amplitude=amp,
                amplitude_se=amp_se,
                amplitude_ci_low=max(0.0, amp - t_crit * amp_se),
                amplitude_ci_high=amp + t_crit * amp_se,
                acrophase_rad=phi_rad,
                acrophase_deg_neg=float(deg_neg),
                acrophase_hours=peak_h,
                acrophase_clock_hours=peak_clock,
                acrophase_hhmm=format_hours_hhmm(peak_clock),
                acrophase_se_hours=phi_se_hours,
                f_stat=f_k,
                p_value=p_k,
                partial_r2=partial_r2,
                percent_rhythm=partial_r2 * 100.0,
                amplitude_ratio_to_h1=(amp / amp1) if (amp1 and amp1 > 1e-12) else (1.0 if k == 1 else 0.0),
            )
        )

    h1 = harmonics[0]
    bathy_h = (h1.acrophase_hours + period_hours / 2.0) % period_hours

    # Composite multi-harmonic waveform over [0, period_hours)
    grid_t = np.linspace(0.0, float(period_hours), 1440, endpoint=False)
    grid_y = np.full_like(grid_t, mesor)
    for h in harmonics:
        grid_y += h.beta * np.cos(h.harmonic_index * omega * grid_t) + h.gamma * np.sin(
            h.harmonic_index * omega * grid_t
        )
    ortho_idx = int(np.argmax(grid_y))
    bathy_idx = int(np.argmin(grid_y))
    ortho_h = float(grid_t[ortho_idx])
    comp_bathy_h = float(grid_t[bathy_idx])
    comp_amp = float((grid_y[ortho_idx] - grid_y[bathy_idx]) / 2.0)

    # Full fitted values aligned to input length (with NaNs preserved if any)
    full_hat = np.full(len(t_hours), np.nan, dtype=float)
    full_resid = np.full(len(t_hours), np.nan, dtype=float)
    full_hat[mask] = y_hat
    full_resid[mask] = resid

    return CosinorResult(
        period_hours=float(period_hours),
        n_harmonics=n_harmonics,
        n_samples=n,
        mesor=mesor,
        mesor_se=mesor_se,
        mesor_ci_low=mesor_ci_low,
        mesor_ci_high=mesor_ci_high,
        amplitude=h1.amplitude,
        amplitude_se=h1.amplitude_se,
        amplitude_ci_low=h1.amplitude_ci_low,
        amplitude_ci_high=h1.amplitude_ci_high,
        acrophase_rad=h1.acrophase_rad,
        acrophase_deg_neg=h1.acrophase_deg_neg,
        acrophase_hours=h1.acrophase_clock_hours,
        acrophase_hhmm=h1.acrophase_hhmm,
        bathyphase_hours=float(bathy_h % 24.0),
        bathyphase_hhmm=format_hours_hhmm(bathy_h % 24.0),
        orthophase_hours=float(ortho_h % 24.0),
        orthophase_hhmm=format_hours_hhmm(ortho_h % 24.0),
        orthophase_value=float(grid_y[ortho_idx]),
        composite_bathyphase_hours=float(comp_bathy_h % 24.0),
        composite_bathyphase_hhmm=format_hours_hhmm(comp_bathy_h % 24.0),
        composite_bathyphase_value=float(grid_y[bathy_idx]),
        composite_amplitude=comp_amp,
        r2=r2,
        adj_r2=adj_r2,
        percent_rhythm=r2 * 100.0,
        rmse=rmse,
        f_stat=f_stat,
        p_value=p_val,
        harmonics=harmonics,
        fitted_values=full_hat,
        residuals=full_resid,
    )


def evaluate_cosinor_curve(result: CosinorResult, t_hours: np.ndarray, up_to_harmonic: Optional[int] = None) -> np.ndarray:
    """Evaluate fitted cosinor (or first `up_to_harmonic` components) at arbitrary `t_hours`."""
    t = np.asarray(t_hours, dtype=float)
    omega = 2.0 * np.pi / result.period_hours
    y = np.full_like(t, result.mesor, dtype=float)
    limit = up_to_harmonic or len(result.harmonics)
    for h in result.harmonics[:limit]:
        y += h.beta * np.cos(h.harmonic_index * omega * t) + h.gamma * np.sin(h.harmonic_index * omega * t)
    return y


def fit_daily_cosinor_series(
    binned_df: pd.DataFrame,
    value_col: str = "activity",
    period_hours: float = 24.0,
    n_harmonics: int = 1,
    min_coverage_frac: float = 0.5,
) -> pd.DataFrame:
    """Fit cosinor separately for each calendar day (serial section) to track MESOR, Amplitude, and Acrophase."""
    rows: List[Dict[str, object]] = []
    for day_idx, grp in binned_df.groupby("day_index", sort=True):
        valid = grp[np.isfinite(grp[value_col])]
        expected_bins = max(1, int(round(24.0 * 60.0 / max(1, (grp["timestamp"].diff().dt.total_seconds().median() or 600) / 60.0))))
        if len(valid) < max(8, int(expected_bins * min_coverage_frac)):
            continue
        try:
            res = fit_cosinor(
                valid["clock_hour"].to_numpy(),
                valid[value_col].to_numpy(),
                period_hours=period_hours,
                n_harmonics=n_harmonics,
            )
        except ValueError:
            continue

        h1 = res.harmonics[0]
        rows.append(
            {
                "day_index": int(day_idx),
                "date": str(grp["date"].iloc[0]),
                "n_valid_epochs": len(valid),
                "mean_value": float(valid[value_col].mean()),
                "max_value": float(valid[value_col].max()),
                "mesor": res.mesor,
                "amplitude": res.amplitude,
                "beta": h1.beta,
                "gamma": h1.gamma,
                "acrophase_hours": res.acrophase_hours,
                "acrophase_hhmm": res.acrophase_hhmm,
                "acrophase_deg_neg": res.acrophase_deg_neg,
                "acrophase_rad": res.acrophase_rad,
                "orthophase_hours": res.orthophase_hours,
                "orthophase_hhmm": res.orthophase_hhmm,
                "r2": res.r2,
                "percent_rhythm": res.percent_rhythm,
                "p_value": res.p_value,
            }
        )
    return pd.DataFrame(rows)


def fit_population_mean_cosinor(
    daily_df: pd.DataFrame,
    period_hours: float = 24.0,
) -> Optional[PopulationCosinorResult]:
    """Compute Population-Mean Cosinor across daily (beta, gamma) estimates using bivariate Hotelling F-test."""
    if daily_df is None or len(daily_df) < 3:
        return None
    betas = daily_df["beta"].to_numpy(dtype=float)
    gammas = daily_df["gamma"].to_numpy(dtype=float)
    mesors = daily_df["mesor"].to_numpy(dtype=float)
    mask = np.isfinite(betas) & np.isfinite(gammas) & np.isfinite(mesors)
    betas, gammas, mesors = betas[mask], gammas[mask], mesors[mask]
    k = len(betas)
    if k < 3:
        return None

    b_bar = float(np.mean(betas))
    g_bar = float(np.mean(gammas))
    m_bar = float(np.mean(mesors))
    m_se = float(np.std(mesors, ddof=1) / np.sqrt(k))

    amp = float(np.hypot(b_bar, g_bar))
    phi_rad = float(np.arctan2(-g_bar, b_bar))
    omega = 2.0 * np.pi / period_hours
    acro_h = float((-phi_rad / omega) % period_hours) % 24.0
    deg_neg = -((acro_h / period_hours) * 360.0) % -360.0
    if deg_neg > 0:
        deg_neg -= 360.0

    # Hotelling's T^2 test against (0, 0)
    Z = np.column_stack([betas, gammas])
    S = np.cov(Z, rowvar=False, ddof=1)
    mean_vec = np.array([b_bar, g_bar])
    try:
        S_inv = np.linalg.pinv(S)
        T2 = float(k * (mean_vec @ S_inv @ mean_vec))
        f_stat = ((k - 2) / (2.0 * (k - 1))) * T2
        p_val = float(stats.f.sf(f_stat, 2, k - 2))
    except Exception:
        f_stat, p_val = 0.0, 1.0

    return PopulationCosinorResult(
        n_cycles=k,
        period_hours=period_hours,
        mesor_mean=m_bar,
        mesor_se=m_se,
        beta_mean=b_bar,
        gamma_mean=g_bar,
        amplitude=amp,
        acrophase_hours=acro_h,
        acrophase_hhmm=format_hours_hhmm(acro_h),
        acrophase_deg_neg=float(deg_neg),
        f_stat=f_stat,
        p_value=p_val,
    )
