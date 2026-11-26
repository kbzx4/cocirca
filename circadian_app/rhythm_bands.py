"""Ultradian / circadian / infradian rhythm detection and bimodal waveform analysis.

Rhythm bands (standard chronobiology definitions, adjustable):
    Ultradian  : period_min  <= tau <  circadian_min   (default  9 h – 20 h)
    Circadian  : circadian_min <= tau <= circadian_max (default 20 h – 28 h)
    Infradian  : circadian_max <  tau <= period_max    (default 28 h – 36 h)

Detection uses sequential prewhitening rather than testing each band on the raw spectrum. A strong
non-sinusoidal 24 h rhythm puts genuine-looking peaks at 12 h (its 2nd harmonic) and spectral sidelobes
in neighbouring bands; testing those on raw data would report "ultradian" or "infradian" rhythms that
are only echoes of the circadian one. So we repeatedly:
    1. take the strongest local peak of the single-cosinor R² spectrum in a band not yet assigned,
    2. test it (zero-amplitude F-test, Šidák-corrected for the number of independent periods in that
       band, plus a minimum share of total variance so huge-N recordings don't flag trivial rhythms),
    3. subtract the fitted rhythm *with its harmonics* and rescan the residual.

Caveat: an independent 12 h rhythm and the 2nd harmonic of a 24 h rhythm are mathematically the same
signal. When the 24 h component is stronger, 12 h power is attributed to it (and shows up as waveform
shape / bimodality); it is reported as an independent ultradian rhythm only if it is the stronger one.

Bimodality uses a 2-harmonic (tau + tau/2) cosinor fit, which can describe at most two peaks per cycle.
A cycle is bimodal when the fitted waveform has a second peak whose prominence is at least
`min_prominence_fraction` of the waveform range AND the tau/2 component is significant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from .cosinor import CosinorResult, evaluate_cosinor_curve, fit_cosinor, format_hours_hhmm

BAND_NAMES = ("Ultradian", "Circadian", "Infradian")


# --------------------------------------------------------------------------------------------------
# Rhythm bands
# --------------------------------------------------------------------------------------------------


@dataclass
class BandRhythm:
    """Detection result for one period band."""

    band: str
    low_hours: float
    high_hours: float
    detected: bool
    classification: str
    period_hours: float = float("nan")
    amplitude: float = float("nan")
    acrophase_hhmm: str = "--:--"
    percent_variance: float = float("nan")  # % of the ORIGINAL series variance explained by this rhythm
    p_value: float = float("nan")  # single-period zero-amplitude F-test
    p_adjusted: float = float("nan")  # Šidák-corrected for independent periods searched in the band
    n_independent_periods: float = float("nan")
    extraction_order: Optional[int] = None  # 1 = dominant rhythm
    raw_peak_period_hours: float = float("nan")  # strongest raw-spectrum local peak in band (if any)
    raw_peak_r2: float = float("nan")
    note: str = ""
    scanned: bool = True  # False when the period scan range excludes this band entirely

    @property
    def range_label(self) -> str:
        """Band limits in hours; open-ended for an ultradian/infradian band the scan range excludes."""
        if not self.scanned:
            if self.band == "Ultradian":
                return f"< {self.high_hours:g}"
            if self.band == "Infradian":
                return f"> {self.low_hours:g}"
        return f"{self.low_hours:g}–{self.high_hours:g}"


@dataclass
class RhythmBandsResult:
    """Ultradian / circadian / infradian classification for one channel."""

    period_min_hours: float
    period_max_hours: float
    circadian_min_hours: float
    circadian_max_hours: float
    alpha: float
    min_percent_variance: float
    bands: List[BandRhythm]
    dominant_band: Optional[str]
    dominant_period_hours: float
    periods_hours: np.ndarray
    raw_r2: np.ndarray
    residual_r2: np.ndarray  # R² spectrum after removing every detected rhythm and its harmonics
    span_hours: float

    def band(self, name: str) -> BandRhythm:
        return next(b for b in self.bands if b.band == name)

    @property
    def detected_bands(self) -> List[BandRhythm]:
        return sorted((b for b in self.bands if b.detected), key=lambda b: b.extraction_order or 99)


def r2_period_sweep(t_hours: np.ndarray, values: np.ndarray, periods_hours: np.ndarray) -> np.ndarray:
    """Single-cosinor R² (fraction of variance explained) at each trial period. NaNs are ignored."""
    t = np.asarray(t_hours, dtype=float)
    y = np.asarray(values, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    periods = np.asarray(periods_hours, dtype=float)
    out = np.zeros(len(periods), dtype=float)
    n = len(y)
    if n < 4 or len(periods) == 0:
        return out
    yc = y - y.mean()
    sst = float(np.dot(yc, yc))
    if sst <= 1e-15:
        return out

    # Vectorized chunked normal-equations solve across candidate periods (64 periods per BLAS batch)
    chunk = 64
    for start in range(0, len(periods), chunk):
        p_blk = periods[start : start + chunk]
        w = (2.0 * np.pi) / p_blk
        phi = t[:, None] * w[None, :]
        C = np.cos(phi)
        S = np.sin(phi)
        c0 = C.mean(axis=0)
        s0 = S.mean(axis=0)
        g11 = np.einsum("ij,ij->j", C, C) / n - c0 * c0
        g22 = np.einsum("ij,ij->j", S, S) / n - s0 * s0
        g12 = np.einsum("ij,ij->j", C, S) / n - c0 * s0
        rc = (C.T @ yc) / n
        rs = (S.T @ yc) / n
        det = g11 * g22 - g12 * g12
        good = det > 1e-14 * np.maximum(g11 * g22, 1e-18)
        r2_blk = np.zeros(len(p_blk), dtype=float)
        if np.any(good):
            beta = (g22[good] * rc[good] - g12[good] * rs[good]) / det[good]
            gamma = (g11[good] * rs[good] - g12[good] * rc[good]) / det[good]
            r2_blk[good] = (n * (beta * rc[good] + gamma * rs[good])) / sst
        if not np.all(good):
            ones = np.ones(n, dtype=float)
            for j in np.where(~good)[0]:
                X = np.column_stack([ones, C[:, j], S[:, j]])
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                r2_blk[j] = 1.0 - float(np.sum((y - X @ coef) ** 2)) / sst
        out[start : start + chunk] = r2_blk

    return np.clip(out, 0.0, 1.0)


def band_edges(
    period_min: float, period_max: float, circadian_min: float, circadian_max: float
) -> Dict[str, Tuple[float, float]]:
    """Band limits clipped to the scanned range (a band can be empty if the range excludes it)."""
    return {
        "Ultradian": (period_min, min(circadian_min, period_max)),
        "Circadian": (max(circadian_min, period_min), min(circadian_max, period_max)),
        "Infradian": (max(circadian_max, period_min), period_max),
    }


def _band_of(period: float, edges: Dict[str, Tuple[float, float]]) -> Optional[str]:
    lo_u, hi_u = edges["Ultradian"]
    lo_c, hi_c = edges["Circadian"]
    lo_i, hi_i = edges["Infradian"]
    if hi_u > lo_u and lo_u <= period < hi_u:
        return "Ultradian"
    if hi_c > lo_c and lo_c <= period <= hi_c:
        return "Circadian"
    if hi_i > lo_i and lo_i < period <= hi_i:
        return "Infradian"
    return None


def _n_independent_periods(span_hours: float, low: float, high: float) -> float:
    """Approximate number of independent frequencies between periods `low` and `high` over `span_hours`."""
    if high <= low or span_hours <= 0:
        return 1.0
    return max(1.0, span_hours * (1.0 / low - 1.0 / high))


def _sidak(p: float, m: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p >= 1.0:
        return 1.0
    return float(-np.expm1(m * np.log1p(-p)))


def _resolution_hours(period: float, span_hours: float) -> float:
    """Period resolution (~ one frequency bin) used as the tolerance for harmonic matching."""
    return max(0.5, period**2 / max(span_hours, 1e-9))


def _max_harmonic(period: float, period_min: float, dt_hours: float) -> int:
    """Harmonics k whose period P/k reaches just below the scan floor, capped by Nyquist and at 8."""
    k_floor = int(period // period_min) + 1
    k_nyq = int(period // (2.0 * dt_hours)) if dt_hours > 0 else k_floor
    return int(max(1, min(8, k_floor, k_nyq)))


def _harmonic_match(period: float, base: float, period_min: float, span: float) -> Optional[int]:
    """Return k if `period` ≈ base / k for some harmonic k >= 2 inside the scanned range."""
    for k in range(2, int(base // period_min) + 2):
        if abs(period - base / k) <= _resolution_hours(base / k, span):
            return k
    return None


def analyze_rhythm_bands(
    t_hours: np.ndarray,
    values: np.ndarray,
    periods_hours: np.ndarray,
    raw_r2: Optional[np.ndarray] = None,
    circadian_bounds: Tuple[float, float] = (20.0, 28.0),
    alpha: float = 0.05,
    min_percent_variance: float = 2.0,
    max_extractions: int = 3,
) -> RhythmBandsResult:
    """Detect significant ultradian, circadian, and infradian rhythms by sequential prewhitening."""
    t = np.asarray(t_hours, dtype=float)
    y = np.asarray(values, dtype=float)
    periods = np.asarray(periods_hours, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    if finite.sum() < 8:
        raise ValueError("Need at least 8 valid samples for rhythm-band analysis.")

    span = float(t[finite].max() - t[finite].min())
    dt = float(np.median(np.diff(np.sort(t[finite])))) if finite.sum() > 1 else 0.0
    p_min, p_max = float(periods.min()), float(periods.max())
    c_min, c_max = circadian_bounds
    edges = band_edges(p_min, p_max, c_min, c_max)
    nominal = {"Ultradian": (c_min, c_min), "Circadian": (c_min, c_max), "Infradian": (c_max, c_max)}
    m_indep = {name: _n_independent_periods(span, *edges[name]) for name in BAND_NAMES}

    sst0 = float(np.sum((y[finite] - y[finite].mean()) ** 2))
    raw = np.asarray(raw_r2, dtype=float) if raw_r2 is not None else r2_period_sweep(t, y, periods)

    detected: Dict[str, BandRhythm] = {}
    series = y.copy()
    r2 = raw.copy()
    sst_cur = sst0

    for order in range(1, max_extractions + 1):
        if sst0 <= 1e-15:
            break
        peak_idx, _ = signal.find_peaks(r2)  # true local maxima only; grid endpoints are excluded
        cands = []
        for i in peak_idx:
            name = _band_of(float(periods[i]), edges)
            if name is not None and name not in detected:
                cands.append((float(r2[i]), float(periods[i]), name))
        cands.sort(reverse=True)

        chosen = None
        for _, period, name in cands:
            fit = fit_cosinor(t, series, period_hours=period, n_harmonics=1)
            p_adj = _sidak(fit.p_value, m_indep[name])
            pct = 100.0 * fit.r2 * sst_cur / sst0
            if p_adj < alpha and pct >= min_percent_variance:
                chosen = (period, name, fit, p_adj, pct)
                break
        if chosen is None:
            break

        period, name, fit, p_adj, pct = chosen
        lo, hi = edges[name]
        detected[name] = BandRhythm(
            band=name,
            low_hours=lo,
            high_hours=hi,
            detected=True,
            classification=f"Significant {name.lower()} rhythm",
            period_hours=period,
            amplitude=fit.amplitude,
            acrophase_hhmm=fit.acrophase_hhmm,
            percent_variance=pct,
            p_value=fit.p_value,
            p_adjusted=p_adj,
            n_independent_periods=m_indep[name],
            extraction_order=order,
        )

        # Prewhiten: remove this rhythm and its harmonics down to (just below) the scan floor
        n_harm = _max_harmonic(period, p_min, dt)
        try:
            full = fit_cosinor(t, series, period_hours=period, n_harmonics=n_harm)
        except ValueError:
            full = fit
        series = series - (full.fitted_values - full.mesor)
        r2 = r2_period_sweep(t, series, periods)
        fin = np.isfinite(series)
        sst_cur = float(np.sum((series[fin] - series[fin].mean()) ** 2))

    # Describe every band, including those without an independent rhythm
    raw_peaks, _ = signal.find_peaks(raw)
    bands: List[BandRhythm] = []
    for name in BAND_NAMES:
        lo, hi = edges[name]
        in_band = [i for i in raw_peaks if _band_of(float(periods[i]), edges) == name]
        best_raw = max(in_band, key=lambda i: raw[i]) if in_band else None
        raw_p = float(periods[best_raw]) if best_raw is not None else float("nan")
        raw_v = float(raw[best_raw]) if best_raw is not None else float("nan")

        if name in detected:
            b = detected[name]
            b.raw_peak_period_hours, b.raw_peak_r2 = raw_p, raw_v
            # Only the weaker (later-extracted) rhythm of an integer-ratio pair is in doubt
            for other in detected.values():
                if other is b or (other.extraction_order or 99) > (b.extraction_order or 99):
                    continue
                k_div = _harmonic_match(b.period_hours, other.period_hours, p_min, span)
                k_mul = _harmonic_match(other.period_hours, b.period_hours, p_min, span)
                if k_div or k_mul:
                    rel = f"τ/{k_div}" if k_div else f"{k_mul}τ"
                    b.classification = f"Significant {name.lower()} rhythm near {other.band.lower()} {rel} (possible harmonic)"
                    b.note = (
                        f"{b.period_hours:.1f} h ≈ {other.band.lower()} {other.period_hours:.1f} h "
                        f"{'÷ ' + str(k_div) if k_div else '× ' + str(k_mul)}. It survived removal of that rhythm's exact "
                        f"harmonics, but day-to-day changes in the {other.band.lower()} waveform also leave power near its "
                        f"harmonics, so treat it as an independent rhythm only with corroborating evidence."
                    )
            bands.append(b)
            continue

        if not (hi > lo):
            n_lo, n_hi = nominal[name]
            bands.append(BandRhythm(name, n_lo, n_hi, False, f"Not scanned (outside {p_min:g}–{p_max:g} h range)", scanned=False))
            continue

        b = BandRhythm(
            band=name,
            low_hours=lo,
            high_hours=hi,
            detected=False,
            classification=f"No significant {name.lower()} rhythm",
            n_independent_periods=m_indep[name],
            raw_peak_period_hours=raw_p,
            raw_peak_r2=raw_v,
        )
        if best_raw is not None:
            # Prefer explaining the band by a significant raw peak sitting on a harmonic of a detected rhythm
            harmonic_of = None
            for i in sorted(in_band, key=lambda i: -raw[i]):
                for d in detected.values():
                    k = _harmonic_match(float(periods[i]), d.period_hours, p_min, span)
                    if k is None:
                        continue
                    f_h = fit_cosinor(t, y, period_hours=float(periods[i]), n_harmonics=1)
                    if _sidak(f_h.p_value, m_indep[name]) < alpha:
                        harmonic_of = (d, k, i, f_h)
                        break
                if harmonic_of is not None:
                    break

            if harmonic_of is not None:
                d, k, i, f_h = harmonic_of
                raw_p, raw_v = float(periods[i]), float(raw[i])
                b.raw_peak_period_hours, b.raw_peak_r2 = raw_p, raw_v
                b.p_value, b.p_adjusted = f_h.p_value, _sidak(f_h.p_value, m_indep[name])
                b.classification = f"Harmonic of {d.band.lower()} rhythm (τ/{k}), not independent"
                b.note = (
                    f"Raw peak at {raw_p:.1f} h ≈ {d.period_hours:.1f} h ÷ {k}: reflects the {d.band.lower()} waveform's "
                    f"shape (e.g. bimodality) and disappears once that rhythm is removed."
                )
                bands.append(b)
                continue

            fit_raw = fit_cosinor(t, y, period_hours=raw_p, n_harmonics=1)
            b.p_value = fit_raw.p_value
            b.p_adjusted = _sidak(fit_raw.p_value, m_indep[name])
            if b.p_adjusted < alpha and detected:
                dom = min(detected.values(), key=lambda d: d.extraction_order or 99)
                b.classification = f"Explained by {dom.band.lower()} rhythm (leakage), not independent"
                b.note = f"Raw peak at {raw_p:.1f} h is significant alone but vanishes after removing the {dom.period_hours:.1f} h rhythm."
            else:
                b.note = f"Strongest raw peak {raw_p:.1f} h (R² {raw_v:.3f}) is not significant."
        else:
            b.note = f"No local spectral peak between {lo:g} and {hi:g} h."
            band_mask = np.array([_band_of(float(p), edges) == name for p in periods])
            if name == "Infradian" and band_mask.any() and int(np.argmax(np.where(band_mask, raw, -1))) == len(periods) - 1:
                b.note += f" Power rises toward the {p_max:g} h scan limit, so a longer infradian period may lie beyond the range."
        bands.append(b)

    dom = min(detected.values(), key=lambda d: d.extraction_order or 99) if detected else None
    return RhythmBandsResult(
        period_min_hours=p_min,
        period_max_hours=p_max,
        circadian_min_hours=c_min,
        circadian_max_hours=c_max,
        alpha=alpha,
        min_percent_variance=min_percent_variance,
        bands=bands,
        dominant_band=dom.band if dom else None,
        dominant_period_hours=dom.period_hours if dom else float("nan"),
        periods_hours=periods,
        raw_r2=raw,
        residual_r2=r2,
        span_hours=span,
    )


# --------------------------------------------------------------------------------------------------
# 12 h components (always reported, even when weak)
# --------------------------------------------------------------------------------------------------


@dataclass
class TwelveHourAnalysis:
    """Side-by-side report of the 12 h rhythm (its own period) and the τ/2 harmonic of the circadian rhythm.

    * 12 h rhythm: after removing ONLY the circadian fundamental, the best single-cosinor period in
      `search_window` is found freely and fitted. Its period is estimated from the data, not assumed.
    * τ/2 harmonic: the 2nd component of a 2-harmonic cosinor at τ (period locked to exactly τ/2).
    If the free period differs from τ/2 by more than the frequency resolution, the 12 h rhythm is distinct
    from the harmonic (evidence of an independent ultradian rhythm); otherwise the two cannot be separated.
    """

    circadian_period_hours: float
    search_window_hours: Tuple[float, float]
    # 12 h rhythm (free period)
    rhythm_period_hours: float
    rhythm_amplitude: float
    rhythm_peaks_hhmm: str
    rhythm_percent_variance: float
    rhythm_p_value: float
    rhythm_p_adjusted: float  # Šidák over independent periods in the search window
    rhythm_significant: bool
    rhythm_at_window_edge: bool
    # τ/2 harmonic of the circadian rhythm (period locked)
    harmonic_period_hours: float
    harmonic_amplitude: float
    harmonic_peaks_hhmm: str
    harmonic_percent_variance: float
    harmonic_p_value: float
    harmonic_significant: bool
    harmonic_to_fundamental_ratio: float
    # Relationship
    period_difference_hours: float
    resolution_hours: float
    distinct_from_harmonic: bool
    interpretation: str
    alpha: float


def _two_peaks_hhmm(first_peak_hours: float, period: float) -> str:
    """A ~12 h component peaks twice a day: list both clock times."""
    return f"{format_hours_hhmm(first_peak_hours)} & {format_hours_hhmm(first_peak_hours + period)}"


def analyze_twelve_hour(
    t_hours: np.ndarray,
    values: np.ndarray,
    circadian_period_hours: float,
    search_window: Tuple[float, float] = (10.0, 14.0),
    alpha: float = 0.05,
) -> Optional[TwelveHourAnalysis]:
    """Report the 12 h rhythm and the τ/2 harmonic of the circadian rhythm, whatever their strength."""
    t = np.asarray(t_hours, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    if finite.sum() < 12:
        return None
    tau = float(circadian_period_hours)
    span = float(t[finite].max() - t[finite].min())
    sst0 = float(np.sum((y[finite] - y[finite].mean()) ** 2))
    if sst0 <= 1e-15:
        return None

    # τ/2 harmonic: 2-harmonic cosinor at τ
    try:
        fit2 = fit_cosinor(t, y, period_hours=tau, n_harmonics=2)
    except ValueError:
        return None
    h1, h2 = fit2.harmonics[0], fit2.harmonics[1]

    # 12 h rhythm: remove only the circadian fundamental, then search the window for the best period
    fit1 = fit_cosinor(t, y, period_hours=tau, n_harmonics=1)
    resid = y - (fit1.fitted_values - fit1.mesor)
    lo, hi = search_window
    grid = np.arange(lo, hi + 1e-9, 0.05)
    r2 = r2_period_sweep(t, resid, grid)
    i_best = int(np.argmax(r2))
    p12 = float(grid[i_best])
    f12 = fit_cosinor(t, resid, period_hours=p12, n_harmonics=1)
    fin = np.isfinite(resid)
    sst_res = float(np.sum((resid[fin] - resid[fin].mean()) ** 2))
    p_adj = _sidak(f12.p_value, _n_independent_periods(span, lo, hi))

    harm_period = tau / 2.0
    diff = p12 - harm_period
    res = max(0.25, harm_period**2 / max(span, 1e-9))
    distinct = abs(diff) > res
    rhythm_sig = p_adj < alpha
    harm_sig = h2.p_value < alpha

    if not rhythm_sig and not harm_sig:
        interp = "No significant 12 h component: both the 12 h rhythm and the τ/2 harmonic are weak (values shown for reference)."
    elif distinct and rhythm_sig:
        interp = (
            f"The 12 h rhythm's own period ({p12:.2f} h) differs from τ/2 ({harm_period:.2f} h) by more than the "
            f"resolution (±{res:.2f} h): evidence of an ultradian rhythm independent of the circadian waveform."
        )
    else:
        interp = (
            f"The 12 h rhythm's period ({p12:.2f} h) matches τ/2 ({harm_period:.2f} h) within ±{res:.2f} h, so it "
            f"cannot be separated from the circadian harmonic. It describes the circadian waveform's shape "
            f"(e.g. a second daily peak or a midday dip)."
        )
    if i_best in (0, len(grid) - 1):
        interp += f" The best period lies at the {lo:g}–{hi:g} h search edge, so the true 12 h-band peak may lie outside it."

    return TwelveHourAnalysis(
        circadian_period_hours=tau,
        search_window_hours=(lo, hi),
        rhythm_period_hours=p12,
        rhythm_amplitude=f12.amplitude,
        rhythm_peaks_hhmm=_two_peaks_hhmm(f12.acrophase_hours, p12),
        rhythm_percent_variance=100.0 * f12.r2 * sst_res / sst0,
        rhythm_p_value=f12.p_value,
        rhythm_p_adjusted=p_adj,
        rhythm_significant=rhythm_sig,
        rhythm_at_window_edge=i_best in (0, len(grid) - 1),
        harmonic_period_hours=harm_period,
        harmonic_amplitude=h2.amplitude,
        harmonic_peaks_hhmm=_two_peaks_hhmm(h2.acrophase_clock_hours, harm_period),
        harmonic_percent_variance=h2.percent_rhythm,
        harmonic_p_value=h2.p_value,
        harmonic_significant=harm_sig,
        harmonic_to_fundamental_ratio=float(h2.amplitude / h1.amplitude) if h1.amplitude > 1e-12 else float("nan"),
        period_difference_hours=diff,
        resolution_hours=res,
        distinct_from_harmonic=distinct,
        interpretation=interp,
        alpha=alpha,
    )


# --------------------------------------------------------------------------------------------------
# Bimodality
# --------------------------------------------------------------------------------------------------


@dataclass
class WaveformShape:
    primary_hours: float
    primary_value: float
    secondary_hours: float = float("nan")
    secondary_value: float = float("nan")
    secondary_prominence: float = 0.0
    dip_hours: float = float("nan")  # trough separating the two peaks (the shallower of the two troughs)
    dip_value: float = float("nan")
    bimodality_index: float = 0.0  # secondary prominence / waveform range (0 = one peak)
    waveform_range: float = 0.0


@dataclass
class BimodalityResult:
    """Bimodal vs unimodal classification of the rhythm waveform (overall and per day)."""

    period_hours: float
    is_bimodal: bool
    classification: str
    shape: WaveformShape
    h2_to_h1_ratio: float
    h2_p_value: float
    r2: float
    min_prominence_fraction: float
    alpha: float
    days_tested: int
    days_bimodal: int
    daily: pd.DataFrame  # day_index, date, bimodal, bimodality_index, peak1_hhmm, peak2_hhmm
    waveform_hours: np.ndarray
    waveform_values: np.ndarray

    @property
    def primary_peak_hhmm(self) -> str:
        return format_hours_hhmm(self.shape.primary_hours)

    @property
    def secondary_peak_hhmm(self) -> str:
        return format_hours_hhmm(self.shape.secondary_hours)

    @property
    def dip_hhmm(self) -> str:
        return format_hours_hhmm(self.shape.dip_hours)


def describe_waveform(fit: CosinorResult, n_grid: int = 1440) -> Tuple[WaveformShape, np.ndarray, np.ndarray]:
    """Locate the peaks of a fitted cosinor waveform over one cycle (circularly)."""
    P = fit.period_hours
    grid = np.linspace(0.0, P, n_grid, endpoint=False)
    wave = evaluate_cosinor_curve(fit, grid)
    rng = float(wave.max() - wave.min())
    i_max = int(np.argmax(wave))
    shape = WaveformShape(primary_hours=float(grid[i_max]), primary_value=float(wave[i_max]), waveform_range=rng)
    if rng <= 1e-12:
        return shape, grid, wave

    tiled = np.concatenate([wave, wave, wave])
    pk, props = signal.find_peaks(tiled, prominence=0.0)
    keep = (pk >= n_grid) & (pk < 2 * n_grid)
    pk, prom = pk[keep] - n_grid, props["prominences"][keep]
    others = [(float(pr), int(i)) for i, pr in zip(pk, prom) if abs(int(i) - i_max) > 1]
    if not others:
        return shape, grid, wave

    sec_prom, j = max(others)
    # Two arcs between the peaks; the shallower trough is the dip that separates them
    if j > i_max:
        arc_a, arc_b = np.arange(i_max, j + 1), np.r_[np.arange(j, n_grid), np.arange(0, i_max + 1)]
    else:
        arc_a, arc_b = np.arange(j, i_max + 1), np.r_[np.arange(i_max, n_grid), np.arange(0, j + 1)]
    mins = [arc[int(np.argmin(wave[arc]))] for arc in (arc_a, arc_b)]
    dip = max(mins, key=lambda k: wave[k])

    shape.secondary_hours = float(grid[j])
    shape.secondary_value = float(wave[j])
    shape.secondary_prominence = sec_prom
    shape.dip_hours = float(grid[dip])
    shape.dip_value = float(wave[dip])
    shape.bimodality_index = float(sec_prom / rng)
    return shape, grid, wave


def _is_bimodal(shape: WaveformShape, h2_p: float, min_prom: float, alpha: float) -> bool:
    return shape.bimodality_index >= min_prom and h2_p < alpha


def analyze_bimodality(
    t_hours: np.ndarray,
    values: np.ndarray,
    period_hours: float,
    binned_df: Optional[pd.DataFrame] = None,
    value_col: str = "activity",
    epoch_minutes: float = 10.0,
    min_prominence_fraction: float = 0.10,
    alpha: float = 0.05,
) -> Optional[BimodalityResult]:
    """Classify the rhythm waveform (period `period_hours`) as bimodal or unimodal, overall and per day."""
    try:
        fit = fit_cosinor(t_hours, values, period_hours=period_hours, n_harmonics=2)
    except ValueError:
        return None
    shape, grid, wave = describe_waveform(fit)
    h1, h2 = fit.harmonics[0], fit.harmonics[1]
    bimodal = _is_bimodal(shape, h2.p_value, min_prominence_fraction, alpha)
    if bimodal:
        classification = "Bimodal"
    elif shape.bimodality_index >= min_prominence_fraction:
        classification = "Unimodal (second peak not significant)"
    elif shape.bimodality_index > 0:
        classification = "Unimodal (minor shoulder)"
    else:
        classification = "Unimodal"

    rows = []
    if binned_df is not None and value_col in binned_df.columns:
        bins_per_day = max(1, int(round(24 * 60 / epoch_minutes)))
        min_valid = max(8, int(np.ceil(0.5 * bins_per_day)))
        for d, sub in binned_df.groupby("day_index"):
            v = sub[value_col].to_numpy(dtype=float)
            if np.isfinite(v).sum() < min_valid or np.nanstd(v) <= 1e-12:
                continue
            try:
                dfit = fit_cosinor(sub["clock_hour"].to_numpy(dtype=float), v, period_hours=24.0, n_harmonics=2)
            except ValueError:
                continue
            dshape, _, _ = describe_waveform(dfit, n_grid=720)
            rows.append(
                {
                    "day_index": int(d),
                    "bimodal": bool(_is_bimodal(dshape, dfit.harmonics[1].p_value, min_prominence_fraction, alpha)),
                    "bimodality_index": round(dshape.bimodality_index, 3),
                    "peak1_hhmm": format_hours_hhmm(dshape.primary_hours),
                    "peak2_hhmm": format_hours_hhmm(dshape.secondary_hours) if dshape.bimodality_index > 0 else "--:--",
                }
            )
    daily = pd.DataFrame(rows, columns=["day_index", "bimodal", "bimodality_index", "peak1_hhmm", "peak2_hhmm"])

    return BimodalityResult(
        period_hours=float(period_hours),
        is_bimodal=bimodal,
        classification=classification,
        shape=shape,
        h2_to_h1_ratio=float(h2.amplitude / h1.amplitude) if h1.amplitude > 1e-12 else float("nan"),
        h2_p_value=float(h2.p_value),
        r2=float(fit.r2),
        min_prominence_fraction=min_prominence_fraction,
        alpha=alpha,
        days_tested=len(daily),
        days_bimodal=int(daily["bimodal"].sum()) if len(daily) else 0,
        daily=daily,
        waveform_hours=grid,
        waveform_values=wave,
    )
