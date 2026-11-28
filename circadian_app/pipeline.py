"""End-to-end circadian rhythmometry, actography, harmonics, circular statistics, and shift analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .circular_stats import (
    AcrophaseShiftReport,
    CircularSummary,
    analyze_acrophase_shift,
    compute_circular_summary,
    wrap_phase_shift_hours,
)
from .cosinor import (
    CosinorResult,
    PopulationCosinorResult,
    evaluate_cosinor_curve,
    fit_cosinor,
    fit_daily_cosinor_series,
    fit_population_mean_cosinor,
)
from .io_loader import RawSensorDataset
from .onsets_nonparam import (
    NonParametricResult,
    OnsetMethod,
    compute_non_parametric_metrics,
    detect_daily_onsets_offsets,
)
from .periodicity import PeriodogramResult, analyze_periodicity
from .preprocessing import ActivityMethod, BinnedCircadianSeries, preprocess_to_epochs
from .rhythm_bands import (
    BimodalityResult,
    RhythmBandsResult,
    TwelveHourAnalysis,
    analyze_bimodality,
    analyze_rhythm_bands,
    analyze_twelve_hour,
)


@dataclass
class CircadianAnalysisConfig:
    """Configuration for circadian rhythmometry and actogram analysis."""

    epoch_minutes: float = 10  # bin size; 1/60 (1 s) to 360 (6 h)
    activity_method: ActivityMethod = "enmo"
    use_log_lux: bool = True
    n_harmonics: int = 3
    period_mode: Literal["auto_consensus", "fixed_24h", "lomb_scargle", "chi2", "cosinor", "custom"] = "auto_consensus"
    custom_period_hours: float = 24.0
    # Period scan range and rhythm bands: ultradian [min, circ_min) | circadian [circ_min, circ_max] | infradian (circ_max, max]
    period_min_hours: float = 9.0
    period_max_hours: float = 36.0
    circadian_min_hours: float = 20.0
    circadian_max_hours: float = 28.0
    rhythm_alpha: float = 0.05
    min_rhythm_percent_variance: float = 2.0  # a band rhythm must explain >= this % of total variance
    bimodality_min_prominence: float = 0.10  # 2nd peak prominence as a fraction of waveform range
    onset_method: OnsetMethod = "template"
    split_day: Optional[int] = None
    interpolate_short_gaps_min: int = 60


@dataclass
class ChannelReport:
    """Complete circadian analysis results for a single channel (Activity or Light)."""

    channel_name: str
    value_col: str
    unit_label: str
    period_used_hours: float
    periodogram: PeriodogramResult
    cosinor_single: CosinorResult
    cosinor_harmonics: CosinorResult
    daily_table: pd.DataFrame
    population_cosinor: Optional[PopulationCosinorResult]
    non_parametric: Optional[NonParametricResult]
    circular_acrophase: Optional[CircularSummary]
    circular_onset: Optional[CircularSummary]
    circular_offset: Optional[CircularSummary]
    shift_report: Optional[AcrophaseShiftReport]
    rhythm_bands: Optional[RhythmBandsResult] = None
    bimodality: Optional[BimodalityResult] = None
    twelve_hour: Optional[TwelveHourAnalysis] = None


@dataclass
class CircadianReport:
    """Top-level report bundling binned epochs and channel-level reports."""

    source_name: str
    config: CircadianAnalysisConfig
    binned: BinnedCircadianSeries
    primary_channel: str
    activity_report: Optional[ChannelReport] = None
    light_report: Optional[ChannelReport] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def primary(self) -> ChannelReport:
        if self.primary_channel == "light" and self.light_report is not None:
            return self.light_report
        if self.activity_report is not None:
            return self.activity_report
        if self.light_report is not None:
            return self.light_report
        raise ValueError("No valid activity or light channel report was generated.")


def _select_period(periodogram: PeriodogramResult, cfg: CircadianAnalysisConfig) -> float:
    if cfg.period_mode == "fixed_24h":
        return 24.0
    if cfg.period_mode == "custom":
        return float(cfg.custom_period_hours)
    if cfg.period_mode == "lomb_scargle":
        return periodogram.best_tau_lomb_scargle
    if cfg.period_mode == "chi2":
        return periodogram.best_tau_chi2
    if cfg.period_mode == "cosinor":
        return periodogram.best_tau_cosinor
    return periodogram.consensus_tau_hours


PERIODOGRAM_MIN_EPOCH_MINUTES = 1.0


def _periodogram_series(
    t_h: np.ndarray, y: np.ndarray, epoch_minutes: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Average sub-minute bins up to 1 min for the 4–36 h period sweep.

    Periods that long carry no information above 1-minute resolution, and the Lomb-Scargle /
    cosinor sweep cost grows linearly with sample count (1 s bins = 60x the work for the same answer).
    """
    if epoch_minutes >= PERIODOGRAM_MIN_EPOCH_MINUTES:
        return t_h, y, epoch_minutes
    factor = int(round(PERIODOGRAM_MIN_EPOCH_MINUTES / epoch_minutes))
    groups = np.arange(len(y)) // factor
    y_ds = pd.Series(y).groupby(groups).mean().to_numpy(dtype=float)
    t_ds = pd.Series(t_h).groupby(groups).mean().to_numpy(dtype=float)
    return t_ds, y_ds, epoch_minutes * factor


def _analyze_single_channel(
    binned: BinnedCircadianSeries,
    channel_name: str,
    value_col: str,
    unit_label: str,
    cfg: CircadianAnalysisConfig,
    other_acrophase_hours: Optional[float] = None,
) -> ChannelReport:
    df = binned.df
    t_h = df["elapsed_hours"].to_numpy(dtype=float)
    y = df[value_col].to_numpy(dtype=float)

    pg_t, pg_y, pg_epoch = _periodogram_series(t_h, y, binned.epoch_minutes)
    periodogram = analyze_periodicity(
        t_hours=pg_t,
        values=pg_y,
        epoch_minutes=pg_epoch,
        min_period_hours=cfg.period_min_hours,
        max_period_hours=cfg.period_max_hours,
        step_hours=0.1,
        circadian_bounds=(cfg.circadian_min_hours, cfg.circadian_max_hours),
    )

    tau = _select_period(periodogram, cfg)

    try:
        rhythm_bands = analyze_rhythm_bands(
            pg_t,
            pg_y,
            periodogram.periods_hours,
            raw_r2=periodogram.cosinor_r2,
            circadian_bounds=(cfg.circadian_min_hours, cfg.circadian_max_hours),
            alpha=cfg.rhythm_alpha,
            min_percent_variance=cfg.min_rhythm_percent_variance,
        )
    except ValueError:
        rhythm_bands = None
    # Bimodality = two peaks per (circadian) day; if τ lies outside the circadian band (e.g. a scan range that
    # excludes it), fall back to 24 h rather than describe the waveform of an ultradian/infradian period.
    bimodal_period = tau if cfg.circadian_min_hours <= tau <= cfg.circadian_max_hours else 24.0
    bimodality = analyze_bimodality(
        t_h,
        y,
        period_hours=bimodal_period,
        binned_df=df,
        value_col=value_col,
        epoch_minutes=binned.epoch_minutes,
        min_prominence_fraction=cfg.bimodality_min_prominence,
        alpha=cfg.rhythm_alpha,
    )
    twelve_hour = analyze_twelve_hour(pg_t, pg_y, circadian_period_hours=bimodal_period, alpha=cfg.rhythm_alpha)
    cos_single = fit_cosinor(t_h, y, period_hours=tau, n_harmonics=1)
    cos_harm = fit_cosinor(t_h, y, period_hours=tau, n_harmonics=cfg.n_harmonics)

    # Daily serial section uses 24h clock projection so clock times of acrophase shift visibly if tau != 24h
    daily_cos = fit_daily_cosinor_series(df, value_col=value_col, period_hours=24.0, n_harmonics=1)
    onsets_df = detect_daily_onsets_offsets(
        df,
        value_col=value_col,
        method=cfg.onset_method,
        epoch_minutes=binned.epoch_minutes,
        cosinor_result=cos_harm,
    )

    if not daily_cos.empty and not onsets_df.empty:
        daily_merged = pd.merge(daily_cos, onsets_df, on=["day_index", "date"], how="outer").sort_values("day_index")
    elif not daily_cos.empty:
        daily_merged = daily_cos.copy()
    else:
        daily_merged = onsets_df.copy()

    # Add daily mean/max light & activity context if available
    if "activity" in df.columns and df["activity"].notna().any():
        act_stats = df.groupby("day_index")["activity"].agg(daily_activity_mean="mean", daily_activity_max="max").reset_index()
        daily_merged = pd.merge(daily_merged, act_stats, on="day_index", how="left")
    if "lux" in df.columns and df["lux"].notna().any():
        lux_stats = df.groupby("day_index")["lux"].agg(daily_lux_mean="mean", daily_lux_max="max").reset_index()
        daily_merged = pd.merge(daily_merged, lux_stats, on="day_index", how="left")
    if bimodality is not None and not bimodality.daily.empty and "day_index" in daily_merged.columns:
        daily_merged = pd.merge(daily_merged, bimodality.daily, on="day_index", how="left")

    pop_cos = fit_population_mean_cosinor(daily_cos, period_hours=24.0)
    non_param = compute_non_parametric_metrics(df, value_col=value_col, epoch_minutes=binned.epoch_minutes)

    circ_acro = (
        compute_circular_summary(
            daily_merged["acrophase_hours"].to_numpy(),
            label=f"{channel_name} Acrophase",
            weights=daily_merged["amplitude"].to_numpy() if "amplitude" in daily_merged.columns else None,
        )
        if "acrophase_hours" in daily_merged.columns
        else None
    )
    circ_onset = (
        compute_circular_summary(daily_merged["onset_hours"].to_numpy(), label=f"{channel_name} Onset")
        if "onset_hours" in daily_merged.columns
        else None
    )
    circ_offset = (
        compute_circular_summary(daily_merged["offset_hours"].to_numpy(), label=f"{channel_name} Offset")
        if "offset_hours" in daily_merged.columns
        else None
    )

    shift_rep = analyze_acrophase_shift(
        daily_merged,
        split_day=cfg.split_day,
        period_hours=24.0,
        light_acrophase_hours=other_acrophase_hours if channel_name != "Light" else cos_single.acrophase_hours,
        activity_acrophase_hours=cos_single.acrophase_hours if channel_name != "Light" else other_acrophase_hours,
    )
    if shift_rep is not None:
        daily_merged = shift_rep.daily_shift_df

    return ChannelReport(
        channel_name=channel_name,
        value_col=value_col,
        unit_label=unit_label,
        period_used_hours=tau,
        periodogram=periodogram,
        cosinor_single=cos_single,
        cosinor_harmonics=cos_harm,
        daily_table=daily_merged.reset_index(drop=True),
        population_cosinor=pop_cos,
        non_parametric=non_param,
        circular_acrophase=circ_acro,
        circular_onset=circ_onset,
        circular_offset=circ_offset,
        shift_report=shift_rep,
        rhythm_bands=rhythm_bands,
        bimodality=bimodality,
        twelve_hour=twelve_hour,
    )


def run_full_analysis(
    dataset: RawSensorDataset,
    config: Optional[CircadianAnalysisConfig] = None,
) -> CircadianReport:
    """Execute complete preprocessing, periodicity, cosinor/harmonics, onset/offset, circular stats, and shift analysis."""
    cfg = config or CircadianAnalysisConfig()
    binned = preprocess_to_epochs(
        dataset,
        epoch_minutes=cfg.epoch_minutes,
        activity_method=cfg.activity_method,
        interpolate_short_gaps_min=cfg.interpolate_short_gaps_min,
    )
    warnings: List[str] = []
    if binned.n_days < 2.0:
        warnings.append(
            f"Recording duration is {binned.n_days:.2f} days; multi-day periodograms and circular stats are most reliable with >= 3 days."
        )

    light_report: Optional[ChannelReport] = None
    activity_report: Optional[ChannelReport] = None

    if binned.has_light and binned.df["lux"].notna().sum() >= 24:
        light_col = "log_lux" if cfg.use_log_lux else "lux"
        light_unit = "log10(lux + 1)" if cfg.use_log_lux else "lux"
        light_report = _analyze_single_channel(binned, "Light", light_col, light_unit, cfg)

    if binned.has_activity and binned.df["activity"].notna().sum() >= 8:
        light_acro = light_report.cosinor_single.acrophase_hours if light_report is not None else None
        # milli-g only applies when activity was derived from accelerometer axes; precomputed counts keep their own units
        from_accel = all(c in dataset.df.columns and dataset.df[c].notna().any() for c in ("ax", "ay", "az"))
        act_unit = "milli-g" if from_accel and cfg.activity_method in ("enmo", "mad") else "activity units"
        # The importer may route heart rate, temperature or another signal into the main channel
        meta = dataset.metadata or {}
        act_name = meta.get("primary_name") or "Activity"
        if meta.get("primary_unit") and act_name != "Activity":
            act_unit = meta["primary_unit"]
        activity_report = _analyze_single_channel(
            binned, act_name, "activity", act_unit, cfg, other_acrophase_hours=light_acro
        )

        # Attach Activity-Light phase angle onto daily table when both exist
        if light_report is not None and "acrophase_hours" in activity_report.daily_table.columns and "acrophase_hours" in light_report.daily_table.columns:
            light_map = light_report.daily_table.set_index("day_index")["acrophase_hours"].to_dict()
            act_df = activity_report.daily_table.copy()
            act_df["light_acrophase_hours"] = act_df["day_index"].map(light_map)
            act_df["phase_angle_act_minus_light_h"] = act_df.apply(
                lambda r: wrap_phase_shift_hours(r["acrophase_hours"] - r["light_acrophase_hours"], 24.0)
                if pd.notna(r["acrophase_hours"]) and pd.notna(r["light_acrophase_hours"])
                else np.nan,
                axis=1,
            )
            activity_report.daily_table = act_df

    # Populate fitted curves back into binned df for Excel export & plotting
    primary_name = "activity" if activity_report is not None else "light"
    primary_rep = activity_report if activity_report is not None else light_report
    if primary_rep is not None:
        t_h = binned.df["elapsed_hours"].to_numpy(dtype=float)
        binned.df["cosinor_single_fit"] = evaluate_cosinor_curve(primary_rep.cosinor_single, t_h)
        binned.df["cosinor_harmonics_fit"] = evaluate_cosinor_curve(primary_rep.cosinor_harmonics, t_h)
        binned.df["cosinor_residual"] = binned.df[primary_rep.value_col] - binned.df["cosinor_harmonics_fit"]

    return CircadianReport(
        source_name=dataset.source_name,
        config=cfg,
        binned=binned,
        primary_channel=primary_name,
        activity_report=activity_report,
        light_report=light_report,
        warnings=warnings,
    )
