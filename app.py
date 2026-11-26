"""Streamlit dashboard for Phyphox / passive phone circadian activity & light rhythmometry."""

from __future__ import annotations

from datetime import datetime, time as dt_time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from circadian_app.excel_export import export_report_to_excel
from circadian_app.io_loader import (
    PhyphoxLiveRecorder,
    RawSensorDataset,
    generate_synthetic_circadian_data,
    load_tabular_sensor_file,
    merge_datasets,
)
from circadian_app.pipeline import CircadianAnalysisConfig, CircadianReport, run_full_analysis
from circadian_app.plotting import (
    apply_plotly_theme,
    downsample_for_display,
    figure_to_png_bytes,
    make_acrophase_shift_timeline_mpl,
    make_actogram_mpl,
    make_actogram_plotly,
    make_circular_polar_mpl,
    make_cosinor_harmonics_mpl,
    make_periodogram_mpl,
    make_rhythm_bands_mpl,
)
from circadian_app.preprocessing import format_bin_size
from circadian_app.theme import (
    AVAILABLE_LABEL_FONTS,
    BACKGROUND_PRESETS,
    KNOWN_AXIS_LABELS,
    PALETTES,
    ROLE_LABELS,
    build_theme,
    reset_seen_labels,
    seen_axis_labels,
    set_plot_theme,
)

# 1 s → 6 h; each value divides 86,400 s so every day has a whole number of bins
BIN_SIZE_OPTIONS_SEC = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 14400, 21600]
LAZY_EXPORT_EPOCHS = 20_000  # above this, Excel / binned-CSV exports are built on request, not every rerun

st.set_page_config(
    page_title="Circadian Phyphox Studio",
    page_icon="🌓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌓 Circadian Phyphox Studio")
st.caption(
    "Passive phone activity & ambient light rhythmometry • Actograms • Cosinor & Circadian Harmonics • Circular Statistics • Acrophase Shift • Excel Export"
)

if "dataset" not in st.session_state:
    st.session_state.dataset = generate_synthetic_circadian_data(
        days=14, epoch_minutes=10, tau_hours=24.0, acrophase_hour=14.5, shift_day=8, shift_hours=3.5
    )
if "recorder" not in st.session_state:
    st.session_state.recorder = None


with st.sidebar:
    st.header("1. Data Input / Passive Recorder")
    input_mode = st.radio(
        "Source Mode",
        ["Synthetic Demo", "Upload Phyphox / CSV / XLSX", "Live Phyphox Remote Recorder"],
        index=0,
    )

    if input_mode == "Synthetic Demo":
        preset = st.selectbox(
            "Scenario Preset",
            [
                "Entrained + Phase Delay (+3.5h on Day 8)",
                "Free-Running Rhythm (τ = 24.4 h)",
                "Phase Advance (-2.5h on Day 7)",
                "Custom Parameters",
            ],
        )
        if preset == "Entrained + Phase Delay (+3.5h on Day 8)":
            s_days, s_tau, s_acr, s_sday, s_sh = 14, 24.0, 14.5, 8, 3.5
        elif preset == "Free-Running Rhythm (τ = 24.4 h)":
            s_days, s_tau, s_acr, s_sday, s_sh = 14, 24.4, 14.0, None, 0.0
        elif preset == "Phase Advance (-2.5h on Day 7)":
            s_days, s_tau, s_acr, s_sday, s_sh = 14, 24.0, 15.0, 7, -2.5
        else:
            s_days = st.slider("Days", 4, 30, 14)
            s_tau = st.slider("Period τ (h)", 20.0, 28.0, 24.0, 0.1)
            s_acr = st.slider("Baseline Acrophase (h)", 0.0, 23.5, 14.5, 0.5)
            enable_shift = st.checkbox("Inject Phase Shift", value=True)
            s_sday = st.slider("Shift Day", 2, s_days, max(2, s_days // 2)) if enable_shift else None
            s_sh = st.slider("Shift Magnitude (h, +Delay / -Advance)", -8.0, 8.0, 3.5, 0.5) if enable_shift else 0.0

        h2_ratio = st.slider("2nd Harmonic (12h) Ratio", 0.0, 0.6, 0.28, 0.02)
        h3_ratio = st.slider("3rd Harmonic (8h) Ratio", 0.0, 0.4, 0.12, 0.02)

        if st.button("Generate / Load Demo Data", type="primary", use_container_width=True):
            st.session_state.dataset = generate_synthetic_circadian_data(
                days=s_days,
                epoch_minutes=10,
                tau_hours=s_tau,
                acrophase_hour=s_acr,
                shift_day=s_sday,
                shift_hours=s_sh,
                harmonic_2_amp_ratio=h2_ratio,
                harmonic_3_amp_ratio=h3_ratio,
            )
            st.success("Loaded synthetic circadian dataset.")

    elif input_mode == "Upload Phyphox / CSV / XLSX":
        st.info(
            "Supports Phyphox `.zip` exports (with `meta/time.csv`), individual `Light.csv` / `Accelerometer.csv`, Sensor Logger, or generic timestamped CSV/XLSX."
        )
        uploaded_files = st.file_uploader(
            "Upload one or more sensor files",
            type=["zip", "csv", "tsv", "xlsx", "xls"],
            accept_multiple_files=True,
        )
        override_start = st.checkbox("Specify start date/time (if file only has relative Time (s))", value=False)
        start_dt = None
        if override_start:
            d_val = st.date_input("Start Date", value=datetime.now().date())
            t_val = st.time_input("Start Time", value=dt_time(0, 0))
            start_dt = datetime.combine(d_val, t_val)

        if uploaded_files and st.button("Load Uploaded File(s)", type="primary", use_container_width=True):
            try:
                loaded: List[RawSensorDataset] = []
                for uf in uploaded_files:
                    loaded.append(load_tabular_sensor_file(uf.getvalue(), filename=uf.name, start_datetime=start_dt))
                st.session_state.dataset = merge_datasets(loaded)
                st.success(f"Loaded {len(st.session_state.dataset.df):,} samples from {st.session_state.dataset.source_name}")
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    else:
        st.info(
            "In Phyphox on your phone, open an experiment (e.g., Light + Acceleration), tap **⋮ → Allow remote access**, and enter the URL below."
        )
        phy_url = st.text_input("Phyphox Remote URL", value="http://192.168.1.50:8080")
        sim_live = st.checkbox("Simulate live phone stream (no phone needed)", value=True)
        poll_sec = st.slider("Poll Interval (seconds)", 1, 60, 5)
        log_path = st.text_input("CSV Log File", value="recordings/phyphox_live.csv")

        c1, c2 = st.columns(2)
        if c1.button("Probe /config", use_container_width=True):
            try:
                info = PhyphoxLiveRecorder.probe_phyphox(phy_url)
                st.success(f"Connected: {info['title']} | Buffers: {', '.join(info['buffers'])}")
            except Exception as exc:
                st.error(f"Probe failed: {exc}")

        rec: PhyphoxLiveRecorder | None = st.session_state.recorder
        is_running = rec is not None and rec.is_running()
        if not is_running:
            if c2.button("Start Passive Rec", type="primary", use_container_width=True):
                rec = PhyphoxLiveRecorder(
                    base_url=phy_url,
                    output_csv=log_path,
                    poll_interval_sec=float(poll_sec),
                    simulate=sim_live,
                )
                rec.start()
                st.session_state.recorder = rec
                st.rerun()
        else:
            if c2.button("Stop Recording", type="secondary", use_container_width=True):
                rec.stop()
                st.rerun()

        if rec is not None:
            st.write(f"**Status:** {'🟢 Recording' if rec.is_running() else '⚪ Stopped'} | **Samples:** {rec.samples_recorded}")
            if rec.last_sample:
                st.json(rec.last_sample)
            if rec.last_error:
                st.warning(f"Last error: {rec.last_error}")
            if Path(rec.output_csv).exists() and st.button("Analyze Recorded Log", use_container_width=True):
                try:
                    st.session_state.dataset = rec.load_recorded_dataset()
                    st.success("Loaded recorded passive log.")
                except Exception as exc:
                    st.error(str(exc))

    st.divider()
    st.header("2. Analysis Parameters")
    epoch_sec = st.select_slider(
        "Epoch Bin Size",
        options=BIN_SIZE_OPTIONS_SEC,
        value=600,
        format_func=lambda s: format_bin_size(s / 60.0),
        help="Resampling bin width, from 1 second to 6 hours. Every step divides 24 h evenly so actogram rows stay aligned to midnight.",
    )
    epoch_min = epoch_sec / 60.0
    act_method = st.selectbox("Accelerometer Metric", ["enmo", "mad", "vector_magnitude", "raw"], index=0)
    use_log_lux = st.checkbox("Use log10(lux + 1) for Light", value=True)
    n_harm = st.slider("Circadian Harmonics (H)", 1, 6, 3)
    period_mode = st.selectbox(
        "Period (τ) Estimation Mode",
        ["auto_consensus", "fixed_24h", "lomb_scargle", "chi2", "cosinor", "custom"],
        index=0,
    )
    custom_tau = st.number_input("Custom τ (hours)", 12.0, 36.0, 24.0, 0.1) if period_mode == "custom" else 24.0
    period_range = st.slider(
        "Period Scan Range (h)", 4.0, 48.0, (9.0, 36.0), 0.5,
        help="Periods searched by all periodograms and the rhythm-band analysis.",
    )
    circ_band = st.slider(
        "Circadian Band (h)", 12.0, 36.0, (20.0, 28.0), 0.5,
        help="Ultradian = shorter than the lower bound; infradian = longer than the upper bound. τ is estimated inside this band.",
    )
    bimodal_prom = st.slider(
        "Bimodality: Min 2nd-Peak Prominence (% of range)", 5, 50, 10,
        help="How pronounced the second daily peak must be (relative to the waveform's full range) to count as bimodal.",
    )
    onset_method = st.selectbox("Onset / Offset Detector", ["template", "threshold", "cosinor"], index=0)
    use_custom_split = st.checkbox("Set Acrophase Shift Split Day", value=False)
    split_day = st.number_input("Split Day (Pre < Day ≤ Post)", min_value=2, max_value=60, value=8) if use_custom_split else None

    st.divider()
    st.header("3. Chart Colours")
    bg_choice = st.selectbox("Graph Background", [*BACKGROUND_PRESETS, "Custom…"], index=0)
    chart_bg = st.color_picker("Custom background", "#ffffff") if bg_choice == "Custom…" else BACKGROUND_PRESETS[bg_choice]
    palette_name = st.selectbox("Plot Colour Palette", list(PALETTES), index=0)
    base_theme = build_theme(chart_bg, palette_name)
    color_overrides = {}
    with st.expander("Customise individual plot colours"):
        st.caption("Pickers start from the chosen palette, adjusted for the background. Changing the palette or background resets them.")
        for role, role_label in ROLE_LABELS.items():
            default_c = getattr(base_theme, role)
            picked = st.color_picker(role_label, default_c, key=f"color_{role}_{palette_name}_{chart_bg}")
            if picked.lower() != default_c.lower():
                color_overrides[role] = picked
    with st.expander("Axis labels: text, font & colour"):
        font_choice = st.selectbox("Font", ["Default", *AVAILABLE_LABEL_FONTS], index=0, key="axis_font")
        size_choice = st.slider("Font size (0 = chart default)", 0, 24, 0, key="axis_font_size")
        bold_choice = st.checkbox("Bold", value=False, key="axis_bold")
        custom_label_colors = st.checkbox("Custom label colours", value=False, key="axis_custom_color")
        x_color = y_color = None
        if custom_label_colors:
            c_x, c_y = st.columns(2)
            label_fg = base_theme.foreground
            x_color = c_x.color_picker("X-axis labels", label_fg, key=f"axis_x_color_{chart_bg}")
            y_color = c_y.color_picker("Y-axis labels", label_fg, key=f"axis_y_color_{chart_bg}")
        st.markdown("**Rename labels** — add a row, pick the label, type the new text.")
        known_labels = sorted(set(st.session_state.get("seen_axis_labels", [])) | set(KNOWN_AXIS_LABELS))
        label_rows = st.data_editor(
            pd.DataFrame({"Original label": pd.Series(dtype="object"), "New text": pd.Series(dtype="object")}),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            key="axis_label_editor",
            column_config={
                "Original label": st.column_config.SelectboxColumn("Original label", options=known_labels, required=True),
                "New text": st.column_config.TextColumn("New text"),
            },
        )
        label_pairs = tuple(
            (str(o), str(n)) for o, n in zip(label_rows["Original label"], label_rows["New text"])
            if isinstance(o, str) and o and isinstance(n, str) and n.strip()
        )
    chart_theme = build_theme(
        chart_bg,
        palette_name,
        color_overrides,
        label_style={
            "label_font_family": None if font_choice == "Default" else font_choice,
            "label_font_size": float(size_choice) if size_choice else None,
            "label_bold": bold_choice,
            "x_label_color": x_color,
            "y_label_color": y_color,
            "label_text": label_pairs,
        },
    )
    reset_seen_labels()
    set_plot_theme(chart_theme)


dataset: RawSensorDataset = st.session_state.dataset

_ts = pd.to_datetime(dataset.df["timestamp"]).sort_values()
_span_sec = (_ts.iloc[-1] - _ts.iloc[0]).total_seconds() if len(_ts) > 1 else 0.0
_src_dt_sec = float(_ts.diff().dt.total_seconds().median()) if len(_ts) > 1 else float("nan")
if np.isfinite(_src_dt_sec) and epoch_sec < _src_dt_sec * 0.99:
    st.sidebar.caption(
        f"ℹ️ The source is sampled about every {format_bin_size(_src_dt_sec / 60.0)}, so bins of "
        f"{format_bin_size(epoch_min)} are mostly empty. Gaps of up to 60 min are interpolated, which adds no real detail."
    )
if _span_sec > 0 and _span_sec / epoch_sec < 24:
    st.error(
        f"This {_span_sec / 86400:.1f}-day recording yields only {int(_span_sec // epoch_sec)} bins at "
        f"{format_bin_size(epoch_min)}; analysis needs at least 24. Choose a smaller bin size."
    )
    st.stop()

cfg = CircadianAnalysisConfig(
    epoch_minutes=float(epoch_min),
    activity_method=act_method,
    use_log_lux=use_log_lux,
    n_harmonics=int(n_harm),
    period_mode=period_mode,
    custom_period_hours=float(custom_tau),
    period_min_hours=float(period_range[0]),
    period_max_hours=float(period_range[1]),
    circadian_min_hours=float(circ_band[0]),
    circadian_max_hours=float(circ_band[1]),
    bimodality_min_prominence=bimodal_prom / 100.0,
    onset_method=onset_method,
    split_day=int(split_day) if split_day is not None else None,
)

try:
    report: CircadianReport = run_full_analysis(dataset, cfg)
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.stop()

avail_channels = []
if report.activity_report is not None:
    avail_channels.append("Activity")
if report.light_report is not None:
    avail_channels.append("Light")

selected_ch_name = st.radio("Active Inspection Channel", avail_channels, horizontal=True) if len(avail_channels) > 1 else avail_channels[0]
rep = report.activity_report if selected_ch_name == "Activity" else report.light_report

for w in report.warnings:
    st.warning(w)

# Top KPI Cards
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Period τ (Consensus)", f"{rep.periodogram.consensus_tau_hours:.2f} h", f"Used: {rep.period_used_hours:.2f} h")
k2.metric("Single Cosinor MESOR", f"{rep.cosinor_single.mesor:.2f}", f"Amp: {rep.cosinor_single.amplitude:.2f}")
k3.metric("Acrophase (HH:MM)", rep.cosinor_single.acrophase_hhmm, f"{rep.cosinor_single.acrophase_deg_neg:.1f}°")
k4.metric(
    f"{rep.cosinor_harmonics.n_harmonics}-Harmonic R²",
    f"{rep.cosinor_harmonics.r2:.3f}",
    f"Ortho: {rep.cosinor_harmonics.orthophase_hhmm}",
)
if rep.circular_acrophase is not None:
    k5.metric("Circular R̄ / Rayleigh p", f"{rep.circular_acrophase.mean_resultant_length:.3f}", f"p={rep.circular_acrophase.rayleigh_p:.2e}")
else:
    k5.metric("Circular R̄", "N/A")

if rep.shift_report is not None and rep.shift_report.pre_post_acrophase is not None:
    ww = rep.shift_report.pre_post_acrophase
    k6.metric("Acrophase Shift (Δφ)", f"{ww.phase_shift_hours:+.2f} h", f"{ww.shift_direction} (p={ww.p_value:.3g})")
elif rep.shift_report is not None:
    k6.metric("Drift Slope", f"{rep.shift_report.drift_slope_hours_per_day:+.3f} h/d")
else:
    k6.metric("Acrophase Shift", "N/A")

tabs = st.tabs(
    [
        "📈 Overview & Time Series",
        "📊 Actograms",
        "🌊 Cosinor & Harmonics",
        "🔎 Periodicity, Rhythm Bands & Bimodality",
        "🧭 Circular Stats & Acrophase Shift",
        "📥 Excel & Data Export",
    ]
)

with tabs[0]:
    st.subheader(f"Binned Time Series & Cosinor Overlay — {rep.channel_name}")
    bdf_full = report.binned.df
    bdf = downsample_for_display(bdf_full)
    if len(bdf) < len(bdf_full):
        st.caption(
            f"Plot shows {len(bdf):,} averaged points out of {len(bdf_full):,} epochs. Analysis and CSV export use every epoch."
        )
    fig_ts = make_subplots(specs=[[{"secondary_y": report.binned.has_light and report.binned.has_activity}]])
    if report.binned.has_activity:
        fig_ts.add_trace(
            go.Scatter(x=bdf["timestamp"], y=bdf["activity"], name="Activity", line=dict(color=chart_theme.activity, width=1.1), opacity=0.75),
            secondary_y=False,
        )
    if report.binned.has_light:
        l_col = "log_lux" if cfg.use_log_lux else "lux"
        fig_ts.add_trace(
            go.Scatter(x=bdf["timestamp"], y=bdf[l_col], name=f"Light ({l_col})", line=dict(color=chart_theme.light, width=1.1), opacity=0.65),
            secondary_y=report.binned.has_activity,
        )
    if "cosinor_harmonics_fit" in bdf.columns:
        fig_ts.add_trace(
            go.Scatter(
                x=bdf["timestamp"],
                y=bdf["cosinor_harmonics_fit"],
                name=f"{cfg.n_harmonics}-Harmonic Fit",
                line=dict(color=chart_theme.fit, width=2.0),
            ),
            secondary_y=(selected_ch_name == "Light" and report.binned.has_activity),
        )
    fig_ts.update_layout(height=420, hovermode="x unified")
    apply_plotly_theme(fig_ts)
    st.plotly_chart(fig_ts, use_container_width=True)

    c_l, c_r = st.columns(2)
    with c_l:
        st.markdown("**Dataset Summary**")
        st.write(
            {
                "Source": report.source_name,
                "Start": str(report.binned.start_time),
                "End": str(report.binned.end_time),
                "Duration (days)": round(report.binned.n_days, 2),
                "Total Epochs": len(bdf_full),
                "Epoch Size": format_bin_size(report.binned.epoch_minutes),
            }
        )
    with c_r:
        if rep.non_parametric is not None:
            np_m = rep.non_parametric
            st.markdown("**Non-Parametric Circadian Variables (Van Someren)**")
            st.write(
                {
                    "Interdaily Stability (IS)": round(np_m.interdaily_stability_is, 4),
                    "Intradaily Variability (IV)": round(np_m.intradaily_variability_iv, 4),
                    "M10 (mean / onset)": f"{np_m.m10_value:.2f} @ {np_m.m10_onset_hhmm}",
                    "L5 (mean / onset)": f"{np_m.l5_value:.2f} @ {np_m.l5_onset_hhmm}",
                    "Relative Amplitude (RA)": round(np_m.relative_amplitude_ra, 4),
                }
            )

with tabs[1]:
    st.subheader("Actograms (Single & Double-Plotted)")
    ac1, ac2, ac3 = st.columns(3)
    dbl_plot = ac1.checkbox("Double-Plotted (48 h)", value=True)
    light_ov = ac2.checkbox("Overlay Ambient Light", value=True)
    show_mk = ac3.checkbox("Overlay Onsets / Offsets / Acrophases", value=True)

    fig_act_plotly = make_actogram_plotly(report, channel=rep, double_plotted=dbl_plot, show_light_overlay=light_ov)
    st.plotly_chart(fig_act_plotly, use_container_width=True)

    st.markdown("**Publication-Style Bar Actogram**")
    fig_act_mpl = make_actogram_mpl(report, channel=rep, double_plotted=dbl_plot, show_light_overlay=light_ov, show_markers=show_mk)
    png_act = figure_to_png_bytes(fig_act_mpl, dpi=160)
    st.image(png_act, use_container_width=True)
    st.download_button("Download Actogram PNG", data=png_act, file_name=f"actogram_{rep.channel_name.lower()}.png", mime="image/png")

with tabs[2]:
    st.subheader("Cosinor Rhythmometry & Circadian Harmonics")
    fig_harm = make_cosinor_harmonics_mpl(report, channel=rep)
    png_harm = figure_to_png_bytes(fig_harm, dpi=160)
    st.image(png_harm, use_container_width=True)

    harm_rows = [
        {
            "Harmonic": f"H{h.harmonic_index}",
            "Period (h)": round(h.period_hours, 2),
            "Amplitude": round(h.amplitude, 4),
            "95% CI": f"[{h.amplitude_ci_low:.3f}, {h.amplitude_ci_high:.3f}]",
            "Acrophase (HH:MM)": h.acrophase_hhmm,
            "Acrophase (° neg)": round(h.acrophase_deg_neg, 1),
            "Ak / A1": round(h.amplitude_ratio_to_h1, 3),
            "Partial PR (%)": round(h.percent_rhythm, 2),
            "F-stat": round(h.f_stat, 2),
            "p-value": f"{h.p_value:.3e}",
        }
        for h in rep.cosinor_harmonics.harmonics
    ]
    st.dataframe(pd.DataFrame(harm_rows), use_container_width=True, hide_index=True)

    if rep.population_cosinor is not None:
        pc = rep.population_cosinor
        st.info(
            f"**Population-Mean Cosinor ({pc.n_cycles} days):** MESOR = {pc.mesor_mean:.2f} ± {pc.mesor_se:.2f}, "
            f"Amplitude = {pc.amplitude:.2f}, Acrophase = {pc.acrophase_hhmm} ({pc.acrophase_deg_neg:.1f}°), "
            f"Hotelling F = {pc.f_stat:.2f}, p = {pc.p_value:.3e}"
        )

with tabs[3]:
    st.subheader("Periodicity Estimation & Non-Parametric Profile")
    fig_pg = make_periodogram_mpl(report, channel=rep)
    st.image(figure_to_png_bytes(fig_pg, dpi=160), use_container_width=True)

    pg = rep.periodogram
    p_cols = st.columns(4)
    p_cols[0].metric("Lomb-Scargle τ", f"{pg.best_tau_lomb_scargle:.2f} h", f"Power: {pg.best_ls_power:.3f}")
    p_cols[1].metric("Chi-Square (χ²) τ", f"{pg.best_tau_chi2:.2f} h", f"p = {pg.best_chi2_p_value:.2e}")
    p_cols[2].metric("Cosinor Sweep τ", f"{pg.best_tau_cosinor:.2f} h", f"R²: {pg.best_cosinor_r2:.3f}")
    p_cols[3].metric("Consensus τ", f"{pg.consensus_tau_hours:.2f} h")

    rb, bm = rep.rhythm_bands, rep.bimodality
    st.markdown("#### Ultradian · Circadian · Infradian Rhythms")
    if rb is not None:
        if rb.dominant_band:
            others = [
                f"{b.band.lower()} {b.period_hours:.1f} h" + (" (possible harmonic)" if "possible harmonic" in b.classification else "")
                for b in rb.detected_bands[1:]
            ]
            st.info(
                f"**Dominant rhythm: {rb.dominant_band} ({rb.dominant_period_hours:.1f} h)**"
                + (f" · also significant: {', '.join(others)}" if others else " · no other independent rhythm")
            )
        else:
            st.warning(f"No significant rhythm between {rb.period_min_hours:g} and {rb.period_max_hours:g} h.")
        band_rows = [
            {
                "Band": b.band,
                "Range (h)": b.range_label,
                "Result": b.classification,
                "Period (h)": round(b.period_hours, 1) if b.detected else None,
                "% Variance": round(b.percent_variance, 1) if b.detected else None,
                "Amplitude": round(b.amplitude, 3) if b.detected else None,
                "p (Šidák)": f"{b.p_adjusted:.2e}" if b.p_adjusted == b.p_adjusted else "",
                "Order": b.extraction_order,
                "Note": b.note,
            }
            for b in rb.bands
        ]
        st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Rhythms are extracted strongest-first; each is removed together with its harmonics before the next is tested, so "
            "harmonics (e.g. 12 h from a non-sinusoidal 24 h rhythm) and spectral sidelobes are not reported as separate rhythms. "
            f"Significant = Šidák-adjusted p < {rb.alpha:g} and ≥ {rb.min_percent_variance:g}% of total variance."
        )

    th = rep.twelve_hour
    if th is not None:
        st.markdown("#### 12 h Rhythm vs. 12 h Harmonic of the Circadian Rhythm")
        sig = lambda ok: "✅ significant" if ok else "— not significant"  # noqa: E731
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Component": f"12 h rhythm (own period, searched {th.search_window_hours[0]:g}–{th.search_window_hours[1]:g} h)",
                        "Period (h)": round(th.rhythm_period_hours, 2),
                        "Amplitude": round(th.rhythm_amplitude, 3),
                        "Peaks (HH:MM)": th.rhythm_peaks_hhmm,
                        "% Variance": round(th.rhythm_percent_variance, 2),
                        "p": f"{th.rhythm_p_value:.2e} (Šidák {th.rhythm_p_adjusted:.2e})",
                        "Status": sig(th.rhythm_significant),
                    },
                    {
                        "Component": f"τ/2 harmonic of {th.circadian_period_hours:.2f} h circadian rhythm",
                        "Period (h)": round(th.harmonic_period_hours, 2),
                        "Amplitude": round(th.harmonic_amplitude, 3),
                        "Peaks (HH:MM)": th.harmonic_peaks_hhmm,
                        "% Variance": round(th.harmonic_percent_variance, 2),
                        "p": f"{th.harmonic_p_value:.2e}",
                        "Status": sig(th.harmonic_significant) + f" · A2/A1 = {th.harmonic_to_fundamental_ratio:.2f}",
                    },
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Period difference {th.period_difference_hours:+.2f} h vs. resolution ±{th.resolution_hours:.2f} h → "
            f"{'distinct from the harmonic' if th.distinct_from_harmonic else 'indistinguishable from the harmonic'}. "
            + th.interpretation
        )

    if bm is not None:
        st.markdown("#### Bimodality")
        s = bm.shape
        b_cols = st.columns(4)
        b_cols[0].metric("Waveform", bm.classification.split(" (")[0], bm.classification[len(bm.classification.split(" (")[0]):].strip(" ()") or None, delta_color="off")
        b_cols[1].metric("Bimodality Index", f"{s.bimodality_index:.2f}", f"threshold {bm.min_prominence_fraction:.2f}", delta_color="off")
        peaks_txt = f"{bm.primary_peak_hhmm} & {bm.secondary_peak_hhmm}" if s.bimodality_index > 0 else bm.primary_peak_hhmm
        b_cols[2].metric("Peak(s)", peaks_txt, f"dip {bm.dip_hhmm}" if s.bimodality_index > 0 else None, delta_color="off")
        b_cols[3].metric("Bimodal Days", f"{bm.days_bimodal} / {bm.days_tested}", f"A2/A1 {bm.h2_to_h1_ratio:.2f}, p={bm.h2_p_value:.1e}", delta_color="off")

    if rb is not None or bm is not None or th is not None:
        st.image(figure_to_png_bytes(make_rhythm_bands_mpl(report, channel=rep), dpi=160), use_container_width=True)

    if rep.non_parametric is not None:
        prof = rep.non_parametric.hourly_profile
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Bar(x=list(range(24)), y=prof, marker_color=chart_theme.accent, name="24h Mean Profile"))
        fig_prof.update_layout(
            title="Average 24-Hour Profile (used for IS, IV, M10, L5, RA)",
            xaxis=dict(title="Hour of Day", tickvals=list(range(0, 24, 2))),
            yaxis=dict(title=rep.unit_label),
            height=320,
        )
        apply_plotly_theme(fig_prof)
        st.plotly_chart(fig_prof, use_container_width=True)

with tabs[4]:
    st.subheader("Circular Statistics Recorder & Acrophase Shift")
    fig_circ = make_circular_polar_mpl(report, channel=rep)
    st.image(figure_to_png_bytes(fig_circ, dpi=160), use_container_width=True)

    circ_rows = []
    for cs in (rep.circular_acrophase, rep.circular_onset, rep.circular_offset):
        if cs is None:
            continue
        circ_rows.append(
            {
                "Phase Marker": cs.label,
                "N": cs.n,
                "Circular Mean": cs.mean_hhmm,
                "Mean Angle (°)": round(cs.mean_angle_deg, 1),
                "Resultant R̄": round(cs.mean_resultant_length, 4),
                "Circular Variance": round(cs.circular_variance, 4),
                "Circular SD (h)": round(cs.circular_std_hours, 2),
                "κ": round(cs.kappa, 2),
                "Rayleigh Z": round(cs.rayleigh_z, 3),
                "Rayleigh p": f"{cs.rayleigh_p:.3e}",
            }
        )
    if circ_rows:
        st.dataframe(pd.DataFrame(circ_rows), use_container_width=True, hide_index=True)

    fig_shift = make_acrophase_shift_timeline_mpl(report, channel=rep)
    st.image(figure_to_png_bytes(fig_shift, dpi=160), use_container_width=True)

    if rep.shift_report is not None:
        sr = rep.shift_report
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Mean Daily Shift", f"{sr.mean_daily_shift_hours:+.3f} h/d")
        sc2.metric("Cumulative Shift", f"{sr.cumulative_shift_hours:+.2f} h")
        sc3.metric("Drift Slope / Implied τ", f"{sr.drift_slope_hours_per_day:+.3f} h/d", f"τ ≈ {sr.implied_tau_from_drift_hours:.2f} h")
        if sr.activity_light_phase_angle_hours is not None:
            sc4.metric("Phase Angle ψ (Act − Light)", f"{sr.activity_light_phase_angle_hours:+.2f} h")

    st.markdown("**Daily Rhythmometry, Onsets/Offsets & Acrophase Shifts**")
    st.dataframe(rep.daily_table, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Export Full Excel Workbook & Raw Tables")
    st.write(
        "The generated `.xlsx` workbook contains 6 formatted sheets: `Summary_Report`, `Cosinor_Harmonics`, "
        "`Daily_Rhythms_Onsets`, `Circular_Statistics` (with live Excel circular formulas), `Periodogram`, and `Binned_Epochs`, "
        "along with embedded high-resolution Actogram, Harmonics, Polar Rose, Acrophase Shift, and Periodogram charts."
    )
    # Fine bins (e.g. 1 s → 1.2M epochs) make these exports take many seconds and hundreds of MB, so past
    # the threshold build them on request instead of on every rerun. Cache keyed by dataset + config.
    n_epochs = len(report.binned.df)
    export_key = (id(dataset), repr(cfg), repr(chart_theme))  # charts embedded in Excel follow the theme
    big_export = n_epochs > LAZY_EXPORT_EPOCHS
    cached = st.session_state.get("export_cache")
    if cached is not None and cached["key"] != export_key:
        cached = st.session_state.export_cache = None

    if big_export and cached is None:
        st.info(
            f"{n_epochs:,} epochs at {format_bin_size(report.binned.epoch_minutes)} bins. Building the exports takes a while, "
            f"so they are prepared on request. The Excel `Binned_Epochs` sheet averages epochs down to ≤ 20,000 rows; the CSV keeps all of them."
        )
        if st.button("Prepare Excel & Binned CSV Exports", type="primary"):
            with st.spinner("Building exports…"):
                st.session_state.export_cache = cached = {
                    "key": export_key,
                    "xlsx": export_report_to_excel(report),
                    "csv": report.binned.df.to_csv(index=False).encode("utf-8"),
                }
    elif cached is None:
        cached = {
            "key": export_key,
            "xlsx": export_report_to_excel(report),
            "csv": report.binned.df.to_csv(index=False).encode("utf-8"),
        }
        if big_export:
            st.session_state.export_cache = cached

    if cached is not None:
        st.download_button(
            label="📊 Download Complete Excel Report (.xlsx)",
            data=cached["xlsx"],
            file_name=f"circadian_report_{datetime.now():%Y%m%d_%H%M}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

    c_dl1, c_dl2 = st.columns(2)
    c_dl1.download_button(
        "Download Daily Table (CSV)",
        data=rep.daily_table.to_csv(index=False).encode("utf-8"),
        file_name=f"daily_rhythms_{rep.channel_name.lower()}.csv",
        mime="text/csv",
    )
    if cached is not None:
        c_dl2.download_button(
            "Download Binned Epochs (CSV)",
            data=cached["csv"],
            file_name="binned_circadian_epochs.csv",
            mime="text/csv",
        )

# Remember the axis labels used by this run's charts so the sidebar can offer them for renaming
st.session_state["seen_axis_labels"] = seen_axis_labels()
