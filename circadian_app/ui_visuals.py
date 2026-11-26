"""Restrained scientific UI system, SVG phase dial, and lean Plotly figures for cosirca."""

from __future__ import annotations

import html
from typing import Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .cosinor import evaluate_cosinor_curve
from .pipeline import ChannelReport, CircadianReport
from .plotting import apply_plotly_theme
from .theme import PlotTheme, blend, to_rgba
from .theme import current_theme as T


def _esc(val: object) -> str:
    return html.escape(str(val))


def build_cosirca_favicon_uri(accent: str = "#2563eb") -> str:
    """Geometric SVG favicon for cosirca."""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        f'<rect width="64" height="64" rx="14" fill="#0f172a"/>'
        f'<circle cx="32" cy="32" r="20" fill="none" stroke="{accent}" stroke-width="3" stroke-opacity="0.35"/>'
        f'<path d="M32 12 A20 20 0 1 1 15 42" fill="none" stroke="{accent}" stroke-width="4" stroke-linecap="round"/>'
        f'<circle cx="32" cy="32" r="4.5" fill="{accent}"/>'
        f'</svg>'
    )
    return f"data:image/svg+xml,{quote(svg)}"


def inject_cosirca_css(theme: PlotTheme) -> None:
    """Inject clean, low-overhead CSS calibrated to the active PlotTheme."""
    is_dark = theme.is_dark
    bg = theme.background if is_dark else "#f8fafc"
    surface = blend(theme.background, theme.foreground, 0.045) if is_dark else "#ffffff"
    surface_alt = blend(theme.background, theme.foreground, 0.085) if is_dark else "#f1f5f9"
    border = to_rgba(theme.foreground, 0.13 if is_dark else 0.10)
    fg = theme.foreground
    muted = theme.muted
    accent = theme.accent

    st.markdown(
        f"""
        <style>
        @keyframes cosircaPulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.35; }}
        }}

        :root {{
            --cs-bg: {bg};
            --cs-surface: {surface};
            --cs-surface-alt: {surface_alt};
            --cs-border: {border};
            --cs-fg: {fg};
            --cs-muted: {muted};
            --cs-accent: {accent};
            --cs-onset: {theme.onset};
        }}

        .stApp {{
            background-color: var(--cs-bg);
            color: var(--cs-fg);
        }}

        .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1420px;
        }}

        section[data-testid="stSidebar"] {{
            background: {blend(theme.background, theme.foreground, 0.025) if is_dark else "#f8fafc"} !important;
            border-right: 1px solid var(--cs-border);
        }}
        section[data-testid="stSidebar"] h2 {{
            font-size: 0.74rem !important;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: var(--cs-muted) !important;
        }}

        .cosirca-topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
            padding: 0.7rem 1rem;
            margin-bottom: 0.85rem;
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-radius: 10px;
        }}
        .cosirca-brand {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }}
        .cosirca-wordmark {{
            font-size: 1.45rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--cs-fg);
            line-height: 1;
        }}
        .cosirca-meta {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.78rem;
            color: var(--cs-muted);
            font-variant-numeric: tabular-nums;
        }}

        .cosirca-status {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.76rem;
            padding: 0.22rem 0.6rem;
            border-radius: 6px;
            background: var(--cs-surface-alt);
            border: 1px solid var(--cs-border);
        }}
        .cosirca-dot-live {{
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--cs-onset);
            animation: cosircaPulse 1.5s ease-in-out infinite;
        }}
        .cosirca-dot-idle {{
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--cs-muted);
        }}

        div[data-testid="stMetric"] {{
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-radius: 10px;
            padding: 0.72rem 0.9rem !important;
        }}
        div[data-testid="stMetricLabel"] p {{
            font-size: 0.70rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--cs-muted) !important;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
            font-size: 1.24rem !important;
            font-weight: 600 !important;
            font-variant-numeric: tabular-nums;
            white-space: normal !important;
            line-height: 1.2 !important;
        }}
        div[data-testid="stMetricDelta"] {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
            font-size: 0.72rem !important;
            white-space: normal !important;
        }}

        .cosirca-kv-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.5rem;
        }}
        .cosirca-kv {{
            padding: 0.5rem 0.7rem;
            border-radius: 8px;
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
        }}
        .cosirca-kv-k {{
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--cs-muted);
        }}
        .cosirca-kv-v {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.86rem;
            font-weight: 600;
            color: var(--cs-fg);
            margin-top: 0.1rem;
            font-variant-numeric: tabular-nums;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cosirca_header(report: CircadianReport, rep: ChannelReport, bin_label: str) -> None:
    """Render clean cosirca header bar with compact telemetry readout."""
    th = T()
    meta_str = (
        f"{report.source_name}  |  {report.binned.n_days:.1f} d  |  "
        f"{bin_label} bins  |  {len(report.binned.df):,} epochs  |  {rep.channel_name}  |  τ = {rep.period_used_hours:.2f} h"
    )
    st.markdown(
        f"""
        <div class="cosirca-topbar">
            <div class="cosirca-brand">
                <svg width="28" height="28" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="24" cy="24" r="19" stroke="{to_rgba(th.foreground, 0.18)}" stroke-width="2"/>
                    <path d="M24 5 A19 19 0 1 1 8 34" stroke="{th.accent}" stroke-width="3.4" stroke-linecap="round"/>
                    <circle cx="24" cy="24" r="4" fill="{th.accent}"/>
                </svg>
                <span class="cosirca-wordmark">cosirca</span>
            </div>
            <div class="cosirca-meta">{_esc(meta_str)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_status_badge(running: bool, samples: int) -> None:
    dot_cls = "cosirca-dot-live" if running else "cosirca-dot-idle"
    state_txt = "Recording" if running else "Standby"
    st.markdown(
        f'<div class="cosirca-status"><span class="{dot_cls}"></span><span>{_esc(state_txt)} · {samples:,} samples</span></div>',
        unsafe_allow_html=True,
    )


def render_kv_card(items: list[tuple[str, str]]) -> None:
    html_cells = "".join(
        f'<div class="cosirca-kv"><div class="cosirca-kv-k">{_esc(k)}</div><div class="cosirca-kv-v">{_esc(v)}</div></div>'
        for k, v in items
    )
    st.markdown(f'<div class="cosirca-kv-grid">{html_cells}</div>', unsafe_allow_html=True)


def render_phase_summary_svg(report: CircadianReport, rep: ChannelReport) -> None:
    """Render an inline, zero-overhead SVG 24h phase summary (onset-offset arc, acrophase vector, pre/post shift)."""
    th = T()
    is_dark = th.is_dark
    surface = blend(th.background, th.foreground, 0.045) if is_dark else "#ffffff"
    border = to_rgba(th.foreground, 0.13 if is_dark else 0.10)

    acro_h = float(rep.cosinor_single.acrophase_hours)
    onset_h = float(rep.circular_onset.mean_hour) if rep.circular_onset is not None else float((acro_h - 6.0) % 24.0)
    offset_h = float(rep.circular_offset.mean_hour) if rep.circular_offset is not None else float((acro_h + 7.0) % 24.0)
    r_bar = float(rep.circular_acrophase.mean_resultant_length) if rep.circular_acrophase is not None else 0.75

    def polar_xy(hr: float, rad: float, cx: float = 94.0, cy: float = 94.0) -> tuple[float, float]:
        ang = (hr / 24.0) * 2.0 * np.pi - np.pi / 2.0
        return cx + rad * float(np.cos(ang)), cy + rad * float(np.sin(ang))

    def arc_path(h1: float, h2: float, rad: float, cx: float = 94.0, cy: float = 94.0) -> str:
        delta = (h2 - h1) % 24.0
        large = 1 if delta > 12.0 else 0
        x1, y1 = polar_xy(h1, rad, cx, cy)
        x2, y2 = polar_xy(h2, rad, cx, cy)
        return f"M {x1:.1f} {y1:.1f} A {rad:.1f} {rad:.1f} 0 {large} 1 {x2:.1f} {y2:.1f}"

    vx, vy = polar_xy(acro_h, max(16.0, min(64.0, r_bar * 64.0)))
    onx, ony = polar_xy(onset_h, 66.0)
    offx, offy = polar_xy(offset_h, 66.0)

    shift_svg = ""
    shift_txt = "No split shift"
    if rep.shift_report is not None and rep.shift_report.pre_post_acrophase is not None:
        ww = rep.shift_report.pre_post_acrophase
        s_col = th.delay if ww.phase_shift_hours >= 0 else th.advance
        h_a, h_b = (ww.mean1_hour, ww.mean2_hour) if ww.phase_shift_hours >= 0 else (ww.mean2_hour, ww.mean1_hour)
        shift_svg = f'<path d="{arc_path(h_a, h_b, 46.0)}" fill="none" stroke="{s_col}" stroke-width="3.5" stroke-linecap="round"/>'
        shift_txt = f"{ww.mean1_hhmm} -> {ww.mean2_hhmm} ({ww.phase_shift_hours:+.2f} h)"

    ticks_svg = ""
    for hr in (0, 6, 12, 18):
        tx, ty = polar_xy(hr, 78.0)
        ticks_svg += (
            f'<text x="{tx:.1f}" y="{ty + 3:.1f}" text-anchor="middle" '
            f'fill="{th.muted}" font-size="9" font-family="monospace">{hr:02d}</text>'
        )

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap;padding:0.75rem 1rem;
                    background:{surface};border:1px solid {border};border-radius:10px;margin-bottom:0.75rem;">
            <svg width="188" height="188" viewBox="0 0 188 188" style="flex-shrink:0;">
                <circle cx="94" cy="94" r="66" fill="none" stroke="{th.grid}" stroke-width="1.2"/>
                <circle cx="94" cy="94" r="46" fill="none" stroke="{th.grid}" stroke-width="0.9" stroke-dasharray="3 3"/>
                <path d="{arc_path(onset_h, offset_h, 66.0)}" fill="none" stroke="{to_rgba(th.onset, 0.45)}" stroke-width="6" stroke-linecap="round"/>
                {shift_svg}
                <line x1="94" y1="94" x2="{vx:.1f}" y2="{vy:.1f}" stroke="{th.acrophase}" stroke-width="2.6" stroke-linecap="round"/>
                <circle cx="{vx:.1f}" cy="{vy:.1f}" r="4" fill="{th.acrophase}"/>
                <circle cx="94" cy="94" r="3" fill="{th.foreground}"/>
                <circle cx="{onx:.1f}" cy="{ony:.1f}" r="3.5" fill="{th.onset}"/>
                <circle cx="{offx:.1f}" cy="{offy:.1f}" r="3.5" fill="{th.offset}"/>
                {ticks_svg}
            </svg>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:0.6rem 1.2rem;flex:1;min-width:220px;">
                <div><div class="cosirca-kv-k">Acrophase / Bathyphase</div><div class="cosirca-kv-v">{_esc(rep.cosinor_single.acrophase_hhmm)} / {_esc(rep.cosinor_single.bathyphase_hhmm)}</div></div>
                <div><div class="cosirca-kv-k">Orthophase ({rep.cosinor_harmonics.n_harmonics}H)</div><div class="cosirca-kv-v">{_esc(rep.cosinor_harmonics.orthophase_hhmm)} (R² {rep.cosinor_harmonics.r2:.3f})</div></div>
                <div><div class="cosirca-kv-k">Mean Onset / Offset</div><div class="cosirca-kv-v">{_esc(rep.circular_onset.mean_hhmm if rep.circular_onset else "--:--")} / {_esc(rep.circular_offset.mean_hhmm if rep.circular_offset else "--:--")}</div></div>
                <div><div class="cosirca-kv-k">Phase Shift</div><div class="cosirca-kv-v">{_esc(shift_txt)}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_animated_harmonics_plotly(report: CircadianReport, channel: Optional[ChannelReport] = None) -> go.Figure:
    """Interactive Plotly Harmonic Reconstruction (H1 -> H1+H2 -> ... -> H_N) + Harmonic Spectrum."""
    rep = channel or report.primary
    df = report.binned.df
    cos1 = rep.cosinor_single
    cosh = rep.cosinor_harmonics
    th = T()

    t_mod = df["elapsed_hours"].to_numpy(dtype=float) % cosh.period_hours
    y_vals = df[rep.value_col].to_numpy(dtype=float)
    valid = np.isfinite(t_mod) & np.isfinite(y_vals)
    bins = np.floor(t_mod[valid]).astype(int) % 24
    mean_h = [float(np.mean(y_vals[valid][bins == h])) if np.any(bins == h) else np.nan for h in range(24)]
    se_h = [
        float(np.std(y_vals[valid][bins == h], ddof=1) / np.sqrt(np.sum(bins == h))) if np.sum(bins == h) > 1 else 0.0
        for h in range(24)
    ]

    grid = np.linspace(0, cosh.period_hours, 240)
    y_sin1 = evaluate_cosinor_curve(cos1, grid)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=("Folded 24h Profile & Harmonic Synthesis", "Harmonic Amplitude Spectrum"),
        horizontal_spacing=0.09,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(24) + 0.5,
            y=mean_h,
            error_y=dict(type="data", array=se_h, visible=True, thickness=1.1, color=th.muted),
            mode="lines+markers",
            name="Hourly Mean +/- SE",
            line=dict(color=th.foreground, width=1.6),
            marker=dict(size=6, color=th.foreground),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=y_sin1,
            mode="lines",
            name=f"1-Harmonic (R²={cos1.r2:.3f})",
            line=dict(color=th.accent, width=1.8, dash="dash"),
        ),
        row=1,
        col=1,
    )

    y_full = evaluate_cosinor_curve(cosh, grid, up_to_harmonic=cosh.n_harmonics)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=y_full,
            mode="lines",
            name=f"Composite ({cosh.n_harmonics}H, R²={cosh.r2:.3f})",
            line=dict(color=th.fit, width=2.8),
        ),
        row=1,
        col=1,
    )

    h_labels = [f"H{h.harmonic_index} ({h.period_hours:.1f}h)" for h in cosh.harmonics]
    h_amps = [h.amplitude for h in cosh.harmonics]
    h_colors = [th.accent if h.p_value < 0.05 else th.muted for h in cosh.harmonics]
    h_text = [f"{h.percent_rhythm:.0f}%{'*' if h.p_value < 0.05 else ''}" for h in cosh.harmonics]
    fig.add_trace(
        go.Bar(
            x=h_labels,
            y=h_amps,
            marker=dict(color=h_colors, line=dict(color=th.foreground, width=0.6)),
            text=h_text,
            textposition="outside",
            textfont=dict(size=11),
            name="Harmonic Amplitude",
        ),
        row=1,
        col=2,
    )

    frames = []
    if cosh.n_harmonics > 1:
        for k in range(1, cosh.n_harmonics + 1):
            yk = evaluate_cosinor_curve(cosh, grid, up_to_harmonic=k)
            frames.append(
                go.Frame(
                    name=f"H{k}",
                    data=[go.Scatter(x=grid, y=yk)],
                    traces=[2],
                )
            )
    fig.frames = frames
    fig.add_hline(y=cosh.mesor, line_dash="dot", line_color=th.mesor, row=1, col=1)

    if frames:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.0,
                    y=-0.22,
                    xanchor="left",
                    yanchor="top",
                    direction="left",
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"fromcurrent": False, "frame": {"duration": 550, "redraw": False}, "transition": {"duration": 300}}]),
                        dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=len(frames) - 1,
                    x=0.14,
                    y=-0.20,
                    len=0.46,
                    pad={"t": 0, "b": 0},
                    currentvalue={"prefix": "Harmonics: ", "font": {"size": 11}},
                    steps=[
                        dict(method="animate", args=[[f.name], {"mode": "immediate", "frame": {"duration": 350, "redraw": False}}], label=f.name)
                        for f in frames
                    ],
                )
            ],
        )

    fig.update_xaxes(title_text="Phase / Clock Hour (h)", range=[0, cosh.period_hours], tickvals=list(range(0, int(round(cosh.period_hours)) + 1, 4)), row=1, col=1)
    fig.update_yaxes(title_text=f"{rep.channel_name} ({rep.unit_label})", row=1, col=1)
    fig.update_xaxes(title_text="Harmonic Component", row=1, col=2)
    fig.update_yaxes(title_text=f"Amplitude ({rep.unit_label})", range=[0, max(h_amps) * 1.34 if h_amps else 1.0], row=1, col=2)
    fig.update_layout(
        height=540,
        margin=dict(t=60, b=165, l=65, r=25),
        legend=dict(orientation="h", yanchor="top", y=-0.42 if frames else -0.22, x=0),
    )
    return apply_plotly_theme(fig)


def make_interactive_periodogram_plotly(report: CircadianReport, channel: Optional[ChannelReport] = None) -> go.Figure:
    """Interactive 3-panel Plotly Periodogram (Lomb-Scargle, Chi-Square, Cosinor R^2)."""
    rep = channel or report.primary
    pg = rep.periodogram
    th = T()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Lomb-Scargle (τ = {pg.best_tau_lomb_scargle:.2f} h)",
            f"Chi-Square χ² (τ = {pg.best_tau_chi2:.2f} h)",
            f"Cosinor R² Sweep (τ = {pg.best_tau_cosinor:.2f} h)",
        ),
        horizontal_spacing=0.07,
    )

    fig.add_trace(go.Scatter(x=pg.periods_hours, y=pg.lomb_scargle_power, mode="lines", name="Lomb-Scargle", line=dict(color=th.accent, width=2.0)), row=1, col=1)
    fig.add_vline(x=pg.best_tau_lomb_scargle, line_dash="dash", line_color=th.fit, row=1, col=1)

    fig.add_trace(go.Scatter(x=pg.periods_hours, y=pg.chi2_qp, mode="lines", name="Qp Statistic", line=dict(color=th.chi2, width=2.0)), row=1, col=2)
    fig.add_trace(go.Scatter(x=pg.periods_hours, y=pg.chi2_crit_95, mode="lines", name="p = 0.05 Crit", line=dict(color=th.threshold, width=1.5, dash="dot")), row=1, col=2)
    fig.add_vline(x=pg.best_tau_chi2, line_dash="dash", line_color=th.fit, row=1, col=2)

    fig.add_trace(go.Scatter(x=pg.periods_hours, y=pg.cosinor_r2, mode="lines", name="Cosinor R²", line=dict(color=th.sweep, width=2.0)), row=1, col=3)
    fig.add_vline(x=pg.best_tau_cosinor, line_dash="dash", line_color=th.fit, row=1, col=3)

    if rep.rhythm_bands is not None:
        for b in rep.rhythm_bands.bands:
            if b.scanned and b.high_hours > b.low_hours:
                for col_i in (1, 2, 3):
                    fig.add_vrect(x0=b.low_hours, x1=b.high_hours, fillcolor=th.band_colors[b.band], opacity=0.06, line_width=0, row=1, col=col_i)

    for col_i in (1, 2, 3):
        fig.update_xaxes(title_text="Period τ (hours)", row=1, col=col_i)
    fig.update_yaxes(title_text="Normalized Power", row=1, col=1)
    fig.update_yaxes(title_text="Qp", row=1, col=2)
    fig.update_yaxes(title_text="R²", row=1, col=3)
    fig.update_layout(
        height=420,
        margin=dict(t=60, b=105, l=60, r=25),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
    )
    return apply_plotly_theme(fig)


def make_interactive_rhythm_bands_plotly(report: CircadianReport, channel: Optional[ChannelReport] = None) -> go.Figure:
    """Interactive Plotly Rhythm Bands spectrum + Bimodality 2-harmonic waveform."""
    rep = channel or report.primary
    rb, bm = rep.rhythm_bands, rep.bimodality
    th = T()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Rhythm Bands: Single-Cosinor R² Spectrum",
            f"Waveform Bimodality{f': {bm.classification}' if bm else ''}",
        ),
        horizontal_spacing=0.09,
    )

    if rb is not None:
        for b in rb.bands:
            if b.scanned and b.high_hours > b.low_hours:
                fig.add_vrect(x0=b.low_hours, x1=b.high_hours, fillcolor=th.band_colors[b.band], opacity=0.07, line_width=0, row=1, col=1)
                fig.add_annotation(
                    x=(b.low_hours + b.high_hours) / 2.0,
                    y=1.02,
                    yref="y domain",
                    xref="x",
                    text=b.band,
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color=th.band_colors[b.band]),
                )
        fig.add_trace(go.Scatter(x=rb.periods_hours, y=rb.raw_r2, mode="lines", name="Raw spectrum", line=dict(color=th.foreground, width=1.9)), row=1, col=1)
        fig.add_trace(go.Scatter(x=rb.periods_hours, y=rb.residual_r2, mode="lines", name="Residual after prewhitening", line=dict(color=th.fit, width=1.5, dash="dash")), row=1, col=1)
        for b in rb.detected_bands:
            y_at = float(np.interp(b.period_hours, rb.periods_hours, rb.raw_r2))
            fig.add_trace(go.Scatter(x=[b.period_hours], y=[y_at], mode="markers+text", text=[f"#{b.extraction_order} {b.period_hours:.1f}h"], textposition="top center", name=f"{b.band} ({b.period_hours:.1f}h)", marker=dict(size=9, color=th.band_colors[b.band])), row=1, col=1)
        rth = rep.twelve_hour
        if rth is not None:
            fig.add_vline(x=rth.harmonic_period_hours, line_dash="dot", line_color=th.harmonic, row=1, col=1)
            fig.add_vline(x=rth.rhythm_period_hours, line_dash="dashdot", line_color=th.highlight, row=1, col=1)
        top_r2 = float(max(rb.raw_r2.max(), rb.residual_r2.max(), 1e-3))
        fig.update_xaxes(title_text="Period (hours)", range=[rb.period_min_hours, rb.period_max_hours], row=1, col=1)
        fig.update_yaxes(title_text="R² (variance explained)", range=[0, top_r2 * 1.24], row=1, col=1)

    if bm is not None:
        df = report.binned.df
        P = bm.period_hours
        bw = max(0.5, report.binned.epoch_minutes / 60.0)
        phase = df["elapsed_hours"].to_numpy(dtype=float) % P
        vals = df[rep.value_col].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        key = np.floor(phase[ok] / bw) * bw + bw / 2
        prof = pd.Series(vals[ok]).groupby(key).mean()

        fig.add_trace(go.Scatter(x=prof.index, y=prof.values, mode="markers", name="Folded mean", marker=dict(color=th.muted, size=6, opacity=0.7)), row=1, col=2)
        fig.add_trace(go.Scatter(x=bm.waveform_hours, y=bm.waveform_values, mode="lines", name="24h + 12h cosinor fit", line=dict(color=th.foreground, width=2.2)), row=1, col=2)
        s = bm.shape
        fig.add_trace(go.Scatter(x=[s.primary_hours], y=[s.primary_value], mode="markers", name=f"Main peak {bm.primary_peak_hhmm}", marker=dict(symbol="triangle-up", size=11, color=th.fit)), row=1, col=2)
        if s.bimodality_index > 0:
            fig.add_trace(go.Scatter(x=[s.secondary_hours], y=[s.secondary_value], mode="markers", name=f"2nd peak {bm.secondary_peak_hhmm}", marker=dict(symbol="triangle-up", size=11, color=th.highlight)), row=1, col=2)
            fig.add_trace(go.Scatter(x=[s.dip_hours], y=[s.dip_value], mode="markers", name=f"Dip {bm.dip_hhmm}", marker=dict(symbol="triangle-down", size=10, color=th.accent)), row=1, col=2)
        fig.update_xaxes(title_text="Phase / Clock Hour (h)", range=[0, P], tickvals=list(np.arange(0, P + 0.01, 4 if P >= 16 else 2)), row=1, col=2)
        fig.update_yaxes(title_text=rep.unit_label, row=1, col=2)

    # Push subplot titles above the band labels
    for ann in fig.layout.annotations:
        if ann.text in ("Rhythm Bands: Single-Cosinor R² Spectrum",) or ann.text.startswith("Waveform Bimodality"):
            ann.yshift = 22
    fig.update_layout(height=460, margin=dict(t=85, b=115, l=60, r=25), legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    return apply_plotly_theme(fig)


def make_animated_polar_plotly(report: CircadianReport, channel: Optional[ChannelReport] = None) -> go.Figure:
    """Interactive 24-hour Polar Phase Vector & Pre/Post Acrophase Shift Rose Plot with vector progression."""
    rep = channel or report.primary
    dtbl = rep.daily_table
    th = T()

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=(
            f"24h Circular Phase Vectors{f' (R̄={rep.circular_acrophase.mean_resultant_length:.2f})' if rep.circular_acrophase else ''}",
            (
                f"Acrophase Shift: {rep.shift_report.pre_post_acrophase.phase_shift_hours:+.2f} h ({rep.shift_report.pre_post_acrophase.shift_direction})"
                if rep.shift_report and rep.shift_report.pre_post_acrophase
                else "Acrophase Shift Comparison"
            ),
        ),
        horizontal_spacing=0.10,
    )

    acr_deg = (dtbl["acrophase_hours"].dropna().to_numpy() / 24.0 * 360.0) if "acrophase_hours" in dtbl.columns else np.array([])
    on_deg = (dtbl["onset_hours"].dropna().to_numpy() / 24.0 * 360.0) if "onset_hours" in dtbl.columns else np.array([])
    off_deg = (dtbl["offset_hours"].dropna().to_numpy() / 24.0 * 360.0) if "offset_hours" in dtbl.columns else np.array([])

    fig.add_trace(go.Scatterpolar(r=np.full_like(acr_deg, 0.96), theta=acr_deg, mode="markers", name="Daily Acrophase", marker=dict(color=th.acrophase, size=8)), row=1, col=1)
    fig.add_trace(go.Scatterpolar(r=np.full_like(on_deg, 0.84), theta=on_deg, mode="markers", name="Daily Onset", marker=dict(color=th.onset, size=8, symbol="triangle-up")), row=1, col=1)
    fig.add_trace(go.Scatterpolar(r=np.full_like(off_deg, 0.72), theta=off_deg, mode="markers", name="Daily Offset", marker=dict(color=th.offset, size=8, symbol="triangle-down")), row=1, col=1)

    for circ, name, col in [
        (rep.circular_acrophase, "Mean Acrophase", th.acrophase),
        (rep.circular_onset, "Mean Onset", th.onset),
        (rep.circular_offset, "Mean Offset", th.offset),
    ]:
        if circ is not None:
            fig.add_trace(
                go.Scatterpolar(r=[0, circ.mean_resultant_length], theta=[0, circ.mean_angle_deg], mode="lines+markers", name=f"{name} ({circ.mean_hhmm})", line=dict(color=col, width=2.8), marker=dict(size=[3, 9], color=col)),
                row=1,
                col=1,
            )

    sr = rep.shift_report
    if sr is not None and sr.pre_post_acrophase is not None:
        pre_df = sr.daily_shift_df[sr.daily_shift_df["day_index"] < sr.split_day]
        post_df = sr.daily_shift_df[sr.daily_shift_df["day_index"] >= sr.split_day]
        pre_deg = pre_df["acrophase_hours"].dropna().to_numpy() / 24.0 * 360.0
        post_deg = post_df["acrophase_hours"].dropna().to_numpy() / 24.0 * 360.0
        ww = sr.pre_post_acrophase
        fig.add_trace(go.Scatterpolar(r=np.full_like(pre_deg, 0.92), theta=pre_deg, mode="markers", name=f"Baseline (D<{sr.split_day})", marker=dict(color=th.advance, size=8)), row=1, col=2)
        fig.add_trace(go.Scatterpolar(r=np.full_like(post_deg, 0.80), theta=post_deg, mode="markers", name=f"Post-Split (D>={sr.split_day})", marker=dict(color=th.delay, size=8)), row=1, col=2)
        fig.add_trace(go.Scatterpolar(r=[0, ww.r_bar1], theta=[0, ww.mean1_hour / 24.0 * 360.0], mode="lines+markers", name=f"Pre Mean ({ww.mean1_hhmm})", line=dict(color=th.advance, width=3.0), marker=dict(size=[3, 9])), row=1, col=2)
        fig.add_trace(go.Scatterpolar(r=[0, ww.r_bar2], theta=[0, ww.mean2_hour / 24.0 * 360.0], mode="lines+markers", name=f"Post Mean ({ww.mean2_hhmm})", line=dict(color=th.delay, width=3.0), marker=dict(size=[3, 9])), row=1, col=2)

    max_n = max(len(acr_deg), len(on_deg), len(off_deg), 1)
    frames = [
        go.Frame(
            name=f"D{i}",
            data=[
                go.Scatterpolar(r=np.full(min(i, len(acr_deg)), 0.96), theta=acr_deg[:i]),
                go.Scatterpolar(r=np.full(min(i, len(on_deg)), 0.84), theta=on_deg[:i]),
                go.Scatterpolar(r=np.full(min(i, len(off_deg)), 0.72), theta=off_deg[:i]),
            ],
            traces=[0, 1, 2],
        )
        for i in range(1, max_n + 1)
    ] if max_n > 1 else []
    fig.frames = frames

    polar_cfg = dict(
        bgcolor=th.background,
        angularaxis=dict(direction="clockwise", rotation=90, tickmode="array", tickvals=list(range(0, 360, 45)), ticktext=["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"], gridcolor=th.grid, linecolor=th.muted),
        radialaxis=dict(range=[0, 1.12], tickvals=[0.25, 0.5, 0.75, 1.0], gridcolor=th.grid, linecolor=th.muted),
    )

    if frames:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.0,
                    y=-0.16,
                    xanchor="left",
                    yanchor="top",
                    direction="left",
                    buttons=[
                        dict(label="Play", method="animate", args=[None, {"fromcurrent": False, "frame": {"duration": 260, "redraw": True}}]),
                        dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
                    ],
                )
            ]
        )

    fig.update_annotations(yshift=28)
    fig.update_layout(
        polar=polar_cfg,
        polar2=polar_cfg,
        height=540,
        margin=dict(t=95, b=145, l=45, r=45),
        legend=dict(orientation="h", yanchor="top", y=-0.28 if frames else -0.18, x=0),
    )
    return apply_plotly_theme(fig)


def make_interactive_shift_timeline_plotly(report: CircadianReport, channel: Optional[ChannelReport] = None) -> go.Figure:
    """Interactive Plotly daily Phase Trajectory (Acrophase/Onset/Offset) and Daily + Cumulative Shift."""
    rep = channel or report.primary
    sr = rep.shift_report
    dtbl = sr.daily_shift_df if sr is not None else rep.daily_table
    th = T()

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        subplot_titles=("Daily Phase Trajectory (Acrophase, Onset, Offset)", "Acrophase Shift (+Delay / -Advance)"),
        horizontal_spacing=0.09,
    )

    days = dtbl["day_index"].to_numpy()
    if "acrophase_hours" in dtbl.columns:
        fig.add_trace(go.Scatter(x=days, y=dtbl["acrophase_hours"], mode="lines+markers", name="Acrophase (h)", line=dict(color=th.acrophase, width=2.2), marker=dict(size=6)), row=1, col=1)
    if "onset_hours" in dtbl.columns:
        fig.add_trace(go.Scatter(x=days, y=dtbl["onset_hours"], mode="lines+markers", name="Onset (h)", line=dict(color=th.onset, width=1.5, dash="dash"), marker=dict(symbol="triangle-up", size=6)), row=1, col=1)
    if "offset_hours" in dtbl.columns:
        fig.add_trace(go.Scatter(x=days, y=dtbl["offset_hours"], mode="lines+markers", name="Offset (h)", line=dict(color=th.offset, width=1.5, dash="dash"), marker=dict(symbol="triangle-down", size=6)), row=1, col=1)
    if sr is not None:
        fig.add_vline(x=sr.split_day - 0.5, line_dash="dot", line_color=th.delay, row=1, col=1)

    if "acrophase_shift_hours" in dtbl.columns:
        shifts = np.nan_to_num(dtbl["acrophase_shift_hours"].to_numpy(dtype=float), nan=0.0)
        bar_cols = [th.delay if s > 0 else th.advance for s in shifts]
        fig.add_trace(go.Bar(x=days, y=shifts, name="Daily Δφ (h)", marker_color=bar_cols, opacity=0.85), row=1, col=2, secondary_y=False)
        if "cumulative_shift_hours" in dtbl.columns:
            fig.add_trace(go.Scatter(x=days, y=dtbl["cumulative_shift_hours"], mode="lines+markers", name="Cumulative Shift (h)", line=dict(color=th.foreground, width=2.0), marker=dict(size=5)), row=1, col=2, secondary_y=True)

    fig.update_xaxes(title_text="Day Index", row=1, col=1)
    fig.update_xaxes(title_text="Day Index", row=1, col=2)
    fig.update_yaxes(title_text="Clock Time", range=[0, 24], tickvals=list(range(0, 25, 4)), ticktext=[f"{h:02d}:00" for h in range(0, 25, 4)], row=1, col=1)
    fig.update_yaxes(title_text="Day-to-Day Shift Δφ (h)", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Shift (h)", row=1, col=2, secondary_y=True)
    fig.update_layout(
        height=440,
        margin=dict(t=60, b=110, l=60, r=60),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
    )
    return apply_plotly_theme(fig)
