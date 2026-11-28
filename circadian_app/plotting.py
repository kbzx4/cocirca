"""Publication-quality Matplotlib figures (for Excel embedding, CLI export, and static views) and Plotly interactive figures."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from plotly.subplots import make_subplots

from .circular_stats import hours_to_radians
from matplotlib.colors import to_hex

from .cosinor import evaluate_cosinor_curve, format_hours_hhmm
from .pipeline import ChannelReport, CircadianReport
from .theme import current_theme as T
from .theme import record_axis_label, to_rgba


def apply_mpl_theme(fig: Figure) -> Figure:
    """Recolour a Matplotlib figure's background, text, axes, ticks and legends for the active theme.

    Data colours come from theme roles when each figure is built; this pass handles everything left
    at Matplotlib defaults, so it can run once at render time for every figure.
    """
    th = T()
    fig.set_facecolor(th.background)
    for ax in fig.axes:
        ax.set_facecolor(th.background)
        for spine in ax.spines.values():
            spine.set_color(th.muted)
        ax.tick_params(colors=th.foreground, which="both")
        ax.title.set_color(th.foreground)
        text_map = th.label_text_map
        for lab, color in ((ax.xaxis.label, th.x_label_color), (ax.yaxis.label, th.y_label_color)):
            orig = getattr(lab, "_circ_orig", None) or lab.get_text()
            if not orig:
                continue
            lab._circ_orig = orig  # keep the chart's own text so re-rendering the same figure is idempotent
            record_axis_label(orig)
            lab.set_text(text_map.get(orig, orig))
            if th.label_font_family:
                lab.set_fontfamily(th.label_font_family)
            if th.label_font_size:
                lab.set_fontsize(th.label_font_size)
            if th.label_bold:
                lab.set_fontweight("bold")
            lab.set_color(color or th.foreground)
        if ax.name == "polar":
            ax.grid(True, color=th.grid)
        for txt in ax.texts:
            if to_hex(txt.get_color()) == "#000000":
                txt.set_color(th.foreground)
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_facecolor(th.background)
            leg.get_frame().set_edgecolor(th.muted)
            for t in leg.get_texts():
                t.set_color(th.foreground)
    return fig


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Background, font and grid colours for an interactive Plotly figure."""
    th = T()
    fig.update_layout(
        template="plotly_dark" if th.is_dark else "plotly_white",
        paper_bgcolor=th.background,
        plot_bgcolor=th.background,
        font=dict(color=th.foreground),
        legend=dict(bgcolor=to_rgba(th.background, 0.0), font=dict(color=th.foreground)),
    )
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font_color=th.foreground)
    fig.update_xaxes(gridcolor=th.grid, linecolor=th.muted, zerolinecolor=th.grid)
    fig.update_yaxes(gridcolor=th.grid, linecolor=th.muted, zerolinecolor=th.grid)

    text_map = th.label_text_map

    def style_title(axis, color):
        orig = axis.title.text
        if not orig:
            return
        record_axis_label(orig)
        text = text_map.get(orig, orig)
        font = dict(color=color or th.foreground)
        if th.label_font_family:
            font["family"] = f"{th.label_font_family}, sans-serif"
        if th.label_font_size:
            font["size"] = th.label_font_size
        axis.update(title=dict(text=f"<b>{text}</b>" if th.label_bold else text, font=font))

    fig.for_each_xaxis(lambda a: style_title(a, th.x_label_color))
    fig.for_each_yaxis(lambda a: style_title(a, th.y_label_color))
    return fig


def figure_to_png_bytes(fig: Figure, dpi: int = 160) -> bytes:
    """Render a Matplotlib Figure cleanly to PNG bytes using Agg without touching pyplot global state."""
    apply_mpl_theme(fig)
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


ACTOGRAM_MAX_BINS_PER_DAY = 1440  # 1-minute display resolution
TIMESERIES_MAX_POINTS = 20_000
MPL_ACTOGRAM_MAX_BARS_PER_ROW = 96  # above this, draw rows as stepped fills instead of thousands of bar patches


def _build_actogram_matrix(
    binned_df: pd.DataFrame,
    value_col: str,
    epoch_minutes: float,
    double_plotted: bool = True,
    max_bins_per_day: int = ACTOGRAM_MAX_BINS_PER_DAY,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build (n_days, bins_per_row) matrix for single (24h) or double-plotted (48h) actogram.

    Epochs finer than 24h / `max_bins_per_day` are mean-aggregated into display columns so very
    fine bins (e.g. 1 s = 86,400 per day) still render; returned x_hours reflect the display width.
    """
    bins_per_day = min(max(1, int(round((24 * 60) / epoch_minutes))), max_bins_per_day)
    disp_minutes = (24 * 60) / bins_per_day
    day_index = binned_df["day_index"].to_numpy()
    days = np.array(sorted(pd.unique(day_index)))
    n_days = len(days)
    first_dates = binned_df.groupby("day_index")["date"].first()
    day_labels: list[str] = [f"D{d} ({first_dates.loc[d]})" for d in days]

    rows = np.searchsorted(days, day_index)
    cols = np.clip(
        np.floor(binned_df["clock_hour"].to_numpy(dtype=float) * 60.0 / disp_minutes + 1e-6).astype(int),
        0,
        bins_per_day - 1,
    )
    vals = binned_df[value_col].to_numpy(dtype=float)
    ok = np.isfinite(vals)
    flat = rows[ok] * bins_per_day + cols[ok]
    size = n_days * bins_per_day
    sums = np.bincount(flat, weights=vals[ok], minlength=size)
    cnts = np.bincount(flat, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        day_rows = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan).reshape(n_days, bins_per_day)
    epoch_minutes = disp_minutes

    if not double_plotted:
        x_hours = np.arange(bins_per_day) * (epoch_minutes / 60.0)
        return day_rows, day_labels, x_hours

    # Double-plot: row r shows Day r (0-24h) followed by Day r+1 (24-48h)
    mat_48 = np.full((n_days, 2 * bins_per_day), np.nan, dtype=float)
    for r in range(n_days):
        mat_48[r, :bins_per_day] = day_rows[r]
        if r + 1 < n_days:
            mat_48[r, bins_per_day:] = day_rows[r + 1]
    x_hours = np.arange(2 * bins_per_day) * (epoch_minutes / 60.0)
    return mat_48, day_labels, x_hours


def downsample_for_display(df: pd.DataFrame, max_points: int = TIMESERIES_MAX_POINTS) -> pd.DataFrame:
    """Mean-aggregate consecutive rows so a browser time-series plot stays under `max_points` points."""
    if len(df) <= max_points:
        return df
    factor = int(np.ceil(len(df) / max_points))
    groups = np.arange(len(df)) // factor
    num_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    out = df[num_cols].groupby(groups).mean()
    out.insert(0, "timestamp", df["timestamp"].iloc[::factor].to_numpy()[: len(out)])
    return out.reset_index(drop=True)


def make_actogram_mpl(
    report: CircadianReport,
    channel: Optional[ChannelReport] = None,
    double_plotted: bool = True,
    show_light_overlay: bool = True,
    show_markers: bool = True,
) -> Figure:
    """Create a classic chronobiology Actogram (single or double-plotted) with optional light shading and onset/offset/acrophase markers."""
    rep = channel or report.primary
    df = report.binned.df
    epoch_min = report.binned.epoch_minutes

    mat, day_labels, x_hours = _build_actogram_matrix(df, rep.value_col, epoch_min, double_plotted=double_plotted)
    n_days = len(day_labels)

    has_light = show_light_overlay and report.binned.has_light and rep.channel_name != "Light"
    if has_light:
        light_col = "log_lux" if report.config.use_log_lux else "lux"
        light_mat, _, _ = _build_actogram_matrix(df, light_col, epoch_min, double_plotted=double_plotted)
    else:
        light_mat = None

    fig_h = max(4.2, 0.48 * n_days + 1.8)
    fig = Figure(figsize=(11.5, fig_h), facecolor="white")
    ax = fig.add_subplot(111)

    vmax = float(np.nanpercentile(mat, 97)) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-6)
    lmax = float(np.nanpercentile(light_mat, 97)) if (light_mat is not None and np.isfinite(light_mat).any()) else 1.0
    lmax = max(lmax, 1e-6)

    bin_w = (48.0 if double_plotted else 24.0) / mat.shape[1]
    for r in range(n_days):
        y_base = n_days - r - 1
        # Baseline line
        ax.axhline(y_base, color=T().grid, linewidth=0.7, zorder=1)

        if light_mat is not None:
            l_row = np.nan_to_num(light_mat[r], nan=0.0)
            l_norm = np.clip(l_row / lmax, 0.0, 1.0) * 0.85
            ax.fill_between(
                x_hours,
                y_base,
                y_base + l_norm,
                step="post",
                color=T().light,
                alpha=0.28,
                zorder=2,
                label="Ambient Light" if r == 0 else None,
            )

        row_vals = np.nan_to_num(mat[r], nan=0.0)
        heights = np.clip(row_vals / vmax, 0.0, 1.0) * 0.88
        bar_color = T().activity if rep.channel_name != "Light" else T().light
        bar_label = f"{rep.channel_name} ({rep.unit_label})" if r == 0 else None
        if len(x_hours) <= MPL_ACTOGRAM_MAX_BARS_PER_ROW:
            ax.bar(
                x_hours,
                heights,
                width=bin_w * 0.95,
                bottom=y_base,
                align="edge",
                color=bar_color,
                alpha=0.9,
                zorder=3,
                label=bar_label,
            )
        else:
            # One stepped polygon per row: same look once bars are ~pixel-wide, ~50x faster than thousands of patches
            ax.fill_between(
                np.append(x_hours, x_hours[-1] + bin_w),
                y_base,
                y_base + np.append(heights, heights[-1]),
                step="post",
                color=bar_color,
                alpha=0.9,
                linewidth=0,
                zorder=3,
                label=bar_label,
            )

    # Overlay daily onset, offset, and acrophase markers
    if show_markers and not rep.daily_table.empty:
        dtbl = rep.daily_table.set_index("day_index")
        days = sorted(df["day_index"].unique())
        offsets_to_plot = [0.0, 24.0] if double_plotted else [0.0]

        for idx_pass, shift_x in enumerate(offsets_to_plot):
            on_x, on_y = [], []
            off_x, off_y = [], []
            acr_x, acr_y = [], []
            for r, d in enumerate(days):
                src_day = d if shift_x == 0.0 else d + 1
                if src_day not in dtbl.index:
                    continue
                y_mid = (n_days - r - 1) + 0.45
                row = dtbl.loc[src_day]
                if "onset_hours" in row and pd.notna(row["onset_hours"]):
                    on_x.append(float(row["onset_hours"]) + shift_x)
                    on_y.append(y_mid)
                if "offset_hours" in row and pd.notna(row["offset_hours"]):
                    off_h = float(row["offset_hours"])
                    # If offset wrapped past midnight relative to onset, display continuously if double-plotted
                    if double_plotted and "onset_hours" in row and pd.notna(row["onset_hours"]) and off_h < float(row["onset_hours"]):
                        if shift_x == 0.0:
                            off_x.append(off_h + 24.0)
                            off_y.append(y_mid)
                    else:
                        off_x.append(off_h + shift_x)
                        off_y.append(y_mid)
                if "acrophase_hours" in row and pd.notna(row["acrophase_hours"]):
                    acr_x.append(float(row["acrophase_hours"]) + shift_x)
                    acr_y.append(y_mid)

            lbl = idx_pass == 0
            if on_x:
                ax.scatter(on_x, on_y, color=T().onset, marker="^", s=46, zorder=5, label="Onset" if lbl else None)
            if off_x:
                ax.scatter(off_x, off_y, color=T().offset, marker="v", s=46, zorder=5, label="Offset" if lbl else None)
            if acr_x:
                ax.scatter(acr_x, acr_y, color=T().acrophase, marker="o", s=34, zorder=6, label="Acrophase" if lbl else None)

    if double_plotted:
        ax.axvline(24.0, color=T().muted, linestyle="--", linewidth=1.0, zorder=4)
        ax.set_xlim(0, 48)
        ax.set_xticks(range(0, 49, 4))
        ax.set_xticklabels([f"{h % 24:02d}:00" for h in range(0, 49, 4)])
        ax.set_xlabel("Clock Time (Double-Plotted 48 h)", fontsize=10)
    else:
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 2))
        ax.set_xticklabels([f"{h % 24:02d}:00" for h in range(0, 25, 2)])
        ax.set_xlabel("Clock Time (24 h)", fontsize=10)

    ax.set_yticks([n_days - r - 0.55 for r in range(n_days)])
    ax.set_yticklabels(day_labels, fontsize=8.5)
    ax.set_ylim(-0.15, n_days + 0.15)
    title_kind = "Double-Plotted" if double_plotted else "Single-Plotted"
    ax.set_title(
        f"{title_kind} Actogram — {rep.channel_name} (τ = {rep.period_used_hours:.2f} h, MESOR = {rep.cosinor_single.mesor:.1f})",
        fontsize=12,
        fontweight="bold",
        pad=28,
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=5, fontsize=8, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_actogram_plotly(
    report: CircadianReport,
    channel: Optional[ChannelReport] = None,
    double_plotted: bool = True,
    show_light_overlay: bool = True,
) -> go.Figure:
    """Interactive Plotly actogram heatmap with overlaid onset, offset, and acrophase trajectories."""
    rep = channel or report.primary
    df = report.binned.df
    epoch_min = report.binned.epoch_minutes

    mat, day_labels, x_hours = _build_actogram_matrix(df, rep.value_col, epoch_min, double_plotted=double_plotted)
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=mat[::-1],
            x=x_hours,
            y=day_labels[::-1],
            colorscale=[[0, T().background], [1, T().activity if rep.channel_name != "Light" else T().light]],
            colorbar=dict(title=rep.unit_label, len=0.75),
            hovertemplate="Day: %{y}<br>Hour: %{x:.2f} h<br>Value: %{z:.2f}<extra></extra>",
            name=rep.channel_name,
        )
    )

    if show_light_overlay and report.binned.has_light and rep.channel_name != "Light":
        light_col = "log_lux" if report.config.use_log_lux else "lux"
        lmat, _, _ = _build_actogram_matrix(df, light_col, epoch_min, double_plotted=double_plotted)
        fig.add_trace(
            go.Heatmap(
                z=lmat[::-1],
                x=x_hours,
                y=day_labels[::-1],
                colorscale=[[0, to_rgba(T().light, 0.0)], [1, to_rgba(T().light, 0.38)]],
                showscale=False,
                hoverinfo="skip",
                name="Ambient Light",
            )
        )

    if not rep.daily_table.empty:
        dtbl = rep.daily_table.set_index("day_index")
        days = sorted(df["day_index"].unique())
        day_to_label = dict(zip(days, day_labels))
        shifts = [0.0, 24.0] if double_plotted else [0.0]

        for col, name, color, symbol in [
            ("onset_hours", "Onset", T().onset, "triangle-up"),
            ("offset_hours", "Offset", T().offset, "triangle-down"),
            ("acrophase_hours", "Acrophase", T().acrophase, "circle"),
        ]:
            if col not in dtbl.columns:
                continue
            xs, ys = [], []
            for sx in shifts:
                for d in days:
                    src = d if sx == 0.0 else d + 1
                    if src in dtbl.index and pd.notna(dtbl.loc[src, col]):
                        xs.append(float(dtbl.loc[src, col]) + sx)
                        ys.append(day_to_label[d])
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=name,
                        marker=dict(color=color, size=9, symbol=symbol, line=dict(width=1, color=T().background)),
                    )
                )

    if double_plotted:
        fig.add_vline(x=24.0, line_dash="dash", line_color=T().muted, line_width=1.5)

    max_x = 48 if double_plotted else 24
    fig.update_layout(
        title=dict(text=f"{'Double-Plotted' if double_plotted else 'Single-Plotted'} Actogram — {rep.channel_name}", y=0.97),
        xaxis=dict(
            title="Clock Time (hours)",
            tickmode="array",
            tickvals=list(range(0, max_x + 1, 4)),
            ticktext=[f"{h % 24:02d}:00" for h in range(0, max_x + 1, 4)],
            range=[0, max_x],
        ),
        yaxis=dict(title="Recording Day"),
        height=max(440, 32 * len(day_labels) + 160),
        margin=dict(t=85, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return apply_plotly_theme(fig)


def make_cosinor_harmonics_mpl(report: CircadianReport, channel: Optional[ChannelReport] = None) -> Figure:
    """Plot 24h folded profile with Single Cosinor, Multi-Harmonic composite, and harmonic amplitude spectrum."""
    rep = channel or report.primary
    df = report.binned.df
    cos1 = rep.cosinor_single
    cosh = rep.cosinor_harmonics

    fig = Figure(figsize=(11.5, 4.6), facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Left: Folded 24h profile + single cosinor + multi-harmonic fit
    t_mod = df["elapsed_hours"].to_numpy(dtype=float) % cosh.period_hours
    y_vals = df[rep.value_col].to_numpy(dtype=float)
    valid = np.isfinite(t_mod) & np.isfinite(y_vals)
    ax1.scatter(t_mod[valid], y_vals[valid], s=8, alpha=0.22, color=T().muted, label="Observed Epochs")

    # Hourly mean +/- SE
    bins = np.floor(t_mod[valid]).astype(int) % 24
    mean_h = [np.mean(y_vals[valid][bins == h]) if np.any(bins == h) else np.nan for h in range(24)]
    ax1.plot(np.arange(24) + 0.5, mean_h, "o-", color=T().foreground, markersize=4, linewidth=1.2, label="Hourly Mean")

    grid = np.linspace(0, cosh.period_hours, 400)
    y_sin1 = evaluate_cosinor_curve(cos1, grid)
    y_harm = evaluate_cosinor_curve(cosh, grid)
    ax1.plot(grid, y_sin1, "--", color=T().accent, linewidth=2.0, label=f"1-Harmonic Cosinor (R²={cos1.r2:.3f})")
    ax1.plot(grid, y_harm, "-", color=T().fit, linewidth=2.4, label=f"{cosh.n_harmonics}-Harmonic Composite (R²={cosh.r2:.3f})")
    ax1.axhline(cosh.mesor, color=T().mesor, linestyle=":", linewidth=1.4, label=f"MESOR = {cosh.mesor:.2f}")
    ax1.axvline(cos1.acrophase_hours, color=T().accent, alpha=0.35, linestyle="-.")

    ax1.set_xlim(0, cosh.period_hours)
    ax1.set_xticks(range(0, int(round(cosh.period_hours)) + 1, 4))
    ax1.set_xlabel("Phase / Clock Hour (h)")
    ax1.set_ylabel(f"{rep.channel_name} ({rep.unit_label})")
    ax1.set_title("Folded Waveform & Circadian Harmonics Fit", fontweight="bold")
    ax1.legend(fontsize=7.8, loc="best")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Harmonic amplitude spectrum & variance contribution
    ks = [h.harmonic_index for h in cosh.harmonics]
    amps = [h.amplitude for h in cosh.harmonics]
    prs = [h.percent_rhythm for h in cosh.harmonics]
    labels = [f"H{h.harmonic_index}\n({h.period_hours:.1f}h)" for h in cosh.harmonics]
    colors = [T().accent if h.p_value < 0.05 else T().muted for h in cosh.harmonics]

    bars = ax2.bar(ks, amps, color=colors, width=0.55, edgecolor=T().foreground)
    for bar, h in zip(bars, cosh.harmonics):
        sig = "*" if h.p_value < 0.05 else "ns"
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{h.amplitude:.1f}\n({h.percent_rhythm:.1f}%, {sig})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_xticks(ks)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel(f"Harmonic Amplitude ({rep.unit_label})")
    ax2.set_title("Circadian Harmonic Spectrum (Blue = p < 0.05)", fontweight="bold")
    ax2.set_ylim(0, max(amps) * 1.28 if amps else 1.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def make_periodogram_mpl(report: CircadianReport, channel: Optional[ChannelReport] = None) -> Figure:
    """Plot Lomb-Scargle, Sokolove-Bushell Chi-Square, and Cosinor R^2 periodograms."""
    rep = channel or report.primary
    pg = rep.periodogram

    fig = Figure(figsize=(11.5, 4.0), facecolor="white")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(pg.periods_hours, pg.lomb_scargle_power, color=T().accent, linewidth=1.8)
    ax1.axvline(pg.best_tau_lomb_scargle, color=T().fit, linestyle="--", label=f"Peak τ = {pg.best_tau_lomb_scargle:.2f} h")
    ax1.set_title("Lomb-Scargle Periodogram", fontweight="bold")
    ax1.set_xlabel("Period τ (hours)")
    ax1.set_ylabel("Normalized Power")
    ax1.legend(fontsize=8.5)

    ax2.plot(pg.periods_hours, pg.chi2_qp, color=T().chi2, linewidth=1.8, label="Qp Statistic")
    ax2.plot(pg.periods_hours, pg.chi2_crit_95, color=T().threshold, linestyle=":", linewidth=1.6, label="p = 0.05 Threshold")
    ax2.axvline(pg.best_tau_chi2, color=T().fit, linestyle="--", label=f"Peak τ = {pg.best_tau_chi2:.2f} h")
    ax2.set_title("Chi-Square (χ²) Periodogram", fontweight="bold")
    ax2.set_xlabel("Period τ (hours)")
    ax2.set_ylabel("Qp")
    ax2.legend(fontsize=8)

    ax3.plot(pg.periods_hours, pg.cosinor_r2, color=T().sweep, linewidth=1.8)
    ax3.axvline(pg.best_tau_cosinor, color=T().fit, linestyle="--", label=f"Peak τ = {pg.best_tau_cosinor:.2f} h")
    ax3.set_title("Cosinor R² Period Sweep", fontweight="bold")
    ax3.set_xlabel("Period τ (hours)")
    ax3.set_ylabel("R²")
    ax3.legend(fontsize=8.5)

    for ax in (ax1, ax2, ax3):
        if rep.rhythm_bands is not None:
            _shade_rhythm_bands(ax, rep.rhythm_bands, label=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def _shade_rhythm_bands(ax, rb, label: bool = True) -> None:
    """Shade ultradian / circadian / infradian period bands behind a periodogram axis."""
    for b in rb.bands:
        if b.scanned and b.high_hours > b.low_hours:
            ax.axvspan(b.low_hours, b.high_hours, color=T().band_colors[b.band], alpha=0.07, zorder=0, lw=0)
            if label:
                ax.text(
                    (b.low_hours + b.high_hours) / 2, 0.98, b.band, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color=T().band_colors[b.band], fontweight="bold",
                )


def make_rhythm_bands_mpl(report: CircadianReport, channel: Optional[ChannelReport] = None) -> Figure:
    """Left: R² spectrum (raw vs. after removing detected rhythms) with rhythm bands.
    Right: folded waveform with 2-harmonic fit and peak / dip markers for bimodality."""
    rep = channel or report.primary
    rb, bm = rep.rhythm_bands, rep.bimodality
    fig = Figure(figsize=(11.5, 4.2), facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    if rb is not None:
        _shade_rhythm_bands(ax1, rb)
        ax1.plot(rb.periods_hours, rb.raw_r2, color=T().foreground, lw=1.8, label="Raw spectrum")
        ax1.plot(rb.periods_hours, rb.residual_r2, color=T().fit, lw=1.3, ls="--", label="After removing detected rhythms")
        for b in rb.detected_bands:
            y_at = float(np.interp(b.period_hours, rb.periods_hours, rb.raw_r2))
            ax1.scatter([b.period_hours], [y_at], color=T().band_colors[b.band], s=70, zorder=5, edgecolor=T().background)
            ax1.annotate(
                f"#{b.extraction_order} {b.period_hours:.1f} h", (b.period_hours, y_at), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=T().band_colors[b.band],
            )
        top = float(max(rb.raw_r2.max(), rb.residual_r2.max(), 1e-3))
        th = rep.twelve_hour
        if th is not None:
            # Always mark both 12 h components, however weak
            ax1.axvline(th.harmonic_period_hours, color=T().harmonic, ls=":", lw=1.4,
                        label=f"τ/2 harmonic {th.harmonic_period_hours:.1f} h ({'sig.' if th.harmonic_significant else 'n.s.'})")
            ax1.axvline(th.rhythm_period_hours, color=T().highlight, ls="-.", lw=1.2,
                        label=f"12 h rhythm {th.rhythm_period_hours:.2f} h ({'sig.' if th.rhythm_significant else 'n.s.'})")
        ax1.set_ylim(0, top * 1.25)
        ax1.set_xlim(rb.period_min_hours, rb.period_max_hours)
        ax1.legend(fontsize=8, loc="center right")
    ax1.set_title("Rhythm Bands: Single-Cosinor R² Spectrum", fontweight="bold")
    ax1.set_xlabel("Period (hours)")
    ax1.set_ylabel("R² (variance explained)")

    if bm is not None:
        df = report.binned.df
        P = bm.period_hours
        bw = max(0.5, report.binned.epoch_minutes / 60.0)
        phase = df["elapsed_hours"].to_numpy(dtype=float) % P
        vals = df[rep.value_col].to_numpy(dtype=float)
        ok = np.isfinite(vals)
        key = np.floor(phase[ok] / bw) * bw + bw / 2
        prof = pd.Series(vals[ok]).groupby(key).mean()
        ax2.scatter(prof.index, prof.values, color=T().muted, s=16, label="Folded mean", zorder=2)
        ax2.plot(bm.waveform_hours, bm.waveform_values, color=T().foreground, lw=2.2, label="24 h + 12 h cosinor fit", zorder=3)
        s = bm.shape
        ax2.scatter([s.primary_hours], [s.primary_value], marker="^", s=90, color=T().fit, zorder=5, label=f"Main peak {bm.primary_peak_hhmm}")
        if s.bimodality_index > 0:
            ax2.scatter([s.secondary_hours], [s.secondary_value], marker="^", s=90, color=T().highlight, zorder=5, label=f"2nd peak {bm.secondary_peak_hhmm}")
            ax2.scatter([s.dip_hours], [s.dip_value], marker="v", s=70, color=T().accent, zorder=5, label=f"Dip {bm.dip_hhmm}")
        ax2.set_xlim(0, P)
        ax2.set_xticks(np.arange(0, P + 0.01, 4 if P >= 16 else 2))
        verdict = f"{bm.classification} (BI = {s.bimodality_index:.2f}, 12h p = {bm.h2_p_value:.2g})"
        ax2.set_title(f"Waveform Bimodality: {verdict}", fontweight="bold", fontsize=10.5)
        ax2.legend(fontsize=8, loc="upper left", framealpha=0.9)
    else:
        ax2.text(0.5, 0.5, "Not enough data for bimodality analysis", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Phase / Clock Hour (h)")
    ax2.set_ylabel(rep.unit_label)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def make_circular_polar_mpl(report: CircadianReport, channel: Optional[ChannelReport] = None) -> Figure:
    """24-hour clock polar plot of daily Acrophases, Onsets, Offsets, Mean Resultant Vector, and Pre/Post shift."""
    rep = channel or report.primary
    dtbl = rep.daily_table

    fig = Figure(figsize=(10.5, 4.6), facecolor="white")
    ax1 = fig.add_subplot(121, projection="polar")
    ax2 = fig.add_subplot(122, projection="polar")

    for ax in (ax1, ax2):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
        ax.set_xticklabels(["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"], fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color=T().muted)

    # Left polar: Acrophase vectors + Onsets + Offsets
    if "acrophase_hours" in dtbl.columns and dtbl["acrophase_hours"].notna().any():
        th_acr = hours_to_radians(dtbl["acrophase_hours"].dropna().to_numpy())
        ax1.scatter(th_acr, np.full_like(th_acr, 0.96), color=T().acrophase, s=38, alpha=0.8, label="Daily Acrophase")
    if "onset_hours" in dtbl.columns and dtbl["onset_hours"].notna().any():
        th_on = hours_to_radians(dtbl["onset_hours"].dropna().to_numpy())
        ax1.scatter(th_on, np.full_like(th_on, 0.84), color=T().onset, marker="^", s=38, alpha=0.8, label="Daily Onset")
    if "offset_hours" in dtbl.columns and dtbl["offset_hours"].notna().any():
        th_off = hours_to_radians(dtbl["offset_hours"].dropna().to_numpy())
        ax1.scatter(th_off, np.full_like(th_off, 0.72), color=T().offset, marker="v", s=38, alpha=0.8, label="Daily Offset")

    for circ, color, lw in [
        (rep.circular_acrophase, T().acrophase, 2.8),
        (rep.circular_onset, T().onset, 2.0),
        (rep.circular_offset, T().offset, 2.0),
    ]:
        if circ is not None:
            ax1.annotate(
                "",
                xy=(circ.mean_angle_rad, circ.mean_resultant_length),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw),
            )
    r_txt = f"R̄={rep.circular_acrophase.mean_resultant_length:.2f}, p={rep.circular_acrophase.rayleigh_p:.2e}" if rep.circular_acrophase else ""
    ax1.set_title(f"24h Circular Phase Vectors\n{r_txt}", fontweight="bold", pad=12, fontsize=10)
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.18), ncol=3, fontsize=7.5)

    # Right polar: Baseline (Pre-split) vs Post-split Acrophase Shift
    sr = rep.shift_report
    if sr is not None and sr.pre_post_acrophase is not None:
        pre_df = sr.daily_shift_df[sr.daily_shift_df["day_index"] < sr.split_day]
        post_df = sr.daily_shift_df[sr.daily_shift_df["day_index"] >= sr.split_day]
        th_pre = hours_to_radians(pre_df["acrophase_hours"].dropna().to_numpy())
        th_post = hours_to_radians(post_df["acrophase_hours"].dropna().to_numpy())
        ax2.scatter(th_pre, np.full_like(th_pre, 0.92), color=T().advance, s=40, alpha=0.85, label=f"Baseline (D<{sr.split_day})")
        ax2.scatter(th_post, np.full_like(th_post, 0.80), color=T().delay, s=40, alpha=0.85, label=f"Post-Split (D≥{sr.split_day})")

        ww = sr.pre_post_acrophase
        ax2.annotate(
            "",
            xy=(hours_to_radians(ww.mean1_hour), ww.r_bar1),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=T().advance, lw=2.8),
        )
        ax2.annotate(
            "",
            xy=(hours_to_radians(ww.mean2_hour), ww.r_bar2),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=T().delay, lw=2.8),
        )
        ax2.set_title(
            f"Acrophase Shift: {ww.phase_shift_hours:+.2f} h ({ww.shift_direction})\nWatson-Williams p={ww.p_value:.3g}",
            fontweight="bold",
            pad=12,
            fontsize=10,
        )
        ax2.legend(loc="lower left", bbox_to_anchor=(-0.05, -0.18), ncol=2, fontsize=8)
    else:
        ax2.set_title("Acrophase Shift (Need ≥ 4 Days)", fontweight="bold", pad=12, fontsize=10)

    fig.tight_layout()
    return fig


def make_acrophase_shift_timeline_mpl(report: CircadianReport, channel: Optional[ChannelReport] = None) -> Figure:
    """Plot daily Acrophase/Onset/Offset progression and day-to-day Acrophase Shift (Δφ)."""
    rep = channel or report.primary
    sr = rep.shift_report
    dtbl = sr.daily_shift_df if sr is not None else rep.daily_table

    fig = Figure(figsize=(11.5, 4.2), facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    days = dtbl["day_index"].to_numpy()
    if "acrophase_hours" in dtbl.columns:
        ax1.plot(days, dtbl["acrophase_hours"], "o-", color=T().acrophase, linewidth=2.0, label="Acrophase (h)")
    if "onset_hours" in dtbl.columns:
        ax1.plot(days, dtbl["onset_hours"], "^--", color=T().onset, linewidth=1.4, label="Onset (h)")
    if "offset_hours" in dtbl.columns:
        ax1.plot(days, dtbl["offset_hours"], "v--", color=T().offset, linewidth=1.4, label="Offset (h)")
    if sr is not None:
        ax1.axvline(sr.split_day - 0.5, color=T().delay, linestyle=":", linewidth=1.5, label=f"Split Day {sr.split_day}")
    ax1.set_ylim(0, 24)
    ax1.set_yticks(range(0, 25, 4))
    ax1.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax1.set_xlabel("Day Index")
    ax1.set_ylabel("Clock Time")
    ax1.set_title("Daily Phase Trajectory (Acrophase, Onset, Offset)", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    if "acrophase_shift_hours" in dtbl.columns:
        shifts = np.nan_to_num(dtbl["acrophase_shift_hours"].to_numpy(dtype=float), nan=0.0)
        colors = [T().delay if s > 0 else T().advance for s in shifts]
        ax2.bar(days, shifts, color=colors, width=0.6, alpha=0.85, label="Daily Δφ (h)")
        ax2.axhline(0, color=T().foreground, linewidth=0.9)
        if "cumulative_shift_hours" in dtbl.columns:
            ax2_t = ax2.twinx()
            ax2_t.plot(days, dtbl["cumulative_shift_hours"], "o-", color=T().foreground, linewidth=1.8, label="Cumulative Shift (h)")
            ax2_t.set_ylabel("Cumulative Shift (h)")
        ax2.set_xlabel("Day Index")
        ax2.set_ylabel("Day-to-Day Shift Δφ (h)")
        ax2.set_title("Acrophase Shift (+Delay / −Advance)", fontweight="bold")
        ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    return fig
