"""Multi-sheet Excel (.xlsx) report generator with formatted tables, live Excel formulas, and embedded circadian charts."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .pipeline import ChannelReport, CircadianReport
from .plotting import (
    figure_to_png_bytes,
    make_acrophase_shift_timeline_mpl,
    make_actogram_mpl,
    make_circular_polar_mpl,
    make_cosinor_harmonics_mpl,
    make_periodogram_mpl,
    make_rhythm_bands_mpl,
)
from .preprocessing import format_bin_size

EPOCHS_SHEET_MAX_ROWS = 20_000  # openpyxl styles cell-by-cell; 1 s bins over 2 weeks would be 1.2M rows

HEADER_FILL = PatternFill("solid", fgColor="1E293B")
SUBHEADER_FILL = PatternFill("solid", fgColor="E2E8F0")
ALT_ROW_FILL = PatternFill("solid", fgColor="F8FAFC")
ACCENT_FILL = PatternFill("solid", fgColor="EFF6FF")
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
SECTION_FONT = Font(name="Calibri", size=12, bold=True, color="0F172A")
BOLD_FONT = Font(name="Calibri", size=10, bold=True, color="0F172A")
BODY_FONT = Font(name="Calibri", size=10, color="1E293B")
THIN_BORDER = Border(
    left=Side(style="thin", color="CBD5E1"),
    right=Side(style="thin", color="CBD5E1"),
    top=Side(style="thin", color="CBD5E1"),
    bottom=Side(style="thin", color="CBD5E1"),
)


def _auto_fit_columns(ws: openpyxl.worksheet.worksheet.Worksheet, min_width: int = 12, max_width: int = 38, max_scan_rows: int = 200) -> None:
    for col_idx in range(1, ws.max_column + 1):
        col_letter = get_column_letter(col_idx)
        max_len = 0
        limit = min(ws.max_row, max_scan_rows)
        for r_idx in range(1, limit + 1):
            v = ws.cell(row=r_idx, column=col_idx).value
            val = "" if v is None else str(v)
            if val.startswith("="):
                val = "00000000"
            max_len = max(max_len, len(val))
        ws.column_dimensions[col_letter].width = max(min_width, min(max_width, max_len + 3))


def _write_table(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    start_row: int,
    headers: List[str],
    rows: List[List[Any]],
    number_formats: Optional[Dict[int, str]] = None,
) -> int:
    """Write a styled table starting at `start_row` and return the next free row."""
    fmt_map = number_formats or {}
    for c_idx, h in enumerate(headers, start=1):
        cell = ws.cell(row=start_row, column=c_idx, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER

    curr_row = start_row + 1
    for r_i, row_vals in enumerate(rows):
        fill = ALT_ROW_FILL if r_i % 2 == 1 else PatternFill(fill_type=None)
        for c_idx, val in enumerate(row_vals, start=1):
            if isinstance(val, (np.floating, float)) and not np.isfinite(val):
                val = None
            elif isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            cell = ws.cell(row=curr_row, column=c_idx, value=val)
            cell.font = BODY_FONT
            cell.border = THIN_BORDER
            if fill.fill_type:
                cell.fill = fill
            if c_idx in fmt_map and val is not None:
                cell.number_format = fmt_map[c_idx]
        curr_row += 1
    return curr_row


def _embed_chart(ws: openpyxl.worksheet.worksheet.Worksheet, png_bytes: bytes, anchor: str, scale: float = 0.62) -> None:
    img = XLImage(io.BytesIO(png_bytes))
    img.width = int(img.width * scale)
    img.height = int(img.height * scale)
    ws.add_image(img, anchor)


def _populate_summary_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, report: CircadianReport) -> None:
    ws.title = "Summary_Report"
    ws.views.sheetView[0].showGridLines = True

    ws["A1"] = "Circadian Rhythmometry, Actography & Circular Statistics Report"
    ws["A1"].font = Font(name="Calibri", size=15, bold=True, color="0F172A")
    ws["A2"] = f"Source: {report.source_name}  |  Span: {report.binned.start_time:%Y-%m-%d %H:%M} to {report.binned.end_time:%Y-%m-%d %H:%M} ({report.binned.n_days:.2f} days)  |  Epoch: {format_bin_size(report.binned.epoch_minutes)}"
    ws["A2"].font = Font(name="Calibri", size=10, italic=True, color="475569")

    row = 4
    channels = [c for c in (report.activity_report, report.light_report) if c is not None]
    headers = ["Category", "Metric", "Value", "Details / 95% CI / Significance"]

    rows: List[List[Any]] = []
    for rep in channels:
        ch = rep.channel_name
        pg = rep.periodogram
        c1 = rep.cosinor_single
        chm = rep.cosinor_harmonics
        np_m = rep.non_parametric
        ca = rep.circular_acrophase
        sr = rep.shift_report

        rows.append([ch, "Consensus Circadian Period (τ)", round(pg.consensus_tau_hours, 2), f"LS={pg.best_tau_lomb_scargle:.2f}h, χ²={pg.best_tau_chi2:.2f}h (p={pg.best_chi2_p_value:.2e}), Cos={pg.best_tau_cosinor:.2f}h"])
        rows.append([ch, "Period Used in Cosinor (τ)", round(rep.period_used_hours, 2), f"Mode: {report.config.period_mode}"])
        rows.append([ch, f"Single Cosinor MESOR ({rep.unit_label})", round(c1.mesor, 4), f"95% CI [{c1.mesor_ci_low:.3f}, {c1.mesor_ci_high:.3f}]"])
        rows.append([ch, f"Single Cosinor Amplitude ({rep.unit_label})", round(c1.amplitude, 4), f"95% CI [{c1.amplitude_ci_low:.3f}, {c1.amplitude_ci_high:.3f}]"])
        rows.append([ch, "Single Cosinor Acrophase (HH:MM)", c1.acrophase_hhmm, f"{c1.acrophase_deg_neg:.1f}° | Bathyphase={c1.bathyphase_hhmm} | Zero-Amp p={c1.p_value:.2e}"])
        rows.append([ch, "Single Cosinor R² / Percent Rhythm", f"{c1.r2:.4f} ({c1.percent_rhythm:.1f}%)", f"F={c1.f_stat:.2f}, p={c1.p_value:.2e}"])
        rows.append([ch, f"{chm.n_harmonics}-Harmonic Composite R² / PR", f"{chm.r2:.4f} ({chm.percent_rhythm:.1f}%)", f"Orthophase={chm.orthophase_hhmm}, Bathyphase={chm.composite_bathyphase_hhmm}, Amp={chm.composite_amplitude:.2f}"])

        rb = rep.rhythm_bands
        if rb is not None:
            dom = f"{rb.dominant_band} ({rb.dominant_period_hours:.1f} h)" if rb.dominant_band else "None significant"
            rows.append([ch, f"Dominant Rhythm ({rb.period_min_hours:g}–{rb.period_max_hours:g} h)", dom, "Strongest significant rhythm; see Rhythm_Bands sheet"])
            for b in rb.bands:
                val = f"{b.period_hours:.1f} h ({b.percent_variance:.1f}% var)" if b.detected else "—"
                rows.append([ch, f"{b.band} Rhythm ({b.range_label} h)", val, b.classification + (f"; p(Šidák)={b.p_adjusted:.2e}" if b.detected else "")])
        th = rep.twelve_hour
        if th is not None:
            rows.append([ch, "12 h Rhythm (own period)", f"{th.rhythm_period_hours:.2f} h, amp {th.rhythm_amplitude:.3f}", f"Peaks {th.rhythm_peaks_hhmm}, {th.rhythm_percent_variance:.2f}% var, p(Šidák)={th.rhythm_p_adjusted:.2e} ({'sig.' if th.rhythm_significant else 'n.s.'})"])
            rows.append([ch, "12 h Harmonic of Circadian (τ/2)", f"{th.harmonic_period_hours:.2f} h, amp {th.harmonic_amplitude:.3f}", f"Peaks {th.harmonic_peaks_hhmm}, {th.harmonic_percent_variance:.2f}% var, p={th.harmonic_p_value:.2e} ({'sig.' if th.harmonic_significant else 'n.s.'}), A2/A1={th.harmonic_to_fundamental_ratio:.2f}; {'distinct' if th.distinct_from_harmonic else 'indistinguishable'}"])
        bm = rep.bimodality
        if bm is not None:
            peaks =f"{bm.primary_peak_hhmm} & {bm.secondary_peak_hhmm}" if bm.is_bimodal else bm.primary_peak_hhmm
            rows.append([ch, "Waveform Bimodality", bm.classification, f"BI={bm.shape.bimodality_index:.2f}, peak(s) {peaks}, A2/A1={bm.h2_to_h1_ratio:.2f} (p={bm.h2_p_value:.2e}), bimodal days {bm.days_bimodal}/{bm.days_tested}"])

        if np_m is not None:
            rows.append([ch, "Interdaily Stability (IS) / Intradaily Variability (IV)", f"IS={np_m.interdaily_stability_is:.3f} | IV={np_m.intradaily_variability_iv:.3f}", f"M10={np_m.m10_value:.2f} (@{np_m.m10_onset_hhmm}), L5={np_m.l5_value:.2f} (@{np_m.l5_onset_hhmm}), RA={np_m.relative_amplitude_ra:.3f}"])

        if ca is not None:
            rows.append([ch, "Circular Mean Acrophase (HH:MM)", ca.mean_hhmm, f"R̄={ca.mean_resultant_length:.3f}, CSD={ca.circular_std_hours:.2f}h, κ={ca.kappa:.2f}, Rayleigh Z={ca.rayleigh_z:.2f} (p={ca.rayleigh_p:.2e})"])

        if rep.circular_onset is not None and rep.circular_offset is not None:
            rows.append([ch, "Circular Mean Onset / Offset", f"{rep.circular_onset.mean_hhmm} / {rep.circular_offset.mean_hhmm}", f"Onset R̄={rep.circular_onset.mean_resultant_length:.3f} | Offset R̄={rep.circular_offset.mean_resultant_length:.3f}"])

        if sr is not None:
            ww = sr.pre_post_acrophase
            ww_txt = f"Pre={ww.mean1_hhmm} vs Post={ww.mean2_hhmm} => Δφ={ww.phase_shift_hours:+.2f}h ({ww.shift_direction}), WW p={ww.p_value:.3g}" if ww else "Insufficient split days"
            rows.append([ch, "Acrophase Shift & Drift", f"Drift={sr.drift_slope_hours_per_day:+.3f} h/day (Cum={sr.cumulative_shift_hours:+.2f}h)", ww_txt])
            if sr.activity_light_phase_angle_hours is not None:
                rows.append([ch, "Phase Angle (Activity − Light)", f"{sr.activity_light_phase_angle_hours:+.2f} h", "Positive = Activity lags Light"])

    next_row = _write_table(ws, row, headers, rows)
    _auto_fit_columns(ws, min_width=14, max_width=52)

    # Embed primary double-plotted actogram on Summary sheet
    acto_png = figure_to_png_bytes(make_actogram_mpl(report, double_plotted=True), dpi=140)
    _embed_chart(ws, acto_png, f"A{next_row + 2}", scale=0.58)


def _populate_harmonics_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, report: CircadianReport) -> None:
    ws.title = "Cosinor_Harmonics"
    ws["A1"] = "Cosinor Rhythmometry & Circadian Harmonics Decomposition"
    ws["A1"].font = SECTION_FONT

    curr_row = 3
    for rep in [c for c in (report.activity_report, report.light_report) if c is not None]:
        ws.cell(row=curr_row, column=1, value=f"{rep.channel_name} Channel ({rep.unit_label}) — τ = {rep.period_used_hours:.2f} h, MESOR = {rep.cosinor_harmonics.mesor:.4f}").font = BOLD_FONT
        curr_row += 1
        headers = [
            "Harmonic (k)",
            "Period (h)",
            "Beta (β)",
            "Gamma (γ)",
            "Amplitude (A)",
            "Amp SE",
            "Amp 95% CI Low",
            "Amp 95% CI High",
            "Acrophase (HH:MM)",
            "Acrophase (° neg)",
            "Acrophase SE (h)",
            "Ratio Ak/A1",
            "Partial PR (%)",
            "F-stat",
            "p-value",
        ]
        rows = []
        for h in rep.cosinor_harmonics.harmonics:
            rows.append(
                [
                    f"H{h.harmonic_index}",
                    h.period_hours,
                    h.beta,
                    h.gamma,
                    h.amplitude,
                    h.amplitude_se,
                    h.amplitude_ci_low,
                    h.amplitude_ci_high,
                    h.acrophase_hhmm,
                    h.acrophase_deg_neg,
                    h.acrophase_se_hours,
                    h.amplitude_ratio_to_h1,
                    h.percent_rhythm,
                    h.f_stat,
                    h.p_value,
                ]
            )
        fmts = {2: "0.00", 3: "0.0000", 4: "0.0000", 5: "0.0000", 6: "0.0000", 7: "0.0000", 8: "0.0000", 10: "0.0", 11: "0.00", 12: "0.000", 13: "0.00", 14: "0.00", 15: "0.00E+00"}
        curr_row = _write_table(ws, curr_row, headers, rows, fmts) + 2

    _auto_fit_columns(ws, min_width=11, max_width=24)
    harm_png = figure_to_png_bytes(make_cosinor_harmonics_mpl(report), dpi=140)
    _embed_chart(ws, harm_png, f"A{curr_row + 1}", scale=0.58)


def _populate_daily_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, report: CircadianReport) -> None:
    ws.title = "Daily_Rhythms_Onsets"
    ws["A1"] = f"Daily Rhythmometry, Onsets/Offsets & Acrophase Shifts ({report.primary.channel_name})"
    ws["A1"].font = SECTION_FONT

    rep = report.primary
    dtbl = rep.daily_table.copy()
    cols_wanted = [
        ("day_index", "Day"),
        ("date", "Date"),
        ("mean_value", f"Mean {rep.channel_name}"),
        ("max_value", f"Max {rep.channel_name}"),
        ("mesor", "Daily MESOR"),
        ("amplitude", "Daily Amplitude"),
        ("acrophase_hours", "Acrophase (h)"),
        ("acrophase_hhmm", "Acrophase (HH:MM)"),
        ("acrophase_deg_neg", "Acrophase (°)"),
        ("r2", "Cosinor R²"),
        ("p_value", "Cosinor p"),
        ("onset_hours", "Onset (h)"),
        ("onset_hhmm", "Onset (HH:MM)"),
        ("offset_hours", "Offset (h)"),
        ("offset_hhmm", "Offset (HH:MM)"),
        ("active_duration_hours", "Active α (h)"),
        ("midpoint_hhmm", "Midpoint (HH:MM)"),
        ("acrophase_shift_hours", "Daily Δφ (h)"),
        ("acrophase_shift_deg", "Daily Δφ (°)"),
        ("cumulative_shift_hours", "Cumulative Δφ (h)"),
        ("phase_angle_act_minus_light_h", "Phase Angle ψ (h)"),
        ("bimodal", "Bimodal Day"),
        ("bimodality_index", "Bimodality Index"),
        ("peak1_hhmm", "Main Peak (HH:MM)"),
        ("peak2_hhmm", "2nd Peak (HH:MM)"),
    ]
    present = [(k, lbl) for k, lbl in cols_wanted if k in dtbl.columns]
    headers = [lbl for _, lbl in present]
    cols_only = [k for k, _ in present]
    rows = dtbl[cols_only].to_numpy(dtype=object).tolist()

    fmts = {}
    for idx, (k, _) in enumerate(present, start=1):
        if k in ("mean_value", "max_value", "mesor", "amplitude", "acrophase_hours", "onset_hours", "offset_hours", "active_duration_hours", "acrophase_shift_hours", "cumulative_shift_hours", "phase_angle_act_minus_light_h"):
            fmts[idx] = "0.00"
        elif k in ("acrophase_deg_neg", "acrophase_shift_deg"):
            fmts[idx] = "0.0"
        elif k == "r2":
            fmts[idx] = "0.000"
        elif k == "p_value":
            fmts[idx] = "0.00E+00"

    next_row = _write_table(ws, 3, headers, rows, fmts)

    # Add Excel summary AVERAGE row for numeric columns
    if len(rows) > 0:
        avg_row = next_row
        ws.cell(row=avg_row, column=1, value="Mean").font = BOLD_FONT
        for idx, (k, _) in enumerate(present, start=1):
            if k in ("mean_value", "max_value", "mesor", "amplitude", "r2", "active_duration_hours", "acrophase_shift_hours", "phase_angle_act_minus_light_h"):
                col_l = get_column_letter(idx)
                c = ws.cell(row=avg_row, column=idx, value=f"=AVERAGE({col_l}4:{col_l}{next_row - 1})")
                c.font = BOLD_FONT
                c.number_format = fmts.get(idx, "0.00")
                c.border = THIN_BORDER

    _auto_fit_columns(ws, min_width=11, max_width=22)
    shift_png = figure_to_png_bytes(make_acrophase_shift_timeline_mpl(report), dpi=140)
    _embed_chart(ws, shift_png, f"A{next_row + 3}", scale=0.58)


def _populate_circular_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, report: CircadianReport) -> None:
    ws.title = "Circular_Statistics"
    ws["A1"] = "Circular Statistics Recorder (24h Clock Angles, Rayleigh Uniformity & Acrophase Shift Tests)"
    ws["A1"].font = SECTION_FONT

    headers = [
        "Channel / Variable",
        "N",
        "Circular Mean (HH:MM)",
        "Mean Hour (h)",
        "Mean Angle (°)",
        "Resultant Length R̄",
        "Circular Variance",
        "Circular SD (h)",
        "Circular SD (°)",
        "Concentration κ",
        "Rayleigh Z",
        "Rayleigh p-value",
    ]
    rows: List[List[Any]] = []
    for rep in [c for c in (report.activity_report, report.light_report) if c is not None]:
        for cs in (rep.circular_acrophase, rep.circular_onset, rep.circular_offset):
            if cs is None:
                continue
            rows.append(
                [
                    cs.label,
                    cs.n,
                    cs.mean_hhmm,
                    cs.mean_hour,
                    cs.mean_angle_deg,
                    cs.mean_resultant_length,
                    cs.circular_variance,
                    cs.circular_std_hours,
                    cs.circular_std_deg,
                    cs.kappa,
                    cs.rayleigh_z,
                    cs.rayleigh_p,
                ]
            )
    fmts = {4: "0.00", 5: "0.0", 6: "0.0000", 7: "0.0000", 8: "0.00", 9: "0.0", 10: "0.00", 11: "0.000", 12: "0.00E+00"}
    next_row = _write_table(ws, 3, headers, rows, fmts) + 1

    # Pre vs Post Watson-Williams Acrophase Shift Table
    ww_rows: List[List[Any]] = []
    for rep in [c for c in (report.activity_report, report.light_report) if c is not None]:
        sr = rep.shift_report
        if sr is None:
            continue
        for lbl, ww in [("Acrophase", sr.pre_post_acrophase), ("Onset", sr.pre_post_onset), ("Offset", sr.pre_post_offset)]:
            if ww is None:
                continue
            ww_rows.append(
                [
                    f"{rep.channel_name} {lbl}",
                    f"Split Day {sr.split_day}",
                    ww.n1,
                    ww.mean1_hhmm,
                    ww.n2,
                    ww.mean2_hhmm,
                    ww.phase_shift_hours,
                    ww.phase_shift_deg,
                    ww.shift_direction,
                    ww.f_stat,
                    ww.p_value,
                ]
            )
    if ww_rows:
        ws.cell(row=next_row, column=1, value="Pre- vs. Post-Split Circular Shift Tests (Watson-Williams)").font = BOLD_FONT
        next_row += 1
        ww_headers = ["Variable", "Split", "N Pre", "Pre Mean", "N Post", "Post Mean", "Shift (h)", "Shift (°)", "Direction", "WW F-stat", "p-value"]
        next_row = _write_table(ws, next_row, ww_headers, ww_rows, {7: "+0.00;-0.00;0.00", 8: "+0.0;-0.0;0.0", 10: "0.000", 11: "0.00E+00"}) + 1

    # Live Excel Circular Vector Table for primary channel daily acrophases (using Excel SIN/COS/ATAN2 formulas)
    rep = report.primary
    if "acrophase_hours" in rep.daily_table.columns and rep.daily_table["acrophase_hours"].notna().any():
        ws.cell(row=next_row, column=1, value="Daily Circular Vector Decomposition (with Live Excel Formulas)").font = BOLD_FONT
        next_row += 1
        vec_headers = ["Day", "Acrophase (h)", "Angle θ (rad)", "cos(θ)", "sin(θ)"]
        for c_i, h in enumerate(vec_headers, 1):
            cell = ws.cell(row=next_row, column=c_i, value=h)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.border = THIN_BORDER
        start_vec = next_row + 1
        r_ptr = start_vec
        for _, dr in rep.daily_table.dropna(subset=["acrophase_hours"]).iterrows():
            ws.cell(row=r_ptr, column=1, value=int(dr["day_index"])).border = THIN_BORDER
            c_hr = ws.cell(row=r_ptr, column=2, value=float(dr["acrophase_hours"]))
            c_hr.number_format = "0.000"
            c_hr.border = THIN_BORDER
            c_rad = ws.cell(row=r_ptr, column=3, value=f"=2*PI()*B{r_ptr}/24")
            c_rad.number_format = "0.0000"
            c_rad.border = THIN_BORDER
            c_cos = ws.cell(row=r_ptr, column=4, value=f"=COS(C{r_ptr})")
            c_cos.number_format = "0.0000"
            c_cos.border = THIN_BORDER
            c_sin = ws.cell(row=r_ptr, column=5, value=f"=SIN(C{r_ptr})")
            c_sin.number_format = "0.0000"
            c_sin.border = THIN_BORDER
            r_ptr += 1
        end_vec = r_ptr - 1
        if end_vec >= start_vec:
            ws.cell(row=r_ptr, column=1, value="Formula Mean:").font = BOLD_FONT
            ws.cell(row=r_ptr, column=3, value=f"=MOD(ATAN2(AVERAGE(D{start_vec}:D{end_vec}),AVERAGE(E{start_vec}:E{end_vec})),2*PI())").number_format = "0.0000"
            ws.cell(row=r_ptr, column=4, value=f"=AVERAGE(D{start_vec}:D{end_vec})").number_format = "0.0000"
            ws.cell(row=r_ptr, column=5, value=f"=AVERAGE(E{start_vec}:E{end_vec})").number_format = "0.0000"
            ws.cell(row=r_ptr + 1, column=1, value="Formula R̄ & Mean (h):").font = BOLD_FONT
            ws.cell(row=r_ptr + 1, column=2, value=f"=24*C{r_ptr}/(2*PI())").number_format = "0.000"
            ws.cell(row=r_ptr + 1, column=4, value=f"=SQRT(D{r_ptr}^2+E{r_ptr}^2)").number_format = "0.0000"
            next_row = r_ptr + 3

    _auto_fit_columns(ws, min_width=12, max_width=26)
    polar_png = figure_to_png_bytes(make_circular_polar_mpl(report), dpi=140)
    _embed_chart(ws, polar_png, f"A{next_row + 1}", scale=0.60)


def _populate_periodogram_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, report: CircadianReport) -> None:
    ws.title = "Periodogram"
    ws["A1"] = f"Periodicity Analysis ({report.primary.channel_name}): Lomb-Scargle, Chi-Square & Cosinor Sweep"
    ws["A1"].font = SECTION_FONT

    pg = report.primary.periodogram
    headers = ["Period τ (h)", "Lomb-Scargle Power", "Chi-Square Qp", "Chi-Square p=0.05 Crit", "Cosinor R²"]
    rows = [
        [float(p), float(ls), float(qp), float(cr), float(r2)]
        for p, ls, qp, cr, r2 in zip(
            pg.periods_hours, pg.lomb_scargle_power, pg.chi2_qp, pg.chi2_crit_95, pg.cosinor_r2
        )
    ]
    _write_table(ws, 3, headers, rows, {1: "0.00", 2: "0.0000", 3: "0.00", 4: "0.00", 5: "0.0000"})
    _auto_fit_columns(ws, min_width=14, max_width=22)

    pg_png = figure_to_png_bytes(make_periodogram_mpl(report), dpi=140)
    _embed_chart(ws, pg_png, "G3", scale=0.58)


def _populate_rhythm_bands_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, report: CircadianReport) -> None:
    ws.title = "Rhythm_Bands"
    rep = report.primary
    rb, bm = rep.rhythm_bands, rep.bimodality
    ws["A1"] = f"Ultradian / Circadian / Infradian Rhythms & Bimodality ({rep.channel_name})"
    ws["A1"].font = SECTION_FONT
    row = 3
    if rb is not None:
        ws.cell(
            row=2, column=1,
            value=(
                f"Scan {rb.period_min_hours:g}–{rb.period_max_hours:g} h; circadian band {rb.circadian_min_hours:g}–{rb.circadian_max_hours:g} h. "
                f"Rhythms extracted strongest-first; each removed with its harmonics before the next is tested. "
                f"Significant = Šidák-adjusted p < {rb.alpha:g} and ≥ {rb.min_percent_variance:g}% of total variance."
            ),
        ).font = Font(name="Calibri", size=9, italic=True, color="475569")
        headers = [
            "Band", "Range (h)", "Status", "Period (h)", "% Variance", "Amplitude", "Acrophase",
            "p", "p (Šidák)", "Indep. Periods", "Order", "Raw Peak (h)", "Raw Peak R²", "Note",
        ]
        rows = [
            [
                b.band, b.range_label, b.classification, b.period_hours, b.percent_variance,
                b.amplitude, b.acrophase_hhmm if b.detected else "", b.p_value, b.p_adjusted,
                b.n_independent_periods, b.extraction_order, b.raw_peak_period_hours, b.raw_peak_r2, b.note,
            ]
            for b in rb.bands
        ]
        row = _write_table(ws, row, headers, rows, {4: "0.0", 5: "0.0", 6: "0.000", 8: "0.00E+00", 9: "0.00E+00", 10: "0.0", 12: "0.0", 13: "0.000"})
        row += 1

    th = rep.twelve_hour
    if th is not None:
        ws.cell(row=row, column=1, value="12 h Rhythm vs. 12 h Harmonic of the Circadian Rhythm (always reported)").font = BOLD_FONT
        row += 1
        yn = lambda ok: "Yes" if ok else "No"  # noqa: E731
        rows = [
            [
                f"12 h rhythm (own period, {th.search_window_hours[0]:g}–{th.search_window_hours[1]:g} h search)",
                th.rhythm_period_hours, th.rhythm_amplitude, th.rhythm_peaks_hhmm, th.rhythm_percent_variance,
                th.rhythm_p_value, th.rhythm_p_adjusted, yn(th.rhythm_significant),
            ],
            [
                f"τ/2 harmonic of {th.circadian_period_hours:.2f} h circadian rhythm",
                th.harmonic_period_hours, th.harmonic_amplitude, th.harmonic_peaks_hhmm, th.harmonic_percent_variance,
                th.harmonic_p_value, None, yn(th.harmonic_significant),
            ],
        ]
        row = _write_table(
            ws, row,
            ["Component", "Period (h)", "Amplitude", "Peaks (HH:MM)", "% Variance", "p", "p (Šidák)", "Significant"],
            rows, {2: "0.00", 3: "0.000", 5: "0.00", 6: "0.00E+00", 7: "0.00E+00"},
        )
        ws.cell(
            row=row, column=1,
            value=(
                f"Period difference {th.period_difference_hours:+.2f} h vs resolution ±{th.resolution_hours:.2f} h → "
                f"{'distinct' if th.distinct_from_harmonic else 'indistinguishable'}. {th.interpretation}"
            ),
        ).font = Font(name="Calibri", size=9, italic=True, color="475569")
        row += 2

    if bm is not None:
        s = bm.shape
        ws.cell(row=row, column=1, value="Waveform Bimodality (2-harmonic cosinor)").font = BOLD_FONT
        row += 1
        rows = [
            ["Classification", bm.classification, f"Bimodal = 2nd peak prominence ≥ {bm.min_prominence_fraction:.0%} of range AND 12 h component p < {bm.alpha:g}"],
            ["Bimodality Index", round(s.bimodality_index, 3), "2nd-peak prominence ÷ waveform range (0 = single peak)"],
            ["Main Peak", bm.primary_peak_hhmm, round(s.primary_value, 3)],
            ["Second Peak", bm.secondary_peak_hhmm if s.bimodality_index > 0 else "none", round(s.secondary_value, 3) if s.bimodality_index > 0 else None],
            ["Dip Between Peaks", bm.dip_hhmm if s.bimodality_index > 0 else "none", round(s.dip_value, 3) if s.bimodality_index > 0 else None],
            ["12 h ÷ 24 h Amplitude (A2/A1)", round(bm.h2_to_h1_ratio, 3), f"12 h component p = {bm.h2_p_value:.2e}"],
            ["2-Harmonic Fit R²", round(bm.r2, 4), f"Period {bm.period_hours:.2f} h"],
            ["Bimodal Days", f"{bm.days_bimodal} / {bm.days_tested}", "Same test on each day's 24 h + 12 h fit"],
        ]
        row = _write_table(ws, row, ["Metric", "Value", "Details"], rows)
        row += 1

    _auto_fit_columns(ws, min_width=11, max_width=48)
    png = figure_to_png_bytes(make_rhythm_bands_mpl(report), dpi=140)
    _embed_chart(ws, png, f"A{row + 1}", scale=0.58)


def _populate_epochs_sheet(ws: openpyxl.worksheet.worksheet.Worksheet, report: CircadianReport) -> None:
    ws.title = "Binned_Epochs"
    df = report.binned.df.copy()
    export_cols = [
        "timestamp",
        "day_index",
        "elapsed_hours",
        "clock_hour",
        "activity",
        "lux",
        "log_lux",
        "cosinor_single_fit",
        "cosinor_harmonics_fit",
        "cosinor_residual",
    ]
    present = [c for c in export_cols if c in df.columns]
    headers = [c.replace("_", " ").title() for c in present]
    df = df[present]

    note = None
    if len(df) > EPOCHS_SHEET_MAX_ROWS:
        factor = int(np.ceil(len(df) / EPOCHS_SHEET_MAX_ROWS))
        groups = np.arange(len(df)) // factor
        agg = df.drop(columns=["timestamp", "day_index"], errors="ignore").groupby(groups).mean()
        agg.insert(0, "timestamp", df["timestamp"].iloc[::factor].to_numpy()[: len(agg)])
        if "day_index" in present:
            agg.insert(1, "day_index", df["day_index"].iloc[::factor].to_numpy()[: len(agg)])
        df = agg[present].reset_index(drop=True)
        note = (
            f"{len(report.binned.df):,} epochs at {format_bin_size(report.binned.epoch_minutes)} exceed this sheet's "
            f"{EPOCHS_SHEET_MAX_ROWS:,}-row limit; rows here are means of {factor} consecutive epochs "
            f"({format_bin_size(report.binned.epoch_minutes * factor)}). Use the app's Binned Epochs CSV for full resolution."
        )

    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    rows: List[List[Any]] = out.to_numpy(dtype=object).tolist()

    fmts = {i: "0.000" for i, c in enumerate(present, start=1) if c not in ("timestamp", "day_index")}
    _write_table(ws, 1, headers, rows, fmts)
    if note:
        ws.cell(row=1, column=len(present) + 2, value=note).font = BOLD_FONT
    _auto_fit_columns(ws, min_width=12, max_width=24)


def export_report_to_excel(
    report: CircadianReport,
    output_path: Optional[Union[str, Path]] = None,
) -> bytes:
    """Build the multi-sheet Excel workbook, optionally save to `output_path`, and return raw .xlsx bytes."""
    wb = openpyxl.Workbook()
    ws_summary = wb.active
    _populate_summary_sheet(ws_summary, report)
    _populate_harmonics_sheet(wb.create_sheet(), report)
    _populate_daily_sheet(wb.create_sheet(), report)
    _populate_circular_sheet(wb.create_sheet(), report)
    _populate_periodogram_sheet(wb.create_sheet(), report)
    _populate_rhythm_bands_sheet(wb.create_sheet(), report)
    _populate_epochs_sheet(wb.create_sheet(), report)

    buf = io.BytesIO()
    wb.save(buf)
    xlsx_bytes = buf.getvalue()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(xlsx_bytes)

    return xlsx_bytes
