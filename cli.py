"""Headless CLI runner for Phyphox / CSV / XLSX circadian rhythmometry, actograms, circular stats, and Excel export."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from circadian_app.excel_export import export_report_to_excel
from circadian_app.io_loader import (
    generate_synthetic_circadian_data,
    load_tabular_sensor_file,
    merge_datasets,
)
from circadian_app.pipeline import CircadianAnalysisConfig, run_full_analysis
from circadian_app.plotting import (
    figure_to_png_bytes,
    make_acrophase_shift_timeline_mpl,
    make_actogram_mpl,
    make_circular_polar_mpl,
    make_cosinor_harmonics_mpl,
    make_periodogram_mpl,
    make_rhythm_bands_mpl,
)
from circadian_app.theme import AVAILABLE_LABEL_FONTS, BACKGROUND_PRESETS, PALETTES, build_theme, set_plot_theme


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Circadian rhythmometry, cosinor/harmonics, actograms, circular stats & Excel exporter for Phyphox/phone data."
    )
    parser.add_argument("inputs", nargs="*", help="Input Phyphox .zip, .csv, .tsv, or .xlsx files. Omit to run --demo.")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic 14-day circadian activity + light dataset.")
    parser.add_argument("--output-dir", default="output", help="Directory to write Excel workbook and PNG charts.")
    parser.add_argument("--excel-name", default="circadian_report.xlsx", help="Output Excel filename.")
    parser.add_argument(
        "--epoch-min",
        type=float,
        default=10,
        help="Epoch bin duration in minutes, 1 s to 6 h (e.g. 0.5 = 30 s, 360 = 6 h; default: 10).",
    )
    parser.add_argument("--harmonics", type=int, default=3, help="Number of circadian harmonics (default: 3).")
    parser.add_argument(
        "--period-mode",
        default="auto_consensus",
        choices=["auto_consensus", "fixed_24h", "lomb_scargle", "chi2", "cosinor", "custom"],
    )
    parser.add_argument("--custom-tau", type=float, default=24.0, help="Custom period tau in hours when --period-mode=custom.")
    parser.add_argument(
        "--onset-method",
        default="template",
        choices=["template", "threshold", "cosinor"],
        help="Daily onset/offset detection algorithm.",
    )
    parser.add_argument("--split-day", type=int, default=None, help="Day index splitting baseline (<) vs post-shift (>=).")
    parser.add_argument(
        "--background",
        default="White",
        help=f"Chart background: one of {', '.join(BACKGROUND_PRESETS)} or a hex colour like '#202020' (default: White).",
    )
    parser.add_argument("--palette", default="Default", choices=list(PALETTES), help="Plot colour palette (default: Default).")
    parser.add_argument("--axis-font", default=None, help=f"Axis label font, e.g. {', '.join(AVAILABLE_LABEL_FONTS[:4])}.")
    parser.add_argument("--axis-font-size", type=float, default=None, help="Axis label font size in points.")
    parser.add_argument("--axis-bold", action="store_true", help="Bold axis labels.")
    parser.add_argument("--x-label-color", default=None, help="X-axis label colour as #RRGGBB.")
    parser.add_argument("--y-label-color", default=None, help="Y-axis label colour as #RRGGBB.")
    parser.add_argument(
        "--axis-label", action="append", default=[], metavar="ORIGINAL=NEW",
        help="Rename an axis label, e.g. --axis-label 'Period τ (hours)=Period (h)'. Repeatable.",
    )
    parser.add_argument("--period-min",type=float, default=9.0, help="Shortest period scanned, hours (default: 9).")
    parser.add_argument("--period-max", type=float, default=36.0, help="Longest period scanned, hours (default: 36).")
    parser.add_argument("--circadian-min", type=float, default=20.0, help="Ultradian/circadian boundary, hours (default: 20).")
    parser.add_argument("--circadian-max", type=float, default=28.0, help="Circadian/infradian boundary, hours (default: 28).")
    parser.add_argument("--start-datetime", default=None, help="Optional ISO start datetime for relative Time(s) files.")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.fromisoformat(args.start_datetime) if args.start_datetime else None
    bg = BACKGROUND_PRESETS.get(args.background, args.background)
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", bg):
        parser.error(f"--background must be one of {', '.join(BACKGROUND_PRESETS)} or a #RRGGBB colour, got {args.background!r}")
    for name, value in (("--x-label-color", args.x_label_color), ("--y-label-color", args.y_label_color)):
        if value is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
            parser.error(f"{name} must be a #RRGGBB colour, got {value!r}")
    label_pairs = []
    for item in args.axis_label:
        if "=" not in item:
            parser.error(f"--axis-label must look like ORIGINAL=NEW, got {item!r}")
        original, new = item.rsplit("=", 1)
        label_pairs.append((original, new))
    set_plot_theme(
        build_theme(
            bg,
            args.palette,
            label_style={
                "label_font_family": args.axis_font,
                "label_font_size": args.axis_font_size,
                "label_bold": args.axis_bold,
                "x_label_color": args.x_label_color,
                "y_label_color": args.y_label_color,
                "label_text": tuple(label_pairs),
            },
        )
    )

    if args.demo or not args.inputs:
        dataset = generate_synthetic_circadian_data(days=14, epoch_minutes=args.epoch_min, shift_day=8, shift_hours=3.5)
    else:
        loaded = [load_tabular_sensor_file(p, start_datetime=start_dt) for p in args.inputs]
        dataset = merge_datasets(loaded)

    cfg = CircadianAnalysisConfig(
        epoch_minutes=args.epoch_min,
        n_harmonics=args.harmonics,
        period_mode=args.period_mode,
        custom_period_hours=args.custom_tau,
        onset_method=args.onset_method,
        split_day=args.split_day,
        period_min_hours=args.period_min,
        period_max_hours=args.period_max,
        circadian_min_hours=args.circadian_min,
        circadian_max_hours=args.circadian_max,
    )
    report = run_full_analysis(dataset, cfg)

    excel_path = out_dir / args.excel_name
    export_report_to_excel(report, output_path=excel_path)

    # Save standalone PNG charts as well
    charts = {
        "actogram_double.png": make_actogram_mpl(report, double_plotted=True),
        "cosinor_harmonics.png": make_cosinor_harmonics_mpl(report),
        "periodogram.png": make_periodogram_mpl(report),
        "rhythm_bands_bimodality.png": make_rhythm_bands_mpl(report),
        "circular_polar.png": make_circular_polar_mpl(report),
        "acrophase_shift.png": make_acrophase_shift_timeline_mpl(report),
    }
    for fname, fig in charts.items():
        (out_dir / fname).write_bytes(figure_to_png_bytes(fig, dpi=160))

    rep = report.primary
    print(f"=== Circadian Analysis Complete ({report.source_name}) ===")
    print(f"Primary Channel      : {rep.channel_name} ({rep.unit_label})")
    print(f"Consensus Period tau : {rep.periodogram.consensus_tau_hours:.2f} h (used: {rep.period_used_hours:.2f} h)")
    print(
        f"Single Cosinor       : MESOR={rep.cosinor_single.mesor:.2f}, Amp={rep.cosinor_single.amplitude:.2f}, "
        f"Acrophase={rep.cosinor_single.acrophase_hhmm} ({rep.cosinor_single.acrophase_deg_neg:.1f} deg), "
        f"R2={rep.cosinor_single.r2:.3f}, p={rep.cosinor_single.p_value:.2e}"
    )
    print(
        f"{rep.cosinor_harmonics.n_harmonics}-Harmonic Cosinor   : R2={rep.cosinor_harmonics.r2:.3f}, "
        f"Orthophase={rep.cosinor_harmonics.orthophase_hhmm}, Bathyphase={rep.cosinor_harmonics.composite_bathyphase_hhmm}"
    )
    if rep.circular_acrophase:
        print(
            f"Circular Acrophase   : Mean={rep.circular_acrophase.mean_hhmm}, R_bar={rep.circular_acrophase.mean_resultant_length:.3f}, "
            f"CSD={rep.circular_acrophase.circular_std_hours:.2f} h, Rayleigh p={rep.circular_acrophase.rayleigh_p:.2e}"
        )
    if rep.shift_report and rep.shift_report.pre_post_acrophase:
        ww = rep.shift_report.pre_post_acrophase
        print(
            f"Acrophase Shift      : {ww.mean1_hhmm} -> {ww.mean2_hhmm} | Delta={ww.phase_shift_hours:+.2f} h "
            f"({ww.shift_direction}), WW p={ww.p_value:.3e}"
        )
    rb = rep.rhythm_bands
    if rb is not None:
        dom = f"{rb.dominant_band} {rb.dominant_period_hours:.1f} h" if rb.dominant_band else "none significant"
        print(f"Rhythm Bands ({rb.period_min_hours:g}-{rb.period_max_hours:g} h): dominant = {dom}")
        for b in rb.bands:
            detail = f"{b.period_hours:5.1f} h, {b.percent_variance:4.1f}% var, p(Sidak)={b.p_adjusted:.1e}" if b.detected else ""
            print(f"  {b.band:<9} ({b.range_label} h): {b.classification}{'  [' + detail + ']' if detail else ''}")
    th = rep.twelve_hour
    if th is not None:
        s = lambda ok: "sig." if ok else "n.s."  # noqa: E731
        print(
            f"12 h Rhythm          : {th.rhythm_period_hours:.2f} h, amp={th.rhythm_amplitude:.3f}, peaks {th.rhythm_peaks_hhmm}, "
            f"{th.rhythm_percent_variance:.2f}% var, p(Sidak)={th.rhythm_p_adjusted:.2e} [{s(th.rhythm_significant)}]"
        )
        print(
            f"12 h Harmonic (tau/2): {th.harmonic_period_hours:.2f} h, amp={th.harmonic_amplitude:.3f}, peaks {th.harmonic_peaks_hhmm}, "
            f"{th.harmonic_percent_variance:.2f}% var, p={th.harmonic_p_value:.2e} [{s(th.harmonic_significant)}], "
            f"A2/A1={th.harmonic_to_fundamental_ratio:.2f} -> {'distinct' if th.distinct_from_harmonic else 'indistinguishable'}"
        )
    bm = rep.bimodality
    if bm is not None:
        peaks = f"{bm.primary_peak_hhmm} & {bm.secondary_peak_hhmm}" if bm.is_bimodal else bm.primary_peak_hhmm
        print(
            f"Bimodality           : {bm.classification} (BI={bm.shape.bimodality_index:.2f}, peak {peaks}, "
            f"A2/A1={bm.h2_to_h1_ratio:.2f}, 12h p={bm.h2_p_value:.2e}; bimodal days {bm.days_bimodal}/{bm.days_tested})"
        )
    print(f"Excel Workbook       : {excel_path.resolve()}")


if __name__ == "__main__":
    main()
