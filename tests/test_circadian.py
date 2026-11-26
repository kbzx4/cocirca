"""Comprehensive automated tests for circadian rhythmometry, harmonics, actograms, circular stats, Phyphox IO, and Excel export."""

from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from circadian_app.circular_stats import (
    analyze_acrophase_shift,
    compute_circular_summary,
    watson_williams_test,
    wrap_phase_shift_hours,
)
from circadian_app.cosinor import fit_cosinor
from circadian_app.excel_export import export_report_to_excel
from circadian_app.io_loader import (
    PhyphoxLiveRecorder,
    RawSensorDataset,
    generate_synthetic_circadian_data,
    load_phyphox_zip,
)
from circadian_app.onsets_nonparam import compute_non_parametric_metrics, detect_daily_onsets_offsets
from circadian_app.periodicity import analyze_periodicity
from circadian_app.pipeline import CircadianAnalysisConfig, run_full_analysis
from circadian_app.preprocessing import preprocess_to_epochs
from circadian_app.rhythm_bands import analyze_bimodality, analyze_rhythm_bands, analyze_twelve_hour
from circadian_app.theme import PlotTheme, build_theme, contrast_ratio


class TestCircadianSuite(unittest.TestCase):
    def test_cosinor_single_and_harmonics_recovery(self) -> None:
        t = np.arange(0, 7 * 24, 1 / 6)  # 10-min bins for 7 days
        w = 2.0 * np.pi / 24.0
        # Exact H1=24h (A=20, peak=15:00), H2=12h (A=8, peak=09:00), H3=8h (A=4, peak=04:00)
        y = (
            50.0
            + 20.0 * np.cos(w * (t - 15.0))
            + 8.0 * np.cos(2 * w * (t - 9.0))
            + 4.0 * np.cos(3 * w * (t - 4.0))
        )
        res1 = fit_cosinor(t, y, period_hours=24.0, n_harmonics=1)
        self.assertAlmostEqual(res1.mesor, 50.0, places=3)
        self.assertAlmostEqual(res1.amplitude, 20.0, places=3)
        self.assertAlmostEqual(res1.acrophase_hours, 15.0, places=2)
        self.assertEqual(res1.acrophase_hhmm, "15:00")

        res3 = fit_cosinor(t, y, period_hours=24.0, n_harmonics=3)
        self.assertGreater(res3.r2, 0.999)
        self.assertAlmostEqual(res3.harmonics[0].amplitude, 20.0, places=3)
        self.assertAlmostEqual(res3.harmonics[1].amplitude, 8.0, places=3)
        self.assertAlmostEqual(res3.harmonics[2].amplitude, 4.0, places=3)
        self.assertAlmostEqual(res3.harmonics[1].period_hours, 12.0, places=3)
        self.assertAlmostEqual(res3.harmonics[2].period_hours, 8.0, places=3)

    def test_periodicity_non_24h_recovery(self) -> None:
        ds = generate_synthetic_circadian_data(
            days=12, epoch_minutes=15, tau_hours=25.0, shift_day=None, shift_hours=0.0, noise_level=0.08
        )
        binned = preprocess_to_epochs(ds, epoch_minutes=15, activity_method="raw")
        pg = analyze_periodicity(
            binned.df["elapsed_hours"].to_numpy(),
            binned.df["activity"].to_numpy(),
            epoch_minutes=15,
        )
        self.assertAlmostEqual(pg.best_tau_lomb_scargle, 25.0, delta=0.3)
        self.assertAlmostEqual(pg.best_tau_cosinor, 25.0, delta=0.3)
        self.assertAlmostEqual(pg.consensus_tau_hours, 25.0, delta=0.4)

    def test_circular_statistics_and_midnight_wrap(self) -> None:
        # 23:00 and 01:00 should average to 00:00 (0.0h), NOT 12:00!
        cs = compute_circular_summary(np.array([23.0, 23.5, 0.5, 1.0]), label="Midnight Cluster")
        self.assertIsNotNone(cs)
        self.assertTrue(cs.mean_hour < 0.2 or cs.mean_hour > 23.8)
        self.assertGreater(cs.mean_resultant_length, 0.95)
        self.assertLess(cs.rayleigh_p, 0.05)

        # Uniformly spaced hours around clock -> R_bar ~ 0, Rayleigh p > 0.5
        cs_unif = compute_circular_summary(np.arange(0, 24, 3), label="Uniform")
        self.assertLess(cs_unif.mean_resultant_length, 0.1)
        self.assertGreater(cs_unif.rayleigh_p, 0.5)

        # Phase shift wrap across midnight: 23:00 -> 02:00 is +3.0h delay
        self.assertAlmostEqual(wrap_phase_shift_hours(2.0 - 23.0), 3.0)
        # 01:00 -> 22:00 is -3.0h advance
        self.assertAlmostEqual(wrap_phase_shift_hours(22.0 - 1.0), -3.0)

        # Watson-Williams two-sample test
        ww = watson_williams_test(np.array([23.0, 23.2, 23.1, 22.9]), np.array([2.0, 2.2, 1.9, 2.1]))
        self.assertIsNotNone(ww)
        self.assertAlmostEqual(ww.phase_shift_hours, 3.0, delta=0.2)
        self.assertEqual(ww.shift_direction, "Phase Delay")
        self.assertLess(ww.p_value, 0.001)

    def test_onsets_offsets_and_non_parametric(self) -> None:
        # Active 08:00 to 23:00 every day
        start = datetime(2026, 11, 1, 0, 0, 0)
        ts = pd.date_range(start, periods=5 * 24 * 6, freq="10min")
        tod = ts.hour + ts.minute / 60.0
        act = np.where((tod >= 8.0) & (tod < 23.0), 80.0, 2.0)
        ds = RawSensorDataset(
            df=pd.DataFrame({"timestamp": ts, "activity_raw": act}),
            source_name="square_wave",
            source_type="test",
            has_activity=True,
            has_light=False,
        )
        binned = preprocess_to_epochs(ds, epoch_minutes=10, activity_method="raw")
        onsets = detect_daily_onsets_offsets(binned.df, value_col="activity", method="template", epoch_minutes=10)
        self.assertGreaterEqual(len(onsets), 4)
        self.assertAlmostEqual(onsets["onset_hours"].median(), 8.0, delta=0.5)
        self.assertAlmostEqual(onsets["offset_hours"].median(), 23.0, delta=0.5)
        self.assertAlmostEqual(onsets["active_duration_hours"].median(), 15.0, delta=0.6)

        np_res = compute_non_parametric_metrics(binned.df, value_col="activity")
        self.assertIsNotNone(np_res)
        self.assertGreater(np_res.interdaily_stability_is, 0.95)
        self.assertGreater(np_res.relative_amplitude_ra, 0.90)

    def test_phyphox_zip_import_and_live_recorder_sim(self) -> None:
        # Build synthetic Phyphox zip in memory
        buf = io.BytesIO()
        start_unix = datetime(2026, 11, 20, 6, 0, 0).timestamp()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "meta/time.csv",
                f"event,experiment time,system time,system time text\nSTART,0.0,{start_unix},2026-11-20 06:00:00\n",
            )
            t_s = np.arange(0, 4 * 24 * 3600, 600)
            lux = 10.0 + 300.0 * np.maximum(0.0, np.cos(2 * np.pi * (t_s / 3600.0 - 6.0) / 24.0))
            light_df = pd.DataFrame({"Time (s)": t_s, "Illuminance (lx)": lux})
            zf.writestr("Light.csv", light_df.to_csv(index=False))

            az = 9.81 + 0.5 * np.maximum(0.0, np.cos(2 * np.pi * (t_s / 3600.0 - 8.0) / 24.0))
            acc_df = pd.DataFrame(
                {
                    "Time (s)": t_s,
                    "Acceleration x (m/s^2)": np.zeros_like(t_s, dtype=float),
                    "Acceleration y (m/s^2)": np.zeros_like(t_s, dtype=float),
                    "Acceleration z (m/s^2)": az,
                }
            )
            zf.writestr("Accelerometer.csv", acc_df.to_csv(index=False))

        buf.seek(0)
        ds = load_phyphox_zip(buf)
        self.assertTrue(ds.has_light)
        self.assertTrue(ds.has_activity)
        self.assertGreater(len(ds.df), 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "live.csv"
            rec = PhyphoxLiveRecorder(output_csv=csv_path, simulate=True)
            row = rec.poll_once()
            self.assertIsNotNone(row)
            loaded = rec.load_recorded_dataset()
            self.assertEqual(len(loaded.df), 1)

    def test_rhythm_bands_and_bimodality(self) -> None:
        rng = np.random.default_rng(7)
        t = np.arange(0, 14 * 24, 10 / 60)
        w = lambda p: 2 * np.pi / p  # noqa: E731
        periods = np.arange(9.0, 36.05, 0.1)
        noise = rng.normal(0, 8, t.size)

        # 24 h + strong 12 h: one circadian rhythm, 12 h flagged as its harmonic, bimodal waveform
        y = 50 + 20 * np.cos(w(24) * (t - 14)) + 14 * np.cos(w(12) * (t - 8)) + noise
        rb = analyze_rhythm_bands(t, y, periods)
        self.assertEqual(rb.dominant_band, "Circadian")
        self.assertAlmostEqual(rb.band("Circadian").period_hours, 24.0, delta=0.3)
        self.assertFalse(rb.band("Ultradian").detected)
        self.assertIn("Harmonic", rb.band("Ultradian").classification)
        self.assertFalse(rb.band("Infradian").detected)
        bm = analyze_bimodality(t, y, 24.0)
        self.assertTrue(bm.is_bimodal)
        self.assertGreater(bm.shape.bimodality_index, 0.1)

        # 24 h + independent 15 h: both detected, waveform unimodal
        y = 50 + 20 * np.cos(w(24) * (t - 14)) + 10 * np.cos(w(15) * t) + noise
        rb = analyze_rhythm_bands(t, y, periods)
        self.assertTrue(rb.band("Ultradian").detected)
        self.assertAlmostEqual(rb.band("Ultradian").period_hours, 15.0, delta=0.3)
        self.assertFalse(analyze_bimodality(t, 50 + 20 * np.cos(w(24) * (t - 14)) + noise, 24.0).is_bimodal)

        # Pure 32 h: infradian dominant; its circadian-band sidelobe must not count as a circadian rhythm
        rb = analyze_rhythm_bands(t, 50 + 20 * np.cos(w(32) * t) + noise, periods)
        self.assertEqual(rb.dominant_band, "Infradian")
        self.assertAlmostEqual(rb.dominant_period_hours, 32.0, delta=0.4)
        self.assertFalse(rb.band("Circadian").detected)

        # Noise only: nothing detected
        rb = analyze_rhythm_bands(t, 50 + noise, periods)
        self.assertIsNone(rb.dominant_band)

        # 12 h components are always reported. Harmonic at exactly τ/2 -> indistinguishable from a 12 h rhythm
        th = analyze_twelve_hour(t, 50 + 20 * np.cos(w(24) * (t - 14)) + 14 * np.cos(w(12) * (t - 8)) + noise, 24.0)
        self.assertTrue(th.harmonic_significant)
        self.assertAlmostEqual(th.rhythm_period_hours, 12.0, delta=0.15)
        self.assertFalse(th.distinct_from_harmonic)
        # Independent 13 h rhythm next to 24 h -> distinct from the τ/2 = 12 h harmonic
        th = analyze_twelve_hour(t, 50 + 20 * np.cos(w(24) * (t - 14)) + 10 * np.cos(w(13) * t) + noise, 24.0)
        self.assertTrue(th.rhythm_significant)
        self.assertAlmostEqual(th.rhythm_period_hours, 13.0, delta=0.15)
        self.assertTrue(th.distinct_from_harmonic)
        # Weak 12 h content: still reported, flagged not significant
        th = analyze_twelve_hour(t, 50 + 20 * np.cos(w(24) * (t - 14)) + noise, 24.0)
        self.assertIsNotNone(th)
        self.assertFalse(th.rhythm_significant)
        self.assertFalse(th.harmonic_significant)
        self.assertTrue(np.isfinite(th.harmonic_amplitude) and np.isfinite(th.rhythm_amplitude))

    def test_chart_themes(self) -> None:
        light = build_theme("#ffffff", "Default")
        self.assertEqual(light.activity, PlotTheme().activity)
        dark = build_theme("#000000", "Default")
        self.assertTrue(dark.is_dark)
        # Dark-on-dark roles are swapped for the light foreground so they stay visible
        self.assertGreater(contrast_ratio(dark.activity, dark.background), 1.8)
        # Explicit user picks are kept even when low-contrast, and palettes apply
        self.assertEqual(build_theme("#000000", "Default", {"activity": "#111111"}).activity, "#111111")
        self.assertEqual(build_theme("#ffffff", "Colourblind-safe (Okabe-Ito)").onset, "#009E73")

    def test_axis_label_styling(self) -> None:
        import plotly.graph_objects as go
        from matplotlib.figure import Figure

        from circadian_app.plotting import apply_mpl_theme, apply_plotly_theme
        from circadian_app.theme import reset_seen_labels, seen_axis_labels, set_plot_theme

        theme = build_theme(
            "#ffffff",
            label_style={
                "label_font_family": "DejaVu Serif", "label_font_size": 15.0, "label_bold": True,
                "x_label_color": "#ff0000", "y_label_color": "#0000ff", "label_text": (("Hours", "Time (h)"),),
            },
        )
        try:
            reset_seen_labels()
            set_plot_theme(theme)
            fig = Figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel("Hours")
            ax.set_ylabel("Counts")
            apply_mpl_theme(fig)
            apply_mpl_theme(fig)  # idempotent
            self.assertEqual(ax.xaxis.label.get_text(), "Time (h)")
            self.assertEqual(ax.yaxis.label.get_text(), "Counts")
            self.assertEqual(ax.xaxis.label.get_fontsize(), 15.0)
            self.assertEqual(ax.xaxis.label.get_fontweight(), "bold")
            self.assertEqual(ax.xaxis.label.get_color(), "#ff0000")
            self.assertEqual(ax.yaxis.label.get_color(), "#0000ff")

            pf = go.Figure()
            pf.update_layout(xaxis=dict(title="Hours"), yaxis=dict(title="Counts"))
            apply_plotly_theme(pf)
            self.assertEqual(pf.layout.xaxis.title.text, "<b>Time (h)</b>")
            self.assertEqual(pf.layout.xaxis.title.font.color, "#ff0000")
            self.assertEqual(pf.layout.yaxis.title.font.size, 15.0)
            self.assertIn("Hours", seen_axis_labels())
        finally:
            set_plot_theme(build_theme("#ffffff"))

    def test_full_pipeline_acrophase_shift_and_excel_export(self) -> None:
        ds = generate_synthetic_circadian_data(
            days=14, epoch_minutes=10, tau_hours=24.0, acrophase_hour=14.0, shift_day=8, shift_hours=3.5
        )
        cfg = CircadianAnalysisConfig(
            epoch_minutes=10,
            activity_method="raw",
            n_harmonics=3,
            period_mode="fixed_24h",
            split_day=8,
        )
        report = run_full_analysis(ds, cfg)
        self.assertIsNotNone(report.activity_report)
        self.assertIsNotNone(report.light_report)
        sr = report.activity_report.shift_report
        self.assertIsNotNone(sr)
        self.assertIsNotNone(sr.pre_post_acrophase)
        self.assertAlmostEqual(sr.pre_post_acrophase.phase_shift_hours, 3.3, delta=0.8)
        self.assertEqual(sr.pre_post_acrophase.shift_direction, "Phase Delay")

        with tempfile.TemporaryDirectory() as tmpdir:
            out_xlsx = Path(tmpdir) / "test_circadian.xlsx"
            xlsx_bytes = export_report_to_excel(report, output_path=out_xlsx)
            self.assertGreater(len(xlsx_bytes), 20000)
            self.assertTrue(out_xlsx.exists())

            wb = openpyxl.load_workbook(out_xlsx)
            self.assertEqual(
                wb.sheetnames,
                [
                    "Summary_Report",
                    "Cosinor_Harmonics",
                    "Daily_Rhythms_Onsets",
                    "Circular_Statistics",
                    "Periodogram",
                    "Rhythm_Bands",
                    "Binned_Epochs",
                ],
            )
            # Verify embedded charts exist on sheets
            self.assertGreaterEqual(len(wb["Summary_Report"]._images), 1)
            self.assertGreaterEqual(len(wb["Cosinor_Harmonics"]._images), 1)
            self.assertGreaterEqual(len(wb["Circular_Statistics"]._images), 1)


if __name__ == "__main__":
    unittest.main()
