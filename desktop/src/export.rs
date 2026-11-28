//! Excel workbook (with native, theme-styled charts) and full-resolution CSV export.

use crate::data::ms_to_naive;
use crate::pipeline::{ChannelReport, Report};
use crate::stats::hhmm;
use crate::theme::Theme;
use eframe::egui::Color32;
use rust_xlsxwriter::{Chart, ChartFont, ChartFormat, ChartLine, ChartSolidFill, ChartType, Color, Format, Workbook, Worksheet, XlsxError};
use std::io::Write;
use std::path::Path;

pub const EXCEL_EPOCH_ROWS: usize = 100_000;

fn xc(c: Color32) -> Color {
    Color::RGB(((c.r() as u32) << 16) | ((c.g() as u32) << 8) | c.b() as u32)
}

/// Writes NaN / inf as blank cells (Excel has no NaN).
fn num(ws: &mut Worksheet, r: u32, c: u16, v: f64) -> Result<(), XlsxError> {
    if v.is_finite() {
        ws.write_number(r, c, v)?;
    }
    Ok(())
}

fn kv(ws: &mut Worksheet, r: &mut u32, key: &str, val: impl Into<Cell>) -> Result<(), XlsxError> {
    ws.write_string(*r, 0, key)?;
    match val.into() {
        Cell::N(v) => num(ws, *r, 1, v)?,
        Cell::S(s) => {
            ws.write_string(*r, 1, s)?;
        }
    }
    *r += 1;
    Ok(())
}

enum Cell {
    N(f64),
    S(String),
}
impl From<f64> for Cell {
    fn from(v: f64) -> Self {
        Cell::N(v)
    }
}
impl From<String> for Cell {
    fn from(v: String) -> Self {
        Cell::S(v)
    }
}
impl From<&str> for Cell {
    fn from(v: &str) -> Self {
        Cell::S(v.to_string())
    }
}

fn style_chart(chart: &mut Chart, theme: &Theme, title: &str, x: &str, y: &str) {
    let mut fx = ChartFont::new();
    let mut fy = ChartFont::new();
    fx.set_color(xc(theme.x_label_color()));
    fy.set_color(xc(theme.y_label_color()));
    if let Some(f) = &theme.label.font {
        fx.set_name(f);
        fy.set_name(f);
    }
    if theme.label.size > 0.0 {
        fx.set_size(theme.label.size as f64);
        fy.set_size(theme.label.size as f64);
    }
    if theme.label.bold {
        fx.set_bold();
        fy.set_bold();
    }
    chart.title().set_name(title);
    chart.x_axis().set_name(&theme.label_text(x)).set_name_font(&fx);
    chart.y_axis().set_name(&theme.label_text(y)).set_name_font(&fy);
    let bg = ChartFormat::new().set_solid_fill(ChartSolidFill::new().set_color(xc(theme.background))).clone();
    chart.chart_area().set_format(&mut bg.clone());
    chart.plot_area().set_format(&mut bg.clone());
    chart.set_width(720).set_height(360);
}

fn line_fmt(c: Color32, w: f64) -> ChartFormat {
    ChartFormat::new().set_line(ChartLine::new().set_color(xc(c)).set_width(w)).clone()
}

fn channel_summary(ws: &mut Worksheet, r: &mut u32, ch: &ChannelReport, bold: &Format) -> Result<(), XlsxError> {
    ws.write_string_with_format(*r, 0, format!("{} ({})", ch.name, ch.unit), bold)?;
    *r += 1;
    let pg = &ch.periodogram;
    kv(ws, r, "Period used τ (h)", ch.tau)?;
    kv(ws, r, "Consensus τ (h)", pg.consensus_tau)?;
    kv(ws, r, "Lomb-Scargle τ (h)", pg.best_tau_ls)?;
    kv(ws, r, "Lomb-Scargle power", pg.best_ls)?;
    kv(ws, r, "χ² periodogram τ (h)", pg.best_tau_chi2)?;
    kv(ws, r, "χ² Qp", pg.best_qp)?;
    kv(ws, r, "χ² p", pg.best_chi2_p)?;
    let s = &ch.single;
    kv(ws, r, "MESOR", s.mesor)?;
    kv(ws, r, "Amplitude", s.amplitude())?;
    kv(ws, r, "Acrophase", s.acrophase_hhmm())?;
    kv(ws, r, "Acrophase (deg)", s.deg_neg())?;
    kv(ws, r, "Bathyphase", hhmm(s.bathy_h))?;
    kv(ws, r, "Single cosinor R²", s.r2)?;
    kv(ws, r, "Single cosinor p", s.p)?;
    let h = &ch.harmonics;
    kv(ws, r, &format!("{}-harmonic R²", h.nh), h.r2)?;
    kv(ws, r, "Orthophase", hhmm(h.ortho_h))?;
    kv(ws, r, "Composite bathyphase", hhmm(h.comp_bathy_h))?;
    kv(ws, r, "Composite amplitude", h.comp_amp)?;
    if let Some(np) = &ch.non_param {
        kv(ws, r, "Interdaily stability (IS)", np.is)?;
        kv(ws, r, "Intradaily variability (IV)", np.iv)?;
        kv(ws, r, "Relative amplitude (RA)", np.ra)?;
        kv(ws, r, "M10 onset", hhmm(np.m10_onset))?;
        kv(ws, r, "L5 onset", hhmm(np.l5_onset))?;
    }
    if let Some(c) = &ch.circ_acrophase {
        kv(ws, r, "Circular mean acrophase", c.mean_hhmm())?;
        kv(ws, r, "R̄ (acrophase)", c.r_bar)?;
        kv(ws, r, "Rayleigh p", c.rayleigh_p)?;
    }
    if let Some(sh) = &ch.shift {
        kv(ws, r, "Drift slope (h/day)", sh.drift_slope)?;
        kv(ws, r, "Implied τ from drift (h)", sh.implied_tau)?;
        if let Some(t) = &sh.acrophase {
            kv(ws, r, &format!("Acrophase shift at day {} (h)", sh.split_day), t.shift_h)?;
            kv(ws, r, "Watson-Williams p", t.p)?;
            kv(ws, r, "Shift direction", t.direction)?;
        }
        if let Some(pa) = sh.act_light_phase_angle {
            kv(ws, r, "Activity − light phase angle (h)", pa)?;
        }
    }
    *r += 1;
    Ok(())
}

pub fn export_excel(rep: &Report, theme: &Theme, path: &Path) -> Result<(), String> {
    write_excel(rep, theme, path).map_err(|e| e.to_string())
}

fn write_excel(rep: &Report, theme: &Theme, path: &Path) -> Result<(), XlsxError> {
    let mut wb = Workbook::new();
    let bold = Format::new().set_bold();
    let b = &rep.binned;
    let p = rep.primary();

    // Summary
    {
        let ws = wb.add_worksheet();
        ws.set_name("Summary")?;
        ws.set_column_width(0, 36)?;
        ws.set_column_width(1, 26)?;
        ws.write_string_with_format(0, 0, "Circadian Rhythmometry, Actography & Circular Statistics Report", &bold)?;
        let mut r = 2;
        kv(ws, &mut r, "Source", rep.source.as_str())?;
        kv(ws, &mut r, "Bin size", crate::data::format_bin_size(b.epoch_sec))?;
        kv(ws, &mut r, "Bins", b.len() as f64)?;
        kv(ws, &mut r, "Recording (days)", b.n_days)?;
        kv(ws, &mut r, "Start", ms_to_naive(b.start_ms).format("%Y-%m-%d %H:%M:%S").to_string())?;
        for w in &rep.warnings {
            kv(ws, &mut r, "Warning", w.as_str())?;
        }
        r += 1;
        for ch in rep.channels() {
            channel_summary(ws, &mut r, ch, &bold)?;
        }
    }

    // Rhythm bands, 12 h, bimodality
    {
        let ws = wb.add_worksheet();
        ws.set_name("Rhythm_Bands")?;
        let mut r = 0u32;
        for ch in rep.channels() {
            ws.write_string_with_format(r, 0, format!("{} rhythm bands", ch.name), &bold)?;
            r += 1;
            for (c, h) in ["Band", "Range (h)", "Classification", "Period (h)", "Amplitude", "Acrophase", "% variance", "p", "p (Šidák)", "Raw peak (h)", "Note"].iter().enumerate() {
                ws.write_string_with_format(r, c as u16, *h, &bold)?;
            }
            r += 1;
            if let Some(bands) = &ch.bands {
                for band in &bands.bands {
                    ws.write_string(r, 0, band.band)?;
                    ws.write_string(r, 1, band.range_label())?;
                    ws.write_string(r, 2, &band.classification)?;
                    num(ws, r, 3, band.period)?;
                    num(ws, r, 4, band.amplitude)?;
                    ws.write_string(r, 5, &band.acrophase_hhmm)?;
                    num(ws, r, 6, band.percent_variance)?;
                    num(ws, r, 7, band.p_value)?;
                    num(ws, r, 8, band.p_adjusted)?;
                    num(ws, r, 9, band.raw_peak_period)?;
                    ws.write_string(r, 10, &band.note)?;
                    r += 1;
                }
            }
            r += 1;
            if let Some(th) = &ch.twelve {
                ws.write_string_with_format(r, 0, "12 h components", &bold)?;
                r += 1;
                for (c, h) in ["Component", "Period (h)", "Amplitude", "Peaks", "% variance", "p", "Significant"].iter().enumerate() {
                    ws.write_string_with_format(r, c as u16, *h, &bold)?;
                }
                r += 1;
                for (name, per, amp, peaks, pct, pv, sig) in [
                    ("12 h rhythm (free period)", th.rhythm_period, th.rhythm_amp, &th.rhythm_peaks, th.rhythm_pct, th.rhythm_p_adj, th.rhythm_sig),
                    ("τ/2 harmonic of circadian rhythm", th.harm_period, th.harm_amp, &th.harm_peaks, th.harm_pct, th.harm_p, th.harm_sig),
                ] {
                    ws.write_string(r, 0, name)?;
                    num(ws, r, 1, per)?;
                    num(ws, r, 2, amp)?;
                    ws.write_string(r, 3, peaks)?;
                    num(ws, r, 4, pct)?;
                    num(ws, r, 5, pv)?;
                    ws.write_string(r, 6, if sig { "yes" } else { "no" })?;
                    r += 1;
                }
                ws.write_string(r, 0, &th.interpretation)?;
                r += 2;
            }
            if let Some(bm) = &ch.bimodality {
                ws.write_string_with_format(r, 0, "Bimodality", &bold)?;
                r += 1;
                kv(ws, &mut r, "Classification", bm.classification.as_str())?;
                kv(ws, &mut r, "Bimodality index", bm.shape.bimodality_index)?;
                kv(ws, &mut r, "Primary peak", hhmm(bm.shape.primary_h))?;
                kv(ws, &mut r, "Secondary peak", hhmm(bm.shape.secondary_h))?;
                kv(ws, &mut r, "Dip", hhmm(bm.shape.dip_h))?;
                kv(ws, &mut r, "A2/A1", bm.h2_h1)?;
                kv(ws, &mut r, "12 h component p", bm.h2_p)?;
                kv(ws, &mut r, "Bimodal days", format!("{}/{}", bm.days_bimodal, bm.days_tested))?;
                r += 1;
            }
        }
        ws.set_column_width(0, 30)?;
        ws.set_column_width(2, 52)?;
    }

    // Daily tables
    for ch in rep.channels() {
        let ws = wb.add_worksheet();
        ws.set_name(format!("Daily_{}", ch.name))?;
        let heads = [
            "Day", "Date", "Valid bins", "Mean", "Max", "MESOR", "Amplitude", "Acrophase (h)", "Acrophase", "Orthophase", "R²", "p", "Onset", "Offset", "Active (h)", "Rest (h)", "Midpoint", "Bimodal", "Bimodality index", "Peak 1", "Peak 2", "Shift (h)", "Cumulative shift (h)", "Light acrophase (h)", "Phase angle act−light (h)",
        ];
        for (c, h) in heads.iter().enumerate() {
            ws.write_string_with_format(0, c as u16, *h, &bold)?;
        }
        for (i, d) in ch.daily.iter().enumerate() {
            let r = i as u32 + 1;
            ws.write_number(r, 0, d.day as f64)?;
            ws.write_string(r, 1, &d.date)?;
            ws.write_number(r, 2, d.n_valid as f64)?;
            for (c, v) in [(3, d.mean_value), (4, d.max_value), (5, d.mesor), (6, d.amplitude), (7, d.acrophase_h), (10, d.r2), (11, d.p), (14, d.active_h), (15, d.rest_h), (18, d.bimodality_index), (21, d.shift_h), (22, d.cumulative_shift_h), (23, d.light_acrophase_h), (24, d.phase_angle_h)] {
                num(ws, r, c, v)?;
            }
            for (c, v) in [(8, d.acrophase_h), (9, d.orthophase_h), (12, d.onset_h), (13, d.offset_h), (16, d.midpoint_h), (19, d.peak1_h), (20, d.peak2_h)] {
                if v.is_finite() {
                    ws.write_string(r, c, hhmm(v))?;
                }
            }
            if let Some(bm) = d.bimodal {
                ws.write_string(r, 17, if bm { "yes" } else { "no" })?;
            }
        }
        ws.set_freeze_panes(1, 0)?;
    }

    // Periodogram + chart
    {
        let pg = &p.periodogram;
        let n = pg.periods.len() as u32;
        let ws = wb.add_worksheet();
        ws.set_name("Periodogram")?;
        for (c, h) in ["Period (h)", "Lomb-Scargle / cosinor R²", "χ² Qp", "χ² critical (α=0.05)", "Residual R² after prewhitening"].iter().enumerate() {
            ws.write_string_with_format(0, c as u16, *h, &bold)?;
        }
        for i in 0..pg.periods.len() {
            let r = i as u32 + 1;
            num(ws, r, 0, pg.periods[i])?;
            num(ws, r, 1, pg.ls_power[i])?;
            num(ws, r, 2, pg.chi2_qp[i])?;
            num(ws, r, 3, pg.chi2_crit[i])?;
            if let Some(bands) = &p.bands {
                num(ws, r, 4, bands.residual_r2.get(i).copied().unwrap_or(f64::NAN))?;
            }
        }
        let mut chart = Chart::new(ChartType::ScatterStraight);
        chart.add_series().set_name("Lomb-Scargle / R²").set_categories(("Periodogram", 1, 0, n, 0)).set_values(("Periodogram", 1, 1, n, 1)).set_format(&mut line_fmt(theme.role("accent"), 1.75));
        if p.bands.is_some() {
            chart.add_series().set_name("Residual R²").set_categories(("Periodogram", 1, 0, n, 0)).set_values(("Periodogram", 1, 4, n, 4)).set_format(&mut line_fmt(theme.role("sweep"), 1.25));
        }
        style_chart(&mut chart, theme, &format!("{} periodogram", p.name), "Period τ (hours)", "Normalized Power");
        ws.insert_chart(1, 6, &chart)?;
        let mut chart2 = Chart::new(ChartType::ScatterStraight);
        chart2.add_series().set_name("χ² Qp").set_categories(("Periodogram", 1, 0, n, 0)).set_values(("Periodogram", 1, 2, n, 2)).set_format(&mut line_fmt(theme.role("chi2"), 1.75));
        chart2.add_series().set_name("Critical (α=0.05)").set_categories(("Periodogram", 1, 0, n, 0)).set_values(("Periodogram", 1, 3, n, 3)).set_format(&mut line_fmt(theme.role("threshold"), 1.25));
        style_chart(&mut chart2, theme, "Sokolove-Bushell χ² periodogram", "Period τ (hours)", "Qp");
        ws.insert_chart(20, 6, &chart2)?;
    }

    // Cosinor waveform + chart
    {
        let ws = wb.add_worksheet();
        ws.set_name("Cosinor_Profile")?;
        for (c, h) in ["Phase (h)", "Single cosinor", "Multi-harmonic cosinor", "Mean of data"].iter().enumerate() {
            ws.write_string_with_format(0, c as u16, *h, &bold)?;
        }
        let tau = p.tau;
        let nb = 288usize;
        let (mut sums, mut cnts) = (vec![0.0; nb], vec![0u32; nb]);
        for (t, v) in b.elapsed_h.iter().zip(&p.values) {
            if v.is_finite() {
                let k = ((t.rem_euclid(tau) / tau) * nb as f64).floor() as usize % nb;
                sums[k] += v;
                cnts[k] += 1;
            }
        }
        for k in 0..nb {
            let ph = (k as f64 + 0.5) * tau / nb as f64;
            let r = k as u32 + 1;
            ws.write_number(r, 0, ph)?;
            ws.write_number(r, 1, p.single.eval(ph))?;
            ws.write_number(r, 2, p.harmonics.eval(ph))?;
            if cnts[k] > 0 {
                ws.write_number(r, 3, sums[k] / cnts[k] as f64)?;
            }
        }
        let n = nb as u32;
        let mut chart = Chart::new(ChartType::ScatterStraight);
        chart.add_series().set_name("Mean of data").set_categories(("Cosinor_Profile", 1, 0, n, 0)).set_values(("Cosinor_Profile", 1, 3, n, 3)).set_format(&mut line_fmt(theme.role("activity"), 1.0));
        chart.add_series().set_name("Single cosinor").set_categories(("Cosinor_Profile", 1, 0, n, 0)).set_values(("Cosinor_Profile", 1, 1, n, 1)).set_format(&mut line_fmt(theme.role("accent"), 1.5));
        chart.add_series().set_name(&format!("{}-harmonic cosinor", p.harmonics.nh)).set_categories(("Cosinor_Profile", 1, 0, n, 0)).set_values(("Cosinor_Profile", 1, 2, n, 2)).set_format(&mut line_fmt(theme.role("fit"), 2.0));
        style_chart(&mut chart, theme, &format!("{} waveform over τ = {:.2} h", p.name, tau), "Phase / Clock Hour (h)", &format!("{} ({})", p.name, p.unit));
        ws.insert_chart(1, 5, &chart)?;
    }

    // Non-parametric hourly profile
    if let Some(np) = &p.non_param {
        let ws = wb.add_worksheet();
        ws.set_name("NonParametric")?;
        ws.write_string_with_format(0, 0, "Hour", &bold)?;
        ws.write_string_with_format(0, 1, "Mean", &bold)?;
        for h in 0..24 {
            ws.write_number(h as u32 + 1, 0, h as f64)?;
            num(ws, h as u32 + 1, 1, np.profile[h])?;
        }
        let mut chart = Chart::new(ChartType::Column);
        chart.add_series().set_name("Hourly mean").set_categories(("NonParametric", 1, 0, 24, 0)).set_values(("NonParametric", 1, 1, 24, 1)).set_format(ChartSolidFill::new().set_color(xc(theme.role("activity"))));
        style_chart(&mut chart, theme, "24 h profile (IS / IV / L5 / M10)", "Hour of Day", &format!("{} ({})", p.name, p.unit));
        ws.insert_chart(1, 3, &chart)?;
    }

    // Epochs (capped; the CSV export has every bin)
    {
        let ws = wb.add_worksheet();
        ws.set_name("Epochs")?;
        let cols = epoch_columns(b);
        let mut heads: Vec<String> = ["Timestamp", "Elapsed (h)", "Day", "Clock hour"].iter().map(|s| s.to_string()).collect();
        heads.extend(cols.iter().map(|c| c.0.clone()));
        heads.push(format!("{} single cosinor fit", p.name));
        heads.push(format!("{} harmonic cosinor fit", p.name));
        for (c, h) in heads.iter().enumerate() {
            ws.write_string_with_format(0, c as u16, h, &bold)?;
        }
        let n = b.len().min(EXCEL_EPOCH_ROWS);
        let k = cols.len() as u16;
        for i in 0..n {
            let r = i as u32 + 1;
            ws.write_string(r, 0, ms_to_naive(b.bin_start_ms(i)).format("%Y-%m-%d %H:%M:%S").to_string())?;
            ws.write_number(r, 1, b.elapsed_h[i])?;
            ws.write_number(r, 2, b.day_index[i] as f64)?;
            ws.write_number(r, 3, b.clock_h[i])?;
            for (j, c) in cols.iter().enumerate() {
                num(ws, r, 4 + j as u16, c.1[i])?;
            }
            ws.write_number(r, 4 + k, p.single.eval(b.elapsed_h[i]))?;
            ws.write_number(r, 5 + k, p.harmonics.eval(b.elapsed_h[i]))?;
        }
        if b.len() > n {
            ws.write_string(n as u32 + 2, 0, format!("Truncated at {EXCEL_EPOCH_ROWS} of {} bins; use File → Export epochs CSV for all of them.", b.len()))?;
        }
        ws.set_freeze_panes(1, 0)?;
        ws.set_column_width(0, 20)?;
    }
    wb.save(path)?;
    Ok(())
}

/// (header, values) for every binned channel present: motion, other signals, light.
fn epoch_columns(b: &crate::data::Binned) -> Vec<(String, &[f64])> {
    let mut cols: Vec<(String, &[f64])> = Vec::new();
    if b.has_activity {
        cols.push((format!("Activity ({})", b.activity_unit), &b.activity));
    }
    for x in &b.extra {
        cols.push((if x.unit.is_empty() { x.name.clone() } else { format!("{} ({})", x.name, x.unit) }, &x.values));
    }
    if b.has_light {
        cols.push(("Lux".into(), &b.lux));
        cols.push(("log10(lux+1)".into(), &b.log_lux));
    }
    cols
}

/// Every bin, with both cosinor fits, as CSV. Rows are formatted in parallel chunks.
pub fn export_csv(rep: &Report, path: &Path) -> Result<(), String> {
    use rayon::prelude::*;
    let f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
    let b = &rep.binned;
    let p = rep.primary();
    let cols = epoch_columns(b);
    let csv_name = |s: &str| if s.contains([',', '"']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s.to_string() };
    let mut head = vec!["timestamp".to_string(), "elapsed_hours".into(), "day_index".into(), "clock_hour".into()];
    head.extend(cols.iter().map(|c| csv_name(&c.0)));
    head.push("cosinor_single_fit".into());
    head.push("cosinor_harmonics_fit".into());
    // UTF-8 BOM so Excel shows units such as °C correctly
    writeln!(w, "\u{feff}{}", head.join(",")).map_err(|e| e.to_string())?;
    let idx: Vec<usize> = (0..b.len()).collect();
    let chunks: Vec<String> = idx
        .par_chunks(50_000)
        .map(|ch| {
            use std::fmt::Write as _;
            let mut s = String::with_capacity(ch.len() * 96);
            for &i in ch {
                let t = b.elapsed_h[i];
                let _ = write!(s, "{},{},{},{}", ms_to_naive(b.bin_start_ms(i)).format("%Y-%m-%d %H:%M:%S"), t, b.day_index[i], b.clock_h[i]);
                for c in &cols {
                    let v = c.1[i];
                    if v.is_finite() {
                        let _ = write!(s, ",{v}");
                    } else {
                        s.push(',');
                    }
                }
                let _ = writeln!(s, ",{},{}", p.single.eval(t), p.harmonics.eval(t));
            }
            s
        })
        .collect();
    for c in chunks {
        w.write_all(c.as_bytes()).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}
