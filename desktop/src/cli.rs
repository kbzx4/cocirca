//! Headless command-line runner (the `cocirca-cli` console binary).

use crate::{data, export, import, pipeline, stats, theme};
use std::path::PathBuf;

pub const USAGE: &str = "cocirca-cli [FILES...] [--demo] [--bin-sec N] [--harmonics N] [--period-min H] [--period-max H]
            [--circadian-min H] [--circadian-max H] [--tau fixed24|ls|chi2|cosinor|H]
            [--onsets template|threshold|cosinor] [--split-day N] [--start \"YYYY-MM-DD HH:MM\"]
            [--excel OUT.xlsx] [--csv OUT.csv]
   import: [--date-order dmy|mdy|ymd] [--time-unit ms|s|min|h|d] [--interval SECONDS] [--header-line N]
            [--layout channels|subjects] [--subject N] [--merge|--separate]
       cocirca-cli --batch FILE|ARCHIVE.zip|FOLDER [analysis & import options] [--out summary.csv]
            analyses every subject (file, DAM channel, plate well, gene row …) separately, in parallel";

fn parse_start(s: &str) -> Result<i64, String> {
    ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"]
        .iter()
        .find_map(|f| chrono::NaiveDateTime::parse_from_str(s.trim(), f).ok())
        .map(data::naive_to_ms)
        .ok_or_else(|| format!("--start: cannot parse {s:?} (use \"YYYY-MM-DD HH:MM\")"))
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn f(v: f64, d: usize) -> String {
    if v.is_finite() {
        format!("{v:.d$}")
    } else {
        String::new()
    }
}

/// One summary row per recording (all BATCH_HEAD columns except the leading name and trailing error).
pub fn batch_row(d: &data::RawDataset, rep: &pipeline::Report, ms: f64) -> Vec<String> {
    let ch = rep.primary();
    let s = &ch.single;
    let (rr_n, rr_bad) = d.rr_counts.unwrap_or((d.len(), 0));
    let t = ch.twelve.as_ref();
    let bm = ch.bimodality.as_ref();
    let b = ch.bands.as_ref();
    vec![
        ch.name.to_string(),
        rr_n.to_string(),
        rr_bad.to_string(),
        f(rep.binned.n_days * 24.0, 2),
        f(stats::finite_mean(&ch.values), 2),
        f(ch.tau, 2),
        f(s.mesor, 3),
        f(s.amplitude(), 3),
        f(s.acrophase_h(), 3),
        s.acrophase_hhmm(),
        f(s.r2, 4),
        format!("{:.3e}", s.p),
        f(ch.harmonics.r2, 4),
        t.map_or(String::new(), |t| f(t.harm_amp, 3)),
        t.map_or(String::new(), |t| format!("{:.3e}", t.harm_p)),
        t.map_or(String::new(), |t| f(t.rhythm_period, 2)),
        t.map_or(String::new(), |t| format!("{:.3e}", t.rhythm_p_adj)),
        bm.map_or(String::new(), |b| b.classification.clone()),
        bm.map_or(String::new(), |b| f(b.shape.bimodality_index, 3)),
        b.and_then(|b| b.dominant_band).unwrap_or("none").to_string(),
        f(ms, 1),
    ]
}

pub const BATCH_HEAD: &str = "subject,signal,samples,rejected,hours,mean,tau_h,mesor,amplitude,acrophase_h,acrophase_hhmm,r2,p,harmonic_r2,tau2_harmonic_amp,tau2_harmonic_p,rhythm12_period_h,rhythm12_p_sidak,bimodality,bimodality_index,dominant_band,analysis_ms,error";

pub fn write_batch_csv(path: &std::path::Path, rows: &[(String, Result<Vec<String>, String>)]) -> Result<(), String> {
    let width = BATCH_HEAD.split(',').count();
    let mut s = String::from(BATCH_HEAD);
    s.push('\n');
    for (name, r) in rows {
        let cells: Vec<String> = match r {
            Ok(row) => std::iter::once(name.clone()).chain(row.iter().cloned()).chain(std::iter::once(String::new())).collect(),
            Err(e) => std::iter::once(name.clone()).chain(std::iter::repeat_n(String::new(), width - 2)).chain(std::iter::once(e.lines().next().unwrap_or("").to_string())).collect(),
        };
        s.push_str(&cells.iter().map(|c| csv_field(c)).collect::<Vec<_>>().join(","));
        s.push('\n');
    }
    std::fs::write(path, s).map_err(|e| e.to_string())
}

/// Every subject in every file of an archive / folder / multi-subject file, analysed in parallel.
fn run_batch(path: &std::path::Path, cfg: &pipeline::Config, opts: &import::ImportOptions, out: Option<PathBuf>) -> Result<(), String> {
    use rayon::prelude::*;
    let t0 = std::time::Instant::now();
    let sources: Vec<import::Source> = if path.is_dir() {
        let mut v = Vec::new();
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path).map_err(|e| e.to_string())?.flatten().map(|e| e.path()).filter(|p| p.is_file()).collect();
        entries.sort();
        for p in entries {
            match import::read_path(&p) {
                Ok(s) => v.extend(s),
                Err(e) => println!("  skipped {}: {e}", p.display()),
            }
        }
        v
    } else {
        import::read_path(path)?
    };
    let parsed: Vec<(String, Result<import::Parsed, String>)> = sources.par_iter().map(|s| (s.name.clone(), import::parse_source(s, opts, None))).collect();
    let read_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut subjects = Vec::new();
    let mut formats: Vec<String> = Vec::new();
    for (name, p) in &parsed {
        match p {
            Ok(p) => {
                if !formats.contains(&p.format) {
                    formats.push(p.format.clone());
                }
                subjects.extend(p.subjects.iter().cloned());
            }
            Err(e) => println!("  skipped {name}: {}", e.lines().next().unwrap_or("")),
        }
    }
    let results: Vec<(String, Result<Vec<String>, String>, f64)> = subjects
        .par_iter()
        .map(|d| {
            let t = std::time::Instant::now();
            let r = pipeline::run_analysis(d, cfg).map(|rep| batch_row(d, &rep, t.elapsed().as_secs_f64() * 1000.0));
            (d.name.clone(), r, t.elapsed().as_secs_f64() * 1000.0)
        })
        .collect();
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut lat: Vec<f64> = results.iter().filter(|r| r.1.is_ok()).map(|r| r.2).collect();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ok = lat.len();
    println!("=== CoCirca batch: {} ===", path.display());
    println!("Formats: {}", formats.join(", "));
    println!(
        "{} files → {} subjects | read+parse {:.0} ms | {} analysed, {} failed | wall {:.0} ms | per-subject analysis: median {:.1} ms, max {:.1} ms",
        parsed.len(),
        subjects.len(),
        read_ms,
        ok,
        results.len() - ok,
        total_ms,
        stats::median(&lat),
        lat.last().copied().unwrap_or(0.0)
    );
    for (name, r, _) in results.iter().filter(|r| r.1.is_err()).take(20) {
        println!("  failed {name}: {}", r.as_ref().err().unwrap().lines().next().unwrap_or(""));
    }
    if let Some(p) = out {
        let rows: Vec<(String, Result<Vec<String>, String>)> = results.into_iter().map(|(n, r, _)| (n, r)).collect();
        write_batch_csv(&p, &rows)?;
        println!("Summary: {}", p.display());
    }
    Ok(())
}

fn num<T: std::str::FromStr>(name: &str, v: Option<&String>) -> Result<T, String> {
    v.ok_or(format!("{name} needs a value"))?.parse().map_err(|_| format!("{name}: invalid value"))
}

pub fn run(args: &[String]) -> Result<(), String> {
    let mut files = Vec::new();
    let mut cfg = pipeline::Config::default();
    let mut excel: Option<PathBuf> = None;
    let mut csv: Option<PathBuf> = None;
    let mut demo = false;
    let mut batch: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut start: Option<i64> = None;
    let mut iopts = import::ImportOptions::default();
    let mut subject: usize = 0;
    let mut merge = import::MergeMode::Auto;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--demo" => demo = true,
            "--batch" => batch = Some(it.next().ok_or("--batch needs an archive or folder")?.into()),
            "--out" => out = Some(it.next().ok_or("--out needs a path")?.into()),
            "--start" => start = Some(parse_start(it.next().ok_or("--start needs a value")?)?),
            "--bin-sec" => cfg.epoch_sec = data::epoch_seconds(num::<f64>(a, it.next())? / 60.0),
            "--harmonics" => cfg.n_harmonics = num(a, it.next())?,
            "--period-min" => cfg.period_min = num(a, it.next())?,
            "--period-max" => cfg.period_max = num(a, it.next())?,
            "--circadian-min" => cfg.circ_min = num(a, it.next())?,
            "--circadian-max" => cfg.circ_max = num(a, it.next())?,
            "--split-day" => cfg.split_day = Some(num(a, it.next())?),
            "--tau" => {
                let v = it.next().ok_or("--tau needs a value")?;
                cfg.period_mode = match v.as_str() {
                    "fixed24" => pipeline::PeriodMode::Fixed24,
                    "ls" => pipeline::PeriodMode::LombScargle,
                    "chi2" => pipeline::PeriodMode::Chi2,
                    "cosinor" => pipeline::PeriodMode::Cosinor,
                    "consensus" => pipeline::PeriodMode::Consensus,
                    h => {
                        cfg.custom_tau = h.parse().map_err(|_| format!("--tau: unknown value {h}"))?;
                        pipeline::PeriodMode::Custom
                    }
                }
            }
            "--onsets" => {
                cfg.onset_method = match it.next().map(|s| s.as_str()) {
                    Some("threshold") => crate::nonparam::OnsetMethod::Threshold,
                    Some("cosinor") => crate::nonparam::OnsetMethod::Cosinor,
                    _ => crate::nonparam::OnsetMethod::Template,
                }
            }
            "--excel" => excel = Some(it.next().ok_or("--excel needs a path")?.into()),
            "--csv" => csv = Some(it.next().ok_or("--csv needs a path")?.into()),
            "--date-order" => {
                iopts.date_order = match it.next().map(|s| s.to_lowercase()).as_deref() {
                    Some("dmy") | Some("day") => import::DateOrder::DayFirst,
                    Some("mdy") | Some("month") => import::DateOrder::MonthFirst,
                    Some("ymd") | Some("year") => import::DateOrder::YearFirst,
                    _ => import::DateOrder::Auto,
                }
            }
            "--time-unit" => {
                iopts.time_unit = match it.next().map(|s| s.to_lowercase()).as_deref() {
                    Some("ms") => Some(import::TimeUnit::Milliseconds),
                    Some("s") => Some(import::TimeUnit::Seconds),
                    Some("min") => Some(import::TimeUnit::Minutes),
                    Some("h") => Some(import::TimeUnit::Hours),
                    Some("d") => Some(import::TimeUnit::Days),
                    _ => None,
                }
            }
            "--interval" => iopts.interval_s = Some(num(a, it.next())?),
            "--header-line" => iopts.header_line = Some(num::<usize>(a, it.next())?.saturating_sub(1)),
            "--layout" => {
                iopts.layout = match it.next().map(|s| s.as_str()) {
                    Some("subjects") => import::Layout::Subjects,
                    Some("channels") => import::Layout::Channels,
                    _ => import::Layout::Auto,
                }
            }
            "--subject" => subject = num::<usize>(a, it.next())?.saturating_sub(1),
            "--merge" => merge = import::MergeMode::Merge,
            "--separate" => merge = import::MergeMode::Separate,
            f if f.starts_with("--") => return Err(format!("unknown option {f}\n{USAGE}")),
            f => files.push(PathBuf::from(f)),
        }
    }
    iopts.start = start;
    if let Some(b) = batch {
        return run_batch(&b, &cfg, &iopts, out);
    }
    let t_load = std::time::Instant::now();
    let (ds, n_subjects, import_notes) = if demo || files.is_empty() {
        (data::synthetic(14, 10.0, Some(8), 3.5), 1, Vec::new())
    } else {
        let sources: Vec<import::Source> = files.iter().map(|f| import::read_path(f)).collect::<Result<Vec<_>, _>>()?.into_iter().flatten().collect();
        let parsed: Vec<import::Parsed> = sources.iter().map(|s| import::parse_source(s, &iopts, None)).collect::<Result<Vec<_>, _>>()?;
        let res = import::combine(parsed, merge);
        let n = res.subjects.len();
        let ds = res.subjects.into_iter().nth(subject).ok_or(format!("--subject {}: the input has {n} subject(s)", subject + 1))?;
        (ds, n, res.notes)
    };
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    let rep = pipeline::run_analysis(&ds, &cfg)?;
    println!("=== CoCirca analysis ({}) ===", rep.source);
    if !ds.format.is_empty() {
        println!("Format: {} | channels: {}", ds.format, ds.describe_channels().join(", "));
    }
    if n_subjects > 1 {
        println!("Subject {} of {n_subjects} (choose with --subject N, or analyse all with --batch)", subject + 1);
    }
    for n in &import_notes {
        println!("Note: {n}");
    }
    println!(
        "Samples {} | bins {} x {} | {:.2} days | load {:.1} ms | analysis {:.1} ms",
        ds.len(),
        rep.binned.len(),
        data::format_bin_size(cfg.epoch_sec),
        rep.binned.n_days,
        load_ms,
        rep.elapsed_ms
    );
    for w in &rep.warnings {
        println!("Warning: {w}");
    }
    for ch in rep.channels() {
        let pg = &ch.periodogram;
        let s = &ch.single;
        let h = &ch.harmonics;
        println!("--- {} ({}) ---", ch.name, ch.unit);
        println!(
            "tau used {:.2} h | consensus {:.2} | LS {:.2} ({:.4}) | chi2 {:.2} (Qp {:.1}, p {:.2e})",
            ch.tau, pg.consensus_tau, pg.best_tau_ls, pg.best_ls, pg.best_tau_chi2, pg.best_qp, pg.best_chi2_p
        );
        println!("Single cosinor: MESOR {:.4} Amp {:.4} Acrophase {} ({:.1} deg) R2 {:.4} p {:.2e}", s.mesor, s.amplitude(), s.acrophase_hhmm(), s.deg_neg(), s.r2, s.p);
        println!("{}-harmonic: R2 {:.4} orthophase {} bathyphase {}", h.nh, h.r2, stats::hhmm(h.ortho_h), stats::hhmm(h.comp_bathy_h));
        if let Some(c) = &ch.circ_acrophase {
            println!("Circular acrophase: mean {} Rbar {:.3} CSD {:.2} h Rayleigh p {:.2e}", c.mean_hhmm(), c.r_bar, c.circ_std_hours, c.rayleigh_p);
        }
        if let Some(sh) = &ch.shift {
            if let Some(t) = &sh.acrophase {
                println!("Acrophase shift: {} -> {} | {:+.2} h ({}) WW p {:.3e}", stats::hhmm(t.mean1), stats::hhmm(t.mean2), t.shift_h, t.direction, t.p);
            }
        }
        if let Some(np) = &ch.non_param {
            println!("IS {:.3} IV {:.3} RA {:.3} M10 {} L5 {}", np.is, np.iv, np.ra, stats::hhmm(np.m10_onset), stats::hhmm(np.l5_onset));
        }
        if let Some(b) = &ch.bands {
            println!(
                "Rhythm bands ({}-{} h): dominant {}",
                b.period_min,
                b.period_max,
                b.dominant_band.map_or("none".into(), |d| format!("{d} {:.1} h", b.dominant_period))
            );
            for band in &b.bands {
                let detail = if band.detected { format!("  [{:.1} h, {:.1}% var, p(Sidak)={:.1e}]", band.period, band.percent_variance, band.p_adjusted) } else { String::new() };
                println!("  {:<9} ({} h): {}{}", band.band, band.range_label(), band.classification, detail);
            }
        }
        if let Some(t) = &ch.twelve {
            println!("12 h rhythm: {:.2} h amp {:.3} peaks {} {:.2}% p(Sidak) {:.2e}", t.rhythm_period, t.rhythm_amp, t.rhythm_peaks, t.rhythm_pct, t.rhythm_p_adj);
            println!(
                "tau/2 harmonic: {:.2} h amp {:.3} peaks {} {:.2}% p {:.2e} A2/A1 {:.2} -> {}",
                t.harm_period,
                t.harm_amp,
                t.harm_peaks,
                t.harm_pct,
                t.harm_p,
                t.ratio,
                if t.distinct { "distinct" } else { "indistinguishable" }
            );
        }
        if let Some(bm) = &ch.bimodality {
            println!(
                "Bimodality: {} (BI {:.2}, peaks {} & {}, A2/A1 {:.2}, 12h p {:.2e}; bimodal days {}/{})",
                bm.classification,
                bm.shape.bimodality_index,
                stats::hhmm(bm.shape.primary_h),
                stats::hhmm(bm.shape.secondary_h),
                bm.h2_h1,
                bm.h2_p,
                bm.days_bimodal,
                bm.days_tested
            );
        }
        println!("Daily rows: {} | onsets found: {}", ch.daily.len(), ch.daily.iter().filter(|d| d.onset_h.is_finite()).count());
    }
    let theme = theme::ThemeSettings::default().build();
    if let Some(p) = excel {
        let t = std::time::Instant::now();
        export::export_excel(&rep, &theme, &p)?;
        println!("Excel: {} ({:.0} ms)", p.display(), t.elapsed().as_secs_f64() * 1000.0);
    }
    if let Some(p) = csv {
        let t = std::time::Instant::now();
        export::export_csv(&rep, &p)?;
        println!("CSV: {} ({:.0} ms)", p.display(), t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(())
}
