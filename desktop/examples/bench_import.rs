//! Latency benchmark: import (read, 5,000-row preview, full parse, merge) and analysis on real files.
//!
//!     cargo run --release --example bench_import -- [--bins 600,60,1] [--export DIR] [--no-merge] FILES...
//!
//! `scripts/fetch_big_ideas.sh` downloads a 1 GB Empatica E4 recording (PhysioNet BIG IDEAs, participant 001)
//! to test with.
use cocirca::{data, export, import, pipeline, theme};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(windows)]
mod win {
    #[repr(C)]
    #[allow(dead_code)]
    #[derive(Default)]
    pub struct Pmc { pub cb: u32, pub faults: u32, pub peak_ws: usize, pub ws: usize, pub a: usize, pub b: usize, pub c: usize, pub d: usize, pub pagefile: usize, pub peak_pagefile: usize }
    #[link(name = "kernel32")]
    extern "system" {
        pub fn GetCurrentProcess() -> isize;
        pub fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    }
}
/// (current, peak) working set in GB (Windows only; 0 elsewhere)
fn mem() -> (f64, f64) {
    #[cfg(windows)]
    {
        let mut p = win::Pmc { cb: std::mem::size_of::<win::Pmc>() as u32, ..Default::default() };
        unsafe { win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut p, p.cb) };
        (p.ws as f64 / 1e9, p.peak_ws as f64 / 1e9)
    }
    #[cfg(not(windows))]
    (0.0, 0.0)
}
fn ms(t: Instant) -> f64 { t.elapsed().as_secs_f64() * 1000.0 }
fn fmt_rows(n: usize) -> String {
    let s = n.to_string();
    let mut o = String::new();
    for (i, c) in s.chars().enumerate() { if i > 0 && (s.len() - i) % 3 == 0 { o.push(','); } o.push(c); }
    o
}

fn main() {
    let mut bins = vec![600.0, 60.0];
    let mut export_dir: Option<PathBuf> = None;
    let mut merge = import::MergeMode::Auto;
    let mut files = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bins" => bins = it.next().unwrap().split(',').map(|s| s.parse().unwrap()).collect(),
            "--export" => export_dir = Some(PathBuf::from(it.next().unwrap())),
            "--no-merge" => merge = import::MergeMode::Separate,
            f => files.push(PathBuf::from(f)),
        }
    }
    let opts = import::ImportOptions::default();
    let t_all = Instant::now();
    let mut parsed = Vec::new();
    println!("{:<14} {:>9} {:>9} {:>10} {:>10} {:>12}  format / channels", "file", "MB", "read ms", "preview ms", "parse ms", "rows");
    for f in &files {
        let t = Instant::now();
        let srcs = import::read_path(f).unwrap_or_else(|e| panic!("{}: {e}", f.display()));
        let read = ms(t);
        for s in srcs {
            let mb = s.bytes.len() as f64 / 1e6;
            let t = Instant::now();
            let pv = import::parse_source(&s, &opts, Some(5000));
            let prev = ms(t);
            let t = Instant::now();
            let p = import::parse_source(&s, &opts, None).unwrap_or_else(|e| panic!("{}: {e}", s.name));
            let full = ms(t);
            let chans: Vec<String> = p.subjects.iter().flat_map(|d| d.describe_channels()).collect();
            println!("{:<14} {:>9.1} {:>9.0} {:>10.1} {:>10.0} {:>12}  {} | {}", Path::new(&s.name).file_name().unwrap().to_string_lossy(), mb, read, prev, full, fmt_rows(p.rows), p.format, chans.join(", "));
            if let Ok(pv) = &pv { println!("{:<14} preview: {} | time: {}", "", pv.summary(), pv.time_desc); }
            for n in &p.notes { println!("{:<14} note: {n}", ""); }
            if p.skipped_rows > 0 { println!("{:<14} skipped rows: {}", "", p.skipped_rows); }
            parsed.push(p);
        }
    }
    let (_, peak) = mem();
    println!("after parse: peak working set {:.2} GB", peak);
    let t = Instant::now();
    let res = import::combine(parsed, merge);
    println!("combine: {:.0} ms -> {} subject(s); peak {:.2} GB", ms(t), res.subjects.len(), mem().1);
    for n in &res.notes { println!("  note: {n}"); }
    println!("import total (read+parse+combine): {:.2} s", t_all.elapsed().as_secs_f64());
    // GUI-style import: what the app does after "Import" (all files parsed in parallel, then combined)
    {
        use rayon::prelude::*;
        let t = Instant::now();
        let srcs: Vec<import::Source> = files.iter().flat_map(|f| import::read_path(f).unwrap()).collect();
        let read = ms(t);
        let t1 = Instant::now();
        let parsed: Vec<import::Parsed> = srcs.par_iter().map(|s| import::parse_source(s, &opts, None).unwrap()).collect();
        let par = ms(t1);
        let t2 = Instant::now();
        let r = import::combine(parsed, merge);
        println!("GUI-style import: read {read:.0} ms + parallel parse {par:.0} ms + combine {:.0} ms = {:.2} s -> {} subject(s)", ms(t2), t.elapsed().as_secs_f64(), r.subjects.len());
    }
    let th = theme::ThemeSettings::default().build();
    for d in &res.subjects {
        let (t0, t1) = d.time_range().unwrap_or((0, 0));
        println!("\n=== {} | {} samples | {:.2} days | {}", d.name, fmt_rows(d.len()), (t1 - t0) as f64 / 86_400_000.0, d.describe_channels().join(", "));
        for &b in &bins {
            let mut cfg = pipeline::Config::default();
            cfg.epoch_sec = data::epoch_seconds(b / 60.0);
            let t = Instant::now();
            match pipeline::run_analysis(d, &cfg) {
                Err(e) => println!("  bin {b:>5} s: error {e}"),
                Ok(rep) => {
                    let wall = ms(t);
                    println!("  bin {:>5} s: {:>9} bins | wall {:>8.1} ms (binning + analysis)", b, fmt_rows(rep.binned.len()), wall);
                    for ch in rep.channels() {
                        let s = &ch.single;
                        println!("      {:<22} tau {:>6.2} h  MESOR {:>10.3}  amp {:>10.3}  acro {}  R2 {:.3}", format!("{} ({})", ch.name, ch.unit), ch.tau, s.mesor, s.amplitude(), s.acrophase_hhmm(), s.r2);
                    }
                    for w in &rep.warnings { println!("      warning: {w}"); }
                    if let Some(dir) = &export_dir {
                        std::fs::create_dir_all(dir).ok();
                        let t = Instant::now();
                        export::export_csv(&rep, &dir.join(format!("epochs_{b}s.csv"))).unwrap();
                        let c = ms(t);
                        let t = Instant::now();
                        export::export_excel(&rep, &th, &dir.join(format!("report_{b}s.xlsx"))).unwrap();
                        println!("      export: CSV {:.0} ms, Excel {:.0} ms", c, ms(t));
                    }
                }
            }
        }
    }
    println!("\npeak working set {:.2} GB | total {:.2} s", mem().1, t_all.elapsed().as_secs_f64());
}
