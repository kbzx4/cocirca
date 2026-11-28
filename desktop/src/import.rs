//! File import for chronobiology data: format sniffing and regex-driven parsing.
//!
//! Recognised formats (besides any delimited table with a time column):
//! * Actigraphy: ActiGraph ActiLife CSV, Philips Respironics Actiware, Condor ActTrust, CamNtech / Actiwatch
//!   `.awd`, GENEActiv CSV, Axivity CSV, MotionWare-style exports
//! * Locomotor activity: TriKinetics DAM monitor files (32 flies per file), wheel / beam-break counts
//! * Loggers: HOBO (light, temperature), iButton / Thermochron temperature
//! * Heart: Polar exports, RR / IBI / NN interval files, Empatica E4 (ACC, TEMP, HR, EDA, IBI)
//! * Phones & wearables: Phyphox, Sensor Logger, Apple Health `export.xml`, Fitbit JSON
//! * Wide tables: bioluminescence / plate-reader columns (one subject per column) and gene-expression
//!   time courses with ZT / CT column headers (one subject per row, JTK_CYCLE / MetaCycle layout)
//! * Containers & encodings: .zip, .gz, .xlsx / .xls / .ods, .json / .jsonl, .xml; UTF-8, UTF-16, Windows-1252
//!
//! Timestamps: ISO 8601, day- or month-first dates (auto-detected), month names, AM/PM, separate date and
//! time columns, Unix s / ms / µs / ns, Excel serial dates, MATLAB datenums, relative seconds / minutes /
//! hours / days, ZT / CT hours, elapsed hh:mm:ss, and fixed-interval files without time stamps.

use crate::data::{ms_to_naive, naive_to_ms, rr_dataset, today_midnight_ms, Channel, RawDataset, Role, MS_PER_DAY};
use chrono::{Local, NaiveDate, TimeZone};
use rayon::prelude::*;
use regex::Regex;
use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::OnceLock;

// ================================================================================================
// Public types
// ================================================================================================

#[derive(Clone, Debug)]
pub struct Source {
    pub name: String,
    pub bytes: Vec<u8>,
    /// Start time found elsewhere in the container (Phyphox meta/time.csv)
    pub start_hint: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DateOrder {
    #[default]
    Auto,
    DayFirst,
    MonthFirst,
    YearFirst,
}

impl DateOrder {
    pub const ALL: [DateOrder; 4] = [DateOrder::Auto, DateOrder::DayFirst, DateOrder::MonthFirst, DateOrder::YearFirst];
    pub fn label(self) -> &'static str {
        match self {
            DateOrder::Auto => "Auto-detect",
            DateOrder::DayFirst => "Day first (31/12/2024)",
            DateOrder::MonthFirst => "Month first (12/31/2024)",
            DateOrder::YearFirst => "Year first (2024-12-31)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimeUnit {
    Milliseconds,
    Seconds,
    Minutes,
    Hours,
    Days,
}

impl TimeUnit {
    pub const ALL: [TimeUnit; 5] = [TimeUnit::Milliseconds, TimeUnit::Seconds, TimeUnit::Minutes, TimeUnit::Hours, TimeUnit::Days];
    pub fn label(self) -> &'static str {
        match self {
            TimeUnit::Milliseconds => "milliseconds",
            TimeUnit::Seconds => "seconds",
            TimeUnit::Minutes => "minutes",
            TimeUnit::Hours => "hours",
            TimeUnit::Days => "days",
        }
    }
    pub fn ms(self) -> f64 {
        match self {
            TimeUnit::Milliseconds => 1.0,
            TimeUnit::Seconds => 1e3,
            TimeUnit::Minutes => 6e4,
            TimeUnit::Hours => 3.6e6,
            TimeUnit::Days => 8.64e7,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Layout {
    #[default]
    Auto,
    /// Columns are different measurements of one recording
    Channels,
    /// Columns are different subjects / samples (plate reader, LumiCycle dishes, cages)
    Subjects,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MergeMode {
    #[default]
    Auto,
    Merge,
    Separate,
}

impl MergeMode {
    pub const ALL: [MergeMode; 3] = [MergeMode::Auto, MergeMode::Merge, MergeMode::Separate];
    pub fn label(self) -> &'static str {
        match self {
            MergeMode::Auto => "Auto (merge different sensors / consecutive files, keep people apart)",
            MergeMode::Merge => "Merge into one recording",
            MergeMode::Separate => "Keep files as separate subjects",
        }
    }
}

/// What a column is used for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColRole {
    Ignore,
    /// Date and time, or the only time column
    Time,
    /// Date part when date and clock time are in separate columns
    Date,
    Data(Role),
    /// RR / inter-beat interval (converted to heart rate)
    RR,
    /// Non-zero / "EXCLUDED" / "off-wrist" marks invalid samples
    Mask,
}

impl ColRole {
    pub fn label(self) -> String {
        match self {
            ColRole::Ignore => "— ignore —".into(),
            ColRole::Time => "Time / timestamp".into(),
            ColRole::Date => "Date (separate column)".into(),
            ColRole::Data(r) => r.label().into(),
            ColRole::RR => "RR / IBI interval → heart rate".into(),
            ColRole::Mask => "Invalid-data flag (off-wrist)".into(),
        }
    }
    pub fn choices() -> Vec<ColRole> {
        let mut v = vec![ColRole::Ignore, ColRole::Time, ColRole::Date];
        v.extend(Role::ALL.iter().map(|&r| ColRole::Data(r)));
        v.extend([ColRole::RR, ColRole::Mask]);
        v
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportOptions {
    pub date_order: DateOrder,
    /// Start for relative / elapsed times (and files without time stamps)
    pub start: Option<i64>,
    /// Unit of a numeric relative time column
    pub time_unit: Option<TimeUnit>,
    /// Sampling interval for files without time stamps
    pub interval_s: Option<f64>,
    /// 0-based line holding the column names (data follow it)
    pub header_line: Option<usize>,
    pub roles: BTreeMap<usize, ColRole>,
    pub layout: Layout,
}

#[derive(Clone, Debug)]
pub struct ColumnInfo {
    pub name: String,
    pub role: ColRole,
    pub example: String,
}

/// Result of parsing one file: the data plus everything the import dialog shows.
#[derive(Clone, Debug, Default)]
pub struct Parsed {
    pub source: String,
    pub format: String,
    pub notes: Vec<String>,
    pub subjects: Vec<RawDataset>,
    pub preview: Vec<String>,
    pub header_line: Option<usize>,
    pub data_line: usize,
    pub columns: Vec<ColumnInfo>,
    pub time_desc: String,
    pub relative_time: bool,
    pub numeric_time: bool,
    pub needs_interval: bool,
    pub date_ambiguous: bool,
    pub can_layout: bool,
    pub layout: Layout,
    pub rows: usize,
    pub skipped_rows: usize,
    pub truncated: bool,
}

impl Parsed {
    /// "1 subject · 80,265 rows · 21.1 h · every 0.79 s"
    pub fn summary(&self) -> String {
        let n = self.subjects.len();
        let (mut t0, mut t1) = (i64::MAX, i64::MIN);
        for d in &self.subjects {
            if let Some((a, b)) = d.time_range() {
                t0 = t0.min(a);
                t1 = t1.max(b);
            }
        }
        let dur = if t1 > t0 { format_duration((t1 - t0) as f64 / 1000.0) } else { "—".into() };
        let step = self.subjects.first().map(median_step_s).filter(|s| s.is_finite()).map(|s| format!(" · every {}", format_duration(s))).unwrap_or_default();
        format!(
            "{} subject{} · {} rows{} · {}{}",
            n,
            if n == 1 { "" } else { "s" },
            group_thousands(self.rows),
            if self.truncated { " (preview)" } else { "" },
            dur,
            step
        )
    }
}

pub struct ImportResult {
    pub subjects: Vec<RawDataset>,
    pub notes: Vec<String>,
    pub formats: Vec<String>,
}

pub fn group_thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

pub fn format_duration(s: f64) -> String {
    if s < 1.0 {
        format!("{:.0} ms", s * 1000.0)
    } else if s < 120.0 {
        let v = format!("{s:.3}");
        format!("{} s", v.trim_end_matches('0').trim_end_matches('.'))
    } else if s < 7200.0 {
        format!("{:.1} min", s / 60.0)
    } else if s < 3.0 * 86400.0 {
        format!("{:.1} h", s / 3600.0)
    } else {
        format!("{:.1} days", s / 86400.0)
    }
}

fn median_step_s(d: &RawDataset) -> f64 {
    let n = d.t_ms.len().min(5001);
    if n < 2 {
        return f64::NAN;
    }
    let diffs: Vec<f64> = d.t_ms[..n].windows(2).map(|w| (w[1] - w[0]) as f64 / 1000.0).filter(|&x| x > 0.0).collect();
    crate::stats::median(&diffs)
}

// ================================================================================================
// Reading files and containers
// ================================================================================================

const BINARY_HINTS: [(&str, &str); 6] = [
    (".gt3x", "ActiGraph .gt3x is a binary format: export a CSV (raw or epoch) from ActiLife, or use the .agd → CSV export."),
    (".agd", "ActiGraph .agd is a database: in ActiLife use File → Export → CSV, then open the CSV."),
    (".cwa", "Axivity .cwa is binary: export CSV with OmGui (Export Raw CSV), then open the CSV."),
    (".bin", "GENEActiv .bin: convert with GENEActiv PC software (Convert → CSV), then open the CSV."),
    (".mtn", "MotionWatch .mtn: export CSV (or .awd) from MotionWare, then open it."),
    (".edf", "EDF recordings: export the channel of interest to CSV (e.g. with EDFbrowser), then open it."),
];

/// Read a path; .zip archives and .gz files are expanded into their members.
pub fn read_path(path: &Path) -> Result<Vec<Source>, String> {
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "data".into());
    let lower = name.to_lowercase();
    if let Some((_, hint)) = BINARY_HINTS.iter().find(|(e, _)| lower.ends_with(e)) {
        return Err(format!("{name}: {hint}"));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
    expand(Source { name, bytes, start_hint: None }, 0)
}

fn skip_member(name: &str) -> bool {
    static RX: OnceLock<Regex> = OnceLock::new();
    let rx = RX.get_or_init(|| Regex::new(r"(?i)(^|/)(__macosx/|\.|license|licence|readme|sha\d*sums|md5sums|checksums?)|\.(pdf|md|html?|png|jpe?g|gif|docx?|pptx?|hea|dat\.hea|log)$").unwrap());
    rx.is_match(name) || name.ends_with('/')
}

pub fn expand(src: Source, depth: usize) -> Result<Vec<Source>, String> {
    let lower = src.name.to_lowercase();
    if depth > 3 {
        return Ok(vec![src]);
    }
    if lower.ends_with(".gz") && src.bytes.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&src.bytes[..]).read_to_end(&mut out).map_err(|e| format!("{}: {e}", src.name))?;
        let inner = src.name[..src.name.len() - 3].to_string();
        return expand(Source { name: inner, bytes: out, start_hint: src.start_hint }, depth + 1);
    }
    if lower.ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(Cursor::new(src.bytes)).map_err(|e| format!("{}: {e}", src.name))?;
        let mut members = Vec::new();
        let mut phyphox_meta = None;
        for i in 0..archive.len() {
            let mut f = archive.by_index(i).map_err(|e| e.to_string())?;
            let name = f.name().to_string();
            if f.is_dir() {
                continue;
            }
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).map_err(|e| e.to_string())?;
            let l = name.to_lowercase();
            if l.ends_with("meta/time.csv") || l == "time.csv" {
                phyphox_meta = phyphox_start(&buf);
                continue;
            }
            if l.contains("meta/") || skip_member(&name) {
                continue;
            }
            let short = name.rsplit('/').next().unwrap_or(&name).to_string();
            members.push(Source { name: short, bytes: buf, start_hint: src.start_hint });
        }
        let mut out = Vec::new();
        for mut m in members {
            if m.start_hint.is_none() {
                m.start_hint = phyphox_meta;
            }
            out.extend(expand(m, depth + 1)?);
        }
        if out.is_empty() {
            return Err(format!("{}: the archive contains no data files.", src.name));
        }
        return Ok(out);
    }
    Ok(vec![src])
}

/// Phyphox meta/time.csv → recording start (local time).
fn phyphox_start(bytes: &[u8]) -> Option<i64> {
    let text = decode(bytes);
    let mut lines = text.lines();
    let header: Vec<String> = lines.next()?.split([',', ';', '\t']).map(|s| s.trim().trim_matches('"').to_lowercase()).collect();
    let col = |pat: &str| header.iter().position(|h| h.contains(pat));
    let (ev, sys, exp) = (col("event"), header.iter().position(|h| h.starts_with("system time") && !h.contains("text")), col("experiment time"));
    for line in lines {
        let f: Vec<&str> = line.split([',', ';', '\t']).map(|s| s.trim().trim_matches('"')).collect();
        if ev.is_none_or(|e| f.get(e).is_some_and(|s| s.to_uppercase().contains("START"))) {
            let sys_v: f64 = f.get(sys?)?.parse().ok()?;
            let exp_v: f64 = exp.and_then(|e| f.get(e)?.parse().ok()).unwrap_or(0.0);
            return Some(unix_ms_to_local((sys_v - exp_v) * 1000.0));
        }
    }
    None
}

/// Text decoding: UTF-8 (with or without BOM), UTF-16 LE/BE, otherwise Windows-1252.
pub fn decode(b: &[u8]) -> String {
    if b.starts_with(&[0xEF, 0xBB, 0xBF]) {
        return String::from_utf8_lossy(&b[3..]).into_owned();
    }
    let utf16 = |le: bool, body: &[u8]| {
        let units: Vec<u16> = body.chunks_exact(2).map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) }).collect();
        String::from_utf16_lossy(&units)
    };
    if b.starts_with(&[0xFF, 0xFE]) {
        return utf16(true, &b[2..]);
    }
    if b.starts_with(&[0xFE, 0xFF]) {
        return utf16(false, &b[2..]);
    }
    // BOM-less UTF-16: many NUL bytes in alternating positions
    let probe = &b[..b.len().min(4096)];
    if probe.len() >= 8 {
        let odd_nul = probe.iter().skip(1).step_by(2).filter(|&&x| x == 0).count();
        let even_nul = probe.iter().step_by(2).filter(|&&x| x == 0).count();
        if odd_nul * 3 > probe.len() / 2 && even_nul * 10 < probe.len() / 2 {
            return utf16(true, b);
        }
        if even_nul * 3 > probe.len() / 2 && odd_nul * 10 < probe.len() / 2 {
            return utf16(false, b);
        }
    }
    match std::str::from_utf8(b) {
        Ok(s) => s.to_string(),
        Err(_) => b.iter().map(|&c| cp1252(c)).collect(),
    }
}

fn cp1252(c: u8) -> char {
    const HI: [u16; 32] = [
        0x20AC, 0x81, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x8D, 0x017D, 0x8F, 0x90, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x9D, 0x017E, 0x0178,
    ];
    if (0x80..0xA0).contains(&c) {
        char::from_u32(HI[(c - 0x80) as usize] as u32).unwrap_or('?')
    } else {
        c as char
    }
}

// ================================================================================================
// Date / time parsing (byte-level tokenizer: ~50–100 ns per value)
// ================================================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dmy {
    Dmy,
    Mdy,
    Ymd,
}

#[derive(Clone, Copy, Default)]
struct Num {
    v: u64,
    d: u8,
    sep: u8,
}

#[derive(Default)]
struct Tok {
    nums: [Num; 10],
    n: usize,
    month: Option<u32>,
    month_at: usize,
    pm: Option<bool>,
}

fn month_of(w: &str) -> Option<u32> {
    let w = w.to_lowercase();
    let k3: String = w.chars().take(3).collect();
    let k = k3.as_str();
    Some(match k {
        "jan" | "ene" | "gen" => 1,
        "feb" | "fev" | "fév" => 2,
        "mar" | "mär" | "mrz" => 3,
        "apr" | "abr" | "avr" => 4,
        "may" | "mai" | "mag" => 5,
        "jun" | "jui" if w.starts_with("juin") || k == "jun" => 6,
        "jul" | "lug" => 7,
        "aug" | "ago" | "aoû" => 8,
        "sep" | "set" => 9,
        "oct" | "out" | "okt" | "ott" => 10,
        "nov" => 11,
        "dec" | "dez" | "dic" | "déc" => 12,
        _ if w.starts_with("juil") => 7,
        _ => return None,
    })
}

fn tokenize(s: &str) -> Tok {
    let b = s.as_bytes();
    let mut t = Tok::default();
    let mut i = 0;
    let mut sep = b' ';
    while i < b.len() && t.n < 10 {
        let c = b[i];
        if c.is_ascii_digit() {
            let st = i;
            let mut v: u64 = 0;
            while i < b.len() && b[i].is_ascii_digit() {
                if i - st < 18 {
                    v = v * 10 + (b[i] - b'0') as u64;
                }
                i += 1;
            }
            t.nums[t.n] = Num { v, d: (i - st).min(255) as u8, sep };
            t.n += 1;
            sep = 0;
        } else if c.is_ascii_alphabetic() || c >= 0x80 {
            let st = i;
            while i < b.len() && (b[i].is_ascii_alphabetic() || b[i] >= 0x80) {
                i += 1;
            }
            let w = &s[st..i];
            let lw = w.to_ascii_lowercase();
            match lw.as_str() {
                "am" | "a.m" | "a" => t.pm = Some(false),
                "pm" | "p.m" | "p" => t.pm = Some(true),
                "t" => {}
                "z" | "utc" | "gmt" => break,
                _ => {
                    if let Some(m) = month_of(&lw) {
                        if t.month.is_none() {
                            t.month = Some(m);
                            t.month_at = t.n;
                        }
                    }
                }
            }
            sep = b' ';
        } else if (c == b'+' || c == b'-') && t.n >= 5 {
            break; // timezone offset: keep the local wall-clock time as written
        } else {
            sep = c;
            i += 1;
        }
    }
    t
}

fn year_of(n: Num) -> i32 {
    let v = n.v as i32;
    if n.d <= 2 {
        if v < 70 {
            2000 + v
        } else {
            1900 + v
        }
    } else {
        v
    }
}

/// (date as (y, m, d) if present, time-of-day ms if present)
fn interpret(t: &Tok, order: Dmy) -> (Option<(i32, u32, u32)>, Option<i64>) {
    let nums = &t.nums[..t.n];
    // time part starts at the first number followed by a ':'-separated number
    let k = (0..nums.len().saturating_sub(1)).find(|&k| nums[k + 1].sep == b':');
    let mut tod = None;
    if let Some(k) = k {
        let h = nums[k].v as i64;
        let m = nums[k + 1].v as i64;
        let mut idx = k + 2;
        let mut s = 0i64;
        if idx < nums.len() && nums[idx].sep == b':' && nums[idx].d <= 2 {
            s = nums[idx].v as i64;
            idx += 1;
        }
        let mut ms = 0i64;
        if idx < nums.len() && matches!(nums[idx].sep, b'.' | b',' | b':') {
            let f = nums[idx];
            ms = match f.d {
                1 => f.v as i64 * 100,
                2 => f.v as i64 * 10,
                3 => f.v as i64,
                d => (f.v as f64 / 10f64.powi(d as i32 - 3)).round() as i64,
            };
        }
        let mut h = h;
        match t.pm {
            Some(true) if h < 12 => h += 12,
            Some(false) if h == 12 => h = 0,
            _ => {}
        }
        if h <= 24 && m < 60 && s <= 60 {
            tod = Some(((h * 60 + m) * 60 + s) * 1000 + ms);
        }
    }
    let dn = &nums[..k.unwrap_or(nums.len())];
    let date = if let Some(mo) = t.month {
        match dn.len() {
            2 => {
                let (a, b) = (dn[0], dn[1]);
                let (day, yr) = if a.d == 4 { (b, a) } else if b.d == 4 { (a, b) } else if t.month_at == 0 { (a, b) } else { (a, b) };
                Some((year_of(yr), mo, day.v as u32))
            }
            _ => None,
        }
    } else if dn.len() >= 3 {
        let (a, b, c) = (dn[0], dn[1], dn[2]);
        if a.d == 4 {
            Some((a.v as i32, b.v as u32, c.v as u32))
        } else {
            match order {
                Dmy::Dmy => Some((year_of(c), b.v as u32, a.v as u32)),
                Dmy::Mdy => Some((year_of(c), a.v as u32, b.v as u32)),
                Dmy::Ymd => Some((year_of(a), b.v as u32, c.v as u32)),
            }
        }
    } else if dn.len() == 1 && dn[0].d == 8 {
        let v = dn[0].v;
        Some(((v / 10000) as i32, ((v / 100) % 100) as u32, (v % 100) as u32))
    } else {
        None
    };
    let date = date.filter(|&(y, m, d)| (1900..=2200).contains(&y) && (1..=12).contains(&m) && (1..=31).contains(&d));
    (date, tod)
}

fn days_from_civil(y: i32, m: u32, d: u32) -> Option<i64> {
    NaiveDate::from_ymd_opt(y, m, d).map(|nd| naive_to_ms(nd.and_hms_opt(0, 0, 0).unwrap()) / MS_PER_DAY)
}

/// Full date (time optional) → ms.
pub fn parse_dt(s: &str, order: Dmy) -> Option<i64> {
    let t = tokenize(s);
    let (date, tod) = interpret(&t, order);
    let (y, m, d) = date?;
    Some(days_from_civil(y, m, d)? * MS_PER_DAY + tod.unwrap_or(0))
}

/// Time of day only (no date) → ms since midnight.
pub fn parse_tod(s: &str) -> Option<i64> {
    let t = tokenize(s);
    let (date, tod) = interpret(&t, Dmy::Ymd);
    if date.is_some() || t.month.is_some() {
        return None;
    }
    tod
}

/// Decide day-first vs month-first from sample values. Returns (order, ambiguous).
fn detect_order(samples: &[&str]) -> (Dmy, bool) {
    let (mut dmy, mut mdy, mut needs) = (0, 0, 0);
    for s in samples {
        let t = tokenize(s);
        if t.month.is_some() || t.n < 3 || t.nums[0].d == 4 || t.nums[2].sep == b':' {
            continue;
        }
        needs += 1;
        if t.nums[0].v > 12 {
            dmy += 1;
        }
        if t.nums[1].v > 12 {
            mdy += 1;
        }
    }
    if needs == 0 {
        return (Dmy::Ymd, false);
    }
    if dmy > 0 && mdy == 0 {
        return (Dmy::Dmy, false);
    }
    if mdy > 0 && dmy == 0 {
        return (Dmy::Mdy, false);
    }
    // All values ≤ 12: pick the reading in which time runs forward and spans the shortest period
    // (10/02 → 11/02 → 12/02 is two days read day-first, but two months read month-first)
    let score = |o: Dmy| {
        let ts: Vec<i64> = samples.iter().filter_map(|s| parse_dt(s, o)).collect();
        let back = ts.windows(2).filter(|w| w[1] < w[0]).count();
        let span = ts.iter().max().copied().unwrap_or(0) - ts.iter().min().copied().unwrap_or(0);
        (back, span)
    };
    let (a, b) = (score(Dmy::Dmy), score(Dmy::Mdy));
    if a.0 < b.0 || (a.0 == b.0 && a.1 < b.1) {
        (Dmy::Dmy, a == b)
    } else {
        (Dmy::Mdy, a == b)
    }
}

fn unix_ms_to_local(ms: f64) -> i64 {
    let ms = ms.round() as i64;
    match Local.timestamp_millis_opt(ms).single() {
        Some(dt) => naive_to_ms(dt.naive_local()),
        None => ms,
    }
}

// ================================================================================================
// Numbers & fields
// ================================================================================================

#[inline]
fn clean(s: &str) -> &str {
    s.trim().trim_matches('"').trim_matches('\'').trim()
}

#[inline]
fn num(s: &str, comma: bool) -> f64 {
    let s = clean(s);
    if s.is_empty() {
        return f64::NAN;
    }
    if comma && s.contains(',') {
        return s.replace('.', "").replace(',', ".").parse().unwrap_or(f64::NAN);
    }
    match s.parse::<f64>() {
        Ok(v) => v,
        Err(_) => match s.to_ascii_lowercase().as_str() {
            "true" | "yes" => 1.0,
            "false" | "no" => 0.0,
            _ => f64::NAN,
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Delim {
    Tab,
    Semi,
    Comma,
    Pipe,
    Space,
}

impl Delim {
    fn ch(self) -> u8 {
        match self {
            Delim::Tab => b'\t',
            Delim::Semi => b';',
            Delim::Comma => b',',
            Delim::Pipe => b'|',
            Delim::Space => b' ',
        }
    }
}

fn split<'a>(line: &'a str, d: Delim, out: &mut Vec<&'a str>) {
    out.clear();
    if d == Delim::Space {
        out.extend(line.split_whitespace());
        return;
    }
    let ch = d.ch();
    if !line.as_bytes().contains(&b'"') {
        out.extend(line.split(ch as char));
        return;
    }
    let b = line.as_bytes();
    let (mut st, mut q) = (0, false);
    for (i, &c) in b.iter().enumerate() {
        if c == b'"' {
            q = !q;
        } else if c == ch && !q {
            out.push(&line[st..i]);
            st = i + 1;
        }
    }
    out.push(&line[st..]);
}

/// Starts like a number or date (digit, sign, decimal point) and contains only number / date characters
/// (month names and AM/PM allowed after the first digit). "ZT0", "A1", "Per2" are labels, not data.
fn looks_numberish(s: &str) -> bool {
    let b = clean(s).as_bytes();
    if b.is_empty() {
        return false;
    }
    let first_ok = b[0].is_ascii_digit() || (matches!(b[0], b'-' | b'+' | b'.') && b.len() > 1 && (b[1].is_ascii_digit() || b[1] == b'.'));
    first_ok && b.iter().all(|&c| c.is_ascii_digit() || b" .,:/-+".contains(&c) || c.is_ascii_alphabetic())
}

fn is_missing_token(s: &str) -> bool {
    matches!(clean(s).to_ascii_lowercase().as_str(), "" | "nan" | "na" | "n/a" | "null" | "none" | "-" | "--" | "inf" | "-inf")
}

fn data_like(fields: &[&str]) -> bool {
    let non_empty: Vec<&&str> = fields.iter().filter(|f| !is_missing_token(f)).collect();
    if non_empty.is_empty() {
        return false;
    }
    let ok = non_empty.iter().filter(|f| looks_numberish(f)).count();
    ok * 10 >= non_empty.len() * 6
}

struct Shape {
    delim: Delim,
    header: Option<usize>,
    data_start: usize,
    ncols: usize,
}

/// Find the data table among preamble lines: the delimiter giving the longest run of consistent,
/// data-like lines, and the column-name line just above it.
fn sniff(lines: &[&str], forced_header: Option<usize>) -> Option<Shape> {
    let mut buf = Vec::new();
    let mut best: Option<(usize, Delim, usize, usize)> = None; // (score, delim, start, ncols)
    for d in [Delim::Tab, Delim::Semi, Delim::Comma, Delim::Pipe, Delim::Space] {
        // Whitespace is only a fallback: dates like "26 Feb 18" would otherwise split into extra columns
        if d == Delim::Space && best.is_some() {
            break;
        }
        let mut info: Vec<(usize, bool)> = Vec::with_capacity(lines.len());
        for l in lines {
            if l.trim().is_empty() {
                info.push((0, false));
                continue;
            }
            split(l, d, &mut buf);
            info.push((buf.len(), data_like(&buf)));
        }
        if let Some(h) = forced_header {
            if h + 1 >= lines.len() {
                continue;
            }
            let c = info[(h + 1).min(info.len() - 1)].0;
            let score = info[h + 1..].iter().take_while(|x| x.0 == c || x.0 == 0).count();
            if c >= 2 && best.is_none_or(|b| score > b.0) {
                best = Some((score, d, h + 1, c));
            }
            continue;
        }
        let mut i = 0;
        while i < info.len() {
            let (c, dl) = info[i];
            if c < 2 || !dl {
                i += 1;
                continue;
            }
            let st = i;
            while i < info.len() && info[i].0 == c && info[i].1 {
                i += 1;
            }
            let run = i - st;
            let score = run * 4 + c.min(64);
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, d, st, c));
            }
        }
    }
    let (score, delim, start, ncols) = best?;
    if forced_header.is_none() && score < 4 * lines.len().min(3) {
        return None;
    }
    let header = match forced_header {
        Some(h) => Some(h),
        None => {
            let mut h = start;
            while h > 0 && lines[h - 1].trim().is_empty() {
                h -= 1;
            }
            if h == 0 {
                None
            } else {
                split(lines[h - 1], delim, &mut buf);
                (buf.len() + 1 >= ncols && buf.len() >= 2 && !data_like(&buf)).then_some(h - 1)
            }
        }
    };
    Some(Shape { delim, header, data_start: start, ncols })
}

// ================================================================================================
// Column-name catalogue
// ================================================================================================

struct Catalog {
    time_abs: Regex,
    time_rel: Regex,
    date_only: Regex,
    tod_only: Regex,
    ignore: Regex,
    mask: Regex,
    light: Regex,
    light_ignore: Regex,
    hr: Regex,
    rr: Regex,
    temp: Regex,
    temp_ignore: Regex,
    accel: Regex,
    xyz: Regex,
    axis: Regex,
    act_hi: Regex,
    act_lo: Regex,
    steps: Regex,
    signal: Regex,
    sample_like: Regex,
    time_label: Regex,
    id_col: Regex,
    unit: Regex,
}

fn cat() -> &'static Catalog {
    static C: OnceLock<Catalog> = OnceLock::new();
    C.get_or_init(|| {
        let r = |p: &str| Regex::new(&format!("(?i){p}")).unwrap();
        Catalog {
            time_abs: r(r"^(timestamp|time ?stamp|date ?time|date/time|datetime|date_time|utc|local ?time|time ?\(utc\)|date time.*|record(ed)? ?time|sample ?time|dateTime)$|^date\s*time|timestamp"),
            time_rel: r(r"^(time|t|elapsed|elapsed ?time|seconds|secs?|ms|millis|minutes?|mins?|hours?|hrs?|days?|zt|ct|circadian ?time|zeitgeber ?time|time ?\((s|sec|seconds|ms|min|minutes|h|hr|hrs|hours|d|days)\)|time_?(s|ms|min|h|hours|days)|seconds_elapsed|time ?since ?start.*|unix.*|epoch ?time|ts|time ?\[.*\])$"),
            date_only: r(r"^(date|day|datum|fecha|data|dia|tag|date ?\(.*\))$"),
            tod_only: r(r"^(time|clock|clock ?time|hora|zeit|heure|time of day|tod|time ?\(hh:mm(:ss)?\))$"),
            ignore: r(r"^(#|no\.?|n|line|index|idx|row|epoch ?#|record|id|sample ?#|seq|sequence|ms)$|sleep|wake|state|marker|event|button|status|flag|battery|inclinometer|orientation|interval|annotation|comment|note|unit$|quality|confidence|counter"),
            mask: r(r"off[- _]?wrist|non[- _]?wear|nonwear|interval ?status|excluded"),
            light: r(r"^(light|lux|illuminance|lx)$|\blux\b|\(lx\)|illuminance|white ?light|^amb(ient)? ?light|^light ?(level|intensity|lux)?\b|intensity, ?lux|melanopic|photopic|^light\b"),
            light_ignore: r(r"^(red|green|blue|ir|uva|uvb|uv|clear|infrared)[ _]?(light)?\b|\b(r|g|b)$"),
            hr: r(r"^(hr|bpm|heart ?rate|pulse|pulse ?rate|heartrate|heart_rate)\b|heart[ _]?rate|\(bpm\)|\bbpm\b"),
            rr: r(r"^(rr|rri|ibi|nn|rr ?intervals?|nn ?intervals?)( ?\((ms|s)\)|_ms|_s)?$|inter[- _]?beat|rr[ _-]?interval"),
            temp: r(r"\btemp(erature)?\b|temp_|°\s*[cf]\b|\bdeg ?[cf]\b|\bcbt\b|skin ?temp|core ?temp|body ?temp|wrist ?temp|thermo|^value$"),
            temp_ignore: r(r"\b(ext(ernal)?|ambient|room|amb|env(ironment(al)?)?)[ _]?temp"),
            accel: r(r"(?:^|[\s_\-])(?:linear[ _]?)?acc(?:el(?:eration|erometer)?)?[\s_\-(:]*([xyz])\b|^a([xyz])$|^acc_?([xyz])$|^([xyz])[ _]?\(m/s|^([xyz])[ _]?\(g\)|^([xyz])[ _]?axis$|^axis ?([xyz])$"),
            xyz: r(r"^([xyz])$"),
            axis: r(r"^axis ?([123])$"),
            act_hi: r(r"^(activity|act|actv|activity ?counts?|activity ?index|activity ?score|motion ?index|pim|vector ?magnitude|vm|svm|svmgs|enmo|counts?|wheel.*|revolutions|revs|locomot\w*|beam ?breaks?|crossings|movement|motion|actigraphy|zcm|tat|mad)\b|\b(enmo|pim|activity)\b|counts?$|wheel|running"),
            act_lo: r(r"^(pimn|zcmn|tatn|movement ?intensity)$"),
            steps: r(r"^steps?\b|step ?count"),
            signal: r(r"lumin(escence)?|biolum|\brlu\b|photon|\bcps\b|counts?/s|luciferase|\bluc\b|per2|bmal|melatonin|cortisol|conc(entration)?|pg/ml|ng/ml|nmol|µg|\beda\b|gsr|skin ?conductance|spo2|resp(iration)?|glucose|blood ?pressure|systolic|diastolic|expression|signal|intensity|fluorescence|\bvalue\b|\bmean\b"),
            sample_like: r(r"^[a-p] ?\d{1,2}$|^(dish|well|sample|ch|channel|cage|mouse|mice|fly|animal|subj(ect)?|id|plate|tube|m|f|wt|ko|rat|hamster)[ _\-#]*\d+|^\d+$|_rep\d|^r\d+$"),
            time_label: r(r"^\s*(?:(zt|ct|t|time|hr?|hours?)\s*[_\- ]?)?(-?\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours)?(?:\s*[_\-. ]\s*(?:rep|r|n|s)?\s*\d+)?\s*$"),
            id_col: r(r"^(id|gene|genes|probe|probe ?id|probeset|symbol|gene ?symbol|name|transcript|ensembl.*|accession|feature|cpg|metabolite|protein)$"),
            unit: r(r"\(([^()]{1,24})\)\s*$|\[([^\[\]]{1,24})\]\s*$|,\s*([^,]{1,12})$"),
        }
    })
}

fn header_unit(h: &str) -> Option<String> {
    let c = cat().unit.captures(h)?;
    let u = c.get(1).or(c.get(2)).or(c.get(3))?.as_str().trim().to_string();
    (!u.to_lowercase().contains("s/n") && u.len() <= 16).then_some(u)
}

fn header_name(h: &str) -> String {
    let n = cat().unit.replace(h, "").trim().trim_end_matches(',').trim().to_string();
    if n.is_empty() {
        h.to_string()
    } else {
        n
    }
}

fn unit_from_header(h: &str) -> Option<TimeUnit> {
    let l = h.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| l.contains(w));
    if has(&["(ms)", "_ms", "millis", "[ms]"]) || l == "ms" {
        Some(TimeUnit::Milliseconds)
    } else if has(&["(s)", "(sec", "_s", "[s]", "second", "secs"]) || l == "s" || l == "sec" || l == "t" {
        Some(TimeUnit::Seconds)
    } else if has(&["(min", "_min", "minute", "[min]"]) || l == "min" || l == "mins" {
        Some(TimeUnit::Minutes)
    } else if has(&["(h)", "(hr", "hour", "_h", "[h]", "zt", "ct", "zeitgeber", "circadian time"]) || l == "h" || l == "hr" || l == "hrs" {
        Some(TimeUnit::Hours)
    } else if has(&["day", "(d)", "_d", "[d]"]) {
        Some(TimeUnit::Days)
    } else {
        None
    }
}

// ================================================================================================
// Format-specific metadata
// ================================================================================================

#[derive(Default, Clone)]
struct Meta {
    format: Option<&'static str>,
    start: Option<i64>,
    epoch_s: Option<f64>,
    order: Option<Dmy>,
    /// Column names / roles for headerless tables
    positional: Option<Vec<(&'static str, ColRole)>>,
    /// Role overrides by column name (case-insensitive exact match)
    by_name: Vec<(&'static str, ColRole)>,
    header_line: Option<usize>,
    /// (column, required value) — rows whose value differs are skipped (e.g. DAM status)
    notes: Vec<String>,
}

fn meta_from_preamble(lines: &[&str], name: &str) -> Meta {
    let mut m = Meta::default();
    let head = lines.iter().take(120).copied().collect::<Vec<_>>().join("\n");
    let rx = |p: &str| Regex::new(&format!("(?im){p}")).unwrap();
    // ActiGraph ActiLife CSV
    if head.contains("ActiGraph") {
        m.format = Some("ActiGraph ActiLife CSV");
        let order = rx(r"date format\s+(\S+)").captures(&head).map(|c| c[1].to_string());
        m.order = order.as_deref().map(|f| if f.starts_with('d') { Dmy::Dmy } else if f.starts_with('y') { Dmy::Ymd } else { Dmy::Mdy });
        let date = rx(r"^Start Date\s+(\S+)").captures(&head).map(|c| c[1].to_string());
        let time = rx(r"^Start Time\s+(\S+)").captures(&head).map(|c| c[1].to_string());
        if let (Some(d), Some(t)) = (date, time) {
            m.start = parse_dt(&format!("{d} {t}"), m.order.unwrap_or(Dmy::Mdy));
        }
        if let Some(c) = rx(r"Epoch Period \(hh:mm:ss\)\s+(\d+):(\d+):(\d+)").captures(&head) {
            let s = c[1].parse::<f64>().unwrap_or(0.0) * 3600.0 + c[2].parse::<f64>().unwrap_or(0.0) * 60.0 + c[3].parse::<f64>().unwrap_or(0.0);
            if s > 0.0 {
                m.epoch_s = Some(s);
            }
        }
        if m.epoch_s.is_none() {
            if let Some(c) = rx(r"at\s+(\d+)\s*Hz").captures(&head) {
                m.epoch_s = c[1].parse::<f64>().ok().map(|hz| 1.0 / hz);
            }
        }
        m.positional = Some(vec![
            ("Axis1", ColRole::Data(Role::Activity)),
            ("Axis2", ColRole::Ignore),
            ("Axis3", ColRole::Ignore),
            ("Steps", ColRole::Ignore),
            ("Lux", ColRole::Data(Role::Light)),
            ("Inclinometer Off", ColRole::Ignore),
            ("Inclinometer Standing", ColRole::Ignore),
            ("Inclinometer Sitting", ColRole::Ignore),
            ("Inclinometer Lying", ColRole::Ignore),
        ]);
        m.notes.push("Header-less ActiGraph table: columns read as Axis1, Axis2, Axis3, Steps, Lux, inclinometer; activity = Axis1 counts.".into());
    }
    // GENEActiv CSV (100-line header, headerless data: time, x, y, z, lux, button, temperature)
    else if head.contains("GENEActiv") || head.contains("GENEA") && head.contains("Device") {
        m.format = Some("GENEActiv CSV");
        m.order = Some(Dmy::Ymd);
        m.positional = Some(vec![
            ("Timestamp", ColRole::Time),
            ("x", ColRole::Data(Role::AccelX)),
            ("y", ColRole::Data(Role::AccelY)),
            ("z", ColRole::Data(Role::AccelZ)),
            ("Lux", ColRole::Data(Role::Light)),
            ("Button", ColRole::Ignore),
            ("Temperature", ColRole::Data(Role::Temperature)),
        ]);
    }
    // Philips Respironics Actiware
    else if rx(r"epoch-by-epoch data|off-wrist status|actiware").is_match(&head) || lines.iter().any(|l| l.contains("Epoch-by-Epoch")) {
        m.format = Some("Philips Respironics Actiware");
        m.by_name = vec![
            ("activity", ColRole::Data(Role::Activity)),
            ("white light", ColRole::Data(Role::Light)),
            ("off-wrist status", ColRole::Mask),
            ("interval status", ColRole::Mask),
            ("sleep/wake", ColRole::Ignore),
            ("marker", ColRole::Ignore),
        ];
    }
    // Condor ActTrust
    else if lines.iter().any(|l| l.trim_start().to_uppercase().starts_with("DATE/TIME;")) {
        m.format = Some("Condor ActTrust");
        m.order = Some(Dmy::Dmy);
        m.by_name = vec![
            ("pim", ColRole::Data(Role::Activity)),
            ("light", ColRole::Data(Role::Light)),
            ("temperature", ColRole::Data(Role::Temperature)),
            ("ext temperature", ColRole::Ignore),
            ("amb light", ColRole::Ignore),
            ("state", ColRole::Ignore),
            ("event", ColRole::Ignore),
            ("ms", ColRole::Ignore),
        ];
    }
    // Polar Flow / Polar ProTrainer CSV: summary line, values, then the sample table
    else if rx(r#"^"?name"?[,;]"?sport"?[,;]"?date"?[,;]"?start time"?"#).is_match(lines.first().copied().unwrap_or("")) {
        m.format = Some("Polar CSV");
        m.order = Some(Dmy::Dmy);
        if let (Some(h), Some(v)) = (lines.first(), lines.get(1)) {
            let hs: Vec<String> = h.split([',', ';']).map(|s| clean(s).to_lowercase()).collect();
            let vs: Vec<&str> = v.split([',', ';']).map(clean).collect();
            let get = |k: &str| hs.iter().position(|x| x == k).and_then(|i| vs.get(i).copied());
            if let (Some(d), Some(t)) = (get("date"), get("start time")) {
                m.start = parse_dt(&format!("{d} {t}"), Dmy::Dmy);
            }
        }
        m.header_line = Some(2);
    } else if head.contains("Plot Title") || head.contains("LGR S/N") {
        m.format = Some("HOBO logger (HOBOware CSV)");
    } else if rx(r"date/time\s*[,;\t]\s*unit\s*[,;\t]\s*value").is_match(&head) {
        m.format = Some("iButton / Thermochron (OneWireViewer)");
        m.by_name = vec![("value", ColRole::Data(Role::Temperature)), ("unit", ColRole::Ignore)];
    } else if head.contains("seconds_elapsed") {
        m.format = Some("Sensor Logger");
        m.by_name = vec![("seconds_elapsed", ColRole::Ignore), ("time", ColRole::Time)];
    } else if rx(r"^\s*time\s*\(s\)").is_match(&head) {
        m.format = Some("Phyphox");
    } else if rx(r"motionware|camntech|motionwatch").is_match(&head) {
        m.format = Some("CamNtech MotionWare export");
    }
    let _ = name;
    m
}

// ================================================================================================
// Parsing entry point
// ================================================================================================

/// Parse one file. `row_limit` caps rows for a fast preview.
pub fn parse_source(src: &Source, opts: &ImportOptions, row_limit: Option<usize>) -> Result<Parsed, String> {
    let lower = src.name.to_lowercase();
    let mut opts = opts.clone();
    if opts.start.is_none() {
        opts.start = src.start_hint;
    }
    if [".xlsx", ".xlsm", ".xls", ".ods"].iter().any(|e| lower.ends_with(e)) {
        let (text, notes) = spreadsheet_to_text(&src.bytes)?;
        let mut p = parse_text(&text, &src.name, &opts, row_limit)?;
        p.notes.extend(notes);
        return Ok(p);
    }
    let text = decode(&src.bytes);
    parse_text(&text, &src.name, &opts, row_limit)
}

fn spreadsheet_to_text(bytes: &[u8]) -> Result<(String, Vec<String>), String> {
    use calamine::{Data, Reader};
    let mut wb = calamine::open_workbook_auto_from_rs(Cursor::new(bytes.to_vec())).map_err(|e| e.to_string())?;
    let names = wb.sheet_names().to_vec();
    let mut notes = Vec::new();
    for name in &names {
        let Ok(range) = wb.worksheet_range(name) else { continue };
        if range.height() < 3 || range.width() < 1 {
            continue;
        }
        let mut text = String::with_capacity(range.height() * range.width() * 8);
        for row in range.rows() {
            for (c, cell) in row.iter().enumerate() {
                if c > 0 {
                    text.push('\t');
                }
                match cell {
                    Data::DateTime(d) => {
                        if let Some(dt) = d.as_datetime() {
                            text.push_str(&dt.format("%Y-%m-%d %H:%M:%S%.3f").to_string());
                        }
                    }
                    Data::Float(f) => text.push_str(&format!("{f}")),
                    Data::Int(i) => text.push_str(&i.to_string()),
                    Data::Empty => {}
                    other => text.push_str(&other.to_string().replace(['\t', '\n', '\r'], " ")),
                }
            }
            text.push('\n');
        }
        if names.len() > 1 {
            notes.push(format!("Workbook has {} sheets; using \"{name}\".", names.len()));
        }
        return Ok((text, notes));
    }
    Err("No sheet with a data table was found in the workbook.".into())
}

fn parse_text(text: &str, name: &str, opts: &ImportOptions, row_limit: Option<usize>) -> Result<Parsed, String> {
    let trimmed = text.trim_start();
    if trimmed.starts_with('<') && (text.contains("<HealthData") || text.contains("HKQuantityTypeIdentifier")) {
        return parse_apple_health(text, name, row_limit);
    }
    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        return parse_json(text, name, opts, row_limit);
    }
    let head: Vec<&str> = text.lines().take(600).collect();
    if head.is_empty() {
        return Err(format!("{name} is empty."));
    }
    let stem = name.rsplit_once('.').map_or(name, |p| p.0).to_uppercase();
    if ["ACC", "BVP", "EDA", "HR", "TEMP", "IBI"].contains(&stem.as_str()) && head[0].split(',').next().and_then(|s| s.trim().parse::<f64>().ok()).is_some_and(|v| v > 1e9) {
        return parse_empatica(text, name, &stem, row_limit);
    }
    if name.to_lowercase().ends_with(".awd") || is_awd(&head) {
        return parse_awd(text, name, opts, row_limit);
    }
    let meta = meta_from_preamble(&head, name);
    let forced = opts.header_line.or(meta.header_line);
    let shape = sniff(&head, forced);
    match shape {
        Some(s) if s.ncols >= 2 => parse_table(text, name, opts, row_limit, meta, s, &head),
        _ => parse_single_column(text, name, opts, row_limit, &head),
    }
}

fn preview_lines(head: &[&str]) -> Vec<String> {
    head.iter().take(40).map(|l| if l.len() > 400 { format!("{}…", &l[..l.char_indices().nth(400).map_or(l.len(), |c| c.0)]) } else { l.to_string() }).collect()
}

// ------------------------------------------------------------------------------------------------
// Single-column files: RR intervals, fixed-interval series
// ------------------------------------------------------------------------------------------------

fn parse_single_column(text: &str, name: &str, opts: &ImportOptions, row_limit: Option<usize>, head: &[&str]) -> Result<Parsed, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty()).peekable();
    let first = *lines.peek().ok_or(format!("{name} is empty."))?;
    let mut header = None;
    if num(first.split([',', ';', '\t']).next().unwrap_or(""), false).is_nan() {
        header = Some(first.to_string());
        lines.next();
    }
    let mut vals = Vec::new();
    let mut bad = 0usize;
    for l in lines {
        let v = num(l.split([',', ';', '\t', ' ']).next().unwrap_or(""), l.contains(',') && !l.contains('.'));
        if v.is_nan() {
            bad += 1;
        } else {
            vals.push(v);
        }
        if row_limit.is_some_and(|r| vals.len() >= r) {
            break;
        }
    }
    if vals.len() < 4 {
        return Err(format!("{name}: no data table or time column was recognised."));
    }
    let hdr = header.clone().unwrap_or_default();
    let med = crate::stats::median(&vals);
    let rr_header = cat().rr.is_match(clean(&hdr));
    let forced_rr = matches!(opts.roles.get(&0), Some(ColRole::RR));
    let forced_other = matches!(opts.roles.get(&0), Some(ColRole::Data(_)));
    let looks_rr = !forced_other && (forced_rr || rr_header || (header.is_none() && ((200.0..=3000.0).contains(&med) || (0.2..=3.0).contains(&med))));
    let mut p = Parsed { source: name.into(), preview: preview_lines(head), header_line: header.as_ref().map(|_| 0), data_line: header.is_some() as usize, ..Default::default() };
    if looks_rr && opts.interval_s.is_none() {
        let rr: Vec<f64> = if med < 10.0 { vals.iter().map(|v| v * 1000.0).collect() } else { vals };
        let d = rr_dataset(name, &rr, None, opts.start);
        p.format = "RR / inter-beat intervals".into();
        p.time_desc = "Beat times = cumulative sum of the RR intervals".into();
        p.relative_time = opts.start.is_none();
        p.columns = vec![ColumnInfo { name: if hdr.is_empty() { "RR interval (ms)".into() } else { hdr }, role: ColRole::RR, example: format!("{:.0}", rr[0]) }];
        p.rows = rr.len();
        p.notes = d.notes.clone();
        p.truncated = row_limit.is_some_and(|r| rr.len() >= r);
        p.subjects = vec![d];
        return Ok(p);
    }
    // A plain series of values: needs a sampling interval
    let role = match opts.roles.get(&0) {
        Some(ColRole::Data(r)) => *r,
        _ => Role::Activity,
    };
    p.format = "Single column of values".into();
    p.columns = vec![ColumnInfo { name: if hdr.is_empty() { "Values".into() } else { hdr.clone() }, role: ColRole::Data(role), example: format!("{}", vals[0]) }];
    p.needs_interval = true;
    p.relative_time = opts.start.is_none();
    let Some(step) = opts.interval_s.filter(|s| *s > 0.0) else {
        return Err(format!("{name}: one column of numbers without time stamps. Set the sampling interval (and start time) in the import options."));
    };
    let base = opts.start.unwrap_or_else(today_midnight_ms);
    let t: Vec<i64> = (0..vals.len()).map(|i| base + (i as f64 * step * 1000.0).round() as i64).collect();
    p.time_desc = format!("No time stamps: every {} from {}", format_duration(step), ms_to_naive(base).format("%Y-%m-%d %H:%M"));
    p.rows = vals.len();
    let nm = if hdr.is_empty() { "Activity".to_string() } else { header_name(&hdr) };
    p.subjects = vec![RawDataset { name: name.into(), format: p.format.clone(), t_ms: t, channels: vec![Channel::new(role, &nm, "", vals)], ..Default::default() }];
    let _ = bad;
    Ok(p)
}

// ------------------------------------------------------------------------------------------------
// Actiwatch / CamNtech AWD
// ------------------------------------------------------------------------------------------------

fn is_awd(head: &[&str]) -> bool {
    static D: OnceLock<Regex> = OnceLock::new();
    let d = D.get_or_init(|| Regex::new(r"^\s*\d{1,2}-[A-Za-z]{3}-\d{2,4}\s*$").unwrap());
    head.len() > 8 && d.is_match(head[1]) && head[2].trim().len() <= 8 && head[2].contains(':') && !head[2].contains(',')
}

fn parse_awd(text: &str, name: &str, opts: &ImportOptions, row_limit: Option<usize>) -> Result<Parsed, String> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 8 {
        return Err(format!("{name}: too short for an AWD file."));
    }
    let start = parse_dt(&format!("{} {}", lines[1].trim(), lines[2].trim()), Dmy::Dmy);
    let code = lines[3].trim();
    let epoch = match code {
        "1" => Some(15.0),
        "2" => Some(30.0),
        "4" => Some(60.0),
        "8" => Some(120.0),
        "20" => Some(300.0),
        "81" => Some(2.0),
        "C1" | "c1" => Some(5.0),
        _ => None,
    };
    let step = opts.interval_s.or(epoch).ok_or(format!("{name}: unknown AWD epoch code '{code}'. Set the sampling interval in the import options."))?;
    let base = opts.start.or(start).unwrap_or_else(today_midnight_ms);
    let mut vals = Vec::new();
    for l in &lines[7..] {
        if l.trim().is_empty() {
            continue;
        }
        vals.push(num(l.split([',', ' ', '\t']).next().unwrap_or(""), false));
        if row_limit.is_some_and(|r| vals.len() >= r) {
            break;
        }
    }
    let t: Vec<i64> = (0..vals.len()).map(|i| base + (i as f64 * step * 1000.0).round() as i64).collect();
    let n = vals.len();
    let d = RawDataset {
        name: if lines[0].trim().is_empty() { name.into() } else { format!("{} ({})", name, lines[0].trim()) },
        format: "Actiwatch / CamNtech AWD".into(),
        t_ms: t,
        channels: vec![Channel::new(Role::Activity, "Activity", "counts", vals)],
        ..Default::default()
    };
    Ok(Parsed {
        source: name.into(),
        format: "Actiwatch / CamNtech AWD".into(),
        notes: vec![format!("Subject '{}', start {}, epoch {} (code {code}).", lines[0].trim(), ms_to_naive(base).format("%Y-%m-%d %H:%M"), format_duration(step))],
        preview: preview_lines(&lines[..lines.len().min(40)]),
        header_line: None,
        data_line: 7,
        columns: vec![ColumnInfo { name: "Activity counts".into(), role: ColRole::Data(Role::Activity), example: lines[7].trim().into() }],
        time_desc: format!("Header start time + {} epochs", format_duration(step)),
        rows: n,
        truncated: row_limit.is_some_and(|r| n >= r),
        subjects: vec![d],
        ..Default::default()
    })
}

// ------------------------------------------------------------------------------------------------
// Empatica E4
// ------------------------------------------------------------------------------------------------

fn parse_empatica(text: &str, name: &str, stem: &str, row_limit: Option<usize>) -> Result<Parsed, String> {
    let lines: Vec<&str> = text.lines().collect();
    let first: Vec<f64> = lines[0].split(',').map(|s| num(s, false)).collect();
    let start = unix_ms_to_local(first[0] * 1000.0);
    let mut p = Parsed { source: name.into(), format: "Empatica E4".into(), preview: preview_lines(&lines[..lines.len().min(40)]), data_line: 2, ..Default::default() };
    if stem == "IBI" {
        let mut times = Vec::new();
        let mut rr = Vec::new();
        for l in &lines[1..] {
            let f: Vec<f64> = l.split(',').map(|s| num(s, false)).collect();
            if f.len() >= 2 && f[0].is_finite() && f[1].is_finite() {
                times.push(start + (f[0] * 1000.0).round() as i64);
                rr.push(f[1] * 1000.0);
            }
            if row_limit.is_some_and(|r| rr.len() >= r) {
                break;
            }
        }
        let mut d = rr_dataset(name, &rr, Some(&times), None);
        d.channels[0].name = "Heart rate (IBI)".into();
        d.format = "Empatica E4".into();
        p.columns = vec![ColumnInfo { name: "IBI (s)".into(), role: ColRole::RR, example: lines.get(1).unwrap_or(&"").to_string() }];
        p.time_desc = format!("Unix start {} + beat times", ms_to_naive(start).format("%Y-%m-%d %H:%M:%S"));
        p.rows = rr.len();
        p.notes = d.notes.clone();
        p.subjects = vec![d];
        return Ok(p);
    }
    let hz = lines.get(1).and_then(|l| l.split(',').next()).map(|s| num(s, false)).filter(|h| *h > 0.0).ok_or("Empatica file without a sampling rate line")?;
    let (roles, unit, scale): (Vec<(Role, &str)>, &str, f64) = match stem {
        "ACC" => (vec![(Role::AccelX, "x"), (Role::AccelY, "y"), (Role::AccelZ, "z")], "g", 1.0 / 64.0),
        "TEMP" => (vec![(Role::Temperature, "Skin temperature")], "°C", 1.0),
        "HR" => (vec![(Role::HeartRate, "Heart rate")], "bpm", 1.0),
        "EDA" => (vec![(Role::Signal, "EDA")], "µS", 1.0),
        _ => (vec![(Role::Signal, "BVP")], "", 1.0),
    };
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); roles.len()];
    let mut n = 0usize;
    for l in &lines[2..] {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<f64> = l.split(',').map(|s| num(s, false) * scale).collect();
        for (k, c) in cols.iter_mut().enumerate() {
            c.push(f.get(k).copied().unwrap_or(f64::NAN));
        }
        n += 1;
        if row_limit.is_some_and(|r| n >= r) {
            break;
        }
    }
    let t: Vec<i64> = (0..n).map(|i| start + (i as f64 * 1000.0 / hz).round() as i64).collect();
    p.columns = roles.iter().map(|(r, nm)| ColumnInfo { name: nm.to_string(), role: ColRole::Data(*r), example: String::new() }).collect();
    p.time_desc = format!("Unix start {} + {hz} Hz", ms_to_naive(start).format("%Y-%m-%d %H:%M:%S"));
    p.rows = n;
    p.truncated = row_limit.is_some_and(|r| n >= r);
    if stem == "BVP" {
        p.notes.push("BVP is the raw pulse waveform (64 Hz); for circadian heart rate use HR.csv or IBI.csv.".into());
    }
    p.subjects = vec![RawDataset {
        name: name.into(),
        format: "Empatica E4".into(),
        t_ms: t,
        channels: roles.iter().zip(cols).map(|((r, nm), v)| Channel::new(*r, nm, unit, v)).collect(),
        ..Default::default()
    }];
    Ok(p)
}

// ------------------------------------------------------------------------------------------------
// Apple Health export.xml (regex record scan)
// ------------------------------------------------------------------------------------------------

fn parse_apple_health(text: &str, name: &str, row_limit: Option<usize>) -> Result<Parsed, String> {
    let rec = Regex::new(r#"<Record\s([^>]*?)/?>"#).unwrap();
    let attr = Regex::new(r#"(\w+)="([^"]*)""#).unwrap();
    let kinds: [(&str, Role, &str, &str, bool); 5] = [
        ("HKQuantityTypeIdentifierHeartRate", Role::HeartRate, "Heart rate", "bpm", false),
        ("HKQuantityTypeIdentifierStepCount", Role::Activity, "Steps", "steps", true),
        ("HKQuantityTypeIdentifierBodyTemperature", Role::Temperature, "Body temperature", "°C", false),
        ("HKQuantityTypeIdentifierAppleSleepingWristTemperature", Role::Temperature, "Wrist temperature", "°C", false),
        ("HKQuantityTypeIdentifierRestingHeartRate", Role::Signal, "Resting heart rate", "bpm", false),
    ];
    let mut series: Vec<Vec<(i64, f64)>> = vec![Vec::new(); kinds.len()];
    let mut sources: Vec<std::collections::BTreeSet<String>> = vec![Default::default(); kinds.len()];
    let mut count = 0usize;
    for c in rec.captures_iter(text) {
        let a = &c[1];
        let mut ty = "";
        let (mut start, mut val, mut src) = (None, f64::NAN, "");
        for m in attr.captures_iter(a) {
            match m.get(1).unwrap().as_str() {
                "type" => ty = m.get(2).unwrap().as_str(),
                "startDate" => start = parse_dt(m.get(2).unwrap().as_str(), Dmy::Ymd),
                "value" => val = num(m.get(2).unwrap().as_str(), false),
                "sourceName" => src = m.get(2).unwrap().as_str(),
                _ => {}
            }
        }
        if let Some(k) = kinds.iter().position(|x| x.0 == ty) {
            if let Some(t) = start {
                series[k].push((t, val));
                sources[k].insert(src.to_string());
                count += 1;
            }
        }
        if row_limit.is_some_and(|r| count >= r) {
            break;
        }
    }
    let mut sets = Vec::new();
    let mut notes = Vec::new();
    for (k, s) in series.into_iter().enumerate() {
        if s.is_empty() {
            continue;
        }
        let (role, nm, unit, sum) = (kinds[k].1, kinds[k].2, kinds[k].3, kinds[k].4);
        notes.push(format!("{nm}: {} records from {}", group_thousands(s.len()), sources[k].iter().cloned().collect::<Vec<_>>().join(", ")));
        if sum && sources[k].len() > 1 {
            notes.push(format!("{nm} come from {} devices; overlapping records are added together (may double-count).", sources[k].len()));
        }
        let mut ch = Channel::new(role, nm, unit, s.iter().map(|x| x.1).collect());
        ch.sum = sum;
        let mut d = RawDataset { name: name.into(), format: "Apple Health export.xml".into(), t_ms: s.iter().map(|x| x.0).collect(), channels: vec![ch], ..Default::default() };
        d.sort_by_time();
        sets.push(d);
    }
    if sets.is_empty() {
        return Err("No heart-rate, step or temperature records found in the Apple Health export.".into());
    }
    let d = RawDataset::merge(sets)?;
    Ok(Parsed {
        source: name.into(),
        format: "Apple Health export.xml".into(),
        notes,
        preview: preview_lines(&text.lines().take(40).collect::<Vec<_>>()),
        columns: d.channels.iter().map(|c| ColumnInfo { name: c.name.clone(), role: ColRole::Data(c.role), example: String::new() }).collect(),
        time_desc: "Record startDate (local wall-clock time)".into(),
        rows: count,
        truncated: row_limit.is_some_and(|r| count >= r),
        subjects: vec![d],
        ..Default::default()
    })
}

// ------------------------------------------------------------------------------------------------
// JSON: Fitbit exports, intraday API responses, generic arrays of records
// ------------------------------------------------------------------------------------------------

fn role_from_filename(name: &str) -> (Role, &'static str, &'static str, bool) {
    let l = name.to_lowercase();
    if l.contains("heart") {
        (Role::HeartRate, "Heart rate", "bpm", false)
    } else if l.contains("step") {
        (Role::Activity, "Steps", "steps", true)
    } else if l.contains("temp") {
        (Role::Temperature, "Temperature", "°C", false)
    } else if l.contains("calor") {
        (Role::Signal, "Calories", "kcal", true)
    } else if l.contains("distance") {
        (Role::Signal, "Distance", "", true)
    } else {
        (Role::Signal, "Value", "", false)
    }
}

fn json_number(v: &serde_json::Value) -> f64 {
    match v {
        serde_json::Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        serde_json::Value::String(s) => num(s, false),
        serde_json::Value::Object(o) => ["bpm", "value", "temperature", "steps"].iter().find_map(|k| o.get(*k)).map_or(f64::NAN, json_number),
        serde_json::Value::Bool(b) => *b as i32 as f64,
        _ => f64::NAN,
    }
}

fn parse_json(text: &str, name: &str, opts: &ImportOptions, row_limit: Option<usize>) -> Result<Parsed, String> {
    use serde_json::Value;
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            // JSON Lines
            let rows: Result<Vec<Value>, _> = text.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect();
            Value::Array(rows.map_err(|_| format!("{name}: invalid JSON ({e})"))?)
        }
    };
    let preview = preview_lines(&text.lines().take(40).collect::<Vec<_>>());
    // Fitbit Web API intraday: {"activities-heart":[{"dateTime":"2020-01-01"}], "activities-heart-intraday":{"dataset":[{"time":"00:00:00","value":64}]}}
    if let Value::Object(o) = &v {
        if let Some((key, intr)) = o.iter().find(|(k, _)| k.ends_with("-intraday")) {
            let base_key = key.trim_end_matches("-intraday");
            let date = o.get(base_key).and_then(|a| a.get(0)).and_then(|x| x.get("dateTime")).and_then(|d| d.as_str()).and_then(|d| parse_dt(d, Dmy::Ymd)).or(opts.start).unwrap_or_else(today_midnight_ms);
            let ds = intr.get("dataset").and_then(|d| d.as_array()).ok_or("intraday dataset missing")?;
            let (role, nm, unit, sum) = role_from_filename(key);
            let (mut t, mut val) = (Vec::new(), Vec::new());
            let mut prev = -1i64;
            let mut day_off = 0i64;
            for x in ds {
                let Some(tod) = x.get("time").and_then(|s| s.as_str()).and_then(parse_tod) else { continue };
                if tod < prev {
                    day_off += MS_PER_DAY;
                }
                prev = tod;
                t.push(date + day_off + tod);
                val.push(x.get("value").map_or(f64::NAN, json_number));
            }
            let mut ch = Channel::new(role, nm, unit, val);
            ch.sum = sum;
            let n = t.len();
            return Ok(Parsed {
                source: name.into(),
                format: "Fitbit intraday JSON".into(),
                preview,
                columns: vec![ColumnInfo { name: key.clone(), role: ColRole::Data(role), example: String::new() }],
                time_desc: format!("Date {} + time of day", ms_to_naive(date).format("%Y-%m-%d")),
                rows: n,
                subjects: vec![RawDataset { name: name.into(), format: "Fitbit intraday JSON".into(), t_ms: t, channels: vec![ch], ..Default::default() }],
                ..Default::default()
            });
        }
    }
    let arr = match &v {
        Value::Array(a) => a.clone(),
        Value::Object(o) => o.values().find_map(|x| x.as_array().cloned()).ok_or(format!("{name}: no array of records found in the JSON."))?,
        _ => return Err(format!("{name}: unsupported JSON layout.")),
    };
    // Fitbit account export: [{"dateTime": "01/01/20 00:00:05", "value": {"bpm": 72, ...}}]
    if arr.first().and_then(|x| x.get("dateTime")).is_some() && arr.first().and_then(|x| x.get("value")).is_some() {
        let samples: Vec<&str> = arr.iter().take(200).filter_map(|x| x.get("dateTime")?.as_str()).collect();
        let order = match opts.date_order {
            DateOrder::DayFirst => Dmy::Dmy,
            DateOrder::MonthFirst => Dmy::Mdy,
            DateOrder::YearFirst => Dmy::Ymd,
            DateOrder::Auto => detect_order(&samples).0,
        };
        let (role, nm, unit, sum) = role_from_filename(name);
        let (mut t, mut val) = (Vec::new(), Vec::new());
        for x in arr.iter().take(row_limit.unwrap_or(usize::MAX)) {
            if let Some(ts) = x.get("dateTime").and_then(|s| s.as_str()).and_then(|s| parse_dt(s, order)) {
                t.push(ts);
                val.push(x.get("value").map_or(f64::NAN, json_number));
            }
        }
        let mut ch = Channel::new(role, nm, unit, val);
        ch.sum = sum;
        let n = t.len();
        let mut d = RawDataset { name: name.into(), format: "Fitbit JSON".into(), t_ms: t, channels: vec![ch], ..Default::default() };
        d.sort_by_time();
        return Ok(Parsed {
            source: name.into(),
            format: "Fitbit JSON".into(),
            preview,
            columns: vec![ColumnInfo { name: "dateTime".into(), role: ColRole::Time, example: samples.first().unwrap_or(&"").to_string() }, ColumnInfo { name: "value".into(), role: ColRole::Data(role), example: String::new() }],
            time_desc: format!("dateTime ({})", if order == Dmy::Dmy { "day first" } else if order == Dmy::Mdy { "month first" } else { "year first" }),
            rows: n,
            truncated: row_limit.is_some_and(|r| n >= r),
            subjects: vec![d],
            ..Default::default()
        });
    }
    // Generic array of flat records → tab-separated table → generic parser
    let mut keys: Vec<String> = Vec::new();
    for x in arr.iter().take(500) {
        if let Value::Object(o) = x {
            for k in o.keys() {
                if !keys.contains(k) {
                    keys.push(k.clone());
                }
            }
        }
    }
    if keys.is_empty() {
        return Err(format!("{name}: the JSON records have no fields."));
    }
    let mut tsv = keys.join("\t");
    tsv.push('\n');
    for x in arr.iter().take(row_limit.unwrap_or(usize::MAX)) {
        let row: Vec<String> = keys
            .iter()
            .map(|k| match x.get(k) {
                Some(Value::String(s)) => s.replace(['\t', '\n'], " "),
                Some(Value::Null) | None => String::new(),
                Some(o @ Value::Object(_)) => format!("{}", json_number(o)),
                Some(v) => v.to_string(),
            })
            .collect();
        tsv.push_str(&row.join("\t"));
        tsv.push('\n');
    }
    let mut p = parse_text(&tsv, name, opts, row_limit)?;
    p.format = format!("JSON records → {}", p.format);
    p.preview = preview;
    Ok(p)
}

// ------------------------------------------------------------------------------------------------
// Delimited tables (the general case)
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum TimeKind {
    /// Full date-time text
    Text(usize, Dmy),
    /// Separate date and time-of-day columns
    DateTod(usize, usize, Dmy),
    /// Time of day only: rollover across midnight is resolved after parsing
    Tod(usize, i64),
    /// t = v · scale + offset (+ local-zone shift for Unix times)
    Numeric(usize, f64, f64, bool),
    /// No time column: start + i · step
    Implicit(i64, f64),
}

struct ColClass {
    num: f32,
    comma: bool,
    dt: f32,
    dt_time: f32,
    date_only: f32,
    tod: f32,
    example: String,
    distinct: bool,
}

fn classify(samples: &[Vec<&str>], c: usize, order: Dmy) -> ColClass {
    let vals: Vec<&str> = samples.iter().filter_map(|r| r.get(c).map(|s| clean(s))).filter(|s| !s.is_empty()).collect();
    let n = vals.len().max(1) as f32;
    // Decimal commas (European exports): any "12,5"-style value and no "12.5"-style value
    let comma_dec = vals.iter().filter(|s| s.contains(',') && !s.contains('.') && s.chars().all(|ch| ch.is_ascii_digit() || ch == ',' || ch == '-')).count();
    let dot_dec = vals.iter().filter(|s| s.contains('.') && s.chars().all(|ch| ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == 'e' || ch == 'E')).count();
    let comma = comma_dec > 0 && dot_dec == 0;
    let num_ok = vals.iter().filter(|s| !num(s, comma).is_nan()).count() as f32;
    let (mut dt, mut dtt, mut d_only, mut tod) = (0f32, 0f32, 0f32, 0f32);
    for s in &vals {
        let t = tokenize(s);
        let (date, tm) = interpret(&t, order);
        if date.is_some() {
            dt += 1.0;
            if tm.is_some() {
                dtt += 1.0;
            } else {
                d_only += 1.0;
            }
        } else if tm.is_some() && t.month.is_none() {
            tod += 1.0;
        }
    }
    let distinct = vals.first().is_some_and(|f| vals.iter().any(|v| v != f));
    ColClass { num: num_ok / n, comma, dt: dt / n, dt_time: dtt / n, date_only: d_only / n, tod: tod / n, example: vals.first().map(|s| s.to_string()).unwrap_or_default(), distinct }
}

/// Automatic role for a column from its name (None = unknown numeric).
fn auto_role(h: &str, cls: &ColClass) -> Option<ColRole> {
    let c = cat();
    let h = clean(h);
    if cls.num < 0.6 {
        return if c.mask.is_match(h) { Some(ColRole::Mask) } else { Some(ColRole::Ignore) };
    }
    if c.mask.is_match(h) {
        return Some(ColRole::Mask);
    }
    if c.rr.is_match(h) {
        return Some(ColRole::RR);
    }
    if c.hr.is_match(h) && !h.to_lowercase().contains("hrs") {
        return Some(ColRole::Data(Role::HeartRate));
    }
    if c.light_ignore.is_match(h) {
        return Some(ColRole::Ignore);
    }
    if c.light.is_match(h) && !h.to_lowercase().contains("lumines") {
        return Some(ColRole::Data(Role::Light));
    }
    if c.temp_ignore.is_match(h) {
        return Some(ColRole::Ignore);
    }
    if c.temp.is_match(h) && h.to_lowercase() != "value" {
        return Some(ColRole::Data(Role::Temperature));
    }
    if let Some(m) = c.accel.captures(h) {
        let axis = (1..m.len()).find_map(|i| m.get(i)).map(|x| x.as_str().to_lowercase());
        return Some(ColRole::Data(match axis.as_deref() {
            Some("x") => Role::AccelX,
            Some("y") => Role::AccelY,
            _ => Role::AccelZ,
        }));
    }
    if c.act_lo.is_match(h) {
        return Some(ColRole::Ignore);
    }
    if c.act_hi.is_match(h) || c.axis.is_match(h) {
        return Some(ColRole::Data(Role::Activity));
    }
    if c.steps.is_match(h) {
        return Some(ColRole::Data(Role::Activity));
    }
    if c.signal.is_match(h) {
        return Some(ColRole::Data(Role::Signal));
    }
    if c.ignore.is_match(h) {
        return Some(ColRole::Ignore);
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn parse_table(text: &str, name: &str, opts: &ImportOptions, row_limit: Option<usize>, meta: Meta, shape: Shape, head: &[&str]) -> Result<Parsed, String> {
    let c = cat();
    let mut notes = meta.notes.clone();
    let mut buf = Vec::new();
    // Column names
    let mut headers: Vec<String> = match shape.header {
        Some(h) => {
            split(head[h], shape.delim, &mut buf);
            buf.iter().map(|s| clean(s).to_string()).collect()
        }
        None => Vec::new(),
    };
    let ncols = shape.ncols;
    let positional = shape.header.is_none() && meta.positional.is_some();
    if !positional {
        notes.retain(|n| !n.starts_with("Header-less"));
    }
    if positional {
        let pos = meta.positional.as_ref().unwrap();
        // ActiGraph headerless tables may start with date & time columns
        headers = pos.iter().map(|p| p.0.to_string()).collect();
    }
    headers.resize(ncols, String::new());
    for (i, h) in headers.iter_mut().enumerate() {
        if h.is_empty() {
            *h = format!("Column {}", i + 1);
        }
    }
    // Sample rows for classification
    let samples: Vec<Vec<&str>> = head[shape.data_start..]
        .iter()
        .filter(|l| !l.trim().is_empty())
        .take(400)
        .map(|l| {
            let mut v = Vec::new();
            split(l, shape.delim, &mut v);
            v
        })
        .collect();

    // TriKinetics DAM monitor file
    let dam_date = Regex::new(r"^\d{1,2} [A-Za-z]{3} \d{2,4}$").unwrap();
    if shape.delim == Delim::Tab && ncols >= 42 && samples.first().is_some_and(|r| r.get(1).is_some_and(|s| dam_date.is_match(clean(s)))) {
        return parse_dam(text, name, row_limit, &shape, head);
    }

    // Date order
    let order_samples: Vec<&str> = samples.iter().flat_map(|r| r.iter().take(3).map(|s| clean(s))).filter(|s| s.contains(['/', '-', '.']) && s.bytes().any(|b| b.is_ascii_digit())).take(2000).collect();
    let (auto_order, ambiguous) = detect_order(&order_samples);
    let order = match opts.date_order {
        DateOrder::DayFirst => Dmy::Dmy,
        DateOrder::MonthFirst => Dmy::Mdy,
        DateOrder::YearFirst => Dmy::Ymd,
        DateOrder::Auto => meta.order.unwrap_or(auto_order),
    };
    let date_ambiguous = ambiguous && meta.order.is_none() && opts.date_order == DateOrder::Auto;
    let classes: Vec<ColClass> = (0..ncols).map(|i| classify(&samples, i, order)).collect();

    // Gene-expression / ZT-CT wide table: first column IDs, other column headers are time points
    if shape.header.is_some() && opts.layout != Layout::Channels {
        let tl: Vec<Option<f64>> = headers.iter().map(|h| c.time_label.captures(h).and_then(|m| m.get(2)?.as_str().parse().ok())).collect();
        let n_tl = tl[1..].iter().filter(|x| x.is_some()).count();
        if n_tl >= 4 && n_tl * 10 >= (ncols - 1) * 8 && (classes[0].num < 0.5 || c.id_col.is_match(&headers[0])) {
            return parse_wide_rows(text, name, opts, row_limit, &shape, &headers, &tl, head);
        }
    }

    // ---- Roles -------------------------------------------------------------------------------
    let mut roles: Vec<ColRole> = vec![ColRole::Ignore; ncols];
    let mut unknown: Vec<usize> = Vec::new();
    for i in 0..ncols {
        let hl = headers[i].to_lowercase();
        let r = if let Some(r) = opts.roles.get(&i) {
            Some(*r)
        } else if let Some(p) = meta.by_name.iter().find(|p| p.0 == hl) {
            Some(p.1)
        } else if positional {
            meta.positional.as_ref().and_then(|p| p.get(i)).map(|p| p.1)
        } else {
            auto_role(&headers[i], &classes[i])
        };
        match r {
            Some(r) => roles[i] = r,
            None => unknown.push(i),
        }
    }
    // Plain x / y / z next to each other → accelerometer
    let xyz: Vec<usize> = (0..ncols).filter(|&i| c.xyz.is_match(&headers[i]) && classes[i].num > 0.6 && !opts.roles.contains_key(&i)).collect();
    if xyz.len() == 3 {
        for &i in &xyz {
            roles[i] = ColRole::Data(match headers[i].to_lowercase().as_str() {
                "x" => Role::AccelX,
                "y" => Role::AccelY,
                _ => Role::AccelZ,
            });
            unknown.retain(|&u| u != i);
        }
    }

    // ---- Time column(s) ------------------------------------------------------------------------
    let forced_time = (0..ncols).find(|&i| opts.roles.get(&i) == Some(&ColRole::Time));
    let forced_date = (0..ncols).find(|&i| opts.roles.get(&i) == Some(&ColRole::Date));
    let name_score = |i: usize| -> i32 {
        let h = headers[i].as_str();
        if c.time_abs.is_match(h) {
            3
        } else if c.date_only.is_match(h) || c.time_rel.is_match(h) || c.tod_only.is_match(h) {
            2
        } else {
            0
        }
    };
    let mut time_kind: Option<TimeKind> = None;
    let mut time_desc = String::new();
    let mut relative = false;
    let mut numeric_time = false;
    let mut needs_interval = false;
    let base = opts.start.or(meta.start);
    let pick_numeric = |i: usize, base: Option<i64>| -> (TimeKind, String, bool) {
        let vals: Vec<f64> = samples.iter().filter_map(|r| r.get(i)).map(|s| num(s, classes[i].comma)).filter(|v| v.is_finite()).collect();
        let med = crate::stats::median(&vals);
        let hl = headers[i].to_lowercase();
        let unit = opts.time_unit.or_else(|| unit_from_header(&headers[i]));
        let date_like = hl.contains("date") || hl.contains("serial") || hl.contains("datenum");
        if opts.time_unit.is_none() && date_like && (20000.0..80000.0).contains(&med) {
            return (TimeKind::Numeric(i, 86_400_000.0, -25569.0 * 86_400_000.0, false), format!("'{}' as Excel serial dates", headers[i]), false);
        }
        if opts.time_unit.is_none() && date_like && (690000.0..800000.0).contains(&med) {
            return (TimeKind::Numeric(i, 86_400_000.0, -719529.0 * 86_400_000.0, false), format!("'{}' as MATLAB datenums", headers[i]), false);
        }
        if opts.time_unit.is_none() && !hl.contains("elapsed") && unit.is_none_or(|u| u == TimeUnit::Seconds || u == TimeUnit::Milliseconds) {
            let (scale, label) = if med > 1e17 {
                (1e-6, "nanoseconds")
            } else if med > 1e14 {
                (1e-3, "microseconds")
            } else if med > 1e11 {
                (1.0, "milliseconds")
            } else if med > 3e8 {
                (1e3, "seconds")
            } else {
                (0.0, "")
            };
            if scale > 0.0 {
                return (TimeKind::Numeric(i, scale, 0.0, true), format!("'{}' as Unix time in {label} (converted to local time)", headers[i]), false);
            }
        }
        let u = unit.unwrap_or(TimeUnit::Seconds);
        let b = base.unwrap_or_else(today_midnight_ms);
        (TimeKind::Numeric(i, u.ms(), b as f64, false), format!("'{}' as {} since {}", headers[i], u.label(), ms_to_naive(b).format("%Y-%m-%d %H:%M")), true)
    };
    if let Some(t) = forced_time {
        let cl = &classes[t];
        if let Some(d) = forced_date {
            time_kind = Some(TimeKind::DateTod(d, t, order));
            time_desc = format!("'{}' + '{}'", headers[d], headers[t]);
        } else if cl.dt > 0.7 {
            time_kind = Some(TimeKind::Text(t, order));
            time_desc = format!("'{}' (date & time)", headers[t]);
        } else if cl.tod > 0.7 {
            let b = base.unwrap_or_else(today_midnight_ms);
            time_kind = Some(TimeKind::Tod(t, b));
            relative = base.is_none();
            time_desc = format!("'{}' clock times from {}", headers[t], ms_to_naive(b).format("%Y-%m-%d"));
        } else {
            let (k, d, rel) = pick_numeric(t, base);
            numeric_time = true;
            relative = rel;
            time_kind = Some(k);
            time_desc = d;
        }
    }
    if time_kind.is_none() {
        // 1) full date-time column (best name score, must vary)
        let mut cands: Vec<usize> = (0..ncols).filter(|&i| classes[i].dt_time > 0.8 && classes[i].distinct && roles[i] != ColRole::Mask).collect();
        cands.sort_by_key(|&i| -name_score(i));
        if let Some(&i) = cands.first() {
            time_kind = Some(TimeKind::Text(i, order));
            time_desc = format!("'{}' (date & time)", headers[i]);
        }
    }
    if time_kind.is_none() {
        // 2) separate date + clock-time columns
        let d = (0..ncols).find(|&i| classes[i].date_only > 0.8);
        let t = (0..ncols).find(|&i| classes[i].tod > 0.8 && Some(i) != d);
        if let (Some(d), Some(t)) = (d, t) {
            time_kind = Some(TimeKind::DateTod(d, t, order));
            time_desc = format!("'{}' + '{}'", headers[d], headers[t]);
        } else if let Some(d) = d.filter(|&d| classes[d].distinct) {
            time_kind = Some(TimeKind::Text(d, order));
            time_desc = format!("'{}' (dates)", headers[d]);
        }
    }
    if time_kind.is_none() {
        // 3) numeric time by name (relative, Unix, Excel serial, ZT/CT)
        let date_named = |h: &str| {
            let l = h.to_lowercase();
            l.contains("date") || l.contains("serial") || l.contains("datenum")
        };
        let mut cands: Vec<usize> = (0..ncols).filter(|&i| classes[i].num > 0.9 && (c.time_rel.is_match(&headers[i]) || c.time_abs.is_match(&headers[i]) || date_named(&headers[i]))).collect();
        cands.sort_by_key(|&i| -name_score(i));
        if let Some(&i) = cands.first() {
            let (k, d, rel) = pick_numeric(i, base);
            numeric_time = true;
            relative = rel;
            time_kind = Some(k);
            time_desc = d;
        }
    }
    if time_kind.is_none() {
        // 4) time-of-day / elapsed hh:mm:ss without a date
        if let Some(i) = (0..ncols).find(|&i| classes[i].tod > 0.8) {
            let b = base.unwrap_or_else(today_midnight_ms);
            time_kind = Some(TimeKind::Tod(i, b));
            relative = base.is_none();
            time_desc = format!("'{}' times from {}", headers[i], ms_to_naive(b).format("%Y-%m-%d %H:%M"));
        }
    }
    if time_kind.is_none() {
        // 5) no time column: start + fixed interval (ActiGraph epochs, user-set interval)
        if let Some(step) = opts.interval_s.or(meta.epoch_s) {
            let b = base.unwrap_or_else(today_midnight_ms);
            time_kind = Some(TimeKind::Implicit(b, step * 1000.0));
            relative = base.is_none();
            time_desc = format!("No time column: every {} from {}", format_duration(step), ms_to_naive(b).format("%Y-%m-%d %H:%M:%S"));
        } else {
            needs_interval = true;
        }
    }
    let Some(time_kind) = time_kind else {
        let mut p = Parsed {
            source: name.into(),
            format: meta.format.unwrap_or("Delimited table").into(),
            preview: preview_lines(head),
            header_line: shape.header,
            data_line: shape.data_start,
            columns: (0..ncols).map(|i| ColumnInfo { name: headers[i].clone(), role: roles[i], example: classes[i].example.clone() }).collect(),
            needs_interval,
            ..Default::default()
        };
        p.notes.push("No time column was recognised. Mark the time column, or set a sampling interval for evenly spaced rows.".into());
        if headers.iter().any(|h| h.contains("Circadian Rhythmometry") && h.contains("Report")) {
            return Err("This is an analysis report generated by CoCirca, not sensor data. Open the data file you analysed instead.".into());
        }
        return Err(format!("{name}: no time column found in {:?}. {}", headers, p.notes[0]));
    };
    let time_cols: Vec<usize> = match time_kind {
        TimeKind::Text(i, _) | TimeKind::Tod(i, _) | TimeKind::Numeric(i, ..) => vec![i],
        TimeKind::DateTod(d, t, _) => vec![d, t],
        TimeKind::Implicit(..) => vec![],
    };
    for &i in &time_cols {
        roles[i] = if time_cols.len() == 2 && i == time_cols[0] { ColRole::Date } else { ColRole::Time };
        unknown.retain(|&u| u != i);
    }
    // Columns we mistook for data because they look like dates
    for i in 0..ncols {
        if !time_cols.contains(&i) && matches!(roles[i], ColRole::Time | ColRole::Date) && !opts.roles.contains_key(&i) {
            roles[i] = ColRole::Ignore;
        }
    }

    // ---- Resolve conflicts & unknown numeric columns -------------------------------------------
    // ActiGraph count axes → vector magnitude unless a VM column exists
    let axis_cols: Vec<usize> = if positional { vec![] } else { (0..ncols).filter(|&i| c.axis.is_match(&headers[i]) && !opts.roles.contains_key(&i)).collect() };
    let vm_rx = Regex::new(r"(?i)^(vector ?magnitude|vm)$").unwrap();
    let vm_col = if positional { None } else { (0..ncols).find(|&i| vm_rx.is_match(&headers[i]) && classes[i].num > 0.6 && !opts.roles.contains_key(&i)) };
    let mut vm_from_axes = false;
    if let Some(v) = vm_col {
        for &a in &axis_cols {
            roles[a] = ColRole::Ignore;
        }
        roles[v] = ColRole::Data(Role::Activity);
    } else if axis_cols.len() == 3 && opts.roles.is_empty() {
        vm_from_axes = true;
        for i in 0..ncols {
            if axis_cols.contains(&i) || roles[i] == ColRole::Data(Role::Activity) {
                roles[i] = ColRole::Ignore;
            }
        }
    }
    // One activity / light / HR column (first, by catalogue priority); extra temperatures are kept
    let keep_one = |roles: &mut Vec<ColRole>, role: ColRole, prefer: &dyn Fn(&str) -> i32| {
        let mut idx: Vec<usize> = (0..ncols).filter(|&i| roles[i] == role && !opts.roles.contains_key(&i)).collect();
        idx.sort_by_key(|&i| -prefer(&headers[i]));
        for &i in idx.iter().skip(1) {
            roles[i] = ColRole::Ignore;
        }
    };
    keep_one(&mut roles, ColRole::Data(Role::Activity), &|h: &str| if c.steps.is_match(h) { 0 } else if Regex::new(r"(?i)^(activity|pim|vector ?magnitude|vm|enmo|counts?)$").unwrap().is_match(h) { 3 } else { 1 });
    keep_one(&mut roles, ColRole::Data(Role::Light), &|h: &str| if Regex::new(r"(?i)^(light|lux|white light|illuminance)").unwrap().is_match(h) { 2 } else { 1 });
    keep_one(&mut roles, ColRole::Data(Role::HeartRate), &|_| 0);
    keep_one(&mut roles, ColRole::RR, &|_| 0);
    let known_data = (0..ncols).any(|i| matches!(roles[i], ColRole::Data(_) | ColRole::RR)) || vm_from_axes;
    let numeric_unknown: Vec<usize> = unknown.iter().copied().filter(|&i| classes[i].num > 0.6 && !time_cols.contains(&i)).collect();
    let sample_named = numeric_unknown.iter().filter(|&&i| c.sample_like.is_match(&headers[i])).count();
    let signal_cols: Vec<usize> = (0..ncols).filter(|&i| roles[i] == ColRole::Data(Role::Signal) && !opts.roles.contains_key(&i)).collect();
    let can_layout = numeric_unknown.len() + signal_cols.len() >= 2 && !(0..ncols).any(|i| matches!(roles[i], ColRole::Data(r) if r != Role::Signal));
    let layout = match opts.layout {
        Layout::Auto if can_layout && (numeric_unknown.len() + signal_cols.len() >= 4 || sample_named >= 2) => Layout::Subjects,
        Layout::Auto => Layout::Channels,
        l => l,
    };
    for &i in &numeric_unknown {
        roles[i] = if !known_data || layout == Layout::Subjects { ColRole::Data(Role::Signal) } else { ColRole::Ignore };
    }

    // ---- Parse every row ------------------------------------------------------------------------
    let data_cols: Vec<usize> = (0..ncols).filter(|&i| matches!(roles[i], ColRole::Data(_) | ColRole::RR)).chain(if vm_from_axes { axis_cols.clone() } else { vec![] }).collect();
    let mask_cols: Vec<usize> = (0..ncols).filter(|&i| roles[i] == ColRole::Mask).collect();
    let comma: Vec<bool> = classes.iter().map(|c| c.comma).collect();
    let data_text = {
        let mut off = 0usize;
        let mut line_no = 0usize;
        for l in text.split_inclusive('\n') {
            if line_no == shape.data_start {
                break;
            }
            off += l.len();
            line_no += 1;
        }
        &text[off.min(text.len())..]
    };
    let plan = RowPlan { delim: shape.delim, time: time_kind, data_cols: data_cols.clone(), mask_cols: mask_cols.clone(), comma: comma.clone(), min_fields: ncols.saturating_sub(2).max(1) };
    let mut rows = parse_rows(data_text, &plan, row_limit);
    let skipped = rows.bad;
    post_process_times(&mut rows, time_kind);
    if rows.times.is_empty() {
        return Err(format!("{name}: no rows could be read with the time column {time_desc}."));
    }
    if skipped > 0 {
        notes.push(format!("{} rows skipped (time could not be read).", group_thousands(skipped)));
    }

    // ---- Build channels --------------------------------------------------------------------------
    let col_values = |i: usize| -> Vec<f64> { rows.cols[data_cols.iter().position(|&d| d == i).unwrap()].clone() };
    let mut channels: Vec<Channel> = Vec::new();
    let mut rr_series: Option<Vec<f64>> = None;
    let mut subject_cols: Vec<usize> = Vec::new();
    for &i in &data_cols {
        if vm_from_axes && axis_cols.contains(&i) {
            continue;
        }
        match roles[i] {
            ColRole::RR => {
                // Intervals in seconds (Empatica / Polar IBI "0.83") → ms
                let mut v = col_values(i);
                let unit = header_unit(&headers[i]).map(|u| u.to_lowercase());
                let secs = match unit.as_deref() {
                    Some("s" | "sec" | "secs" | "seconds") => true,
                    Some("ms" | "msec") => false,
                    _ => crate::stats::median(&v.iter().copied().filter(|x| x.is_finite()).collect::<Vec<_>>()) < 10.0,
                };
                if secs {
                    v.iter_mut().for_each(|x| *x *= 1000.0);
                    notes.push(format!("'{}' read as seconds and converted to ms.", headers[i].trim()));
                }
                rr_series = Some(v);
            }
            ColRole::Data(Role::Signal) if layout == Layout::Subjects => subject_cols.push(i),
            ColRole::Data(r) => {
                let unit = header_unit(&headers[i]).unwrap_or_else(|| default_unit(r, &headers[i]));
                let nm = channel_name(r, &headers[i]);
                channels.push(Channel::new(r, &nm, &unit, col_values(i)));
            }
            _ => {}
        }
    }
    if vm_from_axes {
        let a: Vec<Vec<f64>> = axis_cols.iter().map(|&i| col_values(i)).collect();
        let vm: Vec<f64> = (0..rows.times.len()).map(|k| (a[0][k].powi(2) + a[1][k].powi(2) + a[2][k].powi(2)).sqrt()).collect();
        channels.push(Channel::new(Role::Activity, "Activity", "counts (vector magnitude of axes 1–3)", vm));
        notes.push("Activity = vector magnitude of ActiGraph axes 1–3.".into());
    }
    // iButton unit column
    if let Some(u) = (0..ncols).find(|&i| headers[i].eq_ignore_ascii_case("unit")) {
        if let Some(v) = samples.first().and_then(|r| r.get(u)).map(|s| clean(s).to_uppercase()) {
            for ch in channels.iter_mut().filter(|c| c.role == Role::Temperature) {
                ch.unit = if v.starts_with('F') { "°F".into() } else { "°C".into() };
            }
        }
    }
    let accel_linear = (0..ncols).any(|i| matches!(roles[i], ColRole::Data(Role::AccelX | Role::AccelY | Role::AccelZ)) && (headers[i].to_lowercase().contains("linear") || headers[i].to_lowercase().contains("without g")));
    // Raw device units (Empatica 1/64 g, Axivity 1/256 g, milli-g …) → g, judged from the resting magnitude
    if !accel_linear {
        let axis = |r: Role| channels.iter().position(|c| c.role == r);
        if let (Some(x), Some(y), Some(z)) = (axis(Role::AccelX), axis(Role::AccelY), axis(Role::AccelZ)) {
            let n = channels[x].values.len();
            let step = (n / 200_000).max(1);
            let vm: Vec<f64> = (0..n).step_by(step).map(|k| (channels[x].values[k].powi(2) + channels[y].values[k].powi(2) + channels[z].values[k].powi(2)).sqrt()).collect();
            let med = crate::stats::median(&vm);
            if let (g, Some(desc)) = crate::data::accel_g_units(med) {
                for i in [x, y, z] {
                    channels[i].values.par_iter_mut().for_each(|v| *v /= g);
                    channels[i].unit = "g".into();
                }
                notes.push(format!("Acceleration at rest ≈ {med:.1} units: read as {desc} and converted to g."));
            }
        }
    }
    if !mask_cols.is_empty() {
        let masked = rows.masked.iter().filter(|&&m| m).count();
        for ch in &mut channels {
            for (v, &m) in ch.values.iter_mut().zip(&rows.masked) {
                if m {
                    *v = f64::NAN;
                }
            }
        }
        if masked > 0 {
            notes.push(format!("{} rows flagged as off-wrist / excluded were set to missing.", group_thousands(masked)));
        }
    }
    let format = meta.format.unwrap_or("Delimited table").to_string();
    let mut subjects = Vec::new();
    if let Some(rr) = rr_series {
        let times = rows.times.clone();
        let mut d = rr_dataset(name, &rr, Some(&times), None);
        // distinct from a separate heart-rate file's channel, so the two merge side by side
        d.channels[0].name = "Heart rate (IBI)".into();
        d.channels.extend(channels.drain(..));
        notes.extend(std::mem::take(&mut d.notes)); // added back once, below
        subjects.push(d);
    } else if layout == Layout::Subjects {
        let (sig_name, sig_unit) = if headers.iter().any(|h| c.signal.is_match(h) && h.to_lowercase().contains("lum")) || name.to_lowercase().contains("lum") {
            ("Bioluminescence", "counts")
        } else {
            ("Signal", "")
        };
        for &i in &subject_cols {
            let mut chs = channels.clone();
            chs.push(Channel::new(Role::Signal, sig_name, sig_unit, col_values(i)));
            subjects.push(RawDataset { name: format!("{} · {}", name, headers[i]), format: format.clone(), t_ms: rows.times.clone(), channels: chs, accel_linear, ..Default::default() });
        }
        notes.push(format!("{} columns treated as separate subjects / samples (change under Layout).", subject_cols.len()));
        for d in &mut subjects {
            d.format = "Multi-sample table (one subject per column)".into();
        }
    } else {
        subjects.push(RawDataset { name: name.into(), format: format.clone(), t_ms: rows.times.clone(), channels, accel_linear, ..Default::default() });
    }
    for d in &mut subjects {
        d.sort_by_time();
        d.notes.extend(notes.iter().cloned());
        if relative {
            d.notes.push("Times are relative (no calendar date in the file); clock phases are measured from the chosen start.".into());
        }
    }
    let n = rows.times.len();
    let format = if layout == Layout::Subjects && format == "Delimited table" { "Multi-sample table (one subject per column)".to_string() } else { format };
    Ok(Parsed {
        source: name.into(),
        format,
        notes,
        subjects,
        preview: preview_lines(head),
        header_line: shape.header,
        data_line: shape.data_start,
        columns: (0..ncols)
            .map(|i| ColumnInfo { name: headers[i].clone(), role: if vm_from_axes && axis_cols.contains(&i) { ColRole::Data(Role::Activity) } else { roles[i] }, example: classes[i].example.clone() })
            .collect(),
        time_desc,
        relative_time: relative,
        numeric_time,
        needs_interval,
        date_ambiguous,
        can_layout,
        layout,
        rows: n,
        skipped_rows: skipped,
        truncated: row_limit.is_some_and(|r| n >= r),
    })
}

fn default_unit(r: Role, h: &str) -> String {
    match r {
        Role::Light => "lux".into(),
        Role::HeartRate => "bpm".into(),
        Role::Temperature => {
            if h.contains('F') && h.contains('°') {
                "°F".into()
            } else {
                "°C".into()
            }
        }
        Role::Activity => "activity units".into(),
        Role::Signal if matches!(header_name(h).to_lowercase().as_str(), "eda" | "gsr" | "scl" | "skin conductance" | "electrodermal activity") => "µS".into(),
        _ => String::new(),
    }
}

fn channel_name(r: Role, h: &str) -> String {
    match r {
        Role::Activity => "Activity".into(),
        Role::Light => "Light".into(),
        Role::HeartRate => "Heart rate".into(),
        Role::AccelX => "x".into(),
        Role::AccelY => "y".into(),
        Role::AccelZ => "z".into(),
        Role::Temperature => {
            let n = header_name(h);
            let l = n.to_lowercase();
            if l == "value" || l.starts_with("temp") {
                "Temperature".into()
            } else {
                n
            }
        }
        Role::Signal => {
            let n = header_name(h);
            match n.to_lowercase().as_str() {
                "eda" | "gsr" | "scl" | "skin conductance" | "electrodermal activity" => "EDA".into(),
                "bvp" | "ppg" => n.to_uppercase(),
                _ => n,
            }
        }
    }
}

struct RowPlan {
    delim: Delim,
    time: TimeKind,
    data_cols: Vec<usize>,
    mask_cols: Vec<usize>,
    comma: Vec<bool>,
    min_fields: usize,
}

#[derive(Default)]
struct Rows {
    times: Vec<i64>,
    cols: Vec<Vec<f64>>,
    masked: Vec<bool>,
    bad: usize,
}

fn mask_true(s: &str) -> bool {
    let s = clean(s);
    if s.is_empty() {
        return false;
    }
    match s.parse::<f64>() {
        Ok(v) => v != 0.0,
        Err(_) => {
            let l = s.to_ascii_lowercase();
            l.contains("excluded") || l.contains("off") || l == "true" || l == "yes" || l.contains("nonwear") || l.contains("non-wear")
        }
    }
}

fn parse_chunk(chunk: &str, plan: &RowPlan, limit: usize) -> Rows {
    let mut r = Rows { cols: vec![Vec::new(); plan.data_cols.len()], ..Default::default() };
    let mut f: Vec<&str> = Vec::with_capacity(64);
    for line in chunk.lines() {
        if r.times.len() >= limit {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        split(line, plan.delim, &mut f);
        if f.len() < plan.min_fields.min(2) {
            r.bad += 1;
            continue;
        }
        let t = match plan.time {
            TimeKind::Text(i, o) => f.get(i).and_then(|s| parse_dt(clean(s), o)),
            TimeKind::DateTod(d, t, o) => match (f.get(d).and_then(|s| parse_dt(clean(s), o)), f.get(t).and_then(|s| parse_tod(clean(s)))) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            },
            TimeKind::Tod(i, _) => f.get(i).and_then(|s| parse_tod(clean(s))),
            TimeKind::Numeric(i, scale, off, _) => f.get(i).map(|s| num(s, plan.comma[i])).filter(|v| v.is_finite()).map(|v| (v * scale + off).round() as i64),
            TimeKind::Implicit(..) => Some(0),
        };
        let Some(t) = t else {
            r.bad += 1;
            continue;
        };
        r.times.push(t);
        for (k, &c) in plan.data_cols.iter().enumerate() {
            r.cols[k].push(f.get(c).map_or(f64::NAN, |s| num(s, plan.comma[c])));
        }
        if !plan.mask_cols.is_empty() {
            r.masked.push(plan.mask_cols.iter().any(|&m| f.get(m).is_some_and(|s| mask_true(s))));
        }
    }
    r
}

/// Parse all rows: large inputs are split at line boundaries and parsed in parallel.
fn parse_rows(data: &str, plan: &RowPlan, row_limit: Option<usize>) -> Rows {
    if let Some(lim) = row_limit {
        return parse_chunk(data, plan, lim);
    }
    const CHUNK: usize = 1 << 22;
    let mut bounds = vec![0usize];
    let b = data.as_bytes();
    while *bounds.last().unwrap() < b.len() {
        let mut e = (*bounds.last().unwrap() + CHUNK).min(b.len());
        while e < b.len() && b[e - 1] != b'\n' {
            e += 1;
        }
        bounds.push(e);
    }
    let parts: Vec<Rows> = bounds.windows(2).collect::<Vec<_>>().par_iter().map(|w| parse_chunk(&data[w[0]..w[1]], plan, usize::MAX)).collect();
    let total: usize = parts.iter().map(|p| p.times.len()).sum();
    let mut out = Rows { times: Vec::with_capacity(total), cols: vec![Vec::with_capacity(total); plan.data_cols.len()], ..Default::default() };
    for p in parts {
        out.times.extend(p.times);
        for (k, c) in p.cols.into_iter().enumerate() {
            out.cols[k].extend(c);
        }
        out.masked.extend(p.masked);
        out.bad += p.bad;
    }
    out
}

fn post_process_times(rows: &mut Rows, kind: TimeKind) {
    match kind {
        TimeKind::Implicit(start, step) => {
            for (i, t) in rows.times.iter_mut().enumerate() {
                *t = start + (i as f64 * step).round() as i64;
            }
        }
        TimeKind::Tod(_, base) => {
            // Clock times without dates: a drop of > 12 h means the next day
            let mut day = 0i64;
            let mut prev = i64::MIN;
            for t in rows.times.iter_mut() {
                if prev != i64::MIN && *t + 12 * 3_600_000 < prev {
                    day += MS_PER_DAY;
                }
                prev = *t;
                *t = base + day + *t;
            }
        }
        TimeKind::Numeric(_, _, _, true) => {
            // Unix UTC → local wall-clock time (one offset unless the recording crosses a DST change)
            let (Some(&a), Some(&b)) = (rows.times.first(), rows.times.last()) else { return };
            let off = |ms: i64| unix_ms_to_local(ms as f64) - ms;
            let (oa, ob) = (off(a), off(b));
            if oa == ob {
                for t in rows.times.iter_mut() {
                    *t += oa;
                }
            } else {
                let mut cache = (i64::MIN, 0i64);
                for t in rows.times.iter_mut() {
                    let hour = t.div_euclid(3_600_000);
                    if hour != cache.0 {
                        cache = (hour, off(*t));
                    }
                    *t += cache.1;
                }
            }
        }
        _ => {}
    }
}

// ------------------------------------------------------------------------------------------------
// TriKinetics DAM monitor files: one subject (fly / mouse) per channel
// ------------------------------------------------------------------------------------------------

fn parse_dam(text: &str, name: &str, row_limit: Option<usize>, shape: &Shape, head: &[&str]) -> Result<Parsed, String> {
    let ncol = shape.ncols;
    let n_ch = ncol - 10;
    let mut times = Vec::new();
    let mut light = Vec::new();
    let mut ch: Vec<Vec<f64>> = vec![Vec::new(); n_ch];
    let mut f = Vec::new();
    let mut bad_status = 0usize;
    for line in text.lines().skip(shape.data_start) {
        if line.trim().is_empty() {
            continue;
        }
        split(line, Delim::Tab, &mut f);
        if f.len() < ncol {
            continue;
        }
        if clean(f[3]) != "1" {
            bad_status += 1;
            continue;
        }
        let Some(t) = parse_dt(&format!("{} {}", clean(f[1]), clean(f[2])), Dmy::Dmy) else { continue };
        times.push(t);
        light.push(num(f[9], false));
        for k in 0..n_ch {
            ch[k].push(num(f[10 + k], false));
        }
        if row_limit.is_some_and(|r| times.len() >= r) {
            break;
        }
    }
    let light_varies = light.iter().any(|&v| v == 1.0) && light.iter().any(|&v| v == 0.0);
    let mut subjects = Vec::new();
    let mut empty = 0;
    for (k, v) in ch.into_iter().enumerate() {
        if v.iter().all(|&x| !x.is_finite() || x == 0.0) {
            empty += 1;
            continue;
        }
        let mut chans = vec![Channel::new(Role::Activity, "Activity", "beam crossings", v)];
        if light_varies {
            chans.push(Channel::new(Role::Light, "Light", "on/off", light.iter().map(|&x| x * 1000.0).collect()));
        }
        subjects.push(RawDataset { name: format!("{} · channel {}", name, k + 1), format: "TriKinetics DAM".into(), t_ms: times.clone(), channels: chans, ..Default::default() });
    }
    let mut notes = vec![format!("{} active channels (one animal each); {empty} empty channels skipped.", subjects.len())];
    if bad_status > 0 {
        notes.push(format!("{bad_status} readings with monitor status ≠ 1 skipped."));
    }
    if light_varies {
        notes.push("Column 10 (light on/off) is used as the light channel (on = 1000 lux).".into());
    }
    if subjects.is_empty() {
        return Err(format!("{name}: all DAM channels are empty."));
    }
    let n = times.len();
    Ok(Parsed {
        source: name.into(),
        format: "TriKinetics DAM monitor".into(),
        notes,
        subjects,
        preview: preview_lines(head),
        header_line: None,
        data_line: shape.data_start,
        columns: vec![
            ColumnInfo { name: "Date + time (columns 2–3)".into(), role: ColRole::Time, example: String::new() },
            ColumnInfo { name: format!("Channels 1–{n_ch} (columns 11–{ncol})"), role: ColRole::Data(Role::Activity), example: String::new() },
        ],
        time_desc: "Columns 2 (date, e.g. 26 Feb 18) + 3 (time)".into(),
        rows: n,
        truncated: row_limit.is_some_and(|r| n >= r),
        ..Default::default()
    })
}

// ------------------------------------------------------------------------------------------------
// Wide tables with time points as column headers (gene expression, hormone panels)
// ------------------------------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn parse_wide_rows(text: &str, name: &str, opts: &ImportOptions, row_limit: Option<usize>, shape: &Shape, headers: &[String], tl: &[Option<f64>], head: &[&str]) -> Result<Parsed, String> {
    let base = opts.start.unwrap_or_else(today_midnight_ms);
    let unit = opts.time_unit.unwrap_or(TimeUnit::Hours).ms();
    let cols: Vec<(usize, i64)> = tl.iter().enumerate().skip(1).filter_map(|(i, h)| h.map(|h| (i, base + (h * unit).round() as i64))).collect();
    let mut f = Vec::new();
    let mut subjects = Vec::new();
    let comma = shape.delim != Delim::Comma;
    for line in text.lines().skip(shape.data_start) {
        if line.trim().is_empty() {
            continue;
        }
        split(line, shape.delim, &mut f);
        let id = clean(f.first().copied().unwrap_or("")).to_string();
        let mut pts: Vec<(i64, f64)> = cols.iter().map(|&(i, t)| (t, f.get(i).map_or(f64::NAN, |s| num(s, comma && s.contains(','))))).collect();
        pts.sort_by_key(|p| p.0);
        subjects.push(RawDataset {
            name: if id.is_empty() { format!("{name} · row {}", subjects.len() + 1) } else { id },
            format: "Time-course table (time points as columns)".into(),
            t_ms: pts.iter().map(|p| p.0).collect(),
            channels: vec![Channel::new(Role::Signal, "Expression", "", pts.iter().map(|p| p.1).collect())],
            ..Default::default()
        });
        if row_limit.is_some_and(|r| subjects.len() >= r) {
            break;
        }
    }
    let label = headers[cols[0].0].to_uppercase();
    let zt = label.starts_with("ZT") || label.starts_with("CT");
    let notes = vec![
        format!("{} rows (genes / features) × {} time points; each row is analysed as a separate subject.", group_thousands(subjects.len()), cols.len()),
        if zt { "Times come from ZT / CT column headers (hours); ZT0 / CT0 = 00:00 on the chosen start date.".into() } else { "Times come from the column headers (hours).".into() },
        "Sparse sampling: cosinor fits are reliable; periodograms need ≥ 2 cycles at ≤ 4 h spacing.".into(),
    ];
    let n = subjects.len();
    for d in &mut subjects {
        d.notes = notes.clone();
    }
    Ok(Parsed {
        source: name.into(),
        format: "Time-course table (ZT / CT columns)".into(),
        notes,
        subjects,
        preview: preview_lines(head),
        header_line: shape.header,
        data_line: shape.data_start,
        columns: std::iter::once(ColumnInfo { name: headers[0].clone(), role: ColRole::Ignore, example: String::new() })
            .chain(cols.iter().map(|&(i, _)| ColumnInfo { name: headers[i].clone(), role: ColRole::Time, example: String::new() }))
            .collect(),
        time_desc: format!("Column headers as {} from {}", opts.time_unit.unwrap_or(TimeUnit::Hours).label(), ms_to_naive(base).format("%Y-%m-%d %H:%M")),
        relative_time: true,
        numeric_time: true,
        rows: n,
        truncated: row_limit.is_some_and(|r| n >= r),
        ..Default::default()
    })
}

// ================================================================================================
// Combining files
// ================================================================================================

/// Combine parsed files into subjects. Auto merges files that hold different sensors of one recording
/// (e.g. Light.csv + Accelerometer.csv) or consecutive time ranges, and keeps overlapping recordings of
/// the same kind (different people / animals) as separate subjects.
/// Whether `combine` turns these recordings into one. Auto merges files with different channels (sensors of
/// one recording) or consecutive time ranges, and keeps multi-subject files and overlapping recordings apart.
pub fn will_merge(sets: &[&RawDataset], single_each: bool, mode: MergeMode) -> bool {
    match mode {
        MergeMode::Merge => true,
        MergeMode::Separate => false,
        MergeMode::Auto => {
            if !single_each || sets.len() < 2 {
                return false;
            }
            let keys: Vec<Vec<(Role, &str)>> = sets.iter().map(|d| d.channels.iter().map(|c| (c.role, c.name.as_str())).collect()).collect();
            let disjoint = (0..keys.len()).all(|i| (i + 1..keys.len()).all(|j| keys[i].iter().all(|k| !keys[j].contains(k))));
            let mut ranges: Vec<(i64, i64)> = sets.iter().filter_map(|d| d.time_range()).collect();
            ranges.sort();
            let consecutive = ranges.windows(2).all(|w| w[1].0 > w[0].1);
            disjoint || consecutive
        }
    }
}

pub fn combine(parsed: Vec<Parsed>, mode: MergeMode) -> ImportResult {
    let mut notes = Vec::new();
    let mut formats: Vec<String> = Vec::new();
    for p in &parsed {
        if !formats.contains(&p.format) {
            formats.push(p.format.clone());
        }
    }
    let single_each = parsed.iter().all(|p| p.subjects.len() == 1);
    let all: Vec<RawDataset> = parsed.into_iter().flat_map(|p| p.subjects).filter(|d| !d.is_empty()).collect();
    if all.len() <= 1 {
        return ImportResult { subjects: all, notes, formats };
    }
    let merge = will_merge(&all.iter().collect::<Vec<_>>(), single_each, mode);
    if merge {
        notes.push(format!("{} files merged into one recording.", all.len()));
        match RawDataset::merge(all) {
            Ok(d) => ImportResult { subjects: vec![d], notes, formats },
            Err(e) => {
                notes.push(e);
                ImportResult { subjects: vec![], notes, formats }
            }
        }
    } else {
        notes.push(format!("{} subjects (analyse one at a time, or all together with \"Analyse all subjects\").", all.len()));
        ImportResult { subjects: all, notes, formats }
    }
}

/// Read, parse and combine a path in one call (CLI, tests).
pub fn load_path(path: &Path, opts: &ImportOptions, mode: MergeMode) -> Result<ImportResult, String> {
    let sources = read_path(path)?;
    let results: Vec<(String, Result<Parsed, String>)> = sources.par_iter().map(|s| (s.name.clone(), parse_source(s, opts, None))).collect();
    let mut errors = Vec::new();
    let mut ok = Vec::new();
    for (n, r) in results {
        match r {
            Ok(p) => ok.push(p),
            Err(e) => errors.push(format!("{n}: {}", e.lines().next().unwrap_or(""))),
        }
    }
    if ok.is_empty() {
        return Err(errors.join("\n"));
    }
    let mut res = combine(ok, mode);
    for e in errors {
        res.notes.push(format!("skipped {e}"));
    }
    Ok(res)
}

// ================================================================================================
// Tests (fixtures follow the published layouts of each format)
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(name: &str, text: &str) -> Parsed {
        parse_source(&Source { name: name.into(), bytes: text.as_bytes().to_vec(), start_hint: None }, &ImportOptions::default(), None).unwrap_or_else(|e| panic!("{name}: {e}"))
    }
    fn hhmm(ms: i64) -> String {
        ms_to_naive(ms).format("%Y-%m-%d %H:%M:%S").to_string()
    }

    #[test]
    fn datetime_forms() {
        let cases = [
            ("2023-01-26 01:00:00", Dmy::Mdy, "2023-01-26 01:00:00"),
            ("2023-01-26T01:00:00.250+02:00", Dmy::Mdy, "2023-01-26 01:00:00"),
            ("26/01/2023 13:30", Dmy::Dmy, "2023-01-26 13:30:00"),
            ("1/26/2023 1:30:05 PM", Dmy::Mdy, "2023-01-26 13:30:05"),
            ("26 Feb 18 09:30:00", Dmy::Mdy, "2018-02-26 09:30:00"),
            ("21-Mar-2017 10:00", Dmy::Mdy, "2017-03-21 10:00:00"),
            ("2013-08-22 11:12:41:500", Dmy::Ymd, "2013-08-22 11:12:41"),
            ("01.03.2019 12:00:00", Dmy::Dmy, "2019-03-01 12:00:00"),
            ("06/01/20 10:00:00 AM", Dmy::Mdy, "2020-06-01 10:00:00"),
            ("20200115", Dmy::Ymd, "2020-01-15 00:00:00"),
        ];
        for (s, o, want) in cases {
            assert_eq!(parse_dt(s, o).map(hhmm).as_deref(), Some(want), "{s}");
        }
        assert_eq!(parse_dt("2013-08-22 11:12:41:500", Dmy::Ymd).unwrap() % 1000, 500);
        assert_eq!(parse_tod("12:30:00 AM"), Some(30 * 60_000));
        assert_eq!(detect_order(&["01/02/2020 23:50", "01/02/2020 23:55", "02/02/2020 00:00"]).0, Dmy::Dmy);
        assert_eq!(detect_order(&["13/02/2020"]).0, Dmy::Dmy);
        assert_eq!(detect_order(&["02/13/2020"]).0, Dmy::Mdy);
        // ambiguous days: the reading spanning hours, not months, wins
        assert_eq!(detect_order(&["10/02/24 23:50", "11/02/24 00:00", "12/02/24 00:10"]).0, Dmy::Dmy);
        assert_eq!(detect_order(&["02/10/24 23:50", "02/11/24 00:00", "02/12/24 00:10"]).0, Dmy::Mdy);
    }

    #[test]
    fn generic_csv_and_separate_date_time() {
        let p = parse("a.csv", "timestamp,activity\n2023-01-26 01:00:00,0\n2023-01-26 02:00:00,5\n2023-01-26 03:00:00,7\n");
        assert_eq!(p.subjects[0].channel(Role::Activity).unwrap().values, vec![0.0, 5.0, 7.0]);
        let p = parse("b.csv", "Date;Time;Activity;Light\n26.01.2023;23:00;1,5;10\n26.01.2023;23:30;2,5;20\n27.01.2023;00:00;3,5;30\n");
        assert_eq!(hhmm(p.subjects[0].t_ms[2]), "2023-01-27 00:00:00");
        assert_eq!(p.subjects[0].channel(Role::Activity).unwrap().values[0], 1.5);
        assert!(p.subjects[0].has_light());
    }

    #[test]
    fn actigraph_with_and_without_columns() {
        let pre = "------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.3 Firmware v2.5.0 date format d/M/yyyy at 30 Hz  Filter Normal -----------\nSerial Number: MOS2D12345678\nStart Time 08:00:00\nStart Date 15/1/2020\nEpoch Period (hh:mm:ss) 00:01:00\nDownload Time 10:00:00\nDownload Date 22/1/2020\nCurrent Memory Address: 0\nCurrent Battery Voltage: 4.15     Mode = 61\n--------------------------------------------------\n";
        let headerless = format!("{pre}120,80,95,10,250,0,1,0,0,173\n60,30,20,4,100,0,1,0,0,70\n0,0,0,0,5,1,0,0,0,0\n");
        let p = parse("ag.csv", &headerless);
        assert_eq!(p.format, "ActiGraph ActiLife CSV");
        assert_eq!(hhmm(p.subjects[0].t_ms[1]), "2020-01-15 08:01:00");
        assert_eq!(p.subjects[0].channel(Role::Activity).unwrap().values[0], 120.0);
        assert_eq!(p.subjects[0].channel(Role::Light).unwrap().values[1], 100.0);
        let with = format!("{pre}Date,Time,Axis1,Axis2,Axis3,Steps,Lux\n15/1/2020,08:00:00,3,4,12,0,250\n15/1/2020,08:01:00,0,0,0,0,240\n15/1/2020,08:02:00,1,0,0,0,230\n");
        let p = parse("ag2.csv", &with);
        assert_eq!(p.subjects[0].channel(Role::Activity).unwrap().values[0], 13.0); // vector magnitude
        assert_eq!(hhmm(p.subjects[0].t_ms[2]), "2020-01-15 08:02:00");
    }

    #[test]
    fn actiware_masks_off_wrist() {
        let t = "\"Actiware Export File\"\n\"Identity:\",\"S01\"\n\n\"-------------------- Epoch-by-Epoch Data -------------------\"\n\"Line\",\"Date\",\"Time\",\"Off-Wrist Status\",\"Activity\",\"Marker\",\"White Light\",\"Red Light\",\"Sleep/Wake\",\"Interval Status\"\n\"1\",\"3/20/2019\",\"12:00:00 PM\",\"0\",\"120\",\"0\",\"300.5\",\"10\",\"1\",\"ACTIVE\"\n\"2\",\"3/20/2019\",\"12:00:30 PM\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"EXCLUDED\"\n\"3\",\"3/20/2019\",\"12:01:00 PM\",\"0\",\"95\",\"0\",\"280\",\"9\",\"1\",\"ACTIVE\"\n";
        let p = parse("aw.csv", t);
        assert_eq!(p.format, "Philips Respironics Actiware");
        let d = &p.subjects[0];
        assert_eq!(hhmm(d.t_ms[0]), "2019-03-20 12:00:00");
        assert!(d.channel(Role::Activity).unwrap().values[1].is_nan());
        assert_eq!(d.channel(Role::Light).unwrap().values[2], 280.0);
    }

    #[test]
    fn acttrust_awd_dam() {
        let at = "+-----------------------+\n| Condor Instruments    |\n+-----------------------+\nDATE/TIME;MS;EVENT;TEMPERATURE;EXT TEMPERATURE;ORIENTATION;PIM;PIMn;TAT;TATn;ZCM;ZCMn;LIGHT;AMB LIGHT;STATE\n01/03/2019 12:00:00;0;0;33,5;25,1;0;1200;10;50;1;30;1;150,5;120;0\n01/03/2019 12:01:00;0;0;33,6;25,0;0;900;8;40;1;20;1;140;110;0\n01/03/2019 12:02:00;0;0;33,7;25,0;0;0;0;0;0;0;0;0;0;1\n";
        let p = parse("at.txt", at);
        assert_eq!(p.format, "Condor ActTrust");
        let d = &p.subjects[0];
        assert_eq!(d.channel(Role::Activity).unwrap().values[0], 1200.0);
        assert_eq!(d.channel(Role::Temperature).unwrap().values[0], 33.5);
        assert_eq!(d.channel(Role::Light).unwrap().values[0], 150.5);
        assert_eq!(hhmm(d.t_ms[0]), "2019-03-01 12:00:00");

        let awd = "S01\n21-Mar-2017\n10:00\n4\n33\nV12345\nM\n0 , M\n12\n30\n5\n";
        let p = parse("s01.awd", awd);
        assert_eq!(p.subjects[0].t_ms.len(), 4);
        assert_eq!(hhmm(p.subjects[0].t_ms[3]), "2017-03-21 10:03:00");

        let mut dam = String::new();
        for i in 0..5 {
            let mut row = format!("{}\t26 Feb 18\t09:3{}:00\t1\t0\t0\t0\t0\t0\t{}", i + 1, i, i % 2);
            for ch in 0..32 {
                row += &format!("\t{}", if ch < 3 { (i * (ch + 1)) as i32 } else { 0 });
            }
            dam += &row;
            dam += "\n";
        }
        let p = parse("Monitor1.txt", &dam);
        assert_eq!(p.format, "TriKinetics DAM monitor");
        assert_eq!(p.subjects.len(), 3);
        assert_eq!(p.subjects[1].channel(Role::Activity).unwrap().values[4], 8.0);
    }

    #[test]
    fn hobo_ibutton_geneactiv_polar() {
        let hobo = "\"Plot Title: 10234567\"\n\"#\",\"Date Time, GMT-05:00\",\"Temp, °F (LGR S/N: 10234567)\",\"Intensity, Lux (LGR S/N: 10234567)\"\n1,06/01/20 10:00:00 AM,72.1,1234.5\n2,06/01/20 10:05:00 AM,72.4,1500\n3,06/01/20 10:10:00 AM,72.6,1800\n";
        let p = parse("hobo.csv", hobo);
        assert_eq!(p.format, "HOBO logger (HOBOware CSV)");
        let d = &p.subjects[0];
        assert_eq!(d.channel(Role::Light).unwrap().values[2], 1800.0);
        assert_eq!(d.channel(Role::Temperature).unwrap().unit, "°F");
        assert_eq!(hhmm(d.t_ms[1]), "2020-06-01 10:05:00");

        let ib = "1-Wire/iButton Part Number: DS1921H\nDate/Time,Unit,Value\n01/02/20 08:00:01 AM,C,36.5\n01/02/20 08:10:01 AM,C,36.6\n01/02/20 08:20:01 AM,C,36.4\n";
        let p = parse("ibutton.csv", ib);
        assert_eq!(p.subjects[0].channel(Role::Temperature).unwrap().values[1], 36.6);

        let mut ge = String::from("Device Type,GENEActiv\nDevice Model,1.1\nMeasurement Frequency,100 Hz\n");
        for i in 0..5 {
            ge += &format!("2013-08-22 11:12:41:{:03},0.05,-0.95,0.23,{},0,23.6\n", i * 10, 10 + i);
        }
        let p = parse("ge.csv", &ge);
        assert_eq!(p.format, "GENEActiv CSV");
        assert!(p.subjects[0].has_accel());
        assert_eq!(p.subjects[0].t_ms[1] - p.subjects[0].t_ms[0], 10);

        let polar = "Name,Sport,Date,Start time,Duration\nJo,RUNNING,15-01-2020,08:00:00,01:00:00\nSample rate,Time,HR (bpm),Speed (km/h)\n1,00:00:00,80,0\n1,00:00:01,82,1.2\n1,00:00:02,85,2.0\n";
        let p = parse("polar.csv", polar);
        assert_eq!(p.format, "Polar CSV");
        assert_eq!(hhmm(p.subjects[0].t_ms[2]), "2020-01-15 08:00:02");
        assert_eq!(p.subjects[0].channel(Role::HeartRate).unwrap().values[1], 82.0);
    }

    #[test]
    fn numeric_times_and_wide_tables() {
        let p = parse("phyphox.csv", "\"Time (s)\",\"Illuminance (lx)\"\n0.0,10\n0.5,11\n1.0,12\n");
        assert_eq!(p.subjects[0].t_ms[1] - p.subjects[0].t_ms[0], 500);
        assert!(p.relative_time);
        let p = parse("sl.csv", "time,seconds_elapsed,z,y,x\n1700000000000000000,0,9.8,0.1,0.2\n1700000000500000000,0.5,9.7,0.1,0.2\n1700000001000000000,1.0,9.8,0.2,0.1\n");
        assert!(p.subjects[0].has_accel());
        assert_eq!(p.subjects[0].t_ms[2] - p.subjects[0].t_ms[0], 1000);
        let p = parse("xl.csv", "Date serial,Activity\n44927.0,1\n44927.5,2\n44928.0,3\n");
        assert_eq!(hhmm(p.subjects[0].t_ms[1]), "2023-01-01 12:00:00");
        let p = parse("zt.csv", "ZT,Melatonin (pg/mL)\n0,5\n4,3\n8,2\n12,10\n16,60\n20,40\n");
        assert_eq!(p.subjects[0].t_ms[1] - p.subjects[0].t_ms[0], 4 * 3_600_000);
        assert_eq!(p.subjects[0].channels[0].unit, "pg/mL");
        // plate reader: one subject per well
        let p = parse("lumi.csv", "Time (h),A1,A2,A3,A4\n0,100,200,150,90\n1,110,190,160,95\n2,120,180,170,99\n");
        assert_eq!(p.layout, Layout::Subjects);
        assert_eq!(p.subjects.len(), 4);
        // gene expression: one subject per row
        let p = parse("jtk.csv", "ID,ZT0,ZT4,ZT8,ZT12,ZT16,ZT20\nPer2,1,2,5,9,6,2\nBmal1,9,6,2,1,3,7\n");
        assert_eq!(p.subjects.len(), 2);
        assert_eq!(p.subjects[1].name, "Bmal1");
        assert_eq!(p.subjects[0].t_ms[5] - p.subjects[0].t_ms[0], 20 * 3_600_000);
    }

    #[test]
    fn json_xml_empatica_rr() {
        let fb = r#"[{"dateTime":"01/01/20 00:00:05","value":{"bpm":72,"confidence":2}},{"dateTime":"01/01/20 00:00:10","value":{"bpm":74,"confidence":2}}]"#;
        let p = parse("heart_rate-2020-01-01.json", fb);
        assert_eq!(p.subjects[0].channel(Role::HeartRate).unwrap().values, vec![72.0, 74.0]);
        let xml = r#"<?xml version="1.0"?><HealthData><Record type="HKQuantityTypeIdentifierHeartRate" sourceName="Watch" unit="count/min" startDate="2020-01-01 10:00:00 +0100" endDate="2020-01-01 10:00:00 +0100" value="61"/>
<Record type="HKQuantityTypeIdentifierStepCount" sourceName="Watch" unit="count" startDate="2020-01-01 10:01:00 +0100" endDate="2020-01-01 10:02:00 +0100" value="30"/></HealthData>"#;
        let p = parse("export.xml", xml);
        let d = &p.subjects[0];
        assert_eq!(hhmm(d.t_ms[0]), "2020-01-01 10:00:00");
        assert!(d.channel(Role::Activity).unwrap().sum);
        let e4 = "1495437325.000000\n1.000000\n36.1\n36.2\n36.3\n";
        let p = parse("TEMP.csv", e4);
        assert_eq!(p.subjects[0].channel(Role::Temperature).unwrap().values.len(), 3);
        let rr: String = (0..150).map(|_| "800\n").collect();
        let p = parse("000.txt", &rr);
        assert_eq!(p.format, "RR / inter-beat intervals");
        let p = parse("ibi.csv", &format!("IBI (s)\n{}", "0.8\n".repeat(150)));
        assert!((p.subjects[0].channel(Role::HeartRate).unwrap().values[0] - 75.0).abs() < 1e-9);
        // time-stamped IBI table in seconds (BIG IDEAs / Empatica re-export), with an extra column
        let mut t = String::from("datetime, ibi, temp\n");
        for i in 0..60 {
            let v = if i == 30 { "3.5" } else { "0.8" }; // one artifact
            t += &format!("2020-02-13 15:{:02}:{:02}.000000,{v},33.{}\n", i / 60 + 33, i % 60, i % 10);
        }
        let p = parse("IBI_001.csv", &t);
        let d = &p.subjects[0];
        let hr = &d.channel(Role::HeartRate).unwrap().values;
        assert_eq!((d.t_ms.len(), hr.len(), d.channel(Role::Temperature).unwrap().values.len()), (60, 60, 60));
        assert!((hr[0] - 75.0).abs() < 1e-9 && hr[30].is_nan());
        assert_eq!(d.channel(Role::HeartRate).unwrap().name, "Heart rate (IBI)");
        assert_eq!(d.notes.iter().filter(|n| n.starts_with("RR intervals")).count(), 1);
    }

    #[test]
    fn raw_accel_units_and_eda() {
        // Empatica re-export: axes in 1/64 g, sub-second time stamps
        let mut t = String::from("datetime, acc_x, acc_y, acc_z\n");
        for i in 0..64 {
            t += &format!("2020-02-13 15:28:{:02}.{:06},{},{},{}\n", 50 + i / 32, (i % 32) * 31250, -2 + i % 3, 1, 64);
        }
        let p = parse("ACC_001.csv", &t);
        let z = p.subjects[0].channel(Role::AccelZ).unwrap();
        assert_eq!(z.unit, "g");
        assert!((z.values[0] - 1.0).abs() < 1e-12);
        assert!(p.notes.iter().any(|n| n.contains("1/64 g")));
        // already in g / m/s²: untouched
        let p = parse("a.csv", "time,ax,ay,az\n2020-01-01 00:00:00,0,0,9.8\n2020-01-01 00:00:01,0,0,9.7\n2020-01-01 00:00:02,0,0,9.9\n");
        assert!((p.subjects[0].channel(Role::AccelZ).unwrap().values[0] - 9.8).abs() < 1e-12);
        assert_eq!(crate::data::accel_g_units(1.02).0, 1.0);
        assert_eq!(crate::data::accel_g_units(250.0).0, 256.0);
        assert_eq!(crate::data::accel_g_units(37.0).0, 37.0);
        let p = parse("EDA_001.csv", "datetime, eda\n2020-02-13 15:28:50.000,0.1\n2020-02-13 15:28:50.250,0.2\n2020-02-13 15:28:50.500,0.3\n");
        let c = &p.subjects[0].channels[0];
        assert_eq!((c.name.as_str(), c.unit.as_str()), ("EDA", "µS"));
    }

    #[test]
    fn utf16_and_cp1252() {
        let s = "timestamp\tlux\n2020-01-01 00:00\t5\n2020-01-01 00:01\t6\n2020-01-01 00:02\t7\n";
        let mut b = vec![0xFF, 0xFE];
        for u in s.encode_utf16() {
            b.extend(u.to_le_bytes());
        }
        assert_eq!(decode(&b), s);
        assert_eq!(decode(&[b'T', 0xB0, b'F']), "T°F");
    }

    #[test]
    fn combine_rules() {
        let light = parse("Light.csv", "\"Time (s)\",\"Illuminance (lx)\"\n0,1\n1,2\n2,3\n");
        let acc = parse("Acc.csv", "\"Time (s)\",\"Acceleration x (m/s^2)\",\"Acceleration y (m/s^2)\",\"Acceleration z (m/s^2)\"\n0,0,0,9.8\n1,0,0,9.8\n2,0,0,9.8\n");
        assert_eq!(combine(vec![light, acc], MergeMode::Auto).subjects.len(), 1);
        let a = parse("s1.txt", &"800\n".repeat(150));
        let b = parse("s2.txt", &"700\n".repeat(150));
        assert_eq!(combine(vec![a, b], MergeMode::Auto).subjects.len(), 2);
    }
}
