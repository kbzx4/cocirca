//! Core data model (multi-channel recordings), synthetic demo data and uniform epoch binning.
//!
//! A recording is a time axis plus any number of channels, each with a role (activity, light, heart
//! rate, temperature, …), a display name and a unit. File parsing lives in `import`.
//! Timestamps are naive local times in milliseconds since 1970-01-01 (treated as UTC for arithmetic).

use chrono::{DateTime, Local, NaiveDate, NaiveDateTime, Timelike};
use std::path::Path;

pub const MS_PER_HOUR: f64 = 3_600_000.0;
pub const MS_PER_DAY: i64 = 86_400_000;
pub const MIN_EPOCH_SECONDS: i64 = 1;
pub const MAX_EPOCH_SECONDS: i64 = 6 * 3600;

/// What a channel measures. Activity and the three accelerometer axes combine into the motion channel;
/// light is analysed on a log scale and overlaid on actograms; everything else is analysed as its own channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    Activity,
    AccelX,
    AccelY,
    AccelZ,
    Light,
    HeartRate,
    Temperature,
    Signal,
}

impl Role {
    pub const ALL: [Role; 8] = [Role::Activity, Role::AccelX, Role::AccelY, Role::AccelZ, Role::Light, Role::HeartRate, Role::Temperature, Role::Signal];
    pub fn label(self) -> &'static str {
        match self {
            Role::Activity => "Activity / counts",
            Role::AccelX => "Accelerometer X",
            Role::AccelY => "Accelerometer Y",
            Role::AccelZ => "Accelerometer Z",
            Role::Light => "Light (lux)",
            Role::HeartRate => "Heart rate",
            Role::Temperature => "Temperature",
            Role::Signal => "Other signal",
        }
    }
    pub fn is_motion(self) -> bool {
        matches!(self, Role::Activity | Role::AccelX | Role::AccelY | Role::AccelZ)
    }
}

#[derive(Clone, Debug)]
pub struct Channel {
    pub role: Role,
    pub name: String,
    pub unit: String,
    pub values: Vec<f64>,
    /// Aggregate by sum per bin (event counts such as steps); empty bins inside the recording count as 0
    pub sum: bool,
}

impl Channel {
    pub fn new(role: Role, name: &str, unit: &str, values: Vec<f64>) -> Self {
        Channel { role, name: name.to_string(), unit: unit.to_string(), values, sum: false }
    }
    pub fn any_finite(&self) -> bool {
        self.values.iter().any(|v| v.is_finite())
    }
}

#[derive(Clone, Debug, Default)]
pub struct RawDataset {
    pub name: String,
    /// Detected file format, e.g. "ActiGraph ActiLife CSV"
    pub format: String,
    pub t_ms: Vec<i64>,
    pub channels: Vec<Channel>,
    /// Accelerometer excludes gravity (linear acceleration)
    pub accel_linear: bool,
    /// (RR intervals read, intervals rejected as artifact / ectopic)
    pub rr_counts: Option<(usize, usize)>,
    /// Loader notes shown with the analysis (e.g. missing start time)
    pub notes: Vec<String>,
}

impl RawDataset {
    pub fn len(&self) -> usize {
        self.t_ms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t_ms.is_empty()
    }
    pub fn channel(&self, role: Role) -> Option<&Channel> {
        self.channels.iter().find(|c| c.role == role && c.any_finite())
    }
    pub fn has_light(&self) -> bool {
        self.channel(Role::Light).is_some()
    }
    pub fn has_accel(&self) -> bool {
        [Role::AccelX, Role::AccelY, Role::AccelZ].iter().all(|&r| self.channel(r).is_some())
    }
    pub fn has_motion(&self) -> bool {
        self.channel(Role::Activity).is_some() || self.has_accel()
    }
    /// Channels analysed on their own (heart rate, temperature, other signals)
    pub fn signals(&self) -> impl Iterator<Item = &Channel> {
        self.channels.iter().filter(|c| matches!(c.role, Role::HeartRate | Role::Temperature | Role::Signal) && c.any_finite())
    }
    /// Short human-readable channel list
    pub fn describe_channels(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.has_accel() {
            out.push(if self.accel_linear { "3-axis linear acceleration".into() } else { "3-axis acceleration".into() });
        }
        for c in self.channels.iter().filter(|c| !matches!(c.role, Role::AccelX | Role::AccelY | Role::AccelZ) && c.any_finite()) {
            out.push(if c.unit.is_empty() { c.name.clone() } else { format!("{} ({})", c.name, c.unit) });
        }
        out
    }
    pub fn time_range(&self) -> Option<(i64, i64)> {
        Some((*self.t_ms.first()?, *self.t_ms.last()?))
    }

    /// Union of datasets on a shared time axis (e.g. Light.csv + Accelerometer.csv), sorted by time.
    /// Channels with the same role and name are concatenated; missing samples are NaN.
    pub fn merge(mut sets: Vec<RawDataset>) -> Result<RawDataset, String> {
        if sets.is_empty() {
            return Err("No datasets to merge.".into());
        }
        if sets.len() == 1 {
            return Ok(sets.pop().unwrap());
        }
        let total: usize = sets.iter().map(|d| d.len()).sum();
        let mut keys: Vec<(Role, String, String, bool)> = Vec::new();
        for d in &sets {
            for c in &d.channels {
                if !keys.iter().any(|k| k.0 == c.role && k.1 == c.name) {
                    keys.push((c.role, c.name.clone(), c.unit.clone(), c.sum));
                }
            }
        }
        let (t, vals) = if sets.iter().all(|d| d.t_ms.windows(2).all(|w| w[0] <= w[1])) {
            // Every set is already in time order (the importer sorts): k-way merge, then fill each channel
            // in parallel. Ties keep set order, exactly like the stable sort of the concatenation below.
            use rayon::prelude::*;
            let ts: Vec<&[i64]> = sets.iter().map(|d| d.t_ms.as_slice()).collect();
            let src = merge_order(&ts);
            let mut pos = vec![0usize; sets.len()];
            let t: Vec<i64> = src.iter().map(|&s| {
                let s = s as usize;
                pos[s] += 1;
                ts[s][pos[s] - 1]
            }).collect();
            let vals: Vec<Vec<f64>> = keys
                .par_iter()
                .map(|key| {
                    let cols: Vec<Option<&[f64]>> = sets.iter().map(|d| d.channels.iter().find(|c| c.role == key.0 && c.name == key.1).map(|c| c.values.as_slice())).collect();
                    let mut pos = vec![0usize; sets.len()];
                    src.iter()
                        .map(|&s| {
                            let s = s as usize;
                            pos[s] += 1;
                            cols[s].map_or(f64::NAN, |v| v[pos[s] - 1])
                        })
                        .collect()
                })
                .collect();
            (t, vals)
        } else {
            let mut t = Vec::with_capacity(total);
            let mut vals: Vec<Vec<f64>> = vec![Vec::with_capacity(total); keys.len()];
            for d in &sets {
                t.extend_from_slice(&d.t_ms);
                for (k, key) in keys.iter().enumerate() {
                    match d.channels.iter().find(|c| c.role == key.0 && c.name == key.1) {
                        Some(c) => vals[k].extend_from_slice(&c.values),
                        None => vals[k].extend(std::iter::repeat_n(f64::NAN, d.len())),
                    }
                }
            }
            (t, vals)
        };
        let mut formats: Vec<&str> = sets.iter().map(|d| d.format.as_str()).filter(|f| !f.is_empty()).collect();
        formats.dedup();
        let mut merged = RawDataset {
            name: sets.iter().map(|d| d.name.as_str()).collect::<Vec<_>>().join(" + "),
            format: formats.join(" + "),
            t_ms: t,
            channels: keys.into_iter().zip(vals).map(|((role, name, unit, sum), values)| Channel { role, name, unit, values, sum }).collect(),
            accel_linear: sets.iter().any(|d| d.accel_linear),
            rr_counts: None,
            notes: sets.iter().flat_map(|d| d.notes.iter().cloned()).collect(),
        };
        merged.sort_by_time();
        Ok(merged)
    }

    pub fn sort_by_time(&mut self) {
        if self.t_ms.windows(2).all(|w| w[0] <= w[1]) {
            return;
        }
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_by_key(|&i| self.t_ms[i]);
        self.t_ms = order.iter().map(|&i| self.t_ms[i]).collect();
        for c in &mut self.channels {
            c.values = order.iter().map(|&i| c.values[i]).collect();
        }
    }
}

/// Which set each row of the time-ordered merge comes from, for sets that are each sorted by time.
/// Ties go to the lower set index (a stable merge).
fn merge_order(ts: &[&[i64]]) -> Vec<u32> {
    let total: usize = ts.iter().map(|t| t.len()).sum();
    let mut out = Vec::with_capacity(total);
    let mut pos = vec![0usize; ts.len()];
    if ts.len() <= 8 {
        loop {
            let mut best = usize::MAX;
            let mut bt = i64::MAX;
            for (s, t) in ts.iter().enumerate() {
                if let Some(&v) = t.get(pos[s]) {
                    if best == usize::MAX || v < bt {
                        best = s;
                        bt = v;
                    }
                }
            }
            if best == usize::MAX {
                break;
            }
            // take the whole run from `best` up to the next set's head time
            let limit = ts.iter().enumerate().filter(|&(s, t)| s != best && pos[s] < t.len()).map(|(s, t)| (t[pos[s]], s)).min();
            let t = ts[best];
            let start = pos[best];
            let mut end = start + 1;
            while end < t.len() && limit.is_none_or(|(lt, ls)| t[end] < lt || (t[end] == lt && best < ls)) {
                end += 1;
            }
            out.extend(std::iter::repeat_n(best as u32, end - start));
            pos[best] = end;
        }
    } else {
        use std::cmp::Reverse;
        let mut heap: std::collections::BinaryHeap<Reverse<(i64, usize)>> = ts.iter().enumerate().filter(|(_, t)| !t.is_empty()).map(|(s, t)| Reverse((t[0], s))).collect();
        while let Some(Reverse((_, s))) = heap.pop() {
            out.push(s as u32);
            pos[s] += 1;
            if let Some(&v) = ts[s].get(pos[s]) {
                heap.push(Reverse((v, s)));
            }
        }
    }
    out
}

/// Load one file as a single recording (all subjects / members merged). Used by the parity test and simple
/// CLI runs; the GUI and batch mode use `import` directly to keep subjects apart.
pub fn load_file(path: &Path, start: Option<i64>) -> Result<RawDataset, String> {
    let opts = crate::import::ImportOptions { start, ..Default::default() };
    let subjects = crate::import::load_path(path, &opts, crate::import::MergeMode::Auto)?.subjects;
    RawDataset::merge(subjects)
}

// ------------------------------------------------------------------------------------------------
// Time helpers
// ------------------------------------------------------------------------------------------------

pub fn naive_to_ms(dt: NaiveDateTime) -> i64 {
    dt.and_utc().timestamp_millis()
}

pub fn ms_to_naive(ms: i64) -> NaiveDateTime {
    DateTime::from_timestamp_millis(ms).map(|d| d.naive_utc()).unwrap_or_default()
}

pub fn today_midnight_ms() -> i64 {
    naive_to_ms(Local::now().date_naive().and_hms_opt(0, 0, 0).unwrap())
}

// ------------------------------------------------------------------------------------------------
// RR / inter-beat intervals → heart rate
// ------------------------------------------------------------------------------------------------

/// Heart rate from RR intervals (ms). `times` gives each beat's time if known (e.g. Empatica IBI);
/// otherwise beat times are the cumulative sum from `start`. Rejects intervals outside 250–2500 ms or
/// more than 25 % from the median of the surrounding 11 beats (artifacts, ectopic / compensatory beats).
pub fn rr_dataset(name: &str, rr: &[f64], times: Option<&[i64]>, start: Option<i64>) -> RawDataset {
    let n = rr.len();
    let mut t_ms = Vec::with_capacity(n);
    let mut hr = Vec::with_capacity(n);
    let base = start.unwrap_or_else(today_midnight_ms);
    let mut elapsed = 0.0f64;
    let mut rejected = 0usize;
    let mut window: Vec<f64> = Vec::with_capacity(11);
    for i in 0..n {
        let v = rr[i];
        if v.is_finite() && v > 0.0 {
            elapsed += v; // every interval occupies time, accepted or not
        }
        window.clear();
        window.extend(rr[i.saturating_sub(5)..(i + 6).min(n)].iter().copied().filter(|x| x.is_finite()));
        window.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let local = window.get(window.len() / 2).copied().unwrap_or(f64::NAN);
        let ok = v.is_finite() && (250.0..=2500.0).contains(&v) && (v - local).abs() <= 0.25 * local;
        rejected += !ok as usize;
        match times {
            // Own time stamps: keep every row (other columns in the same table stay aligned),
            // rejected beats become missing values.
            Some(t) => {
                t_ms.push(t[i]);
                hr.push(if ok { 60_000.0 / v } else { f64::NAN });
            }
            None if ok => {
                t_ms.push(base + elapsed.round() as i64);
                hr.push(60_000.0 / v);
            }
            None => {}
        }
    }
    let hours = match times {
        Some(t) if t.len() > 1 => (t.iter().max().unwrap() - t.iter().min().unwrap()) as f64 / MS_PER_HOUR,
        _ => elapsed / MS_PER_HOUR,
    };
    let mut notes = vec![format!(
        "RR intervals: {n} beats over {hours:.1} h; heart rate = 60000 / RR. {rejected} intervals ({:.2}%) rejected as artifact / ectopic (outside 250–2500 ms or > 25% from the local median).",
        100.0 * rejected as f64 / n.max(1) as f64
    )];
    if start.is_none() && times.is_none() {
        notes.push(format!(
            "No recording start time in the file: times start at {} and clock phases (acrophase etc.) are hours after the recording began, not real clock time.",
            ms_to_naive(base).format("%Y-%m-%d %H:%M")
        ));
    }
    RawDataset {
        name: name.to_string(),
        format: "RR / inter-beat intervals".into(),
        t_ms,
        channels: vec![Channel::new(Role::HeartRate, "Heart rate", "bpm", hr)],
        rr_counts: Some((n, rejected)),
        notes,
        ..Default::default()
    }
}

// ------------------------------------------------------------------------------------------------
// Synthetic demo data
// ------------------------------------------------------------------------------------------------

/// Small deterministic PRNG (SplitMix64) with normal / gamma sampling, so the demo needs no extra crates.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Marsaglia-Tsang gamma(shape >= 1, scale)
    fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v * scale;
            }
        }
    }
}

/// Multi-day activity + light with a 24 h rhythm, 12 h / 8 h harmonics, a phase shift from `shift_day`, and noise.
pub fn synthetic(days: usize, epoch_minutes: f64, shift_day: Option<i64>, shift_hours: f64) -> RawDataset {
    let mut rng = Rng(42);
    let tau = 24.0;
    let acro = 14.5;
    let step_s = ((epoch_minutes * 60.0).round() as i64).max(1);
    let n = ((days as f64 * 24.0 * 60.0) / epoch_minutes) as usize;
    let start = naive_to_ms(NaiveDate::from_ymd_opt(2026, 11, 10).unwrap().and_hms_opt(0, 0, 0).unwrap());
    let w = 2.0 * std::f64::consts::PI / tau;
    let (phi1, phi2, phi3) = (-w * acro, -2.0 * w * (acro - 1.5), -3.0 * w * (acro + 0.5));
    let half = tau / 2.0;
    let sleep_c = (acro + half) % tau;
    let light_acro = (acro - 1.0).rem_euclid(tau);
    let (mut t_ms, mut act, mut lux) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let th = i as f64 * epoch_minutes / 60.0;
        let day = (th / 24.0).floor() as i64 + 1;
        let off = match shift_day {
            Some(sd) if shift_hours != 0.0 => shift_hours * (((day - sd + 1) as f64) / 2.0).clamp(0.0, 1.0),
            _ => 0.0,
        };
        let ph = th - off;
        let drive = 45.0 + 36.0 * (w * ph + phi1).cos() + 36.0 * 0.28 * (2.0 * w * ph + phi2).cos() + 36.0 * 0.12 * (3.0 * w * ph + phi3).cos();
        let tod = ph.rem_euclid(tau);
        let dist = (((tod - sleep_c + half).rem_euclid(tau)) - half).abs();
        let awake = 1.0 / (1.0 + (-(dist - 3.8) * 3.0).exp());
        let burst = if rng.uniform() < 0.25 { rng.gamma(2.0, 12.0) } else { 0.0 };
        let mut a = (drive * awake + burst * awake + rng.normal() * 36.0 * 0.18).clamp(0.5, 250.0);
        if awake < 0.25 {
            a = 0.2 + rng.uniform() * 3.8;
        }
        let ld = (((tod - light_acro + half).rem_euclid(tau)) - half).abs();
        let daylight = (-0.5 * (ld / 3.8).powi(2)).exp();
        let peak = if daylight > 0.5 && rng.uniform() < 0.12 { 800.0 + rng.uniform() * 3700.0 } else { 0.0 };
        let mut l = (2.0 + 380.0 * daylight * awake + peak * awake + rng.normal() * 15.0).clamp(0.1, 15000.0);
        if awake < 0.2 {
            l = rng.uniform() * 2.5;
        }
        t_ms.push(start + i as i64 * step_s * 1000);
        act.push((a * 1000.0).round() / 1000.0);
        lux.push((l * 100.0).round() / 100.0);
    }
    RawDataset {
        name: format!("Synthetic_{days}d_tau24.0_shift{shift_hours:+.1}h"),
        format: "Synthetic demo".into(),
        t_ms,
        channels: vec![Channel::new(Role::Activity, "Activity", "activity units", act), Channel::new(Role::Light, "Light", "lux", lux)],
        ..Default::default()
    }
}

// ------------------------------------------------------------------------------------------------
// Epoch binning
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivityMethod {
    Enmo,
    Mad,
    VectorMagnitude,
    Raw,
}

impl ActivityMethod {
    pub const ALL: [ActivityMethod; 4] = [ActivityMethod::Enmo, ActivityMethod::Mad, ActivityMethod::VectorMagnitude, ActivityMethod::Raw];
    pub fn label(self) -> &'static str {
        match self {
            ActivityMethod::Enmo => "ENMO (milli-g)",
            ActivityMethod::Mad => "MAD (milli-g)",
            ActivityMethod::VectorMagnitude => "Vector magnitude",
            ActivityMethod::Raw => "Raw activity column",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BinnedChannel {
    pub role: Role,
    pub name: String,
    pub unit: String,
    pub values: Vec<f64>,
}

/// Uniformly binned recording. Index i covers [start_ms + i·epoch, start_ms + (i+1)·epoch).
#[derive(Clone, Debug)]
pub struct Binned {
    pub epoch_sec: i64,
    pub start_ms: i64,
    pub midnight_ms: i64,
    /// Motion channel (activity counts, or ENMO/MAD/VM from the accelerometer)
    pub activity: Vec<f64>,
    pub activity_unit: String,
    pub lux: Vec<f64>,
    pub log_lux: Vec<f64>,
    /// Heart rate, temperature and other signals, each analysed separately
    pub extra: Vec<BinnedChannel>,
    /// Bin-centre hours since midnight of the first day
    pub elapsed_h: Vec<f64>,
    /// Bin-centre clock hour (may exceed 24 by half a bin)
    pub clock_h: Vec<f64>,
    pub day_index: Vec<i32>,
    pub has_activity: bool,
    pub has_light: bool,
    pub n_days: f64,
}

impl Binned {
    pub fn len(&self) -> usize {
        self.elapsed_h.len()
    }
    pub fn epoch_minutes(&self) -> f64 {
        self.epoch_sec as f64 / 60.0
    }
    pub fn bin_start_ms(&self, i: usize) -> i64 {
        self.start_ms + i as i64 * self.epoch_sec * 1000
    }
    pub fn date_of_day(&self, day: i32) -> String {
        ms_to_naive(self.midnight_ms + (day as i64 - 1) * MS_PER_DAY).format("%Y-%m-%d").to_string()
    }
    pub fn n_day_rows(&self) -> i32 {
        self.day_index.last().copied().unwrap_or(0)
    }
    /// Contiguous index range of each day: (day_index, start, end_exclusive)
    pub fn day_ranges(&self) -> Vec<(i32, usize, usize)> {
        let mut out = Vec::new();
        let mut s = 0;
        for i in 1..=self.len() {
            if i == self.len() || self.day_index[i] != self.day_index[s] {
                out.push((self.day_index[s], s, i));
                s = i;
            }
        }
        out
    }
}

pub fn epoch_seconds(epoch_minutes: f64) -> i64 {
    ((epoch_minutes * 60.0).round() as i64).clamp(MIN_EPOCH_SECONDS, MAX_EPOCH_SECONDS)
}

pub fn format_bin_size(sec: i64) -> String {
    if sec < 60 {
        format!("{sec} s")
    } else if sec < 3600 {
        let (m, s) = (sec / 60, sec % 60);
        if s == 0 {
            format!("{m} min")
        } else {
            format!("{m} min {s} s")
        }
    } else {
        let (h, m) = (sec / 3600, (sec % 3600) / 60);
        if m == 0 {
            format!("{h} h")
        } else {
            format!("{h} h {m} min")
        }
    }
}

/// Per-sample motion values and their unit.
fn sample_activity(d: &RawDataset, method: ActivityMethod) -> (Vec<f64>, String) {
    let n = d.len();
    if let Some(c) = d.channel(Role::Activity) {
        if method == ActivityMethod::Raw || !d.has_accel() {
            let unit = if c.unit.is_empty() { "activity units".to_string() } else { c.unit.clone() };
            return (c.values.iter().map(|&v| if v.is_finite() { v.max(0.0) } else { f64::NAN }).collect(), unit);
        }
    }
    if !d.has_accel() {
        return (vec![f64::NAN; n], String::new());
    }
    let (ax, ay, az) = (&d.channel(Role::AccelX).unwrap().values, &d.channel(Role::AccelY).unwrap().values, &d.channel(Role::AccelZ).unwrap().values);
    let vm: Vec<f64> = (0..n).map(|i| (ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i]).sqrt()).collect();
    let unit = d.channel(Role::AccelX).map(|c| c.unit.clone()).unwrap_or_default();
    if d.accel_linear || method == ActivityMethod::VectorMagnitude {
        return (vm, if unit.is_empty() { "vector magnitude".into() } else { unit });
    }
    let g = accel_g_units(crate::stats::median(&vm)).0;
    let out = match method {
        // NaN must stay missing (rows from another sensor in a merged recording, masked epochs):
        // f64::max(NaN, 0.0) is 0.0, which would count them as zero movement.
        ActivityMethod::Enmo => vm.iter().map(|&v| if v.is_finite() { (v - g).max(0.0) * (1000.0 / g) } else { f64::NAN }).collect(),
        _ => vm.iter().map(|&v| (v - g).abs() * (1000.0 / g)).collect(),
    };
    (out, "milli-g".into())
}

/// How many file units make 1 g, judged from the median vector magnitude of a 3-axis accelerometer
/// (≈ gravity, since a worn device is mostly still). The text names units that are neither g nor m/s².
pub fn accel_g_units(median_vm: f64) -> (f64, Option<&'static str>) {
    let m = median_vm;
    if !(m.is_finite() && m > 0.0) {
        return (9.80665, None);
    }
    if m > 0.5 && m < 2.0 {
        return (1.0, None);
    }
    if m > 5.0 && m < 20.0 {
        return (9.80665, None);
    }
    const KNOWN: [(f64, &str); 3] = [(64.0, "1/64 g (Empatica E4)"), (256.0, "1/256 g (Axivity raw)"), (1000.0, "milli-g")];
    match KNOWN.iter().find(|(s, _)| (m / s - 1.0).abs() < 0.3) {
        Some(&(s, d)) => (s, Some(d)),
        None => (m, Some("unrecognised units, scaled so the resting magnitude is 1 g")),
    }
}

/// Linear interpolation of up to `limit` consecutive NaNs after a valid value (pandas `interpolate(limit=)`).
fn interpolate_gaps(v: &mut [f64], limit: usize) {
    if limit == 0 {
        return;
    }
    let n = v.len();
    let mut i = 0;
    while i < n {
        if v[i].is_finite() {
            i += 1;
            continue;
        }
        let gap_start = i;
        while i < n && !v[i].is_finite() {
            i += 1;
        }
        if gap_start == 0 {
            continue; // leading NaNs stay NaN
        }
        let left = v[gap_start - 1];
        let right = if i < n { v[i] } else { left }; // trailing NaNs: hold the last value
        let span = (i - gap_start + 1) as f64;
        for (k, j) in (gap_start..i).enumerate().take(limit) {
            v[j] = left + (right - left) * (k + 1) as f64 / span;
        }
    }
}

/// Mean (or sum) of `values` per bin given precomputed bin indices.
fn aggregate(bins: &[u32], values: &[f64], nb: usize, sum: bool, clamp0: bool) -> Vec<f64> {
    let mut s = vec![0.0; nb];
    let mut c = vec![0u32; nb];
    for (&b, &v) in bins.iter().zip(values) {
        if v.is_finite() {
            s[b as usize] += if clamp0 { v.max(0.0) } else { v };
            c[b as usize] += 1;
        }
    }
    s.iter().zip(&c).map(|(&s, &c)| if sum { s } else if c > 0 { s / c as f64 } else { f64::NAN }).collect()
}

/// Resample raw samples into uniform epochs (bin edges anchored at midnight of the first day).
pub fn bin_epochs(d: &RawDataset, epoch_sec: i64, method: ActivityMethod, interp_gap_min: f64) -> Result<Binned, String> {
    if d.t_ms.is_empty() {
        return Err("Dataset has no timestamps.".into());
    }
    let epoch_ms = epoch_sec * 1000;
    let t0 = d.t_ms[0];
    let midnight = t0.div_euclid(MS_PER_DAY) * MS_PER_DAY;
    let first = (t0 - midnight).div_euclid(epoch_ms);
    let last = (d.t_ms[d.len() - 1] - midnight).div_euclid(epoch_ms);
    let nb = (last - first + 1) as usize;
    if nb > 40_000_000 {
        return Err(format!("{nb} bins is too many; choose a larger bin size."));
    }
    let bins: Vec<u32> = d.t_ms.iter().map(|&t| ((t - midnight).div_euclid(epoch_ms) - first) as u32).collect();
    let epoch_min = epoch_sec as f64 / 60.0;
    let max_gap = (interp_gap_min / epoch_min).floor().max(0.0) as usize;

    let has_act = d.has_motion();
    let (mut activity, activity_unit) = if has_act {
        let (v, unit) = sample_activity(d, method);
        let sum = d.channel(Role::Activity).is_some_and(|c| c.sum) && (method == ActivityMethod::Raw || !d.has_accel());
        (aggregate(&bins, &v, nb, sum, false), unit)
    } else {
        (vec![f64::NAN; nb], String::new())
    };
    let has_light = d.has_light();
    let mut lux = match d.channel(Role::Light) {
        Some(c) if has_light => aggregate(&bins, &c.values, nb, false, true),
        _ => vec![f64::NAN; nb],
    };
    if has_act {
        interpolate_gaps(&mut activity, max_gap);
    }
    if has_light {
        interpolate_gaps(&mut lux, max_gap);
    }
    let log_lux = lux.iter().map(|&l| if l.is_finite() { (l.max(0.0) + 1.0).log10() } else { f64::NAN }).collect();
    let extra = d
        .signals()
        .map(|c| {
            let mut v = aggregate(&bins, &c.values, nb, c.sum, false);
            if !c.sum {
                interpolate_gaps(&mut v, max_gap);
            }
            BinnedChannel { role: c.role, name: c.name.clone(), unit: c.unit.clone(), values: v }
        })
        .collect();

    let start_ms = midnight + first * epoch_ms;
    let half_h = epoch_min / 120.0;
    let mut elapsed_h = Vec::with_capacity(nb);
    let mut clock_h = Vec::with_capacity(nb);
    let mut day_index = Vec::with_capacity(nb);
    for i in 0..nb {
        let ts = start_ms + i as i64 * epoch_ms;
        let since = ts - midnight;
        elapsed_h.push(since as f64 / MS_PER_HOUR + half_h);
        day_index.push((since.div_euclid(MS_PER_DAY) + 1) as i32);
        let nt = ms_to_naive(ts);
        clock_h.push(nt.hour() as f64 + nt.minute() as f64 / 60.0 + nt.second() as f64 / 3600.0 + half_h);
    }
    Ok(Binned {
        epoch_sec,
        start_ms,
        midnight_ms: midnight,
        activity,
        activity_unit,
        lux,
        log_lux,
        extra,
        elapsed_h,
        clock_h,
        day_index,
        has_activity: has_act,
        has_light,
        n_days: nb as f64 * epoch_min / 60.0 / 24.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_and_channels() {
        let d = synthetic(3, 10.0, None, 0.0);
        let b = bin_epochs(&d, 3600, ActivityMethod::Enmo, 60.0).unwrap();
        assert_eq!(b.len(), 72);
        assert!((b.elapsed_h[0] - 0.5).abs() < 1e-12);
        assert_eq!(b.day_index[71], 3);
        assert!(b.has_activity && b.has_light && b.extra.is_empty());
    }

    #[test]
    fn enmo_ignores_other_sensors_rows() {
        // accelerometer (constant 1.05 g → ENMO 50 mg) merged with a light sensor on its own time stamps
        let acc = RawDataset {
            t_ms: (0..3600).map(|i| i * 1000).collect(),
            channels: vec![
                Channel::new(Role::AccelX, "x", "g", vec![0.0; 3600]),
                Channel::new(Role::AccelY, "y", "g", vec![0.0; 3600]),
                Channel::new(Role::AccelZ, "z", "g", vec![1.05; 3600]),
            ],
            ..Default::default()
        };
        let light = RawDataset { t_ms: (0..3600).map(|i| i * 1000 + 500).collect(), channels: vec![Channel::new(Role::Light, "Light", "lux", vec![100.0; 3600])], ..Default::default() };
        let m = RawDataset::merge(vec![acc, light]).unwrap();
        let b = bin_epochs(&m, 600, ActivityMethod::Enmo, 60.0).unwrap();
        assert!(b.activity.iter().all(|&a| (a - 50.0).abs() < 1e-9), "{:?}", &b.activity[..3]);
    }

    #[test]
    fn gap_interpolation_matches_pandas() {
        let mut v = vec![f64::NAN, 1.0, f64::NAN, f64::NAN, 4.0, f64::NAN];
        interpolate_gaps(&mut v, 1);
        assert_eq!(v[1..3], [1.0, 2.0]);
        assert!(v[3].is_nan());
        assert_eq!(v[5], 4.0);
        assert!(v[0].is_nan());
    }

    #[test]
    fn rr_conversion() {
        let mut rr = vec![800.0; 200];
        rr[50] = 400.0; // ectopic
        rr[120] = 9000.0; // artifact
        let d = rr_dataset("000.txt", &rr, None, Some(0));
        assert_eq!(d.rr_counts, Some((200, 2)));
        let hr = &d.channel(Role::HeartRate).unwrap().values;
        assert!(hr.iter().all(|&h| (h - 75.0).abs() < 1e-9));
        // rejected intervals still advance the clock
        assert_eq!(*d.t_ms.last().unwrap(), 198 * 800 + 400 + 9000);
    }

    #[test]
    fn kway_merge_matches_stable_sort() {
        // interleaved sensors with ties, plus one empty set and > 8 sets for the heap path
        let mk = |name: &str, role: Role, t: Vec<i64>| {
            let v = t.iter().map(|&x| x as f64 + 0.5).collect();
            RawDataset { name: name.into(), t_ms: t, channels: vec![Channel::new(role, name, "", v)], ..Default::default() }
        };
        let base = vec![
            mk("acc", Role::Activity, (0..200).map(|i| i * 31).collect()),
            mk("temp", Role::Temperature, (0..30).map(|i| i * 250).collect()),
            mk("hr", Role::HeartRate, vec![0, 0, 1000, 1000, 5000]),
            mk("none", Role::Light, vec![]),
        ];
        for sets in [base.clone(), (0..12).map(|k| mk(&format!("s{k}"), Role::Signal, (0..50).map(|i| i * (k + 3)).collect())).collect::<Vec<_>>()] {
            let m = RawDataset::merge(sets.clone()).unwrap();
            // reference: concatenate in set order, stable sort by time
            let mut rows: Vec<(i64, usize, f64)> = sets.iter().enumerate().flat_map(|(s, d)| d.t_ms.iter().zip(&d.channels[0].values).map(move |(&t, &v)| (t, s, v))).collect();
            rows.sort_by_key(|r| r.0);
            assert_eq!(m.t_ms, rows.iter().map(|r| r.0).collect::<Vec<_>>());
            for (k, c) in m.channels.iter().enumerate() {
                let src = sets.iter().position(|d| d.channels[0].name == c.name).unwrap();
                let want: Vec<f64> = rows.iter().map(|r| if r.1 == src { r.2 } else { f64::NAN }).collect();
                assert!(c.values.iter().zip(&want).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())), "channel {k}");
            }
        }
    }

    #[test]
    fn merge_by_channel() {
        let a = RawDataset { name: "a".into(), t_ms: vec![0, 2000], channels: vec![Channel::new(Role::Light, "Light", "lux", vec![1.0, 2.0])], ..Default::default() };
        let b = RawDataset { name: "b".into(), t_ms: vec![1000], channels: vec![Channel::new(Role::HeartRate, "Heart rate", "bpm", vec![60.0])], ..Default::default() };
        let m = RawDataset::merge(vec![a, b]).unwrap();
        assert_eq!(m.t_ms, vec![0, 1000, 2000]);
        assert_eq!(m.channels.len(), 2);
        assert!(m.channel(Role::Light).unwrap().values[1].is_nan());
        assert_eq!(m.channel(Role::HeartRate).unwrap().values[1], 60.0);
    }
}
