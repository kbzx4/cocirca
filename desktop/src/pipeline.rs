//! End-to-end analysis: binning → periodograms → rhythm bands / 12 h / bimodality → cosinor →
//! daily serial section, onsets, circular stats, shift analysis → non-parametric metrics.

use crate::circular::{acrophase_shift, circular_summary, wrap_shift, CircularSummary, ShiftReport};
use crate::cosinor::{fit_cosinor, CosinorFit};
use crate::data::{bin_epochs, ActivityMethod, Binned, RawDataset};
use crate::nonparam::{detect_onsets, non_parametric, NonParametric, OnsetMethod};
use crate::rhythm::{analyze_bimodality, analyze_periodicity, analyze_rhythm_bands, analyze_twelve_hour, Bimodality, Periodogram, RhythmBands, TwelveHour};
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeriodMode {
    Consensus,
    Fixed24,
    LombScargle,
    Chi2,
    Cosinor,
    Custom,
}

impl PeriodMode {
    pub const ALL: [PeriodMode; 6] = [PeriodMode::Consensus, PeriodMode::Fixed24, PeriodMode::LombScargle, PeriodMode::Chi2, PeriodMode::Cosinor, PeriodMode::Custom];
    pub fn label(self) -> &'static str {
        match self {
            PeriodMode::Consensus => "Auto (consensus τ)",
            PeriodMode::Fixed24 => "Fixed 24 h",
            PeriodMode::LombScargle => "Lomb-Scargle peak",
            PeriodMode::Chi2 => "χ² periodogram peak",
            PeriodMode::Cosinor => "Cosinor R² peak",
            PeriodMode::Custom => "Custom τ",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub epoch_sec: i64,
    pub activity_method: ActivityMethod,
    pub use_log_lux: bool,
    pub n_harmonics: usize,
    pub period_mode: PeriodMode,
    pub custom_tau: f64,
    pub period_min: f64,
    pub period_max: f64,
    pub circ_min: f64,
    pub circ_max: f64,
    pub alpha: f64,
    pub min_pct_variance: f64,
    pub bimodality_min_prominence: f64,
    pub onset_method: OnsetMethod,
    pub split_day: Option<i32>,
    pub interp_gap_min: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            epoch_sec: 600,
            activity_method: ActivityMethod::Enmo,
            use_log_lux: true,
            n_harmonics: 3,
            period_mode: PeriodMode::Consensus,
            custom_tau: 24.0,
            period_min: 9.0,
            period_max: 36.0,
            circ_min: 20.0,
            circ_max: 28.0,
            alpha: 0.05,
            min_pct_variance: 2.0,
            bimodality_min_prominence: 0.10,
            onset_method: OnsetMethod::Template,
            split_day: None,
            interp_gap_min: 60.0,
        }
    }
}

/// One row of the per-day table (NaN = not available).
#[derive(Clone, Debug)]
pub struct DailyRow {
    pub day: i32,
    pub date: String,
    pub n_valid: usize,
    pub mean_value: f64,
    pub max_value: f64,
    pub mesor: f64,
    pub amplitude: f64,
    pub acrophase_h: f64,
    pub acrophase_deg_neg: f64,
    pub orthophase_h: f64,
    pub r2: f64,
    pub p: f64,
    pub onset_h: f64,
    pub offset_h: f64,
    pub active_h: f64,
    pub rest_h: f64,
    pub midpoint_h: f64,
    pub act_mean: f64,
    pub act_max: f64,
    pub lux_mean: f64,
    pub lux_max: f64,
    pub bimodal: Option<bool>,
    pub bimodality_index: f64,
    pub peak1_h: f64,
    pub peak2_h: f64,
    pub light_acrophase_h: f64,
    pub phase_angle_h: f64,
    pub shift_h: f64,
    pub cumulative_shift_h: f64,
}

impl DailyRow {
    fn new(day: i32, date: String) -> Self {
        let n = f64::NAN;
        DailyRow {
            day,
            date,
            n_valid: 0,
            mean_value: n,
            max_value: n,
            mesor: n,
            amplitude: n,
            acrophase_h: n,
            acrophase_deg_neg: n,
            orthophase_h: n,
            r2: n,
            p: n,
            onset_h: n,
            offset_h: n,
            active_h: n,
            rest_h: n,
            midpoint_h: n,
            act_mean: n,
            act_max: n,
            lux_mean: n,
            lux_max: n,
            bimodal: None,
            bimodality_index: n,
            peak1_h: n,
            peak2_h: n,
            light_acrophase_h: n,
            phase_angle_h: n,
            shift_h: n,
            cumulative_shift_h: n,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChannelReport {
    pub name: String,
    pub unit: String,
    pub values: Vec<f64>,
    pub tau: f64,
    pub periodogram: Periodogram,
    pub single: CosinorFit,
    pub harmonics: CosinorFit,
    pub daily: Vec<DailyRow>,
    pub non_param: Option<NonParametric>,
    pub circ_acrophase: Option<CircularSummary>,
    pub circ_onset: Option<CircularSummary>,
    pub circ_offset: Option<CircularSummary>,
    pub shift: Option<ShiftReport>,
    pub bands: Option<RhythmBands>,
    pub bimodality: Option<Bimodality>,
    pub twelve: Option<TwelveHour>,
    pub pop_cosinor: Option<PopulationCosinor>,
}

#[derive(Clone, Debug)]
pub struct PopulationCosinor {
    pub n: usize,
    pub mesor: f64,
    pub mesor_se: f64,
    pub amplitude: f64,
    pub acrophase_h: f64,
    pub f: f64,
    pub p: f64,
}

pub struct Report {
    pub source: String,
    pub config: Config,
    pub binned: Binned,
    pub activity: Option<ChannelReport>,
    /// Heart rate, temperature and other signals
    pub extra: Vec<ChannelReport>,
    pub light: Option<ChannelReport>,
    pub warnings: Vec<String>,
    pub elapsed_ms: f64,
}

impl Report {
    pub fn primary(&self) -> &ChannelReport {
        self.activity.as_ref().or(self.extra.first()).or(self.light.as_ref()).expect("report has at least one channel")
    }
    /// Activity, then other signals, then light
    pub fn channels(&self) -> Vec<&ChannelReport> {
        self.activity.iter().chain(self.extra.iter()).chain(self.light.iter()).collect()
    }
}

/// Sub-minute bins are averaged up to 1 min for the period sweeps (no information is lost at ≥ 4 h periods).
fn periodogram_series(t: &[f64], y: &[f64], epoch_min: f64) -> (Vec<f64>, Vec<f64>, f64) {
    if epoch_min >= 1.0 {
        return (t.to_vec(), y.to_vec(), epoch_min);
    }
    let factor = (1.0 / epoch_min).round().max(1.0) as usize;
    let mut td = Vec::with_capacity(t.len() / factor + 1);
    let mut yd = Vec::with_capacity(t.len() / factor + 1);
    for (tc, yc) in t.chunks(factor).zip(y.chunks(factor)) {
        td.push(tc.iter().sum::<f64>() / tc.len() as f64);
        let (s, n) = yc.iter().filter(|v| v.is_finite()).fold((0.0, 0), |(s, n), v| (s + v, n + 1));
        yd.push(if n > 0 { s / n as f64 } else { f64::NAN });
    }
    (td, yd, epoch_min * factor as f64)
}

fn population_cosinor(rows: &[(f64, f64, f64)]) -> Option<PopulationCosinor> {
    // rows: (beta, gamma, mesor) per day
    let k = rows.len();
    if k < 3 {
        return None;
    }
    let kf = k as f64;
    let b = rows.iter().map(|r| r.0).sum::<f64>() / kf;
    let g = rows.iter().map(|r| r.1).sum::<f64>() / kf;
    let m = rows.iter().map(|r| r.2).sum::<f64>() / kf;
    let m_sd = (rows.iter().map(|r| (r.2 - m).powi(2)).sum::<f64>() / (kf - 1.0)).sqrt();
    let (sbb, sgg, sbg) = rows.iter().fold((0.0, 0.0, 0.0), |(a, c, d), r| (a + (r.0 - b).powi(2), c + (r.1 - g).powi(2), d + (r.0 - b) * (r.1 - g)));
    let (sbb, sgg, sbg) = (sbb / (kf - 1.0), sgg / (kf - 1.0), sbg / (kf - 1.0));
    let det = sbb * sgg - sbg * sbg;
    let (f, p) = if det.abs() > 1e-18 {
        let t2 = kf * (b * (sgg * b - sbg * g) + g * (-sbg * b + sbb * g)) / det;
        let f = (kf - 2.0) / (2.0 * (kf - 1.0)) * t2;
        (f, crate::stats::f_sf(f, 2.0, kf - 2.0))
    } else {
        (0.0, 1.0)
    };
    let phi = (-g).atan2(b);
    let w = 2.0 * std::f64::consts::PI / 24.0;
    Some(PopulationCosinor { n: k, mesor: m, mesor_se: m_sd / kf.sqrt(), amplitude: b.hypot(g), acrophase_h: crate::stats::py_mod(-phi / w, 24.0), f, p })
}

fn analyze_channel(b: &Binned, name: &str, values: Vec<f64>, unit: String, cfg: &Config, other_acro: Option<f64>) -> Result<ChannelReport, String> {
    let name = name.to_string();
    let t = &b.elapsed_h;
    // Sparse designs (e.g. 4-hourly sampling) cannot support many harmonics
    let n_valid = values.iter().filter(|v| v.is_finite()).count();
    let n_harm = cfg.n_harmonics.min((n_valid.saturating_sub(3) / 2).max(1));
    let epoch = b.epoch_minutes();
    let (pt, py, pe) = periodogram_series(t, &values, epoch);
    let pg = analyze_periodicity(&pt, &py, pe, cfg.period_min, cfg.period_max, 0.1, (cfg.circ_min, cfg.circ_max))?;
    let tau = match cfg.period_mode {
        PeriodMode::Fixed24 => 24.0,
        PeriodMode::Custom => cfg.custom_tau,
        PeriodMode::LombScargle => pg.best_tau_ls,
        PeriodMode::Chi2 => pg.best_tau_chi2,
        PeriodMode::Cosinor => pg.best_tau_cos,
        PeriodMode::Consensus => pg.consensus_tau,
    };
    let bimodal_period = if cfg.circ_min <= tau && tau <= cfg.circ_max { tau } else { 24.0 };
    let ranges = b.day_ranges();
    let day_slices: Vec<(i32, &[f64], &[f64])> = ranges.iter().map(|&(d, s, e)| (d, &b.clock_h[s..e], &values[s..e])).collect();

    // Independent analyses run concurrently
    let ((bands, twelve), ((bimodality, single), (harmonics, non_param))) = rayon::join(
        || {
            rayon::join(
                || analyze_rhythm_bands(&pt, &py, &pg.periods, &pg.ls_power, (cfg.circ_min, cfg.circ_max), cfg.alpha, cfg.min_pct_variance, 3).ok(),
                || analyze_twelve_hour(&pt, &py, bimodal_period, (10.0, 14.0), cfg.alpha),
            )
        },
        || {
            rayon::join(
                || rayon::join(|| analyze_bimodality(t, &values, bimodal_period, &day_slices, epoch, cfg.bimodality_min_prominence, cfg.alpha), || fit_cosinor(t, &values, tau, 1, 0.05)),
                || rayon::join(|| fit_cosinor(t, &values, tau, n_harm, 0.05), || non_parametric(b, &values)),
            )
        },
    );
    let single = single.ok_or("Not enough valid data for a cosinor fit.")?;
    let harmonics = harmonics.ok_or("Not enough valid data for a multi-harmonic cosinor fit.")?;

    // Daily serial-section cosinor (24 h clock projection)
    let expected = ((1440.0 / epoch.max(1.0)).round() as usize).max(1);
    let need = 8.max((expected as f64 * 0.5) as usize);
    let daily_fits: Vec<(i32, usize, f64, f64, CosinorFit)> = day_slices
        .par_iter()
        .filter_map(|&(d, ch, v)| {
            let n_valid = v.iter().filter(|x| x.is_finite()).count();
            if n_valid < need {
                return None;
            }
            let f = fit_cosinor(ch, v, 24.0, 1, 0.05)?;
            Some((d, n_valid, crate::stats::finite_mean(v), crate::stats::finite_max(v), f))
        })
        .collect();
    let onsets = detect_onsets(b, &values, cfg.onset_method, Some(&harmonics));

    let mut rows: BTreeMap<i32, DailyRow> = BTreeMap::new();
    for (d, nv, mean, max, f) in &daily_fits {
        let r = rows.entry(*d).or_insert_with(|| DailyRow::new(*d, b.date_of_day(*d)));
        r.n_valid = *nv;
        r.mean_value = *mean;
        r.max_value = *max;
        r.mesor = f.mesor;
        r.amplitude = f.amplitude();
        r.acrophase_h = f.acrophase_h();
        r.acrophase_deg_neg = f.deg_neg();
        r.orthophase_h = f.ortho_h;
        r.r2 = f.r2;
        r.p = f.p;
    }
    for o in &onsets {
        let r = rows.entry(o.day).or_insert_with(|| DailyRow::new(o.day, b.date_of_day(o.day)));
        r.onset_h = o.onset_h;
        r.offset_h = o.offset_h;
        r.active_h = o.active_h;
        r.rest_h = o.rest_h;
        r.midpoint_h = o.midpoint_h;
    }
    for &(d, s, e) in &ranges {
        if let Some(r) = rows.get_mut(&d) {
            if b.has_activity {
                r.act_mean = crate::stats::finite_mean(&b.activity[s..e]);
                r.act_max = crate::stats::finite_max(&b.activity[s..e]);
            }
            if b.has_light {
                r.lux_mean = crate::stats::finite_mean(&b.lux[s..e]);
                r.lux_max = crate::stats::finite_max(&b.lux[s..e]);
            }
        }
    }
    if let Some(bm) = &bimodality {
        for db in &bm.daily {
            if let Some(r) = rows.get_mut(&db.day) {
                r.bimodal = Some(db.bimodal);
                r.bimodality_index = db.index;
                r.peak1_h = db.peak1_h;
                r.peak2_h = db.peak2_h;
            }
        }
    }
    let mut daily: Vec<DailyRow> = rows.into_values().collect();
    let pop = population_cosinor(&daily_fits.iter().map(|x| (x.4.harmonics[0].beta, x.4.harmonics[0].gamma, x.4.mesor)).collect::<Vec<_>>());

    let col = |f: fn(&DailyRow) -> f64| daily.iter().map(f).collect::<Vec<f64>>();
    let (acro, amp, on, off) = (col(|r| r.acrophase_h), col(|r| r.amplitude), col(|r| r.onset_h), col(|r| r.offset_h));
    let circ_acrophase = circular_summary(&acro, &format!("{name} Acrophase"), Some(&amp), 24.0, None);
    let circ_onset = circular_summary(&on, &format!("{name} Onset"), None, 24.0, None);
    let circ_offset = circular_summary(&off, &format!("{name} Offset"), None, 24.0, None);
    let days: Vec<i32> = daily.iter().map(|r| r.day).collect();
    let (light_acro, act_acro) = if name != "Light" { (other_acro, Some(single.acrophase_h())) } else { (Some(single.acrophase_h()), other_acro) };
    let shift = acrophase_shift(&days, &acro, &on, &off, cfg.split_day, 24.0, light_acro, act_acro);
    if let Some(s) = &shift {
        for (r, &(sh, cum)) in daily.iter_mut().zip(&s.per_day) {
            r.shift_h = sh;
            r.cumulative_shift_h = cum;
        }
    }
    Ok(ChannelReport {
        name,
        unit,
        values,
        tau,
        periodogram: pg,
        single,
        harmonics,
        daily,
        non_param,
        circ_acrophase,
        circ_onset,
        circ_offset,
        shift,
        bands,
        bimodality,
        twelve,
        pop_cosinor: pop,
    })
}

pub fn run_analysis(d: &RawDataset, cfg: &Config) -> Result<Report, String> {
    let t0 = std::time::Instant::now();
    let b = bin_epochs(d, cfg.epoch_sec, cfg.activity_method, cfg.interp_gap_min)?;
    let mut warnings = d.notes.clone();
    if b.n_days < 2.0 {
        warnings.push(format!("Recording duration is {:.2} days; multi-day periodograms and circular stats are most reliable with ≥ 3 days.", b.n_days));
        if cfg.period_mode != PeriodMode::Fixed24 && cfg.period_mode != PeriodMode::Custom {
            warnings.push("With under 2 cycles the free-running period τ is not identifiable; set Period τ to \"Fixed 24 h\" for a standard single-day cosinor.".into());
        }
    }
    let count = |v: &[f64]| v.iter().filter(|x| x.is_finite()).count();
    const MIN_BINS: usize = 6;
    let light_ok = b.has_light && count(&b.lux) >= MIN_BINS;
    let act_ok = b.has_activity && count(&b.activity) >= MIN_BINS;
    let extras_ok: Vec<usize> = (0..b.extra.len()).filter(|&i| count(&b.extra[i].values) >= MIN_BINS).collect();
    if !light_ok && !act_ok && extras_ok.is_empty() {
        let mut found: Vec<String> = Vec::new();
        if b.has_activity {
            found.push(format!("{} activity", count(&b.activity)));
        }
        if b.has_light {
            found.push(format!("{} light", count(&b.lux)));
        }
        for x in &b.extra {
            found.push(format!("{} {}", count(&x.values), x.name.to_lowercase()));
        }
        return Err(format!(
            "Too few valid bins to analyse ({}; need ≥ {MIN_BINS}). Choose a smaller bin size or a longer recording.",
            if found.is_empty() { "no data channels".into() } else { found.join(", ") }
        ));
    }
    for x in b.extra.iter().filter(|x| count(&x.values) < MIN_BINS) {
        warnings.push(format!("{} has fewer than {MIN_BINS} valid bins and was not analysed.", x.name));
    }
    let (light_vals, light_unit) = if cfg.use_log_lux { (b.log_lux.clone(), "log10(lux + 1)") } else { (b.lux.clone(), "lux") };

    // Every channel is independent: analyse them all in parallel, then attach the light phase angle.
    enum Job {
        Activity,
        Extra(usize),
        Light,
    }
    let mut jobs = Vec::new();
    if act_ok {
        jobs.push(Job::Activity);
    }
    jobs.extend(extras_ok.iter().map(|&i| Job::Extra(i)));
    if light_ok {
        jobs.push(Job::Light);
    }
    let results: Vec<(usize, Result<ChannelReport, String>)> = jobs
        .par_iter()
        .map(|j| match j {
            Job::Activity => (0, analyze_channel(&b, "Activity", b.activity.clone(), b.activity_unit.clone(), cfg, None)),
            Job::Extra(i) => {
                let x = &b.extra[*i];
                (1, analyze_channel(&b, &x.name, x.values.clone(), x.unit.clone(), cfg, None))
            }
            Job::Light => (2, analyze_channel(&b, "Light", light_vals.clone(), light_unit.into(), cfg, None)),
        })
        .collect();
    let (mut activity, mut extra, mut light) = (None, Vec::new(), None);
    for (k, r) in results {
        match (k, r) {
            (0, Ok(c)) => activity = Some(c),
            (1, Ok(c)) => extra.push(c),
            (2, Ok(c)) => light = Some(c),
            (_, Err(e)) => warnings.push(e),
            _ => {}
        }
    }
    if activity.is_none() && extra.is_empty() && light.is_none() {
        return Err(warnings.pop().unwrap_or_else(|| "No channel could be analysed.".into()));
    }
    let primary_motion = activity.is_some();
    let target = if primary_motion { activity.as_mut() } else { extra.first_mut() };
    if let (Some(a), Some(l)) = (target, light.as_ref()) {
        let la = l.single.acrophase_h();
        if let Some(s) = a.shift.as_mut() {
            s.act_light_phase_angle = Some(wrap_shift(a.single.acrophase_h() - la, 24.0));
        }
        let lmap: BTreeMap<i32, f64> = l.daily.iter().map(|r| (r.day, r.acrophase_h)).collect();
        for r in a.daily.iter_mut() {
            if let Some(&lh) = lmap.get(&r.day) {
                r.light_acrophase_h = lh;
                if r.acrophase_h.is_finite() && lh.is_finite() {
                    r.phase_angle_h = wrap_shift(r.acrophase_h - lh, 24.0);
                }
            }
        }
    }
    Ok(Report {
        source: d.name.clone(),
        config: cfg.clone(),
        binned: b,
        activity,
        extra,
        light,
        warnings,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}
