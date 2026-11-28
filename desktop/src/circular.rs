//! Circular statistics (mean direction, R̄, κ, Rayleigh, V-test), Watson-Williams two-sample test,
//! and acrophase shift / drift analysis.

use crate::stats::{f_sf, hhmm, linregress_slope, norm_sf, py_mod};
use std::f64::consts::PI;

pub fn hours_to_rad(h: f64, period: f64) -> f64 {
    2.0 * PI * py_mod(h, period) / period
}

pub fn rad_to_hours(r: f64, period: f64) -> f64 {
    py_mod(r, 2.0 * PI) / (2.0 * PI) * period
}

/// Wrap into (-period/2, +period/2]. Positive = delay, negative = advance.
pub fn wrap_shift(d: f64, period: f64) -> f64 {
    let half = period / 2.0;
    let w = py_mod(d + half, period) - half;
    if (w + half).abs() < 1e-9 {
        half
    } else {
        w
    }
}

pub fn von_mises_kappa(r_bar: f64, n: usize) -> f64 {
    let r = r_bar.clamp(0.0, 0.999999);
    let mut k = if r < 0.53 {
        2.0 * r + r.powi(3) + 5.0 * r.powi(5) / 6.0
    } else if r < 0.85 {
        -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    } else {
        1.0 / (r.powi(3) - 4.0 * r * r + 3.0 * r)
    };
    if n > 0 && n <= 15 {
        let nf = n as f64;
        if k < 2.0 {
            k = (k - 2.0 / (nf * k.max(1e-6))).max(0.0);
        } else {
            k = (nf - 1.0).powi(3) * k / (nf.powi(3) + nf);
        }
    }
    k.max(0.0)
}

#[derive(Clone, Debug)]
pub struct CircularSummary {
    pub label: String,
    pub n: usize,
    pub mean_rad: f64,
    pub mean_deg: f64,
    pub mean_hour: f64,
    pub r_bar: f64,
    pub circ_var: f64,
    pub circ_std_rad: f64,
    pub circ_std_hours: f64,
    pub kappa: f64,
    pub rayleigh_z: f64,
    pub rayleigh_p: f64,
    pub v_u: Option<f64>,
    pub v_p: Option<f64>,
}

impl CircularSummary {
    pub fn mean_hhmm(&self) -> String {
        hhmm(self.mean_hour)
    }
}

pub fn circular_summary(hours: &[f64], label: &str, weights: Option<&[f64]>, period: f64, expected: Option<f64>) -> Option<CircularSummary> {
    let mut th = Vec::new();
    let mut w = Vec::new();
    for (i, &h) in hours.iter().enumerate() {
        if !h.is_finite() {
            continue;
        }
        match weights {
            Some(ws) => {
                let wi = ws[i];
                if wi.is_finite() && wi > 0.0 {
                    th.push(hours_to_rad(h, period));
                    w.push(wi);
                }
            }
            None => {
                th.push(hours_to_rad(h, period));
                w.push(1.0);
            }
        }
    }
    let n = th.len();
    if n == 0 {
        return None;
    }
    let wm = w.iter().sum::<f64>() / n as f64;
    let (mut c, mut s, mut wsum) = (0.0, 0.0, 0.0);
    for (t, wi) in th.iter().zip(&w) {
        let wn = wi / wm;
        c += wn * t.cos();
        s += wn * t.sin();
        wsum += wn;
    }
    let r = c.hypot(s);
    let r_bar = if wsum > 0.0 { (r / wsum).clamp(0.0, 1.0) } else { 0.0 };
    let mean_rad = py_mod(s.atan2(c), 2.0 * PI);
    let mean_hour = rad_to_hours(mean_rad, period) % 24.0;
    let std_rad = if r_bar > 1e-12 { (-2.0 * r_bar.max(1e-12).ln()).max(0.0).sqrt() } else { PI };
    let nf = n as f64;
    let z = nf * r_bar * r_bar;
    let p_ray = if n >= 5 {
        let rn = nf * r_bar;
        ((1.0 + 4.0 * nf + 4.0 * (nf * nf - rn * rn)).sqrt() - (1.0 + 2.0 * nf)).exp()
    } else {
        (-z).exp()
    }
    .clamp(0.0, 1.0);
    let (v_u, v_p) = match expected.filter(|e| e.is_finite()) {
        Some(e) => {
            let v = r * (mean_rad - hours_to_rad(e, period)).cos();
            let u = (2.0 / nf).sqrt() * v;
            (Some(u), Some(norm_sf(u)))
        }
        None => (None, None),
    };
    Some(CircularSummary {
        label: label.to_string(),
        n,
        mean_rad,
        mean_deg: mean_rad.to_degrees(),
        mean_hour,
        r_bar,
        circ_var: 1.0 - r_bar,
        circ_std_rad: std_rad,
        circ_std_hours: std_rad / (2.0 * PI) * period,
        kappa: von_mises_kappa(r_bar, n),
        rayleigh_z: z,
        rayleigh_p: p_ray,
        v_u,
        v_p,
    })
}

#[derive(Clone, Debug)]
pub struct TwoSample {
    pub n1: usize,
    pub n2: usize,
    pub mean1: f64,
    pub mean2: f64,
    pub r1: f64,
    pub r2: f64,
    pub shift_h: f64,
    pub shift_deg: f64,
    pub direction: &'static str,
    pub f: f64,
    pub p: f64,
}

pub fn watson_williams(h1: &[f64], h2: &[f64], period: f64) -> Option<TwoSample> {
    let a: Vec<f64> = h1.iter().copied().filter(|x| x.is_finite()).map(|h| hours_to_rad(h, period)).collect();
    let b: Vec<f64> = h2.iter().copied().filter(|x| x.is_finite()).map(|h| hours_to_rad(h, period)).collect();
    let (n1, n2) = (a.len(), b.len());
    if n1 < 2 || n2 < 2 {
        return None;
    }
    let cs = |v: &[f64]| (v.iter().map(|t| t.cos()).sum::<f64>(), v.iter().map(|t| t.sin()).sum::<f64>());
    let ((c1, s1), (c2, s2)) = (cs(&a), cs(&b));
    let (r1, r2) = (c1.hypot(s1), c2.hypot(s2));
    let rt = (c1 + c2).hypot(s1 + s2);
    let n = (n1 + n2) as f64;
    let mean1 = rad_to_hours(s1.atan2(c1), period) % 24.0;
    let mean2 = rad_to_hours(s2.atan2(c2), period) % 24.0;
    let shift = wrap_shift(mean2 - mean1, period);
    let direction = if shift.abs() < 0.05 {
        "Aligned (No Shift)"
    } else if shift > 0.0 {
        "Phase Delay"
    } else {
        "Phase Advance"
    };
    let kappa = von_mises_kappa((r1 + r2) / n, n1 + n2);
    let corr = 1.0 + 3.0 / (8.0 * kappa.max(0.1));
    let denom = n - (r1 + r2);
    let (f, p) = if denom <= 1e-12 {
        if shift.abs() < 1e-6 {
            (0.0, 1.0)
        } else {
            (1e6, 0.0)
        }
    } else {
        let f = (corr * (n - 2.0) * (r1 + r2 - rt) / denom).max(0.0);
        (f, f_sf(f, 1.0, n - 2.0))
    };
    Some(TwoSample { n1, n2, mean1, mean2, r1: r1 / n1 as f64, r2: r2 / n2 as f64, shift_h: shift, shift_deg: shift / period * 360.0, direction, f, p })
}

#[derive(Clone, Debug)]
pub struct ShiftReport {
    pub split_day: i32,
    pub acrophase: Option<TwoSample>,
    pub onset: Option<TwoSample>,
    pub offset: Option<TwoSample>,
    pub mean_daily_shift: f64,
    pub cumulative_shift: f64,
    pub drift_slope: f64,
    pub implied_tau: f64,
    pub act_light_phase_angle: Option<f64>,
    /// Per input row: (day-to-day shift h, cumulative unwrapped shift h)
    pub per_day: Vec<(f64, f64)>,
}

/// `days`, `acro`, `onset`, `offset` are aligned per-day rows sorted by day.
pub fn acrophase_shift(days: &[i32], acro: &[f64], onset: &[f64], offset: &[f64], split_day: Option<i32>, period: f64, light_acro: Option<f64>, act_acro: Option<f64>) -> Option<ShiftReport> {
    let n = days.len();
    if n < 2 {
        return None;
    }
    let mut shifts = vec![f64::NAN; n];
    for i in 1..n {
        if acro[i].is_finite() && acro[i - 1].is_finite() {
            shifts[i] = wrap_shift(acro[i] - acro[i - 1], period);
        }
    }
    // np.unwrap over the valid phases, relative to the first valid day
    let valid: Vec<usize> = (0..n).filter(|&i| acro[i].is_finite()).collect();
    let mut cum = vec![f64::NAN; n];
    if !valid.is_empty() {
        let mut prev = hours_to_rad(acro[valid[0]], period);
        let mut unwrapped = vec![prev];
        for &i in &valid[1..] {
            let r = hours_to_rad(acro[i], period);
            let mut d = r - prev;
            d = py_mod(d + PI, 2.0 * PI) - PI;
            let next = unwrapped.last().unwrap() + d;
            unwrapped.push(next);
            prev = r;
        }
        let base = unwrapped[0] / (2.0 * PI) * period;
        for (k, &i) in valid.iter().enumerate() {
            cum[i] = unwrapped[k] / (2.0 * PI) * period - base;
        }
    }
    let (slope, cum_shift) = if valid.len() >= 2 {
        let x: Vec<f64> = valid.iter().map(|&i| days[i] as f64).collect();
        let y: Vec<f64> = valid.iter().map(|&i| cum[i]).collect();
        (linregress_slope(&x, &y), *y.last().unwrap())
    } else {
        (0.0, 0.0)
    };
    let vs: Vec<f64> = shifts.iter().copied().filter(|x| x.is_finite()).collect();
    let mean_shift = if vs.is_empty() { 0.0 } else { vs.iter().sum::<f64>() / vs.len() as f64 };
    let split = split_day.unwrap_or_else(|| {
        let mut d: Vec<f64> = days.iter().map(|&d| d as f64).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut s = crate::stats::median(&d) as i32;
        if s <= days[0] {
            s = days[0] + (n as i32 / 2).max(1);
        }
        s
    });
    let pre: Vec<usize> = (0..n).filter(|&i| days[i] < split).collect();
    let post: Vec<usize> = (0..n).filter(|&i| days[i] >= split).collect();
    let pick = |v: &[f64], idx: &[usize]| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>();
    let both = pre.len() >= 2 && post.len() >= 2;
    let test = |v: &[f64]| if both { watson_williams(&pick(v, &pre), &pick(v, &post), period) } else { None };
    let phase_angle = match (light_acro, act_acro) {
        (Some(l), Some(a)) if l.is_finite() && a.is_finite() => Some(wrap_shift(a - l, period)),
        _ => None,
    };
    Some(ShiftReport {
        split_day: split,
        acrophase: test(acro),
        onset: test(onset),
        offset: test(offset),
        mean_daily_shift: mean_shift,
        cumulative_shift: cum_shift,
        drift_slope: slope,
        implied_tau: 24.0 + slope,
        act_light_phase_angle: phase_angle,
        per_day: shifts.into_iter().zip(cum).collect(),
    })
}
