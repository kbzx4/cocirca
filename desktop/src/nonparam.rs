//! Non-parametric rhythm metrics (IS, IV, M10, L5, RA) and daily activity onset / offset detection.

use crate::cosinor::CosinorFit;
use crate::data::{ms_to_naive, Binned};
use crate::stats::{find_peaks, finite_std, percentile};
use chrono::Timelike;

#[derive(Clone, Debug)]
pub struct NonParametric {
    pub is: f64,
    pub iv: f64,
    pub m10: f64,
    pub m10_onset: f64,
    pub m10_mid: f64,
    pub l5: f64,
    pub l5_onset: f64,
    pub l5_mid: f64,
    pub ra: f64,
    pub profile: [f64; 24],
}

const MS_PER_HOUR: i64 = 3_600_000;

/// IS / IV / M10 / L5 / RA from hourly means. Bins coarser than 1 h are held across every hour they span.
pub fn non_parametric(b: &Binned, values: &[f64]) -> Option<NonParametric> {
    let epoch_min = b.epoch_minutes();
    let n_valid = values.iter().filter(|v| v.is_finite()).count();
    let min_valid = if epoch_min <= 60.0 { 24 } else { 4.max((24.0 * 60.0 / epoch_min).ceil() as usize) };
    if n_valid < min_valid {
        return None;
    }
    let h0 = b.start_ms.div_euclid(MS_PER_HOUR);
    let h_last = b.bin_start_ms(b.len() - 1).div_euclid(MS_PER_HOUR);
    let nh = (h_last - h0 + 1) as usize;
    let mut sum = vec![0.0; nh];
    let mut cnt = vec![0u32; nh];
    for (i, &v) in values.iter().enumerate() {
        if v.is_finite() {
            let h = (b.bin_start_ms(i).div_euclid(MS_PER_HOUR) - h0) as usize;
            sum[h] += v;
            cnt[h] += 1;
        }
    }
    let mut hourly: Vec<f64> = sum.iter().zip(&cnt).map(|(&s, &c)| if c > 0 { s / c as f64 } else { f64::NAN }).collect();
    if epoch_min > 60.0 {
        let limit = (epoch_min / 60.0).ceil() as usize - 1;
        let mut last = f64::NAN;
        let mut run = 0;
        for v in hourly.iter_mut() {
            if v.is_finite() {
                last = *v;
                run = 0;
            } else if last.is_finite() && run < limit {
                *v = last;
                run += 1;
            }
        }
    }
    let pairs: Vec<(usize, f64)> = hourly
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(k, &v)| (ms_to_naive((h0 + k as i64) * MS_PER_HOUR).hour() as usize, v))
        .collect();
    let n = pairs.len();
    if n < 24 {
        return None;
    }
    let x: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let xbar = x.iter().sum::<f64>() / n as f64;
    let denom: f64 = x.iter().map(|v| (v - xbar).powi(2)).sum();
    if denom <= 1e-15 {
        return None;
    }
    let mut profile = [0.0; 24];
    let mut counts = [0.0; 24];
    for &(h, v) in &pairs {
        profile[h] += v;
        counts[h] += 1.0;
    }
    for h in 0..24 {
        profile[h] = if counts[h] > 0.0 { profile[h] / counts[h] } else { xbar };
    }
    let nf = n as f64;
    let is = (nf * profile.iter().map(|p| (p - xbar).powi(2)).sum::<f64>() / (24.0 * denom)).clamp(0.0, 1.0);
    let iv = nf * x.windows(2).map(|w| (w[1] - w[0]).powi(2)).sum::<f64>() / ((nf - 1.0) * denom);
    let window_mean = |h: usize, len: usize| (0..len).map(|k| profile[(h + k) % 24]).sum::<f64>() / len as f64;
    let m10s: Vec<f64> = (0..24).map(|h| window_mean(h, 10)).collect();
    let l5s: Vec<f64> = (0..24).map(|h| window_mean(h, 5)).collect();
    let m10_start = (0..24).max_by(|&a, &b| m10s[a].partial_cmp(&m10s[b]).unwrap().then(b.cmp(&a))).unwrap();
    let l5_start = (0..24).min_by(|&a, &b| l5s[a].partial_cmp(&l5s[b]).unwrap().then(a.cmp(&b))).unwrap();
    let (m10, l5) = (m10s[m10_start], l5s[l5_start]);
    Some(NonParametric {
        is,
        iv,
        m10,
        m10_onset: m10_start as f64,
        m10_mid: (m10_start as f64 + 5.0) % 24.0,
        l5,
        l5_onset: l5_start as f64,
        l5_mid: (l5_start as f64 + 2.5) % 24.0,
        ra: if m10 + l5 > 1e-15 { (m10 - l5) / (m10 + l5) } else { 0.0 },
        profile,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnsetMethod {
    Template,
    Threshold,
    Cosinor,
}

impl OnsetMethod {
    pub const ALL: [OnsetMethod; 3] = [OnsetMethod::Template, OnsetMethod::Threshold, OnsetMethod::Cosinor];
    pub fn label(self) -> &'static str {
        match self {
            OnsetMethod::Template => "Template edge detection",
            OnsetMethod::Threshold => "Threshold crossing",
            OnsetMethod::Cosinor => "Cosinor MESOR crossing",
        }
    }
}

#[derive(Clone, Debug)]
pub struct OnsetRow {
    pub day: i32,
    pub onset_h: f64,
    pub offset_h: f64,
    pub active_h: f64,
    pub rest_h: f64,
    pub midpoint_h: f64,
}

/// Linear interpolation across all NaNs, holding the edge values (pandas `limit_direction="both"`).
fn fill_all(v: &[f64]) -> Vec<f64> {
    let idx: Vec<usize> = (0..v.len()).filter(|&i| v[i].is_finite()).collect();
    if idx.is_empty() {
        return v.to_vec();
    }
    let mut out = v.to_vec();
    for i in 0..idx[0] {
        out[i] = v[idx[0]];
    }
    for w in idx.windows(2) {
        let (a, b) = (w[0], w[1]);
        for i in a + 1..b {
            out[i] = v[a] + (v[b] - v[a]) * (i - a) as f64 / (b - a) as f64;
        }
    }
    for i in idx[idx.len() - 1] + 1..v.len() {
        out[i] = v[idx[idx.len() - 1]];
    }
    out
}

/// Centred rolling mean with min_periods = 1 (pandas `rolling(w, center=True)`), O(n) via prefix sums.
fn rolling_center(v: &[f64], w: usize) -> Vec<f64> {
    let n = v.len();
    let mut cs = vec![0.0; n + 1];
    let mut cc = vec![0usize; n + 1];
    for i in 0..n {
        let f = v[i].is_finite();
        cs[i + 1] = cs[i] + if f { v[i] } else { 0.0 };
        cc[i + 1] = cc[i] + f as usize;
    }
    let off = (w - 1) / 2;
    (0..n)
        .map(|i| {
            let hi = (i + off + 1).min(n);
            let lo = (i + off + 1).saturating_sub(w);
            let c = cc[hi] - cc[lo];
            if c > 0 {
                (cs[hi] - cs[lo]) / c as f64
            } else {
                f64::NAN
            }
        })
        .collect()
}

pub fn detect_onsets(b: &Binned, values: &[f64], method: OnsetMethod, fit: Option<&CosinorFit>) -> Vec<OnsetRow> {
    if values.iter().filter(|v| v.is_finite()).count() < 12 {
        return Vec::new();
    }
    let epoch = b.epoch_minutes();
    let n = values.len();
    let y = fill_all(values);
    let ys = rolling_center(&y, ((60.0 / epoch).round() as usize).max(1));
    let min_dist = 4.max(((14.0 * 60.0) / epoch).round() as usize);
    let (mut ons, mut offs): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    match (method, fit) {
        (OnsetMethod::Cosinor, Some(f)) => {
            let above: Vec<bool> = b.elapsed_h.iter().map(|&t| f.eval(t) >= f.mesor).collect();
            for i in 1..n {
                if !above[i - 1] && above[i] {
                    ons.push(i);
                }
                if above[i - 1] && !above[i] {
                    offs.push(i);
                }
            }
        }
        (OnsetMethod::Threshold, _) => {
            let (ql, qh) = (percentile(&ys, 15.0), percentile(&ys, 85.0));
            let thresh = ql + 0.35 * (qh - ql);
            let sustained = ((60.0 / epoch).round() as usize).max(1);
            let state: Vec<bool> = ys.iter().map(|&v| v >= thresh).collect();
            let mut cs = vec![0usize; n + 1];
            for i in 0..n {
                cs[i + 1] = cs[i] + state[i] as usize;
            }
            let win = |i: usize| cs[(i + sustained).min(n)] - cs[i];
            let mut active = state[0];
            let mut i = 0;
            while i + sustained < n {
                if !active && win(i) == sustained {
                    if ons.last().is_none_or(|&l| i - l >= min_dist) {
                        ons.push(i);
                    }
                    active = true;
                    i += sustained;
                    continue;
                } else if active && win(i) == 0 {
                    if offs.last().is_none_or(|&l| i - l >= min_dist) {
                        offs.push(i);
                    }
                    active = false;
                    i += sustained;
                    continue;
                }
                i += 1;
            }
        }
        _ => {
            let w = 2.max(((3.0 * 60.0) / epoch).round() as usize);
            let mut score = vec![0.0; n];
            if n > 2 * w {
                let mut cs = vec![0.0; n + 1];
                for i in 0..n {
                    cs[i + 1] = cs[i] + ys[i];
                }
                for i in w..n - w {
                    score[i] = (cs[i + w] - cs[i]) / w as f64 - (cs[i] - cs[i - w]) / w as f64;
                }
            }
            let prom = (finite_std(&score) * 0.35).max(1e-9);
            ons = find_peaks(&score, Some(min_dist), Some(prom)).0;
            let neg: Vec<f64> = score.iter().map(|v| -v).collect();
            offs = find_peaks(&neg, Some(min_dist), Some(prom)).0;
        }
    }
    let mut rows = Vec::new();
    let mut used = std::collections::HashSet::new();
    for &on in &ons {
        let day = b.day_index[on];
        if !used.insert(day) {
            continue;
        }
        let t_on = b.elapsed_h[on];
        let c_on = b.clock_h[on];
        let off = offs.iter().copied().filter(|&o| o > on).find(|&o| {
            let dt = b.elapsed_h[o] - t_on;
            (3.0..=21.5).contains(&dt)
        });
        let (offset_h, active, rest, mid) = match off {
            Some(o) => {
                let a = b.elapsed_h[o] - t_on;
                (b.clock_h[o], a, (24.0 - a).max(0.0), (c_on + 0.5 * a) % 24.0)
            }
            None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        };
        rows.push(OnsetRow { day, onset_h: c_on, offset_h, active_h: active, rest_h: rest, midpoint_h: mid });
    }
    rows
}
