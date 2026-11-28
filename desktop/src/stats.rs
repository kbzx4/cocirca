//! Small numeric helpers: distributions, robust summaries, and SciPy-compatible peak finding.

use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, StudentsT};

/// Survival function of the F distribution (P[F > f]).
pub fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    if !f.is_finite() || f <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return if f.is_infinite() && f > 0.0 { 0.0 } else { 1.0 };
    }
    FisherSnedecor::new(d1, d2).map(|d| d.sf(f).clamp(0.0, 1.0)).unwrap_or(1.0)
}

/// Quantile of Student's t distribution.
pub fn t_ppf(p: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    StudentsT::new(0.0, 1.0, df).map(|d| d.inverse_cdf(p)).unwrap_or(f64::NAN)
}

pub fn chi2_ppf(p: f64, df: f64) -> f64 {
    ChiSquared::new(df.max(1e-9)).map(|d| d.inverse_cdf(p)).unwrap_or(f64::NAN)
}

pub fn chi2_sf(x: f64, df: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 1.0;
    }
    ChiSquared::new(df.max(1e-9)).map(|d| d.sf(x).clamp(0.0, 1.0)).unwrap_or(1.0)
}

/// Standard normal survival function.
pub fn norm_sf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(z / std::f64::consts::SQRT_2)
}

/// Python-style modulo: result has the sign of `b`.
#[inline]
pub fn py_mod(a: f64, b: f64) -> f64 {
    a - b * (a / b).floor()
}

/// Hour of day (wrapped into [0, 24)) as HH:MM.
pub fn hhmm(hours: f64) -> String {
    if !hours.is_finite() {
        return "--:--".into();
    }
    let h = py_mod(hours, 24.0);
    let total = ((h * 60.0).round() as i64).rem_euclid(1440);
    format!("{:02}:{:02}", total / 60, total % 60)
}

pub fn finite_mean(v: &[f64]) -> f64 {
    let (s, n) = v.iter().filter(|x| x.is_finite()).fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 {
        f64::NAN
    } else {
        s / n as f64
    }
}

pub fn finite_max(v: &[f64]) -> f64 {
    v.iter().copied().filter(|x| x.is_finite()).fold(f64::NAN, |a, b| if a.is_nan() || b > a { b } else { a })
}

/// Population standard deviation of the finite values.
pub fn finite_std(v: &[f64]) -> f64 {
    let m = finite_mean(v);
    if !m.is_finite() {
        return f64::NAN;
    }
    let (s, n) = v.iter().filter(|x| x.is_finite()).fold((0.0, 0usize), |(s, n), x| (s + (x - m) * (x - m), n + 1));
    (s / n as f64).sqrt()
}

/// Median of the finite values (NaN if none).
pub fn median(v: &[f64]) -> f64 {
    let mut w: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if w.is_empty() {
        return f64::NAN;
    }
    let n = w.len();
    let mid = n / 2;
    let (_, m, _) = w.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
    let hi = *m;
    if n % 2 == 1 {
        hi
    } else {
        let lo = w[..mid].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

/// NumPy-style linear-interpolation percentile (q in 0..=100) of the finite values.
pub fn percentile(v: &[f64], q: f64) -> f64 {
    let mut w: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if w.is_empty() {
        return f64::NAN;
    }
    w.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = (q / 100.0).clamp(0.0, 1.0) * (w.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    w[lo] + (w[hi] - w[lo]) * (pos - lo as f64)
}

/// Strict local maxima (plateaus resolve to their middle), matching `scipy.signal.find_peaks` with no filters.
pub fn local_maxima(x: &[f64]) -> Vec<usize> {
    let n = x.len();
    let mut peaks = Vec::new();
    if n < 3 {
        return peaks;
    }
    let i_max = n - 1;
    let mut i = 1;
    while i < i_max {
        if x[i - 1] < x[i] {
            let mut ahead = i + 1;
            while ahead < i_max && x[ahead] == x[i] {
                ahead += 1;
            }
            if x[ahead] < x[i] {
                peaks.push((i + ahead - 1) / 2);
                i = ahead;
            }
        }
        i += 1;
    }
    peaks
}

/// Topographic prominence of each peak (SciPy `peak_prominences`, unlimited window).
pub fn prominences(x: &[f64], peaks: &[usize]) -> Vec<f64> {
    peaks
        .iter()
        .map(|&p| {
            let v = x[p];
            let mut left_min = v;
            let mut i = p as isize;
            while i >= 0 && x[i as usize] <= v {
                left_min = left_min.min(x[i as usize]);
                i -= 1;
            }
            let mut right_min = v;
            let mut j = p;
            while j < x.len() && x[j] <= v {
                right_min = right_min.min(x[j]);
                j += 1;
            }
            v - left_min.max(right_min)
        })
        .collect()
}

/// SciPy-compatible `find_peaks(x, distance=?, prominence=?)`. Returns (peaks, prominences).
pub fn find_peaks(x: &[f64], distance: Option<usize>, min_prominence: Option<f64>) -> (Vec<usize>, Vec<f64>) {
    let mut peaks = local_maxima(x);
    if let Some(d) = distance.filter(|d| *d > 1) {
        let mut keep = vec![true; peaks.len()];
        let mut order: Vec<usize> = (0..peaks.len()).collect();
        order.sort_by(|&a, &b| x[peaks[a]].partial_cmp(&x[peaks[b]]).unwrap_or(std::cmp::Ordering::Equal));
        for &j in order.iter().rev() {
            if !keep[j] {
                continue;
            }
            let mut k = j as isize - 1;
            while k >= 0 && peaks[j] - peaks[k as usize] < d {
                keep[k as usize] = false;
                k -= 1;
            }
            let mut k = j + 1;
            while k < peaks.len() && peaks[k] - peaks[j] < d {
                keep[k] = false;
                k += 1;
            }
        }
        peaks = peaks.into_iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| p).collect();
    }
    let proms = prominences(x, &peaks);
    match min_prominence {
        Some(mp) => {
            let (p, pr): (Vec<usize>, Vec<f64>) = peaks.into_iter().zip(proms).filter(|(_, pr)| *pr >= mp).unzip();
            (p, pr)
        }
        None => (peaks, proms),
    }
}

/// Least-squares slope of y on x.
pub fn linregress_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if x.len() < 2 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
    }
    if sxx <= 0.0 {
        0.0
    } else {
        sxy / sxx
    }
}

/// Invert a small dense symmetric matrix (row-major, dim n) by Gauss-Jordan with partial pivoting.
pub fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max).max(1e-300);
    for col in 0..n {
        let piv = (col..n).max_by(|&r1, &r2| m[r1 * n + col].abs().partial_cmp(&m[r2 * n + col].abs()).unwrap())?;
        if m[piv * n + col].abs() < 1e-13 * scale {
            return None;
        }
        if piv != col {
            for k in 0..n {
                m.swap(piv * n + k, col * n + k);
                inv.swap(piv * n + k, col * n + k);
            }
        }
        let d = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * n + col];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[col * n + k];
                        inv[r * n + k] -= f * inv[col * n + k];
                    }
                }
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peaks_and_prominence() {
        let x = [0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.5, 0.0];
        assert_eq!(local_maxima(&x), vec![1, 3, 6]);
        let pr = prominences(&x, &[1, 3, 6]);
        assert_eq!(pr, vec![1.0, 2.0, 0.5]);
        let (p, _) = find_peaks(&x, Some(3), None);
        assert_eq!(p, vec![3, 6]);
    }

    #[test]
    fn distributions() {
        assert!((f_sf(3.0, 2.0, 100.0) - 0.0542).abs() < 1e-3);
        assert!((t_ppf(0.975, 10.0) - 2.228).abs() < 1e-3);
        assert!((chi2_ppf(0.95, 143.0) - 171.907).abs() < 0.05);
        assert_eq!(hhmm(25.5), "01:30");
        assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), 2.5);
    }
}
