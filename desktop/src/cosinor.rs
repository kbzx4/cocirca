//! Single / multi-harmonic cosinor regression and the vectorised single-cosinor R² period sweep.
//!
//! The fit accumulates X'X and X'y in one streaming pass (harmonics by angle-addition recurrence),
//! so a 1.2 M-bin recording never materialises the n × p design matrix.

use crate::stats::{f_sf, hhmm, invert, py_mod, t_ppf};
use rayon::prelude::*;
use std::f64::consts::PI;

pub const MAX_HARMONICS: usize = 8;
const MAXP: usize = 1 + 2 * MAX_HARMONICS;

#[derive(Clone, Debug)]
pub struct Harmonic {
    pub k: usize,
    pub period: f64,
    pub beta: f64,
    pub gamma: f64,
    pub amp: f64,
    pub amp_se: f64,
    pub amp_ci: (f64, f64),
    pub phi_rad: f64,
    pub deg_neg: f64,
    /// First peak after t = 0, in [0, period / k)
    pub peak_h: f64,
    /// Peak as clock hour [0, 24)
    pub peak_clock: f64,
    pub phi_se_h: f64,
    pub f: f64,
    pub p: f64,
    pub partial_r2: f64,
    pub ratio_h1: f64,
}

#[derive(Clone, Debug)]
pub struct CosinorFit {
    pub period: f64,
    pub nh: usize,
    pub n: usize,
    pub mesor: f64,
    pub mesor_se: f64,
    pub mesor_ci: (f64, f64),
    pub harmonics: Vec<Harmonic>,
    pub r2: f64,
    pub adj_r2: f64,
    pub rmse: f64,
    pub f: f64,
    pub p: f64,
    pub bathy_h: f64,
    pub ortho_h: f64,
    pub ortho_val: f64,
    pub comp_bathy_h: f64,
    pub comp_bathy_val: f64,
    pub comp_amp: f64,
}

impl CosinorFit {
    pub fn amplitude(&self) -> f64 {
        self.harmonics[0].amp
    }
    pub fn acrophase_h(&self) -> f64 {
        self.harmonics[0].peak_clock
    }
    pub fn acrophase_hhmm(&self) -> String {
        hhmm(self.acrophase_h())
    }
    pub fn deg_neg(&self) -> f64 {
        self.harmonics[0].deg_neg
    }
    pub fn eval(&self, t: f64) -> f64 {
        self.eval_upto(t, self.nh)
    }
    pub fn eval_upto(&self, t: f64, upto: usize) -> f64 {
        let w = 2.0 * PI / self.period;
        let mut y = self.mesor;
        for h in self.harmonics.iter().take(upto) {
            let (s, c) = (h.k as f64 * w * t).sin_cos();
            y += h.beta * c + h.gamma * s;
        }
        y
    }
}

/// Fill `x[1..]` with cos/sin of k·ωt for k = 1..=nh via angle-addition recurrence.
#[inline(always)]
fn regressors(x: &mut [f64; MAXP], wt: f64, nh: usize) {
    let (s1, c1) = wt.sin_cos();
    let (mut c, mut s) = (c1, s1);
    x[0] = 1.0;
    for k in 0..nh {
        x[1 + 2 * k] = c;
        x[2 + 2 * k] = s;
        let cn = c * c1 - s * s1;
        s = s * c1 + c * s1;
        c = cn;
    }
}

/// Fit y(t) = M + Σ_k [β_k cos(kωt) + γ_k sin(kωt)], ω = 2π/period. NaN pairs are skipped.
pub fn fit_cosinor(t: &[f64], y: &[f64], period: f64, nh: usize, alpha: f64) -> Option<CosinorFit> {
    let nh = nh.clamp(1, MAX_HARMONICS);
    let p = 1 + 2 * nh;
    let w = 2.0 * PI / period;

    let mut xtx = [0.0f64; MAXP * MAXP];
    let mut xty = [0.0f64; MAXP];
    let mut x = [0.0f64; MAXP];
    let (mut n, mut sy, mut syy) = (0usize, 0.0f64, 0.0f64);
    for (&ti, &yi) in t.iter().zip(y) {
        if !(ti.is_finite() && yi.is_finite()) {
            continue;
        }
        regressors(&mut x, w * ti, nh);
        for a in 0..p {
            let xa = x[a];
            xty[a] += xa * yi;
            for b in a..p {
                xtx[a * MAXP + b] += xa * x[b];
            }
        }
        n += 1;
        sy += yi;
        syy += yi * yi;
    }
    if n <= p + 1 {
        return None;
    }
    let mut m = vec![0.0; p * p];
    for a in 0..p {
        for b in a..p {
            m[a * p + b] = xtx[a * MAXP + b];
            m[b * p + a] = xtx[a * MAXP + b];
        }
    }
    let inv = invert(&m, p).or_else(|| {
        // Near-singular (e.g. period at an alias of the sampling): tiny ridge keeps the fit defined
        let tr = (0..p).map(|i| m[i * p + i]).sum::<f64>() / p as f64;
        let mut r = m.clone();
        for i in 0..p {
            r[i * p + i] += 1e-9 * tr.max(1e-12);
        }
        invert(&r, p)
    })?;
    let coef: Vec<f64> = (0..p).map(|a| (0..p).map(|b| inv[a * p + b] * xty[b]).sum()).collect();

    // Residual sum of squares by a second streaming pass (stable even with large offsets)
    let mut ssr = 0.0;
    for (&ti, &yi) in t.iter().zip(y) {
        if !(ti.is_finite() && yi.is_finite()) {
            continue;
        }
        regressors(&mut x, w * ti, nh);
        let yhat: f64 = (0..p).map(|a| coef[a] * x[a]).sum();
        ssr += (yi - yhat) * (yi - yhat);
    }
    let nf = n as f64;
    let sst = (syy - sy * sy / nf).max(0.0);
    let df_model = (2 * nh) as f64;
    let df_resid = (n - p) as f64;
    let ssm = (sst - ssr).max(0.0);
    let r2 = if sst > 1e-15 { ssm / sst } else { 0.0 };
    let adj_r2 = if df_resid > 0.0 { 1.0 - (1.0 - r2) * (nf - 1.0) / df_resid } else { r2 };
    let sigma2 = if df_resid > 0.0 { ssr / df_resid } else { 0.0 };
    let rmse = (ssr / nf).sqrt();
    let (f_stat, p_val) = if sigma2 > 1e-15 {
        let f = (ssm / df_model) / sigma2;
        (f, f_sf(f, df_model, df_resid))
    } else {
        (0.0, 1.0)
    };
    let tcrit = t_ppf(1.0 - alpha / 2.0, df_resid);
    let cov = |a: usize, b: usize| sigma2 * inv[a * p + b];

    let mesor = coef[0];
    let mesor_se = cov(0, 0).max(0.0).sqrt();

    let mut harmonics = Vec::with_capacity(nh);
    let mut amp1 = 0.0;
    for k in 1..=nh {
        let (ib, ig) = (1 + 2 * (k - 1), 2 + 2 * (k - 1));
        let (bk, gk) = (coef[ib], coef[ig]);
        let amp = bk.hypot(gk);
        if k == 1 {
            amp1 = amp;
        }
        let (vb, vg, cbg) = (cov(ib, ib).max(0.0), cov(ig, ig).max(0.0), cov(ib, ig));
        let (amp_se, phi_se) = if amp > 1e-12 {
            let av = (bk * bk * vb + gk * gk * vg + 2.0 * bk * gk * cbg) / (amp * amp);
            let pv = (gk * gk * vb + bk * bk * vg - 2.0 * bk * gk * cbg) / amp.powi(4);
            (av.max(0.0).sqrt(), pv.max(0.0).sqrt())
        } else {
            ((0.5 * (vb + vg)).sqrt(), PI)
        };
        let phi = (-gk).atan2(bk);
        let k_period = period / k as f64;
        let kw = k as f64 * w;
        let peak_h = py_mod(-phi / kw, k_period);
        let mut deg_neg = py_mod(-(peak_h / k_period) * 360.0, -360.0);
        if deg_neg > 0.0 {
            deg_neg -= 360.0;
        }
        // Partial F-test for harmonic k from the 2×2 block of (X'X)^-1
        let (s00, s01, s10, s11) = (inv[ib * p + ib], inv[ib * p + ig], inv[ig * p + ib], inv[ig * p + ig]);
        let det = s00 * s11 - s01 * s10;
        let ssr_drop = if det.abs() > 1e-18 {
            let (i00, i01, i10, i11) = (s11 / det, -s01 / det, -s10 / det, s00 / det);
            (bk * (i00 * bk + i01 * gk) + gk * (i10 * bk + i11 * gk)).max(0.0)
        } else {
            0.0
        };
        let (fk, pk) = if sigma2 > 1e-15 {
            let f = 0.5 * ssr_drop / sigma2;
            (f, f_sf(f, 2.0, df_resid))
        } else {
            (0.0, 1.0)
        };
        let partial_r2 = if sst > 1e-15 { (ssr_drop / sst).max(0.0) } else { 0.0 };
        harmonics.push(Harmonic {
            k,
            period: k_period,
            beta: bk,
            gamma: gk,
            amp,
            amp_se,
            amp_ci: ((amp - tcrit * amp_se).max(0.0), amp + tcrit * amp_se),
            phi_rad: phi,
            deg_neg,
            peak_h,
            peak_clock: py_mod(peak_h, 24.0),
            phi_se_h: phi_se / kw,
            f: fk,
            p: pk,
            partial_r2,
            ratio_h1: if amp1 > 1e-12 { amp / amp1 } else if k == 1 { 1.0 } else { 0.0 },
        });
    }

    let mut fit = CosinorFit {
        period,
        nh,
        n,
        mesor,
        mesor_se,
        mesor_ci: (mesor - tcrit * mesor_se, mesor + tcrit * mesor_se),
        bathy_h: py_mod(harmonics[0].peak_h + period / 2.0, period) % 24.0,
        harmonics,
        r2,
        adj_r2,
        rmse,
        f: f_stat,
        p: p_val,
        ortho_h: 0.0,
        ortho_val: 0.0,
        comp_bathy_h: 0.0,
        comp_bathy_val: 0.0,
        comp_amp: 0.0,
    };
    // Composite multi-harmonic waveform over one period
    let (mut imax, mut imin, mut vmax, mut vmin) = (0usize, 0usize, f64::NEG_INFINITY, f64::INFINITY);
    for i in 0..1440 {
        let v = fit.eval(period * i as f64 / 1440.0);
        if v > vmax {
            vmax = v;
            imax = i;
        }
        if v < vmin {
            vmin = v;
            imin = i;
        }
    }
    fit.ortho_h = py_mod(period * imax as f64 / 1440.0, 24.0);
    fit.ortho_val = vmax;
    fit.comp_bathy_h = py_mod(period * imin as f64 / 1440.0, 24.0);
    fit.comp_bathy_val = vmin;
    fit.comp_amp = (vmax - vmin) / 2.0;
    Some(fit)
}

/// `y - (fit(t) - mesor)`: remove a fitted rhythm while keeping the series level (NaNs preserved).
pub fn subtract_rhythm(t: &[f64], y: &[f64], fit: &CosinorFit) -> Vec<f64> {
    t.iter().zip(y).map(|(&ti, &yi)| if yi.is_finite() { yi - (fit.eval(ti) - fit.mesor) } else { f64::NAN }).collect()
}

/// Single-cosinor R² at each trial period (equals the floating-mean, normalised Lomb-Scargle power).
/// Parallel over periods; uniformly spaced samples use a rotation recurrence instead of sin/cos per sample.
pub fn r2_sweep(t: &[f64], y: &[f64], periods: &[f64]) -> Vec<f64> {
    let idx: Vec<usize> = (0..t.len()).filter(|&i| t[i].is_finite() && y[i].is_finite()).collect();
    let n = idx.len();
    if n < 4 || periods.is_empty() {
        return vec![0.0; periods.len()];
    }
    let mean = idx.iter().map(|&i| y[i]).sum::<f64>() / n as f64;
    let sst: f64 = idx.iter().map(|&i| (y[i] - mean).powi(2)).sum();
    if sst <= 1e-15 {
        return vec![0.0; periods.len()];
    }
    // Uniform grid check (binned series are uniform; NaN bins are simply skipped)
    let (t0, i_last) = (t[idx[0]], idx[n - 1]);
    let dt = if i_last > idx[0] { (t[i_last] - t0) / (i_last - idx[0]) as f64 } else { 0.0 };
    let uniform = dt > 0.0 && idx.iter().all(|&i| (t[i] - (t0 + (i - idx[0]) as f64 * dt)).abs() <= 1e-6 * dt.max(1e-9));
    let i0 = idx[0];

    periods
        .par_iter()
        .map(|&per| {
            let w = 2.0 * PI / per;
            let (mut sc, mut ss, mut scc, mut sss, mut scs, mut scy, mut ssy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let mut acc = |c: f64, s: f64, yc: f64| {
                sc += c;
                ss += s;
                scc += c * c;
                sss += s * s;
                scs += c * s;
                scy += c * yc;
                ssy += s * yc;
            };
            if uniform {
                let (sd, cd) = (w * dt).sin_cos();
                let (mut s, mut c) = (w * t0).sin_cos();
                let mut j = i0;
                for (step, &i) in idx.iter().enumerate() {
                    while j < i {
                        let cn = c * cd - s * sd;
                        s = s * cd + c * sd;
                        c = cn;
                        j += 1;
                    }
                    if step % 512 == 511 {
                        // resynchronise to bound recurrence drift
                        let (s2, c2) = (w * (t0 + (i - i0) as f64 * dt)).sin_cos();
                        s = s2;
                        c = c2;
                    }
                    acc(c, s, y[i] - mean);
                }
            } else {
                for &i in &idx {
                    let (s, c) = (w * t[i]).sin_cos();
                    acc(c, s, y[i] - mean);
                }
            }
            let nf = n as f64;
            let (c0, s0) = (sc / nf, ss / nf);
            let g11 = scc / nf - c0 * c0;
            let g22 = sss / nf - s0 * s0;
            let g12 = scs / nf - c0 * s0;
            let (rc, rs) = (scy / nf, ssy / nf);
            let det = g11 * g22 - g12 * g12;
            let r2 = if det > 1e-14 * (g11 * g22).max(1e-18) {
                let beta = (g22 * rc - g12 * rs) / det;
                let gamma = (g11 * rs - g12 * rc) / det;
                nf * (beta * rc + gamma * rs) / sst
            } else {
                // cos and sin collinear at this period: best single regressor
                let a = if g11 > 1e-15 { rc * rc / g11 } else { 0.0 };
                let b = if g22 > 1e-15 { rs * rs / g22 } else { 0.0 };
                nf * a.max(b) / sst
            };
            r2.clamp(0.0, 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_rhythm() {
        let t: Vec<f64> = (0..2016).map(|i| i as f64 / 6.0 + 1.0 / 12.0).collect();
        let y: Vec<f64> = t.iter().map(|&t| 10.0 + 4.0 * (2.0 * PI * (t - 15.0) / 24.0).cos() + 1.0 * (4.0 * PI * (t - 3.0) / 24.0).cos()).collect();
        let f = fit_cosinor(&t, &y, 24.0, 2, 0.05).unwrap();
        assert!((f.mesor - 10.0).abs() < 1e-8);
        assert!((f.amplitude() - 4.0).abs() < 1e-8);
        assert!((f.acrophase_h() - 15.0).abs() < 1e-6);
        assert!((f.harmonics[1].amp - 1.0).abs() < 1e-8);
        assert!(f.r2 > 0.999999);
        let periods: Vec<f64> = (0..=270).map(|i| 9.0 + 0.1 * i as f64).collect();
        let r2 = r2_sweep(&t, &y, &periods);
        let best = r2.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert!((periods[best] - 24.0).abs() < 1e-9);
        // uniform recurrence path agrees with a direct single-cosinor fit
        let f1 = fit_cosinor(&t, &y, 17.3, 1, 0.05).unwrap();
        let i = periods.iter().position(|p| (p - 17.3).abs() < 1e-9).unwrap();
        assert!((r2[i] - f1.r2).abs() < 1e-9);
    }
}
