//! Periodograms (Lomb-Scargle ≡ cosinor R², Sokolove-Bushell χ²), ultradian / circadian / infradian
//! detection by sequential prewhitening, the 12 h rhythm vs τ/2 harmonic report, and bimodality.

use crate::cosinor::{fit_cosinor, r2_sweep, subtract_rhythm, CosinorFit};
use crate::stats::{chi2_ppf, chi2_sf, find_peaks, hhmm, local_maxima, median};
use rayon::prelude::*;

// ------------------------------------------------------------------------------------------------
// Periodograms
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Periodogram {
    pub periods: Vec<f64>,
    /// Floating-mean normalised Lomb-Scargle power; identical to the single-cosinor R² sweep
    pub ls_power: Vec<f64>,
    pub chi2_qp: Vec<f64>,
    pub chi2_crit: Vec<f64>,
    pub best_tau_ls: f64,
    pub best_ls: f64,
    pub best_tau_chi2: f64,
    pub best_qp: f64,
    pub best_chi2_p: f64,
    pub best_tau_cos: f64,
    pub best_cos_r2: f64,
    pub consensus_tau: f64,
}

pub fn period_grid(min: f64, max: f64, step: f64) -> Vec<f64> {
    let n = ((max + step * 0.5 - min) / step).ceil().max(0.0) as usize;
    (0..n).map(|i| min + i as f64 * step).collect()
}

/// Sokolove & Bushell χ² periodogram on a uniformly binned series (NaNs filled with the mean).
pub fn chi_square_periodogram(y: &[f64], epoch_min: f64, periods: &[f64], alpha: f64) -> (Vec<f64>, Vec<f64>) {
    let valid = y.iter().filter(|v| v.is_finite()).count();
    let zeros = (vec![0.0; periods.len()], vec![1.0; periods.len()]);
    if valid < 24 {
        return zeros;
    }
    let mean = y.iter().filter(|v| v.is_finite()).sum::<f64>() / valid as f64;
    let yc: Vec<f64> = y.iter().map(|&v| if v.is_finite() { v } else { mean }).collect();
    let n = yc.len();
    let var_total: f64 = yc.iter().map(|v| (v - mean).powi(2)).sum();
    if var_total <= 1e-15 {
        return zeros;
    }
    let dt_h = epoch_min / 60.0;
    let p_int: Vec<usize> = periods.iter().map(|&p| ((p * 60.0 / epoch_min).round() as i64).clamp(4, (n / 2).max(4) as i64) as usize).collect();
    let crit: Vec<f64> = p_int.par_iter().map(|&p| chi2_ppf(1.0 - alpha, (p as f64 - 1.0).max(1.0))).collect();
    let qp: Vec<f64> = periods
        .par_iter()
        .zip(&p_int)
        .map(|(&tau, &p)| {
            if n / p < 2 {
                return 0.0;
            }
            let mut sums = vec![0.0; p];
            let mut counts = vec![0u32; p];
            for (i, &v) in yc.iter().enumerate() {
                let th = i as f64 * dt_h;
                let b = ((th.rem_euclid(tau) / tau) * p as f64).floor() as usize % p;
                sums[b] += v;
                counts[b] += 1;
            }
            let ss_between: f64 = sums.iter().zip(&counts).filter(|(_, &c)| c > 0).map(|(&s, &c)| c as f64 * (s / c as f64 - mean).powi(2)).sum();
            n as f64 * ss_between / var_total
        })
        .collect();
    (qp, crit)
}

pub fn analyze_periodicity(t: &[f64], y: &[f64], epoch_min: f64, pmin: f64, pmax: f64, step: f64, circ: (f64, f64)) -> Result<Periodogram, String> {
    let valid = t.iter().zip(y).filter(|(a, b)| a.is_finite() && b.is_finite()).count();
    if valid < 6 {
        return Err("Need at least 6 valid epochs for periodicity analysis.".into());
    }
    let periods = period_grid(pmin, pmax, step);
    let (ls, (qp, crit)) = rayon::join(|| r2_sweep(t, y, &periods), || chi_square_periodogram(y, epoch_min, &periods, 0.05));
    let mut cidx: Vec<usize> = (0..periods.len()).filter(|&i| periods[i] >= circ.0 && periods[i] <= circ.1).collect();
    if cidx.is_empty() {
        cidx = (0..periods.len()).collect();
    }
    let argmax = |f: &dyn Fn(usize) -> f64| *cidx.iter().max_by(|&&a, &&b| f(a).partial_cmp(&f(b)).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    let i_ls = argmax(&|i| ls[i]);
    let i_chi = argmax(&|i| qp[i] - crit[i]);
    let tau_chi = periods[i_chi];
    let p_df = (((tau_chi * 60.0) / epoch_min).round() - 1.0).max(1.0);
    let consensus = median(&[periods[i_ls], tau_chi, periods[i_ls]]);
    Ok(Periodogram {
        best_tau_ls: periods[i_ls],
        best_ls: ls[i_ls],
        best_tau_chi2: tau_chi,
        best_qp: qp[i_chi],
        best_chi2_p: chi2_sf(qp[i_chi], p_df),
        best_tau_cos: periods[i_ls],
        best_cos_r2: ls[i_ls],
        consensus_tau: consensus,
        periods,
        ls_power: ls,
        chi2_qp: qp,
        chi2_crit: crit,
    })
}

// ------------------------------------------------------------------------------------------------
// Rhythm bands
// ------------------------------------------------------------------------------------------------

pub const BAND_NAMES: [&str; 3] = ["Ultradian", "Circadian", "Infradian"];

#[derive(Clone, Debug)]
pub struct BandRhythm {
    pub band: &'static str,
    pub low: f64,
    pub high: f64,
    pub detected: bool,
    pub classification: String,
    pub period: f64,
    pub amplitude: f64,
    pub acrophase_hhmm: String,
    pub percent_variance: f64,
    pub p_value: f64,
    pub p_adjusted: f64,
    pub n_independent: f64,
    pub order: Option<usize>,
    pub raw_peak_period: f64,
    pub raw_peak_r2: f64,
    pub note: String,
    pub scanned: bool,
}

impl BandRhythm {
    fn empty(band: &'static str, low: f64, high: f64, classification: String) -> Self {
        BandRhythm {
            band,
            low,
            high,
            detected: false,
            classification,
            period: f64::NAN,
            amplitude: f64::NAN,
            acrophase_hhmm: "--:--".into(),
            percent_variance: f64::NAN,
            p_value: f64::NAN,
            p_adjusted: f64::NAN,
            n_independent: f64::NAN,
            order: None,
            raw_peak_period: f64::NAN,
            raw_peak_r2: f64::NAN,
            note: String::new(),
            scanned: true,
        }
    }
    pub fn range_label(&self) -> String {
        if !self.scanned {
            if self.band == "Ultradian" {
                return format!("< {}", self.high);
            }
            if self.band == "Infradian" {
                return format!("> {}", self.low);
            }
        }
        format!("{}–{}", self.low, self.high)
    }
}

#[derive(Clone, Debug)]
pub struct RhythmBands {
    pub period_min: f64,
    pub period_max: f64,
    pub circ_min: f64,
    pub circ_max: f64,
    pub alpha: f64,
    pub bands: Vec<BandRhythm>,
    pub dominant_band: Option<&'static str>,
    pub dominant_period: f64,
    pub periods: Vec<f64>,
    pub raw_r2: Vec<f64>,
    pub residual_r2: Vec<f64>,
    pub span_h: f64,
}

impl RhythmBands {
    pub fn band(&self, name: &str) -> &BandRhythm {
        self.bands.iter().find(|b| b.band == name).unwrap()
    }
}

fn band_edges(pmin: f64, pmax: f64, cmin: f64, cmax: f64) -> [(f64, f64); 3] {
    [(pmin, cmin.min(pmax)), (cmin.max(pmin), cmax.min(pmax)), (cmax.max(pmin), pmax)]
}

fn band_of(p: f64, e: &[(f64, f64); 3]) -> Option<usize> {
    if e[0].1 > e[0].0 && e[0].0 <= p && p < e[0].1 {
        Some(0)
    } else if e[1].1 > e[1].0 && e[1].0 <= p && p <= e[1].1 {
        Some(1)
    } else if e[2].1 > e[2].0 && e[2].0 < p && p <= e[2].1 {
        Some(2)
    } else {
        None
    }
}

fn n_independent(span: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo || span <= 0.0 {
        1.0
    } else {
        (span * (1.0 / lo - 1.0 / hi)).max(1.0)
    }
}

pub fn sidak(p: f64, m: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    if p >= 1.0 {
        1.0
    } else {
        -(m * (-p).ln_1p()).exp_m1()
    }
}

fn resolution(period: f64, span: f64) -> f64 {
    (period * period / span.max(1e-9)).max(0.5)
}

fn max_harmonic(period: f64, pmin: f64, dt: f64) -> usize {
    let k_floor = (period / pmin).floor() as i64 + 1;
    let k_nyq = if dt > 0.0 { (period / (2.0 * dt)).floor() as i64 } else { k_floor };
    k_floor.min(k_nyq).clamp(1, 8) as usize
}

fn harmonic_match(period: f64, base: f64, pmin: f64, span: f64) -> Option<usize> {
    let kmax = (base / pmin).floor() as usize + 1;
    (2..=kmax).find(|&k| (period - base / k as f64).abs() <= resolution(base / k as f64, span))
}

pub fn analyze_rhythm_bands(t: &[f64], y: &[f64], periods: &[f64], raw_r2: &[f64], circ: (f64, f64), alpha: f64, min_pct: f64, max_extractions: usize) -> Result<RhythmBands, String> {
    let fin: Vec<usize> = (0..t.len()).filter(|&i| t[i].is_finite() && y[i].is_finite()).collect();
    if fin.len() < 8 {
        return Err("Need at least 8 valid samples for rhythm-band analysis.".into());
    }
    let tmin = fin.iter().map(|&i| t[i]).fold(f64::INFINITY, f64::min);
    let tmax = fin.iter().map(|&i| t[i]).fold(f64::NEG_INFINITY, f64::max);
    let span = tmax - tmin;
    let dt = {
        let mut ts: Vec<f64> = fin.iter().map(|&i| t[i]).collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median(&ts.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>())
    };
    let (pmin, pmax) = (periods[0], *periods.last().unwrap());
    let (cmin, cmax) = circ;
    let edges = band_edges(pmin, pmax, cmin, cmax);
    let nominal = [(cmin, cmin), (cmin, cmax), (cmax, cmax)];
    let m_ind: Vec<f64> = edges.iter().map(|e| n_independent(span, e.0, e.1)).collect();
    let ss = |v: &[f64]| {
        let vals: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        let m = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
        vals.iter().map(|x| (x - m).powi(2)).sum::<f64>()
    };
    let sst0 = ss(&fin.iter().map(|&i| y[i]).collect::<Vec<_>>());

    let mut detected: Vec<Option<BandRhythm>> = vec![None, None, None];
    let mut series = y.to_vec();
    let mut r2 = raw_r2.to_vec();
    let mut sst_cur = sst0;
    for order in 1..=max_extractions {
        if sst0 <= 1e-15 {
            break;
        }
        let mut cands: Vec<(f64, f64, usize)> = local_maxima(&r2)
            .into_iter()
            .filter_map(|i| band_of(periods[i], &edges).filter(|&b| detected[b].is_none()).map(|b| (r2[i], periods[i], b)))
            .collect();
        cands.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let mut chosen = None;
        for &(_, period, b) in &cands {
            let Some(fit) = fit_cosinor(t, &series, period, 1, 0.05) else { continue };
            let p_adj = sidak(fit.p, m_ind[b]);
            let pct = 100.0 * fit.r2 * sst_cur / sst0;
            if p_adj < alpha && pct >= min_pct {
                chosen = Some((period, b, fit, p_adj, pct));
                break;
            }
        }
        let Some((period, b, fit, p_adj, pct)) = chosen else { break };
        let name = BAND_NAMES[b];
        let mut br = BandRhythm::empty(name, edges[b].0, edges[b].1, format!("Significant {} rhythm", name.to_lowercase()));
        br.detected = true;
        br.period = period;
        br.amplitude = fit.amplitude();
        br.acrophase_hhmm = fit.acrophase_hhmm();
        br.percent_variance = pct;
        br.p_value = fit.p;
        br.p_adjusted = p_adj;
        br.n_independent = m_ind[b];
        br.order = Some(order);
        detected[b] = Some(br);
        // Prewhiten: remove this rhythm with its harmonics down to the scan floor
        let full = fit_cosinor(t, &series, period, max_harmonic(period, pmin, dt), 0.05).unwrap_or(fit);
        series = subtract_rhythm(t, &series, &full);
        r2 = r2_sweep(t, &series, periods);
        sst_cur = ss(&series);
    }

    let raw_peaks = local_maxima(raw_r2);
    let det_list: Vec<BandRhythm> = detected.iter().flatten().cloned().collect();
    let mut bands = Vec::new();
    for (b, &name) in BAND_NAMES.iter().enumerate() {
        let (lo, hi) = edges[b];
        let in_band: Vec<usize> = raw_peaks.iter().copied().filter(|&i| band_of(periods[i], &edges) == Some(b)).collect();
        let best_raw = in_band.iter().copied().max_by(|&a, &c| raw_r2[a].partial_cmp(&raw_r2[c]).unwrap());
        let (mut raw_p, mut raw_v) = best_raw.map_or((f64::NAN, f64::NAN), |i| (periods[i], raw_r2[i]));

        if let Some(mut br) = detected[b].clone() {
            br.raw_peak_period = raw_p;
            br.raw_peak_r2 = raw_v;
            for other in &det_list {
                if other.band == br.band || other.order.unwrap_or(99) > br.order.unwrap_or(99) {
                    continue;
                }
                let k_div = harmonic_match(br.period, other.period, pmin, span);
                let k_mul = harmonic_match(other.period, br.period, pmin, span);
                if k_div.is_some() || k_mul.is_some() {
                    let rel = k_div.map_or_else(|| format!("{}τ", k_mul.unwrap()), |k| format!("τ/{k}"));
                    let ob = other.band.to_lowercase();
                    br.classification = format!("Significant {} rhythm near {ob} {rel} (possible harmonic)", name.to_lowercase());
                    let op = k_div.map_or_else(|| format!("× {}", k_mul.unwrap()), |k| format!("÷ {k}"));
                    br.note = format!(
                        "{:.1} h ≈ {ob} {:.1} h {op}. It survived removal of that rhythm's exact harmonics, but day-to-day changes in the {ob} waveform also leave power near its harmonics, so treat it as an independent rhythm only with corroborating evidence.",
                        br.period, other.period
                    );
                }
            }
            bands.push(br);
            continue;
        }
        if hi <= lo {
            let (nl, nh) = nominal[b];
            let mut br = BandRhythm::empty(name, nl, nh, format!("Not scanned (outside {pmin}–{pmax} h range)"));
            br.scanned = false;
            bands.push(br);
            continue;
        }
        let mut br = BandRhythm::empty(name, lo, hi, format!("No significant {} rhythm", name.to_lowercase()));
        br.n_independent = m_ind[b];
        br.raw_peak_period = raw_p;
        br.raw_peak_r2 = raw_v;
        if best_raw.is_some() {
            let mut sorted = in_band.clone();
            sorted.sort_by(|&a, &c| raw_r2[c].partial_cmp(&raw_r2[a]).unwrap());
            let mut harmonic_of: Option<(BandRhythm, usize, usize, CosinorFit)> = None;
            'outer: for &i in &sorted {
                for d in &det_list {
                    let Some(k) = harmonic_match(periods[i], d.period, pmin, span) else { continue };
                    if let Some(fh) = fit_cosinor(t, y, periods[i], 1, 0.05) {
                        if sidak(fh.p, m_ind[b]) < alpha {
                            harmonic_of = Some((d.clone(), k, i, fh));
                            break 'outer;
                        }
                    }
                }
            }
            if let Some((d, k, i, fh)) = harmonic_of {
                raw_p = periods[i];
                raw_v = raw_r2[i];
                br.raw_peak_period = raw_p;
                br.raw_peak_r2 = raw_v;
                br.p_value = fh.p;
                br.p_adjusted = sidak(fh.p, m_ind[b]);
                let db = d.band.to_lowercase();
                br.classification = format!("Harmonic of {db} rhythm (τ/{k}), not independent");
                br.note = format!("Raw peak at {raw_p:.1} h ≈ {:.1} h ÷ {k}: reflects the {db} waveform's shape (e.g. bimodality) and disappears once that rhythm is removed.", d.period);
                bands.push(br);
                continue;
            }
            if let Some(fr) = fit_cosinor(t, y, raw_p, 1, 0.05) {
                br.p_value = fr.p;
                br.p_adjusted = sidak(fr.p, m_ind[b]);
            }
            if br.p_adjusted < alpha && !det_list.is_empty() {
                let dom = det_list.iter().min_by_key(|d| d.order.unwrap_or(99)).unwrap();
                br.classification = format!("Explained by {} rhythm (leakage), not independent", dom.band.to_lowercase());
                br.note = format!("Raw peak at {raw_p:.1} h is significant alone but vanishes after removing the {:.1} h rhythm.", dom.period);
            } else {
                br.note = format!("Strongest raw peak {raw_p:.1} h (R² {raw_v:.3}) is not significant.");
            }
        } else {
            br.note = format!("No local spectral peak between {lo} and {hi} h.");
            if b == 2 {
                let best = (0..periods.len()).filter(|&i| band_of(periods[i], &edges) == Some(2)).max_by(|&a, &c| raw_r2[a].partial_cmp(&raw_r2[c]).unwrap());
                if best == Some(periods.len() - 1) {
                    br.note += &format!(" Power rises toward the {pmax} h scan limit, so a longer infradian period may lie beyond the range.");
                }
            }
        }
        bands.push(br);
    }
    let dom = det_list.iter().min_by_key(|d| d.order.unwrap_or(99));
    Ok(RhythmBands {
        period_min: pmin,
        period_max: pmax,
        circ_min: cmin,
        circ_max: cmax,
        alpha,
        bands,
        dominant_band: dom.map(|d| d.band),
        dominant_period: dom.map_or(f64::NAN, |d| d.period),
        periods: periods.to_vec(),
        raw_r2: raw_r2.to_vec(),
        residual_r2: r2,
        span_h: span,
    })
}

// ------------------------------------------------------------------------------------------------
// 12 h rhythm vs τ/2 harmonic
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TwelveHour {
    pub tau: f64,
    pub window: (f64, f64),
    pub rhythm_period: f64,
    pub rhythm_amp: f64,
    pub rhythm_peaks: String,
    pub rhythm_pct: f64,
    pub rhythm_p: f64,
    pub rhythm_p_adj: f64,
    pub rhythm_sig: bool,
    pub at_edge: bool,
    pub harm_period: f64,
    pub harm_amp: f64,
    pub harm_peaks: String,
    pub harm_pct: f64,
    pub harm_p: f64,
    pub harm_sig: bool,
    pub ratio: f64,
    pub diff: f64,
    pub resolution: f64,
    pub distinct: bool,
    pub interpretation: String,
}

fn two_peaks(first: f64, period: f64) -> String {
    format!("{} & {}", hhmm(first), hhmm(first + period))
}

pub fn analyze_twelve_hour(t: &[f64], y: &[f64], tau: f64, window: (f64, f64), alpha: f64) -> Option<TwelveHour> {
    let fin: Vec<usize> = (0..t.len()).filter(|&i| t[i].is_finite() && y[i].is_finite()).collect();
    if fin.len() < 12 {
        return None;
    }
    let tmin = fin.iter().map(|&i| t[i]).fold(f64::INFINITY, f64::min);
    let tmax = fin.iter().map(|&i| t[i]).fold(f64::NEG_INFINITY, f64::max);
    let span = tmax - tmin;
    let ss = |v: &[f64]| {
        let vals: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        let m = vals.iter().sum::<f64>() / vals.len().max(1) as f64;
        vals.iter().map(|x| (x - m).powi(2)).sum::<f64>()
    };
    let sst0 = ss(&fin.iter().map(|&i| y[i]).collect::<Vec<_>>());
    if sst0 <= 1e-15 {
        return None;
    }
    let fit2 = fit_cosinor(t, y, tau, 2, 0.05)?;
    let (h1, h2) = (&fit2.harmonics[0], &fit2.harmonics[1]);
    let fit1 = fit_cosinor(t, y, tau, 1, 0.05)?;
    let resid = subtract_rhythm(t, y, &fit1);
    let (lo, hi) = window;
    let grid: Vec<f64> = (0..).map(|i| lo + 0.05 * i as f64).take_while(|&p| p <= hi + 1e-9).collect();
    let r2 = r2_sweep(t, &resid, &grid);
    let ib = (0..grid.len()).max_by(|&a, &b| r2[a].partial_cmp(&r2[b]).unwrap()).unwrap_or(0);
    let p12 = grid[ib];
    let f12 = fit_cosinor(t, &resid, p12, 1, 0.05)?;
    let sst_res = ss(&resid);
    let p_adj = sidak(f12.p, n_independent(span, lo, hi));
    let harm_period = tau / 2.0;
    let diff = p12 - harm_period;
    let res = (harm_period * harm_period / span.max(1e-9)).max(0.25);
    let distinct = diff.abs() > res;
    let rhythm_sig = p_adj < alpha;
    let harm_sig = h2.p < alpha;
    let mut interp = if !rhythm_sig && !harm_sig {
        "No significant 12 h component: both the 12 h rhythm and the τ/2 harmonic are weak (values shown for reference).".to_string()
    } else if distinct && rhythm_sig {
        format!("The 12 h rhythm's own period ({p12:.2} h) differs from τ/2 ({harm_period:.2} h) by more than the resolution (±{res:.2} h): evidence of an ultradian rhythm independent of the circadian waveform.")
    } else {
        format!("The 12 h rhythm's period ({p12:.2} h) matches τ/2 ({harm_period:.2} h) within ±{res:.2} h, so it cannot be separated from the circadian harmonic. It describes the circadian waveform's shape (e.g. a second daily peak or a midday dip).")
    };
    let at_edge = ib == 0 || ib == grid.len() - 1;
    if at_edge {
        interp += &format!(" The best period lies at the {lo}–{hi} h search edge, so the true 12 h-band peak may lie outside it.");
    }
    Some(TwelveHour {
        tau,
        window,
        rhythm_period: p12,
        rhythm_amp: f12.amplitude(),
        rhythm_peaks: two_peaks(f12.harmonics[0].peak_h, p12),
        rhythm_pct: 100.0 * f12.r2 * sst_res / sst0,
        rhythm_p: f12.p,
        rhythm_p_adj: p_adj,
        rhythm_sig,
        at_edge,
        harm_period,
        harm_amp: h2.amp,
        harm_peaks: two_peaks(h2.peak_clock, harm_period),
        harm_pct: h2.partial_r2 * 100.0,
        harm_p: h2.p,
        harm_sig,
        ratio: if h1.amp > 1e-12 { h2.amp / h1.amp } else { f64::NAN },
        diff,
        resolution: res,
        distinct,
        interpretation: interp,
    })
}

// ------------------------------------------------------------------------------------------------
// Bimodality
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct WaveShape {
    pub primary_h: f64,
    pub primary_val: f64,
    pub secondary_h: f64,
    pub secondary_val: f64,
    pub secondary_prom: f64,
    pub dip_h: f64,
    pub dip_val: f64,
    pub bimodality_index: f64,
    pub range: f64,
}

#[derive(Clone, Debug)]
pub struct DayBimodality {
    pub day: i32,
    pub bimodal: bool,
    pub index: f64,
    pub peak1_h: f64,
    pub peak2_h: f64,
}

#[derive(Clone, Debug)]
pub struct Bimodality {
    pub period: f64,
    pub is_bimodal: bool,
    pub classification: String,
    pub shape: WaveShape,
    pub h2_h1: f64,
    pub h2_p: f64,
    pub r2: f64,
    pub min_prominence: f64,
    pub days_tested: usize,
    pub days_bimodal: usize,
    pub daily: Vec<DayBimodality>,
    pub wave_h: Vec<f64>,
    pub wave_v: Vec<f64>,
}

pub fn describe_waveform(fit: &CosinorFit, n_grid: usize) -> (WaveShape, Vec<f64>, Vec<f64>) {
    let p = fit.period;
    let grid: Vec<f64> = (0..n_grid).map(|i| p * i as f64 / n_grid as f64).collect();
    let wave: Vec<f64> = grid.iter().map(|&g| fit.eval(g)).collect();
    let (mut i_max, mut vmax, mut vmin) = (0, f64::NEG_INFINITY, f64::INFINITY);
    for (i, &v) in wave.iter().enumerate() {
        if v > vmax {
            vmax = v;
            i_max = i;
        }
        vmin = vmin.min(v);
    }
    let rng = vmax - vmin;
    let mut shape = WaveShape { primary_h: grid[i_max], primary_val: vmax, range: rng, secondary_h: f64::NAN, secondary_val: f64::NAN, dip_h: f64::NAN, dip_val: f64::NAN, ..Default::default() };
    if rng <= 1e-12 {
        return (shape, grid, wave);
    }
    let tiled: Vec<f64> = wave.iter().chain(&wave).chain(&wave).copied().collect();
    let (pk, prom) = find_peaks(&tiled, None, Some(0.0));
    let others: Vec<(f64, usize)> = pk
        .iter()
        .zip(&prom)
        .filter(|(&i, _)| i >= n_grid && i < 2 * n_grid)
        .map(|(&i, &pr)| (pr, i - n_grid))
        .filter(|&(_, i)| (i as i64 - i_max as i64).abs() > 1)
        .collect();
    let Some(&(sec_prom, j)) = others.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) else { return (shape, grid, wave) };
    let arc_min = |arc: Vec<usize>| *arc.iter().min_by(|&&a, &&b| wave[a].partial_cmp(&wave[b]).unwrap()).unwrap();
    let (arc_a, arc_b): (Vec<usize>, Vec<usize>) = if j > i_max {
        ((i_max..=j).collect(), (j..n_grid).chain(0..=i_max).collect())
    } else {
        ((j..=i_max).collect(), (i_max..n_grid).chain(0..=j).collect())
    };
    let (ma, mb) = (arc_min(arc_a), arc_min(arc_b));
    let dip = if wave[ma] >= wave[mb] { ma } else { mb };
    shape.secondary_h = grid[j];
    shape.secondary_val = wave[j];
    shape.secondary_prom = sec_prom;
    shape.dip_h = grid[dip];
    shape.dip_val = wave[dip];
    shape.bimodality_index = sec_prom / rng;
    (shape, grid, wave)
}

/// `days` = per-day (day_index, clock hours, values) used for the day-by-day classification.
pub fn analyze_bimodality(t: &[f64], y: &[f64], period: f64, days: &[(i32, &[f64], &[f64])], epoch_min: f64, min_prom: f64, alpha: f64) -> Option<Bimodality> {
    let fit = fit_cosinor(t, y, period, 2, 0.05)?;
    let (shape, grid, wave) = describe_waveform(&fit, 1440);
    let (h1, h2) = (&fit.harmonics[0], &fit.harmonics[1]);
    let bimodal = shape.bimodality_index >= min_prom && h2.p < alpha;
    let classification = if bimodal {
        "Bimodal"
    } else if shape.bimodality_index >= min_prom {
        "Unimodal (second peak not significant)"
    } else if shape.bimodality_index > 0.0 {
        "Unimodal (minor shoulder)"
    } else {
        "Unimodal"
    }
    .to_string();

    let bins_per_day = ((24.0 * 60.0 / epoch_min).round() as usize).max(1);
    let min_valid = 8.max((0.5 * bins_per_day as f64).ceil() as usize);
    let daily: Vec<DayBimodality> = days
        .par_iter()
        .filter_map(|&(d, ch, v)| {
            let n_valid = v.iter().filter(|x| x.is_finite()).count();
            if n_valid < min_valid || crate::stats::finite_std(v) <= 1e-12 {
                return None;
            }
            let dfit = fit_cosinor(ch, v, 24.0, 2, 0.05)?;
            let (ds, _, _) = describe_waveform(&dfit, 720);
            Some(DayBimodality {
                day: d,
                bimodal: ds.bimodality_index >= min_prom && dfit.harmonics[1].p < alpha,
                index: (ds.bimodality_index * 1000.0).round() / 1000.0,
                peak1_h: ds.primary_h,
                peak2_h: if ds.bimodality_index > 0.0 { ds.secondary_h } else { f64::NAN },
            })
        })
        .collect();
    Some(Bimodality {
        period,
        is_bimodal: bimodal,
        classification,
        h2_h1: if h1.amp > 1e-12 { h2.amp / h1.amp } else { f64::NAN },
        h2_p: h2.p,
        r2: fit.r2,
        min_prominence: min_prom,
        days_tested: daily.len(),
        days_bimodal: daily.iter().filter(|d| d.bimodal).count(),
        daily,
        shape,
        wave_h: grid,
        wave_v: wave,
    })
}
