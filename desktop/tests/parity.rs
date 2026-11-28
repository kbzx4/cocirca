//! Parity with the Python app (`cli.py --epoch-min 60`) on the bundled hourly activity dataset.

use cocirca::{data, pipeline, stats::hhmm};
use std::path::Path;

#[test]
fn matches_python_on_hourly_counts() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/activity_hourly_2023-01-26_to_02-03.csv");
    let ds = data::load_file(&path, None).expect("load");
    let cfg = pipeline::Config { epoch_sec: 3600, ..Default::default() };
    let rep = pipeline::run_analysis(&ds, &cfg).expect("analysis");
    let a = rep.activity.as_ref().expect("activity channel");
    assert_eq!(a.unit, "activity units");
    assert!((a.tau - 24.0).abs() < 1e-9);
    let s = &a.single;
    // Python: MESOR=14.90, Amp=20.09, Acrophase=12:07 (-181.7 deg), R2=0.559, p=3.26e-30
    assert!((s.mesor - 14.8981).abs() < 1e-3);
    assert!((s.amplitude() - 20.0862).abs() < 1e-3);
    assert_eq!(s.acrophase_hhmm(), "12:07");
    assert!((s.deg_neg() + 181.7).abs() < 0.05);
    assert!((s.r2 - 0.5587).abs() < 1e-3);
    assert!((s.p / 3.26e-30 - 1.0).abs() < 0.01);
    // 3-harmonic: R2=0.632, Orthophase=09:54, Bathyphase=03:51
    assert!((a.harmonics.r2 - 0.6325).abs() < 1e-3);
    assert_eq!(hhmm(a.harmonics.ortho_h), "09:54");
    assert_eq!(hhmm(a.harmonics.comp_bathy_h), "03:51");
    // Circular acrophase: Mean=12:07, R_bar=0.993, Rayleigh p=8.47e-05
    let c = a.circ_acrophase.as_ref().unwrap();
    assert_eq!(c.mean_hhmm(), "12:07");
    assert!((c.r_bar - 0.993).abs() < 1e-3);
    assert!((c.rayleigh_p / 8.47e-5 - 1.0).abs() < 0.01);
    // Watson-Williams: 11:52 -> 12:22, +0.51 h, p=0.1778
    let ww = a.shift.as_ref().unwrap().acrophase.as_ref().unwrap();
    assert_eq!((hhmm(ww.mean1), hhmm(ww.mean2)), ("11:52".to_string(), "12:22".to_string()));
    assert!((ww.p - 0.1778).abs() < 5e-4);
    // Rhythm bands, 12 h and bimodality
    let b = a.bands.as_ref().unwrap();
    assert_eq!(b.dominant_band, Some("Circadian"));
    assert!(b.band("Ultradian").classification.starts_with("Explained by circadian"));
    let t = a.twelve.as_ref().unwrap();
    assert!((t.harm_amp - 3.513).abs() < 1e-3 && !t.rhythm_sig && t.harm_sig && !t.distinct);
    let bm = a.bimodality.as_ref().unwrap();
    assert_eq!(bm.classification, "Unimodal");
    assert_eq!((bm.days_bimodal, bm.days_tested), (0, 7));
}
