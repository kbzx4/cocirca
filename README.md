# Circadian Phyphox Studio

A passive phone activity & ambient light circadian rhythm recorder, cosinor rhythmometry suite, circular statistics analyzer, actogram generator, and Excel report builder.

## Features

- **Data Ingestion & Live Passive Recording**
  - **Phyphox `.zip` archives**: Automatically reads `meta/time.csv` for real wall-clock timestamps alongside `Light.csv`, `Accelerometer.csv`, and `Linear Acceleration.csv`.
  - **CSV / TSV / XLSX**: Auto-detects timestamps (ISO datetime, Unix epoch, or elapsed `Time (s)`), illuminance (`lux`), 3-axis acceleration (`ENMO`, `MAD`, `Vector Magnitude`), or precomputed activity counts.
  - **Live Phyphox Remote Access Recorder**: Polls Phyphox's `/config` and `/get` HTTP API on your local network (plus a simulated live phone stream mode for testing) and logs passively to CSV.
  - **Synthetic Circadian Demo Generator**: Multi-day activity + light datasets with configurable $\tau$, harmonics, and phase-shift events.
- **Periodicity & Rhythmometry**
  - **Periodograms**: Lomb-Scargle, Sokolove-Bushell $\chi^2$ Periodogram, FFT power spectrum, and Cosinor $R^2$ $\tau$-sweep with consensus $\tau$, scanning 9–36 h by default (adjustable).
  - **Ultradian / Circadian / Infradian Rhythms**: classifies significant rhythms into ultradian (< 20 h), circadian (20–28 h) and infradian (> 28 h) bands (boundaries adjustable). Rhythms are extracted strongest-first; each is removed with its harmonics before the next is tested (Šidák-corrected per band, ≥ 2 % of variance), so harmonics (e.g. 12 h from a non-sinusoidal 24 h rhythm) and spectral sidelobes are not misreported as separate rhythms.
  - **Bimodality**: 24 h + 12 h cosinor waveform test for two daily peaks (e.g. morning & evening activity), with bimodality index, peak and dip times, and a per-day bimodal count.
  - **Single & Multi-Harmonic Cosinor**: MESOR, Amplitude, Acrophase (HH:MM, degrees, radians), 95% CIs, Zero-Amplitude $F$-test, per-harmonic decomposition ($H_1..H_6$), Partial %PR, Orthophase/Bathyphase, Daily Serial Section, and Population-Mean Cosinor.
- **Onsets / Offsets, Non-Parametric & Circular Statistics**
  - **Onset / Offset Detection**: Edge-template, adaptive threshold with hysteresis, and cosinor MESOR-crossing detectors for daily onset, offset, active duration ($\alpha$), midpoint, and activity–light phase angle ($\psi$).
  - **Non-Parametric Variables**: Interdaily Stability (IS), Intradaily Variability (IV), M10, L5, and Relative Amplitude (RA).
  - **Circular Statistics Recorder**: Circular mean, resultant length $\bar{R}$, circular variance & SD, von Mises $\kappa$, Rayleigh uniformity test, and Watson-Williams two-sample circular test.
  - **Acrophase Shift Analysis**: Day-to-day wrapped $\Delta\phi$, unwrapped cumulative drift slope, implied $\tau$, and pre- vs. post-split shift test.
- **Onscreen & Excel Outputs**
  - **Interactive Streamlit Dashboard** (`app.py`) with Single/Double-plotted Actograms, Harmonics, Periodograms, Polar Rose plots, and Shift timelines.
  - **Multi-Sheet Excel Workbook** (`Summary_Report`, `Cosinor_Harmonics`, `Daily_Rhythms_Onsets`, `Circular_Statistics` with live Excel circular formulas, `Periodogram`, `Rhythm_Bands`, `Binned_Epochs`) with embedded charts.

## Quick Start

### Launch the Interactive Dashboard
```powershell
streamlit run app.py
```

### Run Headless CLI Analysis & Excel Export
```powershell
# Run on synthetic 14-day demo dataset
python cli.py --demo --output-dir output

# Run on a Phyphox zip export or CSV files
python cli.py path/to/phyphox_export.zip --epoch-min 10 --harmonics 3 --output-dir output

# Change the period scan range and the circadian band (ultradian < 20 h < circadian < 28 h < infradian)
python cli.py data.csv --period-min 9 --period-max 36 --circadian-min 20 --circadian-max 28
```

### Run Automated Tests
```powershell
python -m unittest discover -s tests -v
```
