# Circadian Phyphox Studio

A passive phone activity & ambient light circadian rhythm recorder, cosinor rhythmometry suite, circular statistics analyzer, actogram generator, and Excel report builder.

## CoSirca desktop app (Windows)

The same analysis as a single native Windows program, with no Python, browser or installer needed.
**[Download the latest release](https://github.com/kbzx4/cosirca/releases/latest)**: `cosirca.exe` (the app) and
`cosirca-cli.exe` (command line, for scripts and batch runs). Details and build instructions are in
[desktop/README.md](desktop/README.md).

### Latency

Measured on a real recording: an Empatica E4 wristband worn for 9.1 days (PhysioNet
[BIG IDEAs](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/), participant 001). It is
five CSV files (accelerometer at 32 Hz, EDA, skin temperature, heart rate, IBI) totalling 1.07 GB and
26.3 million rows, merged into one recording with five signals. Ryzen 7 7445HS, 32 GB RAM.

| Step | Desktop app | Streamlit app |
|---|---|---|
| Import dialog open with all 5 previews (from app launch) | 1.1–1.4 s | — |
| Import: read, parse and merge all 5 files | 1.7–2.3 s | 55 s |
| Analysis at 10-minute bins | 0.40 s (all 5 signals) | 2.7 s (main signal) |
| Analysis at 1-second bins (786,434 bins) | 0.71 s | — |
| **Import + analysis** | **about 2.5 s** | **58 s** |
| Peak memory | 3.1 GB | 8.5 GB |

The import preview of the largest file (879 MB accelerometer, first 5,000 rows) appears in 0.13 s.
Both apps give the same results. The Streamlit app's default upload limit is 200 MB, so files this
size need the desktop app.

## Features

- **Data Ingestion & Live Passive Recording** (see [Supported file formats](#supported-file-formats))
  - **Upload & review**: drop one or many files. The app shows the detected format, how time was read, which column feeds which channel (editable), and a raw preview before importing. Archives with several people / animals, DAM monitors, plate readers and gene tables become separate **subjects**: pick one, or **Analyse all subjects** for a summary table.
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

## Supported file formats

Both the Streamlit app (`circadian_app/formats.py`) and the desktop app (`desktop/src/import.rs`) use the same detection rules. Example files for every layout are in [data/examples](data/examples) (`python data/examples/make_examples.py` regenerates them).

| Area | Formats |
|---|---|
| Actigraphy | ActiGraph ActiLife CSV (with or without the column-name row; `.agd` in the Streamlit app), Philips Respironics Actiware (off-wrist / excluded epochs masked), Condor ActTrust (PIM, light, wrist temperature; decimal commas), Actiwatch / CamNtech `.awd`, GENEActiv CSV, Axivity CSV, MotionWare exports |
| Animal activity | TriKinetics DAM monitor files: each of the 32 channels is a subject; the light column is used when it toggles. Running-wheel, beam-break and generic counts |
| Light & temperature | HOBO (HOBOware CSV), iButton / Thermochron (OneWireViewer), any lux / °C / °F column |
| Heart | RR / IBI / NN interval files (heart rate = 60000 / RR, artifacts and ectopic beats rejected), Polar CSV, Empatica E4 (ACC, TEMP, HR, EDA, IBI) |
| Phones & wearables | Phyphox (`.zip` with `meta/time.csv` or single CSVs), Sensor Logger, Apple Health `export.xml`, Fitbit JSON (account export and Web-API intraday) |
| Lab tables | Plate-reader / bioluminescence tables (one subject per well column), gene-expression / hormone time courses with ZT / CT column headers (one subject per row, JTK_CYCLE / MetaCycle layout) |
| Containers | `.csv .tsv .txt .dat .xlsx .xls .xlsm .ods .json .jsonl .xml .zip .gz`; UTF-8, UTF-16 and Windows-1252 text |

Time stamps that are recognised:
- ISO 8601, with any time-zone suffix dropped so the wall-clock time is kept
- Day-first or month-first dates. When every day is ≤ 12 the order is decided by which reading spans the shorter time, and the import review flags it.
- Month names, 2-digit years and AM/PM
- Separate date and clock-time columns
- Unix seconds, milliseconds, microseconds and nanoseconds, converted to local time
- Excel serial dates and MATLAB datenums
- Relative seconds, minutes, hours or days, taken from the column name or set in the review
- ZT / CT hours and elapsed `hh:mm:ss`
- Files with no time column at all: set a start and a sampling interval

Binary formats (`.gt3x`, `.cwa`, GENEActiv `.bin`, `.mtn`, `.edf`) give a message saying how to export a CSV.

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

# Any supported format; pick a subject / main signal, or fix ambiguous dates
python cli.py Monitor7.txt --subject 3
python cli.py acttrust.txt --signal Temperature
python cli.py ibutton.csv --date-order dmy
python cli.py counts.txt --interval 60 --start-datetime 2024-05-01T08:00
```

### Run Automated Tests
```powershell
python -m unittest discover -s tests -v
```
