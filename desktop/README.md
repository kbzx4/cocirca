# CoCirca desktop (Rust)

A native Windows desktop version of Circadian Phyphox Studio. It does the same analysis as the Python
app, gives the same numbers, and is 15–50× faster. It is a single 7.5 MB executable with no Python,
browser or server needed.

## Run

```
desktop\target\release\cocirca.exe                 # GUI
desktop\target\release\cocirca.exe data.csv        # GUI, opening a file
desktop\target\release\cocirca-cli.exe data.csv --bin-sec 600 --excel report.xlsx --csv epochs.csv
```

In the GUI, open files from the toolbar (Ctrl+O), click the welcome screen's drop zone, or drag files
onto the window. The **import dialog** then shows, for every file:
- the detected format and a summary (subjects, rows, duration, sampling interval)
- how time was read
- each column's role, which you can change from a drop-down
- the first and last parsed times
- a raw preview with the column-name row highlighted

From the same dialog you can override the date order, the time unit, the recording start (for relative
times), the sampling interval (for files with no time stamps), the header row, and the layout (columns
as channels or as subjects). For multiple files, choose whether to merge them or keep them as separate
subjects. "Import directly next time" skips the dialog when the format is recognised, and
**Data → Import settings…** reopens it.

Multi-subject imports (DAM monitors, plate readers, gene tables, archives of recordings) get a
searchable subject picker and **Analyse all subjects**, which fills a summary table you can export as CSV.

Every channel in a file is analysed: activity, light, heart rate, temperature, and other signals.
The full list of supported formats is in the main [README](../README.md#supported-file-formats).
`cocirca-cli` accepts the same import options (`--date-order`, `--time-unit`, `--interval`,
`--header-line`, `--layout`, `--subject`, `--merge`/`--separate`), and `--batch` runs every subject
of a file, archive or folder.

**RR / inter-beat interval files** (Holter, chest straps; one interval per line in ms or s, optional
`RR` / `IBI` / `NN` header) are recognised automatically and analysed as heart rate (bpm).
- **Beat times:** built from the cumulative RR intervals.
- **Rejected intervals:** anything outside 250–2500 ms or more than 25% from the median of the surrounding 11 beats is rejected as an artifact or ectopic beat. Rejected intervals still count toward elapsed time.
- **Start time:** these files carry none, so pass `--start "YYYY-MM-DD HH:MM"` to get real clock phases. Without it, phases are hours after the recording began.
- **Single-day recordings:** use `--tau fixed24`.

An archive or folder of separate recordings (one per person) is never merged. Use batch mode, which
analyses each file in parallel and writes one summary row per file:

```
cocirca-cli --batch rr-interval-healthy-subjects-1.0.0.zip --tau fixed24 --bin-sec 600 --out summary.csv
```

## Features

| Area | What you get |
|---|---|
| Binning | Bin size from 1 s to 6 h, with bin-centre timestamps and short-gap interpolation |
| Periodicity | Lomb-Scargle (≡ cosinor R² sweep), Sokolove-Bushell χ² with critical line, and consensus τ over a 9–36 h scan (adjustable) |
| Cosinor | Single fit and 1–6 harmonics, with 95% CIs, partial F-tests, orthophase/bathyphase and a population-mean cosinor |
| Rhythm bands | Ultradian / circadian / infradian detection by sequential prewhitening with Šidák correction, plus harmonic and leakage classification |
| 12 h | The free-period 12 h rhythm and the τ/2 harmonic, both always shown, with a test of whether they can be told apart |
| Bimodality | 2-harmonic waveform peaks, dip and bimodality index, overall and per day |
| Circular | Mean phase, R̄, κ and Rayleigh for acrophase, onset and offset, with a polar plot |
| Phase shift | Day-to-day shift, unwrapped drift, implied τ, Watson-Williams pre/post test and the activity–light phase angle |
| Non-parametric | IS, IV, RA, M10, L5 and the 24 h profile |
| Charts | Double-plotted actogram with light overlay and onset/offset markers, plus time series, waveform, periodograms, bands, polar plot and shift timeline. All charts zoom, pan and show hover readouts |
| Theme | Background (preset or custom), 5 palettes and per-role colours. Axis labels: font (installed Windows fonts), size, bold, separate X/Y colours, and renaming any label |
| Export | Excel workbook with native, theme-styled charts; full-resolution epochs CSV; PNG of the current chart |

## Performance

Measured on 14 days of 1-second data (1,209,600 rows, activity + light). Times are wall-clock and include the Excel export.

| Bin size | Python app (`cli.py`) | CoCirca desktop | Speed-up |
|---|---|---|---|
| 10 min | 8.5 s | 0.63 s | ≈ 13× |
| 1 s | 50.5 s | 1.7 s | ≈ 30× |

Analysis alone takes 12 ms at 10-minute bins and 0.27 s at 1-second bins. CSV parsing takes most of the remaining time.

### Large real recording

PhysioNet [BIG IDEAs](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.2/) participant 001:
an Empatica E4 wristband worn for 9.1 days, as five CSV files: accelerometer at 32 Hz (879 MB), EDA, skin
temperature, heart rate and IBI. Together they are 1.07 GB and 26.3 million rows, merged into one
recording with five channels. Measured on a Ryzen 7 7445HS (6 cores) with 32 GB RAM.

| Step | CoCirca desktop | Python app |
|---|---|---|
| Import dialog open with all 5 previews (from app launch) | 1.1–1.4 s | — |
| Import: read, parse and merge all 5 files | 1.7–2.3 s | 55 s |
| Analysis at 10-minute bins | 0.40 s (all 5 channels) | 2.7 s (main signal) |
| Analysis at 1-second bins (786,434 bins) | 0.71 s | — |
| Peak memory | 3.1 GB | 8.5 GB |

Both apps give the same results (activity MESOR 18.95 mg, amplitude 9.47 mg, acrophase 15:31).
The Streamlit app's default upload limit is 200 MB, so files this size need the desktop app or the CLI.

To reproduce (in Git Bash):

```
bash desktop/scripts/fetch_big_ideas.sh            # ~1 GB into bench_data/ (git-ignored), checks SHA-256
cd desktop
cargo run --release --example bench_import -- --bins 600,60,1 ../bench_data/big-ideas-001/*.csv
```

Why it's fast:
- **Single-pass cosinor.** X′X is accumulated in one pass, with harmonics from angle-addition recurrences, so the n×p design matrix is never built.
- **Parallel period sweeps.** The R² / Lomb-Scargle and χ² sweeps run in parallel (rayon). Uniformly spaced data uses a rotation recurrence instead of calling sin/cos for every sample.
- **One spectrum, two uses.** The Lomb-Scargle (floating-mean, normalised) power is numerically identical to the cosinor R² sweep, so it is computed once.
- **Concurrent analyses.** The light and activity channels, and the band, 12 h, bimodality and cosinor analyses within each, run concurrently.
- **Responsive UI.** Analysis runs off the UI thread with a 250 ms debounce. Time series are min/max-decimated to the visible window each frame. The actogram is drawn as one GPU mesh.
- **Optimised build.** The release profile uses full LTO, one codegen unit and `panic = abort`.

## Build from source

```
winget install Rustlang.Rustup                     # or https://rustup.rs
winget install BrechtSanders.WinLibs.POSIX.UCRT    # MinGW binutils (dlltool) for the GNU toolchain
cd desktop
.\build.ps1                                        # cargo build --release + tests
```

The MSVC toolchain (Visual Studio Build Tools) also works. If you use it, you don't need WinLibs.

`cargo test` runs the unit tests and a parity test that checks the results against the Python app's
output on `data/activity_hourly_2023-01-26_to_02-03.csv`.

## Layout

```
src/import.rs      format detection and parsing (all supported formats), parallel row parser, merging
src/import_ui.rs   the import dialog
src/data.rs        dataset model, RR → heart rate, k-way merge, synthetic demo, epoch binning
src/cosinor.rs     streaming cosinor fit, R² period sweep
src/rhythm.rs      periodograms, rhythm bands (prewhitening), 12 h analysis, bimodality
src/circular.rs    circular stats, Watson-Williams, acrophase shift
src/nonparam.rs    IS / IV / M10 / L5 / RA, onset / offset detection
src/pipeline.rs    end-to-end analysis and daily table
src/theme.rs       colour roles, palettes, axis-label style
src/charts.rs      egui_plot charts, actogram mesh, polar plot
src/export.rs      Excel (rust_xlsxwriter) and CSV export
src/app.rs         the egui application
src/cli.rs         headless runner (cocirca-cli)
examples/bench_import.rs     latency benchmark (import + analysis) on any files
scripts/fetch_big_ideas.sh   downloads the 1 GB real recording used in the benchmark
```
