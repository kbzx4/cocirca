//! The desktop UI: settings side panel, chart tabs, background loading / analysis, export.

use crate::charts::{self, family_key, ActoOpts, ChartCtx, Extras};
use crate::data::{self, format_bin_size, ActivityMethod, RawDataset};
use crate::import::{self, ImportResult, Parsed, Source};
use crate::import_ui::{self, ImportSession, PendingFile, PREVIEW_ROWS};
use crate::nonparam::OnsetMethod;
use crate::pipeline::{run_analysis, Config, PeriodMode, Report};
use crate::rhythm::BAND_NAMES;
use crate::stats::hhmm;
use crate::theme::{self, Theme, ThemeSettings, BACKGROUNDS, PALETTES, ROLES};
use eframe::egui::{self, Color32, FontData, FontDefinitions, FontFamily, RichText};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BIN_SIZES: [i64; 19] = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 10800, 14400, 21600];

/// Axis labels the charts use; labels with channel units are added once drawn.
const KNOWN_LABELS: [&str; 16] = [
    "Clock Time (Double-Plotted 48 h)",
    "Clock Time (24 h)",
    "Elapsed Time (hours)",
    "Recording Day",
    "Hour of Day",
    "Phase / Clock Hour (h)",
    "Period τ (hours)",
    "Normalized Power",
    "Qp",
    "R² (variance explained)",
    "Day Index",
    "Clock Time",
    "Cumulative Shift (h)",
    "Day-to-Day Shift Δφ (h)",
    "Activity (activity units)",
    "Light (log10(lux + 1))",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Overview,
    Actogram,
    TimeSeries,
    Waveform,
    Periodogram,
    Bands,
    Circular,
    Shift,
    Profile,
    Daily,
    Subjects,
}

const TABS: [(Tab, &str); 11] = [
    (Tab::Overview, "Overview"),
    (Tab::Actogram, "Actogram"),
    (Tab::TimeSeries, "Time series"),
    (Tab::Waveform, "Cosinor waveform"),
    (Tab::Periodogram, "Periodogram"),
    (Tab::Bands, "Rhythm bands & bimodality"),
    (Tab::Circular, "Circular stats"),
    (Tab::Shift, "Acrophase shift"),
    (Tab::Profile, "24 h profile"),
    (Tab::Daily, "Daily table"),
    (Tab::Subjects, "All subjects"),
];

pub struct Computed {
    pub report: Report,
    pub extras: Vec<Extras>,
}

/// One summary row per subject ("Analyse all subjects")
pub struct BatchTable {
    pub head: Vec<String>,
    pub rows: Vec<(String, Result<Vec<String>, String>)>,
    pub ms: f64,
}

enum Msg {
    /// Files read and containers expanded: (load generation, sources, read errors, paths)
    Sources(u64, Vec<Source>, Vec<String>),
    /// Preview parse of one pending file: (session, file index, option version, result)
    Preview(u64, usize, u64, Box<Result<Parsed, String>>),
    Imported(u64, Result<ImportResult, String>, f64),
    Demo(u64, RawDataset),
    Analysed(u64, Result<Computed, String>),
    Batch(u64, BatchTable),
}

pub struct CosircaApp {
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    dataset: Option<Arc<RawDataset>>,
    subjects: Vec<Arc<RawDataset>>,
    subject_idx: usize,
    subject_filter: String,
    import: Option<ImportSession>,
    /// Last confirmed import session, so its settings can be reopened
    last_import: Option<ImportSession>,
    session_seq: u64,
    skip_dialog: bool,
    /// Current session was opened with "import directly" on
    auto_import: bool,
    import_notes: Vec<String>,
    batch: Option<BatchTable>,
    batch_gen: u64,
    batch_running: bool,
    load_gen: u64,
    run_gen: u64,
    loading: bool,
    running: bool,
    cfg: Config,
    requested_cfg: Option<Config>,
    pending_since: Option<Instant>,
    bin_idx: usize,
    computed: Option<Arc<Computed>>,
    error: Option<String>,
    status: String,
    tab: Tab,
    channel: usize,
    acto: ActoOpts,
    show_fits: bool,
    theme_settings: ThemeSettings,
    font_defs: FontDefinitions,
    loaded_fonts: HashSet<String>,
    fonts_next_frame: Vec<String>,
    available_fonts: Vec<&'static str>,
    seen_labels: BTreeSet<String>,
    chart_rect: Option<egui::Rect>,
    pending_png: Option<PathBuf>,
    custom_label_colors: bool,
    /// --shot PATH: render, save a full-window PNG, exit (automated visual checks)
    auto_shot: Option<(PathBuf, u32)>,
    /// --shot-dialog: take the screenshot of the import dialog instead of the analysis
    shot_dialog: bool,
    /// --analyse-all: run "Analyse all subjects" once imported (for screenshots)
    shot_batch: bool,
    /// --subject N: show subject N once imported (for screenshots)
    shot_subject: usize,
}

impl CosircaApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut defs = FontDefinitions::default();
        // Windows symbol fonts as fallbacks so τ, χ², ≈, Δφ, R̄ … always render
        let dir = theme::fonts_dir();
        for (key, file) in [("segoe-ui", "segoeui.ttf"), ("segoe-symbol", "seguisym.ttf")] {
            if let Ok(bytes) = std::fs::read(dir.join(file)) {
                defs.font_data.insert(key.into(), Arc::new(FontData::from_owned(bytes)));
                for fam in [FontFamily::Proportional, FontFamily::Monospace] {
                    defs.families.entry(fam).or_default().push(key.into());
                }
            }
        }
        cc.egui_ctx.set_fonts(defs.clone());
        cc.egui_ctx.style_mut(|s| {
            s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        });
        let (tx, rx) = channel();
        let cfg = Config::default();
        let bin_idx = BIN_SIZES.iter().position(|&b| b == cfg.epoch_sec).unwrap_or(10);
        let mut app = CosircaApp {
            tx,
            rx,
            dataset: None,
            subjects: Vec::new(),
            subject_idx: 0,
            subject_filter: String::new(),
            import: None,
            last_import: None,
            session_seq: 0,
            skip_dialog: false,
            auto_import: false,
            import_notes: Vec::new(),
            batch: None,
            batch_gen: 0,
            batch_running: false,
            load_gen: 0,
            run_gen: 0,
            loading: false,
            running: false,
            cfg,
            requested_cfg: None,
            pending_since: None,
            bin_idx,
            computed: None,
            error: None,
            status: "Open or drop data files (actigraphy, light / temperature loggers, heart rate, wearables, DAM, plate reader, gene expression …), or load the demo.".into(),
            tab: Tab::Overview,
            channel: 0,
            acto: ActoOpts { double: true, light: true, markers: true },
            show_fits: true,
            theme_settings: ThemeSettings::default(),
            font_defs: defs,
            loaded_fonts: HashSet::new(),
            fonts_next_frame: Vec::new(),
            available_fonts: theme::available_fonts(),
            seen_labels: KNOWN_LABELS.iter().map(|s| s.to_string()).collect(),
            chart_rect: None,
            pending_png: None,
            custom_label_colors: false,
            auto_shot: None,
            shot_dialog: false,
            shot_batch: false,
            shot_subject: 0,
        };
        // Files passed on the command line (e.g. "Open with")
        let mut files: Vec<PathBuf> = Vec::new();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let next = args.get(i + 1).cloned().unwrap_or_default();
            match args[i].as_str() {
                "--shot" => {
                    app.auto_shot = Some((PathBuf::from(&next), 0));
                    i += 1;
                }
                "--tab" => {
                    if let Some(t) = TABS.iter().find(|t| t.1.to_lowercase().starts_with(&next.to_lowercase())) {
                        app.tab = t.0;
                    }
                    i += 1;
                }
                "--channel" => {
                    app.channel = next.parse().unwrap_or(0);
                    i += 1;
                }
                "--bg" => {
                    app.theme_settings.background_preset = next.clone();
                    i += 1;
                }
                "--palette" => {
                    app.theme_settings.palette = next.clone();
                    i += 1;
                }
                "--font" => {
                    app.theme_settings.label.font = Some(next.clone());
                    i += 1;
                }
                "--font-size" => {
                    app.theme_settings.label.size = next.parse().unwrap_or(0.0);
                    i += 1;
                }
                "--bold" => app.theme_settings.label.bold = true,
                "--x-color" => {
                    app.theme_settings.label.x_color = Some(theme::hex(&next));
                    i += 1;
                }
                "--y-color" => {
                    app.theme_settings.label.y_color = Some(theme::hex(&next));
                    i += 1;
                }
                "--rename" => {
                    if let Some((o, n)) = next.split_once('=') {
                        app.theme_settings.label.renames.push((o.to_string(), n.to_string()));
                    }
                    i += 1;
                }
                "--tau" => {
                    app.cfg.period_mode = match next.as_str() {
                        "fixed24" => PeriodMode::Fixed24,
                        "ls" => PeriodMode::LombScargle,
                        "chi2" => PeriodMode::Chi2,
                        _ => PeriodMode::Consensus,
                    };
                    i += 1;
                }
                "--bin-sec" => {
                    if let Ok(v) = next.parse::<i64>() {
                        app.bin_idx = BIN_SIZES.iter().position(|&b| b >= v).unwrap_or(BIN_SIZES.len() - 1);
                    }
                    i += 1;
                }
                "--demo" => app.load_demo(&cc.egui_ctx),
                "--direct" => app.skip_dialog = true,
                "--shot-dialog" => app.shot_dialog = true,
                "--analyse-all" => app.shot_batch = true,
                "--subject" => {
                    app.shot_subject = next.parse::<usize>().unwrap_or(1).saturating_sub(1);
                    i += 1;
                }
                a if !a.starts_with("--") => files.push(PathBuf::from(a)),
                _ => {}
            }
            i += 1;
        }
        if !files.is_empty() {
            app.load_files(files, &cc.egui_ctx);
        }
        app
    }

    /// Step 1 of importing: read the files (expanding .zip / .gz) off the UI thread.
    fn load_files(&mut self, files: Vec<PathBuf>, ctx: &egui::Context) {
        self.load_gen += 1;
        let gen = self.load_gen;
        self.loading = true;
        self.error = None;
        self.status = format!("Reading {} file(s)…", files.len());
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let results: Vec<Result<Vec<Source>, String>> = files.par_iter().map(|f| import::read_path(f)).collect();
            let (mut sources, mut errors) = (Vec::new(), Vec::new());
            for r in results {
                match r {
                    Ok(s) => sources.extend(s),
                    Err(e) => errors.push(e),
                }
            }
            let _ = tx.send(Msg::Sources(gen, sources, errors));
            ctx.request_repaint();
        });
    }

    /// Step 2: open the import dialog and preview every file in parallel.
    fn open_session(&mut self, sources: Vec<Source>, ctx: &egui::Context) {
        self.session_seq += 1;
        let mut session = ImportSession { id: self.session_seq, files: sources.into_iter().map(PendingFile::new).collect(), selected: 0, merge: import::MergeMode::Auto, filter: String::new() };
        let idx: Vec<usize> = (0..session.files.len()).collect();
        self.spawn_previews(&mut session, &idx, ctx);
        self.auto_import = self.skip_dialog;
        self.import = Some(session);
    }

    fn spawn_previews(&self, s: &mut ImportSession, idx: &[usize], ctx: &egui::Context) {
        let jobs: Vec<(usize, u64, Arc<Source>, import::ImportOptions)> = idx.iter().map(|&i| (i, s.files[i].version, s.files[i].source.clone(), s.files[i].opts.clone())).collect();
        let sid = s.id;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            jobs.into_par_iter().for_each(|(i, ver, src, opts)| {
                let r = import::parse_source(&src, &opts, Some(PREVIEW_ROWS));
                let _ = tx.send(Msg::Preview(sid, i, ver, Box::new(r)));
                ctx.request_repaint();
            });
        });
    }

    /// Step 3: full parse of the included files with their options, then combine into subjects.
    fn run_import(&mut self, ctx: &egui::Context) {
        let Some(s) = self.import.take() else { return };
        self.load_gen += 1;
        let gen = self.load_gen;
        self.loading = true;
        self.status = "Importing…".into();
        let jobs: Vec<(Arc<Source>, import::ImportOptions)> = s.files.iter().filter(|f| f.include && f.ok()).map(|f| (f.source.clone(), f.opts.clone())).collect();
        let mode = s.merge;
        self.last_import = Some(s);
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            let parsed: Vec<Result<Parsed, String>> = jobs.par_iter().map(|(src, o)| import::parse_source(src, o, None)).collect();
            let mut errs = Vec::new();
            let ok: Vec<Parsed> = parsed.into_iter().filter_map(|r| r.map_err(|e| errs.push(e)).ok()).collect();
            let res = if ok.is_empty() {
                Err(errs.join("\n"))
            } else {
                let mut r = import::combine(ok, mode);
                r.notes.extend(errs.into_iter().map(|e| format!("skipped: {e}")));
                if r.subjects.is_empty() {
                    Err("Nothing to import.".into())
                } else {
                    Ok(r)
                }
            };
            let _ = tx.send(Msg::Imported(gen, res, t.elapsed().as_secs_f64() * 1000.0));
            ctx.request_repaint();
        });
    }

    fn reopen_import(&mut self, ctx: &egui::Context) {
        if let Some(mut s) = self.last_import.take() {
            self.session_seq += 1;
            s.id = self.session_seq;
            for f in &mut s.files {
                f.version += 1;
                f.parsed = None;
                import_ui::sync_text(f);
            }
            let idx: Vec<usize> = (0..s.files.len()).collect();
            self.spawn_previews(&mut s, &idx, ctx);
            self.auto_import = false;
            self.import = Some(s);
        }
    }

    fn set_subjects(&mut self, subjects: Vec<RawDataset>, ctx: &egui::Context) {
        self.subjects = subjects.into_iter().map(Arc::new).collect();
        self.subject_idx = 0;
        self.subject_filter.clear();
        self.batch = None;
        self.select_subject(0, ctx);
    }

    fn select_subject(&mut self, i: usize, ctx: &egui::Context) {
        self.subject_idx = i.min(self.subjects.len().saturating_sub(1));
        self.dataset = self.subjects.get(self.subject_idx).cloned();
        self.computed = None;
        self.requested_cfg = Some(self.cfg.clone());
        self.pending_since = None;
        self.start_analysis(ctx);
    }

    fn load_demo(&mut self, ctx: &egui::Context) {
        self.load_gen += 1;
        let gen = self.load_gen;
        self.loading = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let d = data::synthetic(14, 5.0, Some(8), 3.5);
            let _ = tx.send(Msg::Demo(gen, d));
            ctx.request_repaint();
        });
    }

    fn run_batch(&mut self, ctx: &egui::Context) {
        self.batch_gen += 1;
        let gen = self.batch_gen;
        self.batch_running = true;
        let subjects = self.subjects.clone();
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            let rows = subjects
                .par_iter()
                .map(|d| {
                    let t1 = Instant::now();
                    let r = run_analysis(d, &cfg).map(|rep| crate::cli::batch_row(d, &rep, t1.elapsed().as_secs_f64() * 1000.0));
                    (d.name.clone(), r)
                })
                .collect();
            let head = crate::cli::BATCH_HEAD.split(',').map(|s| s.to_string()).collect();
            let _ = tx.send(Msg::Batch(gen, BatchTable { head, rows, ms: t.elapsed().as_secs_f64() * 1000.0 }));
            ctx.request_repaint();
        });
    }

    fn export_batch(&mut self) {
        let Some(b) = &self.batch else { return };
        if let Some(p) = rfd::FileDialog::new().add_filter("CSV", &["csv"]).set_file_name("subjects_summary.csv").save_file() {
            match crate::cli::write_batch_csv(&p, &b.rows) {
                Ok(()) => self.status = format!("Saved {}", p.display()),
                Err(e) => self.error = Some(format!("Export failed: {e}")),
            }
        }
    }

    fn start_analysis(&mut self, ctx: &egui::Context) {
        let Some(ds) = self.dataset.clone() else { return };
        self.run_gen += 1;
        let gen = self.run_gen;
        self.running = true;
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let res = run_analysis(&ds, &cfg).map(|report| {
                let light_overlay = report.light.as_ref().map(|_| report.binned.log_lux.clone());
                let extras = report
                    .channels()
                    .par_iter()
                    .map(|ch| charts::build_extras(&report.binned, ch, if ch.name != "Light" { light_overlay.as_deref() } else { None }))
                    .collect();
                Computed { report, extras }
            });
            let _ = tx.send(Msg::Analysed(gen, res));
            ctx.request_repaint();
        });
    }

    fn poll_messages(&mut self, ctx: &egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Sources(gen, sources, errors) if gen == self.load_gen => {
                    self.loading = false;
                    if !errors.is_empty() {
                        self.error = Some(errors.join("\n"));
                    }
                    if sources.is_empty() {
                        self.status = "Nothing could be read.".into();
                    } else {
                        self.status = format!("{} file(s) ready to import", sources.len());
                        self.open_session(sources, ctx);
                    }
                }
                Msg::Preview(sid, i, ver, res) => {
                    if let Some(s) = self.import.as_mut().filter(|s| s.id == sid) {
                        if let Some(f) = s.files.get_mut(i).filter(|f| f.version == ver) {
                            f.parsed = Some(*res);
                        }
                        // Direct import once every preview is in and all are recognised
                        if self.auto_import && s.files.iter().all(|f| f.parsed.is_some()) {
                            self.auto_import = false;
                            if s.files.iter().all(|f| f.confident() || !f.include) && s.files.iter().any(|f| f.ok()) {
                                self.run_import(ctx);
                            }
                        }
                    }
                }
                Msg::Imported(gen, res, ms) if gen == self.load_gen => {
                    self.loading = false;
                    match res {
                        Ok(r) => {
                            let n = r.subjects.len();
                            let rows: usize = r.subjects.iter().map(|d| d.len()).sum();
                            self.status = format!("Imported {} ({} rows, {} subject{}) in {ms:.0} ms", r.formats.join(" + "), import::group_thousands(rows), n, if n == 1 { "" } else { "s" });
                            self.import_notes = r.notes;
                            self.error = None;
                            self.set_subjects(r.subjects, ctx);
                        }
                        Err(e) => {
                            self.error = Some(format!("Import failed: {e}"));
                            self.status = "Import failed".into();
                        }
                    }
                }
                Msg::Demo(gen, d) if gen == self.load_gen => {
                    self.loading = false;
                    self.import_notes.clear();
                    self.last_import = None;
                    self.set_subjects(vec![d], ctx);
                }
                Msg::Batch(gen, b) if gen == self.batch_gen => {
                    self.batch_running = false;
                    let ok = b.rows.iter().filter(|r| r.1.is_ok()).count();
                    self.status = format!("Analysed {ok} of {} subjects in {:.0} ms", b.rows.len(), b.ms);
                    self.batch = Some(b);
                    self.tab = Tab::Subjects;
                }
                Msg::Analysed(gen, res) if gen == self.run_gen => {
                    self.running = false;
                    match res {
                        Ok(c) => {
                            self.error = None;
                            self.status = format!(
                                "{} · {} bins × {} · {:.2} days · analysed in {:.0} ms",
                                c.report.source,
                                c.report.binned.len(),
                                format_bin_size(c.report.binned.epoch_sec),
                                c.report.binned.n_days,
                                c.report.elapsed_ms
                            );
                            self.computed = Some(Arc::new(c));
                        }
                        Err(e) => self.error = Some(e),
                    }
                }
                _ => {}
            }
        }
    }

    /// Debounced re-analysis whenever a setting changes.
    fn schedule_analysis(&mut self, ctx: &egui::Context) {
        if self.dataset.is_none() || self.loading {
            return;
        }
        if self.requested_cfg.as_ref() != Some(&self.cfg) {
            if self.pending_since.is_none() {
                self.pending_since = Some(Instant::now());
            }
            let waited = self.pending_since.unwrap().elapsed();
            if waited >= Duration::from_millis(250) {
                self.pending_since = None;
                self.requested_cfg = Some(self.cfg.clone());
                self.start_analysis(ctx);
            } else {
                ctx.request_repaint_after(Duration::from_millis(250) - waited);
            }
        }
    }

    fn ensure_label_fonts(&mut self, ctx: &egui::Context) {
        // Fonts registered last frame are usable now
        self.loaded_fonts.extend(self.fonts_next_frame.drain(..));
        let Some(font) = self.theme_settings.label.font.clone() else { return };
        let Some(&(_, reg, bold)) = theme::FONT_FILES.iter().find(|f| f.0 == font) else { return };
        let mut changed = false;
        for (key, file) in [(family_key(&font, false), reg), (family_key(&font, true), bold)] {
            if self.loaded_fonts.contains(&key) || self.fonts_next_frame.contains(&key) || self.font_defs.font_data.contains_key(&key) {
                continue;
            }
            if let Ok(bytes) = std::fs::read(theme::fonts_dir().join(file)) {
                self.font_defs.font_data.insert(key.clone(), Arc::new(FontData::from_owned(bytes)));
                let mut fallback = vec![key.clone()];
                fallback.extend(self.font_defs.families.get(&FontFamily::Proportional).cloned().unwrap_or_default());
                self.font_defs.families.insert(FontFamily::Name(key.clone().into()), fallback);
                self.fonts_next_frame.push(key);
                changed = true;
            }
        }
        if changed {
            ctx.set_fonts(self.font_defs.clone());
            ctx.request_repaint();
        }
    }

    fn handle_screenshot(&mut self, ctx: &egui::Context) {
        let Some(path) = self.pending_png.clone() else { return };
        let shot = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        let Some(img) = shot else { return };
        self.pending_png = None;
        let ppp = ctx.pixels_per_point();
        let region = if self.auto_shot.is_some() { ctx.screen_rect() } else { self.chart_rect.unwrap_or(ctx.screen_rect()) };
        let crop = img.region(&region, Some(ppp));
        match write_png(&path, &crop) {
            Ok(()) => self.status = format!("Saved chart image to {}", path.display()),
            Err(e) => self.error = Some(format!("PNG export failed: {e}")),
        }
    }

    // --------------------------------------------------------------------------------------------
    // Panels
    // --------------------------------------------------------------------------------------------

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(RichText::new("CoSirca").strong());
                ui.label(RichText::new("circadian rhythmometry").weak());
                ui.separator();
                if ui
                    .button("📂 Open data…")
                    .on_hover_text(
                        "Ctrl+O, or drop files / folders' contents on the window.\n\
                         Actigraphy: ActiGraph, Actiware, ActTrust, Actiwatch .awd, GENEActiv, Axivity\n\
                         Activity monitors: TriKinetics DAM, wheel / beam-break counts\n\
                         Loggers: HOBO, iButton · Heart: Polar, RR/IBI, Empatica E4\n\
                         Phones & wearables: Phyphox, Sensor Logger, Apple Health, Fitbit\n\
                         Tables: CSV, TSV, TXT, XLSX/XLS/ODS, JSON, plate reader, ZT/CT gene expression\n\
                         Archives: .zip, .gz",
                    )
                    .clicked()
                {
                    self.open_dialog(ctx);
                }
                if ui.button("Demo data").on_hover_text("14 days of synthetic activity + light with a 3.5 h phase delay from day 8").clicked() {
                    self.load_demo(ctx);
                }
                ui.separator();
                let has = self.computed.is_some();
                if ui.add_enabled(has, egui::Button::new("📊 Export Excel…")).on_hover_text("Ctrl+E").clicked() {
                    self.export_excel();
                }
                if ui.add_enabled(has, egui::Button::new("Export epochs CSV…")).clicked() {
                    self.export_csv();
                }
                if ui.add_enabled(has && self.tab != Tab::Overview && self.tab != Tab::Daily, egui::Button::new("🖼 Save chart PNG…")).clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("PNG image", &["png"]).set_file_name("chart.png").save_file() {
                        self.pending_png = Some(p);
                        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                    }
                }
            });
        });
    }

    fn open_dialog(&mut self, ctx: &egui::Context) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("Chronobiology data", &["csv", "tsv", "txt", "dat", "awd", "xlsx", "xls", "xlsm", "ods", "json", "jsonl", "xml", "zip", "gz"])
            .add_filter("All files", &["*"])
            .pick_files()
        {
            self.load_files(files, ctx);
        }
    }

    fn export_excel(&mut self) {
        let Some(c) = self.computed.clone() else { return };
        if let Some(p) = rfd::FileDialog::new().add_filter("Excel workbook", &["xlsx"]).set_file_name("circadian_report.xlsx").save_file() {
            let theme = self.theme_settings.build();
            match crate::export::export_excel(&c.report, &theme, &p) {
                Ok(()) => self.status = format!("Saved {}", p.display()),
                Err(e) => self.error = Some(format!("Excel export failed: {e}")),
            }
        }
    }

    fn export_csv(&mut self) {
        let Some(c) = self.computed.clone() else { return };
        if let Some(p) = rfd::FileDialog::new().add_filter("CSV", &["csv"]).set_file_name("circadian_epochs.csv").save_file() {
            match crate::export::export_csv(&c.report, &p) {
                Ok(()) => self.status = format!("Saved {} ({} rows)", p.display(), c.report.binned.len()),
                Err(e) => self.error = Some(format!("CSV export failed: {e}")),
            }
        }
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if self.loading || self.running || self.pending_since.is_some() {
                    ui.spinner();
                    ui.label(if self.loading { "Loading…" } else { "Analysing…" });
                    ui.separator();
                }
                ui.label(RichText::new(&self.status).small());
            });
        });
    }

    fn side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("settings").resizable(true).default_width(330.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.data_section(ui);
                self.analysis_section(ui);
                self.bands_section(ui);
                self.colour_section(ui);
            });
        });
    }

    fn data_section(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::CollapsingHeader::new(RichText::new("1. Data").strong()).default_open(true).show(ui, |ui| {
            let Some(d) = self.dataset.clone() else {
                ui.label("No data loaded.");
                if ui.button("📂 Open data…").clicked() {
                    self.open_dialog(&ctx);
                }
                return;
            };
            // Subject picker for multi-subject imports (DAM monitors, plate readers, gene tables, archives)
            if self.subjects.len() > 1 {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Subject {} of {}", self.subject_idx + 1, self.subjects.len())).strong());
                    if ui.small_button("◀").clicked() && self.subject_idx > 0 {
                        self.select_subject(self.subject_idx - 1, &ctx);
                    }
                    if ui.small_button("▶").clicked() && self.subject_idx + 1 < self.subjects.len() {
                        self.select_subject(self.subject_idx + 1, &ctx);
                    }
                });
                let cur = self.subjects[self.subject_idx].name.clone();
                let mut pick = None;
                egui::ComboBox::from_id_salt("subject").selected_text(cur).width(290.0).height(360.0).show_ui(ui, |ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.subject_filter).hint_text("search…"));
                    let f = self.subject_filter.to_lowercase();
                    let mut shown = 0;
                    for (i, s) in self.subjects.iter().enumerate() {
                        if !f.is_empty() && !s.name.to_lowercase().contains(&f) {
                            continue;
                        }
                        if ui.selectable_label(i == self.subject_idx, &s.name).clicked() {
                            pick = Some(i);
                        }
                        shown += 1;
                        if shown >= 500 {
                            ui.label(RichText::new("… refine the search to see more").weak());
                            break;
                        }
                    }
                });
                if let Some(i) = pick {
                    self.select_subject(i, &ctx);
                }
                ui.horizontal(|ui| {
                    let label = if self.batch_running { "Analysing all…".to_string() } else { format!("Analyse all {} subjects", self.subjects.len()) };
                    if ui.add_enabled(!self.batch_running, egui::Button::new(label)).on_hover_text("Runs the current settings on every subject in parallel; results appear in the \"All subjects\" tab").clicked() {
                        self.run_batch(&ctx);
                    }
                    if self.batch_running {
                        ui.spinner();
                    }
                });
                ui.separator();
            }
            ui.label(RichText::new(&d.name).strong());
            if !d.format.is_empty() {
                ui.label(RichText::new(&d.format).color(Color32::from_rgb(90, 170, 250)));
            }
            if let Some((t0, t1)) = d.time_range() {
                ui.label(format!("{} samples · {}", import::group_thousands(d.len()), import::format_duration((t1 - t0) as f64 / 1000.0)));
                ui.label(format!("{} → {}", data::ms_to_naive(t0).format("%Y-%m-%d %H:%M"), data::ms_to_naive(t1).format("%Y-%m-%d %H:%M")));
            }
            ui.label(format!("Channels: {}", d.describe_channels().join(", ")));
            let notes: Vec<String> = self.import_notes.iter().chain(d.notes.iter()).cloned().collect();
            if !notes.is_empty() {
                egui::CollapsingHeader::new(format!("Import notes ({})", notes.len())).default_open(false).show(ui, |ui| {
                    for n in notes {
                        ui.label(RichText::new(format!("• {n}")).small());
                    }
                });
            }
            ui.horizontal(|ui| {
                if ui.button("📂 Open…").clicked() {
                    self.open_dialog(&ctx);
                }
                if ui.add_enabled(self.last_import.is_some(), egui::Button::new("⚙ Import settings…")).on_hover_text("Reopen the import dialog to change time / column / layout choices").clicked() {
                    self.reopen_import(&ctx);
                }
            });
        });
    }

    fn analysis_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(RichText::new("2. Analysis").strong()).default_open(true).show(ui, |ui| {
            ui.label("Bin size");
            let r = ui.add(egui::Slider::new(&mut self.bin_idx, 0..=BIN_SIZES.len() - 1).show_value(false));
            self.cfg.epoch_sec = BIN_SIZES[self.bin_idx];
            let per_day = 86_400 / self.cfg.epoch_sec;
            let mut info = format!("{} ({} bins/day)", format_bin_size(self.cfg.epoch_sec), per_day);
            if let Some(d) = &self.dataset {
                if let (Some(a), Some(b)) = (d.t_ms.first(), d.t_ms.last()) {
                    let n = (b - a) / 1000 / self.cfg.epoch_sec + 1;
                    info += &format!(" · ≈{n} bins");
                    if n < 6 {
                        ui.colored_label(Color32::from_rgb(220, 120, 0), "Too few bins for analysis (need ≥ 6); pick a smaller bin.");
                    }
                }
            }
            r.on_hover_text("1 s to 6 h. Sub-minute bins are averaged to 1 min for the period scans.");
            ui.label(RichText::new(info).small());
            if self.dataset.as_ref().is_some_and(|d| d.has_accel()) {
                egui::ComboBox::from_label("Activity metric").selected_text(self.cfg.activity_method.label()).show_ui(ui, |ui| {
                    for m in ActivityMethod::ALL {
                        ui.selectable_value(&mut self.cfg.activity_method, m, m.label());
                    }
                });
            }
            ui.checkbox(&mut self.cfg.use_log_lux, "Analyse light as log10(lux + 1)");
            ui.add(egui::Slider::new(&mut self.cfg.n_harmonics, 1..=6).text("cosinor harmonics"));
            egui::ComboBox::from_label("Period τ").selected_text(self.cfg.period_mode.label()).show_ui(ui, |ui| {
                for m in PeriodMode::ALL {
                    ui.selectable_value(&mut self.cfg.period_mode, m, m.label());
                }
            });
            if self.cfg.period_mode == PeriodMode::Custom {
                ui.add(egui::DragValue::new(&mut self.cfg.custom_tau).range(4.0..=48.0).speed(0.05).suffix(" h").prefix("τ = "));
            }
            egui::ComboBox::from_label("Onsets").selected_text(self.cfg.onset_method.label()).show_ui(ui, |ui| {
                for m in OnsetMethod::ALL {
                    ui.selectable_value(&mut self.cfg.onset_method, m, m.label());
                }
            });
            ui.horizontal(|ui| {
                let mut manual = self.cfg.split_day.is_some();
                if ui.checkbox(&mut manual, "Split day for pre/post test").changed() {
                    self.cfg.split_day = if manual { Some(2) } else { None };
                }
                if let Some(s) = self.cfg.split_day.as_mut() {
                    ui.add(egui::DragValue::new(s).range(2..=3650));
                }
            });
            ui.add(egui::Slider::new(&mut self.cfg.interp_gap_min, 0.0..=240.0).step_by(5.0).text("min gap fill"));
        });
    }

    fn bands_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(RichText::new("Rhythm bands & bimodality").strong()).default_open(false).show(ui, |ui| {
            let c = &mut self.cfg;
            ui.horizontal(|ui| {
                ui.label("Scan");
                ui.add(egui::DragValue::new(&mut c.period_min).range(2.0..=47.0).speed(0.1).suffix(" h"));
                ui.label("to");
                ui.add(egui::DragValue::new(&mut c.period_max).range(3.0..=96.0).speed(0.1).suffix(" h"));
            });
            if c.period_max <= c.period_min + 1.0 {
                c.period_max = c.period_min + 1.0;
            }
            ui.horizontal(|ui| {
                ui.label("Circadian");
                ui.add(egui::DragValue::new(&mut c.circ_min).range(10.0..=30.0).speed(0.1).suffix(" h"));
                ui.label("to");
                ui.add(egui::DragValue::new(&mut c.circ_max).range(14.0..=40.0).speed(0.1).suffix(" h"));
            });
            if c.circ_max <= c.circ_min {
                c.circ_max = c.circ_min + 0.5;
            }
            ui.label(RichText::new(format!("Ultradian {}–{} h · Circadian {}–{} h · Infradian {}–{} h", c.period_min, c.circ_min, c.circ_min, c.circ_max, c.circ_max, c.period_max)).small());
            ui.add(egui::Slider::new(&mut c.alpha, 0.001..=0.1).logarithmic(true).text("α"));
            ui.add(egui::Slider::new(&mut c.min_pct_variance, 0.0..=20.0).step_by(0.5).text("min % variance"));
            ui.add(egui::Slider::new(&mut c.bimodality_min_prominence, 0.02..=0.5).step_by(0.01).text("2nd-peak prominence"));
        });
    }

    fn colour_section(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new(RichText::new("3. Chart colours").strong()).default_open(false).show(ui, |ui| {
            let ts = &mut self.theme_settings;
            egui::ComboBox::from_label("Background").selected_text(&ts.background_preset).show_ui(ui, |ui| {
                for (name, _) in BACKGROUNDS {
                    ui.selectable_value(&mut ts.background_preset, name.to_string(), name);
                }
                ui.selectable_value(&mut ts.background_preset, "Custom".to_string(), "Custom");
            });
            if ts.background_preset == "Custom" {
                ui.horizontal(|ui| {
                    ui.color_edit_button_srgba(&mut ts.custom_background);
                    ui.label("custom background");
                });
            }
            egui::ComboBox::from_label("Palette").selected_text(&ts.palette).show_ui(ui, |ui| {
                for (name, _) in PALETTES {
                    ui.selectable_value(&mut ts.palette, name.to_string(), name);
                }
            });
            egui::CollapsingHeader::new("Individual plot colours").show(ui, |ui| {
                let base = ThemeSettings { overrides: Default::default(), ..ts.clone() }.build();
                for (role, label) in ROLES {
                    ui.horizontal(|ui| {
                        let mut c = ts.overrides.get(role).copied().unwrap_or(base.role(role));
                        if ui.color_edit_button_srgba(&mut c).changed() {
                            ts.overrides.insert(role.to_string(), c);
                        }
                        ui.label(label);
                        if ts.overrides.contains_key(role) && ui.small_button("↺").on_hover_text("Back to palette colour").clicked() {
                            ts.overrides.remove(role);
                        }
                    });
                }
                if !ts.overrides.is_empty() && ui.button("Reset all to palette").clicked() {
                    ts.overrides.clear();
                }
            });
            egui::CollapsingHeader::new("Axis labels: text, font & colour").show(ui, |ui| {
                let bg_now = ts.background();
                let l = &mut ts.label;
                egui::ComboBox::from_label("Font").selected_text(l.font.clone().unwrap_or_else(|| "Default".into())).show_ui(ui, |ui| {
                    ui.selectable_value(&mut l.font, None, "Default");
                    for f in &self.available_fonts {
                        ui.selectable_value(&mut l.font, Some(f.to_string()), *f);
                    }
                });
                ui.add(egui::Slider::new(&mut l.size, 0.0..=28.0).step_by(1.0).text("size (0 = default)"));
                ui.checkbox(&mut l.bold, "Bold");
                if ui.checkbox(&mut self.custom_label_colors, "Custom label colours").changed() && !self.custom_label_colors {
                    l.x_color = None;
                    l.y_color = None;
                }
                if self.custom_label_colors {
                    let fg = if theme::contrast(bg_now, Color32::BLACK) > 7.0 { Color32::from_rgb(15, 23, 42) } else { Color32::from_rgb(226, 232, 240) };
                    ui.horizontal(|ui| {
                        let mut x = l.x_color.unwrap_or(fg);
                        if ui.color_edit_button_srgba(&mut x).changed() {
                            l.x_color = Some(x);
                        }
                        ui.label("X label");
                        let mut y = l.y_color.unwrap_or(fg);
                        if ui.color_edit_button_srgba(&mut y).changed() {
                            l.y_color = Some(y);
                        }
                        ui.label("Y label");
                    });
                }
                ui.separator();
                ui.label("Rename labels");
                let mut remove = None;
                for (i, (orig, new)) in l.renames.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt(("rename", i)).width(150.0).selected_text(if orig.is_empty() { "pick label…" } else { orig.as_str() }).show_ui(ui, |ui| {
                            for s in &self.seen_labels {
                                ui.selectable_value(orig, s.clone(), s);
                            }
                        });
                        ui.add(egui::TextEdit::singleline(new).desired_width(110.0).hint_text("new text"));
                        if ui.small_button("✖").clicked() {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    l.renames.remove(i);
                }
                if ui.button("➕ Add rename").clicked() {
                    l.renames.push((String::new(), String::new()));
                }
            });
        });
    }

    // --------------------------------------------------------------------------------------------
    // Central content
    // --------------------------------------------------------------------------------------------

    fn central(&mut self, ctx: &egui::Context, theme: &Theme) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(e) = self.error.clone() {
                egui::Frame::default().fill(Color32::from_rgba_unmultiplied(220, 50, 50, 28)).corner_radius(6.0).inner_margin(8.0).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.colored_label(Color32::from_rgb(235, 90, 90), RichText::new(e).strong());
                        if ui.small_button("✖").clicked() {
                            self.error = None;
                        }
                    });
                });
            }
            if self.tab == Tab::Subjects && self.batch.is_some() {
                ui.horizontal_wrapped(|ui| {
                    for (t, name) in TABS {
                        ui.selectable_value(&mut self.tab, t, name);
                    }
                });
                ui.separator();
                self.batch_view(ui);
                return;
            }
            let Some(comp) = self.computed.clone() else {
                if welcome(ui, self.loading || self.running) && self.import.is_none() && !self.loading {
                    let ctx = ui.ctx().clone();
                    self.open_dialog(&ctx);
                }
                return;
            };
            ui.horizontal_wrapped(|ui| {
                for (t, name) in TABS {
                    if t == Tab::Subjects && self.batch.is_none() {
                        continue;
                    }
                    ui.selectable_value(&mut self.tab, t, name);
                }
            });
            let chans = comp.report.channels();
            if chans.len() > 1 && self.tab != Tab::Overview {
                ui.horizontal(|ui| {
                    ui.label("Channel:");
                    for (i, ch) in chans.iter().enumerate() {
                        ui.selectable_value(&mut self.channel, i, ch.name.as_str());
                    }
                });
            }
            self.channel = self.channel.min(chans.len() - 1);
            ui.separator();
            let cx = ChartCtx::new(theme, &self.loaded_fonts);
            let ch = chans[self.channel];
            let ex = &comp.extras[self.channel];
            let rep = &comp.report;
            egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| match self.tab {
                Tab::Overview => overview(ui, rep),
                Tab::Actogram => {
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.acto.double, "Double-plotted");
                        if rep.light.is_some() && ch.name != "Light" {
                            ui.checkbox(&mut self.acto.light, "Light overlay");
                        }
                        ui.checkbox(&mut self.acto.markers, "Onset / offset markers");
                    });
                    charts::actogram(ui, &cx, rep, ch, &ex.acto, &self.acto);
                }
                Tab::TimeSeries => {
                    ui.checkbox(&mut self.show_fits, "Show cosinor fits");
                    for c in &chans {
                        charts::time_series(ui, &cx, rep, c, self.show_fits, 260.0, Some("ts"));
                    }
                }
                Tab::Waveform => {
                    cosinor_table(ui, ch);
                    charts::waveform(ui, &cx, ch, ex);
                }
                Tab::Periodogram => {
                    let pg = &ch.periodogram;
                    ui.label(format!(
                        "Lomb-Scargle / cosinor τ = {:.2} h (power {:.3}) · χ² τ = {:.2} h (Qp {:.1}, p = {}) · consensus τ = {:.2} h · used τ = {:.2} h",
                        pg.best_tau_ls, pg.best_ls, pg.best_tau_chi2, pg.best_qp, fmt_p(pg.best_chi2_p), pg.consensus_tau, ch.tau
                    ));
                    charts::periodogram(ui, &cx, ch);
                }
                Tab::Bands => {
                    bands_tables(ui, ch);
                    charts::bands_spectrum(ui, &cx, ch);
                    ui.add_space(6.0);
                    charts::bimodality_wave(ui, &cx, ch);
                }
                Tab::Circular => {
                    circular_tables(ui, ch);
                    charts::polar(ui, &cx, ch);
                }
                Tab::Shift => {
                    shift_summary(ui, ch);
                    charts::shift_timeline(ui, &cx, ch);
                }
                Tab::Profile => {
                    if let Some(np) = &ch.non_param {
                        ui.label(format!(
                            "IS {:.3} · IV {:.3} · RA {:.3} · M10 {:.3} from {} · L5 {:.3} from {}",
                            np.is,
                            np.iv,
                            np.ra,
                            np.m10,
                            hhmm(np.m10_onset),
                            np.l5,
                            hhmm(np.l5_onset)
                        ));
                    }
                    charts::hourly_profile(ui, &cx, ch);
                }
                Tab::Daily => daily_table(ui, ch),
                Tab::Subjects => {}
            });
            self.seen_labels.extend(cx.seen_labels.into_inner());
            self.chart_rect = cx.chart_rect.into_inner();
        });
    }
}

const FORMAT_GROUPS: [(&str, &str); 7] = [
    ("Actigraphy", "ActiGraph (ActiLife CSV), Philips Actiware, Condor ActTrust, Actiwatch / CamNtech .awd, GENEActiv, Axivity, MotionWare"),
    ("Animal activity", "TriKinetics DAM monitors (one fly per channel), running-wheel and beam-break counts"),
    ("Light & temperature", "HOBO loggers, iButton / Thermochron, any lux / °C column"),
    ("Heart", "RR / IBI / NN intervals (Holter, chest straps), Polar exports, Empatica E4"),
    ("Phones & wearables", "Phyphox, Sensor Logger, Apple Health export.xml, Fitbit JSON"),
    ("Lab tables", "Bioluminescence / plate-reader wells, ZT / CT gene-expression and hormone time courses"),
    ("Files", "CSV · TSV · TXT · XLSX · XLS · ODS · JSON · XML · ZIP · GZ — UTF-8, UTF-16 or Windows-1252"),
];

/// Welcome screen with a drop zone; returns true when the drop zone was clicked.
fn welcome(ui: &mut egui::Ui, busy: bool) -> bool {
    let mut clicked = false;
    ui.add_space(24.0);
    ui.vertical_centered(|ui| {
        ui.heading(RichText::new("CoSirca").size(28.0).strong());
        ui.label(RichText::new("Circadian rhythmometry for actigraphy, light, temperature, heart rate and molecular time courses").weak());
        ui.add_space(16.0);
        let w = ui.available_width().min(760.0);
        let (rect, resp) = ui.allocate_exact_size(egui::vec2(w, 150.0), egui::Sense::click());
        clicked = resp.clicked();
        let hovered = resp.hovered();
        let stroke_col = if hovered { Color32::from_rgb(90, 170, 250) } else { ui.visuals().widgets.noninteractive.bg_stroke.color };
        ui.painter().rect_filled(rect, 12.0, if hovered { Color32::from_rgba_unmultiplied(90, 170, 250, 18) } else { ui.visuals().extreme_bg_color });
        // dashed border
        let pts = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom(), rect.left_top()];
        for w2 in pts.windows(2) {
            ui.painter().extend(egui::Shape::dashed_line(&[w2[0], w2[1]], egui::Stroke::new(1.5, stroke_col), 8.0, 6.0));
        }
        let c = rect.center();
        ui.painter().text(c - egui::vec2(0.0, 22.0), egui::Align2::CENTER_CENTER, "⬇", egui::FontId::proportional(30.0), stroke_col);
        ui.painter().text(c + egui::vec2(0.0, 14.0), egui::Align2::CENTER_CENTER, if busy { "Reading…" } else { "Drop files here, or click to choose (Ctrl+O)" }, egui::FontId::proportional(17.0), ui.visuals().strong_text_color());
        ui.painter().text(c + egui::vec2(0.0, 40.0), egui::Align2::CENTER_CENTER, "Several files at once: sensors of one recording are merged; different people / animals stay separate", egui::FontId::proportional(12.0), ui.visuals().weak_text_color());
        if busy {
            ui.spinner();
        }
        ui.add_space(18.0);
        egui::Grid::new("formats").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
            for (k, v) in FORMAT_GROUPS {
                ui.label(RichText::new(k).strong());
                ui.label(RichText::new(v).small());
                ui.end_row();
            }
        });
    });
    clicked
}

impl CosircaApp {
    fn batch_view(&mut self, ui: &mut egui::Ui) {
        let Some(b) = &self.batch else { return };
        let ok = b.rows.iter().filter(|r| r.1.is_ok()).count();
        let mut export = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{ok} of {} subjects analysed in {:.0} ms with the current settings", b.rows.len(), b.ms)).strong());
            if ui.button("Export summary CSV…").clicked() {
                export = true;
            }
        });
        // Quick aggregate of the key rhythm metrics
        let col = |name: &str| b.head.iter().position(|h| h == name).map(|i| i - 1);
        let vals = |name: &str| -> Vec<f64> { col(name).map(|c| b.rows.iter().filter_map(|r| r.1.as_ref().ok()?.get(c)?.parse::<f64>().ok()).collect()).unwrap_or_default() };
        let summ = |v: &[f64]| if v.is_empty() { "—".to_string() } else { format!("median {:.3} (IQR {:.3}–{:.3})", crate::stats::median(v), crate::stats::percentile(v, 25.0), crate::stats::percentile(v, 75.0)) };
        let (amp, r2, p) = (vals("amplitude"), vals("r2"), vals("p"));
        let sig = p.iter().filter(|&&x| x < 0.05).count();
        ui.label(format!("Amplitude {} · R² {} · significant rhythm (p < 0.05) in {sig} of {}", summ(&amp), summ(&r2), p.len()));
        ui.separator();
        egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            egui::Grid::new("batch").striped(true).spacing([12.0, 3.0]).show(ui, |ui| {
                for h in &b.head {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for (name, r) in &b.rows {
                    ui.label(name);
                    match r {
                        Ok(row) => {
                            for c in row {
                                ui.label(c);
                            }
                        }
                        Err(e) => {
                            ui.colored_label(Color32::from_rgb(220, 90, 90), e.lines().next().unwrap_or(""));
                        }
                    }
                    ui.end_row();
                }
            });
        });
        if export {
            self.export_batch();
        }
    }
}

fn write_png(path: &std::path::Path, img: &egui::ColorImage) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), img.size[0] as u32, img.size[1] as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut w = enc.write_header().map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
    w.write_image_data(&bytes).map_err(|e| e.to_string())
}

fn fmt_p(p: f64) -> String {
    if !p.is_finite() {
        "—".into()
    } else if p < 1e-300 {
        "< 1e-300".into()
    } else if p < 0.001 {
        format!("{p:.1e}")
    } else {
        format!("{p:.3}")
    }
}

fn fmt_num(v: f64, d: usize) -> String {
    if v.is_finite() {
        format!("{v:.d$}")
    } else {
        "—".into()
    }
}

fn kv_grid(ui: &mut egui::Ui, id: &str, rows: &[(&str, String)]) {
    egui::Grid::new(id).num_columns(2).striped(true).spacing([18.0, 4.0]).show(ui, |ui| {
        for (k, v) in rows {
            ui.label(*k);
            ui.add(egui::Label::new(RichText::new(v).strong()).wrap());
            ui.end_row();
        }
    });
}

fn overview(ui: &mut egui::Ui, rep: &Report) {
    for w in &rep.warnings {
        ui.colored_label(Color32::from_rgb(220, 120, 0), w);
    }
    let b = &rep.binned;
    ui.label(format!(
        "{} · {} bins of {} · {:.2} days from {} · analysis {:.0} ms",
        rep.source,
        b.len(),
        format_bin_size(b.epoch_sec),
        b.n_days,
        data::ms_to_naive(b.start_ms).format("%Y-%m-%d %H:%M"),
        rep.elapsed_ms
    ));
    ui.add_space(4.0);
    // Responsive cards: as many per row as fit at ≥ 470 pt each
    let chans = rep.channels();
    let per_row = ((ui.available_width() / 470.0).floor() as usize).clamp(1, chans.len().max(1));
    for row in chans.chunks(per_row) {
        ui.columns(per_row, |cols| {
            for (ui, ch) in cols.iter_mut().zip(row.iter()) {
                overview_card(ui, ch);
            }
        });
        ui.add_space(6.0);
    }
}

fn overview_card(ui: &mut egui::Ui, ch: &crate::pipeline::ChannelReport) {
    {
        {
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.heading(format!("{} ({})", ch.name, ch.unit));
                let s = &ch.single;
                let h = &ch.harmonics;
                let mut rows = vec![
                    ("Period τ used", format!("{:.2} h", ch.tau)),
                    ("Consensus τ", format!("{:.2} h", ch.periodogram.consensus_tau)),
                    ("MESOR", fmt_num(s.mesor, 3)),
                    ("Amplitude", fmt_num(s.amplitude(), 3)),
                    ("Acrophase", format!("{} ({:.1}°)", s.acrophase_hhmm(), s.deg_neg())),
                    ("Single cosinor R² / p", format!("{:.3} / {}", s.r2, fmt_p(s.p))),
                    ("Harmonic cosinor R²", format!("{:.3} ({} harmonics)", h.r2, h.nh)),
                    ("Orthophase / bathyphase", format!("{} / {}", hhmm(h.ortho_h), hhmm(h.comp_bathy_h))),
                ];
                if let Some(np) = &ch.non_param {
                    rows.push(("IS / IV / RA", format!("{:.3} / {:.3} / {:.3}", np.is, np.iv, np.ra)));
                }
                if let Some(c) = &ch.circ_acrophase {
                    rows.push(("Circular mean acrophase", format!("{} (R̄ {:.3}, Rayleigh p {})", c.mean_hhmm(), c.r_bar, fmt_p(c.rayleigh_p))));
                }
                if let Some(sh) = &ch.shift {
                    if let Some(t) = &sh.acrophase {
                        rows.push(("Acrophase shift", format!("{} → {} ({:+.2} h), p {}", hhmm(t.mean1), hhmm(t.mean2), t.shift_h, fmt_p(t.p))));
                    }
                    rows.push(("Drift / implied τ", format!("{:+.2} h/day / {:.2} h", sh.drift_slope, sh.implied_tau)));
                    if let Some(pa) = sh.act_light_phase_angle.filter(|_| ch.name != "Light") {
                        rows.push(("Activity − light phase angle", format!("{pa:+.2} h")));
                    }
                }
                if let Some(bd) = &ch.bands {
                    rows.push(("Dominant rhythm", bd.dominant_band.map_or("none significant".into(), |d| format!("{d} {:.1} h", bd.dominant_period))));
                }
                if let Some(bm) = &ch.bimodality {
                    rows.push(("Waveform", format!("{} (index {:.2}; bimodal days {}/{})", bm.classification, bm.shape.bimodality_index, bm.days_bimodal, bm.days_tested)));
                }
                kv_grid(ui, &format!("ov-{}", ch.name), &rows);
            });
        }
    }
}

fn cosinor_table(ui: &mut egui::Ui, ch: &crate::pipeline::ChannelReport) {
    let h = &ch.harmonics;
    egui::Grid::new(format!("harm-{}", ch.name)).striped(true).spacing([14.0, 3.0]).show(ui, |ui| {
        for t in ["k", "Period (h)", "Amplitude", "95% CI", "Peak", "Partial R²", "p", "A_k / A_1"] {
            ui.label(RichText::new(t).strong());
        }
        ui.end_row();
        for c in &h.harmonics {
            ui.label(c.k.to_string());
            ui.label(format!("{:.2}", c.period));
            ui.label(format!("{:.3}", c.amp));
            ui.label(format!("{:.3}–{:.3}", c.amp_ci.0, c.amp_ci.1));
            ui.label(hhmm(c.peak_clock));
            ui.label(format!("{:.4}", c.partial_r2));
            ui.label(fmt_p(c.p));
            ui.label(format!("{:.2}", c.ratio_h1));
            ui.end_row();
        }
    });
    ui.label(format!(
        "MESOR {:.3} (95% CI {:.3}–{:.3}) · overall R² {:.3}, adj. R² {:.3}, F {:.1}, p {} · composite amplitude {:.3}",
        h.mesor, h.mesor_ci.0, h.mesor_ci.1, h.r2, h.adj_r2, h.f, fmt_p(h.p), h.comp_amp
    ));
    if let Some(pc) = &ch.pop_cosinor {
        ui.label(format!(
            "Population-mean cosinor over {} days: MESOR {:.3} ± {:.3}, amplitude {:.3}, acrophase {}, F {:.2}, p {}",
            pc.n,
            pc.mesor,
            pc.mesor_se,
            pc.amplitude,
            hhmm(pc.acrophase_h),
            pc.f,
            fmt_p(pc.p)
        ));
    }
}

fn bands_tables(ui: &mut egui::Ui, ch: &crate::pipeline::ChannelReport) {
    if let Some(b) = &ch.bands {
        ui.label(RichText::new(format!("Rhythm bands ({}–{} h scan, α = {}, Šidák-corrected)", b.period_min, b.period_max, b.alpha)).strong());
        egui::Grid::new(format!("bands-{}", ch.name)).striped(true).spacing([12.0, 3.0]).show(ui, |ui| {
            for t in ["Band", "Range (h)", "Classification", "Period", "Amplitude", "Peak", "% var", "p (Šidák)"] {
                ui.label(RichText::new(t).strong());
            }
            ui.end_row();
            for name in BAND_NAMES {
                let x = b.band(name);
                ui.label(x.band);
                ui.label(x.range_label());
                ui.label(&x.classification).on_hover_text(&x.note);
                ui.label(if x.period.is_finite() { format!("{:.1} h", x.period) } else { "—".into() });
                ui.label(fmt_num(x.amplitude, 3));
                ui.label(&x.acrophase_hhmm);
                ui.label(fmt_num(x.percent_variance, 1));
                ui.label(fmt_p(x.p_adjusted));
                ui.end_row();
            }
        });
        for x in b.bands.iter().filter(|x| !x.note.is_empty()) {
            ui.label(RichText::new(format!("{}: {}", x.band, x.note)).small().weak());
        }
    }
    if let Some(t) = &ch.twelve {
        ui.add_space(6.0);
        ui.label(RichText::new("12 h components (shown even when weak)").strong());
        egui::Grid::new(format!("twelve-{}", ch.name)).striped(true).spacing([12.0, 3.0]).show(ui, |ui| {
            for h in ["Component", "Period", "Amplitude", "Peaks", "% var", "p", ""] {
                ui.label(RichText::new(h).strong());
            }
            ui.end_row();
            let sig = |s: bool| if s { "significant" } else { "n.s." };
            ui.label("12 h rhythm (free period, after removing τ)");
            ui.label(format!("{:.2} h", t.rhythm_period));
            ui.label(format!("{:.3}", t.rhythm_amp));
            ui.label(&t.rhythm_peaks);
            ui.label(format!("{:.2}", t.rhythm_pct));
            ui.label(format!("{} (Šidák)", fmt_p(t.rhythm_p_adj)));
            ui.label(sig(t.rhythm_sig));
            ui.end_row();
            ui.label("τ/2 harmonic of the circadian rhythm");
            ui.label(format!("{:.2} h", t.harm_period));
            ui.label(format!("{:.3}", t.harm_amp));
            ui.label(&t.harm_peaks);
            ui.label(format!("{:.2}", t.harm_pct));
            ui.label(fmt_p(t.harm_p));
            ui.label(sig(t.harm_sig));
            ui.end_row();
        });
        ui.label(RichText::new(format!("A2/A1 = {:.2}. {}", t.ratio, t.interpretation)).small());
    }
    if let Some(bm) = &ch.bimodality {
        ui.add_space(6.0);
        ui.label(RichText::new(format!(
            "Bimodality over {:.2} h: {} · index {:.2} (threshold {:.2}) · peaks {} & {} · dip {} · 12 h component p {} · bimodal days {}/{}",
            bm.period,
            bm.classification,
            bm.shape.bimodality_index,
            bm.min_prominence,
            hhmm(bm.shape.primary_h),
            hhmm(bm.shape.secondary_h),
            hhmm(bm.shape.dip_h),
            fmt_p(bm.h2_p),
            bm.days_bimodal,
            bm.days_tested
        )));
    }
}

fn circular_tables(ui: &mut egui::Ui, ch: &crate::pipeline::ChannelReport) {
    egui::Grid::new(format!("circ-{}", ch.name)).striped(true).spacing([14.0, 3.0]).show(ui, |ui| {
        for t in ["Phase marker", "n", "Mean", "R̄", "Circular SD (h)", "κ", "Rayleigh z", "Rayleigh p"] {
            ui.label(RichText::new(t).strong());
        }
        ui.end_row();
        for s in [&ch.circ_acrophase, &ch.circ_onset, &ch.circ_offset].into_iter().flatten() {
            ui.label(&s.label);
            ui.label(s.n.to_string());
            ui.label(s.mean_hhmm());
            ui.label(format!("{:.3}", s.r_bar));
            ui.label(format!("{:.2}", s.circ_std_hours));
            ui.label(format!("{:.2}", s.kappa));
            ui.label(format!("{:.2}", s.rayleigh_z));
            ui.label(fmt_p(s.rayleigh_p));
            ui.end_row();
        }
    });
}

fn shift_summary(ui: &mut egui::Ui, ch: &crate::pipeline::ChannelReport) {
    let Some(s) = &ch.shift else {
        ui.label("Not enough days for shift analysis.");
        return;
    };
    let mut rows = vec![
        ("Split day (pre < day ≤ post)", s.split_day.to_string()),
        ("Mean day-to-day shift", format!("{:+.2} h", s.mean_daily_shift)),
        ("Cumulative shift", format!("{:+.2} h", s.cumulative_shift)),
        ("Drift slope / implied τ", format!("{:+.3} h/day / {:.2} h", s.drift_slope, s.implied_tau)),
    ];
    for (name, t) in [("Acrophase", &s.acrophase), ("Onset", &s.onset), ("Offset", &s.offset)] {
        if let Some(t) = t {
            rows.push((name, format!("{} → {} = {:+.2} h ({}), Watson-Williams F {:.2}, p {}", hhmm(t.mean1), hhmm(t.mean2), t.shift_h, t.direction, t.f, fmt_p(t.p))));
        }
    }
    kv_grid(ui, &format!("shift-{}", ch.name), &rows);
}

fn daily_table(ui: &mut egui::Ui, ch: &crate::pipeline::ChannelReport) {
    egui::ScrollArea::horizontal().show(ui, |ui| {
        egui::Grid::new(format!("daily-{}", ch.name)).striped(true).spacing([12.0, 3.0]).show(ui, |ui| {
            for t in ["Day", "Date", "MESOR", "Amplitude", "Acrophase", "R²", "Onset", "Offset", "Active (h)", "Midpoint", "Bimodal", "Index", "Peak 1", "Peak 2", "Shift (h)", "Cum. shift (h)", "Act−light (h)"] {
                ui.label(RichText::new(t).strong());
            }
            ui.end_row();
            let t = |v: f64| if v.is_finite() { hhmm(v) } else { "—".into() };
            for d in &ch.daily {
                ui.label(d.day.to_string());
                ui.label(&d.date);
                ui.label(fmt_num(d.mesor, 3));
                ui.label(fmt_num(d.amplitude, 3));
                ui.label(t(d.acrophase_h));
                ui.label(fmt_num(d.r2, 3));
                ui.label(t(d.onset_h));
                ui.label(t(d.offset_h));
                ui.label(fmt_num(d.active_h, 2));
                ui.label(t(d.midpoint_h));
                ui.label(match d.bimodal {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "—",
                });
                ui.label(fmt_num(d.bimodality_index, 2));
                ui.label(t(d.peak1_h));
                ui.label(t(d.peak2_h));
                ui.label(fmt_num(d.shift_h, 2));
                ui.label(fmt_num(d.cumulative_shift_h, 2));
                ui.label(fmt_num(d.phase_angle_h, 2));
                ui.end_row();
            }
        });
    });
}

impl eframe::App for CosircaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_messages(ctx);
        self.ensure_label_fonts(ctx);
        self.handle_screenshot(ctx);
        if let Some((path, frames)) = self.auto_shot.clone() {
            let dialog_ready = self.shot_dialog && (self.import.as_ref().is_some_and(|s| s.files.iter().all(|f| f.parsed.is_some())) || (self.load_gen == 0 && self.import.is_none()));
            if self.shot_subject > 0 && self.subjects.len() > self.shot_subject && self.subject_idx != self.shot_subject && !self.running {
                let s = self.shot_subject;
                self.select_subject(s, ctx);
            }
            if self.shot_batch && self.subjects.len() > 1 && self.batch.is_none() && !self.batch_running && self.computed.is_some() {
                self.run_batch(ctx);
            }
            let batch_done = !self.shot_batch || self.batch.is_some();
            let ready = dialog_ready || (!self.shot_dialog && batch_done && self.computed.is_some() && !self.running && !self.loading && self.pending_since.is_none() && self.fonts_next_frame.is_empty());
            if ready {
                if frames == 4 && self.pending_png.is_none() {
                    self.pending_png = Some(path.clone());
                    self.chart_rect = None;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
                } else if frames > 4 && self.pending_png.is_none() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                self.auto_shot = Some((path, frames + 1));
            }
            ctx.request_repaint();
        }
        // Drag & drop: highlight while hovering, import on drop
        let hovering = ctx.input(|i| i.raw.hovered_files.len());
        if hovering > 0 {
            let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop-overlay")));
            let r = ctx.screen_rect();
            p.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(20, 60, 110, 170));
            p.text(r.center(), egui::Align2::CENTER_CENTER, format!("Drop {} file{} to import", hovering, if hovering == 1 { "" } else { "s" }), egui::FontId::proportional(28.0), Color32::WHITE);
        }
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() {
            self.load_files(dropped, ctx);
        }
        // Import dialog
        if let Some(mut s) = self.import.take() {
            let action = if self.auto_import { import_ui::Action::None } else { import_ui::show(ctx, &mut s, &mut self.skip_dialog) };
            self.import = Some(s);
            match action {
                import_ui::Action::Import => self.run_import(ctx),
                import_ui::Action::Cancel => {
                    self.import = None;
                    self.status = "Import cancelled.".into();
                }
                import_ui::Action::Reparse(idx) => {
                    if let Some(mut s) = self.import.take() {
                        self.spawn_previews(&mut s, &idx, ctx);
                        self.import = Some(s);
                    }
                }
                import_ui::Action::None => {}
            }
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::O)) {
            self.open_dialog(ctx);
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::E)) {
            self.export_excel();
        }
        let theme = self.theme_settings.build();
        self.top_bar(ctx);
        self.status_bar(ctx);
        self.side_panel(ctx);
        self.central(ctx, &theme);
        self.schedule_analysis(ctx);
    }
}
