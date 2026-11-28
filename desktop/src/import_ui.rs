//! The import dialog: detected format, time interpretation, column roles, raw preview, and per-file
//! overrides, so any chronobiology export can be brought in without editing the file.

use crate::data::{ms_to_naive, naive_to_ms};
use crate::import::{format_duration, ColRole, DateOrder, ImportOptions, Layout, MergeMode, Parsed, Source, TimeUnit};
use eframe::egui::{self, Color32, RichText};
use std::sync::Arc;

pub const PREVIEW_ROWS: usize = 5000;

pub struct PendingFile {
    pub source: Arc<Source>,
    pub opts: ImportOptions,
    pub parsed: Option<Result<Parsed, String>>,
    /// Bumped on every option change; stale previews are dropped
    pub version: u64,
    pub include: bool,
    start_text: String,
    start_err: bool,
}

impl PendingFile {
    pub fn new(source: Source) -> Self {
        PendingFile { source: Arc::new(source), opts: ImportOptions::default(), parsed: None, version: 0, include: true, start_text: String::new(), start_err: false }
    }
    pub fn ok(&self) -> bool {
        matches!(self.parsed, Some(Ok(_)))
    }
    /// Recognised with nothing left for the user to decide
    pub fn confident(&self) -> bool {
        match &self.parsed {
            Some(Ok(p)) => !p.date_ambiguous && !p.needs_interval && p.format != "Delimited table" && p.format != "Single column of values",
            _ => false,
        }
    }
}

pub struct ImportSession {
    pub id: u64,
    pub files: Vec<PendingFile>,
    pub selected: usize,
    pub merge: MergeMode,
    pub filter: String,
}

pub enum Action {
    None,
    Import,
    Cancel,
    /// Re-parse these files (their options changed)
    Reparse(Vec<usize>),
}

fn status_icon(f: &PendingFile) -> (&'static str, Color32) {
    match &f.parsed {
        None => ("…", Color32::GRAY),
        Some(Err(_)) => ("✖", Color32::from_rgb(220, 70, 70)),
        Some(Ok(p)) if p.date_ambiguous || p.needs_interval => ("⚠", Color32::from_rgb(230, 160, 40)),
        Some(Ok(_)) => ("✔", Color32::from_rgb(70, 180, 90)),
    }
}

fn parse_start(s: &str) -> Option<i64> {
    let s = s.trim();
    ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d"]
        .iter()
        .find_map(|f| {
            chrono::NaiveDateTime::parse_from_str(s, f).ok().or_else(|| chrono::NaiveDate::parse_from_str(s, f).ok().map(|d| d.and_hms_opt(0, 0, 0).unwrap()))
        })
        .map(naive_to_ms)
}

/// Draw the dialog. Returns what the app should do next.
pub fn show(ctx: &egui::Context, s: &mut ImportSession, skip_dialog: &mut bool) -> Action {
    let mut action = Action::None;
    let mut reparse: Vec<usize> = Vec::new();
    let screen = ctx.screen_rect();
    let resp = egui::Modal::new(egui::Id::new(("import", s.id))).show(ctx, |ui| {
        ui.set_width((screen.width() - 80.0).clamp(560.0, 1180.0));
        let n_files = s.files.len();
        ui.horizontal(|ui| {
            ui.heading("Import data");
            ui.label(RichText::new(format!("{} file{}", n_files, if n_files == 1 { "" } else { "s" })).weak());
        });
        ui.separator();
        let body_h = (screen.height() - 220.0).clamp(300.0, 760.0);
        ui.horizontal_top(|ui| {
            // A separator in a horizontal row grows to the available height, which inside a modal is the
            // whole screen; cap the row so the footer (Import / Cancel) always stays visible.
            ui.set_max_height(body_h);
            // ---- file list --------------------------------------------------------------------
            if n_files > 1 {
                ui.vertical(|ui| {
                    ui.set_width(240.0);
                    ui.add(egui::TextEdit::singleline(&mut s.filter).hint_text("filter files…").desired_width(230.0));
                    let ok = s.files.iter().filter(|f| f.ok()).count();
                    ui.label(RichText::new(format!("{ok} of {n_files} readable")).small());
                    ui.horizontal(|ui| {
                        if ui.small_button("all").clicked() {
                            s.files.iter_mut().for_each(|f| f.include = true);
                        }
                        if ui.small_button("none").clicked() {
                            s.files.iter_mut().for_each(|f| f.include = false);
                        }
                        if ui.small_button("readable only").clicked() {
                            s.files.iter_mut().for_each(|f| f.include = f.ok());
                        }
                    });
                    egui::ScrollArea::vertical().id_salt("files").max_height(body_h - 60.0).show(ui, |ui| {
                        let filt = s.filter.to_lowercase();
                        for i in 0..n_files {
                            if !filt.is_empty() && !s.files[i].source.name.to_lowercase().contains(&filt) {
                                continue;
                            }
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut s.files[i].include, "");
                                let (icon, col) = status_icon(&s.files[i]);
                                ui.colored_label(col, icon);
                                let name = s.files[i].source.name.clone();
                                if ui.selectable_label(s.selected == i, name).clicked() {
                                    s.selected = i;
                                }
                            });
                        }
                    });
                });
                ui.separator();
            }
            // ---- details of the selected file ----------------------------------------------------
            ui.vertical(|ui| {
                egui::ScrollArea::vertical().id_salt("details").max_height(body_h).min_scrolled_height(body_h).auto_shrink([false, false]).show(ui, |ui| {
                    let i = s.selected.min(n_files.saturating_sub(1));
                    if let Some(changed) = file_details(ui, &mut s.files[i]) {
                        if changed {
                            reparse.push(i);
                        }
                    }
                    if n_files > 1 && ui.button("Apply these settings to all files").on_hover_text("Copies date order, start time, time unit, interval, layout and column roles").clicked() {
                        let opts = s.files[i].opts.clone();
                        let st = s.files[i].start_text.clone();
                        for (k, f) in s.files.iter_mut().enumerate() {
                            if k != i {
                                f.opts = opts.clone();
                                f.start_text = st.clone();
                                f.version += 1;
                                f.parsed = None;
                                reparse.push(k);
                            }
                        }
                    }
                });
            });
        });
        ui.separator();
        // ---- footer ---------------------------------------------------------------------------
        let included: Vec<&PendingFile> = s.files.iter().filter(|f| f.include).collect();
        let ok_inc = included.iter().filter(|f| f.ok()).count();
        let subjects: usize = included.iter().filter_map(|f| f.parsed.as_ref()?.as_ref().ok()).map(|p| p.subjects.len()).sum();
        let busy = s.files.iter().any(|f| f.parsed.is_none());
        if included.len() > 1 {
            ui.horizontal(|ui| {
                ui.label("Combine files:");
                egui::ComboBox::from_id_salt("merge").selected_text(s.merge.label()).width(420.0).show_ui(ui, |ui| {
                    for m in MergeMode::ALL {
                        ui.selectable_value(&mut s.merge, m, m.label());
                    }
                });
            });
        }
        ui.horizontal(|ui| {
            ui.checkbox(skip_dialog, "Import directly next time when the format is recognised").on_hover_text("You can always reopen this dialog from Data → Import settings");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // same rule as import::combine (on the preview rows), so the label says what will happen
                let parsed: Vec<&Parsed> = included.iter().filter_map(|f| f.parsed.as_ref()?.as_ref().ok()).collect();
                let sets: Vec<&crate::data::RawDataset> = parsed.iter().flat_map(|p| p.subjects.iter()).filter(|d| !d.is_empty()).collect();
                let merged = sets.len() > 1 && crate::import::will_merge(&sets, parsed.iter().all(|p| p.subjects.len() == 1), s.merge);
                let label = if merged {
                    format!("Import {ok_inc} file(s) → 1 recording")
                } else if subjects > 1 {
                    format!("Import {ok_inc} file(s) · {subjects} subjects")
                } else {
                    format!("Import {ok_inc} file(s)")
                };
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter)) && !ui.memory(|m| m.focused().is_some());
                if (ui.add_enabled(ok_inc > 0 && !busy, egui::Button::new(RichText::new(label).strong())).clicked() || (enter && ok_inc > 0 && !busy)) && matches!(action, Action::None) {
                    action = Action::Import;
                }
                if ui.button("Cancel").clicked() {
                    action = Action::Cancel;
                }
                if busy {
                    ui.spinner();
                    ui.label("reading…");
                }
            });
        });
    });
    if resp.should_close() && matches!(action, Action::None) {
        action = Action::Cancel;
    }
    if !reparse.is_empty() && matches!(action, Action::None) {
        reparse.sort_unstable();
        reparse.dedup();
        action = Action::Reparse(reparse);
    }
    action
}

/// Returns Some(true) when an option changed (re-parse needed).
fn file_details(ui: &mut egui::Ui, f: &mut PendingFile) -> Option<bool> {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(&f.source.name).strong().size(16.0));
        ui.label(RichText::new(format!("{:.1} KB", f.source.bytes.len() as f64 / 1024.0)).weak());
    });
    let parsed = match &f.parsed {
        None => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Reading…");
            });
            return Some(false);
        }
        Some(Err(e)) => {
            ui.colored_label(Color32::from_rgb(220, 70, 70), e);
            ui.label(RichText::new("Adjust the options below (header row, time column, interval) or deselect this file.").small());
            None
        }
        Some(Ok(p)) => Some(p.clone()),
    };
    if let Some(p) = &parsed {
        ui.horizontal_wrapped(|ui| {
            ui.label("Detected:");
            ui.label(RichText::new(&p.format).strong().color(Color32::from_rgb(90, 170, 250)));
            ui.label(RichText::new(p.summary()).weak());
        });
        if !p.notes.is_empty() {
            for n in &p.notes {
                ui.label(RichText::new(format!("• {n}")).small());
            }
        }
    }
    ui.add_space(4.0);
    // ---- time ----------------------------------------------------------------------------------
    egui::Grid::new("time-opts").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        ui.label("Time");
        let desc = parsed.as_ref().map(|p| p.time_desc.clone()).unwrap_or_default();
        ui.label(RichText::new(if desc.is_empty() { "—".into() } else { desc }).strong());
        ui.end_row();

        ui.label("Date order");
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("dateorder").selected_text(f.opts.date_order.label()).show_ui(ui, |ui| {
                for o in DateOrder::ALL {
                    changed |= ui.selectable_value(&mut f.opts.date_order, o, o.label()).changed();
                }
            });
            if parsed.as_ref().is_some_and(|p| p.date_ambiguous) {
                ui.colored_label(Color32::from_rgb(230, 160, 40), "⚠ ambiguous (all days ≤ 12): check the first/last dates below");
            }
        });
        ui.end_row();

        let show_unit = parsed.as_ref().is_some_and(|p| p.numeric_time) || f.opts.time_unit.is_some();
        if show_unit {
            ui.label("Time unit");
            let sel = f.opts.time_unit.map_or("Auto", |u| u.label());
            egui::ComboBox::from_id_salt("tunit").selected_text(sel).show_ui(ui, |ui| {
                changed |= ui.selectable_value(&mut f.opts.time_unit, None, "Auto").changed();
                for u in TimeUnit::ALL {
                    changed |= ui.selectable_value(&mut f.opts.time_unit, Some(u), u.label()).changed();
                }
            });
            ui.end_row();
        }

        let show_start = parsed.as_ref().is_none_or(|p| p.relative_time || p.needs_interval) || f.opts.start.is_some();
        if show_start {
            ui.label("Recording start");
            ui.horizontal(|ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut f.start_text).hint_text("YYYY-MM-DD HH:MM").desired_width(150.0));
                if r.lost_focus() || (r.changed() && parse_start(&f.start_text).is_some()) {
                    let new = if f.start_text.trim().is_empty() { None } else { parse_start(&f.start_text) };
                    f.start_err = !f.start_text.trim().is_empty() && new.is_none();
                    if new != f.opts.start && !f.start_err {
                        f.opts.start = new;
                        changed = true;
                    }
                }
                if f.start_err {
                    ui.colored_label(Color32::from_rgb(220, 70, 70), "use YYYY-MM-DD HH:MM");
                } else {
                    ui.label(RichText::new("no calendar time in the file; phases are measured from this start").small().weak());
                }
            });
            ui.end_row();
        }

        let needs_iv = parsed.as_ref().is_none_or(|p| p.needs_interval) || f.opts.interval_s.is_some();
        ui.label("Sampling interval");
        ui.horizontal(|ui| {
            let mut on = f.opts.interval_s.is_some();
            if ui.checkbox(&mut on, if needs_iv { "set (no time column found)" } else { "override (evenly spaced rows)" }).changed() {
                f.opts.interval_s = if on { Some(60.0) } else { None };
                changed = true;
            }
            if let Some(iv) = f.opts.interval_s.as_mut() {
                changed |= ui.add(egui::DragValue::new(iv).range(0.001..=86400.0).speed(1.0).suffix(" s")).changed();
                ui.label(RichText::new(format_duration(*iv)).weak());
            }
        });
        ui.end_row();

        ui.label("Header row");
        ui.horizontal(|ui| {
            let mut manual = f.opts.header_line.is_some();
            if ui.checkbox(&mut manual, "set manually").changed() {
                f.opts.header_line = if manual { Some(parsed.as_ref().and_then(|p| p.header_line).unwrap_or(0)) } else { None };
                changed = true;
            }
            if let Some(h) = f.opts.header_line.as_mut() {
                let mut line1 = *h + 1;
                if ui.add(egui::DragValue::new(&mut line1).range(1..=500).prefix("line ")).changed() {
                    *h = line1 - 1;
                    changed = true;
                }
            } else if let Some(p) = &parsed {
                ui.label(RichText::new(match p.header_line {
                    Some(h) => format!("auto: column names on line {}, data from line {}", h + 1, p.data_line + 1),
                    None => format!("auto: no column-name row, data from line {}", p.data_line + 1),
                }).weak());
            }
        });
        ui.end_row();

        if parsed.as_ref().is_some_and(|p| p.can_layout) || f.opts.layout != Layout::Auto {
            ui.label("Layout");
            ui.horizontal(|ui| {
                let cur = parsed.as_ref().map_or(f.opts.layout, |p| p.layout);
                let mut sel = if f.opts.layout == Layout::Auto { cur } else { f.opts.layout };
                let a = ui.radio_value(&mut sel, Layout::Channels, "Columns are measurements of one recording");
                let b = ui.radio_value(&mut sel, Layout::Subjects, "Columns are separate subjects / samples");
                if a.changed() || b.changed() {
                    f.opts.layout = sel;
                    changed = true;
                }
            });
            ui.end_row();
        }
    });
    // ---- columns -------------------------------------------------------------------------------
    if let Some(p) = &parsed {
        if !p.columns.is_empty() {
            ui.add_space(6.0);
            ui.label(RichText::new("Columns").strong());
            let choices = ColRole::choices();
            egui::ScrollArea::vertical().id_salt("cols").max_height(220.0).show(ui, |ui| {
                egui::Grid::new("cols-grid").striped(true).num_columns(3).spacing([14.0, 4.0]).show(ui, |ui| {
                    ui.label(RichText::new("Column").weak());
                    ui.label(RichText::new("Used as").weak());
                    ui.label(RichText::new("First value").weak());
                    ui.end_row();
                    let editable = !matches!(p.format.as_str(), "TriKinetics DAM monitor" | "Apple Health export.xml" | "Empatica E4" | "Fitbit JSON" | "Fitbit intraday JSON" | "Time-course table (ZT / CT columns)");
                    for (ci, col) in p.columns.iter().enumerate().take(400) {
                        ui.label(&col.name);
                        let cur = f.opts.roles.get(&ci).copied().unwrap_or(col.role);
                        if editable {
                            let mut sel = cur;
                            egui::ComboBox::from_id_salt(("role", ci)).selected_text(sel.label()).width(210.0).show_ui(ui, |ui| {
                                for ch in &choices {
                                    ui.selectable_value(&mut sel, *ch, ch.label());
                                }
                            });
                            if sel != cur {
                                f.opts.roles.insert(ci, sel);
                                changed = true;
                            }
                        } else {
                            ui.label(cur.label());
                        }
                        let role_col = match cur {
                            ColRole::Ignore => Color32::GRAY,
                            _ => ui.visuals().text_color(),
                        };
                        ui.label(RichText::new(&col.example).monospace().color(role_col));
                        ui.end_row();
                    }
                });
            });
            if !f.opts.roles.is_empty() && ui.small_button("↺ reset column roles").clicked() {
                f.opts.roles.clear();
                changed = true;
            }
        }
        // First / last parsed time for a quick sanity check
        if let Some(d) = p.subjects.first() {
            if let Some((a, b)) = d.time_range() {
                ui.label(RichText::new(format!(
                    "First sample {}  ·  last {}{}",
                    ms_to_naive(a).format("%a %Y-%m-%d %H:%M:%S"),
                    ms_to_naive(b).format("%a %Y-%m-%d %H:%M:%S"),
                    if p.truncated { " (preview rows)" } else { "" }
                ))
                .small());
                ui.label(RichText::new(format!("Channels: {}", d.describe_channels().join(", "))).small());
            }
        }
    }
    // ---- raw preview ---------------------------------------------------------------------------
    ui.add_space(6.0);
    egui::CollapsingHeader::new("File preview").default_open(true).show(ui, |ui| {
        let (lines, header, data_line) = match &parsed {
            Some(p) => (p.preview.clone(), p.header_line, p.data_line),
            None => (crate::import::decode(&f.source.bytes[..f.source.bytes.len().min(8192)]).lines().take(40).map(|s| s.to_string()).collect(), None, 0),
        };
        egui::ScrollArea::both().id_salt("raw").max_height(200.0).show(ui, |ui| {
            for (k, l) in lines.iter().enumerate() {
                let txt = format!("{:>4} │ {}", k + 1, l);
                let rt = RichText::new(txt).monospace().size(12.0);
                let rt = if Some(k) == header {
                    rt.strong().color(Color32::from_rgb(90, 170, 250))
                } else if k < data_line {
                    rt.weak()
                } else {
                    rt
                };
                ui.label(rt);
            }
        });
        ui.label(RichText::new("blue = column names · grey = header / metadata lines").small().weak());
    });
    if changed {
        f.version += 1;
        f.parsed = None;
    }
    Some(changed)
}

/// Fill the start text box from options (when the session opens).
pub fn sync_text(f: &mut PendingFile) {
    if let Some(s) = f.opts.start {
        f.start_text = ms_to_naive(s).format("%Y-%m-%d %H:%M").to_string();
    }
}
