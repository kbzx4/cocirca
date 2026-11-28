//! Charts: themed egui_plot wrappers plus custom-painted actogram and polar plot.
//!
//! Performance notes
//! * Long series are min/max-decimated to the *visible* x-range every frame (O(visible bins)), so a
//!   1 M-bin recording pans and zooms smoothly while zoomed views still show every sample.
//! * The actogram is one GPU mesh; columns narrower than a pixel are merged by max before meshing.
//! * Folded waveforms and actogram grids are precomputed off the UI thread (see `Extras`).

use crate::data::Binned;
use crate::pipeline::{ChannelReport, Report};
use crate::stats::hhmm;
use crate::theme::Theme;
use eframe::egui::{self, epaint::TextShape, Align2, Color32, FontFamily, FontId, Mesh, Pos2, Rect, RichText, Sense, Shape, Stroke, Vec2};
use egui_plot::{Bar, BarChart, Corner, Legend, Line, LineStyle, MarkerShape, Plot, PlotPoint, PlotPoints, PlotUi, Points, Polygon, Text, VLine, HLine};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::f32::consts::{FRAC_PI_2, TAU};

// ------------------------------------------------------------------------------------------------
// Precomputed chart data (built in the analysis thread)
// ------------------------------------------------------------------------------------------------

pub struct ActogramData {
    pub days: usize,
    pub cols: usize,
    pub values: Vec<f32>,
    pub light: Option<Vec<f32>>,
    pub scale: f32,
    /// Bar baseline: 0 for counts, the 1st percentile for signals like heart rate or temperature
    pub lo: f32,
    /// Light overlay is scaled between its 5th (dark) and 99th (bright) percentiles
    pub light_lo: f32,
    pub light_scale: f32,
}

pub struct Extras {
    pub acto: ActogramData,
    /// (phase hour, mean) over 288 phase bins of τ
    pub fold_mean: Vec<[f64; 2]>,
    /// up to 6000 raw (phase, value) points for the scatter
    pub fold_points: Vec<[f64; 2]>,
}

pub fn build_extras(b: &Binned, ch: &ChannelReport, light_overlay: Option<&[f64]>) -> Extras {
    let n = b.len();
    let days = b.n_day_rows().max(1) as usize;
    let bins_per_day = ((86_400 / b.epoch_sec.max(1)) as usize).max(1);
    let cols = bins_per_day.min(1440).max(1);
    let mut sum = vec![0.0f64; days * cols];
    let mut cnt = vec![0u32; days * cols];
    let mut lsum = vec![0.0f64; if light_overlay.is_some() { days * cols } else { 0 }];
    let mut lcnt = vec![0u32; lsum.len()];
    for i in 0..n {
        let d = (b.day_index[i] - 1) as usize;
        let since_mid = ((b.bin_start_ms(i) - b.midnight_ms).rem_euclid(86_400_000)) as f64 / 86_400_000.0;
        let c = ((since_mid * cols as f64) as usize).min(cols - 1);
        let k = d * cols + c;
        let v = ch.values[i];
        if v.is_finite() {
            sum[k] += v;
            cnt[k] += 1;
        }
        if let Some(l) = light_overlay {
            if l[i].is_finite() {
                lsum[k] += l[i];
                lcnt[k] += 1;
            }
        }
    }
    let mean = |s: &[f64], c: &[u32]| -> Vec<f32> { s.iter().zip(c).map(|(&s, &c)| if c > 0 { (s / c as f64) as f32 } else { f32::NAN }).collect() };
    let values = mean(&sum, &cnt);
    let p99 = |v: &[f32]| {
        let f: Vec<f64> = v.iter().filter(|x| x.is_finite()).map(|&x| x as f64).collect();
        (crate::stats::percentile(&f, 99.0) as f32).max(1e-9)
    };
    let scale = p99(&values);
    let lo = if ch.name == "Activity" {
        0.0
    } else {
        let f: Vec<f64> = values.iter().filter(|x| x.is_finite()).map(|&x| x as f64).collect();
        (crate::stats::percentile(&f, 1.0) as f32).min(scale - 1e-6)
    };
    let light = light_overlay.map(|_| mean(&lsum, &lcnt));
    let light_scale = light.as_ref().map(|l| p99(l)).unwrap_or(1.0);
    let light_lo = light
        .as_ref()
        .map(|l| {
            let f: Vec<f64> = l.iter().filter(|x| x.is_finite()).map(|&x| x as f64).collect();
            crate::stats::percentile(&f, 5.0) as f32
        })
        .unwrap_or(0.0);

    let tau = ch.tau;
    let nb = 288;
    let (mut s, mut c) = (vec![0.0; nb], vec![0u32; nb]);
    for (t, v) in b.elapsed_h.iter().zip(&ch.values) {
        if v.is_finite() {
            let k = ((t.rem_euclid(tau) / tau) * nb as f64) as usize % nb;
            s[k] += v;
            c[k] += 1;
        }
    }
    let fold_mean = (0..nb).filter(|&k| c[k] > 0).map(|k| [(k as f64 + 0.5) * tau / nb as f64, s[k] / c[k] as f64]).collect();
    let valid: Vec<usize> = (0..n).filter(|&i| ch.values[i].is_finite()).collect();
    let step = (valid.len() / 6000).max(1);
    let fold_points = valid.iter().step_by(step).map(|&i| [b.elapsed_h[i].rem_euclid(tau), ch.values[i]]).collect();
    Extras { acto: ActogramData { days, cols, values, light, scale, lo, light_lo, light_scale }, fold_mean, fold_points }
}

// ------------------------------------------------------------------------------------------------
// Chart context: theme, axis-label fonts, label registry, export rectangle
// ------------------------------------------------------------------------------------------------

pub struct ChartCtx<'a> {
    pub theme: &'a Theme,
    pub loaded_fonts: &'a HashSet<String>,
    pub seen_labels: RefCell<BTreeSet<String>>,
    pub chart_rect: RefCell<Option<Rect>>,
}

pub fn family_key(font: &str, bold: bool) -> String {
    if bold {
        format!("lbl-{font}-bold")
    } else {
        format!("lbl-{font}")
    }
}

impl<'a> ChartCtx<'a> {
    pub fn new(theme: &'a Theme, loaded_fonts: &'a HashSet<String>) -> Self {
        ChartCtx { theme, loaded_fonts, seen_labels: RefCell::new(BTreeSet::new()), chart_rect: RefCell::new(None) }
    }

    /// (family, needs synthetic bold)
    fn family(&self) -> (FontFamily, bool) {
        let l = &self.theme.label;
        match &l.font {
            Some(f) => {
                let bold_key = family_key(f, true);
                if l.bold && self.loaded_fonts.contains(&bold_key) {
                    (FontFamily::Name(bold_key.into()), false)
                } else if self.loaded_fonts.contains(&family_key(f, false)) {
                    (FontFamily::Name(family_key(f, false).into()), l.bold)
                } else {
                    (FontFamily::Proportional, l.bold)
                }
            }
            None => (FontFamily::Proportional, l.bold),
        }
    }

    pub fn label_size(&self) -> f32 {
        if self.theme.label.size > 0.0 {
            self.theme.label.size
        } else {
            13.0
        }
    }

    pub fn label_font(&self) -> FontId {
        FontId::new(self.label_size(), self.family().0)
    }

    /// Styled axis-label text for egui_plot; records the original label for the rename list.
    pub fn axis(&self, original: &str, is_x: bool) -> RichText {
        self.seen_labels.borrow_mut().insert(original.to_string());
        let (fam, strong) = self.family();
        let color = if is_x { self.theme.x_label_color() } else { self.theme.y_label_color() };
        let t = RichText::new(self.theme.label_text(original)).family(fam).size(self.label_size()).color(color);
        if strong {
            t.strong()
        } else {
            t
        }
    }

    pub fn mark(&self, r: Rect) {
        let mut cr = self.chart_rect.borrow_mut();
        *cr = Some(cr.map_or(r, |c| c.union(r)));
    }

    /// Paint an axis label with the label style (optionally rotated for a y-axis).
    fn paint_label(&self, p: &egui::Painter, pos: Pos2, original: &str, is_x: bool, anchor: Align2) {
        self.seen_labels.borrow_mut().insert(original.to_string());
        let color = if is_x { self.theme.x_label_color() } else { self.theme.y_label_color() };
        let (fam, strong) = self.family();
        let text = self.theme.label_text(original);
        let galley = p.layout_no_wrap(text, FontId::new(self.label_size(), fam), color);
        let sz = galley.size();
        if is_x {
            let r = anchor.anchor_size(pos, sz);
            p.galley(r.min, galley.clone(), color);
            if strong {
                p.galley(r.min + Vec2::new(0.6, 0.0), galley, color);
            }
        } else {
            // rotated −90°: the galley's width runs upward
            let origin = Pos2::new(pos.x - sz.y / 2.0, pos.y + sz.x / 2.0);
            p.add(TextShape::new(origin, galley.clone(), color).with_angle(-FRAC_PI_2));
            if strong {
                p.add(TextShape::new(origin + Vec2::new(0.0, -0.6), galley, color).with_angle(-FRAC_PI_2));
            }
        }
    }
}

pub struct PlotOpts {
    pub height: f32,
    pub legend: bool,
    pub include_y: Vec<f64>,
    pub include_x: Vec<f64>,
    pub link_x: Option<&'static str>,
}

impl Default for PlotOpts {
    fn default() -> Self {
        PlotOpts { height: 260.0, legend: true, include_y: Vec::new(), include_x: Vec::new(), link_x: None }
    }
}

/// egui_plot with the chart theme applied (background, text, grid, axis labels).
pub fn themed_plot<R>(ui: &mut egui::Ui, cx: &ChartCtx, id: &str, x: &str, y: &str, o: PlotOpts, body: impl FnOnce(&mut PlotUi) -> R) -> R {
    let th = cx.theme;
    ui.scope(|ui| {
        let v = ui.visuals_mut();
        v.extreme_bg_color = th.background;
        v.override_text_color = Some(th.foreground);
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0f32, th.grid);
        v.window_fill = th.background;
        v.window_stroke = Stroke::new(1.0f32, th.grid);
        let mut plot = Plot::new(id).height(o.height).x_axis_label(cx.axis(x, true)).y_axis_label(cx.axis(y, false)).y_axis_min_width(40.0);
        if o.legend {
            plot = plot.legend(Legend::default().position(Corner::RightTop));
        }
        for &yv in &o.include_y {
            plot = plot.include_y(yv);
        }
        for &xv in &o.include_x {
            plot = plot.include_x(xv);
        }
        if let Some(g) = o.link_x {
            plot = plot.link_axis(g, egui::Vec2b::new(true, false)).link_cursor(g, egui::Vec2b::new(true, false));
        }
        // Frame the plot in the chart background so the axis area matches too
        let resp = egui::Frame::default().fill(th.background).inner_margin(6.0).show(ui, |ui| plot.show(ui, body));
        cx.mark(resp.response.rect);
        resp.inner.inner
    })
    .inner
}

fn alpha(c: Color32, a: f32) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (a * 255.0) as u8)
}

/// Min/max decimation of a uniform series (x = x0 + i·dx) restricted to [xmin, xmax].
pub fn decimate(x0: f64, dx: f64, y: &[f64], xmin: f64, xmax: f64, buckets: usize) -> Vec<[f64; 2]> {
    let n = y.len();
    if n == 0 {
        return Vec::new();
    }
    let (lo, hi) = if xmin.is_finite() && xmax.is_finite() && xmax > xmin {
        let lo = ((xmin - x0) / dx).floor().max(0.0) as usize;
        let hi = (((xmax - x0) / dx).ceil() as usize + 1).min(n);
        (lo.min(n), hi)
    } else {
        (0, n)
    };
    if hi <= lo {
        return Vec::new();
    }
    let count = hi - lo;
    if count <= buckets * 2 {
        return (lo..hi).filter(|&i| y[i].is_finite()).map(|i| [x0 + i as f64 * dx, y[i]]).collect();
    }
    let per = count as f64 / buckets as f64;
    let mut out = Vec::with_capacity(buckets * 2);
    for bk in 0..buckets {
        let s = lo + (bk as f64 * per) as usize;
        let e = (lo + ((bk + 1) as f64 * per) as usize).min(hi);
        let (mut imin, mut imax) = (usize::MAX, usize::MAX);
        for i in s..e {
            let v = y[i];
            if !v.is_finite() {
                continue;
            }
            if imin == usize::MAX || v < y[imin] {
                imin = i;
            }
            if imax == usize::MAX || v > y[imax] {
                imax = i;
            }
        }
        if imin == usize::MAX {
            continue;
        }
        let (a, b) = if imin <= imax { (imin, imax) } else { (imax, imin) };
        out.push([x0 + a as f64 * dx, y[a]]);
        if b != a {
            out.push([x0 + b as f64 * dx, y[b]]);
        }
    }
    out
}

fn vspan(x0: f64, x1: f64, y0: f64, y1: f64, c: Color32, name: &str) -> Polygon<'static> {
    Polygon::new(PlotPoints::from(vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]])).fill_color(c).stroke(Stroke::NONE).name(name)
}

fn channel_color(th: &Theme, ch: &ChannelReport) -> Color32 {
    let n = ch.name.to_lowercase();
    if n == "light" {
        th.role("light")
    } else if n == "activity" {
        th.role("activity")
    } else if n.contains("heart") {
        th.role("fit")
    } else if n.contains("temp") {
        th.role("highlight")
    } else {
        th.role("accent")
    }
}

fn value_label(ch: &ChannelReport) -> String {
    format!("{} ({})", ch.name, ch.unit)
}

// ------------------------------------------------------------------------------------------------
// Actogram
// ------------------------------------------------------------------------------------------------

pub struct ActoOpts {
    pub double: bool,
    pub light: bool,
    pub markers: bool,
}

pub fn actogram(ui: &mut egui::Ui, cx: &ChartCtx, rep: &Report, ch: &ChannelReport, a: &ActogramData, o: &ActoOpts) {
    let th = cx.theme;
    let days = a.days;
    let span = if o.double { 48.0 } else { 24.0 };
    let row_h = (560.0 / days as f32).clamp(5.0, 30.0);
    let (left, right, top, bottom) = (64.0, 12.0, 10.0, 48.0);
    let width = ui.available_width().max(320.0);
    let height = top + bottom + row_h * days as f32;
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, height), Sense::hover());
    cx.mark(rect);
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, th.background);
    let plot = Rect::from_min_max(Pos2::new(rect.left() + left, rect.top() + top), Pos2::new(rect.right() - right, rect.bottom() - bottom));
    let pw = plot.width();
    let px_per_h = pw / span;
    let col_w = px_per_h * 24.0 / a.cols as f32;
    // Merge columns narrower than one pixel (max keeps activity bouts visible)
    let group = ((1.0 / col_w).ceil() as usize).max(1);
    let gw = col_w * group as f32;
    let groups = a.cols.div_ceil(group);
    let bar_c = channel_color(th, ch);
    let light_c = th.role("light");
    let mut mesh = Mesh::default();
    let halves: &[(usize, f32)] = if o.double { &[(0, 0.0), (1, 24.0)] } else { &[(0, 0.0)] };
    for r in 0..days {
        let y_base = plot.top() + (r as f32 + 1.0) * row_h;
        for &(off, x_off) in halves {
            let d = r + off;
            if d >= days {
                continue;
            }
            let row = &a.values[d * a.cols..(d + 1) * a.cols];
            let lrow = a.light.as_ref().filter(|_| o.light).map(|l| &l[d * a.cols..(d + 1) * a.cols]);
            for g in 0..groups {
                let s = g * group;
                let e = (s + group).min(a.cols);
                let x0 = plot.left() + x_off * px_per_h + g as f32 * gw;
                let x1 = x0 + gw.max(1.0);
                if let Some(lr) = lrow {
                    let lv = lr[s..e].iter().copied().filter(|v| v.is_finite()).fold(f32::NAN, |m, v| if m.is_nan() || v > m { v } else { m });
                    if lv.is_finite() {
                        let f = ((lv - a.light_lo) / (a.light_scale - a.light_lo).max(1e-9)).clamp(0.0, 1.0);
                        if f > 0.15 {
                            mesh.add_colored_rect(Rect::from_min_max(Pos2::new(x0, y_base - row_h), Pos2::new(x1, y_base)), alpha(light_c, 0.30 * f));
                        }
                    }
                }
                let v = row[s..e].iter().copied().filter(|v| v.is_finite()).fold(f32::NAN, |m, v| if m.is_nan() || v > m { v } else { m });
                if v.is_finite() && v > 0.0 {
                    let h = ((v - a.lo) / (a.scale - a.lo)).clamp(0.0, 1.0) * (row_h - 1.5);
                    if h >= 0.3 {
                        mesh.add_colored_rect(Rect::from_min_max(Pos2::new(x0, y_base - h), Pos2::new(x1, y_base)), bar_c);
                    }
                }
            }
        }
    }
    // Gridlines & baselines
    let step = if o.double { 6 } else { 3 };
    for h in (0..=(span as i32)).step_by(step) {
        let x = plot.left() + h as f32 * px_per_h;
        p.line_segment([Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())], Stroke::new(if h % 24 == 0 { 1.0f32 } else { 0.5 }, th.grid));
        p.text(Pos2::new(x, plot.bottom() + 4.0), Align2::CENTER_TOP, format!("{:02}", h % 24), FontId::proportional(11.0), th.foreground);
    }
    p.add(Shape::mesh(mesh));
    let label_every = ((12.0 / row_h).ceil() as usize).max(1);
    for r in 0..days {
        let y_base = plot.top() + (r as f32 + 1.0) * row_h;
        p.line_segment([Pos2::new(plot.left(), y_base), Pos2::new(plot.right(), y_base)], Stroke::new(0.5f32, th.grid));
        if r % label_every == 0 {
            p.text(Pos2::new(plot.left() - 6.0, y_base - row_h / 2.0), Align2::RIGHT_CENTER, format!("{}", r + 1), FontId::proportional(11.0), th.foreground);
        }
    }
    // Onset / offset markers on the day's row (and the previous row's right half when double-plotted)
    if o.markers {
        for d in &ch.daily {
            let r = (d.day - 1) as usize;
            if r >= days {
                continue;
            }
            let mark = |hour: f64, c: Color32| {
                if !hour.is_finite() {
                    return;
                }
                let put = |row: usize, xh: f64| {
                    if xh > span as f64 {
                        return;
                    }
                    let x = plot.left() + xh as f32 * px_per_h;
                    let yb = plot.top() + (row as f32 + 1.0) * row_h;
                    p.line_segment([Pos2::new(x, yb - row_h * 0.95), Pos2::new(x, yb)], Stroke::new(2.0f32, c));
                };
                put(r, hour);
                if o.double && r > 0 {
                    put(r - 1, hour + 24.0);
                }
            };
            mark(d.onset_h, th.role("onset"));
            let off = if d.offset_h.is_finite() && d.onset_h.is_finite() && d.offset_h < d.onset_h { d.offset_h + 24.0 } else { d.offset_h };
            mark(off, th.role("offset"));
        }
    }
    cx.paint_label(&p, Pos2::new(plot.center().x, rect.bottom() - 4.0), if o.double { "Clock Time (Double-Plotted 48 h)" } else { "Clock Time (24 h)" }, true, Align2::CENTER_BOTTOM);
    cx.paint_label(&p, Pos2::new(rect.left() + 12.0, plot.center().y), "Recording Day", false, Align2::CENTER_CENTER);

    if let Some(pos) = resp.hover_pos() {
        if plot.contains(pos) {
            let r = ((pos.y - plot.top()) / row_h) as usize;
            let xh = ((pos.x - plot.left()) / px_per_h) as f64;
            let d = r + if xh >= 24.0 { 1 } else { 0 };
            if d < days {
                let hour = xh % 24.0;
                let c = ((hour / 24.0) * a.cols as f64) as usize;
                let v = a.values[d * a.cols + c.min(a.cols - 1)];
                let date = rep.binned.date_of_day(d as i32 + 1);
                let txt = format!("Day {} ({date}) {}\n{}: {}", d + 1, hhmm(hour), ch.name, if v.is_finite() { format!("{v:.3}") } else { "no data".into() });
                resp.on_hover_text_at_pointer(txt);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Time series
// ------------------------------------------------------------------------------------------------

pub fn time_series(ui: &mut egui::Ui, cx: &ChartCtx, rep: &Report, ch: &ChannelReport, show_fits: bool, height: f32, link: Option<&'static str>) {
    let th = cx.theme;
    let b = &rep.binned;
    let (x0, dx) = (b.elapsed_h.first().copied().unwrap_or(0.0), b.epoch_sec as f64 / 3600.0);
    let opts = PlotOpts { height, link_x: link, ..Default::default() };
    themed_plot(ui, cx, &format!("ts-{}", ch.name), "Elapsed Time (hours)", &value_label(ch), opts, |pu| {
        // Auto-bounds (initial view / after double-click reset): decimate everything so the view fits the
        // whole recording. Once the user zooms or pans, decimate only the visible window at full detail.
        let (xmin, xmax) = if pu.auto_bounds().x {
            (f64::NAN, f64::NAN)
        } else {
            let b = pu.plot_bounds();
            (b.min()[0], b.max()[0])
        };
        let width_px = pu.response().rect.width().max(200.0) as usize;
        let pts = decimate(x0, dx, &ch.values, xmin, xmax, width_px);
        let xs: Vec<f64> = if pts.len() > 1 { pts.iter().map(|p| p[0]).step_by((pts.len() / 1500).max(1)).collect() } else { Vec::new() };
        pu.line(Line::new(PlotPoints::from(pts)).color(channel_color(th, ch)).width(1.0f32).name(&ch.name));
        if show_fits && !xs.is_empty() {
            let single: Vec<[f64; 2]> = xs.iter().map(|&x| [x, ch.single.eval(x)]).collect();
            let multi: Vec<[f64; 2]> = xs.iter().map(|&x| [x, ch.harmonics.eval(x)]).collect();
            pu.line(Line::new(PlotPoints::from(single)).color(th.role("accent")).width(1.2f32).name("Single cosinor"));
            pu.line(Line::new(PlotPoints::from(multi)).color(th.role("fit")).width(1.8f32).name(format!("{}-harmonic cosinor", ch.harmonics.nh)));
        }
    });
}

// ------------------------------------------------------------------------------------------------
// Waveform (folded over τ) with cosinor fits
// ------------------------------------------------------------------------------------------------

pub fn waveform(ui: &mut egui::Ui, cx: &ChartCtx, ch: &ChannelReport, ex: &Extras) {
    let th = cx.theme;
    let tau = ch.tau;
    let opts = PlotOpts { height: 340.0, include_x: vec![0.0, tau], ..Default::default() };
    themed_plot(ui, cx, &format!("wave-{}", ch.name), "Phase / Clock Hour (h)", &value_label(ch), opts, |pu| {
        pu.points(Points::new(PlotPoints::from(ex.fold_points.clone())).color(alpha(th.muted, 0.35)).radius(1.2f32).name("Data"));
        pu.line(Line::new(PlotPoints::from(ex.fold_mean.clone())).color(channel_color(th, ch)).width(1.5f32).name("Mean per phase bin"));
        let grid: Vec<f64> = (0..=480).map(|i| tau * i as f64 / 480.0).collect();
        pu.line(Line::new(PlotPoints::from(grid.iter().map(|&x| [x, ch.single.eval(x)]).collect::<Vec<_>>())).color(th.role("accent")).width(1.5f32).name("Single cosinor"));
        pu.line(Line::new(PlotPoints::from(grid.iter().map(|&x| [x, ch.harmonics.eval(x)]).collect::<Vec<_>>())).color(th.role("fit")).width(2.2f32).name(format!("{}-harmonic cosinor", ch.harmonics.nh)));
        pu.hline(HLine::new(ch.single.mesor).color(th.role("mesor")).style(LineStyle::Dashed { length: 6.0 }).name(format!("MESOR {:.3}", ch.single.mesor)));
        pu.vline(VLine::new(ch.single.harmonics[0].peak_h).color(th.role("acrophase")).width(1.5f32).name(format!("Acrophase {}", ch.single.acrophase_hhmm())));
        pu.vline(VLine::new(ch.harmonics.ortho_h).color(th.role("highlight")).style(LineStyle::Dashed { length: 4.0 }).name(format!("Orthophase {}", hhmm(ch.harmonics.ortho_h))));
    });
}

// ------------------------------------------------------------------------------------------------
// Periodograms
// ------------------------------------------------------------------------------------------------

fn band_shading(pu: &mut PlotUi, th: &Theme, ch: &ChannelReport, ymax: f64) {
    // Translucent fills read much stronger on dark backgrounds
    let a = if th.is_dark() { 0.045 } else { 0.10 };
    if let Some(bands) = &ch.bands {
        for b in bands.bands.iter().filter(|b| b.scanned && b.high > b.low) {
            pu.polygon(vspan(b.low, b.high, 0.0, ymax, alpha(th.band(b.band), a), b.band));
        }
    }
}

pub fn periodogram(ui: &mut egui::Ui, cx: &ChartCtx, ch: &ChannelReport) {
    let th = cx.theme;
    let pg = &ch.periodogram;
    let ls: Vec<[f64; 2]> = pg.periods.iter().zip(&pg.ls_power).map(|(&p, &v)| [p, v]).collect();
    let ymax = pg.ls_power.iter().copied().fold(0.0, f64::max) * 1.08 + 1e-9;
    let opts = PlotOpts { height: 280.0, include_y: vec![0.0], link_x: Some("pgram"), ..Default::default() };
    themed_plot(ui, cx, &format!("ls-{}", ch.name), "Period τ (hours)", "Normalized Power", opts, |pu| {
        band_shading(pu, th, ch, ymax);
        pu.line(Line::new(PlotPoints::from(ls)).color(th.role("accent")).width(2.0f32).name("Lomb-Scargle (= cosinor R²)"));
        if let Some(b) = &ch.bands {
            let res: Vec<[f64; 2]> = b.periods.iter().zip(&b.residual_r2).map(|(&p, &v)| [p, v]).collect();
            pu.line(Line::new(PlotPoints::from(res)).color(th.role("sweep")).style(LineStyle::Dashed { length: 5.0 }).width(1.2f32).name("Residual after prewhitening"));
        }
        pu.vline(VLine::new(ch.tau).color(th.role("fit")).width(1.5f32).name(format!("τ used {:.2} h", ch.tau)));
    });
    let qp: Vec<[f64; 2]> = pg.periods.iter().zip(&pg.chi2_qp).map(|(&p, &v)| [p, v]).collect();
    let crit: Vec<[f64; 2]> = pg.periods.iter().zip(&pg.chi2_crit).map(|(&p, &v)| [p, v]).collect();
    let opts = PlotOpts { height: 240.0, include_y: vec![0.0], link_x: Some("pgram"), ..Default::default() };
    themed_plot(ui, cx, &format!("chi-{}", ch.name), "Period τ (hours)", "Qp", opts, |pu| {
        pu.line(Line::new(PlotPoints::from(qp)).color(th.role("chi2")).width(2.0f32).name("χ² Qp"));
        pu.line(Line::new(PlotPoints::from(crit)).color(th.role("threshold")).style(LineStyle::Dashed { length: 6.0 }).name("Critical (α = 0.05)"));
        pu.vline(VLine::new(pg.best_tau_chi2).color(th.role("fit")).width(1.2f32).name(format!("χ² peak {:.2} h", pg.best_tau_chi2)));
    });
}

pub fn bands_spectrum(ui: &mut egui::Ui, cx: &ChartCtx, ch: &ChannelReport) {
    let th = cx.theme;
    let Some(b) = &ch.bands else { return };
    let raw: Vec<[f64; 2]> = b.periods.iter().zip(&b.raw_r2).map(|(&p, &v)| [p, v]).collect();
    let res: Vec<[f64; 2]> = b.periods.iter().zip(&b.residual_r2).map(|(&p, &v)| [p, v]).collect();
    let ymax = b.raw_r2.iter().copied().fold(0.0, f64::max) * 1.1 + 1e-9;
    let opts = PlotOpts { height: 280.0, include_y: vec![0.0], ..Default::default() };
    themed_plot(ui, cx, &format!("bands-{}", ch.name), "Period τ (hours)", "R² (variance explained)", opts, |pu| {
        band_shading(pu, th, ch, ymax);
        pu.line(Line::new(PlotPoints::from(raw)).color(th.role("sweep")).width(2.0f32).name("Raw R² spectrum"));
        pu.line(Line::new(PlotPoints::from(res)).color(th.muted).style(LineStyle::Dashed { length: 5.0 }).name("Residual after removing detected rhythms"));
        for d in b.bands.iter().filter(|d| d.detected) {
            let i = b.periods.iter().position(|&p| (p - d.period).abs() < 1e-9).unwrap_or(0);
            pu.points(Points::new(PlotPoints::from(vec![[d.period, b.raw_r2[i]]])).shape(MarkerShape::Diamond).radius(6.0f32).color(th.role("fit")).name(format!("{} {:.1} h", d.band, d.period)));
        }
        if let Some(t) = &ch.twelve {
            pu.vline(VLine::new(t.harm_period).color(th.role("harmonic")).style(LineStyle::Dashed { length: 4.0 }).name(format!("τ/2 = {:.2} h", t.harm_period)));
            pu.vline(VLine::new(t.rhythm_period).color(th.role("highlight")).width(1.2f32).name(format!("12 h rhythm {:.2} h", t.rhythm_period)));
        }
    });
}

pub fn bimodality_wave(ui: &mut egui::Ui, cx: &ChartCtx, ch: &ChannelReport) {
    let th = cx.theme;
    let Some(bm) = &ch.bimodality else { return };
    let wave: Vec<[f64; 2]> = bm.wave_h.iter().zip(&bm.wave_v).map(|(&h, &v)| [h, v]).collect();
    let opts = PlotOpts { height: 260.0, include_x: vec![0.0, bm.period], ..Default::default() };
    themed_plot(ui, cx, &format!("bimod-{}", ch.name), "Phase / Clock Hour (h)", &value_label(ch), opts, |pu| {
        pu.line(Line::new(PlotPoints::from(wave)).color(channel_color(th, ch)).width(2.2f32).name("2-harmonic waveform (τ + τ/2)"));
        let s = &bm.shape;
        pu.points(Points::new(PlotPoints::from(vec![[s.primary_h, s.primary_val]])).radius(6.0f32).color(th.role("acrophase")).name(format!("Peak 1 {}", hhmm(s.primary_h))));
        if s.secondary_h.is_finite() {
            pu.points(Points::new(PlotPoints::from(vec![[s.secondary_h, s.secondary_val]])).radius(6.0f32).color(th.role("highlight")).name(format!("Peak 2 {} (index {:.2})", hhmm(s.secondary_h), s.bimodality_index)));
            pu.points(Points::new(PlotPoints::from(vec![[s.dip_h, s.dip_val]])).shape(MarkerShape::Down).radius(6.0f32).color(th.muted).name(format!("Dip {}", hhmm(s.dip_h))));
        }
    });
}

// ------------------------------------------------------------------------------------------------
// Circular (polar) plot
// ------------------------------------------------------------------------------------------------

pub fn polar(ui: &mut egui::Ui, cx: &ChartCtx, ch: &ChannelReport) {
    let th = cx.theme;
    let size = ui.available_width().min(460.0).max(260.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(size, size + 30.0), Sense::hover());
    cx.mark(rect);
    let p = ui.painter_at(rect);
    p.rect_filled(rect, 4.0, th.background);
    let c = Pos2::new(rect.center().x, rect.top() + size / 2.0);
    let r = size / 2.0 - 34.0;
    let at = |hour: f64, rad: f32| {
        let a = (hour / 24.0) as f32 * TAU;
        Pos2::new(c.x + rad * a.sin(), c.y - rad * a.cos())
    };
    for k in 1..=4 {
        p.circle_stroke(c, r * k as f32 / 4.0, Stroke::new(if k == 4 { 1.2f32 } else { 0.5 }, th.grid));
    }
    for h in (0..24).step_by(3) {
        p.line_segment([c, at(h as f64, r)], Stroke::new(0.5f32, th.grid));
        p.text(at(h as f64, r + 14.0), Align2::CENTER_CENTER, format!("{h:02}"), FontId::proportional(12.0), th.foreground);
    }
    let series = [("Acrophase", th.role("acrophase"), 0.92f32), ("Onset", th.role("onset"), 0.78), ("Offset", th.role("offset"), 0.64)];
    for (name, col, rad) in series {
        let hours: Vec<f64> = ch.daily.iter().map(|d| match name {
            "Acrophase" => d.acrophase_h,
            "Onset" => d.onset_h,
            _ => d.offset_h,
        }).collect();
        for &h in hours.iter().filter(|h| h.is_finite()) {
            p.circle_filled(at(h, r * rad), 4.0, col);
        }
        let summ = match name {
            "Acrophase" => &ch.circ_acrophase,
            "Onset" => &ch.circ_onset,
            _ => &ch.circ_offset,
        };
        if let Some(s) = summ {
            let tip = at(s.mean_hour, r * s.r_bar as f32);
            p.arrow(c, tip - c, Stroke::new(2.5f32, col));
        }
    }
    // Legend
    let mut x = rect.left() + 10.0;
    let y = rect.bottom() - 14.0;
    for (name, col, _) in series {
        p.circle_filled(Pos2::new(x + 5.0, y), 4.5, col);
        let g = p.layout_no_wrap(name.to_string(), FontId::proportional(12.0), th.foreground);
        let w = g.size().x;
        p.galley(Pos2::new(x + 14.0, y - g.size().y / 2.0), g, th.foreground);
        x += w + 32.0;
    }
    p.text(Pos2::new(rect.right() - 8.0, y), Align2::RIGHT_CENTER, "arrow length = R̄ (phase consistency)", FontId::proportional(11.0), th.muted);
}

// ------------------------------------------------------------------------------------------------
// Acrophase shift timeline
// ------------------------------------------------------------------------------------------------

pub fn shift_timeline(ui: &mut egui::Ui, cx: &ChartCtx, ch: &ChannelReport) {
    let th = cx.theme;
    let pts = |f: fn(&crate::pipeline::DailyRow) -> f64| -> Vec<[f64; 2]> { ch.daily.iter().filter(|d| f(d).is_finite()).map(|d| [d.day as f64, f(d)]).collect() };
    let split = ch.shift.as_ref().map(|s| s.split_day as f64);
    let opts = PlotOpts { height: 260.0, include_y: vec![0.0, 24.0], link_x: Some("shift"), ..Default::default() };
    themed_plot(ui, cx, &format!("shift-phase-{}", ch.name), "Day Index", "Clock Time", opts, |pu| {
        let acro = pts(|d| d.acrophase_h);
        pu.line(Line::new(PlotPoints::from(acro.clone())).color(th.role("acrophase")).width(1.2f32));
        pu.points(Points::new(PlotPoints::from(acro)).radius(4.0f32).color(th.role("acrophase")).name("Acrophase"));
        pu.points(Points::new(PlotPoints::from(pts(|d| d.onset_h))).radius(3.5f32).shape(MarkerShape::Up).color(th.role("onset")).name("Onset"));
        pu.points(Points::new(PlotPoints::from(pts(|d| d.offset_h))).radius(3.5f32).shape(MarkerShape::Down).color(th.role("offset")).name("Offset"));
        if let Some(s) = split {
            pu.vline(VLine::new(s - 0.5).color(th.muted).style(LineStyle::Dashed { length: 6.0 }).name(format!("Split (day {s})")));
        }
    });
    let opts = PlotOpts { height: 220.0, link_x: Some("shift"), ..Default::default() };
    themed_plot(ui, cx, &format!("shift-cum-{}", ch.name), "Day Index", "Cumulative Shift (h)", opts, |pu| {
        pu.line(Line::new(PlotPoints::from(pts(|d| d.cumulative_shift_h))).color(th.role("delay")).width(2.0f32).name("Cumulative shift"));
        pu.hline(HLine::new(0.0).color(th.grid));
    });
    let bars: Vec<Bar> = ch
        .daily
        .iter()
        .filter(|d| d.shift_h.is_finite())
        .map(|d| Bar::new(d.day as f64, d.shift_h).width(0.7).fill(if d.shift_h >= 0.0 { th.role("delay") } else { th.role("advance") }))
        .collect();
    let opts = PlotOpts { height: 200.0, link_x: Some("shift"), legend: false, ..Default::default() };
    themed_plot(ui, cx, &format!("shift-daily-{}", ch.name), "Day Index", "Day-to-Day Shift Δφ (h)", opts, |pu| {
        pu.bar_chart(BarChart::new(bars).name("Δφ (+ delay / − advance)"));
        pu.hline(HLine::new(0.0).color(th.grid));
    });
}

// ------------------------------------------------------------------------------------------------
// 24 h profile (non-parametric)
// ------------------------------------------------------------------------------------------------

pub fn hourly_profile(ui: &mut egui::Ui, cx: &ChartCtx, ch: &ChannelReport) {
    let th = cx.theme;
    let Some(np) = &ch.non_param else { return };
    let ymax = np.profile.iter().copied().fold(0.0, f64::max) * 1.1 + 1e-9;
    let bars: Vec<Bar> = (0..24).map(|h| Bar::new(h as f64 + 0.5, np.profile[h]).width(0.85).fill(channel_color(th, ch))).collect();
    let opts = PlotOpts { height: 280.0, include_y: vec![0.0], include_x: vec![0.0, 24.0], ..Default::default() };
    themed_plot(ui, cx, &format!("prof-{}", ch.name), "Hour of Day", &value_label(ch), opts, |pu| {
        let span = |start: f64, len: f64, c: Color32, name: &str, pu: &mut PlotUi| {
            let end = start + len;
            if end <= 24.0 {
                pu.polygon(vspan(start, end, 0.0, ymax, c, name));
            } else {
                pu.polygon(vspan(start, 24.0, 0.0, ymax, c, name));
                pu.polygon(vspan(0.0, end - 24.0, 0.0, ymax, c, name));
            }
        };
        let a = if th.is_dark() { 0.06 } else { 0.13 };
        span(np.m10_onset, 10.0, alpha(th.role("highlight"), a), "M10 window", pu);
        span(np.l5_onset, 5.0, alpha(th.role("advance"), a), "L5 window", pu);
        pu.bar_chart(BarChart::new(bars).name("Hourly mean"));
        pu.text(Text::new(PlotPoint::new(np.m10_onset + 5.0, ymax * 0.97), RichText::new("M10").color(th.foreground)));
    });
}
