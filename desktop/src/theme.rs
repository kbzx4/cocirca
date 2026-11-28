//! Chart colour themes: a background plus named colour roles, and axis-label styling.

use eframe::egui::Color32;
use std::collections::BTreeMap;

pub const ROLES: [(&str, &str); 18] = [
    ("activity", "Activity"),
    ("light", "Ambient light"),
    ("fit", "Fits & detected peaks"),
    ("accent", "Secondary lines (1-harmonic fit, Lomb-Scargle)"),
    ("acrophase", "Acrophase markers"),
    ("onset", "Onset markers"),
    ("offset", "Offset markers"),
    ("mesor", "MESOR line"),
    ("chi2", "Chi-square periodogram"),
    ("sweep", "Cosinor R² sweep"),
    ("threshold", "Significance threshold"),
    ("delay", "Phase delay / post-split"),
    ("advance", "Phase advance / baseline"),
    ("highlight", "12 h rhythm & 2nd peak"),
    ("harmonic", "τ/2 harmonic marker"),
    ("band_ultradian", "Ultradian band"),
    ("band_circadian", "Circadian band"),
    ("band_infradian", "Infradian band"),
];

const DEFAULT_ROLES: [&str; 18] = [
    "#1e293b", "#d97706", "#dc2626", "#2563eb", "#2563eb", "#16a34a", "#dc2626", "#059669", "#0f766e", "#7c3aed", "#f59e0b", "#ea580c", "#0284c7", "#f97316", "#0ea5e9", "#a855f7", "#3b82f6", "#22c55e",
];

pub const PALETTES: [(&str, [&str; 18]); 5] = [
    ("Default", DEFAULT_ROLES),
    (
        "Colourblind-safe (Okabe-Ito)",
        ["#1a1a1a", "#E69F00", "#D55E00", "#0072B2", "#0072B2", "#009E73", "#D55E00", "#009E73", "#009E73", "#CC79A7", "#E69F00", "#D55E00", "#56B4E9", "#E69F00", "#56B4E9", "#CC79A7", "#0072B2", "#009E73"],
    ),
    (
        "High contrast",
        ["#000000", "#ff8c00", "#e00000", "#0000e0", "#0000e0", "#008a00", "#e00000", "#008a00", "#007070", "#8000c0", "#ff8c00", "#e00000", "#0000e0", "#ff8c00", "#00a0e0", "#c000c0", "#0000ff", "#00b000"],
    ),
    (
        "Pastel",
        ["#5b6b8c", "#f4b860", "#e57373", "#64b5f6", "#64b5f6", "#81c784", "#e57373", "#4db6ac", "#4db6ac", "#b39ddb", "#ffb74d", "#ff8a65", "#4fc3f7", "#ffb74d", "#4fc3f7", "#ce93d8", "#90caf9", "#a5d6a7"],
    ),
    (
        "Greyscale (print)",
        ["#111111", "#9a9a9a", "#000000", "#555555", "#333333", "#444444", "#000000", "#666666", "#333333", "#555555", "#888888", "#222222", "#777777", "#666666", "#999999", "#bbbbbb", "#888888", "#555555"],
    ),
];

pub const BACKGROUNDS: [(&str, &str); 5] = [("White", "#ffffff"), ("Light grey", "#f1f5f9"), ("Cream", "#fdf8ee"), ("Dark", "#0f172a"), ("Black", "#000000")];

/// Windows font files for the axis-label font picker: (family, regular, bold).
pub const FONT_FILES: [(&str, &str, &str); 12] = [
    ("Arial", "arial.ttf", "arialbd.ttf"),
    ("Calibri", "calibri.ttf", "calibrib.ttf"),
    ("Comic Sans MS", "comic.ttf", "comicbd.ttf"),
    ("Consolas", "consola.ttf", "consolab.ttf"),
    ("Courier New", "cour.ttf", "courbd.ttf"),
    ("Georgia", "georgia.ttf", "georgiab.ttf"),
    ("Palatino Linotype", "pala.ttf", "palab.ttf"),
    ("Segoe UI", "segoeui.ttf", "segoeuib.ttf"),
    ("Tahoma", "tahoma.ttf", "tahomabd.ttf"),
    ("Times New Roman", "times.ttf", "timesbd.ttf"),
    ("Trebuchet MS", "trebuc.ttf", "trebucbd.ttf"),
    ("Verdana", "verdana.ttf", "verdanab.ttf"),
];

pub fn fonts_dir() -> std::path::PathBuf {
    std::env::var_os("WINDIR").map(std::path::PathBuf::from).unwrap_or_else(|| "C:\\Windows".into()).join("Fonts")
}

/// Fonts whose regular file is installed.
pub fn available_fonts() -> Vec<&'static str> {
    let dir = fonts_dir();
    FONT_FILES.iter().filter(|f| dir.join(f.1).exists()).map(|f| f.0).collect()
}

pub fn hex(c: &str) -> Color32 {
    let c = c.trim_start_matches('#');
    let p = |i: usize| u8::from_str_radix(&c[i..i + 2], 16).unwrap_or(0);
    Color32::from_rgb(p(0), p(2), p(4))
}

pub fn to_hex(c: Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", c.r(), c.g(), c.b())
}

fn lum(c: Color32) -> f64 {
    let lin = |v: u8| {
        let v = v as f64 / 255.0;
        if v <= 0.03928 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
}

pub fn contrast(a: Color32, b: Color32) -> f64 {
    let (la, lb) = (lum(a), lum(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

pub fn blend(a: Color32, b: Color32, t: f32) -> Color32 {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgb(m(a.r(), b.r()), m(a.g(), b.g()), m(a.b(), b.b()))
}

#[derive(Clone, Debug, PartialEq)]
pub struct LabelStyle {
    /// None = egui's default proportional font
    pub font: Option<String>,
    /// 0 = chart default
    pub size: f32,
    pub bold: bool,
    pub x_color: Option<Color32>,
    pub y_color: Option<Color32>,
    pub renames: Vec<(String, String)>,
}

impl Default for LabelStyle {
    fn default() -> Self {
        LabelStyle { font: None, size: 0.0, bold: false, x_color: None, y_color: None, renames: Vec::new() }
    }
}

#[derive(Clone, Debug)]
pub struct Theme {
    pub background: Color32,
    pub foreground: Color32,
    pub muted: Color32,
    pub grid: Color32,
    roles: [Color32; 18],
    pub label: LabelStyle,
}

impl Theme {
    pub fn role(&self, name: &str) -> Color32 {
        ROLES.iter().position(|r| r.0 == name).map(|i| self.roles[i]).unwrap_or(self.foreground)
    }
    pub fn is_dark(&self) -> bool {
        lum(self.background) < 0.35
    }
    pub fn band(&self, band: &str) -> Color32 {
        match band {
            "Ultradian" => self.role("band_ultradian"),
            "Circadian" => self.role("band_circadian"),
            _ => self.role("band_infradian"),
        }
    }
    /// Renamed axis label text, if the user set one.
    pub fn label_text(&self, original: &str) -> String {
        self.label.renames.iter().find(|(o, n)| o == original && !n.is_empty()).map(|(_, n)| n.clone()).unwrap_or_else(|| original.to_string())
    }
    pub fn x_label_color(&self) -> Color32 {
        self.label.x_color.unwrap_or(self.foreground)
    }
    pub fn y_label_color(&self) -> Color32 {
        self.label.y_color.unwrap_or(self.foreground)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThemeSettings {
    pub background_preset: String,
    pub custom_background: Color32,
    pub palette: String,
    pub overrides: BTreeMap<String, Color32>,
    pub label: LabelStyle,
}

impl Default for ThemeSettings {
    fn default() -> Self {
        ThemeSettings { background_preset: "White".into(), custom_background: hex("#202020"), palette: "Default".into(), overrides: BTreeMap::new(), label: LabelStyle::default() }
    }
}

impl ThemeSettings {
    pub fn background(&self) -> Color32 {
        BACKGROUNDS.iter().find(|b| b.0 == self.background_preset).map(|b| hex(b.1)).unwrap_or(self.custom_background)
    }

    /// Combine background, palette and overrides. Roles nearly invisible on the background are swapped
    /// for the foreground colour unless the user picked them explicitly.
    pub fn build(&self) -> Theme {
        let bg = self.background();
        let dark = lum(bg) < 0.35;
        let fg = if dark { hex("#e2e8f0") } else { hex("#0f172a") };
        let pal = PALETTES.iter().find(|p| p.0 == self.palette).map(|p| p.1).unwrap_or(DEFAULT_ROLES);
        let mut roles = [Color32::BLACK; 18];
        for (i, (name, _)) in ROLES.iter().enumerate() {
            roles[i] = match self.overrides.get(*name) {
                Some(&c) => c,
                None => {
                    let c = hex(pal[i]);
                    if contrast(c, bg) < 1.8 {
                        fg
                    } else {
                        c
                    }
                }
            };
        }
        Theme { background: bg, foreground: fg, muted: blend(fg, bg, 0.45), grid: blend(fg, bg, 0.82), roles, label: self.label.clone() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dark_background_fixes_invisible_roles() {
        let s = ThemeSettings { background_preset: "Black".into(), ..Default::default() };
        let t = s.build();
        assert!(t.is_dark());
        assert_eq!(t.role("activity"), t.foreground);
        assert!(contrast(t.role("light"), t.background) >= 1.8);
    }
}
