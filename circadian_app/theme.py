"""Chart colour themes: a background plus named colour roles shared by every Matplotlib and Plotly figure.

Plotting code asks for colours by role (`current_theme().activity`, `.onset`, ...) instead of hard-coding
hex values, so a single theme change recolours every chart, the Excel embeds, and CLI PNGs.

The active theme lives in a ContextVar. Streamlit runs each browser session's script in its own thread,
so one viewer's choice never leaks into another's charts.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional

# Colour roles (everything except the background / foreground-derived ones) that palettes define
ROLE_LABELS: Dict[str, str] = {
    "activity": "Activity",
    "light": "Ambient light",
    "fit": "Fits & detected peaks",
    "accent": "Secondary lines (1-harmonic fit, Lomb-Scargle)",
    "acrophase": "Acrophase markers",
    "onset": "Onset markers",
    "offset": "Offset markers",
    "mesor": "MESOR line",
    "chi2": "Chi-square periodogram",
    "sweep": "Cosinor R² sweep",
    "threshold": "Significance threshold",
    "delay": "Phase delay / post-split",
    "advance": "Phase advance / baseline",
    "highlight": "12 h rhythm & 2nd peak",
    "harmonic": "τ/2 harmonic marker",
    "band_ultradian": "Ultradian band",
    "band_circadian": "Circadian band",
    "band_infradian": "Infradian band",
}


@dataclass(frozen=True)
class PlotTheme:
    background: str = "#ffffff"
    foreground: str = "#0f172a"  # text, axes, dark data lines (derived from background unless overridden)
    muted: str = "#64748b"  # reference lines, faint scatter, non-significant bars
    grid: str = "#d9dee7"  # baselines and gridlines
    activity: str = "#1e293b"
    light: str = "#d97706"
    fit: str = "#dc2626"
    accent: str = "#2563eb"
    acrophase: str = "#2563eb"
    onset: str = "#16a34a"
    offset: str = "#dc2626"
    mesor: str = "#059669"
    chi2: str = "#0f766e"
    sweep: str = "#7c3aed"
    threshold: str = "#f59e0b"
    delay: str = "#ea580c"
    advance: str = "#0284c7"
    highlight: str = "#f97316"
    harmonic: str = "#0ea5e9"
    band_ultradian: str = "#a855f7"
    band_circadian: str = "#3b82f6"
    band_infradian: str = "#22c55e"
    # Axis-label styling. None = leave the chart's own default (colour falls back to `foreground`).
    label_font_family: Optional[str] = None
    label_font_size: Optional[float] = None
    label_bold: bool = False
    x_label_color: Optional[str] = None
    y_label_color: Optional[str] = None
    label_text: tuple = ()  # ((original label, replacement), ...)

    @property
    def label_text_map(self) -> Dict[str, str]:
        return {o: n for o, n in self.label_text if o and n}

    @property
    def is_dark(self) -> bool:
        return relative_luminance(self.background) < 0.35

    @property
    def band_colors(self) -> Dict[str, str]:
        return {"Ultradian": self.band_ultradian, "Circadian": self.band_circadian, "Infradian": self.band_infradian}


BACKGROUND_PRESETS: Dict[str, str] = {
    "White": "#ffffff",
    "Light grey": "#f1f5f9",
    "Cream": "#fdf8ee",
    "Dark": "#0f172a",
    "Black": "#000000",
}

# Palettes only list the roles they change from PlotTheme's defaults
PALETTES: Dict[str, Dict[str, str]] = {
    "Default": {},
    "Colourblind-safe (Okabe-Ito)": {
        "activity": "#1a1a1a", "light": "#E69F00", "fit": "#D55E00", "accent": "#0072B2", "acrophase": "#0072B2",
        "onset": "#009E73", "offset": "#D55E00", "mesor": "#009E73", "chi2": "#009E73", "sweep": "#CC79A7",
        "threshold": "#E69F00", "delay": "#D55E00", "advance": "#56B4E9", "highlight": "#E69F00", "harmonic": "#56B4E9",
        "band_ultradian": "#CC79A7", "band_circadian": "#0072B2", "band_infradian": "#009E73",
    },
    "High contrast": {
        "activity": "#000000", "light": "#ff8c00", "fit": "#e00000", "accent": "#0000e0", "acrophase": "#0000e0",
        "onset": "#008a00", "offset": "#e00000", "mesor": "#008a00", "chi2": "#007070", "sweep": "#8000c0",
        "threshold": "#ff8c00", "delay": "#e00000", "advance": "#0000e0", "highlight": "#ff8c00", "harmonic": "#00a0e0",
        "band_ultradian": "#c000c0", "band_circadian": "#0000ff", "band_infradian": "#00b000",
    },
    "Pastel": {
        "activity": "#5b6b8c", "light": "#f4b860", "fit": "#e57373", "accent": "#64b5f6", "acrophase": "#64b5f6",
        "onset": "#81c784", "offset": "#e57373", "mesor": "#4db6ac", "chi2": "#4db6ac", "sweep": "#b39ddb",
        "threshold": "#ffb74d", "delay": "#ff8a65", "advance": "#4fc3f7", "highlight": "#ffb74d", "harmonic": "#4fc3f7",
        "band_ultradian": "#ce93d8", "band_circadian": "#90caf9", "band_infradian": "#a5d6a7",
    },
    "Greyscale (print)": {
        "activity": "#111111", "light": "#9a9a9a", "fit": "#000000", "accent": "#555555", "acrophase": "#333333",
        "onset": "#444444", "offset": "#000000", "mesor": "#666666", "chi2": "#333333", "sweep": "#555555",
        "threshold": "#888888", "delay": "#222222", "advance": "#777777", "highlight": "#666666", "harmonic": "#999999",
        "band_ultradian": "#bbbbbb", "band_circadian": "#888888", "band_infradian": "#555555",
    },
}


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return tuple(int(c[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def relative_luminance(color: str) -> float:
    def lin(v: float) -> float:
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in _hex_to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a: str, b: str) -> float:
    la, lb = sorted((relative_luminance(a), relative_luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def blend(color: str, other: str, t: float) -> str:
    """Mix `t` of `other` into `color` (0 = color, 1 = other)."""
    a, b = _hex_to_rgb(color), _hex_to_rgb(other)
    return "#" + "".join(f"{int(round((x + (y - x) * t) * 255)):02x}" for x, y in zip(a, b))


def to_rgba(color: str, alpha: float) -> str:
    r, g, b = (int(round(v * 255)) for v in _hex_to_rgb(color))
    return f"rgba({r},{g},{b},{alpha})"


def build_theme(
    background: str = "#ffffff",
    palette: str = "Default",
    overrides: Optional[Dict[str, str]] = None,
    label_style: Optional[Dict[str, object]] = None,
) -> PlotTheme:
    """Combine a background, a named palette, and per-role overrides into a readable theme.

    Foreground, muted and grid colours are derived from the background. Roles that would be nearly
    invisible on it (e.g. dark activity bars on a black background) are swapped for the foreground,
    unless the user picked that colour explicitly.
    """
    overrides = dict(overrides or {})
    dark = relative_luminance(background) < 0.35
    fg = "#e2e8f0" if dark else "#0f172a"
    theme = replace(
        PlotTheme(),
        background=background,
        foreground=fg,
        muted=blend(fg, background, 0.45),
        grid=blend(fg, background, 0.82),
        **PALETTES.get(palette, {}),
    )
    role_names = {f.name for f in fields(PlotTheme)}
    auto_fix = {}
    for role in ROLE_LABELS:
        if role in overrides:
            continue
        if contrast_ratio(getattr(theme, role), background) < 1.8:
            auto_fix[role] = fg
    theme = replace(theme, **auto_fix)
    theme = replace(theme, **{k: v for k, v in overrides.items() if k in ROLE_LABELS})
    if label_style:
        allowed = {"label_font_family", "label_font_size", "label_bold", "x_label_color", "y_label_color", "label_text"}
        theme = replace(theme, **{k: v for k, v in label_style.items() if k in allowed})
    return theme


# Fixed-text axis labels the charts use (labels that embed the channel/unit are added once they have been drawn)
KNOWN_AXIS_LABELS = [
    "Clock Time (Double-Plotted 48 h)", "Clock Time (24 h)", "Clock Time (hours)", "Recording Day", "Hour of Day",
    "Phase / Clock Hour (h)", "Period τ (hours)", "Period (hours)", "Normalized Power", "Qp", "R²",
    "R² (variance explained)", "Day Index", "Clock Time", "Cumulative Shift (h)", "Day-to-Day Shift Δφ (h)",
]

_LABEL_FONT_CANDIDATES = [
    "Arial", "Calibri", "Cambria", "Comic Sans MS", "Consolas", "Courier New", "DejaVu Sans", "DejaVu Sans Mono",
    "DejaVu Serif", "Georgia", "Helvetica", "Palatino Linotype", "Segoe UI", "Tahoma", "Times New Roman",
    "Trebuchet MS", "Verdana",
]


def _installed_label_fonts() -> list:
    try:
        from matplotlib import font_manager

        installed = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover - font discovery is best-effort
        installed = set()
    found = [f for f in _LABEL_FONT_CANDIDATES if f in installed]
    return found or ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]


AVAILABLE_LABEL_FONTS = _installed_label_fonts()

# Axis labels seen while rendering charts (so the app can offer them for renaming)
_SEEN_LABELS: ContextVar[Optional[set]] = ContextVar("circadian_seen_axis_labels", default=None)


def reset_seen_labels() -> None:
    _SEEN_LABELS.set(set())


def record_axis_label(text: str) -> None:
    seen = _SEEN_LABELS.get()
    if seen is not None and text:
        seen.add(text)


def seen_axis_labels() -> list:
    return sorted(_SEEN_LABELS.get() or [])


DEFAULT_THEME = PlotTheme()
_ACTIVE: ContextVar[PlotTheme] = ContextVar("circadian_plot_theme", default=DEFAULT_THEME)


def set_plot_theme(theme: PlotTheme) -> None:
    _ACTIVE.set(theme)


def current_theme() -> PlotTheme:
    return _ACTIVE.get()
