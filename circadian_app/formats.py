"""Chronobiology file formats: format sniffing and regex-driven parsing.

Mirrors the desktop app's importer (desktop/src/import.rs). Recognised besides any delimited table with a
time column:

* Actigraphy: ActiGraph ActiLife CSV, Philips Respironics Actiware, Condor ActTrust, Actiwatch / CamNtech
  ``.awd`` (also Actimetrics ClockLab), GENEActiv CSV, Axivity CSV, MotionWare-style exports; ActiGraph ``.agd`` (SQLite)
* Locomotor activity: TriKinetics DAM monitor files (one subject per channel), wheel / beam-break counts
* Loggers: HOBO (HOBOware CSV), iButton / Thermochron (OneWireViewer)
* Heart: RR / IBI / NN interval files, Polar CSV, Empatica E4 (ACC, TEMP, HR, EDA, IBI)
* Phones & wearables: Phyphox, Sensor Logger, Apple Health ``export.xml``, Fitbit JSON
* Wide tables: plate-reader / bioluminescence columns (one subject per column) and ZT / CT time-course
  tables with time points as column headers (one subject per row; JTK_CYCLE / MetaCycle layout)
* Containers & encodings: .zip, .gz, .xlsx / .xls / .ods, .json / .jsonl, .xml; UTF-8, UTF-16, cp1252
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
import sqlite3
import tempfile
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROLES = ["activity", "accel_x", "accel_y", "accel_z", "light", "heart_rate", "temperature", "signal"]
COL_ROLES = ["ignore", "time", "date", *ROLES, "rr", "mask"]
ROLE_LABELS = {
    "ignore": "— ignore —",
    "time": "Time / timestamp",
    "date": "Date (separate column)",
    "activity": "Activity / counts",
    "accel_x": "Accelerometer X",
    "accel_y": "Accelerometer Y",
    "accel_z": "Accelerometer Z",
    "light": "Light (lux)",
    "heart_rate": "Heart rate",
    "temperature": "Temperature",
    "signal": "Other signal",
    "rr": "RR / IBI interval → heart rate",
    "mask": "Invalid-data flag (off-wrist)",
}
LABEL_TO_ROLE = {v: k for k, v in ROLE_LABELS.items()}


# ================================================================================================
# Data classes
# ================================================================================================


@dataclass
class Source:
    name: str
    data: bytes
    start_hint: Optional[datetime] = None


@dataclass
class ImportOptions:
    date_order: str = "auto"  # auto | dmy | mdy | ymd
    start: Optional[datetime] = None  # for relative times / files without time stamps
    time_unit: Optional[str] = None  # ms | s | min | h | d
    interval_s: Optional[float] = None  # files without time stamps
    header_line: Optional[int] = None  # 0-based line with the column names
    roles: Dict[int, str] = field(default_factory=dict)
    layout: str = "auto"  # auto | channels | subjects


@dataclass
class Channel:
    role: str
    name: str
    unit: str
    values: np.ndarray
    sum: bool = False


@dataclass
class Subject:
    name: str
    format: str
    t: pd.DatetimeIndex
    channels: List[Channel]
    notes: List[str] = field(default_factory=list)
    accel_linear: bool = False
    rr_counts: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        # pandas >= 2 parses to s / ms / us resolution depending on the text; keep one unit (ns) so integer
        # views (.asi8) of different files line up when they are merged.
        self.t = pd.DatetimeIndex(self.t).as_unit("ns")

    def channel(self, role: str) -> Optional[Channel]:
        return next((c for c in self.channels if c.role == role and np.isfinite(c.values).any()), None)

    def describe(self) -> List[str]:
        out = []
        roles = {c.role for c in self.channels}
        if {"accel_x", "accel_y", "accel_z"} <= roles:
            out.append("3-axis linear acceleration" if self.accel_linear else "3-axis acceleration")
        for c in self.channels:
            if c.role.startswith("accel_"):
                continue
            out.append(f"{c.name} ({c.unit})" if c.unit else c.name)
        return out

    @property
    def signal_channels(self) -> List[Channel]:
        """Channels that can be analysed as the main signal (everything except light and raw axes)."""
        prio = {"activity": 0, "heart_rate": 1, "temperature": 2, "signal": 3}
        out = sorted((c for c in self.channels if c.role in prio), key=lambda c: prio[c.role])
        if self.channel("accel_x") is not None and not any(c.role == "activity" for c in out):
            out.insert(0, Channel("activity", "Activity (from accelerometer)", "milli-g", np.array([])))
        return out


@dataclass
class ColumnInfo:
    name: str
    role: str
    example: str


@dataclass
class ParsedFile:
    source: str
    format: str
    subjects: List[Subject]
    notes: List[str] = field(default_factory=list)
    preview: List[str] = field(default_factory=list)
    header_line: Optional[int] = None
    data_line: int = 0
    columns: List[ColumnInfo] = field(default_factory=list)
    time_desc: str = ""
    relative_time: bool = False
    numeric_time: bool = False
    needs_interval: bool = False
    date_ambiguous: bool = False
    can_layout: bool = False
    layout: str = "channels"
    rows: int = 0

    def summary(self) -> str:
        n = len(self.subjects)
        t0 = min((s.t[0] for s in self.subjects if len(s.t)), default=None)
        t1 = max((s.t[-1] for s in self.subjects if len(s.t)), default=None)
        dur = fmt_duration((t1 - t0).total_seconds()) if t0 is not None and t1 is not None and t1 > t0 else "—"
        step = ""
        if self.subjects and len(self.subjects[0].t) > 1:
            d = np.diff(self.subjects[0].t[:5001].asi8) / 1e9
            d = d[d > 0]
            if len(d):
                step = f" · every {fmt_duration(float(np.median(d)))}"
        return f"{n} subject{'s' if n != 1 else ''} · {self.rows:,} rows · {dur}{step}"


def fmt_duration(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 120:
        return f"{s:.3f}".rstrip("0").rstrip(".") + " s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    if s < 3 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


# ================================================================================================
# Reading / decoding
# ================================================================================================

BINARY_HINTS = {
    ".gt3x": "ActiGraph .gt3x is binary: export a CSV from ActiLife (or open the .agd file).",
    ".cwa": "Axivity .cwa is binary: export CSV with OmGui (Export Raw CSV).",
    ".bin": "GENEActiv .bin: convert with GENEActiv PC software (Convert → CSV).",
    ".mtn": "MotionWatch .mtn: export CSV (or .awd) from MotionWare.",
    ".edf": "EDF recordings: export the channel of interest to CSV (e.g. EDFbrowser).",
}
_SKIP = re.compile(r"(?i)(^|/)(__macosx/|\.|license|licence|readme|sha\d*sums|md5sums|checksums?)|\.(pdf|md|html?|png|jpe?g|gif|docx?|pptx?|hea|log)$")


def decode(b: bytes) -> str:
    if b.startswith(b"\xef\xbb\xbf"):
        return b[3:].decode("utf-8", errors="replace")
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return b.decode("utf-16", errors="replace")
    probe = b[:4096]
    if len(probe) >= 8:
        odd = probe[1::2].count(0)
        even = probe[0::2].count(0)
        if odd * 3 > len(probe) // 2 and even * 10 < len(probe) // 2:
            return b.decode("utf-16-le", errors="replace")
        if even * 3 > len(probe) // 2 and odd * 10 < len(probe) // 2:
            return b.decode("utf-16-be", errors="replace")
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("cp1252", errors="replace")


def _phyphox_start(b: bytes) -> Optional[datetime]:
    try:
        df = pd.read_csv(io.StringIO(decode(b)), sep=None, engine="python")
    except Exception:
        return None
    cols = {c.lower().strip(): c for c in df.columns}
    ev = next((cols[c] for c in cols if "event" in c), None)
    sys_c = next((cols[c] for c in cols if c.startswith("system time") and "text" not in c), None)
    exp = next((cols[c] for c in cols if "experiment time" in c), None)
    if sys_c is None:
        return None
    row = df[df[ev].astype(str).str.upper().str.contains("START")].iloc[0] if ev is not None and df[ev].astype(str).str.upper().str.contains("START").any() else df.iloc[0]
    off = float(row[exp]) if exp is not None and pd.notna(row[exp]) else 0.0
    return datetime.fromtimestamp(float(row[sys_c]) - off)


def expand(name: str, data: bytes, depth: int = 0, start_hint: Optional[datetime] = None) -> List[Source]:
    """Expand .zip archives and .gz files into their members."""
    low = name.lower()
    for ext, hint in BINARY_HINTS.items():
        if low.endswith(ext):
            raise ValueError(f"{name}: {hint}")
    if depth > 3:
        return [Source(name, data, start_hint)]
    if low.endswith(".gz") and data[:2] == b"\x1f\x8b":
        return expand(name[:-3], gzip.decompress(data), depth + 1, start_hint)
    if low.endswith(".zip"):
        out: List[Source] = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            meta = next((n for n in zf.namelist() if n.lower().endswith("meta/time.csv") or n.lower() == "time.csv"), None)
            hint = start_hint or (_phyphox_start(zf.read(meta)) if meta else None)
            for m in zf.namelist():
                if m.endswith("/") or "meta/" in m.lower() or _SKIP.search(m):
                    continue
                out.extend(expand(m.rsplit("/", 1)[-1], zf.read(m), depth + 1, hint))
        if not out:
            raise ValueError(f"{name}: the archive contains no data files.")
        return out
    return [Source(name, data, start_hint)]


# ================================================================================================
# Date-time helpers
# ================================================================================================

_MONTHS = {
    "jan": 1, "ene": 1, "gen": 1, "feb": 2, "fev": 2, "fév": 2, "mar": 3, "mär": 3, "mrz": 3, "apr": 4, "abr": 4, "avr": 4,
    "may": 5, "mai": 5, "mag": 5, "jun": 6, "jui": 6, "jul": 7, "lug": 7, "aug": 8, "ago": 8, "aoû": 8, "sep": 9, "set": 9,
    "oct": 10, "out": 10, "okt": 10, "ott": 10, "nov": 11, "dec": 12, "dez": 12, "dic": 12, "déc": 12,
}
_NUM = re.compile(r"\d+")
_TOD = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?(?:[.,:](\d{1,6}))?\s*([AaPp]\.?[Mm]\.?)?\s*$")
_TZ_TAIL = re.compile(r"(?:\s*(?:GMT|UTC)?\s*[+-]\d{2}:?\d{2}|Z|\s(?:GMT|UTC))\s*$")
_GENEA_MS = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}):(\d{3})$")


def _has_month_word(s: str) -> bool:
    return any(w[:3].lower() in _MONTHS for w in re.findall(r"[^\W\d_]+", s))


def detect_order(samples: Sequence[str]) -> Tuple[str, bool]:
    """'dmy' | 'mdy' | 'ymd' plus an ambiguity flag (all day numbers ≤ 12 and both readings plausible)."""
    dmy = mdy = needs = 0
    cand = []
    for s in samples:
        if _has_month_word(s):
            continue
        nums = _NUM.findall(s)
        if len(nums) < 3 or len(nums[0]) == 4 or ":" in s.split(nums[2], 1)[0][-1:]:
            continue
        needs += 1
        cand.append(s)
        dmy += int(nums[0]) > 12
        mdy += int(nums[1]) > 12
    if needs == 0:
        return "ymd", False
    if dmy and not mdy:
        return "dmy", False
    if mdy and not dmy:
        return "mdy", False

    def score(dayfirst: bool) -> Tuple[int, float]:
        ts = pd.to_datetime(pd.Series(cand), dayfirst=dayfirst, errors="coerce", format="mixed").dropna()
        if ts.empty:
            return (10**9, float("inf"))
        back = int((ts.diff().dt.total_seconds() < 0).sum())
        return back, (ts.max() - ts.min()).total_seconds()

    a, b = score(True), score(False)
    if a < b:
        return "dmy", a == b
    return "mdy", a == b


def parse_datetimes(s: pd.Series, order: str) -> pd.Series:
    """Text date-times → datetime64 (local wall-clock time; time-zone suffixes are dropped)."""
    v = s.astype(str).str.strip().str.strip('"').str.strip()
    # pattern strings (not compiled re objects) let pandas use pyarrow's vectorised RE2 instead of a Python
    # re.sub per row (2.5M rows: 6.7 s -> 0.7 s)
    v = v.str.replace(_GENEA_MS.pattern, r"\1.\2", regex=True).str.replace(_TZ_TAIL.pattern, "", regex=True)
    kw = dict(dayfirst=order == "dmy", yearfirst=order == "ymd", errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = pd.to_datetime(v, **kw)
        if out.notna().mean() < 0.8:
            out = pd.to_datetime(v, format="mixed", **kw)
    return out


def parse_tod(s: pd.Series) -> pd.Series:
    """'hh:mm(:ss)( AM/PM)' → seconds since midnight (NaN when not a clock time)."""
    m = s.astype(str).str.strip().str.strip('"').str.extract(_TOD)
    h = pd.to_numeric(m[0], errors="coerce")
    mi = pd.to_numeric(m[1], errors="coerce")
    sec = pd.to_numeric(m[2], errors="coerce").fillna(0)
    frac = m[3].fillna("0")
    fr = pd.to_numeric(frac, errors="coerce").fillna(0) / (10.0 ** frac.str.len().clip(lower=1))
    ap = m[4].fillna("").str.lower().str[0]
    h = h.where(~((ap == "p") & (h < 12)), h + 12).where(~((ap == "a") & (h == 12)), 0)
    return h * 3600 + mi * 60 + sec + fr


def _unix_local(seconds: np.ndarray) -> pd.DatetimeIndex:
    t = pd.to_datetime(seconds, unit="s", utc=True).tz_convert(datetime.now().astimezone().tzinfo)
    return pd.DatetimeIndex(t.tz_localize(None))


def _today() -> datetime:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


# ================================================================================================
# Column-name catalogue (same rules as the desktop app)
# ================================================================================================


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


CAT = {
    "time_abs": _rx(r"^(timestamp|time ?stamp|date ?time|date/time|datetime|date_time|utc|local ?time|time ?\(utc\)|date time.*|record(ed)? ?time|sample ?time)$|^date\s*time|timestamp"),
    "time_rel": _rx(r"^(time|t|elapsed|elapsed ?time|seconds|secs?|ms|millis|minutes?|mins?|hours?|hrs?|days?|zt|ct|circadian ?time|zeitgeber ?time|time ?\((s|sec|seconds|ms|min|minutes|h|hr|hrs|hours|d|days)\)|time_?(s|ms|min|h|hours|days)|seconds_elapsed|time ?since ?start.*|unix.*|epoch ?time|ts|time ?\[.*\])$"),
    "ignore": _rx(r"^(#|no\.?|n|line|index|idx|row|epoch ?#|record|id|sample ?#|seq|sequence|ms)$|sleep|wake|state|marker|event|button|status|flag|battery|inclinometer|orientation|interval|annotation|comment|note|unit$|quality|confidence|counter"),
    "mask": _rx(r"off[- _]?wrist|non[- _]?wear|nonwear|interval ?status|excluded"),
    "light": _rx(r"^(light|lux|illuminance|lx)$|\blux\b|\(lx\)|illuminance|white ?light|^amb(ient)? ?light|^light ?(level|intensity|lux)?\b|intensity, ?lux|melanopic|photopic|^light\b"),
    "light_ignore": _rx(r"^(red|green|blue|ir|uva|uvb|uv|clear|infrared)[ _]?(light)?\b"),
    "hr": _rx(r"^(hr|bpm|heart ?rate|pulse|pulse ?rate|heartrate|heart_rate)\b|heart[ _]?rate|\(bpm\)|\bbpm\b"),
    "rr": _rx(r"^(rr|rri|ibi|nn|rr ?intervals?|nn ?intervals?)( ?\((ms|s)\)|_ms|_s)?$|inter[- _]?beat|rr[ _-]?interval"),
    "temp": _rx(r"\btemp(erature)?\b|temp_|°\s*[cf]\b|\bdeg ?[cf]\b|\bcbt\b|skin ?temp|core ?temp|body ?temp|wrist ?temp|thermo"),
    "temp_ignore": _rx(r"\b(ext(ernal)?|ambient|room|amb|env(ironment(al)?)?)[ _]?temp"),
    "accel": _rx(r"(?:^|[\s_\-])(?:linear[ _]?)?acc(?:el(?:eration|erometer)?)?[\s_\-(:]*([xyz])\b|^a([xyz])$|^acc_?([xyz])$|^([xyz])[ _]?\(m/s|^([xyz])[ _]?\(g\)|^([xyz])[ _]?axis$|^axis ?([xyz])$"),
    "xyz": _rx(r"^([xyz])$"),
    "axis": _rx(r"^axis ?([123])$"),
    "act_hi": _rx(r"^(activity|act|actv|activity ?counts?|activity ?index|activity ?score|motion ?index|pim|vector ?magnitude|vm|svm|svmgs|enmo|counts?|wheel.*|revolutions|revs|locomot\w*|beam ?breaks?|crossings|movement|motion|actigraphy|zcm|tat|mad)\b|\b(enmo|pim|activity)\b|counts?$|wheel|running"),
    "act_lo": _rx(r"^(pimn|zcmn|tatn|movement ?intensity)$"),
    "steps": _rx(r"^steps?\b|step ?count"),
    "signal": _rx(r"lumin(escence)?|biolum|\brlu\b|photon|\bcps\b|counts?/s|luciferase|\bluc\b|per2|bmal|melatonin|cortisol|conc(entration)?|pg/ml|ng/ml|nmol|µg|\beda\b|gsr|skin ?conductance|spo2|resp(iration)?|glucose|blood ?pressure|systolic|diastolic|expression|signal|intensity|fluorescence|\bvalue\b|\bmean\b"),
    "sample_like": _rx(r"^[a-p] ?\d{1,2}$|^(dish|well|sample|ch|channel|cage|mouse|mice|fly|animal|subj(ect)?|id|plate|tube|m|f|wt|ko|rat|hamster)[ _\-#]*\d+|^\d+$|_rep\d|^r\d+$"),
    "time_label": _rx(r"^\s*(?:(zt|ct|t|time|hr?|hours?)\s*[_\- ]?)?(-?\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hours)?(?:\s*[_\-. ]\s*(?:rep|r|n|s)?\s*\d+)?\s*$"),
    "id_col": _rx(r"^(id|gene|genes|probe|probe ?id|probeset|symbol|gene ?symbol|name|transcript|ensembl.*|accession|feature|cpg|metabolite|protein)$"),
    "unit": re.compile(r"\(([^()]{1,24})\)\s*$|\[([^\[\]]{1,24})\]\s*$|,\s*([^,]{1,12})$"),
}


def header_unit(h: str) -> Optional[str]:
    m = CAT["unit"].search(h)
    if not m:
        return None
    u = (m.group(1) or m.group(2) or m.group(3) or "").strip()
    return u if u and "s/n" not in u.lower() and len(u) <= 16 else None


def header_name(h: str) -> str:
    n = CAT["unit"].sub("", h).strip().rstrip(",").strip()
    return n or h


def time_unit_from_header(h: str) -> Optional[str]:
    low = h.lower()
    has = lambda *w: any(x in low for x in w)  # noqa: E731
    if has("(ms)", "_ms", "millis", "[ms]") or low == "ms":
        return "ms"
    if has("(s)", "(sec", "_s", "[s]", "second", "secs") or low in ("s", "sec", "t"):
        return "s"
    if has("(min", "_min", "minute", "[min]") or low in ("min", "mins"):
        return "min"
    if has("(h)", "(hr", "hour", "_h", "[h]", "zt", "ct", "zeitgeber", "circadian time") or low in ("h", "hr", "hrs"):
        return "h"
    if has("day", "(d)", "_d", "[d]"):
        return "d"
    return None


UNIT_SECONDS = {"ms": 1e-3, "s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}


def auto_role(h: str, numeric_frac: float) -> Optional[str]:
    h = h.strip().strip('"')
    if numeric_frac < 0.6:
        return "mask" if CAT["mask"].search(h) else "ignore"
    if CAT["mask"].search(h):
        return "mask"
    if CAT["rr"].search(h):
        return "rr"
    if CAT["hr"].search(h) and "hrs" not in h.lower():
        return "heart_rate"
    if CAT["light_ignore"].search(h):
        return "ignore"
    if CAT["light"].search(h) and "lumines" not in h.lower():
        return "light"
    if CAT["temp_ignore"].search(h):
        return "ignore"
    if CAT["temp"].search(h):
        return "temperature"
    m = CAT["accel"].search(h)
    if m:
        axis = next(g for g in m.groups() if g).lower()
        return f"accel_{axis}"
    if CAT["act_lo"].search(h):
        return "ignore"
    if CAT["act_hi"].search(h) or CAT["axis"].search(h) or CAT["steps"].search(h):
        return "activity"
    if CAT["signal"].search(h):
        return "signal"
    if CAT["ignore"].search(h):
        return "ignore"
    return None


# ================================================================================================
# Table sniffing
# ================================================================================================

_DELIMS = ["\t", ";", ",", "|", "ws"]
_MISSING = {"", "nan", "na", "n/a", "null", "none", "-", "--", "inf", "-inf"}


def _clean(s: str) -> str:
    return s.strip().strip('"').strip("'").strip()


def _split(line: str, d: str) -> List[str]:
    if d == "ws":
        return line.split()
    if '"' not in line:
        return line.split(d)
    return next(iter(__import__("csv").reader([line], delimiter=d)), [])


def _numberish(s: str) -> bool:
    s = _clean(s)
    if not s:
        return False
    first_ok = s[0].isdigit() or (s[0] in "+-." and len(s) > 1 and (s[1].isdigit() or s[1] == "."))
    return first_ok and all(c.isdigit() or c in " .,:/-+" or c.isalpha() for c in s)


def _data_like(fields: List[str]) -> bool:
    ne = [f for f in fields if _clean(f).lower() not in _MISSING]
    return bool(ne) and sum(_numberish(f) for f in ne) * 10 >= len(ne) * 6


def sniff(lines: List[str], forced_header: Optional[int] = None) -> Optional[Tuple[str, Optional[int], int, int]]:
    """(delimiter, header line, first data line, n columns) of the data table among preamble lines."""
    best = None  # (score, delim, start, ncols)
    for d in _DELIMS:
        if d == "ws" and best is not None:
            break
        info = []
        for ln in lines:
            if not ln.strip():
                info.append((0, False))
                continue
            f = _split(ln, d)
            info.append((len(f), _data_like(f)))
        if forced_header is not None:
            if forced_header + 1 >= len(lines):
                continue
            c = info[forced_header + 1][0]
            run = 0
            for x in info[forced_header + 1:]:
                if x[0] in (c, 0):
                    run += 1
                else:
                    break
            if c >= 2 and (best is None or run > best[0]):
                best = (run, d, forced_header + 1, c)
            continue
        i = 0
        while i < len(info):
            c, dl = info[i]
            if c < 2 or not dl:
                i += 1
                continue
            st = i
            while i < len(info) and info[i][0] == c and info[i][1]:
                i += 1
            score = (i - st) * 4 + min(c, 64)
            if best is None or score > best[0]:
                best = (score, d, st, c)
    if best is None or (forced_header is None and best[0] < 4 * min(len(lines), 3)):
        return None
    _, d, start, c = best
    if forced_header is not None:
        return d, forced_header, start, c
    h = start
    while h > 0 and not lines[h - 1].strip():
        h -= 1
    header = None
    if h > 0:
        f = _split(lines[h - 1], d)
        if len(f) + 1 >= c and len(f) >= 2 and not _data_like(f):
            header = h - 1
    return d, header, start, c


# ================================================================================================
# Format signatures
# ================================================================================================


def _meta(lines: List[str]) -> dict:
    head = "\n".join(lines[:120])
    m: dict = {"format": None, "start": None, "epoch_s": None, "order": None, "positional": None, "by_name": {}, "header_line": None, "notes": []}
    if "ActiGraph" in head:
        m["format"] = "ActiGraph ActiLife CSV"
        fm = re.search(r"date format\s+(\S+)", head, re.I)
        if fm:
            f = fm.group(1)
            m["order"] = "dmy" if f.startswith("d") else "ymd" if f.startswith("y") else "mdy"
        d = re.search(r"^Start Date\s+(\S+)", head, re.M)
        t = re.search(r"^Start Time\s+(\S+)", head, re.M)
        if d and t:
            m["start"] = pd.to_datetime(f"{d.group(1)} {t.group(1)}", dayfirst=m["order"] == "dmy", errors="coerce")
        e = re.search(r"Epoch Period \(hh:mm:ss\)\s+(\d+):(\d+):(\d+)", head)
        if e:
            m["epoch_s"] = int(e.group(1)) * 3600 + int(e.group(2)) * 60 + int(e.group(3)) or None
        if m["epoch_s"] is None:
            hz = re.search(r"at\s+(\d+)\s*Hz", head)
            if hz:
                m["epoch_s"] = 1.0 / float(hz.group(1))
        m["positional"] = [("Axis1", "activity"), ("Axis2", "ignore"), ("Axis3", "ignore"), ("Steps", "ignore"), ("Lux", "light"),
                           ("Inclinometer Off", "ignore"), ("Inclinometer Standing", "ignore"), ("Inclinometer Sitting", "ignore"), ("Inclinometer Lying", "ignore")]
    elif "GENEActiv" in head:
        m["format"] = "GENEActiv CSV"
        m["order"] = "ymd"
        m["positional"] = [("Timestamp", "time"), ("x", "accel_x"), ("y", "accel_y"), ("z", "accel_z"), ("Lux", "light"), ("Button", "ignore"), ("Temperature", "temperature")]
    elif re.search(r"epoch-by-epoch data|off-wrist status|actiware", head, re.I):
        m["format"] = "Philips Respironics Actiware"
        m["by_name"] = {"activity": "activity", "white light": "light", "off-wrist status": "mask", "interval status": "mask", "sleep/wake": "ignore", "marker": "ignore"}
    elif any(ln.strip().upper().startswith("DATE/TIME;") for ln in lines):
        m["format"] = "Condor ActTrust"
        m["order"] = "dmy"
        m["by_name"] = {"pim": "activity", "light": "light", "temperature": "temperature", "ext temperature": "ignore", "amb light": "ignore", "state": "ignore", "event": "ignore", "ms": "ignore"}
    elif lines and re.match(r'^"?name"?[,;]"?sport"?[,;]"?date"?[,;]"?start time', lines[0], re.I):
        m["format"] = "Polar CSV"
        m["order"] = "dmy"
        if len(lines) > 1:
            hs = [_clean(x).lower() for x in re.split(r"[,;]", lines[0])]
            vs = [_clean(x) for x in re.split(r"[,;]", lines[1])]
            get = lambda k: vs[hs.index(k)] if k in hs and hs.index(k) < len(vs) else None  # noqa: E731
            if get("date") and get("start time"):
                m["start"] = pd.to_datetime(f"{get('date')} {get('start time')}", dayfirst=True, errors="coerce")
        m["header_line"] = 2
    elif "Plot Title" in head or "LGR S/N" in head:
        m["format"] = "HOBO logger (HOBOware CSV)"
    elif re.search(r"date/time\s*[,;\t]\s*unit\s*[,;\t]\s*value", head, re.I):
        m["format"] = "iButton / Thermochron (OneWireViewer)"
        m["by_name"] = {"value": "temperature", "unit": "ignore"}
    elif "seconds_elapsed" in head:
        m["format"] = "Sensor Logger"
        m["by_name"] = {"seconds_elapsed": "ignore", "time": "time"}
    elif re.search(r"^\s*\"?time\s*\(s\)", head, re.I | re.M):
        m["format"] = "Phyphox"
    elif re.search(r"motionware|camntech|motionwatch", head, re.I):
        m["format"] = "CamNtech MotionWare export"
    if isinstance(m["start"], pd.Timestamp) and pd.isna(m["start"]):
        m["start"] = None
    return m


# ================================================================================================
# Parsers
# ================================================================================================


EDA_NAMES = ("eda", "gsr", "scl", "skin conductance", "electrodermal activity")


def accel_g_units(median_vm: float) -> Tuple[float, Optional[str]]:
    """How many file units make 1 g, judged from the median vector magnitude of a 3-axis accelerometer
    (about gravity, since a worn device is mostly still). The text names units that are neither g nor m/s^2."""
    m = median_vm
    if not (np.isfinite(m) and m > 0):
        return 9.80665, None
    if 0.5 < m < 2.0:
        return 1.0, None
    if 5.0 < m < 20.0:
        return 9.80665, None
    for s, d in ((64.0, "1/64 g (Empatica E4)"), (256.0, "1/256 g (Axivity raw)"), (1000.0, "milli-g")):
        if abs(m / s - 1.0) < 0.3:
            return s, d
    return m, "unrecognised units, scaled so the resting magnitude is 1 g"


def _rr_subject(name: str, rr_ms: np.ndarray, start: Optional[datetime], times: Optional[pd.DatetimeIndex] = None) -> Subject:
    """Heart rate from RR intervals; rejects intervals outside 250–2500 ms or > 25 % from the local median."""
    rr = np.asarray(rr_ms, dtype=float)
    local = pd.Series(rr).rolling(11, center=True, min_periods=1).median().to_numpy()
    ok = np.isfinite(rr) & (rr >= 250) & (rr <= 2500) & (np.abs(rr - local) <= 0.25 * local)
    base = pd.Timestamp(start or _today())
    if times is None:
        t = base + pd.to_timedelta(np.nancumsum(np.where(np.isfinite(rr) & (rr > 0), rr, 0.0)), unit="ms")
        hours = np.nansum(np.where(rr > 0, rr, 0)) / 3.6e6
    else:
        t = pd.DatetimeIndex(times)
        hours = (t.max() - t.min()).total_seconds() / 3600 if len(t) > 1 else 0.0
    notes = [f"RR intervals: {len(rr):,} beats over {hours:.1f} h; heart rate = 60000 / RR. {int((~ok).sum()):,} intervals ({100 * (~ok).mean():.2f}%) rejected as artifact / ectopic."]
    if start is None and times is None:
        notes.append(f"No recording start time in the file: times start at {base:%Y-%m-%d %H:%M}; clock phases are hours after the recording began.")
    if times is not None:
        # own time stamps: keep every row so other columns stay aligned; rejected beats are missing
        return Subject(name, "RR / inter-beat intervals", t, [Channel("heart_rate", "Heart rate", "bpm", np.where(ok, 60000.0 / np.where(ok, rr, 1.0), np.nan))], notes, rr_counts=(len(rr), int((~ok).sum())))
    return Subject(name, "RR / inter-beat intervals", pd.DatetimeIndex(t[ok]), [Channel("heart_rate", "Heart rate", "bpm", 60000.0 / rr[ok])], notes, rr_counts=(len(rr), int((~ok).sum())))


def _single_column(text: str, name: str, opts: ImportOptions, lines: List[str]) -> ParsedFile:
    rows = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header = None
    if rows and pd.isna(pd.to_numeric(re.split(r"[,;\t]", rows[0])[0], errors="coerce")):
        header = rows.pop(0)
    vals = pd.to_numeric(pd.Series([re.split(r"[,;\t ]", r)[0] for r in rows]).str.replace(",", ".", regex=False), errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 4:
        raise ValueError(f"{name}: no data table or time column was recognised.")
    med = float(np.median(vals))
    forced = opts.roles.get(0)
    rr_hdr = bool(header and CAT["rr"].search(_clean(header)))
    looks_rr = forced == "rr" or (forced in (None, "rr") and (rr_hdr or (header is None and (200 <= med <= 3000 or 0.2 <= med <= 3.0))))
    pf = ParsedFile(name, "", [], preview=lines[:40], header_line=0 if header else None, data_line=int(header is not None))
    if looks_rr and opts.interval_s is None:
        rr = vals * 1000 if med < 10 else vals
        s = _rr_subject(name, rr, opts.start)
        pf.format, pf.subjects, pf.rows, pf.notes = "RR / inter-beat intervals", [s], len(rr), list(s.notes)
        pf.columns = [ColumnInfo(header or "RR interval (ms)", "rr", f"{rr[0]:.0f}")]
        pf.time_desc, pf.relative_time = "Beat times = cumulative sum of the RR intervals", opts.start is None
        return pf
    role = forced if forced in ROLES else "activity"
    pf.format, pf.needs_interval, pf.relative_time = "Single column of values", True, opts.start is None
    pf.columns = [ColumnInfo(header or "Values", role, f"{vals[0]:g}")]
    if not opts.interval_s:
        raise ValueError(f"{name}: one column of numbers without time stamps. Set the sampling interval (and start time).")
    base = pd.Timestamp(opts.start or _today())
    t = base + pd.to_timedelta(np.arange(len(vals)) * opts.interval_s, unit="s")
    pf.subjects = [Subject(name, pf.format, pd.DatetimeIndex(t), [Channel(role, header_name(header) if header else "Activity", "", vals)])]
    pf.rows, pf.time_desc = len(vals), f"No time stamps: every {fmt_duration(opts.interval_s)} from {base:%Y-%m-%d %H:%M}"
    return pf


AWD_FORMAT = "AWD (Actiwatch / CamNtech / ClockLab)"


def _awd(lines: List[str], name: str, opts: ImportOptions) -> ParsedFile:
    """Actiwatch / CamNtech and Actimetrics ClockLab text AWD: 7 header lines, then "count" per line,
    optionally ",light" (ClockLab) or " , M" (Actiwatch marker). Negative values are missing data."""
    code = lines[3].strip()
    # Actiwatch's sub-15 s codes; otherwise the code is minutes x 4 (ClockLab; matches Actiwatch 1/2/4/8/20)
    epoch = {"81": 2, "C1": 5, "c1": 5}.get(code) or (int(code) * 15 if code.isdigit() and int(code) > 0 else None)
    step = opts.interval_s or epoch
    if not step:
        raise ValueError(f"{name}: unknown AWD epoch code '{code}'. Set the sampling interval.")
    start = opts.start or pd.to_datetime(f"{lines[1].strip()} {lines[2].strip()}", dayfirst=True, errors="coerce")
    start = start if start is not None and not pd.isna(start) else _today()
    fields = [[x for x in re.split(r"[ ,\t]", ln.strip()) if x] for ln in lines[7:] if ln.strip()]
    vals = pd.to_numeric(pd.Series([f[0] if f else "" for f in fields]), errors="coerce").to_numpy(dtype=float, copy=True)
    light = pd.to_numeric(pd.Series([f[1] if len(f) > 1 else "" for f in fields]), errors="coerce").to_numpy(dtype=float, copy=True)
    gaps = int((vals < 0).sum())
    vals[vals < 0] = np.nan
    light[light < 0] = np.nan
    t = pd.Timestamp(start) + pd.to_timedelta(np.arange(len(vals)) * step, unit="s")
    chans = [Channel("activity", "Activity", "counts", vals)]
    cols = [ColumnInfo("Activity counts", "activity", lines[7].strip() if len(lines) > 7 else "")]
    if np.isfinite(light).any():
        chans.append(Channel("light", "Light", "", light))
        cols.append(ColumnInfo("Light (2nd value)", "light", ""))
    subj = Subject(f"{name} ({lines[0].strip()})" if lines[0].strip() else name, AWD_FORMAT, pd.DatetimeIndex(t), chans)
    notes = [f"Subject '{lines[0].strip()}', start {pd.Timestamp(start):%Y-%m-%d %H:%M}, epoch {fmt_duration(step)} (code {code})."]
    if gaps:
        notes.append(f"{gaps} negative values read as missing data.")
    return ParsedFile(name, subj.format, [subj], notes, lines[:40], None, 7, cols, f"Header start time + {fmt_duration(step)} epochs", rows=len(vals))


def _empatica(lines: List[str], name: str, stem: str) -> ParsedFile:
    start = float(lines[0].split(",")[0])
    t0 = _unix_local(np.array([start]))[0]
    if stem == "IBI":
        arr = np.array([[float(x) for x in ln.split(",")[:2]] for ln in lines[1:] if ln.strip() and len(ln.split(",")) >= 2])
        times = _unix_local(start + arr[:, 0])
        s = _rr_subject(name, arr[:, 1] * 1000, None, times)
        s.channels[0].name, s.format = "Heart rate (IBI)", "Empatica E4"
        return ParsedFile(name, "Empatica E4", [s], list(s.notes), lines[:40], None, 1, [ColumnInfo("IBI (s)", "rr", lines[1] if len(lines) > 1 else "")], f"Unix start {t0:%Y-%m-%d %H:%M:%S} + beat times", rows=len(arr))
    hz = float(lines[1].split(",")[0])
    data = np.array([[float(x) for x in ln.split(",")] for ln in lines[2:] if ln.strip()], dtype=float)
    roles = {"ACC": [("accel_x", "x"), ("accel_y", "y"), ("accel_z", "z")], "TEMP": [("temperature", "Skin temperature")], "HR": [("heart_rate", "Heart rate")], "EDA": [("signal", "EDA")]}.get(stem, [("signal", "BVP")])
    unit = {"ACC": "g", "TEMP": "°C", "HR": "bpm", "EDA": "µS"}.get(stem, "")
    scale = 1 / 64 if stem == "ACC" else 1.0
    t = _unix_local(start + np.arange(len(data)) / hz)
    chans = [Channel(r, n, unit, data[:, k] * scale) for k, (r, n) in enumerate(roles) if k < data.shape[1]]
    return ParsedFile(name, "Empatica E4", [Subject(name, "Empatica E4", t, chans)], [], lines[:40], None, 2, [ColumnInfo(n, r, "") for r, n in roles], f"Unix start {t0:%Y-%m-%d %H:%M:%S} + {hz:g} Hz", rows=len(data))


def _apple_health(text: str, name: str) -> ParsedFile:
    kinds = {
        "HKQuantityTypeIdentifierHeartRate": ("heart_rate", "Heart rate", "bpm", False),
        "HKQuantityTypeIdentifierStepCount": ("activity", "Steps", "steps", True),
        "HKQuantityTypeIdentifierBodyTemperature": ("temperature", "Body temperature", "°C", False),
        "HKQuantityTypeIdentifierAppleSleepingWristTemperature": ("temperature", "Wrist temperature", "°C", False),
    }
    rec = re.compile(r"<Record\s([^>]*?)/?>")
    attr = re.compile(r'(\w+)="([^"]*)"')
    rows: Dict[str, List[Tuple[str, float]]] = {k: [] for k in kinds}
    for m in rec.finditer(text):
        a = dict(attr.findall(m.group(1)))
        if a.get("type") in kinds and "startDate" in a:
            rows[a["type"]].append((a["startDate"], float(a.get("value", "nan") or "nan")))
    frames = []
    notes = []
    for k, rws in rows.items():
        if not rws:
            continue
        role, nm, unit, sm = kinds[k]
        t = parse_datetimes(pd.Series([r[0] for r in rws]), "ymd")
        frames.append((role, nm, unit, sm, pd.DataFrame({"t": t, "v": [r[1] for r in rws]}).dropna(subset=["t"])))
        notes.append(f"{nm}: {len(rws):,} records")
    if not frames:
        raise ValueError("No heart-rate, step or temperature records found in the Apple Health export.")
    allt = pd.concat([f[4].assign(k=i) for i, f in enumerate(frames)]).sort_values("t")
    chans = []
    for i, (role, nm, unit, sm, _) in enumerate(frames):
        v = np.where(allt["k"].to_numpy() == i, allt["v"].to_numpy(dtype=float), np.nan)
        chans.append(Channel(role, nm, unit, v, sm))
    s = Subject(name, "Apple Health export.xml", pd.DatetimeIndex(allt["t"]), chans, notes)
    return ParsedFile(name, s.format, [s], notes, text.splitlines()[:40], columns=[ColumnInfo(c.name, c.role, "") for c in chans], time_desc="Record startDate (local wall-clock time)", rows=len(allt))


def _role_from_filename(name: str) -> Tuple[str, str, str, bool]:
    low = name.lower()
    if "heart" in low:
        return "heart_rate", "Heart rate", "bpm", False
    if "step" in low:
        return "activity", "Steps", "steps", True
    if "temp" in low:
        return "temperature", "Temperature", "°C", False
    if "calor" in low:
        return "signal", "Calories", "kcal", True
    return "signal", "Value", "", False


def _jnum(v) -> float:
    if isinstance(v, dict):
        for k in ("bpm", "value", "temperature", "steps"):
            if k in v:
                return _jnum(v[k])
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _json(text: str, name: str, opts: ImportOptions) -> ParsedFile:
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        v = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    preview = text.splitlines()[:40] if "\n" in text[:5000] else [text[:400]]
    if isinstance(v, dict):
        intr = next(((k, x) for k, x in v.items() if k.endswith("-intraday")), None)
        if intr:
            key, x = intr
            base = v.get(key.replace("-intraday", ""), [{}])
            date = pd.to_datetime(base[0].get("dateTime") if base else None, errors="coerce")
            date = date if date is not None and not pd.isna(date) else pd.Timestamp(opts.start or _today())
            ds = pd.DataFrame(x.get("dataset", []))
            secs = parse_tod(ds["time"]).to_numpy()
            day = np.concatenate([[0], np.cumsum(np.diff(secs) < 0)]) * 86400
            role, nm, unit, sm = _role_from_filename(key)
            t = pd.Timestamp(date) + pd.to_timedelta(secs + day, unit="s")
            s = Subject(name, "Fitbit intraday JSON", pd.DatetimeIndex(t), [Channel(role, nm, unit, ds["value"].map(_jnum).to_numpy(dtype=float), sm)])
            return ParsedFile(name, s.format, [s], [], preview, columns=[ColumnInfo(key, role, "")], time_desc=f"Date {pd.Timestamp(date):%Y-%m-%d} + time of day", rows=len(ds))
        v = next((x for x in v.values() if isinstance(x, list)), None)
        if v is None:
            raise ValueError(f"{name}: no array of records found in the JSON.")
    if v and isinstance(v[0], dict) and "dateTime" in v[0] and "value" in v[0]:
        dts = pd.Series([r.get("dateTime") for r in v], dtype=str)
        order = opts.date_order if opts.date_order != "auto" else detect_order(dts.head(200).tolist())[0]
        t = parse_datetimes(dts, order)
        role, nm, unit, sm = _role_from_filename(name)
        vals = np.array([_jnum(r.get("value")) for r in v], dtype=float)
        ok = t.notna().to_numpy()
        s = Subject(name, "Fitbit JSON", pd.DatetimeIndex(t[ok]), [Channel(role, nm, unit, vals[ok], sm)])
        return ParsedFile(name, s.format, [s], [], preview, columns=[ColumnInfo("dateTime", "time", str(dts.iloc[0])), ColumnInfo("value", role, "")], time_desc=f"dateTime ({order})", rows=int(ok.sum()))
    df = pd.json_normalize(v)
    if df.empty:
        raise ValueError(f"{name}: the JSON records have no fields.")
    pf = _parse_text(df.to_csv(sep="\t", index=False), name, opts)
    pf.format, pf.preview = f"JSON records → {pf.format}", preview
    return pf


def _dam(df: pd.DataFrame, name: str, lines: List[str], data_start: int) -> ParsedFile:
    df = df[df.iloc[:, 3].astype(str).str.strip() == "1"]
    t = parse_datetimes(df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str), "dmy")
    ok = t.notna().to_numpy()
    df, t = df[ok], pd.DatetimeIndex(t[ok])
    light = pd.to_numeric(df.iloc[:, 9], errors="coerce").to_numpy(dtype=float)
    light_varies = np.nanmax(light) == 1 and np.nanmin(light) == 0
    subjects, empty = [], 0
    for k in range(10, df.shape[1]):
        v = pd.to_numeric(df.iloc[:, k], errors="coerce").to_numpy(dtype=float)
        if not np.nansum(v):
            empty += 1
            continue
        ch = [Channel("activity", "Activity", "beam crossings", v)]
        if light_varies:
            ch.append(Channel("light", "Light", "on/off", light * 1000))
        subjects.append(Subject(f"{name} · channel {k - 9}", "TriKinetics DAM", t, ch))
    if not subjects:
        raise ValueError(f"{name}: all DAM channels are empty.")
    notes = [f"{len(subjects)} active channels (one animal each); {empty} empty channels skipped."]
    if light_varies:
        notes.append("Column 10 (light on/off) is used as the light channel (on = 1000 lux).")
    return ParsedFile(name, "TriKinetics DAM monitor", subjects, notes, lines[:40], None, data_start,
                      [ColumnInfo("Date + time (columns 2–3)", "time", ""), ColumnInfo(f"Channels 1–{df.shape[1] - 10}", "activity", "")], "Columns 2 (date) + 3 (time)", rows=len(df))


def _wide_rows(df: pd.DataFrame, name: str, opts: ImportOptions, headers: List[str], labels: List[Optional[float]], lines: List[str], hdr: Optional[int], start: int) -> ParsedFile:
    base = pd.Timestamp(opts.start or _today())
    unit = UNIT_SECONDS[opts.time_unit or "h"]
    cols = [(i, base + pd.to_timedelta(lab * unit, unit="s")) for i, lab in enumerate(labels) if i > 0 and lab is not None]
    subjects = []
    for _, row in df.iterrows():
        pts = sorted((t, pd.to_numeric(str(row.iloc[i]).replace(",", "."), errors="coerce")) for i, t in cols)
        subjects.append(Subject(str(row.iloc[0]).strip(), "Time-course table (time points as columns)", pd.DatetimeIndex([p[0] for p in pts]), [Channel("signal", "Expression", "", np.array([p[1] for p in pts], dtype=float))]))
    zt = headers[cols[0][0]].upper().startswith(("ZT", "CT"))
    notes = [f"{len(subjects):,} rows (genes / features) × {len(cols)} time points; each row is a separate subject.",
             "Times come from ZT / CT column headers (hours); ZT0 / CT0 = 00:00 on the chosen start date." if zt else "Times come from the column headers (hours).",
             "Sparse sampling: cosinor fits are reliable; periodograms need ≥ 2 cycles at ≤ 4 h spacing."]
    for s in subjects:
        s.notes = notes
    return ParsedFile(name, "Time-course table (ZT / CT columns)", subjects, notes, lines[:40], hdr, start,
                      [ColumnInfo(headers[0], "ignore", "")] + [ColumnInfo(headers[i], "time", "") for i, _ in cols],
                      f"Column headers as hours from {base:%Y-%m-%d %H:%M}", relative_time=True, numeric_time=True, rows=len(subjects))


def _to_float(s: pd.Series) -> Tuple[pd.Series, bool]:
    """Numeric conversion with decimal-comma detection."""
    v = s.astype(str).str.strip().str.strip('"')
    comma = v.str.fullmatch(r"-?\d+,\d+").any() and not v.str.fullmatch(r"-?\d+\.\d+([eE][-+]?\d+)?").any()
    if comma:
        v = v.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(v, errors="coerce"), comma


def _parse_table(text: str, name: str, opts: ImportOptions, lines: List[str], meta: dict, shape) -> ParsedFile:
    delim, hdr, start, ncols = shape
    notes = list(meta["notes"])
    positional = hdr is None and meta["positional"] is not None
    if hdr is not None:
        headers = [_clean(x) for x in _split(lines[hdr], delim)]
    elif positional:
        headers = [p[0] for p in meta["positional"]]
        notes.append("Header-less table: columns read in the device's documented order.")
    else:
        headers = []
    headers = (headers + [f"Column {i + 1}" for i in range(len(headers), ncols)])[:ncols]
    headers = [h or f"Column {i + 1}" for i, h in enumerate(headers)]
    body = "\n".join(text.splitlines()[start:])
    sep = r"\s+" if delim == "ws" else delim
    df = pd.read_csv(io.StringIO(body), sep=sep, header=None, names=range(ncols), usecols=range(ncols), dtype=str, engine="python" if delim == "ws" else "c",
                     quotechar='"', skip_blank_lines=True, on_bad_lines="skip", keep_default_na=False)
    if df.empty:
        raise ValueError(f"{name}: no data rows below the header.")
    sample = df.head(400)

    # TriKinetics DAM
    if delim == "\t" and ncols >= 42 and sample.iloc[:, 1].astype(str).str.fullmatch(r"\d{1,2} [A-Za-z]{3} \d{2,4}").mean() > 0.8:
        return _dam(df, name, lines, start)

    order_samples = [str(x) for c in range(min(3, ncols)) for x in sample.iloc[:, c] if re.search(r"[/\-.]", str(x)) and re.search(r"\d", str(x))][:2000]
    auto_order, ambiguous = detect_order(order_samples)
    order = opts.date_order if opts.date_order != "auto" else (meta["order"] or auto_order)
    date_ambiguous = ambiguous and meta["order"] is None and opts.date_order == "auto"

    numeric = {}
    comma = {}
    for i in range(ncols):
        v, cm = _to_float(sample.iloc[:, i])
        ne = sample.iloc[:, i].astype(str).str.strip().str.lower().isin(_MISSING)
        numeric[i] = float(v.notna().sum()) / max(1, int((~ne).sum()))
        comma[i] = cm

    # Gene-expression / ZT-CT wide table
    if hdr is not None and opts.layout != "channels":
        labels = []
        for h in headers:
            m = CAT["time_label"].match(h)
            labels.append(float(m.group(2)) if m else None)
        n_tl = sum(x is not None for x in labels[1:])
        if n_tl >= 4 and n_tl * 10 >= (ncols - 1) * 8 and (numeric[0] < 0.5 or CAT["id_col"].match(headers[0])):
            return _wide_rows(df, name, opts, headers, labels, lines, hdr, start)

    # Classify date/time-ness of each column
    dt_frac, tod_frac, date_only = {}, {}, {}
    for i in range(ncols):
        s = sample.iloc[:, i].astype(str)
        s = s[~s.str.strip().str.lower().isin(_MISSING)]
        if s.empty or numeric[i] > 0.9 or s.str.contains(r"\d").mean() < 0.8:
            dt_frac[i] = tod_frac[i] = date_only[i] = 0.0
            continue
        # Pure clock times ("08:00:00") are time-of-day, never full dates (pandas would put them on today)
        is_tod = parse_tod(s).notna()
        tod_frac[i] = float(is_tod.mean())
        parsed = parse_datetimes(s, order)
        has_time = s.str.contains(":")
        dt_frac[i] = float((parsed.notna() & has_time & ~is_tod).mean())
        date_only[i] = float((parsed.notna() & ~has_time & ~is_tod).mean())

    roles: Dict[int, Optional[str]] = {}
    for i, h in enumerate(headers):
        hl = h.lower()
        if i in opts.roles:
            roles[i] = opts.roles[i]
        elif hl in meta["by_name"]:
            roles[i] = meta["by_name"][hl]
        elif positional and i < len(meta["positional"]):
            roles[i] = meta["positional"][i][1]
        else:
            roles[i] = auto_role(h, numeric[i])
    xyz = [i for i, h in enumerate(headers) if CAT["xyz"].match(h) and numeric[i] > 0.6 and i not in opts.roles]
    if len(xyz) == 3:
        for i in xyz:
            roles[i] = f"accel_{headers[i].lower()}"

    # ---- time column(s)
    base = opts.start or meta["start"]
    time_desc, relative, numeric_time, needs_interval = "", False, False, False
    t: Optional[pd.Series] = None
    time_cols: List[int] = []

    def name_score(i: int) -> int:
        h = headers[i]
        if CAT["time_abs"].search(h):
            return 3
        if re.match(r"(?i)^(date|day|datum|fecha|data)$", h) or CAT["time_rel"].match(h):
            return 2
        return 0

    def numeric_time_col(i: int):
        v, _ = _to_float(df.iloc[:, i])
        med = float(np.nanmedian(v.to_numpy(dtype=float)))
        hl = headers[i].lower()
        unit = opts.time_unit or time_unit_from_header(headers[i])
        date_like = any(w in hl for w in ("date", "serial", "datenum"))
        if opts.time_unit is None and date_like and 20000 < med < 80000:
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(v, unit="D"), f"'{headers[i]}' as Excel serial dates", False
        if opts.time_unit is None and date_like and 690000 < med < 800000:
            return pd.to_datetime("1970-01-01") + pd.to_timedelta(v - 719529, unit="D"), f"'{headers[i]}' as MATLAB datenums", False
        if opts.time_unit is None and "elapsed" not in hl and unit in (None, "s", "ms"):
            scale = 1e-9 if med > 1e17 else 1e-6 if med > 1e14 else 1e-3 if med > 1e11 else 1.0 if med > 3e8 else 0.0
            if scale:
                lab = {1e-9: "nanoseconds", 1e-6: "microseconds", 1e-3: "milliseconds", 1.0: "seconds"}[scale]
                return pd.Series(_unix_local(v.to_numpy(dtype=float) * scale)), f"'{headers[i]}' as Unix time in {lab} (local time)", False
        u = unit or "s"
        b = pd.Timestamp(base or _today())
        return b + pd.to_timedelta(v * UNIT_SECONDS[u], unit="s"), f"'{headers[i]}' as {u} since {b:%Y-%m-%d %H:%M}", True

    forced_t = next((i for i, r in opts.roles.items() if r == "time"), None)
    forced_d = next((i for i, r in opts.roles.items() if r == "date"), None)
    if forced_t is None and positional:
        forced_t = next((i for i, p in enumerate(meta["positional"]) if p[1] == "time" and i < ncols), None)
    if forced_t is None:
        forced_t = next((i for i, h in enumerate(headers) if meta["by_name"].get(h.lower()) == "time"), None)
    if forced_t is not None:
        if forced_d is not None:
            t = parse_datetimes(df.iloc[:, forced_d].astype(str) + " " + df.iloc[:, forced_t].astype(str), order)
            time_cols, time_desc = [forced_d, forced_t], f"'{headers[forced_d]}' + '{headers[forced_t]}'"
        elif dt_frac.get(forced_t, 0) > 0.7 or date_only.get(forced_t, 0) > 0.7:
            t = parse_datetimes(df.iloc[:, forced_t], order)
            time_cols, time_desc = [forced_t], f"'{headers[forced_t]}' (date & time)"
        elif tod_frac.get(forced_t, 0) > 0.7:
            t, time_cols = None, [forced_t]
        else:
            t, time_desc, relative = numeric_time_col(forced_t)
            numeric_time, time_cols = True, [forced_t]
    if not time_cols:
        cands = sorted((i for i in range(ncols) if dt_frac.get(i, 0) > 0.8 and roles.get(i) != "mask" and sample.iloc[:, i].nunique() > 1), key=lambda i: -name_score(i))
        if cands:
            i = cands[0]
            t, time_cols, time_desc = parse_datetimes(df.iloc[:, i], order), [i], f"'{headers[i]}' (date & time)"
    if not time_cols:
        d = next((i for i in range(ncols) if date_only.get(i, 0) > 0.8), None)
        c = next((i for i in range(ncols) if tod_frac.get(i, 0) > 0.8 and i != d), None)
        if d is not None and c is not None:
            t = parse_datetimes(df.iloc[:, d].astype(str) + " " + df.iloc[:, c].astype(str), order)
            time_cols, time_desc = [d, c], f"'{headers[d]}' + '{headers[c]}'"
        elif d is not None and sample.iloc[:, d].nunique() > 1:
            t, time_cols, time_desc = parse_datetimes(df.iloc[:, d], order), [d], f"'{headers[d]}' (dates)"
    if not time_cols:
        cands = sorted((i for i in range(ncols) if numeric[i] > 0.9 and (CAT["time_rel"].match(headers[i]) or CAT["time_abs"].search(headers[i]) or any(w in headers[i].lower() for w in ("date", "serial", "datenum")))), key=lambda i: -name_score(i))
        if cands:
            t, time_desc, relative = numeric_time_col(cands[0])
            numeric_time, time_cols = True, [cands[0]]
    if not time_cols:
        c = next((i for i in range(ncols) if tod_frac.get(i, 0) > 0.8), None)
        if c is not None:
            time_cols = [c]
    if time_cols and t is None and len(time_cols) == 1 and tod_frac.get(time_cols[0], 0) > 0.7:
        i = time_cols[0]
        secs = parse_tod(df.iloc[:, i]).to_numpy(dtype=float)
        day = np.concatenate([[0], np.cumsum(np.diff(np.nan_to_num(secs, nan=0)) < -12 * 3600)]) * 86400
        b = pd.Timestamp(base or _today())
        t = pd.Series(b + pd.to_timedelta(secs + day, unit="s"))
        relative = base is None
        time_desc = f"'{headers[i]}' times from {b:%Y-%m-%d %H:%M}"
    if not time_cols:
        step = opts.interval_s or meta["epoch_s"]
        if step:
            b = pd.Timestamp(base or _today())
            t = pd.Series(b + pd.to_timedelta(np.arange(len(df)) * step, unit="s"))
            relative = base is None
            time_desc = f"No time column: every {fmt_duration(step)} from {b:%Y-%m-%d %H:%M:%S}"
        else:
            needs_interval = True
    if t is None:
        if any("Circadian Rhythmometry" in h and "Report" in h for h in headers):
            raise ValueError("This is an analysis report generated by this app, not sensor data. Upload the data file you analysed instead.")
        raise ValueError(f"{name}: no time column found in {headers}. Mark the time column, or set a sampling interval for evenly spaced rows.")
    for i in time_cols:
        roles[i] = "date" if len(time_cols) == 2 and i == time_cols[0] else "time"
    for i in range(ncols):
        if i not in time_cols and roles.get(i) in ("time", "date") and i not in opts.roles:
            roles[i] = "ignore"

    # ---- resolve roles
    axis_cols = [] if positional else [i for i, h in enumerate(headers) if CAT["axis"].match(h) and i not in opts.roles]
    vm_col = None if positional else next((i for i, h in enumerate(headers) if re.match(r"(?i)^(vector ?magnitude|vm)$", h) and numeric[i] > 0.6 and i not in opts.roles), None)
    vm_from_axes = False
    if vm_col is not None:
        for a in axis_cols:
            roles[a] = "ignore"
        roles[vm_col] = "activity"
    elif len(axis_cols) == 3 and not opts.roles:
        vm_from_axes = True
        for i in range(ncols):
            if i in axis_cols or roles.get(i) == "activity":
                roles[i] = "ignore"

    def keep_one(role: str, prefer) -> None:
        idx = sorted((i for i in range(ncols) if roles.get(i) == role and i not in opts.roles), key=lambda i: -prefer(headers[i]))
        for i in idx[1:]:
            roles[i] = "ignore"

    keep_one("activity", lambda h: 0 if CAT["steps"].search(h) else 3 if re.match(r"(?i)^(activity|pim|vector ?magnitude|vm|enmo|counts?)$", h) else 1)
    keep_one("light", lambda h: 2 if re.match(r"(?i)^(light|lux|white light|illuminance)", h) else 1)
    keep_one("heart_rate", lambda h: 0)
    keep_one("rr", lambda h: 0)
    unknown = [i for i in range(ncols) if roles.get(i) is None and numeric[i] > 0.6 and i not in time_cols]
    known_data = any(roles.get(i) in (*ROLES, "rr") for i in range(ncols)) or vm_from_axes
    sig_cols = [i for i in range(ncols) if roles.get(i) == "signal" and i not in opts.roles]
    can_layout = len(unknown) + len(sig_cols) >= 2 and not any(roles.get(i) in ROLES and roles.get(i) != "signal" for i in range(ncols))
    sample_named = sum(bool(CAT["sample_like"].search(headers[i])) for i in unknown)
    layout = opts.layout
    if layout == "auto":
        layout = "subjects" if can_layout and (len(unknown) + len(sig_cols) >= 4 or sample_named >= 2) else "channels"
    for i in unknown:
        roles[i] = "signal" if (not known_data or layout == "subjects") else "ignore"
    for i in range(ncols):
        roles[i] = roles.get(i) or "ignore"

    # ---- build channels
    tt = pd.to_datetime(pd.Series(t).reset_index(drop=True), errors="coerce")
    ok = tt.notna().to_numpy()
    skipped = int((~ok).sum())
    if skipped:
        notes.append(f"{skipped:,} rows skipped (time could not be read).")
    if not ok.any():
        raise ValueError(f"{name}: no rows could be read with the time column {time_desc}.")
    times = pd.DatetimeIndex(tt[ok])
    vals = {i: _to_float(df.iloc[:, i])[0].to_numpy(dtype=float)[ok] for i in range(ncols) if roles[i] in (*ROLES, "rr") or (vm_from_axes and i in axis_cols)}
    masked = np.zeros(int(ok.sum()), dtype=bool)
    for i in (i for i in range(ncols) if roles[i] == "mask"):
        col = df.iloc[:, i].astype(str).str.strip().str.lower()[ok]
        num = pd.to_numeric(col, errors="coerce")
        masked |= (num.fillna(0) != 0).to_numpy() | col.str.contains("excluded|off|nonwear|non-wear", regex=True).to_numpy()
    channels: List[Channel] = []
    rr_series = None
    subject_cols = []
    for i in range(ncols):
        r = roles[i]
        if r == "rr":
            rr_series = vals[i]
            u = (header_unit(headers[i]) or "").lower()
            fin = rr_series[np.isfinite(rr_series)]
            if u in ("s", "sec", "secs", "seconds") or (u not in ("ms", "msec") and len(fin) and np.median(fin) < 10):
                rr_series = rr_series * 1000.0  # intervals in seconds (Empatica / Polar IBI) -> ms
                notes.append(f"'{headers[i].strip()}' read as seconds and converted to ms.")
        elif r == "signal" and layout == "subjects":
            subject_cols.append(i)
        elif r in ROLES and not (vm_from_axes and i in axis_cols):
            unit = header_unit(headers[i]) or {"light": "lux", "heart_rate": "bpm", "temperature": "°F" if "°F" in headers[i] else "°C", "activity": "activity units"}.get(r, "")
            nm = {"activity": "Activity", "light": "Light", "heart_rate": "Heart rate", "accel_x": "x", "accel_y": "y", "accel_z": "z"}.get(r)
            if nm is None:
                nm = header_name(headers[i])
                if r == "temperature" and (nm.lower() == "value" or nm.lower().startswith("temp")):
                    nm = "Temperature"
                if r == "signal" and nm.lower() in EDA_NAMES:
                    nm, unit = "EDA", unit or "µS"
                elif r == "signal" and nm.lower() in ("bvp", "ppg"):
                    nm = nm.upper()
            channels.append(Channel(r, nm, unit, vals[i]))
    if vm_from_axes:
        a = [vals[i] for i in axis_cols]
        channels.append(Channel("activity", "Activity", "counts (vector magnitude of axes 1–3)", np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)))
        notes.append("Activity = vector magnitude of ActiGraph axes 1–3.")
    unit_col = next((i for i, h in enumerate(headers) if h.lower() == "unit"), None)
    if unit_col is not None:
        u = str(sample.iloc[0, unit_col]).strip().upper()
        for c in channels:
            if c.role == "temperature":
                c.unit = "°F" if u.startswith("F") else "°C"
    if masked.any():
        for c in channels:
            c.values = np.where(masked, np.nan, c.values)
        notes.append(f"{int(masked.sum()):,} rows flagged as off-wrist / excluded were set to missing.")
    accel_linear = any(roles[i].startswith("accel_") and ("linear" in headers[i].lower() or "without g" in headers[i].lower()) for i in range(ncols))
    axes = [next((c for c in channels if c.role == r), None) for r in ("accel_x", "accel_y", "accel_z")]
    if not accel_linear and all(c is not None for c in axes):
        # raw device units (Empatica 1/64 g, Axivity 1/256 g, milli-g ...) -> g, judged from the resting magnitude
        step = max(1, len(axes[0].values) // 200_000)
        vm = np.sqrt(sum(c.values[::step] ** 2 for c in axes))
        med = float(np.nanmedian(vm)) if np.isfinite(vm).any() else float("nan")
        g, desc = accel_g_units(med)
        if desc:
            for c in axes:
                c.values = c.values / g
                c.unit = "g"
            notes.append(f"Acceleration at rest ≈ {med:.1f} units: read as {desc} and converted to g.")
    fmt = meta["format"] or ("Multi-sample table (one subject per column)" if layout == "subjects" else "Delimited table")
    subjects: List[Subject] = []
    if rr_series is not None:
        s = _rr_subject(name, rr_series, None, times)
        s.channels[0].name = "Heart rate (IBI)"  # distinct from a separate heart-rate file, so both merge side by side
        s.channels.extend(channels)
        notes.extend(s.notes)
        subjects.append(s)
    elif layout == "subjects":
        lum = any("lum" in headers[i].lower() for i in range(ncols)) or "lum" in name.lower()
        for i in subject_cols:
            subjects.append(Subject(f"{name} · {headers[i]}", fmt, times, channels + [Channel("signal", "Bioluminescence" if lum else "Signal", "counts" if lum else "", vals[i])]))
        notes.append(f"{len(subject_cols)} columns treated as separate subjects / samples.")
    else:
        subjects.append(Subject(name, fmt, times, channels, accel_linear=accel_linear))
    for s in subjects:
        order_idx = np.argsort(s.t.asi8, kind="stable")
        if not np.all(np.diff(s.t.asi8) >= 0):
            s.t = s.t[order_idx]
            for c in s.channels:
                c.values = c.values[order_idx]
        s.notes = list(dict.fromkeys(s.notes + notes + (["Times are relative (no calendar date in the file); clock phases are measured from the chosen start."] if relative else [])))
    examples = {i: next((str(x) for x in sample.iloc[:, i] if str(x).strip()), "") for i in range(ncols)}
    return ParsedFile(name, fmt, subjects, notes, lines[:40], hdr, start,
                      [ColumnInfo(headers[i], "activity" if (vm_from_axes and i in axis_cols) else roles[i], examples[i]) for i in range(ncols)],
                      time_desc, relative, numeric_time, needs_interval, date_ambiguous, can_layout, layout, int(ok.sum()))


def _parse_text(text: str, name: str, opts: ImportOptions) -> ParsedFile:
    stripped = text.lstrip()
    if stripped.startswith("<") and ("<HealthData" in text or "HKQuantityTypeIdentifier" in text):
        return _apple_health(text, name)
    if stripped.startswith(("[", "{")):
        return _json(text, name, opts)
    lines = text.splitlines()[:600]
    if not lines:
        raise ValueError(f"{name} is empty.")
    stem = name.rsplit(".", 1)[0].upper()
    try:
        first = float(lines[0].split(",")[0])
    except ValueError:
        first = 0.0
    if stem in ("ACC", "BVP", "EDA", "HR", "TEMP", "IBI") and first > 1e9:
        return _empatica(text.splitlines(), name, stem)
    if name.lower().endswith(".awd") or (len(lines) > 8 and re.fullmatch(r"\s*\d{1,2}-[A-Za-z]{3}-\d{2,4}\s*", lines[1]) and ":" in lines[2] and "," not in lines[2] and len(lines[2].strip()) <= 8):
        return _awd(text.splitlines(), name, opts)
    meta = _meta(lines)
    shape = sniff(lines, opts.header_line if opts.header_line is not None else meta["header_line"])
    if shape is None or shape[3] < 2:
        return _single_column(text, name, opts, lines)
    return _parse_table(text, name, opts, lines, meta, shape)


def _agd(data: bytes, name: str) -> ParsedFile:
    """ActiGraph .agd (SQLite): 'data' table (dataTimestamp in .NET ticks, axis1-3, lux) + 'settings'."""
    with tempfile.NamedTemporaryFile(suffix=".agd", delete=False) as f:
        f.write(data)
        path = f.name
    con = sqlite3.connect(path)
    try:
        df = pd.read_sql_query("SELECT * FROM data", con)
    finally:
        con.close()
    cols = {c.lower(): c for c in df.columns}
    t = pd.to_datetime((df[cols["datatimestamp"]].astype("int64") - 621355968000000000) // 10, unit="us")
    axes = [df[cols[a]].to_numpy(dtype=float) for a in ("axis1", "axis2", "axis3") if a in cols]
    act = np.sqrt(sum(a ** 2 for a in axes)) if len(axes) == 3 else axes[0]
    ch = [Channel("activity", "Activity", "counts (vector magnitude)" if len(axes) == 3 else "counts", act)]
    if "lux" in cols:
        ch.append(Channel("light", "Light", "lux", df[cols["lux"]].to_numpy(dtype=float)))
    s = Subject(name, "ActiGraph .agd", pd.DatetimeIndex(t), ch)
    return ParsedFile(name, s.format, [s], [], [f"SQLite database, {len(df):,} epochs"], columns=[ColumnInfo(c, "", "") for c in df.columns], time_desc="dataTimestamp (.NET ticks)", rows=len(df))


def parse_source(src: Source, opts: Optional[ImportOptions] = None) -> ParsedFile:
    opts = opts or ImportOptions()
    if opts.start is None and src.start_hint is not None:
        opts = ImportOptions(**{**opts.__dict__, "start": src.start_hint})
    low = src.name.lower()
    if low.endswith(".agd"):
        return _agd(src.data, src.name)
    if low.endswith((".xlsx", ".xlsm", ".xls", ".ods")):
        sheets = pd.read_excel(io.BytesIO(src.data), sheet_name=None, header=None, dtype=str)
        name, sheet = next(((n, s) for n, s in sheets.items() if s.shape[0] >= 3), (None, None))
        if sheet is None:
            raise ValueError("No sheet with a data table was found in the workbook.")
        pf = _parse_text(sheet.fillna("").to_csv(sep="\t", header=False, index=False), src.name, opts)
        if len(sheets) > 1:
            pf.notes.append(f"Workbook has {len(sheets)} sheets; using '{name}'.")
        return pf
    return _parse_text(decode(src.data), src.name, opts)


def will_merge(parsed: List[ParsedFile], mode: str = "auto") -> bool:
    """Whether `combine` turns these files into one recording. Auto merges files with different channels
    (sensors of one recording) or consecutive time ranges; multi-subject files and overlapping recordings stay apart."""
    subjects = [s for p in parsed for s in p.subjects if len(s.t)]
    if mode != "auto" or len(subjects) < 2:
        return mode == "merge" and len(subjects) > 1
    if not all(len(p.subjects) == 1 for p in parsed):
        return False
    keys = [{(c.role, c.name) for c in s.channels} for s in subjects]
    disjoint = all(not (keys[i] & keys[j]) for i in range(len(keys)) for j in range(i + 1, len(keys)))
    rng = sorted((s.t[0], s.t[-1]) for s in subjects)
    consecutive = all(b[0] > a[1] for a, b in zip(rng, rng[1:]))
    return disjoint or consecutive


def combine(parsed: List[ParsedFile], mode: str = "auto") -> Tuple[List[Subject], List[str]]:
    """auto | merge | separate. Auto merges different sensors of one recording or consecutive files and keeps
    overlapping recordings of the same kind (different people / animals) apart."""
    subjects = [s for p in parsed for s in p.subjects if len(s.t)]
    if len(subjects) <= 1:
        return subjects, []
    mode = "merge" if will_merge(parsed, mode) else "separate"
    if mode == "separate":
        return subjects, [f"{len(subjects)} subjects (pick one, or analyse all)."]
    keys = []
    for s in subjects:
        for c in s.channels:
            if (c.role, c.name) not in [(k[0], k[1]) for k in keys]:
                keys.append((c.role, c.name, c.unit, c.sum))
    t = np.concatenate([s.t.asi8 for s in subjects])
    chans = []
    for role, nm, unit, sm in keys:
        parts = []
        for s in subjects:
            c = next((c for c in s.channels if c.role == role and c.name == nm), None)
            parts.append(c.values if c is not None else np.full(len(s.t), np.nan))
        chans.append(Channel(role, nm, unit, np.concatenate(parts), sm))
    o = np.argsort(t, kind="stable")
    merged = Subject(" + ".join(s.name for s in subjects), " + ".join(dict.fromkeys(s.format for s in subjects)), pd.DatetimeIndex(t[o]),
                     [Channel(c.role, c.name, c.unit, c.values[o], c.sum) for c in chans], [n for s in subjects for n in s.notes], any(s.accel_linear for s in subjects))
    return [merged], [f"{len(subjects)} files merged into one recording."]


def subject_to_dataset(subj: Subject, primary: Optional[str] = None):
    """Convert to the analysis pipeline's RawSensorDataset. `primary` names the channel analysed as the main
    signal (default: activity / accelerometer, else heart rate, temperature, other signal)."""
    from .io_loader import RawSensorDataset

    df = pd.DataFrame({"timestamp": subj.t})
    has_accel = all(subj.channel(r) is not None for r in ("accel_x", "accel_y", "accel_z"))
    cands = [c for c in subj.signal_channels if len(c.values) and np.isfinite(c.values).any()]
    chosen = next((c for c in cands if c.name == primary), None)
    if chosen is None and not (primary or "").startswith("Activity (from") and not has_accel:
        chosen = next((c for c in cands if c.role == "activity"), cands[0] if cands else None)
    if primary and primary.startswith("Activity (from"):
        chosen = None
    if chosen is not None:
        df["activity_raw"] = chosen.values
    if has_accel and (chosen is None or chosen.role == "activity"):
        for r, col in (("accel_x", "ax"), ("accel_y", "ay"), ("accel_z", "az")):
            df[col] = subj.channel(r).values
        df["is_linear_accel"] = subj.accel_linear
    light = subj.channel("light")
    if light is not None:
        df["lux"] = light.values
    meta = {
        "format": subj.format,
        "notes": subj.notes,
        "channels": subj.describe(),
        "primary_name": chosen.name if chosen is not None and chosen.role != "activity" else "Activity",
        "primary_unit": chosen.unit if chosen is not None and chosen.role != "activity" else None,
        "primary_sum": bool(chosen.sum) if chosen is not None else False,
        "other_signals": [c.name for c in cands if c is not chosen],
    }
    has_act = chosen is not None or has_accel
    return RawSensorDataset(df=df, source_name=subj.name, source_type=subj.format, has_activity=has_act, has_light=light is not None, metadata=meta)


def load_any(name: str, data: bytes, opts: Optional[ImportOptions] = None, mode: str = "auto") -> Tuple[List[Subject], List[str], List[ParsedFile]]:
    """Expand, parse and combine one upload. Unreadable archive members are reported in the notes."""
    parsed, notes = [], []
    for src in expand(name, data):
        try:
            parsed.append(parse_source(src, opts))
        except Exception as exc:  # noqa: BLE001 - report and continue with the other members
            notes.append(f"skipped {src.name}: {str(exc).splitlines()[0]}")
    if not parsed:
        raise ValueError("\n".join(notes) or f"{name}: nothing could be read.")
    subjects, n2 = combine(parsed, mode)
    return subjects, notes + n2, parsed
