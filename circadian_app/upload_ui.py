"""Streamlit import review: detected format, time interpretation, column roles, raw preview, subjects."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from .formats import (
    COL_ROLES,
    LABEL_TO_ROLE,
    ROLE_LABELS,
    ImportOptions,
    ParsedFile,
    Source,
    combine,
    expand,
    parse_source,
    will_merge,
)

UPLOAD_TYPES = ["csv", "tsv", "txt", "dat", "awd", "agd", "xlsx", "xls", "xlsm", "ods", "json", "jsonl", "xml", "zip", "gz"]
FORMAT_HELP = (
    "Actigraphy: ActiGraph (CSV, .agd), Philips Actiware, Condor ActTrust, Actiwatch .awd, GENEActiv, Axivity · "
    "Animal activity: TriKinetics DAM, wheel / beam-break counts · Loggers: HOBO, iButton · "
    "Heart: RR/IBI intervals, Polar, Empatica E4 · Phones & wearables: Phyphox, Sensor Logger, Apple Health, Fitbit · "
    "Lab tables: plate-reader / bioluminescence wells, ZT/CT gene-expression time courses · Archives: .zip, .gz"
)
DATE_ORDERS = {"auto": "Auto-detect", "dmy": "Day first (31/12/2024)", "mdy": "Month first (12/31/2024)", "ymd": "Year first (2024-12-31)"}
TIME_UNITS = {None: "Auto", "ms": "milliseconds", "s": "seconds", "min": "minutes", "h": "hours", "d": "days"}
MERGE_MODES = {"auto": "Auto (merge sensors / consecutive files, keep people apart)", "merge": "Merge into one recording", "separate": "Keep files as separate subjects"}


def _key(name: str, data: bytes) -> str:
    return f"{name}:{hashlib.sha1(data).hexdigest()[:12]}"


@st.cache_data(max_entries=64, show_spinner=False)
def _expand_cached(name: str, data: bytes) -> List[Tuple[str, bytes, Optional[datetime]]]:
    return [(s.name, s.data, s.start_hint) for s in expand(name, data)]


@st.cache_data(max_entries=128, show_spinner=False)
def _parse_cached(name: str, data: bytes, hint: Optional[datetime], opts_json: str) -> ParsedFile:
    o = json.loads(opts_json)
    o["roles"] = {int(k): v for k, v in o["roles"].items()}
    o["start"] = datetime.fromisoformat(o["start"]) if o["start"] else None
    return parse_source(Source(name, data, hint), ImportOptions(**o))


def _opts_json(o: ImportOptions) -> str:
    d = dict(o.__dict__)
    d["start"] = o.start.isoformat() if o.start else None
    return json.dumps(d, sort_keys=True, default=str)


def sources_from_uploads(files) -> List[Tuple[str, bytes, Optional[datetime]]]:
    out, errors = [], []
    for f in files:
        try:
            out.extend(_expand_cached(f.name, f.getvalue()))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    for e in errors:
        st.sidebar.error(e)
    return out


def _options_state() -> Dict[str, ImportOptions]:
    return st.session_state.setdefault("import_opts", {})


def parse_all(sources) -> List[Tuple[str, ImportOptions, Optional[ParsedFile], Optional[str]]]:
    state = _options_state()
    out = []
    for name, data, hint in sources:
        k = _key(name, data)
        o = state.setdefault(k, ImportOptions())
        try:
            out.append((k, o, _parse_cached(name, data, hint, _opts_json(o)), None))
        except Exception as exc:  # noqa: BLE001
            out.append((k, o, None, str(exc)))
    return out


def render_review(sources) -> Optional[Tuple[list, List[str]]]:
    """Main-area import review. Returns (subjects, notes) when the user clicks Import."""
    results = parse_all(sources)
    ok = [r for r in results if r[2] is not None]
    st.subheader("Import data")
    st.caption(f"{len(results)} file(s) · {len(ok)} readable · check the detected format, time and columns, then import.")
    names = [n for n, _, _ in sources]
    tabs = st.tabs([("✔ " if r[2] is not None and not (r[2].date_ambiguous or r[2].needs_interval) else "⚠ " if r[2] is not None else "✖ ") + n for r, n in zip(results, names)]) if len(results) <= 12 else None
    if tabs is None:
        pick = st.selectbox("File", range(len(results)), format_func=lambda i: names[i])
        containers = {pick: st.container()}
    else:
        containers = dict(enumerate(tabs))
    changed = False
    for i, cont in containers.items():
        k, o, pf, err = results[i]
        with cont:
            changed |= _file_panel(k, o, pf, err)
    if changed:
        st.rerun()
    c1, c2, c3 = st.columns([3, 1, 1])
    mode = c1.selectbox("Combine files", list(MERGE_MODES), format_func=MERGE_MODES.get, disabled=len(ok) < 2)
    if c3.button("Cancel", width="stretch"):
        st.session_state.import_review = False
        st.rerun()
    n_subj = sum(len(r[2].subjects) for r in ok)
    merged = n_subj > 1 and will_merge([r[2] for r in ok], mode)  # same rule as combine(), on the preview rows
    label = "Import (→ 1 recording)" if merged else f"Import ({n_subj} subject{'s' if n_subj != 1 else ''})"
    if c2.button(label, type="primary", width="stretch", disabled=not ok):
        full = []
        for (name, data, hint), (k, o, pf, err) in zip(sources, results):
            if pf is not None:
                full.append(pf)
        subjects, notes = combine(full, mode)
        return subjects, notes
    return None


def _file_panel(key: str, o: ImportOptions, pf: Optional[ParsedFile], err: Optional[str]) -> bool:
    """Options for one file. Returns True when an option changed (re-parse)."""
    before = _opts_json(o)
    if err:
        st.error(err)
    if pf is not None:
        st.markdown(f"**Detected:** :blue[{pf.format}] &nbsp; · &nbsp; {pf.summary()}")
        for n in pf.notes:
            st.caption(f"• {n}")
        if pf.date_ambiguous:
            st.warning("Dates are ambiguous (all day numbers ≤ 12). Check the first / last sample below and set the date order if needed.")
    c1, c2, c3 = st.columns(3)
    o.date_order = c1.selectbox("Date order", list(DATE_ORDERS), index=list(DATE_ORDERS).index(o.date_order), format_func=DATE_ORDERS.get, key=f"do-{key}")
    if pf is None or pf.numeric_time or o.time_unit:
        o.time_unit = c2.selectbox("Time unit (numeric time column)", list(TIME_UNITS), index=list(TIME_UNITS).index(o.time_unit), format_func=TIME_UNITS.get, key=f"tu-{key}")
    iv_on = c3.checkbox("Set sampling interval" + (" (no time column)" if pf is None or pf.needs_interval else ""), value=o.interval_s is not None, key=f"ivon-{key}")
    o.interval_s = c3.number_input("Interval (s)", min_value=0.001, value=float(o.interval_s or 60.0), key=f"iv-{key}") if iv_on else None
    if pf is None or pf.relative_time or pf.needs_interval or o.start is not None:
        s1, s2, s3 = st.columns([1, 1, 2])
        use_start = s1.checkbox("Set recording start", value=o.start is not None, key=f"son-{key}", help="The file has no calendar date: phases are measured from this start.")
        if use_start:
            d = s2.date_input("Start date", value=(o.start or datetime.now()).date(), key=f"sd-{key}")
            t = s3.time_input("Start time", value=(o.start.time() if o.start else dt_time(0, 0)), key=f"st-{key}")
            o.start = datetime.combine(d, t)
        else:
            o.start = None
    h1, h2 = st.columns([1, 3])
    manual = h1.checkbox("Header row manually", value=o.header_line is not None, key=f"hon-{key}")
    if manual:
        o.header_line = int(h2.number_input("Column names on line", min_value=1, max_value=500, value=(o.header_line or (pf.header_line if pf and pf.header_line is not None else 0)) + 1, key=f"hl-{key}")) - 1
    else:
        o.header_line = None
        if pf is not None:
            h2.caption(f"auto: column names on line {pf.header_line + 1}, data from line {pf.data_line + 1}" if pf.header_line is not None else f"auto: no column-name row, data from line {pf.data_line + 1}")
    if pf is not None and (pf.can_layout or o.layout != "auto"):
        cur = o.layout if o.layout != "auto" else pf.layout
        lay = st.radio("Layout", ["channels", "subjects"], index=["channels", "subjects"].index(cur), horizontal=True, key=f"lay-{key}",
                       format_func=lambda x: "Columns are measurements of one recording" if x == "channels" else "Columns are separate subjects / samples")
        o.layout = lay if lay != pf.layout or o.layout != "auto" else o.layout
    if pf is not None and pf.columns:
        editable = pf.format not in ("TriKinetics DAM monitor", "Apple Health export.xml", "Empatica E4", "Fitbit JSON", "Fitbit intraday JSON", "Time-course table (ZT / CT columns)", "RR / inter-beat intervals", "ActiGraph .agd")
        cols_df = pd.DataFrame({"Column": [c.name for c in pf.columns], "Used as": [ROLE_LABELS.get(o.roles.get(i, c.role), ROLE_LABELS.get(c.role, c.role)) for i, c in enumerate(pf.columns)], "First value": [c.example for c in pf.columns]})
        if editable:
            edited = st.data_editor(
                cols_df,
                column_config={"Used as": st.column_config.SelectboxColumn(options=[ROLE_LABELS[r] for r in COL_ROLES], required=True), "Column": st.column_config.TextColumn(disabled=True), "First value": st.column_config.TextColumn(disabled=True)},
                hide_index=True,
                width="stretch",
                key=f"roles-{key}",
            )
            for i, (lab, orig) in enumerate(zip(edited["Used as"], cols_df["Used as"])):
                if lab != orig:
                    o.roles[i] = LABEL_TO_ROLE.get(lab, "ignore")
            if o.roles and st.button("Reset column roles", key=f"rr-{key}"):
                o.roles.clear()
        else:
            st.dataframe(cols_df, hide_index=True, width="stretch")
        if pf.subjects and len(pf.subjects[0].t):
            s0 = pf.subjects[0]
            st.caption(f"First sample {s0.t[0]:%a %Y-%m-%d %H:%M:%S} · last {s0.t[-1]:%a %Y-%m-%d %H:%M:%S} · channels: {', '.join(s0.describe())}")
    if pf is not None:
        with st.expander("File preview", expanded=False):
            lines = [f"{i + 1:>4} {'▶' if i == pf.header_line else '│'} {ln[:300]}" for i, ln in enumerate(pf.preview)]
            st.code("\n".join(lines), language=None)
            st.caption("▶ marks the column-name row" + (f"; data start on line {pf.data_line + 1}" if pf.data_line else ""))
    return _opts_json(o) != before
