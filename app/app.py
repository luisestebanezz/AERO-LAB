"""HORYZN wing aerodynamics lab - local interactive app (a stand-alone study of the wing, derived from the scout configurator).

Start with run.bat (or: python -m streamlit run app/app.py). Everything runs on this PC.
Tables are drawn as HTML because the 'pyarrow' library is blocked by a Windows security policy on this PC.
"""
from __future__ import annotations

import csv
import datetime
import html
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wingaero import airfoil as AF  # noqa: E402
from wingaero import figures as FG  # noqa: E402
from wingaero.analysis import Case, safety_factors, with_safety  # noqa: E402
from wingaero.geometry import wing_area  # noqa: E402
from wingaero import explore as EX  # noqa: E402
from wingaero import report as RP  # noqa: E402
from wingaero import valleys as VL  # noqa: E402
from wingaero.optimise import OBJECTIVES, Problem, free_variables, metrics_sf, optimise  # noqa: E402
from wingaero.points import CRUISE, STANDARD_POINTS, FlightPoint, cases_for, from_dicts, point_list, to_dicts  # noqa: E402
from wingaero.params import (BY_KEY, DEFAULT_AIRFOIL_POOL, GROUPS, MINIMUM_GROUP, MINIMUM_INPUTS, MINIMUM_WHY, PARAMS, RANGE, active, all_auto, auto_inputs,  # noqa: E402
                             auto_kind, default_fixed, default_lists, default_ranges, defaults, from_json_full, mode_of, options_of, range_options, resolve,
                             to_json, validate)

HZ = {"navy": "#1D3043", "ink": "#1F2933", "orange": "#C2410C", "blue": "#2C6596", "amber": "#E0A040", "slate": "#5B6470", "steel": "#6B7A89",
      "line": "#E5E7EB", "green": "#2E7D5B", "red": "#B23B3B"}
CONFIGURATOR_DESIGNS = ROOT.parent / "configurator" / "data" / "designs"
PRESETS = ROOT / "presets"
ONLINE = (ROOT / "online.flag").exists()      # only the online copy (tools/make_online_copy.py) has this file: a small shared server

st.set_page_config(page_title="HORYZN wing aerodynamics lab", page_icon="✈️", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 1.1rem; max-width: 1600px;}}
.hz-banner {{background: linear-gradient(100deg, {HZ['navy']} 0%, #23405C 55%, {HZ['blue']} 100%); color: #fff; border-radius: 12px;
  padding: 14px 22px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px;
  box-shadow: 0 2px 10px rgba(29,48,67,.18); border-bottom: 3px solid {HZ['orange']};}}
.hz-brand {{font-weight: 800; letter-spacing: .32em; font-size: 1.35rem;}}
.hz-brand small {{display: block; letter-spacing: .04em; font-weight: 400; font-size: .86rem; opacity: .85; margin-top: 2px;}}
.hz-chip {{background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.28); border-radius: 999px; padding: 3px 11px; font-size: .78rem; margin-left: 6px; white-space: nowrap;}}
.hz-cards {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 4px 0 12px;}}
.hz-card {{background: #fff; border: 1px solid {HZ['line']}; border-top: 3px solid {HZ['orange']}; border-radius: 9px; padding: 8px 12px;}}
.hz-card .lbl {{font-size: .70rem; text-transform: uppercase; letter-spacing: .06em; color: {HZ['slate']};}}
.hz-card .val {{font-size: 1.28rem; font-weight: 700; color: {HZ['ink']}; line-height: 1.35;}}
.hz-card .sub {{font-size: .74rem; color: {HZ['steel']};}}
.hz-card.ok {{border-top-color: {HZ['green']};}} .hz-card.bad {{border-top-color: {HZ['red']};}} .hz-card.blue {{border-top-color: {HZ['blue']};}} .hz-card.amber {{border-top-color: {HZ['amber']};}}
table.hz {{border-collapse: collapse; width: 100%; font-size: .83rem; background: #fff;}}
table.hz th {{background: {HZ['navy']}; color: #fff; text-align: left; padding: 6px 9px; position: sticky; top: 0; z-index: 1; font-weight: 600;}}
table.hz td {{padding: 4px 9px; border-bottom: 1px solid {HZ['line']}; vertical-align: top;}}
table.hz tr:nth-child(even) td {{background: #F8FAFC;}}
table.hz tr.group td {{background: #E6ECF2 !important; font-weight: 700; color: {HZ['navy']}; text-transform: uppercase; font-size: .72rem; letter-spacing: .06em;}}
.hz-wrap {{border: 1px solid {HZ['line']}; border-radius: 8px; overflow: auto;}}
.hz-h {{font-weight: 700; color: {HZ['navy']}; border-left: 4px solid {HZ['orange']}; padding-left: 9px; margin: 18px 0 8px; font-size: 1.05rem;}}
.hz-note {{color: {HZ['slate']}; font-size: .82rem;}}
.hz-warn {{background: #FFF6E5; border: 1px solid {HZ['amber']}; border-radius: 8px; padding: 6px 10px; font-size: .84rem; margin: 4px 0;}}
.hz-bad {{background: #F8E1E1; border: 1px solid {HZ['red']}; border-radius: 8px; padding: 6px 10px; font-size: .84rem; margin: 4px 0;}}
.hz-side-brand {{font-weight: 800; letter-spacing: .3em; font-size: 1.15rem; color: #fff; padding: 4px 0 2px;}}
.hz-side-sub {{font-size: .78rem; color: #C9D6E3; margin-bottom: 8px;}}
div[data-testid="stTabs"] button[role="tab"] p {{font-weight: 600;}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ small drawing helpers
def fmt(v, nd: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not math.isfinite(v):
            return "–"
        if v == 0:
            return "0"
        return f"{v:,.{nd}g}" if abs(v) < 1e5 else f"{v:,.0f}"
    return str(v)


def table(rows: list[dict], cols: list[str] | None = None, height: int | None = None) -> None:
    if not rows:
        st.markdown("<span class='hz-note'>(nothing to show)</span>", unsafe_allow_html=True)
        return
    cols = cols or list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = ""
    for r in rows:
        if r.get("_group"):
            body += f"<tr class='group'><td colspan='{len(cols)}'>{html.escape(str(r['_group']))}</td></tr>"
            continue
        body += "<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
    style = f"max-height:{height}px;" if height else ""
    st.markdown(f"<div class='hz-wrap' style='{style}'><table class='hz'><tr>{head}</tr>{body}</table></div>", unsafe_allow_html=True)


def cards(items: list[tuple]) -> None:
    out = "".join(f"<div class='hz-card {k}'><div class='lbl'>{html.escape(l)}</div><div class='val'>{v}</div><div class='sub'>{s}</div></div>" for l, v, s, k in items)
    st.markdown(f"<div class='hz-cards'>{out}</div>", unsafe_allow_html=True)


def heading(text: str) -> None:
    st.markdown(f"<div class='hz-h'>{html.escape(text)}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='hz-note'>{text}</div>", unsafe_allow_html=True)


def chart(fig, key: str) -> None:
    st.plotly_chart(fig, theme=None, key=key)


# ------------------------------------------------------------------------------------------------ input state
# PV: the value of every input; PF: the state of every input (True = a value, False = a range or a list of options, None = auto; see params.py);
# PR: the range of every numeric input; PL: the allowed airfoils of an airfoil station, the allowed options of a choice or a yes/no;
# PRG: whether the range switch was last on (kept while the input is auto, so that switching the input on again restores it).
def init_state() -> None:
    ss = st.session_state
    if "PV" not in ss:
        ss["PV"], ss["PF"] = defaults(), default_fixed()
        ss["PR"], ss["PL"] = default_ranges(), default_lists()
        ss["PRG"] = {k: False for k in BY_KEY}
        ss["PT"] = to_dicts(STANDARD_POINTS)
        ss["PW"] = {"ld": 1.0}
    ss.setdefault("opt", None)
    ss.setdefault("picked", None)
    ss.setdefault("valley", None)


def clamp(p, v):
    if p.kind == "float":
        v = float(v)
        return min(max(v, p.lo), p.hi) if p.lo is not None else v
    if p.kind == "choice":
        return v if v in p.choices else p.default
    if p.kind == "airfoil":
        return v if (v in AF.library_names() or v == AF.PARAM) else p.default
    if p.kind == "bool":
        return bool(v)
    return v


def _drop_widgets(key: str) -> None:
    for prefix in ("v_", "a_", "m_", "au_", "rl_", "rh_", "al_"):
        st.session_state.pop(prefix + key, None)


def apply_inputs(values: dict, fixed: dict, ranges: dict | None = None, lists: dict | None = None) -> None:
    """Replace inputs (a callback: runs before the widgets are drawn)."""
    ss = st.session_state
    for k, p in BY_KEY.items():
        if k in values:
            ss["PV"][k] = clamp(p, values[k])
        if k in fixed:
            f = fixed[k]
            ss["PF"][k] = True if (f is None and k in MINIMUM_INPUTS) else (None if f is None else bool(f))
            if f is not None:
                ss["PRG"][k] = not bool(f)
        if ranges and k in ranges and p.kind == "float":
            lo, hi = sorted((float(ranges[k][0]), float(ranges[k][1])))
            ss["PR"][k] = (min(max(lo, p.lo), p.hi), min(max(hi, p.lo), p.hi))
        if lists and lists.get(k):
            if p.kind == "airfoil":
                ss["PL"][k] = [n for n in lists[k] if n in AF.library_names() or n == AF.PARAM] or list(DEFAULT_AIRFOIL_POOL)
            elif p.kind in ("choice", "bool"):
                ss["PL"][k] = [x for x in lists[k] if x in options_of(p)] or options_of(p)
        _drop_widgets(k)


def reset_inputs() -> None:
    apply_inputs(defaults(), default_fixed(), default_ranges(), default_lists())
    st.session_state["PRG"] = {k: False for k in BY_KEY}
    st.session_state["PT"] = to_dicts(STANDARD_POINTS)
    st.session_state["PW"] = {"ld": 1.0}
    for k in [k for k in st.session_state.keys() if str(k).startswith(("pt_", "w_"))]:
        st.session_state.pop(k, None)
    st.session_state["opt"] = None
    st.session_state["picked"] = None
    st.session_state["valley"] = None


def set_all_auto() -> None:
    """Every input auto except the minimum inputs (which keep being a value or a range)."""
    ss = st.session_state
    for k, f in all_auto(ss["PF"]).items():
        ss["PF"][k] = f
        _drop_widgets(k)


def set_all_values() -> None:
    """Every input back to a value (the overrides back off), the ranges as they were."""
    ss = st.session_state
    for k, f in default_fixed().items():
        ss["PF"][k] = f
        ss["PRG"][k] = False
        _drop_widgets(k)


def go_optimise() -> None:
    st.session_state["page"] = "Optimise"


def load_configurator_wing() -> None:
    f = sorted(CONFIGURATOR_DESIGNS.glob("FW_recommended_*.json"))
    if not f:
        return
    d = json.loads(f[0].read_text(encoding="utf-8-sig"))
    v = {"span_m": d["span"], "aspect_ratio": d["aspect_ratio"], "n_panels": 1, "taper_tip": d["taper"], "sweep_1": d.get("sweep_deg", 0.0),
         "washout_tip": d.get("washout_deg", 0.0), "twist_mode": "linear", "incidence_deg": d.get("incidence_deg", 2.0), "foil_root": d["airfoil"],
         "foil_tip": d["airfoil"], "dihedral_1": d.get("wing_dihedral_deg", 0.0), "mass_kg": 24.9}
    apply_inputs(v, {k: True for k in v})


def apply_upload() -> None:
    up = st.session_state.get("upload")
    if up is None:
        return
    text = up.getvalue().decode("utf-8")
    vals, fx, rng, lst = from_json_full(text)
    apply_inputs(vals, fx, rng, lst)
    pts = from_dicts(json.loads(text).get("points"))
    if pts:
        st.session_state["PT"] = to_dicts(pts[:1] + pts[1:6])
        _drop_point_widgets()


def apply_optimised() -> None:
    r = st.session_state.get("opt")
    if r is not None:
        apply_inputs(r.values, r.fixed)


def apply_point(inputs: dict) -> None:
    apply_inputs(inputs, {k: True for k in inputs})


def pick_point(p: dict) -> None:
    st.session_state["picked"] = p


# ------------------------------------------------------------------------------------------------ flight points and objective weights (shared by Optimise, Explore, Valleys, Results)
# PT: the flight points as dictionaries (points.py): the same wing in several flight conditions, each with its weight in the objective and its own least speed margin;
# PW: the weight of every objective (the sliders of the Optimise page), kept when the page is left.
def points_now() -> list:
    pts = from_dicts(st.session_state["PT"])
    return pts if pts else [CRUISE]


def weights_now(p: dict | None = None) -> dict:
    w = {k: float(x) for k, x in st.session_state["PW"].items() if k in OBJECTIVES and x}
    if p is not None and not p.get("aileron_on"):
        w.pop("roll", None)                                      # roll power needs the ailerons
    return w


def _drop_point_widgets() -> None:
    for k in [k for k in st.session_state.keys() if str(k).startswith("pt_")]:
        st.session_state.pop(k, None)


def standard_points() -> None:
    st.session_state["PT"] = to_dicts(STANDARD_POINTS)
    _drop_point_widgets()


def cruise_only() -> None:
    st.session_state["PT"] = to_dicts([CRUISE])
    _drop_point_widgets()


def add_point() -> None:
    st.session_state["PT"] += to_dicts([FlightPoint(f"Point {len(st.session_state['PT']) + 1}", weight=0.0)])
    _drop_point_widgets()


def remove_point(i: int) -> None:
    if 0 < i < len(st.session_state["PT"]):
        st.session_state["PT"].pop(i)
        _drop_point_widgets()


def points_summary() -> str:
    pts = points_now()
    if len(pts) == 1:
        return "One flight point: the design point of the inputs. Add more on the Optimise page (Flight points)."
    return (f"{len(pts)} flight points: " + "; ".join(pt.label() + (f", weight {pt.weight:g}" if pt.weight else ", limits only") for pt in pts)
            + ". Edit them on the Optimise page.")


def points_editor() -> None:
    """One row per flight point: on, name, speed (x the design speed), altitude (+ m), mass (x), load factor (0: as the inputs), weight in the objective, least speed
    margin (0: the limit of the inputs). The first point is the design point of the inputs: only its name and weight can change."""
    ss = st.session_state
    widths = [0.45, 2.0, 0.95, 1.05, 0.95, 1.05, 0.95, 1.1, 0.45]
    head = st.columns(widths)
    for c, t, h in zip(head, ["on", "Flight point", "Speed ×", "Altitude + m", "Mass ×", "Load factor n", "Weight", "Least margin", ""],
                       ["", "", "Share of the design speed of the inputs.", "Added to the altitude of the inputs.", "Share of the mass of the inputs.",
                        "0: the load factor of the inputs.", "Share of the objective (they are normalised to add up to one); 0: the point only checks the limits.",
                        "Least design speed / stall speed with the safety factor at this point; 0: the limit of the inputs.", ""]):
        c.markdown(f"<span class='hz-note' title='{html.escape(h)}'><b>{t}</b></span>", unsafe_allow_html=True)
    for i, d in enumerate(ss["PT"]):
        cs = st.columns(widths, vertical_alignment="center")
        first = i == 0

        def seed(key, val):
            if key not in ss:
                ss[key] = val
        seed(f"pt_{i}_on", bool(d.get("on", True)) or first)
        cs[0].checkbox("on", key=f"pt_{i}_on", disabled=first, label_visibility="collapsed")
        seed(f"pt_{i}_name", d.get("name", f"Point {i + 1}"))
        cs[1].text_input("name", key=f"pt_{i}_name", label_visibility="collapsed")
        seed(f"pt_{i}_sp", float(d.get("speed_factor", 1.0)))
        cs[2].number_input("speed", 0.3, 2.0, step=0.05, key=f"pt_{i}_sp", disabled=first, label_visibility="collapsed", format="%g")
        seed(f"pt_{i}_alt", float(d.get("altitude_delta_m", 0.0)))
        cs[3].number_input("altitude", -500.0, 4000.0, step=100.0, key=f"pt_{i}_alt", disabled=first, label_visibility="collapsed", format="%g")
        seed(f"pt_{i}_ms", float(d.get("mass_factor", 1.0)))
        cs[4].number_input("mass", 0.3, 1.5, step=0.05, key=f"pt_{i}_ms", disabled=first, label_visibility="collapsed", format="%g")
        seed(f"pt_{i}_n", float(d["load_factor"]) if d.get("load_factor") is not None else 0.0)
        cs[5].number_input("load factor", 0.0, 6.0, step=0.1, key=f"pt_{i}_n", disabled=first, label_visibility="collapsed", format="%g")
        seed(f"pt_{i}_w", float(d.get("weight", 1.0)))
        cs[6].number_input("weight", 0.0, 1.0, step=0.05, key=f"pt_{i}_w", label_visibility="collapsed", format="%g")
        seed(f"pt_{i}_mg", float(d["margin_min"]) if d.get("margin_min") is not None else 0.0)
        cs[7].number_input("margin", 0.0, 2.0, step=0.01, key=f"pt_{i}_mg", disabled=first, label_visibility="collapsed", format="%g")
        cs[8].button("✕", key=f"pt_{i}_rm", on_click=remove_point, args=(i,), disabled=first, help="Remove this point.")
        d.update({"on": True if first else bool(ss[f"pt_{i}_on"]), "name": ss[f"pt_{i}_name"] or f"Point {i + 1}", "weight": float(ss[f"pt_{i}_w"])})
        if not first:
            d.update({"speed_factor": float(ss[f"pt_{i}_sp"]), "altitude_delta_m": float(ss[f"pt_{i}_alt"]), "mass_factor": float(ss[f"pt_{i}_ms"]),
                      "load_factor": float(ss[f"pt_{i}_n"]) if ss[f"pt_{i}_n"] > 0 else None, "margin_min": float(ss[f"pt_{i}_mg"]) if ss[f"pt_{i}_mg"] > 0 else None})
    b1, b2, b3, _ = st.columns([1, 1.3, 1, 3])
    b1.button("Add a point", on_click=add_point, disabled=len(ss["PT"]) >= 6, key="pt_add")
    b2.button("The standard three points", on_click=standard_points, key="pt_std", help="Cruise; slow flight (0.85 x speed, least margin 1.05, weight 0.5); manoeuvre or gust (1.3 x speed, n = 2.5, limits only).")
    b3.button("Cruise only", on_click=cruise_only, key="pt_one")


def point_rows(c: Case, pts: list) -> list[dict]:
    """The key results of a Case at every flight point (design point first), with the safety factor."""
    rows = []
    for i, (pt, cc) in enumerate(zip(point_list(pts), cases_for(c, pts))):
        m = metrics_sf(cc, set())
        lim = float(pt.margin_min) if (i > 0 and pt.margin_min is not None) else float(c.p["speed_margin_min"])
        if not m.get("ok"):
            rows.append({"Flight point": pt.label(), "Speed m/s": fmt(cc.V), "n": fmt(cc.n), "Result": m.get("reason", "no design solution")})
            continue
        rows.append({"Flight point": pt.label(), "Speed m/s": fmt(cc.V), "n": fmt(cc.n), "CL": fmt(m["CL_design"]), "L/D (SF)": fmt(m["LD_sf"]), "Stall speed m/s (SF)": fmt(m["V_stall_mps_sf"]),
                     "Speed margin (SF)": fmt(m["speed_margin_sf"]) + (" ✓" if m["speed_margin_sf"] >= lim else f" ✗ (< {fmt(lim)})"),
                     "Strip utilisation (SF)": fmt(m["util_max_sf"]) + (" ✓" if m["util_max_sf"] <= 1.0 else " ✗"),
                     "Stall starts at η": fmt(m["eta_stall_first"]) + (" ✓" if m["eta_stall_first"] <= c.p["stall_onset_max"] else " ✗")})
    return rows


ON_HELP = ("On: you give this input, as a value or, with the range switch on, as a range. Off (auto): the lab decides. A design input (aspect ratio, taper, sweep, twist, "
           "airfoils, ...) is searched by the optimiser over everything it allows; an input that describes the environment, a calibration or a model switch takes the lab's "
           "default; an override (wing area, wing loading, design lift coefficient) is not used.")
RANGE_HELP = ("Off: you give one value. On: you give a minimum and a maximum (an airfoil station or a choice: a list of allowed options) and the lab searches inside it "
              "(optimiser, sampling, sweeps). For a plain analysis the input takes the middle of its range.")
AUTO_UNUSED = {"area_m2": "off: the aspect ratio decides the area", "wing_loading_Nm2": "off: the aspect ratio decides the area",
               "design_cl": "off: the design speed decides the lift coefficient"}


def unit_of(p) -> str:
    return "" if p.unit == "-" else " " + p.unit


def show_option(p, x) -> str:
    """An option of an airfoil station, a choice or a yes/no as text."""
    if p.kind == "airfoil":
        return "parametric NACA" if x == AF.PARAM else AF.display_name(x)
    return fmt(x)


def auto_text(p) -> str:
    """What the lab does when this input is auto."""
    kind = auto_kind(p)
    if kind == "unused":
        return AUTO_UNUSED.get(p.key, "off: not used")
    if kind == "searched":
        if p.key.startswith("naca_"):
            return f"auto: used only when the airfoil is the parametric NACA (then searched from {fmt(p.lo)} to {fmt(p.hi)}; default {fmt(p.default)})"
        if p.kind == "airfoil":
            return f"auto: the optimiser picks from {len(DEFAULT_AIRFOIL_POOL)} airfoils (the default is {show_option(p, p.default)})"
        return f"auto: searched from {fmt(p.lo)} to {fmt(p.hi)}{unit_of(p)} (default {fmt(p.default)})"
    extra = " (a range lets the optimiser choose)" if p.optimisable else ""
    return f"auto: default {show_option(p, p.default)}{unit_of(p) if p.kind == 'float' else ''}{extra}"


def param_row(p) -> None:
    """One input: an on / off switch (off = auto), a range switch, and the value, the range or the lab's choice."""
    ss = st.session_state
    k = p.key
    vk, ak, mk, lk, hk, lsk = f"v_{k}", f"a_{k}", f"m_{k}", f"rl_{k}", f"rh_{k}", f"al_{k}"
    locked = k in MINIMUM_INPUTS
    state = ss["PF"][k]
    if locked:
        ss[ak] = True
    elif ak not in ss:
        ss[ak] = state is not None
    if mk not in ss:
        ss[mk] = bool(ss["PRG"][k]) if state is None else (state is False)
    c_on, c_rg, c_main, c_unit = st.columns([1.1, 1.6, 4.4, 0.7])
    c_on.toggle("on", key=ak, disabled=locked)
    c_rg.toggle("range", key=mk, disabled=not ss[ak])
    on, rng = bool(ss[ak]), bool(ss[mk])
    ss["PRG"][k] = rng
    ss["PF"][k] = None if not on else (False if rng else True)
    hint = (p.help or "").strip() or None
    if not on:
        c_main.text_input(p.label, value=auto_text(p), disabled=True, key=f"au_{k}", help=hint)
        return
    if rng:
        if p.kind == "float":
            lo0, hi0 = ss["PR"][k]
            if tuple(ss["PR"][k]) == tuple(default_ranges()[k]):          # untouched: a range around the current value
                lo0, hi0 = default_ranges({k: ss["PV"][k]})[k]
            ss.setdefault(lk, float(lo0))
            ss.setdefault(hk, float(hi0))
            a, b = c_main.columns(2)
            step = float(p.step or (p.hi - p.lo) / 100.0)
            a.number_input(p.label + ": minimum", min_value=float(p.lo), max_value=float(p.hi), step=step, key=lk, format="%g", help=hint)
            b.number_input("maximum", min_value=float(p.lo), max_value=float(p.hi), step=step, key=hk, format="%g")
            ss["PR"][k] = tuple(sorted((float(ss[lk]), float(ss[hk]))))
        else:
            options = AF.library_names() + [AF.PARAM] if p.kind == "airfoil" else options_of(p)
            ss.setdefault(lsk, [n for n in ss["PL"][k] if n in options])
            c_main.multiselect(p.label + (": allowed airfoils" if p.kind == "airfoil" else ": allowed options"), options, key=lsk, help=hint,
                               format_func=lambda n, p=p: "parametric NACA 4-digit (its three inputs below)" if (p.kind == "airfoil" and n == AF.PARAM) else show_option(p, n))
            ss["PL"][k] = list(ss[lsk]) or (list(DEFAULT_AIRFOIL_POOL) if p.kind == "airfoil" else options_of(p))
    else:
        if vk not in ss:
            ss[vk] = ss["PV"][k]
        if p.kind == "bool":
            c_main.checkbox(p.label, key=vk, help=hint)
        elif p.kind == "choice":
            c_main.selectbox(p.label, list(p.choices), key=vk, help=hint)
        elif p.kind == "float":
            c_main.number_input(p.label, min_value=float(p.lo), max_value=float(p.hi), step=float(p.step or (p.hi - p.lo) / 100.0), key=vk, help=hint, format="%g")
        else:
            c_main.selectbox(p.label, AF.library_names() + [AF.PARAM], key=vk, help=hint,
                             format_func=lambda n: "parametric NACA 4-digit (its three inputs below)" if n == AF.PARAM else AF.display_name(n))
        ss["PV"][k] = ss[vk]
    c_unit.caption("" if p.unit == "-" else p.unit)


def render_group(group: str) -> None:
    for p in PARAMS:
        if p.group != group:
            continue
        _, f, pres = current()
        if active(p, pres, range_options(f, dict(st.session_state["PL"]))):
            param_row(p)


def current() -> tuple[dict, dict, dict]:
    """(values, states, resolved inputs) as they are now: an input given as a range takes the middle of its range, an auto input the lab's default."""
    ss = st.session_state
    v, f = dict(ss["PV"]), dict(ss["PF"])
    return v, f, resolve(v, f, dict(ss["PR"]), dict(ss["PL"]))


def ranges_lists() -> tuple[dict, dict]:
    return dict(st.session_state["PR"]), dict(st.session_state["PL"])


def inputs_json() -> str:
    v, f, _ = current()
    r, l = ranges_lists()
    return to_json(v, f, r, l, to_dicts(points_now()))


@st.cache_resource(max_entries=48, show_spinner=False)
def get_case(p_json: str, name: str = "") -> Case:
    return Case(json.loads(p_json), name)


def case_of(p: dict, name: str = "") -> Case:
    return get_case(json.dumps(p, sort_keys=True, default=float), name)


def range_rows(vs, f: dict) -> list[dict]:
    """The searched inputs as table rows: where they come from (a range or auto), their minimum and maximum, or their allowed options."""
    rows = []
    for var in vs:
        pr = BY_KEY[var.key]
        src = "auto" if mode_of(f, var.key) == "auto" else "range"
        note_ = "environment or calibration: not optimised" if (src == "range" and not pr.optimisable) else ""
        if var.kind == "float":
            rows.append({"Input": pr.label.strip(), "Unit": "" if pr.unit == "-" else pr.unit, "Given as": src, "Minimum": fmt(var.lo), "Maximum": fmt(var.hi), "Note": note_})
        else:
            names = ", ".join(show_option(pr, n) for n in var.pool[:6]) + (" …" if len(var.pool) > 6 else "")
            rows.append({"Input": pr.label.strip(), "Unit": "", "Given as": src, "Minimum": f"{len(var.pool)} options", "Maximum": names, "Note": note_})
    return rows


def auto_rows(v: dict, f: dict, r: dict, p: dict, lists: dict) -> dict[str, list[dict]]:
    """The auto inputs that are used, by what the lab does with them: searched by the optimiser, default, unused."""
    out = {"searched": [], "default": [], "unused": []}
    opts = range_options(f, lists)
    searched = {x.key for x in free_variables(v, f, r, None, lists)}
    for q in auto_inputs(f):
        if not active(q, p, opts) or (q.key.startswith("naca_") and q.key not in searched):
            continue
        kind = "searched" if q.key in searched else ("unused" if auto_kind(q) == "unused" else "default")
        out[kind].append({"Input": q.label.strip(), "Group": q.group, "What the lab does": auto_text(q).replace("auto: ", "").replace("off: ", "")})
    return out


# ------------------------------------------------------------------------------------------------ the numbers table
ROWS = [
    ("Geometry", None), ("span_m", "Wingspan", "m", False), ("area_m2", "Wing area", "m²", False), ("aspect_ratio", "Aspect ratio", "-", False),
    ("root_chord_m", "Root chord", "m", False), ("tip_chord_m", "Tip chord", "m", False), ("mac_m", "Mean aerodynamic chord", "m", False),
    ("mean_sweep_deg", "Mean quarter-chord sweep", "°", False), ("wing_loading_Nm2", "Wing loading", "N/m²", False),
    ("Flight condition", None), ("mass_kg", "Mass", "kg", False), ("V_design_mps", "Design speed", "m/s", False), ("Re_mac_design", "Reynolds number at the MAC", "-", False),
    ("CL_design", "Design lift coefficient", "-", False), ("alpha_design_deg", "Root angle of attack at the design point", "°", False),
    ("alpha_fus_design_deg", "Fuselage angle of attack at the design point", "°", False),
    ("Lift and stall", None), ("CL_alpha_per_deg", "Lift slope", "per °", False), ("alpha_zero_lift_deg", "Zero-lift angle", "°", False),
    ("CL_max_first", "Usable maximum lift (first strip stalls)", "-", True), ("alpha_stall_first_deg", "Root angle at the first-strip stall", "°", False),
    ("eta_stall_first", "Where the stall starts (share of the semi-span)", "-", False), ("CL_max_nl", "Peak of the non-linear lift curve", "-", False),
    ("alpha_stall_nl_deg", "Root angle at the peak", "°", False), ("V_stall_mps", "Stall speed", "m/s", True), ("speed_margin", "Design speed / stall speed", "-", True),
    ("util_max", "Highest strip utilisation at the design point", "-", True),
    ("Drag and power", None), ("CD", "Drag coefficient at the design point", "-", True), ("CDp", "  of which profile", "-", False), ("CDi", "  of which induced", "-", False),
    ("CDf", "  of which fuselage", "-", False), ("LD", "Lift-to-drag at the design point", "-", True), ("e_design", "Span efficiency at the design point", "-", False),
    ("P_req_W", "Aerodynamic power D·V at the design point", "W", True), ("LD_max", "Best lift-to-drag", "-", True), ("CL_at_LD_max", "  at CL", "-", False),
    ("V_best_glide", "  at speed", "m/s", False), ("P_min_W", "Least aerodynamic power", "W", True), ("V_min_power", "  at speed", "m/s", False),
    ("endurance_factor", "Endurance factor CL^1.5 / CD (best)", "-", True),
    ("Pitching moment", None), ("x_ac_mac", "Aerodynamic centre (share of the MAC)", "-", False), ("Cm_ref", "Cm about the reference point at the design point", "-", True),
    ("Cm_ac", "Cm about the aerodynamic centre", "-", True), ("dCm_dCL_ref", "dCm/dCL about the reference point", "-", False), ("pitch_break", "Pitching-moment break at the stall", "-", True),
    ("pitch_tendency", "Tendency at the stall", "", False), ("trim_CL_wing_alone", "CL where the wing alone has zero moment", "-", False),
    ("Control and lateral", None), ("Cl_p", "Roll damping Clp", "-", False), ("Cl_beta_total", "Dihedral effect Clβ (sweep + dihedral)", "per rad", False),
    ("Cl_beta_dihedral", "  from the dihedral", "per rad", False), ("Cl_beta_sweep", "  from the sweep and taper", "per rad", False),
    ("Cl_da_linear_per_rad", "Aileron roll power, linear", "per rad", True), ("Cl_da_design_per_rad", "  at the design point", "per rad", True),
    ("Cl_da_at_0p9_per_rad", "  at 90 % of the maximum lift", "per rad", True), ("roll_authority_0p9", "  share of the linear power left at 90 %", "-", True),
    ("roll_max_design", "Roll moment at full aileron deflection (design point)", "-", True), ("control_loss_CL", "CL where half the roll power is lost", "-", False),
    ("Loads", None), ("lift_centroid_eta", "Lift centroid (share of the semi-span)", "-", False), ("root_bending_coeff", "Root bending coefficient", "-", True),
    ("Flap", None), ("flap_dCL_same_alpha", "Lift gain at the same angle", "-", False), ("flap_dAlpha_same_CL_deg", "Angle change at the same lift", "°", False),
    ("flap_dCm_same_CL", "Moment change at the same lift", "-", False), ("flap_dCD_same_CL", "Drag change at the same lift", "-", False),
    ("flap_dCLmax", "Gain of the first-strip maximum lift", "-", False),
    ("Model quality", None), ("confidence_min", "Lowest NeuralFoil confidence along the span", "-", False),
]


def metric_table_rows(cases) -> tuple[list[dict], list[str]]:
    cols = ["Quantity", "Unit"]
    data = []
    for lab, c in cases:
        m = c.full()["metrics"]
        data.append((lab, m, with_safety(m, c.sf)))
        cols += [f"{lab}: calculated"] + ([f"{lab}: with safety factor"] if True else [])
    rows, group_buf = [], None
    for r in ROWS:
        if r[1] is None:
            group_buf = {"_group": r[0]}
            continue
        key, label, unit, has_sf = r
        if not any(key in m for _, m, _ in data):
            continue
        if group_buf is not None:
            rows.append(group_buf)
            group_buf = None
        row = {"Quantity": label, "Unit": unit}
        for lab, m, s in data:
            row[f"{lab}: calculated"] = fmt(m.get(key))
            row[f"{lab}: with safety factor"] = fmt(s.get(key + "_sf")) if has_sf else ""
        rows.append(row)
    return rows, cols


def metrics_csv(cases) -> str:
    rows, cols = metric_table_rows(cases)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        if not r.get("_group"):
            w.writerow([r.get(c, "") for c in cols])
    return buf.getvalue()


def warnings_for(c: Case, label: str) -> list[tuple[str, str]]:
    out = []
    p = c.p
    for msg in validate(p):
        out.append(("warn", f"{label}: {msg}"))
    r = c.full()
    m = r["metrics"]
    if m.get("warning"):
        out.append(("bad", f"{label}: {m['warning']}."))
    if r.get("design_ok") and m.get("design_converged") is False:
        out.append(("bad", f"{label}: the non-linear solution at the design point did not converge: do not trust this case (a section may be far outside its table, "
                           "for example a very highly cambered airfoil at a large twist)."))
    if r.get("design_ok") and m.get("curve_points_not_converged", 0) > 0:
        out.append(("warn", f"{label}: {m['curve_points_not_converged']} point(s) of the lift curve did not converge (beyond the stall, where several sections are stalled): read that part as a trend."))
    if r.get("design_ok") and m["confidence_min"] < 0.5:
        out.append(("warn", f"{label}: NeuralFoil's confidence is below 0.5 somewhere along the span: treat the section data with caution."))
    if r.get("design_ok") and m.get("speed_margin_sf", m.get("speed_margin", 9)) is not None:
        s = with_safety(m, c.sf)
        if s.get("speed_margin_sf", 9.0) < 1.2:
            out.append(("warn", f"{label}: with the safety factor the design speed is only {s['speed_margin_sf']:.2f} × the stall speed."))
    if m.get("eta_stall_first", 0.0) > 0.0 and r.get("design_ok") and m["eta_stall_first"] > p["stall_onset_max"] and p["aileron_on"]:
        out.append(("warn", f"{label}: the stall starts at η = {m['eta_stall_first']:.2f}, outboard of the limit {p['stall_onset_max']:.2f}: the ailerons may stall first."))
    if p["flap_on"] and c.flap_defl > 0:
        out.append(("warn", f"{label}: flap results are the least certain part of the lab (thin-airfoil theory with viscous factors, no measured data)."))
    return out


# ------------------------------------------------------------------------------------------------ sidebar
init_state()
st.sidebar.markdown("<div class='hz-side-brand'>HORYZN</div><div class='hz-side-sub'>Wing aerodynamics lab</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Page", ["Inputs", "Results", "Optimise", "Explore", "Valleys", "Sensitivity", "About the model"], key="page")
if ONLINE:
    st.sidebar.markdown("<div class='hz-side-sub'>Online copy on a free shared server: your inputs live only in this browser tab (save them with the button below), "
                        "and long optimisations are slow (see About the model).</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
sf_box = st.sidebar.empty()
st.sidebar.markdown("---")
st.sidebar.button("Load the wing of FW-v6-rec from the configurator", on_click=load_configurator_wing, disabled=not any(CONFIGURATOR_DESIGNS.glob("FW_recommended_*.json")))
st.sidebar.button("Reset every input", on_click=reset_inputs)
dl_box = st.sidebar.empty()
st.sidebar.file_uploader("Load inputs (JSON)", type=["json"], key="upload")
st.sidebar.button("Apply the uploaded file", on_click=apply_upload, disabled=st.session_state.get("upload") is None)

banner_box = st.empty()


# ------------------------------------------------------------------------------------------------ pages
def kpi_cards(c: Case) -> None:
    r = c.full()
    m = r["metrics"]
    s = with_safety(m, c.sf)
    if not r.get("design_ok"):
        cards([("Wing", f"{c.wing.span:.2f} m · AR {c.wing.aspect_ratio:.1f}", f"area {c.wing.area:.3f} m²", "blue"), ("Design point", "not reachable", m.get("warning", ""), "bad")])
        return
    cards([
        ("Wing", f"{m['span_m']:.2f} m · AR {m['aspect_ratio']:.1f}", f"area {m['area_m2']:.3f} m² · loading {m['wing_loading_Nm2']:.0f} N/m²", "blue"),
        ("Lift-to-drag", f"{m['LD']:.1f}", f"with safety factor {s['LD_sf']:.1f} · best {m.get('LD_max', float('nan')):.1f}", "ok"),
        ("Usable CLmax", f"{m['CL_max_first']:.2f}", f"with safety factor {s['CL_max_first_sf']:.2f} · stall starts at η {m['eta_stall_first']:.2f}",
         "ok" if m["eta_stall_first"] <= c.p["stall_onset_max"] else "amber"),
        ("Stall speed", f"{m['V_stall_mps']:.1f} m/s", f"with safety factor {s['V_stall_mps_sf']:.1f} · margin {s['speed_margin_sf']:.2f}", "ok" if s["speed_margin_sf"] >= 1.2 else "amber"),
        ("Cm at the design point", f"{m['Cm_ref']:+.3f}", f"with safety factor {s['Cm_ref_sf']:+.3f} · dCm/dCL {m['dCm_dCL_ref']:+.3f}", "blue"),
        ("Pitch break at the stall", m["pitch_tendency"], f"ΔCm {m['pitch_break']:+.4f}", "blue"),
        ("Roll power near the stall", f"{m['roll_authority_0p9'] * 100:.0f} %" if "roll_authority_0p9" in m else "–", "of the linear value at 90 % CLmax" if "roll_authority_0p9" in m else "ailerons off", "blue"),
        ("Power at the design point", f"{m['P_req_W']:.0f} W", f"with safety factor {s['P_req_W_sf']:.0f} W", "blue"),
    ])


def minimum_tab() -> "st.delta_generator.DeltaGenerator":
    """The first tab: the inputs without which nothing can be computed, and the buttons that set everything else to auto.
    Returns a container the summary of the auto inputs is written into once every tab has been drawn (so it is never one step behind)."""
    note("These four numbers state the problem: what the wing carries, how fast, in which air, and how big it is. Nothing can be computed without them, so they are "
         "<b>always on</b> (their on switch is locked); each can still be a value or a range. <b>Every other input can be left auto</b>: the lab then searches the design inputs "
         "for you (the optimiser, Optimise page) and uses its own default for the rest.")
    for k in MINIMUM_INPUTS:
        param_row(BY_KEY[k])
        note(MINIMUM_WHY[k])
    st.button("Set every other input to auto", type="primary", on_click=set_all_auto, key="btn_all_auto", use_container_width=True,
              help="Switches off every input except the four above: the lab finds a solution from these minimum inputs alone.")
    c2, c3 = st.columns(2)
    c2.button("Switch every input back on (as values)", on_click=set_all_values, key="btn_all_values", use_container_width=True,
              help="Every input becomes a value again (the wing area, wing loading and design lift coefficient stay off, as they start).")
    c3.button("Go to the optimiser", on_click=go_optimise, key="btn_go_opt", use_container_width=True)
    return st.container()


def auto_summary() -> None:
    v, f, p = current()
    r, l = ranges_lists()
    rows = auto_rows(v, f, r, p, l)
    n = sum(len(x) for x in rows.values())
    if n == 0:
        note("No input is auto yet. Press <b>Set every other input to auto</b> to see what the lab then searches and what it takes by default.")
        return
    cards([("Auto inputs", str(n), "of the inputs in use", "blue"), ("Searched by the optimiser", str(len(rows["searched"])), "over everything they allow", "ok"),
           ("Lab default", str(len(rows["default"])), "environment, calibration, structure", "amber"), ("Not used", str(len(rows["unused"])), "overrides", "blue")])
    heading("The optimiser searches these auto inputs")
    table(rows["searched"]) if rows["searched"] else note("None: every design input is a value or a range.")
    with st.expander(f"{len(rows['default'])} auto inputs take the lab's default"):
        table(rows["default"], height=420)
    if rows["unused"]:
        with st.expander(f"{len(rows['unused'])} overrides are off"):
            table(rows["unused"])


def page_inputs() -> None:
    v, f, p = current()
    left, right = st.columns([3.3, 2.4])
    with left:
        note("Every input has two switches. <b>on</b> + range off: a <b>value</b>. <b>on</b> + <b>range</b> on: a <b>minimum and a maximum</b> (an airfoil station or a choice: a list of "
             "allowed options), searched by the optimiser, the sampling and the sweeps. <b>on</b> off: <b>auto</b>, the lab decides. Start with the first tab, <b>Minimum inputs</b>.")
        tabs = st.tabs(GROUPS)
        slot = None
        for t, g in zip(tabs, GROUPS):
            with t:
                if g == MINIMUM_GROUP:
                    slot = minimum_tab()
                else:
                    render_group(g)
        with slot:
            auto_summary()
    with right:
        v, f, p = current()
        r, l = ranges_lists()
        area, how = wing_area(p)
        vs = free_variables(v, f, r, None, l, include_uncertain=True)
        if vs:
            heading("Inputs the lab searches (ranges and auto)")
            table(range_rows(vs, f))
            note("The preview below uses the middle of every range and the lab's default for every auto input. Search on the pages Optimise and Explore.")
        note(f"Wing area {area:.3f} m² (from {how}).")
        for msg in validate(p):
            st.markdown(f"<div class='hz-warn'>{html.escape(msg)}</div>", unsafe_allow_html=True)
        try:
            c = case_of(p, "current inputs")
            kpi_cards(c)
            chart(FG.top_view(c, "Top view of the current inputs"), "iv_top")
        except Exception as exc:  # noqa: BLE001
            st.markdown(f"<div class='hz-bad'>These inputs cannot be evaluated: {html.escape(str(exc))}</div>", unsafe_allow_html=True)


def results_cases() -> list[tuple[str, Case]]:
    v, f, p = current()
    cases = [("current inputs", case_of(p, "current inputs"))]
    opt = st.session_state.get("opt")
    if opt is not None:
        po = resolve(opt.values, opt.fixed, *ranges_lists())
        cases.append(("optimised", case_of(po, "optimised")))
    picked = st.session_state.get("picked")
    if picked is not None:
        cases.append(("picked design", case_of(picked, "picked design")))
    return cases


PDF_LABELS = {"current inputs": "Current inputs", "optimised": "Optimised design", "picked design": "Design picked in Explore"}


def report_parts(which: str) -> dict:
    """What a report (PDF or Excel) of one design of the Results page is made from: the design, the design to compare with, the flight points, the optimisation it
    came from, the valleys around it and the state of every input (for a design that is not an optimum)."""
    ss = st.session_state
    cases = dict(results_cases())
    opt = ss.get("opt") if which == "optimised" else None
    points, start, valley = points_now(), None, None
    if opt is not None:
        points = opt.points or points
        start = ("Starting design", case_of(opt.problem.p0, "starting design")) if opt.problem is not None else ("Current inputs", cases["current inputs"])
        vl = ss.get("valley")
        if vl is not None and vl.get("prob") is opt.problem:
            valley = vl
    elif which != "current inputs":
        start = ("Current inputs", cases["current inputs"])
    return {"design": (PDF_LABELS[which], cases[which]), "start": start, "points": points, "opt": opt, "valley": valley, "fixed": current()[1] if which == "current inputs" else None}


def build_pdf(which: str) -> tuple[bytes, str]:
    """The PDF report of one design of the Results page (and, for the optimised design, of the optimisation and the valleys around it)."""
    parts = report_parts(which)
    pdf = RP.build_report(parts["design"], start=parts["start"], points=parts["points"], opt=parts["opt"], valley=parts["valley"])
    return pdf, f"HORYZN_wing-report_{PDF_LABELS[which].lower().replace(' ', '-')}_{datetime.date.today():%Y-%m-%d}.pdf"


def build_xlsx(which: str) -> tuple[bytes, str]:
    """The Excel workbook of one design: inputs (and where each came from), the optimiser's setup, the calculations as live formulas, the results, the lift against
    the weight, the strips, the curves and the drawings, with native Excel charts."""
    from wingaero import excel as XL
    parts = report_parts(which)
    data = XL.build_workbook(parts["design"], start=parts["start"], points=parts["points"], opt=parts["opt"], valley=parts["valley"], fixed=parts["fixed"])
    return data, f"HORYZN_wing-workbook_{PDF_LABELS[which].lower().replace(' ', '-')}_{datetime.date.today():%Y-%m-%d}.xlsx"


def pdf_signature(which: str) -> str:
    """What the PDF was made from: if it changes, the prepared PDF is out of date."""
    ss = st.session_state
    case = dict(results_cases())[which]
    opt = ss.get("opt") if which == "optimised" else None
    vl = ss.get("valley") if opt is not None else None
    return json.dumps([which, case.p, id(opt), id(vl.get("rep")) if vl else None, [p.label() for p in points_now()]], sort_keys=True, default=str)


def pdf_section(key: str) -> None:
    """The buttons that write the PDF report and the Excel workbook of a design, and the buttons that download them."""
    ss = st.session_state
    names = [n_ for n_, _ in results_cases()]
    heading("Reports: PDF and Excel")
    note("<b>PDF report</b>: a document to keep or send: the design, <b>how every number was calculated</b> (flight condition, geometry, lift, drag, stall, pitching moment, control, loads), "
         "the <b>lift against the weight along the mission</b>, the drawings of the wing and its airfoils, and the charts. For the optimised design it also tells how the optimum was found and, if "
         "you mapped them, the valleys around it.<br><b>Excel workbook</b>: the same design to play with: sheets for the inputs (which are yours, which the optimiser searched, which are the lab's "
         "defaults), the optimiser's setup, the calculations as <b>live formulas</b>, the results, the lift against the weight, the strip-by-strip lifting line, the curves and the drawings, each with its "
         "charts as native Excel charts.")
    c1, c2, c3 = st.columns([1.6, 1.3, 1.5])
    which = c1.selectbox("Design to report", names, index=names.index("optimised") if "optimised" in names else 0, format_func=lambda k: PDF_LABELS.get(k, k), key=f"pdf_which_{key}")
    d1, d2, d3 = st.columns([1.6, 1.3, 1.5])
    for slot, store, label, spinner, make, mime, mark, sig in (
            (c2, "pdf", "Prepare the PDF report", "Writing the report: calculations, drawings and charts (about 15 seconds)…", build_pdf, "application/pdf", "pdf", pdf_signature),
            (c3, "xlsx", "Prepare the Excel workbook", "Writing the workbook: formulas, tables and charts (a few seconds)…", build_xlsx,
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xls", pdf_signature)):
        if slot.button(label, type="primary", key=f"{mark}_make_{key}"):
            with st.spinner(spinner):
                try:
                    data, name = make(which)
                    ss[store] = {"which": which, "bytes": data, "name": name, "sig": sig(which)}
                except Exception as exc:  # noqa: BLE001
                    ss[store] = None
                    st.markdown(f"<div class='hz-bad'>The {'report' if store == 'pdf' else 'workbook'} could not be written: {html.escape(str(exc))}</div>", unsafe_allow_html=True)
        got = ss.get(store)
        target = d2 if store == "pdf" else d3
        if got and got["which"] == which:
            if got["sig"] == sig(which):
                target.download_button(f"Download the {'PDF' if store == 'pdf' else 'Excel'} ({len(got['bytes']) / 1e6:.1f} MB)", got["bytes"], file_name=got["name"], mime=mime, key=f"{mark}_dl_{key}")
            else:
                target.markdown("<span class='hz-note'>The inputs changed since it was written: prepare it again.</span>", unsafe_allow_html=True)


def page_results() -> None:
    all_cases = results_cases()
    names = [n for n, _ in all_cases]
    if len(names) > 1:
        sel = st.radio("Show", names + ["both, overlaid"], horizontal=True, index=1 if "optimised" in names else 0)
        cases = all_cases if sel == "both, overlaid" else [x for x in all_cases if x[0] == sel]
    else:
        cases = all_cases
        note("Run the optimiser (page Optimise) to compare the optimised wing with these inputs.")
    for lab, c in cases:
        for kind, msg in warnings_for(c, lab):
            st.markdown(f"<div class='{'hz-bad' if kind == 'bad' else 'hz-warn'}'>{html.escape(msg)}</div>", unsafe_allow_html=True)
    pdf_section("res")
    ok = [(lab, c) for lab, c in cases if c.full().get("design_ok")]
    cols = st.columns(len(cases))
    for col, (lab, c) in zip(cols, cases):
        with col:
            st.markdown(f"**{lab}**")
            kpi_cards(c)
    # 1 airfoils
    heading("1 · Airfoils")
    cols = st.columns(len(cases))
    for col, (lab, c) in zip(cols, cases):
        with col:
            chart(FG.airfoil_figure(c), f"af_{lab}")
    with st.expander("Thickness and camber distributions, and the section numbers", expanded=False):
        for lab, c in cases:
            st.markdown(f"**{lab}**")
            chart(FG.airfoil_distribution_figure(c), f"afd_{lab}")
            table(FG.airfoil_rows(c))
    # 2 top view
    heading("2 · Top view with the fuselage")
    cols = st.columns(len(cases))
    for col, (lab, c) in zip(cols, cases):
        with col:
            chart(FG.top_view(c, f"Top view: {lab}"), f"tv_{lab}")
    # 3 charts
    heading("3 · Aerodynamic charts")
    if not ok:
        st.markdown("<div class='hz-bad'>No design point could be reached with these inputs; the charts need one.</div>", unsafe_allow_html=True)
        return
    tabs = st.tabs(["Airfoil", "Wing: lift, drag", "Pitching moment", "Stall and span", "Roll control", "Performance", "Flap", "Flight points", "All the numbers"])
    with tabs[0]:
        note("Section data of every airfoil station at its Reynolds number at the design speed (NeuralFoil, n_crit and forced transition as set). Solid: as calculated; "
             "dashed: with the safety factor; dotted: beyond the stall.")
        for lab, c in ok:
            st.markdown(f"**{lab}**")
            figs = FG.airfoil_charts(c)
            for pair in (("cl", "polar"), ("ld", "cm"), ("cmcl", "conf"), ("re_clmax", "re_cd")):
                cc = st.columns(2)
                for col, k in zip(cc, pair):
                    with col:
                        chart(figs[k], f"afc_{lab}_{k}")
    with tabs[1]:
        a, b = st.columns(2)
        with a:
            chart(FG.wing_lift(ok), "wl")
        with b:
            chart(FG.wing_polar(ok), "wp")
        a, b = st.columns(2)
        with a:
            chart(FG.wing_ld(ok), "wld")
        with b:
            chart(FG.wing_drag_breakdown(ok[-1][1]), "wdb")
            note(f"Breakdown of {ok[-1][0]}.")
    with tabs[2]:
        chart(FG.wing_moment(ok), "wm")
        note("Pitching moment of the wing alone about the reference point (the share of the MAC set in the flight condition) and about the aerodynamic centre. A rising Cm "
             "against CL is destabilising about that point; the break near the stall (dash-dot line) is the sections' own moment change: the lab does not model the "
             "tail or the redistribution of load after the stall that makes a swept wing pitch up.")
    with tabs[3]:
        for lab, c in ok:
            st.markdown(f"**{lab}**")
            a, b = st.columns(2)
            with a:
                chart(FG.spanwise_stall_map(c), f"ssm_{lab}")
            with b:
                chart(FG.utilisation_heatmap(c), f"uh_{lab}")
            chart(FG.spanwise_state(c), f"ss_{lab}")
    with tabs[4]:
        chart(FG.wing_control(ok), "wc")
        note("Roll power of the ailerons (Cl per radian of deflection, from ±3° finite differences of the non-linear solution) against the wing's lift. As the outer strips "
             "reach their maximum lift, the aileron on the going-down side cannot add lift and the roll power falls; where it is half of the linear value the control is "
             "marked 'lost'. The aileron is an angle change (no gain of maximum lift): conservative.")
    with tabs[5]:
        chart(FG.wing_performance(ok), "wpf")
        chart(FG.stall_speed_mass(ok), "wsm")
    with tabs[6]:
        for lab, c in ok:
            st.markdown(f"**{lab}**")
            chart(FG.flap_effect(c), f"fe_{lab}")
    with tabs[7]:
        note(points_summary())
        for lab, c in ok:
            st.markdown(f"**{lab}**")
            table(point_rows(c, points_now()))
        note("✓ the limit is met at that point, ✗ it is broken. The same wing, in each flight condition: the speed, the air, the mass and the load factor change, the wing does not.")
    with tabs[8]:
        rows, cols_ = metric_table_rows(ok)
        table(rows, cols_, height=700)
        c1, c2 = st.columns(2)
        c1.download_button("Download the numbers (CSV)", metrics_csv(ok), file_name="wing-results.csv", mime="text/csv")
        c2.download_button("Download the inputs (JSON)", inputs_json(), file_name="wing-inputs.json", mime="application/json")


def page_optimise() -> None:
    ss = st.session_state
    v, f, p = current()
    ranges, lists = ranges_lists()
    st.markdown("Choose what to improve and run. The optimiser searches **inside the ranges you gave** (the *range* switch of an input, in the Inputs page) and **over everything "
                "an auto design input allows** (its *on* switch off; press *Set every other input to auto* in the first tab to search with the minimum inputs only). "
                "The safety factor is applied to every objective and limit. The wing is evaluated at every **flight point** below.")
    c1, c2 = st.columns([1.2, 1.6])
    with c1:
        heading("Objectives (weights; 0 = not used)")
        for k, (lab, _, sgn, _) in OBJECTIVES.items():
            dis = k == "roll" and not p["aileron_on"]
            if f"w_{k}" not in ss:
                ss[f"w_{k}"] = float(ss["PW"].get(k, 1.0 if k == "ld" else 0.0))
            st.slider(lab + (" (switch the ailerons on to use it)" if dis else ""), 0.0, 1.0, step=0.05, key=f"w_{k}", disabled=dis)
            ss["PW"][k] = float(ss[f"w_{k}"])
        w = weights_now(p)
        effort = st.radio("Search effort", ["quick", "normal"] if ONLINE else ["quick", "normal", "thorough"], index=1, horizontal=True,
                          help="With every design input auto (11 inputs) and the three standard flight points: quick about 4 min, normal about 9 min, thorough about 25 min; "
                               "one flight point takes a third of that. Fewer searched inputs run faster.")
        seed = st.number_input("Random seed", 0, 9999, 1, 1, help="A different seed may find a different local optimum.")
    vs = free_variables(v, f, ranges, None, lists)
    with c2:
        heading("Inputs searched, with their ranges")
        if not vs:
            st.markdown("<div class='hz-warn'>Nothing to search. In the Inputs page switch <b>range</b> on for the inputs the optimiser may choose, or leave design inputs "
                        "<b>auto</b> (press <b>Set every other input to auto</b> in the first tab).</div>", unsafe_allow_html=True)
        else:
            table(range_rows(vs, f))
        unc = [x for x in free_variables(v, f, ranges, None, lists, True) if not BY_KEY[x.key].optimisable]
        if unc:
            note(f"{len(unc)} environment, calibration or model-switch input(s) are given as ranges: the optimiser uses their middle value (they belong to the Explore and Sensitivity pages).")
    heading("Flight points")
    note("The same wing is evaluated at every point that is on. Each objective is evaluated at each point and combined with the point weights (they are normalised to add up to one; "
         "a weight of 0 makes the point only check the limits). Every point is tested against the stall-position limit and the strip-utilisation limit, and against its own least speed "
         "margin over the stall (with the safety factor); the pitching-moment, roll and wing-loading limits belong to the first point, the design point of the inputs.")
    points_editor()
    pts = points_now()
    run = st.button("Run the optimiser", type="primary", disabled=not vs or not any(w.values()))
    if run:
        prog = st.progress(0.0, text="starting")
        maxit = {"quick": 25, "normal": 45, "thorough": 90}[effort]

        def cb(gen, best, n_evals):
            prog.progress(min(gen / maxit, 1.0), text=f"generation {gen}: best objective {best:.4f} · {n_evals} candidates evaluated")
        try:
            res = optimise(v, f, w, ranges, None, effort, int(seed), cb, lists, pts)
            ss["opt"] = res
            ss["valley"] = None
            prog.progress(1.0, text=f"done in {res.seconds:.0f} s · {res.n_evals} candidates evaluated")
        except Exception as exc:  # noqa: BLE001
            st.markdown(f"<div class='hz-bad'>{html.escape(str(exc))}</div>", unsafe_allow_html=True)
    res = ss.get("opt")
    if res is None:
        return
    heading("Result")
    if not res.feasible:
        st.markdown(f"<div class='hz-bad'>The best design found violates a limit ({len(res.violations)} constraint(s)). Widen a range or relax a limit (Inputs, 'Limits for the optimiser'), "
                    "move a flight point's least margin, or raise the effort.</div>", unsafe_allow_html=True)
    rows = []
    for var in res.variables:
        now = res.changed[var.key]
        pr = BY_KEY[var.key]
        src = "auto" if mode_of(f, var.key) == "auto" else "range"
        if var.kind == "float":
            pos = (now - var.lo) / (var.hi - var.lo) if var.hi > var.lo else 0.5
            widen = "the input allows nothing lower" if src == "auto" else "widen the range?"
            widen_hi = "the input allows nothing higher" if src == "auto" else "widen the range?"
            edge = f"at the lower end: {widen}" if pos < 0.02 else (f"at the upper end: {widen_hi}" if pos > 0.98 else "")
            rows.append({"Input": pr.label.strip(), "Unit": "" if pr.unit == "-" else pr.unit, "Given as": src, "Range": f"{fmt(var.lo)} to {fmt(var.hi)}", "Optimised": fmt(now),
                         "Position in the range": f"{pos * 100:.0f} %", "Note": edge})
        else:
            rows.append({"Input": pr.label.strip(), "Unit": "", "Given as": src, "Range": f"{len(var.pool)} options", "Optimised": show_option(pr, now),
                         "Position in the range": "", "Note": ""})
    table(rows)
    note("An input that hardly moves the objectives you weighted ends anywhere inside its range (for lift-to-drag alone, the sweep, for instance). "
         "The <b>Valleys</b> page shows how far each input can move from this optimum, and the Explore page, 'Which input moves a result', which inputs matter; weight the pitching moment "
         "or the stall objectives if they do.")
    b0, b1 = res.baseline, res.best
    comp = []
    for key, lab in (("LD_sf", "Lift-to-drag (with safety factor)"), ("P_req_W_sf", "Power at the design point (SF)"), ("V_stall_mps_sf", "Stall speed (SF)"), ("CL_max_first_sf", "Usable CLmax (SF)"),
                     ("eta_stall_first", "Where the stall starts (share of the semi-span)"), ("Cm_ref_sf", "Cm at the design point (SF)"), ("speed_margin_sf", "Design speed / stall speed (SF)"),
                     ("util_max_sf", "Strip utilisation (SF)"), ("area_m2", "Wing area (m²)"), ("wing_loading_Nm2", "Wing loading (N/m²)")):
        if key in b0 and key in b1:
            comp.append({"Quantity": lab + " (design point)", "Starting point (middle of the ranges, auto: default)": fmt(b0[key]), "Optimised": fmt(b1[key])})
    table(comp)
    if len(res.points) > 1:
        heading("Every flight point, starting design and optimised design")
        prow = []
        for i, pt in enumerate(res.points):
            m0, m1 = res.baseline_points[i], (res.best_points[i] if i < len(res.best_points) else {})
            lim = float(pt.margin_min) if (i > 0 and pt.margin_min is not None) else float(p["speed_margin_min"])
            prow.append({"Flight point": pt.label(), "Weight": fmt(res.problem.point_weights[i]), "L/D (SF)": f"{fmt(m0.get('LD_sf'))} → {fmt(m1.get('LD_sf'))}",
                         "Stall speed (SF)": f"{fmt(m0.get('V_stall_mps_sf'))} → {fmt(m1.get('V_stall_mps_sf'))}",
                         f"Speed margin (SF), at least": f"{fmt(m0.get('speed_margin_sf'))} → {fmt(m1.get('speed_margin_sf'))} (≥ {fmt(lim)})",
                         "Strip utilisation (SF)": f"{fmt(m0.get('util_max_sf'))} → {fmt(m1.get('util_max_sf'))}"})
        table(prow)
    e_best = res.problem.evaluate(res.x) if res.problem is not None else None
    if e_best is not None and e_best.g:
        with st.expander("Every limit at the optimum"):
            table([{"Limit": nm, "Value (≤ 0 is met)": f"{gv:+.3f}", "State": "BROKEN" if gv > 1e-6 else ("active" if gv > -0.1 else "met")} for nm, gv in zip(res.names, e_best.g)])
    chart(FG.optimiser_history(res.history), "opt_hist")
    c1, c2 = st.columns(2)
    c1.button("Apply the optimised values to the inputs", on_click=apply_optimised, help="Copies the optimised values into the Inputs page as values (the ranges are kept).")
    c2.markdown("<span class='hz-note'>The Results page shows the optimised wing next to your inputs: airfoils, top view and every chart. The Valleys page shows what can still move.</span>", unsafe_allow_html=True)
    pdf_section("opt")


def page_explore() -> None:
    v, f, p = current()
    ranges, lists = ranges_lists()
    st.markdown("Search **through** the ranges instead of for one optimum: sample the whole space they span, or sweep one or two inputs over their range. Every result has the safety factor "
                "applied and every design is tested against the limits of the Inputs page, at every flight point.")
    note(points_summary())
    vs_all = free_variables(v, f, ranges, None, lists, True)
    if not vs_all:
        st.markdown("<div class='hz-warn'>Nothing to search. In the Inputs page switch <b>range</b> on for the inputs to search, or leave design inputs <b>auto</b> "
                    "(press <b>Set every other input to auto</b> in the first tab): an auto design input is sampled over everything it allows.</div>", unsafe_allow_html=True)
        return
    labels = {k: lab for lab, k in EX.METRICS.items()}
    tabs = st.tabs(["Sample the ranges", "Sweep one input", "Sweep two inputs"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        n = c1.select_slider("Designs in the sample", [64, 128, 256, 512, 1024], 256, key="ex_n", help="About 10 to 30 ms each (the first time an airfoil is used, 50 ms more).")
        rank_by = c2.selectbox("Rank the designs by", list(OBJECTIVES), format_func=lambda k: OBJECTIVES[k][0], key="ex_obj")
        seed = c3.number_input("Random seed", 0, 9999, 1, key="ex_seed")
        inc = st.checkbox("Also vary the environment and calibration inputs given as ranges (shows their effect; the ranking is then not a design choice)", False, key="ex_inc")
        if st.button("Run the sample", type="primary", key="ex_run"):
            prog = st.progress(0.0, text="evaluating the sample")
            try:
                st.session_state["sample"] = EX.sample(v, f, ranges, lists, int(n), int(seed), inc, {rank_by: 1.0}, progress=lambda i, tot: prog.progress(i / tot), points=points_now())
                prog.progress(1.0, text="done")
            except Exception as exc:  # noqa: BLE001
                st.markdown(f"<div class='hz-bad'>{html.escape(str(exc))}</div>", unsafe_allow_html=True)
        s = st.session_state.get("sample")
        if s is None:
            return
        use, feas = s.usable(), s.feasible()
        cards([("Designs", f"{len(s.rows)}", f"{len(use)} could be evaluated", "blue"), ("Meet every limit", f"{len(feas)}", f"{100 * len(feas) / max(len(s.rows), 1):.0f} % of the sample", "ok" if feas else "bad"),
               ("Inputs searched", f"{len(s.variables)}", ", ".join(BY_KEY[x.key].label.strip()[:22] for x in s.variables[:3]) + (" …" if len(s.variables) > 3 else ""), "blue")])
        names = list(s.problem.names)
        counts = {nm: 0 for nm in names}
        counts["the design lift coefficient is beyond the stall (at a flight point)"] = sum(1 for r in s.rows if not r["ok"])
        for r in use:
            for gi, nm in zip(r["g"] or [], names):
                if gi > 1e-6:
                    counts[nm] += 1
        heading("Which limit bites")
        table([{"Limit": nm, "Designs that break it": c, "Share of the sample": f"{100 * c / max(len(s.rows), 1):.0f} %"} for nm, c in counts.items() if c or nm in names])
        heading("The sample")
        cc = st.columns(3)
        lab_list = list(EX.METRICS)
        kx = EX.METRICS[cc[0].selectbox("Horizontal axis", lab_list, index=lab_list.index("Stall speed, m/s (SF)"), key="ex_x")]
        ky = EX.METRICS[cc[1].selectbox("Vertical axis", lab_list, index=0, key="ex_y")]
        kc = EX.METRICS[cc[2].selectbox("Colour", lab_list, index=lab_list.index("Where the stall starts (share of the semi-span)"), key="ex_c")]
        chart(FG.explore_scatter(s, kx, ky, kc, labels), "ex_scatter")
        heading("The best designs that meet every limit")
        top = sorted(feas, key=lambda r: r["objective"])[:10]
        if not top:
            st.markdown("<div class='hz-warn'>No design of the sample meets every limit. Widen a range, relax a limit, or raise the number of designs.</div>", unsafe_allow_html=True)
        else:
            rows = []
            for i, r in enumerate(top):
                row = {"#": i + 1}
                for var in s.variables:
                    pr = BY_KEY[var.key]
                    x = r["inputs"][var.key]
                    row[pr.label.strip()[:26]] = show_option(pr, x) if var.kind != "float" else fmt(x)
                row.update({"L/D (SF)": fmt(r["m"]["LD_sf"]), "Power W (SF)": fmt(r["m"]["P_req_W_sf"]), "Stall m/s (SF)": fmt(r["m"]["V_stall_mps_sf"]), "CLmax (SF)": fmt(r["m"]["CL_max_first_sf"])})
                rows.append(row)
            table(rows)
            k_sel = st.number_input("Design number", 1, len(top), 1, 1, key="ex_pick")
            b1, b2 = st.columns(2)
            b1.button("Use this design as my inputs (values)", on_click=apply_point, args=(top[int(k_sel) - 1]["inputs"],))
            b2.button("Show this design in Results", on_click=pick_point, args=(top[int(k_sel) - 1]["p"],))
        heading("Which input moves a result")
        cc = st.columns(2)
        km = cc[0].selectbox("Result", lab_list, index=0, key="ex_eff")
        only = cc[1].checkbox("Only the designs that meet every limit", True, key="ex_eff_feas")
        eff = EX.effects(s, EX.METRICS[km], only)
        if eff:
            chart(FG.effects_figure(eff, km), "ex_effects")
        else:
            st.markdown("<span class='hz-note'>Not enough designs for this.</span>", unsafe_allow_html=True)
    float_vars = [x for x in vs_all if x.kind == "float"]
    lab_of = {x.key: BY_KEY[x.key].label.strip() for x in float_vars}
    with tabs[1]:
        if not float_vars:
            st.markdown("<span class='hz-note'>Only airfoil lists and choices are searched here; sweeps need a numeric range or a numeric auto design input.</span>", unsafe_allow_html=True)
        else:
            c1, c2, c3 = st.columns(3)
            key = c1.selectbox("Input", [x.key for x in float_vars], format_func=lambda k: lab_of[k], key="sw1_key")
            km = c2.selectbox("Result", list(EX.METRICS), key="sw1_met")
            n = c3.select_slider("Points", [11, 21, 41], 21, key="sw1_n")
            if st.button("Run the sweep", type="primary", key="sw1_run"):
                st.session_state["sweep1"] = (key, km, EX.sweep1d(v, f, ranges, lists, key, int(n), set(), points_now()))
            sw = st.session_state.get("sweep1")
            if sw:
                chart(FG.sweep1d_figure(sw[2], EX.METRICS[sw[1]], sw[1], lab_of.get(sw[0], sw[0])), "sw1_fig")
                note("The other inputs stay at the middle of their ranges (auto: the lab's default). Green: the design meets every limit; red cross: it breaks one.")
    with tabs[2]:
        if len(float_vars) < 2:
            st.markdown("<span class='hz-note'>Needs at least two numeric inputs given as ranges (or auto design inputs).</span>", unsafe_allow_html=True)
        else:
            c1, c2, c3, c4 = st.columns(4)
            kx2 = c1.selectbox("Horizontal input", [x.key for x in float_vars], format_func=lambda k: lab_of[k], key="sw2_x")
            ky2 = c2.selectbox("Vertical input", [x.key for x in float_vars], index=1, format_func=lambda k: lab_of[k], key="sw2_y")
            km2 = c3.selectbox("Result", list(EX.METRICS), key="sw2_met")
            n2 = c4.select_slider("Grid", [9, 13, 17], 13, key="sw2_n")
            if kx2 == ky2:
                st.markdown("<span class='hz-note'>Choose two different inputs.</span>", unsafe_allow_html=True)
            elif st.button("Run the sweep", type="primary", key="sw2_run"):
                st.session_state["sweep2"] = (kx2, ky2, km2, EX.sweep2d(v, f, ranges, lists, kx2, ky2, int(n2), set(), points_now()))
            sw2 = st.session_state.get("sweep2")
            if sw2:
                chart(FG.sweep2d_figure(sw2[3], EX.METRICS[sw2[2]], sw2[2], lab_of.get(sw2[0], sw2[0]), lab_of.get(sw2[1], sw2[1])), "sw2_fig")
                note("The other inputs stay at the middle of their ranges (auto: the lab's default). Grey: a limit is broken (or the design lift coefficient is beyond the stall).")


def apply_valley_ranges() -> None:
    """After the valley analysis: an input with a wide free range becomes a range (its free range), the others a value (their optimum)."""
    ss = st.session_state
    vl = ss.get("valley")
    if vl is None:
        return
    for key, act in VL.ranges_from_report(vl["rep"]).items():
        p = BY_KEY[key]
        if act[0] == "range":
            lo, hi = sorted((min(max(float(act[1]), p.lo), p.hi), min(max(float(act[2]), p.lo), p.hi)))
            ss["PF"][key], ss["PR"][key], ss["PRG"][key] = False, (lo, hi), True
        elif act[0] == "list":
            ss["PF"][key], ss["PRG"][key] = False, True
            ss["PL"][key] = list(act[1])
        else:
            ss["PF"][key], ss["PRG"][key] = True, False
            ss["PV"][key] = clamp(p, act[1])
        _drop_widgets(key)


def run_valleys(prob, x, label: str, tol: float, n: int, combos: bool) -> None:
    ss = st.session_state
    prog = st.progress(0.0, text="scanning every input")
    rep = VL.tolerances(prob, x, tol, n, progress=lambda i, t: prog.progress(0.55 * i / t, text=f"scanning input {i} of {t}"))
    cv = None
    if combos and sum(1 for var in prob.vs if var.kind == "float") >= 2:
        prog.progress(0.55, text="combinations of inputs")
        cv = VL.combinations(prob, x, 0.04, tol, progress=lambda i, t: prog.progress(0.55 + 0.45 * i / t, text="combinations of inputs"))
    prog.progress(1.0, text=f"done in {rep.seconds:.0f} s · {rep.n_evals} designs evaluated")
    ss["valley"] = {"rep": rep, "cv": cv, "label": label, "prob": prob, "x": np.asarray(x, float), "surf": None, "tol": tol}


def valley_rows(rep) -> tuple[list[dict], list[dict]]:
    """The table of the free ranges (numeric inputs, tightest first) and the table of the alternatives (airfoils and choices)."""
    def span(v, k):
        e = v.effect.get(k)
        return f"{fmt(e[0])} to {fmt(e[1])}" if e else "–"
    nums, opts = [], []
    for v in sorted((r for r in rep.valleys if r.kind == "float"), key=lambda r: r.share):
        nums.append({"Input": v.label, "Unit": v.unit, "At the optimum": fmt(v.x_star), "Free from": fmt(v.free_lo), "Free to": fmt(v.free_hi), "Share of its range": f"{100 * v.share:.0f} %",
                     "Verdict": v.verdict, "Lower end limited by": v.bind_lo, "Upper end limited by": v.bind_hi, "L/D (SF) over it": span(v, "LD_sf"),
                     "Stall speed (SF)": span(v, "V_stall_mps_sf"), "Stall starts at η": span(v, "eta_stall_first"), "Cm (SF)": span(v, "Cm_ref_sf")})
    for v in rep.valleys:
        if v.kind != "float":
            pr = BY_KEY[v.key]
            for r in v.options:
                opts.append({"Input": v.label, "Option": show_option(pr, r["value"]) + ("  (the optimum)" if r["current"] else ""), "Loss of the objective": (f"{r['loss']:+.2f} %" if r["loss"] == r["loss"] else "no design solution"),
                             "Acceptable": "yes" if r["ok"] else "no", "Why not": r["why"]})
    return nums, opts


def show_valleys(vl: dict) -> None:
    rep, cv, prob, x, tol = vl["rep"], vl["cv"], vl["prob"], vl["x"], vl["tol"]
    counts = rep.counts()
    cards([("Free", str(counts["free"]), "inputs to play with", "ok"), ("Loose", str(counts["loose"]), "some room", "ok"), ("Tight", str(counts["tight"]), "little room", "amber"),
           ("Critical", str(counts["critical"]), "leave them alone", "bad"), ("Active limits", str(len(rep.active)), "at the centre", "blue" if not rep.active else "amber")])
    for msg in rep.warnings:
        st.markdown(f"<div class='hz-warn'>{html.escape(msg)}</div>", unsafe_allow_html=True)
    if rep.active:
        note("The centre of the valleys sits on <b>" + "</b>, <b>".join(html.escape(a["name"]) for a in rep.active) + "</b>. An input whose free range ends at a limit can move only if "
             "another input compensates (see the combinations below).")
    tol_txt = f"{tol:g} %"
    note(f"<b>Free range</b> of an input: how far it can move from the centre, with every other input held there, before the weighted objective (with the safety factor, at every flight point) "
         f"loses {tol_txt} of its starting value, or a limit is broken, or the range the input allows ends. <b>Free</b>: 60 % of its range or more; <b>loose</b>: 25 to 60 %; "
         "<b>tight</b>: 8 to 25 %; <b>critical</b>: under 8 %. The last four columns show how much the result itself changes over the free range: an input can be free for the "
         "objective and still move the stall or the pitching moment.")
    nums, opts = valley_rows(rep)
    heading("Which inputs can move")
    table(nums)
    chart(FG.valley_bars_figure(rep), "vl_bars")
    if opts:
        heading("Airfoils and choices: the alternatives")
        table(opts)
    heading("The valley of every input")
    chart(FG.valley_slices_figure(rep), "vl_slices")
    c1, c2 = st.columns([1.3, 3])
    c1.button("Use the free ranges as my ranges", on_click=apply_valley_ranges, key="vl_apply",
              help="An input that is free or loose becomes a range (its free range); a tight or critical one becomes a value (its optimum). Then run the optimiser again, for example with another objective, to use the room.")
    c2.markdown("<span class='hz-note'>The inputs that are free or loose become ranges, the others values: the next optimisation then moves only what can move.</span>", unsafe_allow_html=True)
    # ---- the 3D valley of two inputs
    floats = [(i, var) for i, var in enumerate(prob.vs) if var.kind == "float"]
    if len(floats) >= 2:
        heading("The valley of two inputs in 3D")
        lab = {i: BY_KEY[var.key].label.strip() for i, var in floats}
        pair = VL.pair_to_show(prob, cv, rep.valleys) if cv else None
        d_i, d_j = pair if pair else (floats[0][0], floats[1][0])
        ids = [i for i, _ in floats]
        c1, c2, c3, c4 = st.columns([1.5, 1.5, 1, 1])
        si = c1.selectbox("Horizontal input", ids, index=ids.index(d_i), format_func=lambda i: lab[i], key="vl_sx")
        sj = c2.selectbox("Vertical input", ids, index=ids.index(d_j) if d_j != d_i else min(1, len(ids) - 1), format_func=lambda i: lab[i], key="vl_sy")
        gn = c3.select_slider("Grid", [13, 17, 21, 29], 17, key="vl_grid", help="Designs evaluated: the grid squared.")
        win = c4.select_slider("Window", [1.0, 0.5, 0.25], 0.5, format_func=lambda w: {1.0: "whole range", 0.5: "half", 0.25: "a quarter"}[w], key="vl_win",
                               help="The share of each input's range that is shown, centred on the optimum.")
        note("The default pair is the input that moves the most active limit the most (the wall of the valley) and the tightest other input, or the two tightest inputs. "
             f"Cost: {int(gn) ** 2} designs, about {int(gn) ** 2 * 0.03 * len(prob.points):.0f} s.")
        if si == sj:
            note("Choose two different inputs.")
        elif st.button("Draw the valley", type="primary", key="vl_surf"):
            prog = st.progress(0.0, text="evaluating the grid")
            try:
                vl["surf"] = VL.surface(prob, x, si, sj, int(gn), float(win), tol, progress=lambda b, t: prog.progress(b / t))
                prog.progress(1.0, text="done")
            except Exception as exc:  # noqa: BLE001
                st.markdown(f"<div class='hz-bad'>{html.escape(str(exc))}</div>", unsafe_allow_html=True)
        sf = vl.get("surf")
        if sf is not None:
            lx, ly = BY_KEY[sf["key_x"]].label.strip(), BY_KEY[sf["key_y"]].label.strip()
            chart(FG.valley_surface_figure(sf, lx, ly), "vl_3d")
            chart(FG.valley_contour_figure(sf, lx, ly), "vl_2d")
            note("Coloured: designs that meet every limit; grey: designs that break one (a wall of the valley); the orange plane and line: the tolerance. A wide flat floor under the plane is "
                 "a pair of inputs to play with; a narrow trench, often diagonal, means they must move together.")
    # ---- combinations
    if cv is not None:
        heading("Combinations of inputs: what can move together")
        note("Valleys are often diagonal: two inputs that cannot move alone may move together. At the optimum the <b>active limits</b> block the directions along their gradient; the "
             "<b>free combinations</b> are the directions they leave open, sloppiest first. The distance is measured in coordinates scaled to the range of each input (a combination that "
             "is 70 % one input and 70 % another moves each by 0.7 x the distance x its range).")
        if cv["active"]:
            table([{"Limit": a["name"], "Margin (g, ≤ 0 is met)": f"{a['g']:+.3f}", "Room along its gradient": ("broken" if a["room"] < 0 else f"{100 * a['room']:.1f} % of the ranges"),
                    "Moves it the most": a["text"]} for a in cv["active"]])
        else:
            note("No limit is active at the centre: every combination is free up to the tolerance.")
        drows = []
        for k, d in enumerate(cv["directions"]):
            mv = lambda m: ", ".join(f"{k_} {v_:+.3g}" for k_, v_ in m.items())  # noqa: E731
            drows.append({"#": k + 1, "Made of": d["text"], "Can move (+)": f"{100 * d['r_plus']:.0f} %", "ended by (+)": d["why_plus"], "so the inputs move by (+)": mv(d["move_plus"]),
                          "Can move (−)": f"{100 * d['r_minus']:.0f} %", "ended by (−)": d["why_minus"]})
        heading("Free combinations, sloppiest first")
        table(drows)
        a, b = st.columns(2)
        with a:
            chart(FG.combination_radius_figure(cv), "vl_radius")
        with b:
            chart(FG.combination_loadings_figure(cv), "vl_loadings")
        if cv["checks"]:
            with st.expander("Check of the prediction against the real design"):
                table([{"Combination": c["direction"], "Sense": "+" if c["sign"] > 0 else "−", "Step (share of the ranges)": f"{100 * c['step']:.0f} %", "Predicted loss": f"{c['predicted_loss']:+.2f} %",
                        "Real loss": (f"{c['actual_loss']:+.2f} %" if c["actual_loss"] == c["actual_loss"] else "no design solution"), "Limits met (predicted / real)": f"{'yes' if c['predicted_feasible'] else 'no'} / {'yes' if c['actual_feasible'] else 'no'}"}
                       for c in cv["checks"]])
                note("The model has kinks (the stall tables, the limits), so the prediction is a guide: where it differs from the real loss, trust the real loss and the slices above.")
    note(f"Centre: {html.escape(vl['label'])}. {rep.n_evals} designs evaluated in {rep.seconds:.0f} s.")


def page_valleys() -> None:
    ss = st.session_state
    v, f, p = current()
    ranges, lists = ranges_lists()
    res = ss.get("opt")
    st.markdown("Around a design the objective forms a **valley**. Along an input that hardly matters the floor is wide and flat, along an input that matters the walls are close. This page measures "
                "the valley of every searched input, draws it (in 3D for two inputs), and finds the combinations of inputs that can move together, so that you know **what to play with**. "
                "It uses the same objectives, flight points, limits and safety factor as the optimiser.")
    centres = (["The optimised design (the last run of the optimiser)"] if res is not None else []) + ["The nominal design (the middle of the ranges; auto: the default)"]
    c1, c2, c3, c4 = st.columns([2.2, 1.3, 1.2, 1.4])
    centre = c1.radio("Centre of the valleys", centres, key="vl_centre")
    tol = c2.select_slider("Tolerance (loss of the objective, %)", [0.5, 1.0, 2.0, 5.0], 1.0, key="vl_tol",
                           help="How much the weighted objective may lose, as a percentage of its starting value, for an input to still count as free.")
    n = c3.select_slider("Points per input", [21, 41, 61], 41, key="vl_n")
    combos = c4.checkbox("Also the combinations of inputs", True, key="vl_combos")
    note(points_summary())
    if st.button("Map the valleys", type="primary", key="vl_run"):
        try:
            if centre.startswith("The optimised"):
                prob, x = res.problem, np.asarray(res.x, float)
            else:
                prob = Problem(v, f, ranges, lists, weights_now(p) or {"ld": 1.0}, points_now())
                if not prob.vs:
                    raise ValueError("nothing is searched: give inputs as ranges, or leave design inputs auto, so that there is something to map")
                x = prob.nominal_x()
            run_valleys(prob, x, centre, float(tol), int(n), bool(combos))
        except Exception as exc:  # noqa: BLE001
            st.markdown(f"<div class='hz-bad'>{html.escape(str(exc))}</div>", unsafe_allow_html=True)
    vl = ss.get("valley")
    if vl is None:
        note("Press <b>Map the valleys</b>. Run the optimiser first (page Optimise) to map the valleys around its result; without it the valleys are mapped around the nominal design.")
        return
    show_valleys(vl)


def page_sensitivity() -> None:
    v, f, p = current()
    ranges, _ = ranges_lists()
    st.markdown("Which uncertain input moves a result most? Each calibration or environment input goes to the low and the high end of its range (the range you gave it, or its allowed span), "
                "one at a time, with every other input as it is.")
    metrics = {"LD": ("Lift-to-drag at the design point", "LD"), "CL_max_first": ("Usable maximum lift", "CL_max_first"), "V_stall_mps": ("Stall speed", "V_stall_mps"),
               "P_req_W": ("Power at the design point", "P_req_W"), "Cm_ref": ("Cm at the design point", "Cm_ref"), "LD_max": ("Best lift-to-drag", "LD_max")}
    k = st.selectbox("Result", list(metrics), format_func=lambda x: metrics[x][0])
    keys = ["cd_factor", "clmax_factor", "n_crit", "excrescence", "interference", "crossflow_factor", "ctrl_visc"]
    keys = [x for x in keys if active(BY_KEY[x], p)]
    if st.button("Run the sensitivity study", type="primary"):
        base = case_of(p).full()["metrics"].get(k)
        if base is None:
            st.markdown("<div class='hz-bad'>The base case has no design point.</div>", unsafe_allow_html=True)
            return
        rows = []
        prog = st.progress(0.0)
        for i, key in enumerate(keys):
            pr = BY_KEY[key]
            lo, hi = ranges[key] if mode_of(f, key) == RANGE else (pr.lo, pr.hi)
            res = {}
            for side, val in (("lo", lo), ("hi", hi)):
                p2 = dict(p)
                p2[key] = float(val)
                val_m = case_of(p2).full()["metrics"].get(k)
                res[side] = 100.0 * (val_m - base) / abs(base) if (val_m is not None and base) else 0.0
            rows.append({"label": f"{pr.label.strip()} [{lo:g} to {hi:g}]", **res})
            prog.progress((i + 1) / len(keys))
        st.session_state["sens"] = (metrics[k][0], rows)
    s = st.session_state.get("sens")
    if s:
        chart(FG.sensitivity_figure(s[1], s[0]), "sens")


def page_about() -> None:
    st.markdown((ROOT / "docs" / "about.md").read_text(encoding="utf-8") if (ROOT / "docs" / "about.md").exists() else "See docs/physics-lab.md.")


{"Inputs": page_inputs, "Results": page_results, "Optimise": page_optimise, "Explore": page_explore, "Valleys": page_valleys, "Sensitivity": page_sensitivity,
 "About the model": page_about}[page]()

_v, _f, _p = current()
sf_box.markdown(f"**Safety factor {_p['safety_factor']:.2f}**  \n<span class='hz-note'>set in the Inputs page, tab Safety factor</span>", unsafe_allow_html=True)
dl_box.download_button("Save the inputs and ranges (JSON)", inputs_json(), file_name="wing-inputs.json", mime="application/json")
banner_box.markdown(f"<div class='hz-banner'><div class='hz-brand'>HORYZN<small>Wing aerodynamics lab · wing and fuselage only</small></div>"
                    f"<div><span class='hz-chip'>safety factor {_p['safety_factor']:.2f}</span><span class='hz-chip'>{page}</span></div></div>", unsafe_allow_html=True)
