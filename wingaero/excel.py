"""The Excel workbook of a wing design: inputs, the optimiser's setup, the calculations as live formulas, the results, the lift against the weight along the
mission, the strip-by-strip working of the lifting line, the curves and the drawings, each sheet with its data and its charts (native Excel charts: click
on them, change them).

    xlsx_bytes = build_workbook(("Optimised design", case), start=("Starting design", case0), points=points, opt=result, valley=valley, fixed=fixed)

Colour code (also on the first sheet):
    blue on yellow     an input you gave                              orange   the optimiser's setup, or an input it searched
    grey               an input left at the lab's default             green    a copy of an input (it follows the Inputs sheet)
    black on pale blue a formula: change an input and it recalculates purple   a value from the lifting line or the airfoil polars (it does not recalculate)

The formulas are written in Excel with names (`W_N = mass_kg * gravity`), and their results are also stored in the file, computed here by the same formulas, so
a viewer that does not recalculate still shows the numbers. Written with xlsxwriter (plain Python): the same on the PC and on the free online server.
"""
from __future__ import annotations

import datetime
import io
import math
import re

import numpy as np

from . import airfoil as AF
from . import mission as MS
from .analysis import with_safety
from .core import G0
from .params import BY_KEY, GROUPS, MINIMUM_GROUP, PARAMS, active

NAVY, INK, ORANGE, BLUE, STEEL, LINE, MIST = "#1D3043", "#1F2933", "#C2410C", "#2C6596", "#6B7A89", "#D8DEE6", "#F3F7FA"
GREEN, AMBER, RED = "#2E7D5B", "#E0A040", "#B23B3B"
KINDS = {                                   # kind: (font colour, fill, what it means)
    "you": ("#0000FF", "#FFF4C2", "An input you gave"),
    "opt": ("#9A3412", "#FFE1CC", "The optimiser's setup, or an input it searched"),
    "dflt": ("#4B5563", "#EEF1F4", "An input left at the lab's default"),
    "link": ("#166534", "#E4F4E8", "A copy of an input: it follows the Inputs sheet"),
    "calc": ("#111827", "#E3F0FB", "A formula: change an input and it recalculates"),
    "lines": ("#5B2A86", "#EFE7F8", "A value from the lifting line, the airfoil polars or the geometry: it does not recalculate"),
}
SHEETS = ["Read me", "Inputs", "Optimiser", "Calculations", "Results", "Mission lift", "Strips", "Curves", "Wing and airfoils", "Valleys"]


# ------------------------------------------------------------------------------------------------ formulas evaluated here too
def _sum(*a):
    return float(sum(np.sum(x) for x in a))


_FUN = {"SQRT": math.sqrt, "PI": lambda: math.pi, "IF": lambda c, a, b: a if c else b, "MAX": lambda *a: float(max(np.max(x) for x in a)), "MIN": lambda *a: float(min(np.min(x) for x in a)),
        "ABS": abs, "SUM": _sum, "SUMPRODUCT": lambda *a: float(np.sum(np.prod(np.array(a, float), axis=0)))}


class Env:
    """Names -> values, where a value may be a formula in Excel syntax that refers to other names (evaluated on demand). The cached result stored
    next to every formula in the file is computed here, from the same formula."""

    def __init__(self):
        self.defs, self.memo = {}, {}

    def __getitem__(self, k):
        if k in _FUN:
            return _FUN[k]
        if k in self.memo:
            return self.memo[k]
        if k not in self.defs:
            raise KeyError(f"name not defined: {k}")
        d = self.defs[k]
        v = self.run(d) if isinstance(d, str) and d.startswith("=") else d
        self.memo[k] = v
        return v

    def run(self, formula: str):
        return eval(formula[1:].replace("^", "**"), {"__builtins__": {}}, self)


def _name_ok(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name)) and not re.match(r"^[A-Za-z]{1,3}\d+$", name) and name.lower() not in ("c", "r")


def nf_for(v) -> str | None:
    """A number format that shows about four significant digits."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.floating, np.integer)) or not math.isfinite(float(v)):
        return None
    a = abs(float(v))
    if a == 0:
        return "0"
    if a >= 1000:
        return "#,##0"
    if a >= 100:
        return "0.0"
    if a >= 10:
        return "0.00"
    if a >= 1:
        return "0.000"
    if a >= 0.01:
        return "0.0000"
    return "0.00E+00" if a < 1e-4 else "0.00000"


class Book:
    """The workbook with its formats, names and helpers."""

    def __init__(self, buf, sheets: list):
        import xlsxwriter
        self.wb = xlsxwriter.Workbook(buf, {"in_memory": True})
        self.wb.set_calc_mode("auto")
        self.ws = {n: self.wb.add_worksheet(n) for n in sheets}
        self._fmts: dict = {}
        self.env = Env()
        self.names: dict = {}
        self._lower: set = set()
        tabs ={"Read me": "#1D3043", "Inputs": "#FFB000", "Optimiser": "#C2410C", "Calculations": "#2C6596", "Results": "#2E7D5B", "Mission lift": "#2E7D5B", "Strips": "#6B4C9A",
                "Curves": "#6B4C9A", "Wing and airfoils": "#6B7A89", "Valleys": "#6B7A89"}
        for k in sheets:
            self.ws[k].set_tab_color(tabs[k])
            self.ws[k].set_landscape()                   # printing: A4 landscape, one page wide
            self.ws[k].set_paper(9)
            self.ws[k].fit_to_pages(1, 0)
            self.ws[k].set_margins(0.4, 0.4, 0.5, 0.5)

    # ---- formats
    def fmt(self, kind: str = "text", nf: str | None = None, bold: bool = False, wrap: bool = False, align: str | None = None, size: float | None = None, color: str | None = None,
            italic: bool = False, border: bool = True, bg: str | None = None, valign: str = "top"):
        key = (kind, nf, bold, wrap, align, size, color, italic, border, bg, valign)
        if key not in self._fmts:
            d = {"font_name": "Calibri", "font_size": size or 10, "valign": valign}
            if kind in KINDS:
                d.update({"font_color": KINDS[kind][0], "bg_color": KINDS[kind][1]})
                if border:
                    d.update({"border": 1, "border_color": LINE})
            elif kind == "head":
                d.update({"bold": True, "font_color": "#FFFFFF", "bg_color": NAVY, "text_wrap": True, "valign": "vcenter", "border": 1, "border_color": NAVY})
            elif kind == "table":
                d.update({"border": 1, "border_color": LINE} if border else {})
            if nf:
                d["num_format"] = nf
            if bold:
                d["bold"] = True
            if wrap:
                d["text_wrap"] = True
            if align:
                d["align"] = align
            if color:
                d["font_color"] = color
            if italic:
                d["italic"] = True
            if bg:
                d["bg_color"] = bg
            self._fmts[key] = self.wb.add_format(d)
        return self._fmts[key]

    # ---- writing
    def title(self, ws, text: str, sub: str = "") -> int:
        ws.write_string(0, 0, text, self.fmt("text", bold=True, size=16, color=NAVY, border=False))
        if sub:
            ws.write_string(1, 0, sub, self.fmt("text", size=9.5, color=STEEL, italic=True, border=False))
        return 3

    def section(self, ws, r: int, text: str, note: str = "", ncols: int = 8) -> int:
        ws.write_string(r, 0, text, self.fmt("text", bold=True, size=12, color=NAVY, border=False))
        ws.set_row(r, 20)
        if note:
            ws.merge_range(r + 1, 0, r + 1, ncols - 1, note, self.fmt("text", size=9, color=STEEL, italic=True, wrap=True, border=False))
            ws.set_row(r + 1, max(15, 12.5 * (1 + len(note) // (13 * ncols))))
            return r + 2
        return r + 1

    def header(self, ws, r: int, heads: list, c0: int = 0, height: float = 30) -> None:
        for j, h in enumerate(heads):
            ws.write_string(r, c0 + j, h, self.fmt("head"))
        ws.set_row(r, height)

    def put(self, ws, r: int, c: int, v, kind: str = "table", nf: str | None = None, **fk) -> None:
        """A value (number, text, yes/no) in a cell."""
        if isinstance(v, (bool, np.bool_)):
            ws.write_boolean(r, c, bool(v), self.fmt(kind, None, **fk))
        elif isinstance(v, (int, float, np.floating, np.integer)):
            x = float(v)
            if not math.isfinite(x):
                ws.write_string(r, c, "–", self.fmt(kind, None, **fk))
            else:
                ws.write_number(r, c, x, self.fmt(kind, nf or nf_for(x), **fk))
        elif v is None:
            ws.write_blank(r, c, None, self.fmt(kind, None, **fk))
        else:
            ws.write_string(r, c, str(v), self.fmt(kind, None, **fk))

    def formula(self, ws, r: int, c: int, expr: str, cached, kind: str = "calc", nf: str | None = None, **fk) -> None:
        if isinstance(cached, (bool, np.bool_)):
            cached = bool(cached)
        elif isinstance(cached, (np.floating, np.integer)):
            cached = float(cached)
        if isinstance(cached, float) and not math.isfinite(cached):
            cached = 0
        ws.write_formula(r, c, expr, self.fmt(kind, nf or nf_for(cached), **fk), cached)

    def name(self, name: str, sheet: str, r: int, c: int, r2: int | None = None) -> None:
        """A workbook name for a cell (or a column range), so a formula reads `mass_kg * gravity` instead of `Inputs!$C$7 * Calculations!$D$5`."""
        assert _name_ok(name), name
        assert name.lower() not in self._lower, f"name used twice (Excel names are not case-sensitive): {name}"
        self._lower.add(name.lower())
        from xlsxwriter.utility import xl_rowcol_to_cell
        a = xl_rowcol_to_cell(r, c, True, True)
        ref = f"='{sheet}'!{a}" if r2 is None else f"='{sheet}'!{a}:{xl_rowcol_to_cell(r2, c, True, True)}"
        self.wb.define_name(name, ref)
        self.names[name] = ref

    # ---- charts
    def scatter(self, title: str, xname: str, yname: str, series: list, size=(540, 310), xlim=None, ylim=None, legend: str = "bottom", equal: bool = False):
        """series: dicts with sheet, x=(r0, r1, c), y=(r0, r1, c), name, color, and optionally dash, width, marker, line (False: markers only)."""
        ch = self.wb.add_chart({"type": "scatter", "subtype": "straight"})
        for s in series:
            sh = s["sheet"]
            d = {"name": s["name"], "categories": [sh, s["x"][0], s["x"][2], s["x"][1], s["x"][2]], "values": [sh, s["y"][0], s["y"][2], s["y"][1], s["y"][2]]}
            if s.get("line", True):
                d["line"] = {"color": s["color"], "width": s.get("width", 2.0), "dash_type": s.get("dash", "solid")}
            else:
                d["line"] = {"none": True}
            if s.get("marker"):
                d["marker"] = {"type": s["marker"], "size": s.get("msize", 7), "fill": {"color": s["color"]}, "border": {"color": "#FFFFFF"}}
            else:
                d["marker"] = {"type": "none"}
            if s.get("labels"):                     # a text (or a cell, written ='Sheet'!$A$1) beside every point
                d["data_labels"] = {"custom": [{"value": t} for t in s["labels"]], "position": "right", "font": {"size": 8, "color": s["color"]}}
            ch.add_series(d)
        ch.set_title({"name": title, "name_font": {"size": 11, "bold": True, "color": NAVY}})
        grid = {"visible": True, "line": {"color": "#E5E7EB", "width": 0.5}}
        xa = {"name": xname, "major_gridlines": grid, "name_font": {"size": 9}, "num_font": {"size": 8}, "label_position": "low"}      # numbers at the edge even when the axis crosses at 0
        ya = {"name": yname, "major_gridlines": grid, "name_font": {"size": 9}, "num_font": {"size": 8}, "label_position": "low", "num_format": "General"}
        if xlim:
            xa.update({"min": xlim[0], "max": xlim[1]})
        if ylim:
            ya.update({"min": ylim[0], "max": ylim[1]})
        ch.set_x_axis(xa)
        ch.set_y_axis(ya)
        ch.set_legend({"position": legend, "font": {"size": 8}} if legend != "none" else {"none": True})
        ch.set_size({"width": size[0], "height": size[1]})
        return ch

    def place(self, ws, row: int, charts: list, cols: int = 2, w: int = 540, h: int = 310) -> int:
        """Charts in a grid starting at a row; returns the first free row below them."""
        rows_per = int(math.ceil(h / 20.0)) + 1
        for i, ch in enumerate(charts):
            ws.insert_chart(row + (i // cols) * rows_per, 0, ch, {"x_offset": 6 + (i % cols) * (w + 12), "y_offset": 4, "object_position": 1})
        return row + int(math.ceil(len(charts) / cols)) * rows_per + 1


def _s(sheet, r0, r1, c, name, color, **kw):
    d = {"sheet": sheet, "x": (r0, r1, c), "y": (r0, r1, c), "name": name, "color": color}
    d.update(kw)
    return d


def _xy(sheet, xr, yr, name, color, **kw):
    """A series from a column of x values and a column of y values: xr = (r0, r1, col)."""
    d = {"sheet": sheet, "x": xr, "y": yr, "name": name, "color": color}
    d.update(kw)
    return d


# ------------------------------------------------------------------------------------------------ the inputs
def origin_of(key: str, group: str, fixed: dict | None, searched: dict, have_opt: bool) -> str:
    """Where an input of the design came from: you | opt | dflt | used (unknown: the design came from somewhere that does not say)."""
    if key in searched:
        return "opt"
    if group == MINIMUM_GROUP:
        return "you"
    if fixed is None:
        return "used"
    return "you" if fixed.get(key) is True else "dflt"


ORIGIN_TEXT = {"you": "Your input", "opt": "Searched by the optimiser", "dflt": "Lab default", "used": "Value used in this design"}


def input_table(case, fixed, opt) -> list[dict]:
    p = case.p
    searched = {v.key: v for v in opt.variables} if opt is not None else {}
    state = opt.fixed if opt is not None else fixed
    rows = []
    for g in GROUPS:
        for q in PARAMS:
            if q.group != g or not active(q, p):
                continue
            v = p.get(q.key)
            if v is None:
                continue
            org = origin_of(q.key, g, state, searched, opt is not None)
            if q.kind == "airfoil":
                v = AF.display_name(v) if v != AF.PARAM else "parametric NACA 4-digit"
            d = {"group": g, "key": q.key, "label": q.label.strip(), "value": v, "unit": "" if q.unit == "-" else q.unit, "origin": org, "start": None, "range": "", "help": q.help or ""}
            if q.key in searched:
                var = searched[q.key]
                p0 = opt.problem.p0.get(q.key) if opt.problem is not None else None
                if var.kind == "float":
                    d["start"], d["range"] = p0, f"{var.lo:.4g} to {var.hi:.4g}"
                else:
                    nm = (lambda x: AF.display_name(x) if (var.kind == "airfoil" and x != AF.PARAM) else str(x))
                    d["start"], d["range"] = (nm(p0) if p0 is not None else None), f"{len(var.pool)} options"
            rows.append(d)
    return rows


def sheet_inputs(bk: Book, case, fixed, opt, title: str) -> None:
    ws = bk.ws["Inputs"]
    r = bk.title(ws, "Inputs", "Every input of this design: where it came from and its value. Change a blue, orange or grey value and the formulas on the Calculations, Results and Mission sheets recalculate.")
    bk.header(ws, r, ["Group", "Input", "Value", "Unit", "Where it came from", "Started at", "Range searched", "What it is"])
    r += 1
    grp = None
    for d in input_table(case, fixed, opt):
        bk.put(ws, r, 0, d["group"] if d["group"] != grp else "", "table", bold=True, color=NAVY)
        grp = d["group"]
        bk.put(ws, r, 1, d["label"], "table")
        kind = {"used": "dflt"}.get(d["origin"], d["origin"])
        v = d["value"]
        bk.put(ws, r, 2, v, kind, align="right")
        bk.put(ws, r, 3, d["unit"], "table")
        bk.put(ws, r, 4, ORIGIN_TEXT[d["origin"]], kind)
        bk.put(ws, r, 5, d["start"], "table", align="right")
        bk.put(ws, r, 6, d["range"], "table")
        bk.put(ws, r, 7, d["help"], "table", wrap=False, color=STEEL, size=9)
        if _name_ok(d["key"]) and not isinstance(v, str):
            bk.name(d["key"], "Inputs", r, 2)
        if not isinstance(v, str):
            bk.env.defs[d["key"]] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
        r += 1
    ws.set_column(0, 0, 18)
    ws.set_column(1, 1, 42)
    ws.set_column(2, 2, 16)
    ws.set_column(3, 3, 8)
    ws.set_column(4, 4, 24)
    ws.set_column(5, 5, 14)
    ws.set_column(6, 6, 18)
    ws.set_column(7, 7, 70)
    ws.freeze_panes(4, 2)
    ws.autofilter(3, 0, r - 1, 7)


# ------------------------------------------------------------------------------------------------ the calculations
def calc_rows(bk: Book, case, how_area: str, has_strips: bool) -> list:
    """The calculation sheet: sections of rows (quantity, formula as text, expression, unit, kind, name). An expression that starts with = is an Excel formula
    (written as one, with the result stored); anything else is a value from the model. Every named row can be used by the other sheets."""
    r = case.full()
    m, p, air, w = r["metrics"], case.p, case.air, case.wing
    sections: list = []

    def sec(title, note=""):
        sections.append((title, note, []))

    def R(q, ftext, expr, unit="", kind="calc", name=None):
        sections[-1][2].append((q, ftext, expr, unit, kind, name))
    ok = r.get("design_ok")
    # 1
    sec("1  Flight condition and weight", "Standard atmosphere (ISA, troposphere) with the temperature offset; the weight the wing carries and the dynamic pressure it flies at.")
    R("Mass", "input", "=mass_kg", "kg", "link")
    R("Gravity", "constant", G0, "m/s²", "lines", "gravity")
    R("Weight", "W = m g", "=mass_kg*gravity", "N", "calc", "W_N")
    R("Load factor", "input (1 = steady level flight; 2 = a 60° turn)", "=load_factor", "", "link", "n_load")
    R("Lift the wing must carry", "L = n W", "=n_load*W_N", "N", "calc", "L_N")
    R("Altitude", "input", "=altitude_m", "m", "link")
    R("Temperature offset", "input", "=dT_K", "K", "link")
    R("Standard temperature at the altitude", "T = 288.15 − 0.0065 h", "=288.15-0.0065*MIN(MAX(altitude_m,0),11000)", "K", "calc", "T_std")
    R("Static pressure", "p = 101325 (T / 288.15)^(g / (R × 0.0065))", "=101325*(T_std/288.15)^(9.80665/(287.053*0.0065))", "Pa", "calc", "p_Pa")
    R("Air temperature", "T = T_std + ΔT", "=T_std+dT_K", "K", "calc", "T_K")
    R("Air density", "ρ = p / (R T),  R = 287.053 J/(kg K)", "=p_Pa/(287.053*T_K)", "kg/m³", "calc", "rho")
    R("Air viscosity", "Sutherland: μ = 1.458·10⁻⁶ T^1.5 / (T + 110.4)", "=1.458E-6*T_K^1.5/(T_K+110.4)", "Pa·s", "calc", "mu")
    if p["design_cl"] is not None:
        R("Design speed", "from the design lift coefficient: V = √(2 n W / (ρ S CL))", "=SQRT(2*n_load*W_N/(rho*S_m2*design_cl))", "m/s", "calc", "V_ms")
    else:
        R("Design speed", "input", "=speed_mps", "m/s", "link", "V_ms")
    R("Dynamic pressure", "q = ½ ρ V²", "=0.5*rho*V_ms^2", "Pa", "calc", "q_Pa")
    R("Reynolds number at the mean chord", "Re = ρ V c̄ / μ", "=rho*V_ms*mac_m/mu", "", "calc", "Re_mac")
    # 2
    sec("2  Wing geometry", "Up to three trapezoidal panels (or an elliptical chord); the chord at every station is a share of the root chord. The planform integrals come from the geometry model.")
    R("Span", "input", "=span_m", "m", "link")
    area = {"area": ("input", "=area_m2", "link"), "wing loading": ("S = W / (W/S), the wing loading is set", "=W_N/wing_loading_Nm2", "calc"),
            "span and aspect ratio": ("S = b² / AR", "=span_m^2/aspect_ratio", "calc")}[how_area]
    R("Wing area", area[0], area[1], "m²", area[2], "S_m2")
    R("Aspect ratio", "AR = b² / S", "=span_m^2/S_m2", "", "calc", "AR_wing")
    R("Wing loading", "W / S", "=W_N/S_m2", "N/m²", "calc", "WL_Nm2")
    R("Root chord", "c_r = S / (b ∫ (c/c_r) dη)", float(w.root_chord), "m", "lines", "croot_m")
    R("Tip chord ratio", "c_t / c_r (from the panels)", float(w.ratio[-1]), "", "lines", "tip_ratio")
    R("Tip chord", "c_t = c_r × ratio", "=croot_m*tip_ratio", "m", "calc")
    R("Mean aerodynamic chord", "c̄ = (1/S) ∫ c² dy", float(w.mac), "m", "lines", "mac_m")
    R("Mean sweep of the quarter-chord line", "area-weighted over the panels", float(w.mean_sweep_deg), "°", "lines")
    R("Incidence of the root chord on the fuselage axis", "input", "=incidence_deg", "°", "link")
    # 3
    sec("3  Lift at the design point", "The lifting line (Weissinger vortex lattice, solved non-linearly with the NeuralFoil section polars) finds the angle at which the wing carries the lift: see the Strips sheet.")
    R("Lift coefficient required", "CL = n W / (q S)", "=n_load*W_N/(q_Pa*S_m2)", "", "calc", "CL_req")
    R("Lift slope of the wing (linear lifting line)", "CLα from the strips' lift slopes and the planform", float(m["CL_alpha_per_rad"]), "per rad", "lines", "CLa_rad")
    R("Lift slope of the wing", "CLα per degree", "=CLa_rad*PI()/180", "per °", "calc", "CLa_deg")
    R("Lift coefficient at zero angle", "CL0 of the linear solution", float(m["CL0"]), "", "lines", "CL_zero")
    R("Zero-lift angle of the wing", "α₀ = −CL0 / CLα", "=-CL_zero/CLa_deg", "°", "calc", "alpha0_deg")
    if ok:
        R("Root angle of attack, linear estimate", "α = α₀ + CL / CLα", "=alpha0_deg+CL_req/CLa_deg", "°", "calc")
        R("Root angle of attack, non-linear solution", "the angle at which the strip polars carry CL (Strips sheet)", float(m["alpha_design_deg"]), "°", "lines", "alpha_nl_deg")
        R("Fuselage angle of attack", "α_fus = α − incidence", "=alpha_nl_deg-incidence_deg", "°", "calc")
    R("Inviscid span efficiency of the planform", "Oswald-type factor from the lifting line × fuselage factor", float(m["e_inviscid"]), "", "lines")
    # 4
    if ok and has_strips:
        sec("4  Drag at the design point", "Profile drag strip by strip from the airfoil polars (calibration factor and excrescence allowance), induced drag from the lifting line, fuselage drag from skin friction, form and cross-flow.")
        R("Profile drag", "CDp = (1 + excrescence) Σ cd_strip c_strip Δy / S  (Strips sheet)", "=(1+excrescence)*2*SUM(st_wcd)", "", "calc", "CD_profile")
        R("Induced drag", "CDi = −Σ G (T G) Δy / S, from the lifting line", float(m["CDi"]), "", "lines", "CD_induced")
        R("Span efficiency at the design point", "e = CL² / (π AR CDi)", "=CL_req^2/(PI()*AR_wing*CD_induced)", "", "calc", "e_span")
        R("Fuselage drag", "CDf = (Cf FF S_wet + cross-flow) × interference × (1 + excrescence) / S", float(m["CDf"]), "", "lines", "CD_fuselage")
        R("Total drag coefficient", "CD = CDp + CDi + CDf", "=CD_profile+CD_induced+CD_fuselage", "", "calc", "CD_tot")
        R("Lift-to-drag", "L/D = CL / CD", "=CL_req/CD_tot", "", "calc", "LD_ratio")
        R("Drag force", "D = CD q S", "=CD_tot*q_Pa*S_m2", "N", "calc", "D_N")
        R("Aerodynamic power", "P = D V", "=D_N*V_ms", "W", "calc", "P_W")
        R("Endurance factor", "CL^1.5 / CD", "=CL_req^1.5/CD_tot", "", "calc")
    # 5
    if "CL_max_first" in m:
        sec("5  Stall and speed margin", "The wing stalls where the first strip reaches its section maximum lift (the section polar, times the calibration factor, capped at 2π lift slope).")
        R("First strip to stall (share of the semi-span)", "the strip where cl reaches cl_max first", float(m["eta_stall_first"]), "η", "lines")
        R("Root angle of attack at that stall", "", float(m["alpha_stall_first_deg"]), "°", "lines")
        R("Maximum lift coefficient (first strip stalls)", "CLmax", float(m["CL_max_first"]), "", "lines", "CLmax")
        R("Peak of the non-linear lift curve", "CLmax,nl", float(m["CL_max_nl"]), "", "lines")
        R("Usable maximum lift coefficient", "CLmax / factor_lift", "=CLmax/factor_lift", "", "calc", "CLmax_use")
        R("Stall speed", "V_s = √(2 n W / (ρ S CLmax))", "=SQRT(2*n_load*W_N/(rho*S_m2*CLmax))", "m/s", "calc", "V_s")
        R("Stall speed with the safety factor", "V_s √factor_lift", "=V_s*SQRT(factor_lift)", "m/s", "calc", "V_s_sf")
        R("Speed margin", "V / V_s", "=V_ms/V_s", "", "calc", "margin_calc")
        R("Speed margin with the safety factor", "V / (V_s √factor_lift)", "=V_ms/V_s_sf", "", "calc", "margin_sf")
        if ok and has_strips:
            R("Highest strip utilisation at the design point", "max over the strips of cl / cl_cap  (Strips sheet)", "=MAX(st_util)", "", "calc", "util_calc")
            R("Highest strip utilisation with the safety factor", "factor_lift × cl / cl_cap", "=util_calc*factor_lift", "", "calc", "util_sf")
    # 6
    if ok:
        sec("6  Pitching moment of the wing alone", "About the moment reference point (a share of the mean chord) and about the aerodynamic centre; the tail is not modelled.")
        R("Aerodynamic centre", "x_ac, from the lifting line (share of the MAC from its leading edge)", float(m["x_ac_mac"]), "", "lines")
        R("Moment reference point", "input (the aircraft's centre of gravity, share of the MAC)", "=x_cg_mac", "", "link")
        R("Pitching moment about the reference point", "Cm = Σ strip lift × arm + Σ section moments", float(m["Cm_ref"]), "", "lines", "Cm_ref")
        R("Pitching moment about the aerodynamic centre", "Cm_ac", float(m["Cm_ac"]), "", "lines")
        R("Slope of Cm against CL (reference point)", "dCm/dCL (positive: nose-up with more lift)", float(m["dCm_dCL_ref"]), "", "lines")
        R("Break at the stall", f"Cm at the lift peak minus the linear trend ({m['pitch_tendency']})", float(m["pitch_break"]), "", "lines")
        R("CL where the wing alone has zero moment", "below the stall", float(m["trim_CL_wing_alone"]), "", "lines")
    # 7
    if ok:
        sec("7  Roll control, loads and lateral derivatives")
        if p["aileron_on"] and "Cl_da_linear_per_rad" in m:
            R("Aileron roll power, linear", "Cl_δa from ±3° finite differences", float(m["Cl_da_linear_per_rad"]), "per rad", "lines")
            R("Roll power at the design point", "same, non-linear", float(m["Cl_da_design_per_rad"]), "per rad", "lines")
            R("Roll power near the stall (90 % of CLmax)", "same, near the stall", float(m["Cl_da_at_0p9_per_rad"]), "per rad", "lines")
            R("Roll power near the stall, share of linear", "Cl_δa(0.9 CLmax) / Cl_δa,lin", float(m["roll_authority_0p9"]), "", "lines", "roll_09")
            R("Roll moment at full deflection, design point", f"deflection {p['aileron_defl_deg']:.1f}° each side", float(m["roll_max_design"]), "", "lines")
            R("CL where half the roll power is lost", "from the design point upwards", float(m["control_loss_CL"]), "", "lines")
        R("Lift centroid", "η of the centre of the lift distribution", float(m["lift_centroid_eta"]), "η", "lines")
        R("Root bending moment coefficient", "M_root / (q S b/2)", float(m["root_bending_coeff"]), "", "lines", "root_bend")
        R("Roll damping", "Cl_p", float(m["Cl_p"]), "per rad", "lines")
        R("Dihedral effect, from the dihedral", "Cl_β dihedral", float(m["Cl_beta_dihedral"]), "per rad", "lines", "Clb_dih")
        R("Dihedral effect, from the sweep", "Cl_β sweep", float(m["Cl_beta_sweep"]), "per rad", "lines", "Clb_swp")
        R("Dihedral effect, total", "Cl_β = dihedral part + sweep part", "=Clb_dih+Clb_swp", "per rad", "calc")
    # 8
    if "LD_max" in m:
        sec("8  Performance over speed", "The wing flown at every speed between just above the stall and 2.8 times it, at the lift it must carry (L = n W): the Curves sheet has the table.")
        R("Best lift-to-drag", "max (CL / CD) over speed", float(m["LD_max"]), "", "lines")
        R("   at lift coefficient", "", float(m["CL_at_LD_max"]), "", "lines")
        R("   at speed", "best-glide speed", float(m["V_best_glide"]), "m/s", "lines")
        R("Least aerodynamic power", "min (D V) over speed", float(m["P_min_W"]), "W", "lines")
        R("   at speed", "minimum-power speed", float(m["V_min_power"]), "m/s", "lines")
        R("Best endurance factor", "max (CL^1.5 / CD)", float(m["endurance_factor"]), "", "lines")
    # 9
    sec("9  The safety factor", "Applied last, in the adverse direction of every result: drag and power up, lift capability down, moments and loads up, control authority down. A family switched off uses 1.")
    R("Safety factor", "input, 1 to 2", "=safety_factor", "", "link")
    R("Factor on drag and power", "SF if switched on, else 1", "=IF(sf_drag,safety_factor,1)", "", "calc", "factor_drag")
    R("Factor on lift capability", "SF if switched on, else 1", "=IF(sf_lift,safety_factor,1)", "", "calc", "factor_lift")
    R("Factor on moments and loads", "SF if switched on, else 1", "=IF(sf_moment,safety_factor,1)", "", "calc", "factor_moment")
    R("Factor on control authority", "SF if switched on, else 1", "=IF(sf_control,safety_factor,1)", "", "calc", "factor_control")
    return sections


KIND_LABEL = {"link": "Input (copied)", "calc": "Formula", "lines": "Model value"}


def sheet_calculations(bk: Book, case, sections: list) -> dict:
    """Write the calculation sheet; returns {name: (row, col)} of every named row."""
    ws = bk.ws["Calculations"]
    r = bk.title(ws, "Calculations: inputs → formulas → results",
                 "Read down: each row is a quantity, its formula, and its result. Green cells copy an input from the Inputs sheet; pale-blue cells are live Excel formulas (click one to see it); purple values come from the "
                 "lifting line or the airfoil polars and stay as calculated.")
    # definitions first, so every formula can be evaluated whatever its order
    n_id = 0
    ids = []
    for _t, _n, rows in sections:
        for (_q, _f, expr, _u, _k, name) in rows:
            key = name or f"_row{n_id}"
            n_id += 1
            ids.append(key)
            bk.env.defs[key] = expr
    k = 0
    out = {}
    for title, note, rows in sections:
        r = bk.section(ws, r, title, note, 5)
        bk.header(ws, r, ["Quantity", "Formula", "Result", "Unit", "What it is"], height=20)
        r += 1
        for (q, ftext, expr, unit, kind, name) in rows:
            key = ids[k]
            k += 1
            val = bk.env[key]
            bk.put(ws, r, 0, q, "table")
            bk.put(ws, r, 1, ftext, "table", color=STEEL, size=9)
            if isinstance(expr, str) and expr.startswith("="):
                bk.formula(ws, r, 2, expr, val, kind)
            else:
                bk.put(ws, r, 2, val, kind)
            bk.put(ws, r, 3, unit, "table")
            bk.put(ws, r, 4, KIND_LABEL.get(kind, "Constant"), "table", color=STEEL, size=9)
            if name:
                bk.name(name, "Calculations", r, 2)
                out[name] = (r, 2)
            r += 1
        r += 1
    ws.set_column(0, 0, 46)
    ws.set_column(1, 1, 62)
    ws.set_column(2, 2, 16)
    ws.set_column(3, 3, 9)
    ws.set_column(4, 4, 16)
    ws.freeze_panes(3, 1)
    return out


# ------------------------------------------------------------------------------------------------ the strips
def sheet_strips(bk: Book, case, sd: dict) -> None:
    ws = bk.ws["Strips"]
    S = "Strips"
    r = bk.title(ws, "Inside the lifting line: strip by strip at the design point",
                 "The wing is cut into 32 strips across the span (16 on the half shown, root to tip, closer together near the tip). The flight is symmetric, so the other half is the same.")
    steps = [
        "1  Each strip has a chord c, a width Δy and a Reynolds number Re = ρ V c / μ: the airfoil polar (NeuralFoil) of the strip is taken at that Re.",
        "2  The geometric angle of a strip is α_geo = root angle + twist (+ flap effect). The angle the strip actually sees is lower, because the wing's own downwash (induced angle) comes on top: α_eff = α_geo − α_ind.",
        "3  The strip carries a lift coefficient cl(α_eff) from its polar; its circulation is G = ½ c cl (per unit of speed).",
        "4  The vortex lattice says how much downwash every strip's circulation puts on every other strip: α_ind = −[(A G) + G/(π c)] / k. The unknowns are the 32 values of G.",
        "5  Newton's method finds the G for which steps 2, 3 and 4 agree for all strips at once, at the root angle that gives exactly the lift the wing must carry.",
        "6  Then: strip lift = q c cl Δy;   CL = 2 Σ G Δy / S;   CDp = (1 + excrescence) Σ (c Δy / S) cd;   CDi = −Σ G (T G) Δy / S.   The columns below are those numbers; the sums at the bottom are checked against n W.",
        "7  A strip is at its stall when cl reaches its cap (utilisation = 1). With the safety factor the utilisation is multiplied by factor_lift and must stay below 1."]
    for t in steps:
        ws.merge_range(r, 0, r, 12, t, bk.fmt("text", size=9.5, wrap=True, border=False, color=INK))
        ws.set_row(r, 27 if len(t) > 120 else 15)
        r += 1
    r += 1
    heads = ["Strip", "η", "y (m)", "Chord c (m)", "Width Δy (m)", "Re = ρ V c / μ", "α_geo (°)", "α_eff (°)", "α_ind = α_geo − α_eff (°)", "cl at α_eff", "cl uncapped", "cl cap",
             "Utilisation cl / cap", "Utilisation × SF", "cd at α_eff", "Share of area w = c Δy / S", "Circulation G = ½ c cl", "Strip lift q c cl Δy (N)", "Lift per metre q c cl (N/m)",
             "Elliptical lift per metre (N/m)", "w × cd"]
    bk.header(ws, r, heads, height=48)
    r0 = r + 1
    n = len(sd["eta"])
    q, S_area = sd["q"], sd["S"]
    env = bk.env
    sf_lift = env["factor_lift"]
    for j in range(n):
        rr = r0 + j
        x = lambda c_: f"{chr(65 + c_)}{rr + 1}"  # noqa: E731  (columns A..U)
        bk.put(ws, rr, 0, j + 1, "table", nf="0")
        bk.put(ws, rr, 1, float(sd["eta"][j]), "lines")
        bk.put(ws, rr, 2, float(sd["y"][j]), "lines")
        bk.put(ws, rr, 3, float(sd["c"][j]), "lines")
        bk.put(ws, rr, 4, float(sd["dy"][j]), "lines")
        bk.formula(ws, rr, 5, f"=rho*V_ms*{x(3)}/mu", env["rho"] * env["V_ms"] * sd["c"][j] / env["mu"], "calc", nf="#,##0")
        bk.put(ws, rr, 6, float(sd["alpha_geo"][j]), "lines")
        bk.put(ws, rr, 7, float(sd["alpha_eff"][j]), "lines")
        bk.formula(ws, rr, 8, f"={x(6)}-{x(7)}", sd["alpha_geo"][j] - sd["alpha_eff"][j], "calc")
        bk.put(ws, rr, 9, float(sd["cl"][j]), "lines")
        bk.put(ws, rr, 10, float(sd["cl_unc"][j]), "lines")
        bk.put(ws, rr, 11, float(sd["cap"][j]), "lines")
        bk.formula(ws, rr, 12, f"={x(10)}/{x(11)}", sd["cl_unc"][j] / sd["cap"][j], "calc")
        bk.formula(ws, rr, 13, f"={x(12)}*factor_lift", sd["cl_unc"][j] / sd["cap"][j] * sf_lift, "calc")
        bk.put(ws, rr, 14, float(sd["cd"][j]), "lines", nf="0.00000")
        bk.formula(ws, rr, 15, f"={x(3)}*{x(4)}/S_m2", sd["c"][j] * sd["dy"][j] / S_area, "calc", nf="0.00000")
        bk.formula(ws, rr, 16, f"=0.5*{x(3)}*{x(9)}", 0.5 * sd["c"][j] * sd["cl"][j], "calc", nf="0.00000")
        bk.formula(ws, rr, 17, f"=q_Pa*{x(3)}*{x(9)}*{x(4)}", q * sd["c"][j] * sd["cl"][j] * sd["dy"][j], "calc", nf="0.000")
        bk.formula(ws, rr, 18, f"=q_Pa*{x(3)}*{x(9)}", q * sd["c"][j] * sd["cl"][j], "calc", nf="0.00")
        ell = 4 * env["L_N"] / (math.pi * env["span_m"]) * math.sqrt(max(0.0, 1 - sd["eta"][j] ** 2))
        bk.formula(ws, rr, 19, f"=4*L_N/(PI()*span_m)*SQRT(1-{x(1)}^2)", ell, "calc", nf="0.00")
        bk.formula(ws, rr, 20, f"={x(15)}*{x(14)}", sd["w"][j] * sd["cd"][j], "calc", nf="0.000000")
    r1 = r0 + n - 1
    for nm, c_ in (("st_wcd", 20), ("st_util", 12)):
        bk.name(nm, S, r0, c_, r1)
    bk.env.defs["st_wcd"] = sd["w"] * sd["cd"]
    bk.env.defs["st_util"] = sd["cl_unc"] / sd["cap"]
    # sums
    r = r1 + 2
    r = bk.section(ws, r, "What the strips add up to", "Each line is a sum over the strips written as a formula; the right-hand column is the number it must reproduce.", 8)
    bk.header(ws, r, ["Quantity", "", "", "From the strips", "", "Must equal", "Difference", ""], height=20)
    r += 1
    col = lambda c_: f"{chr(65 + c_)}{r0 + 1}:{chr(65 + c_)}{r1 + 1}"  # noqa: E731
    lift = 2 * float(np.sum(sd["lift"]))
    cl_sum = 4 * float(np.sum(sd["gamma"] * sd["dy"])) / S_area
    cdp = (1 + case.p["excrescence"]) * 2 * float(np.sum(sd["w"] * sd["cd"]))
    area_sum = 2 * float(np.sum(sd["c"] * sd["dy"]))
    rows = [("Lift of the wing: 2 × Σ strip lift (N)", f"=2*SUM({col(17)})", lift, "=L_N", env["L_N"], "0.000", "n W"),
            ("Lift coefficient: 4 Σ G Δy / S", f"=4*SUMPRODUCT({col(16)},{col(4)})/S_m2", cl_sum, "=CL_req", env["CL_req"], "0.00000", "n W / (q S)"),
            ("Profile drag coefficient: (1 + excrescence) × 2 Σ w cd", f"=(1+excrescence)*2*SUM({col(20)})", cdp, "=CD_profile", env["CD_profile"], "0.000000", "the same, as used on the Calculations sheet"),
            ("Wing area: 2 Σ c Δy (m²)", f"=2*SUMPRODUCT({col(3)},{col(4)})", area_sum, "=S_m2", env["S_m2"], "0.0000", "the planform area (strips: within 0.1 %)")]
    for lab, f1, v1, f2, v2, nf, why in rows:
        bk.put(ws, r, 0, lab, "table")
        bk.put(ws, r, 1, why, "table", color=STEEL, size=9)
        bk.put(ws, r, 2, "", "table")
        bk.formula(ws, r, 3, f1, v1, "calc", nf=nf)
        bk.put(ws, r, 4, "", "table")
        bk.formula(ws, r, 5, f2, v2, "calc", nf=nf)
        bk.formula(ws, r, 6, f"=(D{r + 1}-F{r + 1})/F{r + 1}", (v1 - v2) / v2, "calc", nf="0.0000%")
        r += 1
    r += 1
    d = lambda c_: (r0, r1, c_)  # noqa: E731
    ch1 = bk.scatter("Section lift coefficient along the span", "η (share of the semi-span)", "cl",
                     [_xy(S, d(1), d(9), "cl at the design point", BLUE, marker="circle", msize=5), _xy(S, d(1), d(11), "cl cap (where the strip stalls)", RED, dash="dash"),
                      _xy(S, d(1), d(10), "cl uncapped", STEEL, dash="round_dot", width=1.2)])
    ch2 = bk.scatter("Lift per metre of span", "η", "N/m", [_xy(S, d(1), d(18), "calculated (strips)", BLUE, marker="circle", msize=5), _xy(S, d(1), d(19), "elliptical, same total lift", STEEL, dash="dash")])
    ch3 = bk.scatter("Angles along the span", "η", "degrees", [_xy(S, d(1), d(6), "α_geo (root angle + twist)", NAVY), _xy(S, d(1), d(7), "α_eff (seen by the section)", BLUE, marker="circle", msize=5),
                                                           _xy(S, d(1), d(8), "α_ind (induced by the wing)", ORANGE, dash="dash")])
    ch4 = bk.scatter("Utilisation: how close each strip is to its stall", "η", "cl / cl cap", [_xy(S, d(1), d(12), "as calculated", BLUE, marker="circle", msize=5), _xy(S, d(1), d(13), "with the safety factor", ORANGE, dash="dash")],
                     ylim=(0, max(1.05, float(np.max(sd["cl_unc"] / sd["cap"] * sf_lift)) * 1.1)))
    bk.place(ws, r, [ch1, ch2, ch3, ch4])
    ws.set_column(0, 0, 44)
    ws.set_column(1, 20, 13)
    ws.freeze_panes(r0, 2)


# ------------------------------------------------------------------------------------------------ the mission
def sheet_mission(bk: Book, case, points, mp: list, sw: list) -> None:
    ws = bk.ws["Mission lift"]
    S = "Mission lift"
    env = bk.env
    r = bk.title(ws, "Lift against weight along the mission",
                 "At every flight point the wing must generate exactly the lift that balances n × its weight; this sheet shows how much more it could generate. Blue cells are the flight point's own conditions: change them and the formulas follow.")
    heads = ["Flight point", "Speed V (m/s)", "Mass (kg)", "Load factor n", "Air density ρ (kg/m³)", "Weight W = m g (N)", "Lift generated L = n W (N)", "Lift ÷ weight", "Dynamic pressure q = ½ρV² (Pa)",
             "CL needed L / (q S)", "CLmax as calculated", "Usable CLmax (÷ SF)", "Most lift it can generate q S CLmax (N)", "… with the safety factor (N)", "Most lift ÷ weight", "… with the safety factor",
             "Share of the usable lift in use", "Stall speed (m/s)", "Speed margin with SF", "Lift ÷ design-point weight"]
    bk.header(ws, r, heads, height=62)
    r0 = r + 1
    sfn = "factor_lift"
    for i, d in enumerate(mp):
        rr = r0 + i
        x = lambda c_: f"{chr(65 + c_)}{rr + 1}"  # noqa: E731
        bk.put(ws, rr, 0, d["label"], "opt", bold=True)
        bk.put(ws, rr, 1, d["V"], "you")
        bk.put(ws, rr, 2, d["mass"], "you")
        bk.put(ws, rr, 3, d["n"], "you")
        bk.put(ws, rr, 4, d["rho"], "lines", nf="0.0000")
        W, L, q = d["W"], d["L"], d["q"]
        bk.formula(ws, rr, 5, f"={x(2)}*gravity", W, nf="0.0")
        bk.formula(ws, rr, 6, f"={x(3)}*{x(5)}", L, nf="0.0")
        bk.formula(ws, rr, 7, f"={x(6)}/{x(5)}", L / W, nf="0.000")
        bk.formula(ws, rr, 8, f"=0.5*{x(4)}*{x(1)}^2", q, nf="0.0")
        bk.formula(ws, rr, 9, f"={x(6)}/({x(8)}*S_m2)", d["CL_req"], nf="0.000")
        if "CLmax" in d:
            bk.put(ws, rr, 10, d["CLmax"], "lines", nf="0.000")
            bk.formula(ws, rr, 11, f"={x(10)}/{sfn}", d["CLmax_sf"], nf="0.000")
            bk.formula(ws, rr, 12, f"={x(8)}*S_m2*{x(10)}", d["Lmax"], nf="0.0")
            bk.formula(ws, rr, 13, f"={x(12)}/{sfn}", d["Lmax_sf"], nf="0.0")
            bk.formula(ws, rr, 14, f"={x(12)}/{x(5)}", d["nmax"], nf="0.000")
            bk.formula(ws, rr, 15, f"={x(13)}/{x(5)}", d["nmax_sf"], nf="0.000")
            bk.formula(ws, rr, 16, f"={x(6)}/{x(13)}", d["use"], nf="0%")
            bk.formula(ws, rr, 17, f"=SQRT(2*{x(6)}/({x(4)}*S_m2*{x(10)}))", d["V_stall"], nf="0.00")
            bk.formula(ws, rr, 18, f"={x(1)}/({x(17)}*SQRT({sfn}))", d["margin_sf"], nf="0.000")
        else:
            for c_ in range(10, 19):
                bk.put(ws, rr, c_, "no stall found", "table")
        bk.formula(ws, rr, 19, f"={x(6)}/W_N", L / case.weight, nf="0.000")
    r1 = r0 + len(mp) - 1
    r = r1 + 2
    # sentences
    r = bk.section(ws, r, "In words", "One sentence per flight point, written by formula from the table above (they change with it).", 12)
    for i, d in enumerate(mp):
        rr = r0 + i
        x = lambda c_: f"{chr(65 + c_)}{rr + 1}"  # noqa: E731
        if "CLmax" in d:
            # ROUND(x,k)& gives the number in the reader's own decimal notation (TEXT(x,"0.0") does not in a Spanish Excel)
            g = lambda v, k: f"{round(v, k):g}"  # noqa: E731
            txt = (f"{d['label']}: at {g(d['V'], 1)} m/s the wing generates {g(d['L'], 0)} N, {g(d['L'] / d['W'], 2)} times its weight of {g(d['W'], 0)} N (CL = {g(d['CL_req'], 3)}). It could generate up to "
                   f"{g(d['Lmax'], 0)} N ({g(d['nmax'], 2)} × weight) as calculated and {g(d['Lmax_sf'], 0)} N ({g(d['nmax_sf'], 2)} × weight) with the safety factor: it uses {g(d['use'] * 100, 0)} % of the usable lift.")
            f = (f'={x(0)}&": at "&ROUND({x(1)},1)&" m/s the wing generates "&ROUND({x(6)},0)&" N, "&ROUND({x(7)},2)&" times its weight of "&ROUND({x(5)},0)&" N (CL = "&ROUND({x(9)},3)'
                 f'&"). It could generate up to "&ROUND({x(12)},0)&" N ("&ROUND({x(14)},2)&" × weight) as calculated and "&ROUND({x(13)},0)&" N ("&ROUND({x(15)},2)'
                 f'&" × weight) with the safety factor: it uses "&ROUND({x(16)}*100,0)&" % of the usable lift."')
            ws.merge_range(r, 0, r, 12, "", bk.fmt("table", wrap=True, size=10))
            bk.formula(ws, r, 0, f, txt, "table", wrap=True, size=10)
            ws.set_row(r, 32)
            r += 1
    r += 1
    # speed sweep
    r = bk.section(ws, r, "The most lift the wing can carry over speed (the stall boundary of a V–n diagram)",
                   "For the weight of the design point. The load factor the wing can carry at a speed is its most lift divided by its weight; above the line lift = weight (1.0) the wing can pull up or fly in a gust. "
                   "CLmax is that of the stall speed (it rises a little with speed, so this is on the safe side).", 10)
    bk.header(ws, r, ["Speed V (m/s)", "Dynamic pressure (Pa)", "Most lift q S CLmax (N)", "… with the safety factor (N)", "Most lift ÷ weight", "… with the safety factor", "Weight line"], height=48)
    s0 = r + 1
    for j, d in enumerate(sw):
        rr = s0 + j
        x = lambda c_: f"{chr(65 + c_)}{rr + 1}"  # noqa: E731
        bk.put(ws, rr, 0, d["V"], "you", nf="0.00")
        bk.formula(ws, rr, 1, f"=0.5*rho*{x(0)}^2", d["q"], nf="0.0")
        bk.formula(ws, rr, 2, f"={x(1)}*S_m2*CLmax", d["Lmax"], nf="0.0")
        bk.formula(ws, rr, 3, f"={x(2)}/factor_lift", d["Lmax_sf"], nf="0.0")
        bk.formula(ws, rr, 4, f"={x(2)}/W_N", d["nmax"], nf="0.000")
        bk.formula(ws, rr, 5, f"={x(3)}/W_N", d["nmax_sf"], nf="0.000")
        bk.formula(ws, rr, 6, "=1", 1.0, nf="0.0")
    s1 = s0 + len(sw) - 1
    r = s1 + 2
    c1 = bk.scatter("Lift the wing can carry against its weight", "speed (m/s)", "lift ÷ design weight",
                    [_xy(S, (s0, s1, 0), (s0, s1, 4), "most lift, as calculated", BLUE), _xy(S, (s0, s1, 0), (s0, s1, 5), "most lift, with the safety factor", BLUE, dash="dash"),
                     _xy(S, (s0, s1, 0), (s0, s1, 6), "weight (lift = weight)", NAVY, width=1.5),
                     _xy(S, (r0, r1, 1), (r0, r1, 19), "mission points: lift generated", ORANGE, line=False, marker="circle", msize=9,
                         labels=[f"='{S}'!$A${r0 + 1 + i}" for i in range(len(mp))])], size=(620, 340), xlim=(0, sw[-1]["V"]),
                    ylim=(0, max(3.0, 1.15 * max(d["L"] / case.weight for d in mp), 1.1 * float(np.interp(1.6 * case.V, [x["V"] for x in sw], [x["nmax_sf"] for x in sw])))))
    ch = bk.wb.add_chart({"type": "column"})
    ch.add_series({"name": "lift generated (N)", "categories": [S, r0, 0, r1, 0], "values": [S, r0, 6, r1, 6], "fill": {"color": NAVY}, "data_labels": {"value": True, "num_format": "0", "font": {"size": 8}}})
    ch.add_series({"name": "most lift with the safety factor (N)", "categories": [S, r0, 0, r1, 0], "values": [S, r0, 13, r1, 13], "fill": {"color": BLUE}, "data_labels": {"value": True, "num_format": "0", "font": {"size": 8}}})
    ch.add_series({"name": "weight (N)", "categories": [S, r0, 0, r1, 0], "values": [S, r0, 5, r1, 5], "fill": {"color": AMBER}, "data_labels": {"value": True, "num_format": "0", "font": {"size": 8}}})
    ch.set_title({"name": "At each point of the mission", "name_font": {"size": 11, "bold": True, "color": NAVY}})
    ch.set_y_axis({"name": "N", "major_gridlines": {"visible": True, "line": {"color": "#E5E7EB", "width": 0.5}}})
    ch.set_legend({"position": "bottom", "font": {"size": 8}})
    ch.set_size({"width": 470, "height": 340})
    ws.insert_chart(r, 0, c1, {"x_offset": 6, "y_offset": 4, "object_position": 1})
    ws.insert_chart(r, 0, ch, {"x_offset": 640, "y_offset": 4, "object_position": 1})
    ws.set_column(0, 0, 34)
    ws.set_column(1, 19, 13)
    ws.freeze_panes(r0, 1)


# ------------------------------------------------------------------------------------------------ results
RES_ROWS = [  # label, unit, as-calculated formula, formula with the safety factor, rule, metrics key, key with the safety factor
    ("Lift coefficient required", "", "=CL_req", None, "", "CL_design", None),
    ("Lift-to-drag", "", "=LD_ratio", "=LD_ratio/factor_drag", "÷ SF on drag", "LD", "LD_sf"),
    ("Drag coefficient", "", "=CD_tot", "=CD_tot*factor_drag", "× SF on drag", "CD", "CD_sf"),
    ("Drag force", "N", "=D_N", "=D_N*factor_drag", "× SF on drag", "drag_N", "drag_N_sf"),
    ("Aerodynamic power", "W", "=P_W", "=P_W*factor_drag", "× SF on drag", "P_req_W", "P_req_W_sf"),
    ("Maximum lift coefficient (first strip stalls)", "", "=CLmax", "=CLmax/factor_lift", "÷ SF on lift", "CL_max_first", "CL_max_first_sf"),
    ("Stall speed", "m/s", "=V_s", "=V_s*SQRT(factor_lift)", "× √SF on lift", "V_stall_mps", "V_stall_mps_sf"),
    ("Speed margin over the stall", "", "=margin_calc", "=margin_sf", "V / (V_s √SF on lift)", "speed_margin", "speed_margin_sf"),
    ("Highest strip utilisation", "", "=util_calc", "=util_sf", "× SF on lift", "util_max", "util_max_sf"),
    ("Pitching moment Cm about the reference point", "", "=Cm_ref", "=Cm_ref*factor_moment", "× SF on moments", "Cm_ref", "Cm_ref_sf"),
    ("Root bending coefficient", "", "=root_bend", "=root_bend*factor_moment", "× SF on moments", "root_bending_coeff", "root_bending_coeff_sf"),
    ("Roll power near the stall (share of linear)", "", "=roll_09", "=roll_09/factor_control", "÷ SF on control", "roll_authority_0p9", "roll_authority_0p9_sf"),
]


def _try(env: Env, f):
    try:
        return env.run(f)
    except (KeyError, NameError):          # a name that this design does not have (eval turns a missing name into NameError)
        return None


def sheet_results(bk: Book, case, start, points, mp: list, cn: dict) -> None:
    from .optimise import metrics_sf
    from .points import cases_for, point_list
    ws, S, env = bk.ws["Results"], "Results", bk.env
    r = bk.title(ws, "Results", "The results of the design, as calculated and with the safety factor (the formulas follow the Calculations sheet).")
    ms = ss = None
    if start is not None:
        ms = start[1].full()["metrics"]
        ss = with_safety(ms, start[1].sf)
    heads = ["Result", "Unit", "As calculated", "Rule of the safety factor", "With the safety factor"]
    if ms is not None:
        heads += [f"{start[0]}: as calculated", f"{start[0]}: with the safety factor", "Change with the safety factor"]
    r = bk.section(ws, r, "Key results", "", 8)
    bk.header(ws, r, heads, height=44)
    r += 1
    first = r
    changes = []
    for lab, unit, f1, f2, rule, mk, sk in RES_ROWS:
        v1 = _try(env, f1)
        if v1 is None:
            continue
        bk.put(ws, r, 0, lab, "table")
        bk.put(ws, r, 1, unit, "table")
        bk.formula(ws, r, 2, f1, v1)
        bk.put(ws, r, 3, rule, "table", color=STEEL, size=9)
        v2 = _try(env, f2) if f2 else None
        if f2 and v2 is not None:
            bk.formula(ws, r, 4, f2, v2)
        else:
            bk.put(ws, r, 4, "", "table")
        if ms is not None:
            a = ms.get(mk)
            b = ss.get(sk) if sk else a
            for c_, v in ((5, a), (6, b)):
                if v is not None and (not isinstance(v, float) or math.isfinite(v)):
                    bk.put(ws, r, c_, v, "lines")
                else:
                    bk.put(ws, r, c_, "–", "table")
            if (v2 if f2 else v1) is not None and b not in (None, 0) and isinstance(b, (int, float)) and math.isfinite(b):
                cur = v2 if f2 else v1
                bk.formula(ws, r, 7, f"=({chr(65 + (4 if f2 else 2))}{r + 1}-G{r + 1})/ABS(G{r + 1})", (cur - b) / abs(b), "calc", nf="0.0%")
                changes.append((lab, r))
            else:
                bk.put(ws, r, 7, None, "table")
        r += 1
    last = r - 1
    r += 1
    # drag breakdown
    ch_list = []
    if "CD_profile" in env.defs:
        r = bk.section(ws, r, "Drag breakdown at the design point", "", 4)
        bk.header(ws, r, ["Part of the drag", "CD", "Share of CD", "Where it comes from"], height=20)
        d0 = r + 1
        for i, (nm, nmk, src) in enumerate((("Profile (the sections, strip by strip)", "CD_profile", "the airfoil polars at each strip's Reynolds number"), ("Induced (the lifting line)", "CD_induced", "the downwash of the wing's own vortex sheet"),
                                            ("Fuselage", "CD_fuselage", "skin friction, form and cross-flow"))):
            bk.put(ws, d0 + i, 0, nm, "table")
            bk.formula(ws, d0 + i, 1, f"={nmk}", env[nmk], nf="0.00000")
            bk.formula(ws, d0 + i, 2, f"=B{d0 + i + 1}/CD_tot", env[nmk] / env["CD_tot"], nf="0.0%")
            bk.put(ws, d0 + i, 3, src, "table", color=STEEL, size=9)
        r = d0 + 4
        ch = bk.wb.add_chart({"type": "doughnut"})
        ch.add_series({"name": "Drag breakdown", "categories": [S, d0, 0, d0 + 2, 0], "values": [S, d0, 1, d0 + 2, 1], "points": [{"fill": {"color": BLUE}}, {"fill": {"color": ORANGE}}, {"fill": {"color": STEEL}}],
                       "data_labels": {"percentage": True, "font": {"size": 9, "color": "#FFFFFF", "bold": True}}})
        ch.set_title({"name": "Where the drag comes from", "name_font": {"size": 11, "bold": True, "color": NAVY}})
        ch.set_legend({"position": "bottom", "font": {"size": 8}})
        ch.set_size({"width": 380, "height": 300})
        ch_list.append(("doughnut", ch))
    if changes:
        ch = bk.wb.add_chart({"type": "bar"})
        ch.add_series({"name": "Change with the safety factor", "categories": [S, changes[0][1], 0, changes[-1][1], 0], "values": [S, changes[0][1], 7, changes[-1][1], 7], "fill": {"color": ORANGE}, "invert_if_negative": False,
                       "data_labels": {"value": True, "num_format": "0%", "font": {"size": 8}}})
        ch.set_title({"name": f"Change against the {start[0].lower()}", "name_font": {"size": 11, "bold": True, "color": NAVY}})
        ch.set_legend({"none": True})
        ch.set_y_axis({"reverse": True, "num_font": {"size": 8}, "label_position": "low"})
        ch.set_x_axis({"num_format": "0%", "major_gridlines": {"visible": True, "line": {"color": "#E5E7EB", "width": 0.5}}})
        ch.set_size({"width": 620, "height": 340})
        ch_list.append(("bar", ch))
    # every flight point
    r = bk.section(ws, r, "At every flight point", "The same wing in each flight condition; every result carries the safety factor. The lift against the weight is on the Mission lift sheet.", 10)
    bk.header(ws, r, ["Flight point", "Speed (m/s)", "Load factor n", "CL needed", "L/D (SF)", "Stall speed (SF, m/s)", "Speed margin (SF)", "Strip utilisation (SF)", "Lift ÷ weight", "Most lift ÷ weight (SF)"], height=44)
    r += 1
    for d, cc in zip(mp, cases_for(case, points)):
        m = metrics_sf(cc, set())
        bk.put(ws, r, 0, d["label"], "table", bold=True)
        for c_, v in ((1, d["V"]), (2, d["n"]), (3, d["CL_req"]), (4, m.get("LD_sf")), (5, m.get("V_stall_mps_sf")), (6, m.get("speed_margin_sf")), (7, m.get("util_max_sf")), (8, d["L"] / d["W"]), (9, d.get("nmax_sf"))):
            bk.put(ws, r, c_, v if v is not None else "–", "lines")
        r += 1
    r += 1
    ws.set_column(0, 0, 46)
    ws.set_column(1, 1, 12)
    ws.set_column(2, 9, 17)
    ws.set_column(3, 3, 24)
    x0 = r
    for kind, ch in ch_list:
        ws.insert_chart(x0, 0, ch, {"x_offset": 6, "y_offset": 4, "object_position": 1})
        x0 += 17
    ws.freeze_panes(3, 1)


# ------------------------------------------------------------------------------------------------ the optimiser
def sheet_optimiser(bk: Book, case, opt, points, start) -> None:
    from .optimise import OBJECTIVES
    from .points import point_list
    from .report import limit_rows
    ws, S = bk.ws["Optimiser"], "Optimiser"
    p = case.p
    r = bk.title(ws, "The optimiser: what it was given and what it found", "Orange cells are the optimiser's own inputs: the objectives and their weights, the inputs it was allowed to change and between which values, the flight points and the limits.")
    r = bk.section(ws, r, "How the search ran", "", 6)
    for k, v, nf in (("Method", "differential evolution (global search of the allowed ranges)", None), ("Candidate designs evaluated", opt.n_evals, "#,##0"), ("Time (s)", opt.seconds, "0.0"),
                     ("Found a design that meets every limit", "yes" if opt.feasible else "NO", None), ("Safety factor used in every limit and objective", float(p["safety_factor"]), "0.00")):
        bk.put(ws, r, 0, k, "table")
        bk.put(ws, r, 1, v, "opt", nf=nf, align="left")
        r += 1
    if opt.message:
        bk.put(ws, r, 0, "Message", "table")
        bk.put(ws, r, 1, opt.message, "opt", align="left")
        r += 1
    r += 1
    # objectives
    r = bk.section(ws, r, "Objectives", "Each objective is divided by its value for the starting design and the results are added with these weights (at every flight point, weighted again by the point's weight).", 6)
    bk.header(ws, r, ["Objective", "Direction", "Weight", "Result used"], height=20)
    r += 1
    for k, w in opt.weights.items():
        lab, mk, sign, _need = OBJECTIVES[k]
        bk.put(ws, r, 0, lab, "table")
        bk.put(ws, r, 1, "maximise" if sign < 0 else "minimise", "opt")
        bk.put(ws, r, 2, float(w), "opt", nf="0.00")
        bk.put(ws, r, 3, mk.replace("_sf", " (with the safety factor)"), "table", color=STEEL, size=9)
        r += 1
    r += 1
    # variables
    r = bk.section(ws, r, "Inputs the optimiser was allowed to change", "From, to: the range it searched. Started at: the starting design. Optimised: the value it settled on. Airfoils and choices are searched among a list of options.", 7)
    bk.header(ws, r, ["Input", "Unit", "From", "To", "Started at", "Optimised", "Change"], height=20)
    r += 1
    v0 = r
    for var in opt.variables:
        pr = BY_KEY[var.key]
        now = opt.changed[var.key]
        was = opt.problem.p0.get(var.key) if opt.problem is not None else None
        bk.put(ws, r, 0, pr.label.strip(), "table")
        bk.put(ws, r, 1, "" if pr.unit == "-" else pr.unit, "table")
        if var.kind == "float":
            bk.put(ws, r, 2, var.lo, "opt")
            bk.put(ws, r, 3, var.hi, "opt")
            bk.put(ws, r, 4, was, "table")
            bk.put(ws, r, 5, now, "opt")
            if isinstance(was, (int, float)) and was:
                bk.formula(ws, r, 6, f"=(F{r + 1}-E{r + 1})/ABS(E{r + 1})", (now - was) / abs(was), "calc", nf="0.0%")
        else:
            nm = lambda x: AF.display_name(x) if (var.kind == "airfoil" and x != AF.PARAM) else str(x)  # noqa: E731
            bk.put(ws, r, 2, f"{len(var.pool)} options", "opt")
            bk.put(ws, r, 3, "", "opt")
            bk.put(ws, r, 4, nm(was) if was is not None else "", "table")
            bk.put(ws, r, 5, nm(now), "opt")
        r += 1
    r += 1
    # flight points
    pts = point_list(points)
    r = bk.section(ws, r, "Flight points", "The same wing is evaluated at each point (speed, altitude, mass, load factor change; the wing does not). A weight of zero makes a point that only checks the limits.", 8)
    bk.header(ws, r, ["Flight point", "× design speed", "Altitude change (m)", "× mass", "Load factor", "Weight in the objective", "… normalised", "Least speed margin (SF)"], height=34)
    r += 1
    pw = opt.problem.point_weights if opt.problem is not None else [1.0 / len(pts)] * len(pts)
    for i, pt in enumerate(pts):
        bk.put(ws, r, 0, pt.name, "opt", bold=True)
        bk.put(ws, r, 1, pt.speed_factor, "opt", nf="0.00")
        bk.put(ws, r, 2, pt.altitude_delta_m, "opt", nf="0")
        bk.put(ws, r, 3, pt.mass_factor, "opt", nf="0.00")
        bk.put(ws, r, 4, pt.load_factor if pt.load_factor is not None else float(p["load_factor"]), "opt", nf="0.0")
        bk.put(ws, r, 5, pt.weight, "opt", nf="0.00")
        bk.put(ws, r, 6, float(pw[i]), "table", nf="0.000")
        bk.put(ws, r, 7, float(pt.margin_min) if pt.margin_min is not None else float(p["speed_margin_min"]), "opt", nf="0.00")
        r += 1
    r += 1
    # limits
    r = bk.section(ws, r, "Limits (all tested with the safety factor)", "A design that breaks a limit is not accepted.", 4)
    bk.header(ws, r, ["Limit", "On", "Value", "Applies to"], height=20)
    r += 1
    for lab, on, val, where in (("The stall must start at η ≤ (share of the semi-span)", p["con_stall_on"], p["stall_onset_max"], "every flight point"),
                                ("Speed margin over the stall ≥", p["con_margin_on"], p["speed_margin_min"], "design point (other points: their own)"),
                                ("Strip utilisation ≤", p["con_util_on"], 1.0, "every flight point"), ("|Cm| about the reference point ≤", p["con_cm_on"], p["cm_abs_max"], "design point"),
                                ("Roll power left near the stall ≥ (share of linear)", p["con_roll_on"] and p["aileron_on"], p["roll_authority_min"], "design point"),
                                ("Wing loading ≤ (N/m²)", p["con_loading_on"], p["wing_loading_max"], "design point")):
        bk.put(ws, r, 0, lab, "table")
        bk.put(ws, r, 1, bool(on), "opt")
        bk.put(ws, r, 2, float(val), "opt")
        bk.put(ws, r, 3, where, "table", color=STEEL, size=9)
        r += 1
    r += 1
    # start -> optimised at each point
    if len(opt.points) >= 1 and opt.baseline_points:
        r = bk.section(ws, r, "Every flight point: starting design → optimised", "Results with the safety factor.", 9)
        bk.header(ws, r, ["Flight point", "L/D: start", "L/D: optimised", "Stall speed: start (m/s)", "Stall speed: optimised", "Speed margin: start", "Speed margin: optimised", "Utilisation: start", "Utilisation: optimised"], height=34)
        r += 1
        for i, ptn in enumerate(opt.points):
            m0 = opt.baseline_points[i]
            m1 = opt.best_points[i] if i < len(opt.best_points) else {}
            bk.put(ws, r, 0, ptn.label(), "table", bold=True)
            for c_, (a, k) in enumerate((("0", "LD_sf"), ("1", "LD_sf"), ("0", "V_stall_mps_sf"), ("1", "V_stall_mps_sf"), ("0", "speed_margin_sf"), ("1", "speed_margin_sf"), ("0", "util_max_sf"), ("1", "util_max_sf")), 1):
                v = (m0 if a == "0" else m1).get(k)
                bk.put(ws, r, c_, v if v is not None else "–", "lines")
            r += 1
        r += 1
    # limits at the optimum
    lrows = limit_rows(case, points)
    if lrows:
        r = bk.section(ws, r, "Every limit at the optimum", "", 5)
        bk.header(ws, r, ["Flight point", "Limit", "Value", "Limit value", "Met"], height=20)
        r += 1
        for name, lim, val, bound, met in lrows:
            bk.put(ws, r, 0, name, "table")
            bk.put(ws, r, 1, lim, "table")
            bk.put(ws, r, 2, float(val) if re.match(r"^-?[0-9.,]+(e-?[0-9]+)?$", str(val).replace(",", "")) else val, "lines", align="right")
            bk.put(ws, r, 3, bound, "table", align="right")
            bk.put(ws, r, 4, met, "calc" if met == "yes" else "table", bold=True, color=None if met == "yes" else RED)
            r += 1
        r += 1
    # convergence
    r = bk.section(ws, r, "Convergence", "Best weighted objective of each generation (lower is better).", 4)
    bk.header(ws, r, ["Generation", "Best objective"], height=20)
    r += 1
    h0 = r
    for i, v in enumerate(opt.history):
        bk.put(ws, r, 0, i + 1, "table", nf="0")
        bk.put(ws, r, 1, float(v), "opt", nf="0.0000")
        r += 1
    if opt.history:
        hv = [float(v) for v in opt.history]
        pad = max((max(hv) - min(hv)) * 0.15, 0.01 * max(abs(min(hv)), abs(max(hv)), 1e-3))
        ch = bk.scatter("Optimiser progress", "generation", "best weighted objective", [_xy(S, (h0, r - 1, 0), (h0, r - 1, 1), "best objective", ORANGE, marker="circle", msize=4)], size=(560, 300),
                        legend="none", ylim=(min(hv) - pad, max(hv) + pad))
        ws.insert_chart(h0 - 2, 3, ch, {"x_offset": 20, "y_offset": 0, "object_position": 1})
    ws.set_column(0, 0, 52)
    ws.set_column(1, 8, 17)
    ws.set_column(1, 1, 26)


# ------------------------------------------------------------------------------------------------ the curves
def sheet_curves(bk: Book, cases: list, cn: dict, perf: dict | None) -> None:
    from xlsxwriter.utility import xl_col_to_name
    ws, S = bk.ws["Curves"], "Curves"
    case = cases[0][1]
    r = bk.title(ws, "Curves: lift, drag, moment and performance",
                 "The charts are at the top, the tables they are drawn from are below them. 'With the safety factor' columns are formulas: change the safety factor on the Inputs sheet and they follow.")
    rows_per = int(math.ceil(310 / 20.0)) + 1
    top = r
    r = top + 4 * rows_per + 1
    hdr = r
    heads = ["Root angle α (°)", "CL", "CD", "CDp (profile)", "CDi (induced)", "CDf (fuselage)", "L/D = CL / CD", "Cm about the reference", "Cm about the ac", "CD with the SF", "L/D with the SF"]
    blocks = []
    for bi, (lab, c) in enumerate(cases):
        c0 = bi * 12
        cv = c.full()["curve"]
        n = int(np.argmax(cv["CL"])) + 1
        ws.merge_range(hdr, c0, hdr, c0 + 10, f"{lab}: against the root angle of attack, up to the lift peak", bk.fmt("text", bold=True, color=NAVY, border=False))
        bk.header(ws, hdr + 1, heads, c0=c0, height=44)
        sfd = bk.env["factor_drag"] if bi == 0 else c.sf["drag"]
        for j in range(n):
            rr = hdr + 2 + j
            x = lambda k: f"{xl_col_to_name(c0 + k)}{rr + 1}"  # noqa: E731
            for k_, key in enumerate(("alpha_deg", "CL", "CD", "CDp", "CDi", "CDf")):
                bk.put(ws, rr, c0 + k_, float(cv[key][j]), "lines", nf="0.00000" if k_ >= 2 else "0.000")
            bk.formula(ws, rr, c0 + 6, f"={x(1)}/{x(2)}", cv["CL"][j] / cv["CD"][j], nf="0.00")
            bk.put(ws, rr, c0 + 7, float(cv["Cm_ref"][j]), "lines", nf="0.0000")
            bk.put(ws, rr, c0 + 8, float(cv["Cm_ac"][j]), "lines", nf="0.0000")
            f = "factor_drag" if bi == 0 else f"{sfd}"
            bk.formula(ws, rr, c0 + 9, f"={x(2)}*{f}", cv["CD"][j] * sfd, nf="0.00000")
            bk.formula(ws, rr, c0 + 10, f"={x(1)}/{x(9)}", cv["CL"][j] / (cv["CD"][j] * sfd), nf="0.00")
        blocks.append((hdr + 2, hdr + 1 + n, c0, lab, c))
    # performance over speed
    pb = None
    if perf and len(perf.get("V", [])) > 2:
        c0 = len(cases) * 12
        ws.merge_range(hdr, c0, hdr, c0 + 5, "Performance over speed (level flight, L = n W)", bk.fmt("text", bold=True, color=NAVY, border=False))
        bk.header(ws, hdr + 1, ["Speed V (m/s)", "CL needed", "CD", "L/D", "Power D V (W)", "Power with the SF (W)"], c0=c0, height=44)
        for j in range(len(perf["V"])):
            rr = hdr + 2 + j
            x = lambda k: f"{xl_col_to_name(c0 + k)}{rr + 1}"  # noqa: E731
            v, cl, cd = float(perf["V"][j]), float(perf["CL"][j]), float(perf["CD"][j])
            bk.put(ws, rr, c0, v, "lines", nf="0.00")
            bk.put(ws, rr, c0 + 1, cl, "lines", nf="0.000")
            bk.put(ws, rr, c0 + 2, cd, "lines", nf="0.00000")
            bk.formula(ws, rr, c0 + 3, f"={x(1)}/{x(2)}", cl / cd, nf="0.00")
            bk.formula(ws, rr, c0 + 4, f"={x(2)}*0.5*rho*{x(0)}^3*S_m2", cd * 0.5 * bk.env["rho"] * v ** 3 * bk.env["S_m2"], nf="0.0")
            bk.formula(ws, rr, c0 + 5, f"={x(4)}*factor_drag", cd * 0.5 * bk.env["rho"] * v ** 3 * bk.env["S_m2"] * bk.env["factor_drag"], nf="0.0")
        pb = (hdr + 2, hdr + 1 + len(perf["V"]), c0)
    # charts
    colr = [BLUE, ORANGE]
    cl_ser, pol_ser, ld_ser, cm_ser, cma_ser = [], [], [], [], []
    for bi, (a0, a1, c0, lab, c) in enumerate(blocks):
        col = colr[bi % 2]
        cl_ser.append(_xy(S, (a0, a1, c0), (a0, a1, c0 + 1), lab, col))
        pol_ser.append(_xy(S, (a0, a1, c0 + 2), (a0, a1, c0 + 1), lab, col))
        ld_ser.append(_xy(S, (a0, a1, c0 + 1), (a0, a1, c0 + 6), lab, col))
        cm_ser.append(_xy(S, (a0, a1, c0 + 1), (a0, a1, c0 + 7), f"{lab}: about the reference point", col))
        cma_ser.append(_xy(S, (a0, a1, c0 + 1), (a0, a1, c0 + 8), f"{lab}: about the aerodynamic centre", col, dash="round_dot", width=1.3))
        if bi == 0:
            pol_ser.append(_xy(S, (a0, a1, c0 + 9), (a0, a1, c0 + 1), f"{lab}, with the safety factor", col, dash="dash", width=1.2))
            ld_ser.append(_xy(S, (a0, a1, c0 + 1), (a0, a1, c0 + 10), f"{lab}, with the safety factor", col, dash="dash", width=1.2))
    if "alpha_nl_deg" in cn:
        ra, ca = cn["alpha_nl_deg"]
        rc, cc_ = cn["CL_req"]
        cl_ser.append({"sheet": "Calculations", "x": (ra, ra, ca), "y": (rc, rc, cc_), "name": "design point", "color": NAVY, "line": False, "marker": "circle", "msize": 9})
        pol_ser.append({"sheet": "Calculations", "x": (cn["CD_tot"][0], cn["CD_tot"][0], 2), "y": (rc, rc, cc_), "name": "design point", "color": NAVY, "line": False, "marker": "circle", "msize": 9}) if "CD_tot" in cn else None
    ch = [bk.scatter("Wing lift curve", "root angle of attack (°)", "CL", cl_ser),
          bk.scatter("Drag polar", "CD", "CL", pol_ser),
          bk.scatter("Lift-to-drag against lift", "CL", "L/D", ld_ser),
          bk.scatter("Pitching moment against lift", "CL", "Cm", cm_ser + cma_ser)]
    a0, a1, c0, lab, c = blocks[0]
    ar = bk.wb.add_chart({"type": "area", "subtype": "stacked"})
    for k_, (nm, col) in enumerate((("profile (strip by strip)", BLUE), ("induced (lifting line)", ORANGE), ("fuselage", STEEL))):
        ar.add_series({"name": nm, "categories": [S, a0, c0 + 1, a1, c0 + 1], "values": [S, a0, c0 + 3 + k_, a1, c0 + 3 + k_], "fill": {"color": col}})
    ar.set_title({"name": "Drag breakdown against lift", "name_font": {"size": 11, "bold": True, "color": NAVY}})
    ar.set_x_axis({"name": "CL", "num_format": "0.0", "interval_unit": max(1, (a1 - a0) // 8), "num_font": {"size": 8}})
    ar.set_y_axis({"name": "CD", "major_gridlines": {"visible": True, "line": {"color": "#E5E7EB", "width": 0.5}}})
    ar.set_legend({"position": "bottom", "font": {"size": 8}})
    ar.set_size({"width": 540, "height": 310})
    ch.append(ar)
    if pb is not None:
        p0, p1, pc = pb
        ch.append(bk.scatter("Power required against speed", "speed (m/s)", "W", [_xy(S, (p0, p1, pc), (p0, p1, pc + 4), "as calculated", BLUE), _xy(S, (p0, p1, pc), (p0, p1, pc + 5), "with the safety factor", BLUE, dash="dash")]))
        ch.append(bk.scatter("Lift-to-drag against speed", "speed (m/s)", "L/D", [_xy(S, (p0, p1, pc), (p0, p1, pc + 3), "L/D", ORANGE)]))
    ch.append(bk.scatter("Pitching moment against angle", "root angle of attack (°)", "Cm", [_xy(S, (a0, a1, c0), (a0, a1, c0 + 7), "about the reference point", BLUE), _xy(S, (a0, a1, c0), (a0, a1, c0 + 8), "about the aerodynamic centre", ORANGE, dash="round_dot")]))
    bk.place(ws, top, ch)
    ws.set_column(0, 40, 12.5)


# ------------------------------------------------------------------------------------------------ the wing and its airfoils
def sheet_wing(bk: Book, case) -> None:
    from .report_figures import _station_foils
    ws, S = bk.ws["Wing and airfoils"], "Wing and airfoils"
    w, p = case.wing, case.p
    m = case.full()["metrics"]
    r = bk.title(ws, "The wing and its airfoils", "Drawings at the top (native charts: the planform and the sections), the coordinates they are drawn from below.")
    rows_per = int(math.ceil(330 / 20.0)) + 1
    top = r
    r = top + 2 * rows_per + 1
    # stations
    r = bk.section(ws, r, "Wing at its stations", "", 8)
    bk.header(ws, r, ["η", "y (m)", "Chord (m)", "Chord / root", "Sweep of the panel (°)", "Dihedral (°)", "Twist (° nose down)", "Airfoil"], height=34)
    r += 1
    foil_at = {round(float(e), 4): af for e, af in zip(case.foil_eta, case.foils)}
    for k, e in enumerate(w.eta):
        af = foil_at.get(round(float(e), 4))
        vals = (float(e), float(e) * 0.5 * w.span, float(w.chord(e)), float(w.ratio[k]), float(w.sweep_deg[min(k, len(w.sweep_deg) - 1)]), float(w.dihedral_deg[min(k, len(w.dihedral_deg) - 1)]), float(w.twist_deg[k]))
        for c_, v in enumerate(vals):
            bk.put(ws, r, c_, v, "lines", nf="0.000")
        bk.put(ws, r, 7, af.label if af else "(blended)", "table")
        r += 1
    r += 1
    # airfoil table
    r = bk.section(ws, r, "Section data at the design Reynolds number (NeuralFoil)", "", 12)
    bk.header(ws, r, ["Station", "η", "Airfoil", "Re", "t/c (%)", "Camber (%)", "cl max", "Stall angle (°)", "cd min", "(cl/cd) max", "α₀ (°)", "cm₀", "Confidence"], height=34)
    r += 1
    foils = _station_foils(case)
    for st, af, eta in foils:
        re_ = float(case.air.density * case.V * case.wing.chord(eta) / case.air.viscosity)
        pol = af.polar(re_)
        vals = (eta, af.label, re_, af.thickness * 100, af.shape["camber_max"] * 100, pol["cl_max"], pol["alpha_stall_deg"], pol["cd_min"], pol["ld_max"], pol["alpha0_deg"], pol["cm0"], pol["confidence_min"])
        bk.put(ws, r, 0, st, "table")
        for c_, v in enumerate(vals, 1):
            bk.put(ws, r, c_, v, "lines" if c_ != 2 else "table", nf="#,##0" if c_ == 3 else None)
        r += 1
    r += 1
    # outline coordinates
    r = bk.section(ws, r, "Planform and fuselage outlines (x nose up, y along the span, metres)", "Each pair of columns is one closed path, as drawn in the top view.", 12)
    heads = ["Wing: y", "Wing: x", "Quarter chord: y", "Quarter chord: x", "Fuselage: y", "Fuselage: x", "MAC: y", "MAC: x", "ac, cg: y", "ac, cg: x"]
    bk.header(ws, r, heads, height=30)
    r += 1
    o = w.outline()
    ys, xl, xt, xq = np.asarray(o["y"], float), np.asarray(o["x_le"], float), np.asarray(o["x_te"], float), np.asarray(o["x_qc"], float)
    wy = np.concatenate([ys, ys[::-1], -ys, -ys[::-1], ys[:1]])
    wx = -np.concatenate([xl, xt[::-1], xt, xl[::-1], xl[:1]])
    qy = np.concatenate([-ys[::-1], ys])
    qx = -np.concatenate([xq[::-1], xq])
    cols = [(wy, wx), (qy, qx)]
    if case.fuse is not None:
        xf, hw = case.fuse.outline()
        x_nose = -p["wing_x_fraction"] * case.fuse.length
        xw = xf + x_nose
        cols.append((np.concatenate([hw, -hw[::-1], hw[:1]]), np.concatenate([-xw, -xw[::-1], -xw[:1]])))
    else:
        cols.append((np.array([]), np.array([])))
    cols.append((np.array([w.y_mac, w.y_mac]), -np.array([w.x_le_mac, w.x_le_mac + w.mac])))
    cols.append((np.array([0.0, 0.0]), np.array([-m["x_ac_m"], -case.x_ref])))
    o0 = r
    for k, (a, b) in enumerate(cols):
        for j in range(len(a)):
            bk.put(ws, o0 + j, 2 * k, float(a[j]), "lines", nf="0.0000")
            bk.put(ws, o0 + j, 2 * k + 1, float(b[j]), "lines", nf="0.0000")
    n_o = [len(a) for a, _b in cols]
    r = o0 + max(n_o) + 2
    # airfoil coordinates
    r = bk.section(ws, r, "Airfoil coordinates (chord = 1): the closed outline, trailing edge → upper surface → leading edge → lower surface", "", 10)
    heads = []
    for st, af, eta in foils:
        heads += [f"{st}: x", f"{st}: y"]
    bk.header(ws, r, heads, height=30)
    r += 1
    a0 = r
    sizes = []
    for k, (st, af, eta) in enumerate(foils):
        up, lo = AF.split_surfaces(af.coords)
        x = np.concatenate([up[:, 0], lo[::-1, 0]])
        y = np.concatenate([up[:, 1], lo[::-1, 1]])
        sizes.append(len(x))
        for j in range(len(x)):
            bk.put(ws, a0 + j, 2 * k, float(x[j]), "lines", nf="0.0000")
            bk.put(ws, a0 + j, 2 * k + 1, float(y[j]), "lines", nf="0.0000")
    r = a0 + max(sizes) + 2
    # thickness and camber
    r = bk.section(ws, r, "Thickness and camber distributions", "", 10)
    heads = []
    for st, af, eta in foils:
        heads += [f"{st}: x/c", f"{st}: thickness", f"{st}: camber"]
    bk.header(ws, r, heads, height=30)
    r += 1
    t0 = r
    tsz = []
    for k, (st, af, eta) in enumerate(foils):
        sh = af.shape
        tsz.append(len(sh["x"]))
        for j in range(len(sh["x"])):
            bk.put(ws, t0 + j, 3 * k, float(sh["x"][j]), "lines", nf="0.000")
            bk.put(ws, t0 + j, 3 * k + 1, float(sh["thickness"][j]), "lines", nf="0.0000")
            bk.put(ws, t0 + j, 3 * k + 2, float(sh["camber"][j]), "lines", nf="0.0000")
    # charts
    half = 0.5 * w.span
    allx = np.concatenate([wx, cols[2][1]]) if len(cols[2][1]) else wx
    yr = (float(np.min(allx)), float(np.max(allx)))
    xr = 1.25 * half
    wpx = 560
    hpx = int(min(520, max(240, wpx * (yr[1] - yr[0] + 0.1) / (2 * xr))))
    ser = [_xy(S, (o0, o0 + n_o[0] - 1, 0), (o0, o0 + n_o[0] - 1, 1), "wing", NAVY, width=2.2),
           _xy(S, (o0, o0 + n_o[1] - 1, 2), (o0, o0 + n_o[1] - 1, 3), "quarter-chord line", STEEL, dash="dash", width=1.2)]
    if n_o[2]:
        ser.append(_xy(S, (o0, o0 + n_o[2] - 1, 4), (o0, o0 + n_o[2] - 1, 5), "fuselage", BLUE, width=1.8))
    ser += [_xy(S, (o0, o0 + 1, 6), (o0, o0 + 1, 7), "mean aerodynamic chord", ORANGE, width=3.5),
            {"sheet": S, "x": (o0, o0, 8), "y": (o0, o0, 9), "name": "aerodynamic centre (wing alone)", "color": GREEN, "line": False, "marker": "diamond", "msize": 9},
            {"sheet": S, "x": (o0 + 1, o0 + 1, 8), "y": (o0 + 1, o0 + 1, 9), "name": "moment reference point", "color": RED, "line": False, "marker": "x", "msize": 9}]
    c1 = bk.scatter(f"Top view: span {w.span:.2f} m, area {w.area:.3f} m², AR {w.aspect_ratio:.1f}", "span position (m)", "along the fuselage, nose up (m)", ser, size=(wpx, hpx), xlim=(-xr, xr), ylim=(yr[0] - 0.05, yr[1] + 0.05))
    ser = [_xy(S, (a0, a0 + sizes[k] - 1, 2 * k), (a0, a0 + sizes[k] - 1, 2 * k + 1), f"{st} (η {eta:.2f}): {af.label}", _col_i(k), width=2.0) for k, (st, af, eta) in enumerate(foils)]
    c2 = bk.scatter("Airfoil at each station (chord = 1)", "x / c", "y / c", ser, size=(620, 300), xlim=(0, 1), ylim=(-0.15, 0.2))
    c3 = bk.scatter("Thickness distribution", "x / c", "t / c", [_xy(S, (t0, t0 + tsz[k] - 1, 3 * k), (t0, t0 + tsz[k] - 1, 3 * k + 1), st, _col_i(k)) for k, (st, af, eta) in enumerate(foils)], size=(440, 300))
    ws.insert_chart(top, 0, c1, {"x_offset": 6, "y_offset": 4, "object_position": 1})
    ws.insert_chart(top, 0, c2, {"x_offset": wpx + 20, "y_offset": 4, "object_position": 1})
    ws.insert_chart(top + int(math.ceil(300 / 20.0)) + 2, 0, c3, {"x_offset": wpx + 20, "y_offset": 4, "object_position": 1})
    ws.set_column(0, 30, 13)
    ws.set_column(7, 7, 22)


def _col_i(i: int) -> str:
    return [BLUE, ORANGE, GREEN, AMBER, NAVY, STEEL][i % 6]


# ------------------------------------------------------------------------------------------------ the valleys
def sheet_valleys(bk: Book, valley: dict) -> None:
    ws, S = bk.ws["Valleys"], "Valleys"
    rep = valley["rep"]
    r = bk.title(ws, "What can move: the valleys around the design", f"The free range of an input is how far it can move, with every other input held, before the objective loses {rep.tol_pct:g} % or a limit is broken. Centre: {valley.get('label', '')}.")
    bk.header(ws, r, ["Input", "Unit", "At the design", "Free from", "Free to", "Allowed from", "Allowed to", "Free share of its range", "Verdict", "Lower end limited by", "Upper end limited by"], height=34)
    r += 1
    r0 = r
    for v in sorted((x for x in rep.valleys if x.kind == "float"), key=lambda x: x.share):
        bk.put(ws, r, 0, v.label, "table")
        bk.put(ws, r, 1, v.unit, "table")
        for c_, val in ((2, v.x_star), (3, v.free_lo), (4, v.free_hi), (5, v.lo), (6, v.hi)):
            bk.put(ws, r, c_, float(val), "lines")
        bk.formula(ws, r, 7, f"=(E{r + 1}-D{r + 1})/(G{r + 1}-F{r + 1})", v.share, "calc", nf="0%")
        bk.put(ws, r, 8, v.verdict, "table", bold=True)
        bk.put(ws, r, 9, v.bind_lo, "table", color=STEEL, size=9)
        bk.put(ws, r, 10, v.bind_hi, "table", color=STEEL, size=9)
        r += 1
    r1 = r - 1
    r += 1
    if r1 >= r0:
        ch = bk.wb.add_chart({"type": "bar"})
        ch.add_series({"name": "Free share of its range", "categories": [S, r0, 0, r1, 0], "values": [S, r0, 7, r1, 7], "fill": {"color": GREEN}, "data_labels": {"value": True, "num_format": "0%", "font": {"size": 8}}})
        ch.set_title({"name": "How much of its range each input can move", "name_font": {"size": 11, "bold": True, "color": NAVY}})
        ch.set_legend({"none": True})
        ch.set_y_axis({"reverse": True, "num_font": {"size": 8}})
        ch.set_x_axis({"num_format": "0%", "min": 0, "max": 1, "major_gridlines": {"visible": True, "line": {"color": "#E5E7EB", "width": 0.5}}})
        ch.set_size({"width": 700, "height": max(260, 22 * (r1 - r0 + 1) + 90)})
        ws.insert_chart(r, 0, ch, {"x_offset": 6, "y_offset": 4, "object_position": 1})
    ws.set_column(0, 0, 40)
    ws.set_column(1, 8, 13)
    ws.set_column(9, 10, 30)
    ws.freeze_panes(4, 1)


# ------------------------------------------------------------------------------------------------ the first sheet
def sheet_readme(bk: Book, label: str, title: str, subtitle: str, opt, start, sheets: list) -> None:
    ws = bk.ws["Read me"]
    ws.hide_gridlines(2)
    ws.write_string(0, 0, "HORYZN wing aerodynamics lab", bk.fmt("text", bold=True, size=18, color=NAVY, border=False))
    ws.write_string(1, 0, f"{title}: {label}" + (f" · {subtitle}" if subtitle else ""), bk.fmt("text", size=11, color=STEEL, border=False))
    ws.write_string(2, 0, f"Generated {datetime.date.today().isoformat()}. Estimates, not measurements: treat absolute drag and stall speed as ±10 %; use the safety factor and compare designs with each other.", bk.fmt("text", size=9, color=STEEL, italic=True, border=False))
    r = 4
    r = bk.section(ws, r, "Colour code", "", 3)
    for k, (_fc, _bg, what) in KINDS.items():
        bk.put(ws, r, 0, 12.5, k, align="center")
        ws.write_string(r, 1, what, bk.fmt("text", border=False))
        r += 1
    r += 1
    r = bk.section(ws, r, "What is in each sheet", "", 3)
    what = {"Inputs": "Every input of the design, where it came from (yours, searched by the optimiser, or the lab's default) and its value.",
            "Optimiser": "What the optimiser was given (objectives, weights, ranges, flight points, limits) and what it found; the convergence.",
            "Calculations": "The calculation sheet: inputs → formulas → results, written as live Excel formulas with the lifting-line values beside them.",
            "Results": "The results as calculated and with the safety factor, the drag breakdown, the comparison with the starting design, every flight point.",
            "Mission lift": "How much lift the wing generates against its weight at every flight point, and the most it can generate over speed (a V–n diagram).",
            "Strips": "The lifting line strip by strip at the design point: the method, every strip's numbers, and the sums checked against the lift needed.",
            "Curves": "Lift, drag, moment and performance curves with their tables and charts (and the starting design's curves for comparison).",
            "Wing and airfoils": "The planform, the fuselage and the airfoils: drawings and coordinates.",
            "Valleys": "How far each searched input can move around the optimum, with the limit that ends each side."}
    for sh in sheets:
        if sh in what:
            ws.write_url(r, 0, f"internal:'{sh}'!A1", bk.fmt("text", bold=True, color=BLUE, border=False), sh)
            ws.write_string(r, 1, what[sh], bk.fmt("text", border=False))
            r += 1
    r += 1
    r = bk.section(ws, r, "How to use it", "", 3)
    for t in ("Change a blue, orange or grey value on the Inputs sheet (or the flight conditions on the Mission lift sheet) and every pale-blue formula recalculates: the calculation sheet, the results, the lift against the weight, the strip sums.",
              "Purple values come from the lifting line and the airfoil polars. They are the numbers the lab calculated for the inputs as exported and do not change when you edit an input: to recalculate them, change the input in the lab (page Inputs) and export again.",
              "Every chart is a native Excel chart: click it to change its colours, ranges or type; its data is on the same sheet.",
              "Names: each input and the main results have a name (for example mass_kg, W_N, q_Pa, CL_req, CD_tot), so a formula reads like the equation: =mass_kg*gravity. Formulas > Name Manager lists them.",
              "The safety factor is applied last, in the adverse direction of every result (drag and power up, usable lift down, moments and loads up, control authority down)."):
        ws.merge_range(r, 0, r, 6, t, bk.fmt("text", wrap=True, border=False))
        ws.set_row(r, 32)
        r += 1
    ws.set_column(0, 0, 22)
    ws.set_column(1, 1, 120)
    ws.set_column(2, 6, 6)


# ------------------------------------------------------------------------------------------------ the workbook
def build_workbook(design: tuple, start: tuple | None = None, points=None, opt=None, valley=None, fixed: dict | None = None, title: str = "Wing design workbook", subtitle: str = "") -> bytes:
    """The workbook as bytes. design: (label, Case); start: (label, Case) of the starting design; points: the flight points; opt: the optimiser Result the design came
    from; valley: the dictionary of the Valleys page; fixed: the state of every input (True value, False range, None auto) when the design is not an optimum."""
    from .geometry import wing_area
    label, case = design
    r = case.full()
    ok = bool(r.get("design_ok"))
    sd = MS.strip_data(case) if ok else None
    use = ["Read me", "Inputs"] + (["Optimiser"] if opt is not None else []) + ["Calculations", "Results", "Mission lift"] + (["Strips", "Curves"] if ok else []) + ["Wing and airfoils"] + (["Valleys"] if valley is not None else [])
    buf = io.BytesIO()
    bk = Book(buf, use)
    case.name = case.name or label
    mp = MS.mission_lift(case, points)
    sw = MS.speed_sweep(case)
    _, how_area = wing_area(case.p)
    sheet_inputs(bk, case, fixed, opt, title)
    if sd is not None:
        bk.env.defs["st_wcd"] = sd["w"] * sd["cd"]
        bk.env.defs["st_util"] = sd["cl_unc"] / sd["cap"]
    sections = calc_rows(bk, case, how_area, sd is not None)
    cn = sheet_calculations(bk, case, sections)
    if sd is not None:
        sheet_strips(bk, case, sd)
    if opt is not None:
        sheet_optimiser(bk, case, opt, points, start)
    sheet_results(bk, case, start, points, mp, cn)
    if mp and "Lmax" in mp[0]:
        sheet_mission(bk, case, points, mp, sw)
    else:
        bk.ws["Mission lift"].write_string(0, 0, "The wing has no stall in the model's range, so no lift capability can be shown.")
    if ok:
        sheet_curves(bk, [design] + ([start] if start is not None else []), cn, r.get("performance"))
    sheet_wing(bk, case)
    if valley is not None:
        sheet_valleys(bk, valley)
    sheet_readme(bk, label, title, subtitle, opt, start, use)
    bk.wb.close()
    return buf.getvalue()
