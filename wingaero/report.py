"""The PDF report of a wing design: what it is, how every number was calculated, the drawings and the charts.

    pdf_bytes = build_report(("Optimised design", case), start=("Starting design", case0), points=points, opt=result, valley=valley)

Written with reportlab (text, tables) and matplotlib (charts, report_figures.py): both are plain Python packages, so the report is made the same way on the PC
and on the free online server, with no browser and no extra program. The calculation sheet follows the model of the lab (docs/physics-lab.md): the numbers
are the ones of `Case.full()`, written with the formula and the working that give them.
"""
from __future__ import annotations

import datetime
import io
import math
import os

import numpy as np

from . import airfoil as AF
from . import mission as MS
from . import report_figures as RF
from .analysis import safety_factors, with_safety
from .core import G0
from .geometry import broadcast_stations, wing_area
from .params import BY_KEY, GROUPS, PARAMS, active, mode_of

NAVY, INK, ORANGE, BLUE, STEEL, LINE, MIST = "#1D3043", "#1F2933", "#C2410C", "#2C6596", "#6B7A89", "#D8DEE6", "#F3F7FA"
FONT, FONT_B, MONO = "DejaVuSans", "DejaVuSans-Bold", "DejaVuSansMono"
_fonts_done = False


def _register_fonts() -> None:
    """DejaVu (shipped with matplotlib) has the Greek letters and the symbols the equations need; the standard PDF fonts do not."""
    global _fonts_done
    if _fonts_done:
        return
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    d = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for name, fn in ((FONT, "DejaVuSans.ttf"), (FONT_B, "DejaVuSans-Bold.ttf"), (MONO, "DejaVuSansMono.ttf"), ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, os.path.join(d, fn)))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT_B, italic="DejaVuSans-Oblique", boldItalic=FONT_B)
    _fonts_done = True


# ------------------------------------------------------------------------------------------------ number formatting
def n(x, nd: int = 4) -> str:
    """A number with nd significant digits; thousands separated; '–' for a missing value."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax < 1e-3:
        return f"{x:.{max(nd - 1, 1)}e}"
    return f"{x:,.{nd}g}" if ax >= 1 else f"{x:.{nd}g}"


def pct(x) -> str:
    return "–" if x is None or not math.isfinite(x) else f"{100 * x:.0f} %"


# ------------------------------------------------------------------------------------------------ the calculation sheet
class Calc:
    """One section of the sheet: a title, a short explanation and rows (quantity, formula, the numbers put in, result)."""

    def __init__(self, title: str, note: str = ""):
        self.title, self.note = title, note
        self.rows: list[tuple] = []

    def add(self, quantity: str, formula: str, working: str, result: str) -> None:
        self.rows.append((quantity, formula, working, result))


def calc_sections(case, points=None) -> list[Calc]:
    """The step-by-step calculation of a design: flight condition, geometry, lift, drag, stall, moment, control and loads, performance."""
    r = case.full()
    m, p, air, w = r["metrics"], case.p, case.air, case.wing
    sf = case.sf
    out: list[Calc] = []
    q = m["q_design_Pa"]
    W = case.weight
    # ---- 1 flight condition
    c = Calc("Flight condition and weight", "Standard atmosphere (ISA) with the temperature offset, the weight the wing carries and the dynamic pressure it flies at.")
    c.add("Mass", "input", "", f"{n(p['mass_kg'])} kg")
    c.add("Weight", "W = m g", f"{n(p['mass_kg'])} × {G0:.5f}", f"{n(W)} N")
    c.add("Load factor", "input (1 = steady level flight; 2 = a 60° turn)", "", n(case.n, 3))
    c.add("Lift the wing must carry", "L = n W", f"{n(case.n, 3)} × {n(W)}", f"{n(case.n * W)} N")
    c.add("Air at the altitude", "ISA troposphere, T = T₀ − 0.0065 h + ΔT", f"h = {n(p['altitude_m'])} m, ΔT = {n(p['dT_K'])} K", f"T = {n(air.temperature_K)} K, p = {n(air.pressure_Pa)} Pa")
    c.add("Air density", "ρ = p / (R T),  R = 287.053 J/(kg K)", f"{n(air.pressure_Pa)} / (287.053 × {n(air.temperature_K)})", f"{n(air.density)} kg/m³")
    c.add("Air viscosity", "Sutherland: μ = 1.458·10⁻⁶ T^1.5 / (T + 110.4)", f"T = {n(air.temperature_K)} K", f"{n(air.viscosity)} Pa·s")
    if p["design_cl"] is not None:
        c.add("Design speed", "from the design lift coefficient: V = √(2 n W / (ρ S CL))", f"CL = {n(p['design_cl'])}", f"{n(case.V)} m/s")
    else:
        c.add("Design speed", "input", "", f"{n(case.V)} m/s")
    c.add("Dynamic pressure", "q = ½ ρ V²", f"0.5 × {n(air.density)} × {n(case.V)}²", f"{n(q)} Pa")
    c.add("Reynolds number at the mean chord", "Re = ρ V c̄ / μ", f"{n(air.density)} × {n(case.V)} × {n(w.mac)} / {n(air.viscosity)}", n(m["Re_mac_design"], 4))
    out.append(c)
    # ---- 2 geometry
    s_area, how = wing_area(p)
    c = Calc("Wing geometry", "Up to three trapezoidal panels (or an elliptical chord); the chord at every station is a share of the root chord.")
    c.add("Span", "input", "", f"{n(w.span)} m")
    area_formula = {"area": ("input", ""), "wing loading": ("S = m g / (W/S), the wing loading is set", f"{n(W)} / {n(p['wing_loading_Nm2'])}"),
                    "span and aspect ratio": ("S = b² / AR", f"{n(w.span)}² / {n(p['aspect_ratio'])}")}[how]
    c.add("Wing area", area_formula[0], area_formula[1], f"{n(w.area)} m²")
    c.add("Aspect ratio", "AR = b² / S", f"{n(w.span)}² / {n(w.area)}", n(w.aspect_ratio))
    c.add("Wing loading", "W / S", f"{n(W)} / {n(w.area)}", f"{n(W / w.area)} N/m²")
    c.add("Root chord", "from S, b and the chord ratios: c_r = S / (b ∫ (c/c_r) dη)", "", f"{n(w.root_chord * 1000)} mm")
    c.add("Tip chord", "c_t = c_r × (tip chord ratio)", f"{n(w.root_chord * 1000)} × {n(float(w.ratio[-1]))}", f"{n(w.tip_chord * 1000)} mm")
    c.add("Mean aerodynamic chord", "c̄ = (1/S) ∫ c² dy  (exact for straight panels)", "", f"{n(w.mac * 1000)} mm at y = {n(w.y_mac)} m")
    c.add("Mean sweep of the quarter-chord line", "area-weighted over the panels", "", f"{n(w.mean_sweep_deg)}°")
    c.add("Incidence of the root chord on the fuselage axis", "input", "", f"{n(p['incidence_deg'])}°")
    out.append(c)
    # ---- 3 lift
    pt = r["design_point"] if r.get("design_ok") else None
    c = Calc("Lift at the design point", "The lifting line (Weissinger vortex lattice, solved non-linearly with the NeuralFoil section polars) finds the angle at which the wing carries the lift.")
    c.add("Lift coefficient required", "CL = n W / (q S)", f"{n(case.n, 3)} × {n(W)} / ({n(q)} × {n(w.area)})", n(m["CL_design"]))
    c.add("Lift slope of the wing (linear lifting line)", "CLα, from the strips' lift slopes and the planform", "", f"{n(m['CL_alpha_per_rad'])} per rad = {n(m['CL_alpha_per_deg'])} per °")
    c.add("Zero-lift angle of the wing", "α₀ = −CL₀ / CLα", f"−{n(m['CL0'])} / {n(m['CL_alpha_per_deg'])}", f"{n(m['alpha_zero_lift_deg'])}°")
    if pt is not None:
        a_lin = m["alpha_zero_lift_deg"] + m["CL_design"] / m["CL_alpha_per_deg"]
        c.add("Root angle of attack, linear estimate", "α = α₀ + CL / CLα", f"{n(m['alpha_zero_lift_deg'])} + {n(m['CL_design'])} / {n(m['CL_alpha_per_deg'])}", f"{n(a_lin)}°")
        c.add("Root angle of attack, non-linear solution", "the angle at which the strip polars carry CL", "", f"{n(m['alpha_design_deg'])}°")
        c.add("Fuselage angle of attack", "α_fus = α − incidence", f"{n(m['alpha_design_deg'])} − {n(p['incidence_deg'])}", f"{n(m['alpha_fus_design_deg'])}°")
    c.add("Inviscid span efficiency of the planform", "Oswald-type factor from the lifting line, times the fuselage factor 1 − 2 (d/b)²", "", n(m["e_inviscid"]))
    out.append(c)
    # ---- 4 drag
    if pt is not None:
        c = Calc("Drag at the design point", "Profile drag strip by strip from the airfoil polars (with the calibration factor on the polars and the excrescence allowance), induced drag from the lifting line, fuselage drag from skin friction, form and cross-flow.")
        c.add("Profile drag", "CDp = (1 + excrescence) Σ cd_strip c_strip Δy / S", f"cd factor {n(p['cd_factor'])}, excrescence {n(p['excrescence'])}", n(m["CDp"]))
        c.add("Induced drag", "CDi = CL² / (π AR e)", f"{n(m['CL_design'])}² / (π × {n(w.aspect_ratio)} × {n(m['e_design'])})", n(m["CDi"]))
        c.add("Fuselage drag", "CDf = (Cf FF S_wet + cross-flow) × interference × (1 + excrescence) / S", "switch " + ("on" if p["fuse_drag_on"] and case.fuse is not None else "off"), n(m["CDf"]))
        c.add("Total drag coefficient", "CD = CDp + CDi + CDf", f"{n(m['CDp'])} + {n(m['CDi'])} + {n(m['CDf'])}", n(m["CD"]))
        c.add("Lift-to-drag", "L/D = CL / CD", f"{n(m['CL_design'])} / {n(m['CD'])}", n(m["LD"]))
        c.add("Drag force", "D = CD q S", f"{n(m['CD'])} × {n(q)} × {n(w.area)}", f"{n(m['drag_N'])} N")
        c.add("Aerodynamic power", "P = D V", f"{n(m['drag_N'])} × {n(case.V)}", f"{n(m['P_req_W'])} W")
        c.add("Endurance factor", "CL^1.5 / CD", f"{n(m['CL_design'])}^1.5 / {n(m['CD'])}", n(m["endurance_factor_design"]))
        out.append(c)
    # ---- 5 stall
    c = Calc("Stall and speed margin", "The wing stalls where the first strip reaches its section maximum lift (the section polar, times the calibration factor, capped at 2π lift slope).")
    if "CL_max_first" in m:
        c.add("First strip to stall", "the strip where cl reaches cl_max first", "", f"η = {n(m['eta_stall_first'])} of the semi-span, root angle {n(m['alpha_stall_first_deg'])}°")
        c.add("Maximum lift coefficient (first strip stalls)", "CLmax", "", n(m["CL_max_first"]))
        c.add("Peak of the non-linear lift curve", "CLmax,nl", "", f"{n(m['CL_max_nl'])} at {n(m['alpha_stall_nl_deg'])}°")
        c.add("Usable CLmax with the safety factor", "CLmax / SF_lift", f"{n(m['CL_max_first'])} / {n(sf['lift'], 3)}", n(m["CL_max_first"] / sf["lift"]))
        c.add("Stall speed", "V_s = √(2 n W / (ρ S CLmax)), at the Reynolds number of the stall speed", f"√(2 × {n(case.n, 3)} × {n(W)} / ({n(air.density)} × {n(w.area)} × {n(m['CL_max_first'])}))", f"{n(m['V_stall_mps'])} m/s")
        c.add("Stall speed with the safety factor", "V_s √SF_lift", f"{n(m['V_stall_mps'])} × √{n(sf['lift'], 3)}", f"{n(m['V_stall_mps'] * math.sqrt(sf['lift']))} m/s")
        c.add("Speed margin", "V / (V_s √SF_lift)", f"{n(case.V)} / {n(m['V_stall_mps'] * math.sqrt(sf['lift']))}", n(m["V_design_mps"] / (m["V_stall_mps"] * math.sqrt(sf["lift"]))))
        if pt is not None:
            c.add("Highest strip utilisation at the design point", "SF_lift × cl / cl_max over the strips", f"at η = {n(m['util_eta'])}", n(m["util_max"] * sf["lift"]))
    out.append(c)
    # ---- 6 moment
    if pt is not None:
        c = Calc("Pitching moment of the wing alone", "About the moment reference point (a share of the mean chord) and about the aerodynamic centre; the tail is not modelled.")
        c.add("Aerodynamic centre", "x_ac, from the lifting line", "", f"{n(m['x_ac_mac'])} of the MAC from its leading edge")
        c.add("Moment reference point", "input (the aircraft's centre of gravity)", "", f"{n(p['x_cg_mac'])} of the MAC")
        c.add("Pitching moment about the reference point", "Cm = Σ strip lift × arm + Σ section moments", "", n(m["Cm_ref"]))
        c.add("Pitching moment about the aerodynamic centre", "Cm_ac", "", n(m["Cm_ac"]))
        c.add("Slope of Cm against CL (reference point)", "dCm/dCL (positive: nose-up with more lift)", "", n(m["dCm_dCL_ref"]))
        c.add("Break at the stall", "Cm at the lift peak minus the linear trend", "", f"{n(m['pitch_break'])} ({m['pitch_tendency']})")
        c.add("CL where the wing alone has zero moment", "below the stall", "", n(m["trim_CL_wing_alone"]))
        out.append(c)
    # ---- 7 control and loads
    if pt is not None:
        c = Calc("Roll control, loads and lateral derivatives")
        if p["aileron_on"] and "Cl_da_linear_per_rad" in m:
            c.add("Aileron roll power, linear", "Cl_δa from ±3° finite differences", "", f"{n(m['Cl_da_linear_per_rad'])} per rad")
            c.add("Roll power at the design point", "same, non-linear", "", f"{n(m['Cl_da_design_per_rad'])} per rad ({pct(m['roll_authority_design'])} of linear)")
            c.add("Roll power at 90 % of CLmax", "same, near the stall", "", f"{n(m['Cl_da_at_0p9_per_rad'])} per rad ({pct(m['roll_authority_0p9'])} of linear)")
            c.add("Roll moment at full deflection", f"deflection {n(p['aileron_defl_deg'])}° each side", "", f"design point {n(m['roll_max_design'])}, at the stall {n(m['roll_max_at_stall'])}")
            c.add("CL where half the roll power is lost", "from the design point upwards", "", n(m["control_loss_CL"]))
        else:
            c.add("Ailerons", "switched off", "", "no roll control studied")
        c.add("Lift centroid", "η of the centre of the lift distribution", "", n(m["lift_centroid_eta"]))
        c.add("Root bending moment coefficient", "M_root / (q S b/2)", "", n(m["root_bending_coeff"]))
        c.add("Roll damping", "Cl_p", "", n(m["Cl_p"]))
        c.add("Dihedral effect", "Cl_β from the dihedral + from the sweep", f"{n(m['Cl_beta_dihedral'])} + {n(m['Cl_beta_sweep'])}", f"{n(m['Cl_beta_total'])} per rad")
        out.append(c)
    # ---- 8 performance
    if "LD_max" in m:
        c = Calc("Performance over speed", "The wing flown at every speed between just above the stall and 2.8 times it, at the lift it must carry (L = n W).")
        c.add("Best lift-to-drag", "max (CL / CD) over speed", "", f"{n(m['LD_max'])} at CL = {n(m['CL_at_LD_max'])}, V = {n(m['V_best_glide'])} m/s")
        c.add("Least aerodynamic power", "min (D V) over speed", "", f"{n(m['P_min_W'])} W at V = {n(m['V_min_power'])} m/s")
        c.add("Best endurance factor", "max (CL^1.5 / CD)", "", n(m["endurance_factor"]))
        out.append(c)
    return out


def safety_rows(case) -> list[tuple]:
    """(quantity, as calculated, with the safety factor, rule) for every result the safety factor changes."""
    m = case.full()["metrics"]
    s = with_safety(m, case.sf)
    f = case.sf
    rules = [("CD", "Drag coefficient", f"× {f['drag']:.2f}"), ("LD", "Lift-to-drag", f"÷ {f['drag']:.2f}"), ("P_req_W", "Aerodynamic power, W", f"× {f['drag']:.2f}"),
             ("CL_max_first", "Usable maximum lift", f"÷ {f['lift']:.2f}"), ("V_stall_mps", "Stall speed, m/s", f"× √{f['lift']:.2f}"),
             ("speed_margin", "Speed margin", f"V / (V_s √{f['lift']:.2f})"), ("util_max", "Strip utilisation", f"× {f['lift']:.2f}"), ("Cm_ref", "Pitching moment Cm", f"× {f['moment']:.2f}"),
             ("root_bending_coeff", "Root bending coefficient", f"× {f['moment']:.2f}"), ("roll_authority_0p9", "Roll power near the stall (share of linear)", f"÷ {f['control']:.2f}")]
    rows = []
    for k, lab, rule in rules:
        if k in m and k + "_sf" in s:
            rows.append((lab, n(m[k]), n(s[k + "_sf"]), rule))
    return rows


def limit_rows(case, points=None) -> list[tuple]:
    """Every limit of the optimiser at every flight point: (point, limit, value, limit value, met)."""
    from .optimise import metrics_sf
    from .points import cases_for, margin_limit, point_list
    pts = point_list(points)
    rows = []
    for i, (pt, cc) in enumerate(zip(pts, cases_for(case, points))):
        need = {"roll"} if (case.p["con_roll_on"] and case.p["aileron_on"] and i == 0) else set()
        mm = metrics_sf(cc, need)
        p = case.p
        name = pt.name
        if not mm.get("ok"):
            rows.append((name, "design point", "no solution: " + str(mm.get("reason", "")), "", "no"))
            continue

        def row(lim, val, bound, ok, fmt=n):
            rows.append((name, lim, fmt(val), bound, "yes" if ok else "NO"))
        if p["con_stall_on"]:
            row("stall starts at η ≤", mm["eta_stall_first"], f"{p['stall_onset_max']:.2f}", mm["eta_stall_first"] <= p["stall_onset_max"] + 1e-9)
        if p["con_margin_on"]:
            lim = margin_limit(pt, p) if i > 0 else float(p["speed_margin_min"])
            row("speed margin (SF) ≥", mm["speed_margin_sf"], f"{lim:.2f}", mm["speed_margin_sf"] >= lim - 1e-9)
        if p["con_util_on"]:
            row("strip utilisation (SF) ≤", mm["util_max_sf"], "1.00", mm["util_max_sf"] <= 1.0 + 1e-9)
        if i == 0:
            if p["con_cm_on"]:
                row("|Cm| (SF) ≤", abs(mm["Cm_ref_sf"]), f"{p['cm_abs_max']:.3f}", abs(mm["Cm_ref_sf"]) <= p["cm_abs_max"] + 1e-9)
            if p["con_roll_on"] and p["aileron_on"] and "roll_authority_0p9_sf" in mm:
                row("roll power near the stall (SF) ≥", mm["roll_authority_0p9_sf"], f"{p['roll_authority_min']:.2f}", mm["roll_authority_0p9_sf"] >= p["roll_authority_min"] - 1e-9)
            if p["con_loading_on"]:
                row("wing loading ≤ (N/m²)", mm["wing_loading_Nm2"], f"{p['wing_loading_max']:.0f}", mm["wing_loading_Nm2"] <= p["wing_loading_max"] + 1e-9)
    return rows


def point_rows(case, points=None) -> list[tuple]:
    """Key results of the design at every flight point."""
    from .optimise import metrics_sf
    from .points import cases_for, point_list
    rows = []
    for pt, cc in zip(point_list(points), cases_for(case, points)):
        m = metrics_sf(cc, set())
        if not m.get("ok"):
            rows.append((pt.label(), n(cc.V), n(cc.n, 3), "no design solution", "", "", "", ""))
            continue
        rows.append((pt.label(), n(cc.V), n(cc.n, 3), n(m["CL_design"]), n(m["LD_sf"]), n(m["V_stall_mps_sf"]), n(m["speed_margin_sf"]), n(m["util_max_sf"])))
    return rows


def station_rows(case) -> list[tuple]:
    """The wing at its stations: η, y, chord, sweep of the next panel, dihedral, twist, airfoil."""
    p, w = case.p, case.wing
    s = broadcast_stations(p)
    rows = []
    foil_at = {round(float(e), 4): af for e, af in zip(case.foil_eta, case.foils)}
    for k, e in enumerate(w.eta):
        sweep = w.sweep_deg[min(k, len(w.sweep_deg) - 1)]
        dih = w.dihedral_deg[min(k, len(w.dihedral_deg) - 1)]
        af = foil_at.get(round(float(e), 4))
        rows.append((n(float(e), 3), n(float(e) * 0.5 * w.span, 3), n(float(w.chord(e)) * 1000, 4), n(float(w.ratio[k]), 3), n(float(sweep), 3), n(float(dih), 3), n(float(w.twist_deg[k]), 3), af.label if af else "(blended)"))
    return rows


def airfoil_table(case) -> list[tuple]:
    rows = []
    for st, af, eta in RF._station_foils(case):
        re = float(case.air.density * case.V * case.wing.chord(eta) / case.air.viscosity)
        pol = af.polar(re)
        rows.append((st, n(eta, 3), af.label, f"{re / 1e3:,.0f} k", f"{af.thickness * 100:.1f}", f"{af.shape['camber_max'] * 100:.2f}", n(pol["cl_max"], 3), n(pol["alpha_stall_deg"], 3),
                     n(pol["cd_min"], 3), n(pol["ld_max"], 3), n(pol["alpha0_deg"], 3), n(pol["cm0"], 3), n(pol["confidence_min"], 2)))
    return rows


def input_rows(case) -> list[tuple]:
    """Every input in use, by group, with its value in this design."""
    p = case.p
    rows = []
    for g in GROUPS:
        for q in PARAMS:
            if q.group != g or not active(q, p):
                continue
            v = p.get(q.key)
            if v is None:
                continue
            if q.kind == "airfoil":
                v = AF.display_name(v) if v != AF.PARAM else "parametric NACA 4-digit"
            rows.append((g, q.label.strip(), n(v) if not isinstance(v, str) else v, "" if q.unit == "-" else q.unit))
    return rows


def mission_tables(mp: list) -> tuple[list, list]:
    """The lift against the weight at every flight point, as two tables of rows: what the wing generates, and the most it could generate."""
    a, b = [], []
    for d in mp:
        a.append((d["label"], n(d["V"]), n(d["mass"]), n(d["W"]), n(d["n"], 3), n(d["L"]), n(d["L"] / d["W"], 3), n(d["CL_req"])))
        if "CLmax" in d:
            b.append((d["name"], n(d["CLmax"]), n(d["CLmax_sf"]), n(d["Lmax"]), n(d["Lmax_sf"]), n(d["nmax"], 3), n(d["nmax_sf"], 3), f"{100 * d['use']:.0f} %"))
        else:
            b.append((d["name"], "no stall found", "", "", "", "", "", ""))
    return a, b


def mission_sentences(mp: list) -> list[str]:
    """One plain sentence per flight point: the lift generated against the weight, and how much more the wing could generate."""
    out = []
    for d in mp:
        t = (f"<b>{_esc(d['label'])}:</b> at {n(d['V'], 3)} m/s the wing generates <b>{n(d['L'], 3)} N, {n(d['L'] / d['W'], 3)} times its weight</b> of {n(d['W'], 3)} N (it needs CL = {n(d['CL_req'], 3)}).")
        if "CLmax" in d:
            t += (f" It could generate up to {n(d['Lmax'], 3)} N ({n(d['nmax'], 3)} × weight) as calculated, and {n(d['Lmax_sf'], 3)} N ({n(d['nmax_sf'], 3)} × weight) with the safety factor: "
                  f"it uses <b>{100 * d['use']:.0f} %</b> of the usable lift, with a stall speed of {n(d['V_stall_sf'], 3)} m/s (with the safety factor).")
        out.append(t)
    return out


def strip_rows(case) -> tuple[list, list]:
    """The strips of the right half-wing at the design point (the working of the lifting line) and the sums that must reproduce the lift needed."""
    sd = MS.strip_data(case)
    if sd is None:
        return [], []
    sf = case.sf["lift"]
    rows = []
    for j in range(len(sd["eta"])):
        rows.append((str(j + 1), n(float(sd["eta"][j]), 3), n(float(sd["y"][j]), 3), n(float(sd["c"][j]) * 1000, 4), f"{float(sd['re'][j]) / 1e3:,.0f} k", n(float(sd["alpha_geo"][j]), 3), n(float(sd["alpha_ind"][j]), 3),
                     n(float(sd["alpha_eff"][j]), 3), n(float(sd["cl"][j]), 3), n(float(sd["cap"][j]), 3), n(float(sd["cl_unc"][j] / sd["cap"][j] * sf), 3), n(float(sd["cd"][j]), 3), n(float(sd["lift"][j]), 3)))
    m = case.full()["metrics"]
    lift = 2 * float(sd["lift"].sum())
    cl = 4 * float((sd["gamma"] * sd["dy"]).sum()) / sd["S"]
    cdp = (1 + case.p["excrescence"]) * 2 * float((sd["w"] * sd["cd"]).sum())
    sums = [("Lift of the wing = 2 Σ q c cl Δy", f"{n(lift)} N", f"n W = {n(case.n * case.weight)} N"), ("Lift coefficient = 4 Σ G Δy / S", n(cl), f"n W / (q S) = {n(m['CL_design'])}"),
            ("Profile drag coefficient = (1 + excrescence) × 2 Σ (c Δy / S) cd", n(cdp), f"CDp of the calculation sheet = {n(m['CDp'])}")]
    return rows, sums


# ------------------------------------------------------------------------------------------------ the PDF
def _styles():
    from reportlab.lib.styles import ParagraphStyle
    base = dict(fontName=FONT, textColor=INK)
    return {
        "title": ParagraphStyle("title", fontName=FONT_B, fontSize=22, leading=26, textColor=NAVY, spaceAfter=4),
        "sub": ParagraphStyle("sub", fontSize=11, leading=14, textColor=STEEL, spaceAfter=8, **{k: v for k, v in base.items() if k != "textColor"}),
        "h1": ParagraphStyle("h1", fontName=FONT_B, fontSize=14, leading=18, textColor=NAVY, spaceBefore=10, spaceAfter=6, borderPadding=(0, 0, 3, 0)),
        "h2": ParagraphStyle("h2", fontName=FONT_B, fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontSize=8.6, leading=11.8, spaceAfter=4, **base),
        "small": ParagraphStyle("small", fontSize=7.4, leading=9.6, textColor=STEEL, spaceAfter=3, fontName=FONT),
        "cell": ParagraphStyle("cell", fontSize=7.4, leading=9.4, **base),
        "cellb": ParagraphStyle("cellb", fontSize=7.4, leading=9.4, fontName=FONT_B, textColor=INK),
        "cellh": ParagraphStyle("cellh", fontSize=7.4, leading=9.4, fontName=FONT_B, textColor="#FFFFFF"),
        "mono": ParagraphStyle("mono", fontSize=7.0, leading=9.2, fontName=MONO, textColor=INK),
        "card_v": ParagraphStyle("card_v", fontSize=13, leading=16, fontName=FONT_B, textColor=NAVY),
        "card_l": ParagraphStyle("card_l", fontSize=6.8, leading=8.4, fontName=FONT, textColor=STEEL),
    }


def _esc(t) -> str:
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_report(design: tuple, start: tuple | None = None, points=None, opt=None, valley=None, title: str = "Wing design report", subtitle: str = "",
                 include_valleys_slices: bool = True) -> bytes:
    """The PDF as bytes. design: (label, Case); start: the starting design to compare with, (label, Case); points: the flight points; opt: the optimiser Result the design
    came from; valley: the dictionary the Valleys page keeps (rep, cv, label, tol)."""
    _register_fonts()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from .optimise import OBJECTIVES
    from .points import point_list

    st = _styles()
    label, case = design
    r = case.full()
    m = r["metrics"]
    cases = [design] + ([start] if start is not None else [])
    pts = point_list(points)
    W_TEXT = A4[0] - 2 * 18 * mm
    buf = io.BytesIO()

    def header_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(FONT_B, 7.5)
        canvas.setFillColor(colors.HexColor(NAVY))
        canvas.drawString(18 * mm, A4[1] - 11 * mm, "HORYZN  wing aerodynamics lab")
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(colors.HexColor(STEEL))
        canvas.drawRightString(A4[0] - 18 * mm, A4[1] - 11 * mm, f"{title}: {label}")
        canvas.setStrokeColor(colors.HexColor(ORANGE))
        canvas.setLineWidth(1.2)
        canvas.line(18 * mm, A4[1] - 13 * mm, A4[0] - 18 * mm, A4[1] - 13 * mm)
        canvas.setFont(FONT, 6.8)
        canvas.drawString(18 * mm, 9 * mm, f"Generated {datetime.date.today().isoformat()} by the HORYZN wing aerodynamics lab. Estimates, not measurements: see the notes at the end.")
        canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    def table(rows, widths, header=None, zebra=True, font="cell", align_right=(), repeat=1):
        data = []
        if header:
            data.append([Paragraph(_esc(h), st["cellh"]) for h in header])
        for row in rows:
            data.append([Paragraph(_esc(c), st[font]) for c in row])
        t = Table(data, colWidths=widths, repeatRows=repeat if header else 0)
        sty = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor(LINE)), ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
               ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if header:
            sty += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY))]
        if zebra:
            for i in range(1 if header else 0, len(data)):
                if (i - (1 if header else 0)) % 2 == 1:
                    sty.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor(MIST)))
        t.setStyle(TableStyle(sty))
        return t

    def figure(png: bytes | None, caption: str = "", width=None):
        if not png:
            return []
        w_img = width or W_TEXT
        iw, ih = ImageReader(io.BytesIO(png)).getSize()
        img = Image(io.BytesIO(png), width=w_img, height=w_img * ih / iw)
        items = [img]
        if caption:
            items.append(Paragraph(_esc(caption), st["small"]))
        return [KeepTogether(items), Spacer(1, 5)]

    def cards(items):
        cells = [[Paragraph(_esc(v), st["card_v"]), Paragraph(_esc(lab), st["card_l"])] for lab, v in items]
        k = 4
        rows = [cells[i:i + k] for i in range(0, len(cells), k)]
        data = []
        for row in rows:
            row = row + [[Paragraph("", st["card_v"]), Paragraph("", st["card_l"])]] * (k - len(row))
            data.append([[c[0], c[1]] for c in row])
        t = Table(data, colWidths=[W_TEXT / k] * k)
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor(LINE)), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(MIST)),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LINEABOVE", (0, 0), (-1, 0), 2, colors.HexColor(ORANGE))]))
        return t

    flow: list = []
    # ---------------------------------------------------------------- page 1: title and summary
    flow += [Paragraph(_esc(title), st["title"]), Paragraph(_esc(label + (f" · {subtitle}" if subtitle else "")), st["sub"])]
    flow += [Paragraph(
        f"This report describes one wing design of the HORYZN wing aerodynamics lab: the wing, its airfoils and a simple fuselage, flown at {n(case.V)} m/s at {n(case.p['altitude_m'])} m with a mass of "
        f"{n(case.p['mass_kg'])} kg. It gives the numbers, <b>how each one was calculated</b>, the drawings of the wing and its sections, and the charts. Every result is shown as calculated and with the "
        f"<b>safety factor {case.p['safety_factor']:.2f}</b> applied in its adverse direction (drag and power × SF, usable maximum lift ÷ SF, moments and loads × SF, roll power ÷ SF).", st["body"])]
    if not r.get("design_ok"):
        flow += [Paragraph(f"<b>Warning:</b> no design point could be reached: {_esc(m.get('warning', ''))}.", st["body"])]
    s = with_safety(m, case.sf)
    summary = [("Span", f"{m['span_m']:.2f} m"), ("Aspect ratio", f"{m['aspect_ratio']:.1f}"), ("Wing area", f"{m['area_m2']:.3f} m²"), ("Wing loading", f"{m['wing_loading_Nm2']:.0f} N/m²")]
    if r.get("design_ok"):
        summary += [("Lift-to-drag (with SF)", f"{n(s['LD_sf'])}  ({n(m['LD'])})"), ("Usable CLmax (with SF)", f"{n(s['CL_max_first_sf'])}  ({n(m['CL_max_first'])})"),
                    ("Stall speed (with SF)", f"{n(s['V_stall_mps_sf'])} m/s  ({n(m['V_stall_mps'])})"), ("Power (with SF)", f"{n(s['P_req_W_sf'])} W  ({n(m['P_req_W'])})"),
                    ("Design lift coefficient", n(m["CL_design"])), ("Stall starts at", f"η {n(m['eta_stall_first'])}"), ("Speed margin (with SF)", n(s["speed_margin_sf"])), ("Cm about the reference (with SF)", f"{n(s['Cm_ref_sf'])}")]
    flow += [Spacer(1, 4), cards(summary), Spacer(1, 6)]
    # what changed (optimiser) or the inputs in short
    if opt is not None and start is not None:
        flow += [Paragraph("What the optimiser changed", st["h2"])]
        rows = []
        for var in opt.variables:
            pr = BY_KEY[var.key]
            now = opt.changed[var.key]
            if var.kind == "float":
                was = opt.problem.p0[var.key] if opt.problem is not None else None
                rows.append((pr.label.strip(), "" if pr.unit == "-" else pr.unit, n(was), n(now), f"{n(var.lo)} to {n(var.hi)}"))
            else:
                was = opt.problem.p0[var.key] if opt.problem is not None else None
                nm = lambda x: AF.display_name(x) if (var.kind == "airfoil" and x != AF.PARAM) else n(x)  # noqa: E731
                rows.append((pr.label.strip(), "", nm(was), nm(now), f"{len(var.pool)} options"))
        flow += [table(rows, [W_TEXT * 0.44, W_TEXT * 0.08, W_TEXT * 0.14, W_TEXT * 0.14, W_TEXT * 0.20], ["Input searched", "Unit", "Starting point", "Optimised", "Searched between"]), Spacer(1, 4)]
    # ---------------------------------------------------------------- lift against weight
    mp = MS.mission_lift(case, points)
    flow += [Paragraph("1  Lift against weight along the mission", st["h1"])]
    d0 = mp[0]
    if "CLmax" in d0:
        flow += [Paragraph(
            "In steady flight the wing generates exactly the lift that balances n times its weight, whatever the speed; what changes along the mission is how much more it <i>could</i> generate before it stalls. "
            "The most lift is q S CLmax (q is the dynamic pressure, S the wing area, CLmax the maximum lift coefficient, where the first strip stalls); with the safety factor it is divided by "
            f"{d0['SFL']:.2f}. Lift divided by weight is the load factor the wing could carry at that speed.", st["body"]),
            cards([("Lift at the design point", f"{n(d0['L'], 3)} N = {n(d0['L'] / d0['W'], 3)} × weight"), ("Most lift there (with SF)", f"{n(d0['Lmax_sf'], 3)} N = {n(d0['nmax_sf'], 3)} × weight"),
                   ("Lift in use (of the usable)", f"{100 * d0['use']:.0f} %"), ("Stall speed (with SF)", f"{n(d0['V_stall_sf'], 3)} m/s")]), Spacer(1, 6)]
    ta, tb = mission_tables(mp)
    flow += [Paragraph("What the wing generates at every flight point", st["h2"]),
             table(ta, [W_TEXT * x for x in (0.31, 0.09, 0.08, 0.09, 0.06, 0.13, 0.12, 0.12)], ["Flight point", "speed (m/s)", "mass (kg)", "weight W = m g (N)", "n", "lift L = n W (N)", "lift ÷ weight", "CL needed"]), Spacer(1, 4)]
    if "CLmax" in d0:
        flow += [KeepTogether([Paragraph("The most it could generate", st["h2"]),
                               table(tb, [W_TEXT * x for x in (0.20, 0.09, 0.11, 0.12, 0.13, 0.11, 0.12, 0.12)],
                                     ["Flight point", "CLmax", "usable CLmax (÷ SF)", "most lift q S CLmax (N)", "… with the SF (N)", "most lift ÷ weight", "… with the SF", "share of the usable lift in use"])]), Spacer(1, 4)]
    flow += [Paragraph(t, st["body"]) for t in mission_sentences(mp)]
    if "CLmax" in d0:
        flow += figure(RF.lift_vs_weight(case, mp), "Left: the most lift the wing can carry over speed, as a multiple of the design weight (solid: as calculated; dashed: with the safety factor; the line at 1 is lift = weight), "
                       "with every flight point of the mission marked. Right: at each point, the lift needed (n × weight) against the most it could generate with the safety factor and the weight.")
    # ---------------------------------------------------------------- drawings
    flow += [CondPageBreak(90 * mm), Paragraph("2  The wing and its airfoils", st["h1"])]
    flow += figure(RF.top_view(case), f"Top view of the design. Span {n(case.wing.span)} m, area {n(case.wing.area)} m², aspect ratio {n(case.wing.aspect_ratio, 3)}, mean aerodynamic chord {n(case.wing.mac * 1000)} mm, "
                   f"mean sweep {n(case.wing.mean_sweep_deg)}°. Orange bar: mean aerodynamic chord; diamond: aerodynamic centre of the wing alone; cross: moment reference point.")
    flow += [Paragraph("Wing at its stations", st["h2"]),
             table(station_rows(case), [W_TEXT * x for x in (0.08, 0.10, 0.12, 0.11, 0.11, 0.11, 0.10, 0.27)], ["η", "y (m)", "chord (mm)", "chord / root", "sweep of the panel (°)", "dihedral (°)", "twist (° nose down)", "airfoil"]),
             Spacer(1, 4)]
    flow += figure(RF.airfoils(case), "Section at every airfoil station, drawn on one chord, with the thickness and camber distributions. Between stations the section is blended linearly.")
    flow += [Paragraph("Section data at the design Reynolds number (NeuralFoil)", st["h2"]),
             table(airfoil_table(case), [W_TEXT * x for x in (0.10, 0.06, 0.15, 0.08, 0.06, 0.07, 0.07, 0.08, 0.06, 0.07, 0.07, 0.06, 0.07)],
                   ["Station", "η", "Airfoil", "Re", "t/c %", "camber %", "cl max", "stall °", "cd min", "(cl/cd) max", "α₀ °", "cm₀", "confidence"], font="cell"), Spacer(1, 4)]
    # ---------------------------------------------------------------- calculations
    flow += [PageBreak(), Paragraph("3  How the numbers were calculated", st["h1"]),
             Paragraph("Each step gives the quantity, the formula, the numbers put into it and the result. The numbers are those of the lab's model: a non-linear lifting line over the strips of the wing, "
                       "the airfoil polars of NeuralFoil, and a safety factor applied last. Where a number comes out of the lifting line (a solve, not a closed formula) the formula column says what it is.", st["body"])]
    widths = [W_TEXT * x for x in (0.25, 0.30, 0.27, 0.18)]
    sections = calc_sections(case, points)
    for k, sec in enumerate(sections, 1):
        block = [Paragraph(f"3.{k}  {sec.title}", st["h2"])]
        if sec.note:
            block.append(Paragraph(_esc(sec.note), st["small"]))
        flow += block + [table(sec.rows, widths, ["Quantity", "Formula", "With the numbers", "Result"]), Spacer(1, 5)]
    srows, ssums = strip_rows(case)
    if srows:
        flow += [Paragraph(f"3.{len(sections) + 1}  Inside the lifting line, strip by strip (at the design point)", st["h2"]), Paragraph(
            "The wing is cut into 32 strips across the span (16 on each half; the table gives the right half, root to tip). Each strip has a chord c, a width Δy and a Reynolds number Re = ρ V c / μ, and the "
            "airfoil polar of the strip is taken at that Re. The angle a strip is set at is α_geo (root angle + twist); the angle it actually sees is lower, α_eff = α_geo − α_ind, because the downwash of "
            "the wing's own vortices comes on top. The strip carries cl(α_eff) from its polar, and its circulation is G = ½ c cl. The vortex lattice gives the downwash every strip's circulation puts on every "
            "other strip, so the 32 values of G are the solution of 32 coupled equations, found by Newton's method at the root angle for which the lift is exactly n W. "
            "Then: strip lift = q c cl Δy;  CL = 2 Σ G Δy / S;  CDp = (1 + excrescence) Σ (c Δy / S) cd;  CDi = −Σ G (T G) Δy / S. A strip is at its stall when cl reaches its cap "
            "(utilisation 1); with the safety factor the utilisation is multiplied by SF_lift.", st["small"]),
            table(srows, [W_TEXT * x for x in (0.045, 0.08, 0.075, 0.08, 0.07, 0.075, 0.075, 0.075, 0.065, 0.07, 0.095, 0.09, 0.085)],
                  ["#", "η", "y (m)", "chord (mm)", "Re", "α_geo (°)", "α_ind (°)", "α_eff (°)", "cl", "cl cap", "utilisation × SF", "cd", "strip lift (N)"], font="cell"),
            Spacer(1, 4), Paragraph("What the strips add up to", st["small"]),
            table(ssums, [W_TEXT * x for x in (0.46, 0.22, 0.32)], ["Sum over the strips", "gives", "which must equal"]), Spacer(1, 5)]
    flow += [Paragraph(f"3.{len(sections) + 2}  The safety factor", st["h2"]), Paragraph(
        f"The safety factor ({case.p['safety_factor']:.2f}, from 1 to 2) is applied last, in the adverse direction of every result: drag and power up, lift capability down, moments and loads up, control authority down. "
        f"Families switched off in the inputs use a factor of 1.", st["small"]),
        table(safety_rows(case), [W_TEXT * x for x in (0.40, 0.18, 0.18, 0.24)], ["Result", "As calculated", "With the safety factor", "Rule"]), Spacer(1, 5)]
    # ---------------------------------------------------------------- flight points and limits
    flow += [Paragraph(f"3.{len(sections) + 3}  Flight points and limits", st["h2"])]
    flow += [Paragraph("The same wing in each flight condition (speed, air, mass and load factor change, the wing does not). Every result carries the safety factor.", st["small"]),
             table(point_rows(case, points), [W_TEXT * x for x in (0.30, 0.09, 0.06, 0.09, 0.10, 0.12, 0.12, 0.12)], ["Flight point", "speed m/s", "n", "CL", "L/D (SF)", "stall speed (SF)", "speed margin (SF)", "strip utilisation (SF)"]), Spacer(1, 5)]
    lrows = limit_rows(case, points)
    if lrows:
        flow += [Paragraph("Limits the optimiser works to, checked on this design", st["small"]),
                 table(lrows, [W_TEXT * x for x in (0.26, 0.34, 0.14, 0.14, 0.12)], ["Flight point", "Limit", "Value", "Limit value", "Met"]), Spacer(1, 5)]
    # ---------------------------------------------------------------- charts
    flow += [PageBreak(), Paragraph("4  Charts", st["h1"])]
    flow += figure(RF.lift_polar(cases), "Wing lift curve, drag polar and lift-to-drag. The design is solid blue" + (", the starting design orange." if start else "."))
    flow += figure(RF.drag_breakdown(case), "Where the drag comes from at each lift coefficient, and its shares at the design point.")
    flow += figure(RF.moment(cases), "Pitching moment of the wing alone about the reference point and about the aerodynamic centre. A rising Cm against CL is destabilising about that point; the tail is not modelled.")
    flow += figure(RF.span_state(case), "Stall map: the lift each strip carries against the lift it can carry, from CL = 0.4 CLmax to the stall, with the state of the strips at the design point.")
    flow += figure(RF.utilisation(case), "Strip utilisation over the lift range: red is beyond the lift the strip is allowed with the safety factor.")
    ctl = RF.control(cases)
    if ctl:
        flow += figure(ctl, "Roll power of the ailerons against lift: it falls as the outer strips approach their maximum lift.")
    flap = RF.flap_effect(case)
    if flap:
        flow += figure(flap, "Effect of the flap on the lift curve.")
    flow += figure(RF.performance(cases), "Power required and lift-to-drag against speed.")
    flow += figure(RF.airfoil_sections(case), "Section data of every airfoil station at its Reynolds number at the design speed.")
    # ---------------------------------------------------------------- optimiser
    if opt is not None:
        flow += [PageBreak(), Paragraph("5  How the optimum was found", st["h1"])]
        wt = ", ".join(f"{OBJECTIVES[k][0].lower()} (weight {w:g})" for k, w in opt.weights.items())
        flow += [Paragraph(f"The optimiser (differential evolution) searched {len(opt.variables)} inputs. Objective: {_esc(wt)}; each normalised by its value for the starting design and combined with the "
                           f"flight-point weights. It evaluated {opt.n_evals:,} candidate designs in {opt.seconds:.0f} s and "
                           f"{'found a design that meets every limit' if opt.feasible else '<b>did not find a design that meets every limit</b>'}. All limits are tested with the safety factor.", st["body"])]
        if len(opt.points) > 1:
            prow = []
            for i, ptn in enumerate(opt.points):
                m0 = opt.baseline_points[i]
                m1 = opt.best_points[i] if i < len(opt.best_points) else {}
                prow.append((ptn.label(), n(opt.problem.point_weights[i], 3) if opt.problem is not None else "", f"{n(m0.get('LD_sf'))} → {n(m1.get('LD_sf'))}", f"{n(m0.get('V_stall_mps_sf'))} → {n(m1.get('V_stall_mps_sf'))}",
                             f"{n(m0.get('speed_margin_sf'))} → {n(m1.get('speed_margin_sf'))}", f"{n(m0.get('util_max_sf'))} → {n(m1.get('util_max_sf'))}"))
            flow += [Paragraph("Every flight point: starting point → optimised", st["h2"]),
                     table(prow, [W_TEXT * x for x in (0.31, 0.08, 0.15, 0.16, 0.15, 0.15)], ["Flight point", "weight", "L/D (SF)", "stall speed (SF)", "speed margin (SF)", "strip utilisation (SF)"])]
        else:
            b0, b1 = opt.baseline, opt.best
            comp = [(lab, n(b0.get(k)), n(b1.get(k))) for k, lab in (("LD_sf", "Lift-to-drag (SF)"), ("P_req_W_sf", "Power, W (SF)"), ("V_stall_mps_sf", "Stall speed, m/s (SF)"), ("CL_max_first_sf", "Usable CLmax (SF)"),
                                                                 ("eta_stall_first", "Where the stall starts (η)"), ("speed_margin_sf", "Speed margin (SF)"), ("util_max_sf", "Strip utilisation (SF)")) if k in b0 and k in b1]
            flow += [Paragraph("Starting point → optimised", st["h2"]), table(comp, [W_TEXT * x for x in (0.5, 0.25, 0.25)], ["Result", "Starting point", "Optimised"])]
        flow += [Spacer(1, 5)] + figure(RF.history(opt.history), "Best weighted objective of each generation (lower is better).")
    # ---------------------------------------------------------------- valleys
    if valley is not None:
        rep = valley["rep"]
        sec = "6" if opt is not None else "5"
        flow += [PageBreak(), Paragraph(f"{sec}  What can move: the valleys around the design", st["h1"]),
                 Paragraph(f"Around the design the weighted objective forms a valley. The <b>free range</b> of an input is how far it can move, with every other input held, before the objective loses "
                           f"{rep.tol_pct:g} % of its starting value, a limit is broken, or the range the input allows ends. Centre: {_esc(valley.get('label', ''))}.", st["body"])]
        for msg in rep.warnings:
            flow += [Paragraph(_esc(msg), st["small"])]
        rows = []
        for v in sorted((x for x in rep.valleys if x.kind == "float"), key=lambda x: x.share):
            rows.append((v.label, v.unit, n(v.x_star), f"{n(v.free_lo)} to {n(v.free_hi)}", f"{100 * v.share:.0f} %", v.verdict, v.bind_lo, v.bind_hi))
        flow += [table(rows, [W_TEXT * x for x in (0.23, 0.05, 0.09, 0.15, 0.07, 0.08, 0.17, 0.16)], ["Input", "unit", "at the design", "free from … to", "of its range", "verdict", "lower end limited by", "upper end limited by"], font="cell"), Spacer(1, 5)]
        orows = []
        for v in rep.valleys:
            if v.kind != "float":
                for o in v.options:
                    orows.append((v.label, AF.display_name(o["value"]) if v.kind == "airfoil" and o["value"] != AF.PARAM else n(o["value"]) + ("  (the design)" if o["current"] else ""),
                                  f"{o['loss']:+.2f} %" if o["loss"] == o["loss"] else "no design solution", "yes" if o["ok"] else "no", o["why"]))
        if orows:
            flow += [Paragraph("Airfoils and choices: the alternatives", st["h2"]), table(orows, [W_TEXT * x for x in (0.26, 0.20, 0.14, 0.10, 0.30)], ["Input", "option", "loss of the objective", "acceptable", "why not"]), Spacer(1, 5)]
        flow += figure(RF.valley_bars(rep), "Free range of every numeric input over the range it allows; the diamond is the design. Green: free (60 % of its range or more); light green: loose; amber: tight; red: critical (under 8 %).")
        if include_valleys_slices:
            flow += figure(RF.valley_slices(rep), "The valley of every input: loss of the objective against the input, the others held at the design.")
        cv = valley.get("cv")
        if cv is not None:
            head = [Paragraph("Combinations of inputs that can move together", st["h2"]),
                    Paragraph("Valleys are often diagonal. The <b>active limits</b> block the directions along their gradient; the <b>free combinations</b> are the directions they leave open, sloppiest first. "
                              "A distance of 30 % means moving along that combination by 0.30 in coordinates scaled to the range of each input.", st["small"])]
            if cv["active"]:
                head += [table([(a["name"], f"{a['g']:+.3f}", a["text"]) for a in cv["active"]], [W_TEXT * x for x in (0.45, 0.12, 0.43)], ["Active limit", "margin (g ≤ 0 is met)", "moved most by"]), Spacer(1, 4)]
            flow += [KeepTogether(head)]
            drow = [(str(k + 1), d["text"], f"{100 * d['r_plus']:.0f} %", d["why_plus"], f"{100 * d['r_minus']:.0f} %", d["why_minus"]) for k, d in enumerate(cv["directions"])]
            flow += [table(drow, [W_TEXT * x for x in (0.04, 0.38, 0.08, 0.18, 0.08, 0.24)], ["#", "made of", "can move +", "ended by", "can move −", "ended by"]), Spacer(1, 4)]
    # ---------------------------------------------------------------- notes and inputs
    flow += [CondPageBreak(75 * mm), Paragraph("Notes on the model", st["h1"])]
    notes = [
        "The wing is a non-linear lifting line (Weissinger vortex lattice) over strips; each strip has the lift, drag and moment of its airfoil at its own Reynolds number, taken from NeuralFoil (a neural-network model of XFOIL) "
        "and followed past the stall. The tail, propeller and other installation effects are not modelled; the fuselage enters only through its drag and the induced-drag factor.",
        "With no wind-tunnel or flight data, treat the absolute drag and the stall speed as ±10 %: compare designs with each other and use the safety factor. NeuralFoil is optimistic at low Reynolds number, "
        "which the calibration factors on drag and maximum lift (inputs, Section physics) correct.",
        "Flaps and ailerons are thin-airfoil theory with viscous factors: the least certain part of the model. The pitching moment after the stall (a swept wing's tip-stall pitch-up) is only partly captured.",
        "Steady, planar, symmetric, incompressible flow. The equations, the checks against a hand calculation and a sweep of random designs, and the limits of the model are in the lab's physics document."]
    flow += [Paragraph(_esc(t), st["body"]) for t in notes]
    flow += [Paragraph("Appendix: every input of this design", st["h1"])]
    grp = None
    rows = []
    for g, lab, val, unit in input_rows(case):
        rows.append((g if g != grp else "", lab, val, unit))
        grp = g
    flow += [table(rows, [W_TEXT * x for x in (0.22, 0.50, 0.18, 0.10)], ["Group", "Input", "Value", "Unit"])]
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=16 * mm, title=f"{title}: {label}", author="HORYZN wing aerodynamics lab")
    doc.build(flow, onFirstPage=header_footer, onLaterPages=header_footer)
    return buf.getvalue()
