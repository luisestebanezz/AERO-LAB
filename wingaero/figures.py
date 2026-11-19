"""Plotly figures of the lab, in the order of the results page: the airfoils, the top view with the fuselage, and the charts of the
airfoil and of the wing. Every chart takes one or more cases (for example the starting design and the optimised one); the
calculated curve is a solid line and the curve with the safety factor applied is dashed in the same colour."""
from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import airfoil as AF
from .analysis import Case

HZ = {"navy": "#1D3043", "ink": "#1F2933", "orange": "#C2410C", "blue": "#2C6596", "amber": "#E0A040", "steel": "#6B7A89", "green": "#2E7D5B",
      "red": "#B23B3B", "mist": "#F3F7FA", "line": "#E5E7EB"}
COLORS = [HZ["blue"], HZ["orange"], HZ["green"], HZ["amber"], HZ["navy"], HZ["steel"]]


def _fig(title: str, xt: str, yt: str, h: int = 420) -> go.Figure:
    f = go.Figure()
    f.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, height=h, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="v", yanchor="top", y=-0.18, x=0), font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=HZ["ink"]))
    f.update_xaxes(gridcolor=HZ["line"], zerolinecolor="#CBD5E1")
    f.update_yaxes(gridcolor=HZ["line"], zerolinecolor="#CBD5E1")
    return f


def _col(i: int) -> str:
    return COLORS[i % len(COLORS)]


def _sf_note(cases) -> str:
    c = cases[0][1]
    f = c.sf
    return f"Dashed: safety factor {c.p['safety_factor']:.2f} applied (drag ×{f['drag']:.2f}, lift ÷{f['lift']:.2f}, moments ×{f['moment']:.2f}, control ÷{f['control']:.2f})"


# ------------------------------------------------------------------------------------------------ 1 airfoils
def station_foils(case: Case) -> list[tuple[str, AF.Airfoil, float]]:
    """(station name, airfoil, span fraction) for every airfoil station of the wing."""
    names = {"root": "Root", "1": "First break", "2": "Second break", "tip": "Tip"}
    keys = {2: ["root", "tip"], 3: ["root", "1", "tip"], 4: ["root", "1", "2", "tip"]}[len(case.foils)]
    return [(names[k], af, float(eta)) for k, af, eta in zip(keys, case.foils, case.foil_eta)]


def airfoil_figure(case: Case) -> go.Figure:
    """The section at every station on one chord, with its camber line."""
    f = _fig("Airfoil at each station (chord = 1)", "x / c", "y / c", 380)
    f.update_yaxes(scaleanchor="x", scaleratio=1)
    seen = {}
    for i, (st, af, eta) in enumerate(station_foils(case)):
        sh = af.shape
        col = _col(i)
        up, lo = AF.split_surfaces(af.coords)
        x = np.concatenate([up[:, 0], lo[::-1, 0]])
        y = np.concatenate([up[:, 1], lo[::-1, 1]])
        lab = f"{st} (η {eta:.2f}): {af.label}, t/c {af.thickness * 100:.1f} %, camber {sh['camber_max'] * 100:.1f} % at {sh['x_camber'] * 100:.0f} %"
        f.add_trace(go.Scatter(x=x, y=y, mode="lines", name=lab, line=dict(color=col, width=2.5), fill="toself" if i == 0 else None,
                               fillcolor="rgba(44,101,150,0.12)"))
        f.add_trace(go.Scatter(x=sh["x"], y=sh["camber"], mode="lines", name=f"camber line, {st}", line=dict(color=col, width=1, dash="dot"), showlegend=False))
    return f


def airfoil_distribution_figure(case: Case) -> go.Figure:
    f = make_subplots(rows=1, cols=2, subplot_titles=("Thickness distribution", "Camber line"))
    for i, (st, af, eta) in enumerate(station_foils(case)):
        sh = af.shape
        f.add_trace(go.Scatter(x=sh["x"], y=sh["thickness"], mode="lines", name=f"{st}: {af.label}", line=dict(color=_col(i), width=2)), row=1, col=1)
        f.add_trace(go.Scatter(x=sh["x"], y=sh["camber"], mode="lines", name=st, line=dict(color=_col(i), width=2), showlegend=False), row=1, col=2)
    f.update_layout(height=320, template="plotly_white", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    f.update_xaxes(title_text="x / c", gridcolor=HZ["line"])
    f.update_yaxes(gridcolor=HZ["line"])
    return f


def airfoil_rows(case: Case) -> list[dict]:
    v = case.V
    rows = []
    for st, af, eta in station_foils(case):
        re = float(case.air.density * v * case.wing.chord(eta) / case.air.viscosity)
        pol = af.polar(re)
        rows.append({"Station": st, "η": round(eta, 2), "Airfoil": af.label, "Re at the design speed": f"{re / 1e3:,.0f} k", "t/c %": round(af.thickness * 100, 1),
                     "camber %": round(af.shape["camber_max"] * 100, 2), "cl max": round(pol["cl_max"], 3), "stall angle °": round(pol["alpha_stall_deg"], 1),
                     "cd min": round(pol["cd_min"], 4), "(cl/cd) max": round(pol["ld_max"], 0), "zero-lift angle °": round(pol["alpha0_deg"], 2),
                     "cm0": round(pol["cm0"], 3), "lift slope /°": round(pol["lift_slope_per_deg"], 4), "NeuralFoil confidence": round(pol["confidence_min"], 2)})
    return rows


# ------------------------------------------------------------------------------------------------ 2 top view
def top_view(case: Case, title: str | None = None) -> go.Figure:
    """Wing, fuselage, control surfaces, mean aerodynamic chord, aerodynamic centre and the moment reference point, nose up."""
    w, p = case.wing, case.p
    f = _fig(title or "Top view", "span position (m)", "along the fuselage, nose up (m)", 620)
    f.update_yaxes(scaleanchor="x", scaleratio=1)
    o = w.outline()
    x_wing0 = -p["wing_x_fraction"] * case.fuse.length if case.fuse is not None else 0.0     # fuselage nose in wing coordinates
    ys = np.concatenate([o["y"], o["y"][::-1]])
    xs = np.concatenate([o["x_le"], o["x_te"][::-1]])
    for sgn in (1, -1):
        f.add_trace(go.Scatter(x=sgn * ys, y=-xs, mode="lines", fill="toself", fillcolor="rgba(44,101,150,0.28)", line=dict(color=HZ["navy"], width=2),
                               name="wing", showlegend=sgn == 1, hoverinfo="skip"))
    # quarter-chord line, panel breaks
    for sgn in (1, -1):
        f.add_trace(go.Scatter(x=sgn * o["y"], y=-o["x_qc"], mode="lines", line=dict(color=HZ["steel"], width=1, dash="dash"), name="quarter-chord line", showlegend=sgn == 1))
    for k in range(1, len(o["y"]) - 1):
        if w.n_panels <= 3 and len(o["y"]) <= 4:
            for sgn in (1, -1):
                f.add_trace(go.Scatter(x=[sgn * o["y"][k]] * 2, y=[-o["x_le"][k], -o["x_te"][k]], mode="lines", line=dict(color=HZ["steel"], width=1, dash="dot"), showlegend=False, hoverinfo="skip"))
    # flap and aileron
    for on, a, b, cf, colr, nm in ((p["flap_on"], p["flap_eta_in"], p["flap_eta_out"], p["flap_chord"], HZ["amber"], "flap"),
                                   (p["aileron_on"], p["aileron_eta_in"], p["aileron_eta_out"], p["aileron_chord"], HZ["orange"], "aileron")):
        if not on:
            continue
        ya, yb = a * 0.5 * w.span, b * 0.5 * w.span
        px = [ya, yb, yb, ya]
        py = [-float(w.x_te(a)), -float(w.x_te(b)), -(float(w.x_te(b)) - cf * float(w.chord(b))), -(float(w.x_te(a)) - cf * float(w.chord(a)))]
        for sgn in (1, -1):
            f.add_trace(go.Scatter(x=[sgn * x_ for x_ in px] + [sgn * px[0]], y=py + [py[0]], mode="lines", fill="toself", fillcolor=colr, opacity=0.7,
                                   line=dict(color=colr, width=1), name=nm, showlegend=sgn == 1))
    # fuselage
    if case.fuse is not None:
        xf, hw = case.fuse.outline()
        xw = xf + x_wing0
        f.add_trace(go.Scatter(x=np.concatenate([hw, -hw[::-1]]), y=np.concatenate([-xw, -xw[::-1]]), mode="lines", fill="toself",
                               fillcolor="rgba(29,48,67,0.55)", line=dict(color=HZ["navy"], width=2), name="fuselage", hoverinfo="skip"))
    # mean aerodynamic chord, aerodynamic centre, reference point
    x_le_m, mac = w.x_le_mac, w.mac
    for sgn in (1, -1):
        f.add_trace(go.Scatter(x=[sgn * w.y_mac] * 2, y=[-x_le_m, -(x_le_m + mac)], mode="lines", line=dict(color=HZ["orange"], width=5), name="mean aerodynamic chord", showlegend=sgn == 1))
    full = case.full()
    x_ac = full["metrics"]["x_ac_m"]
    f.add_trace(go.Scatter(x=[0], y=[-x_ac], mode="markers", marker=dict(symbol="diamond", size=12, color=HZ["green"], line=dict(color="white", width=1)), name="aerodynamic centre (wing alone)"))
    f.add_trace(go.Scatter(x=[0], y=[-case.x_ref], mode="markers", marker=dict(symbol="x", size=13, color=HZ["red"], line=dict(width=2)), name=f"moment reference ({p['x_cg_mac'] * 100:.0f} % MAC)"))
    half = 0.5 * w.span
    f.update_layout(title=(title or "Top view") + f"<br><sup>span {w.span:.2f} m · area {w.area:.3f} m² · AR {w.aspect_ratio:.1f} · MAC {w.mac:.3f} m · "
                    f"root chord {w.root_chord:.3f} m · tip chord {w.tip_chord:.3f} m · mean sweep {w.mean_sweep_deg:.1f}°</sup>")
    f.update_xaxes(range=[-1.08 * half, 1.08 * half])
    return f


# ------------------------------------------------------------------------------------------------ 3 airfoil charts
def _foil_re(case: Case) -> list[tuple[str, AF.Airfoil, float]]:
    out = []
    for st, af, eta in station_foils(case):
        out.append((st, af, float(case.air.density * case.V * case.wing.chord(eta) / case.air.viscosity)))
    return out


def airfoil_charts(case: Case) -> dict[str, go.Figure]:
    """cl-alpha, drag polar, cl/cd, cm and Reynolds-number sensitivity of every station airfoil at its design Reynolds number."""
    sf = case.sf
    figs = {k: _fig(t, xt, yt, 400) for k, (t, xt, yt) in {
        "cl": ("Lift curve", "angle of attack (°)", "cl"), "polar": ("Drag polar", "cd", "cl"), "ld": ("Section glide ratio", "angle of attack (°)", "cl / cd"),
        "cm": ("Pitching moment about the quarter chord", "angle of attack (°)", "cm"), "cmcl": ("Pitching moment against lift", "cl", "cm"),
        "re_clmax": ("Maximum lift against Reynolds number", "Reynolds number", "cl max"), "re_cd": ("Minimum drag against Reynolds number", "Reynolds number", "cd min"),
        "conf": ("NeuralFoil analysis confidence", "angle of attack (°)", "confidence (1 = high)")}.items()}
    shown = set()
    for i, (st, af, re) in enumerate(_foil_re(case)):
        key = (af.name, round(re, -3))
        if key in shown:
            continue
        shown.add(key)
        col = _col(i)
        lab = f"{st}: {af.label}, Re {re / 1e3:,.0f} k"
        pol = af.polar(re)
        a, pre = pol["alpha_deg"], pol["pre_stall"]
        cl, cd, cm = pol["cl"], pol["cd"], pol["cm"]
        figs["cl"].add_trace(go.Scatter(x=a[pre], y=cl[pre], mode="lines", name=lab, line=dict(color=col, width=2.5), legendgroup=lab))
        figs["cl"].add_trace(go.Scatter(x=a[~pre], y=cl[~pre], mode="lines", line=dict(color=col, width=1.5, dash="dot"), showlegend=False, legendgroup=lab))
        figs["cl"].add_trace(go.Scatter(x=a, y=np.minimum(cl, pol["cl_max"] / sf["lift"]), mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False, legendgroup=lab))
        figs["polar"].add_trace(go.Scatter(x=cd[pre], y=cl[pre], mode="lines", name=lab, line=dict(color=col, width=2.5)))
        figs["polar"].add_trace(go.Scatter(x=cd[pre] * sf["drag"], y=cl[pre], mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False))
        figs["ld"].add_trace(go.Scatter(x=a[pre], y=cl[pre] / cd[pre], mode="lines", name=lab, line=dict(color=col, width=2.5)))
        figs["ld"].add_trace(go.Scatter(x=a[pre], y=cl[pre] / (cd[pre] * sf["drag"]), mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False))
        figs["cm"].add_trace(go.Scatter(x=a[pre], y=cm[pre], mode="lines", name=lab, line=dict(color=col, width=2.5)))
        figs["cm"].add_trace(go.Scatter(x=a[pre], y=cm[pre] * sf["moment"], mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False))
        figs["cmcl"].add_trace(go.Scatter(x=cl[pre], y=cm[pre], mode="lines", name=lab, line=dict(color=col, width=2.5)))
        figs["conf"].add_trace(go.Scatter(x=a, y=pol["confidence"], mode="lines", name=lab, line=dict(color=col, width=2)))
        figs["re_clmax"].add_trace(go.Scatter(x=AF.REYNOLDS, y=af.clmax * case.p["clmax_factor"], mode="lines+markers", name=af.label, line=dict(color=col, width=2)))
        figs["re_clmax"].add_trace(go.Scatter(x=AF.REYNOLDS, y=af.clmax * case.p["clmax_factor"] / sf["lift"], mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False))
        figs["re_cd"].add_trace(go.Scatter(x=AF.REYNOLDS, y=af.cdmin * case.p["cd_factor"], mode="lines+markers", name=af.label, line=dict(color=col, width=2)))
        figs["re_cd"].add_trace(go.Scatter(x=AF.REYNOLDS, y=af.cdmin * case.p["cd_factor"] * sf["drag"], mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False))
    for k in ("re_clmax", "re_cd"):
        figs[k].update_xaxes(type="log")
    figs["conf"].update_yaxes(range=[0, 1.02])
    figs["conf"].add_hline(y=0.5, line=dict(color=HZ["red"], dash="dot"), annotation_text="below 0.5: treat with caution")
    figs["cl"].add_annotation(xref="paper", yref="paper", x=0, y=1.0, text="Dotted: beyond the stall (lower confidence)", showarrow=False, font=dict(size=10, color=HZ["steel"]))
    return figs


# ------------------------------------------------------------------------------------------------ 4 wing charts
def _pre(cv: dict, st: dict | None):
    return np.ones(len(cv["CL"]), bool)


def wing_lift(cases) -> go.Figure:
    f = _fig("Wing lift curve", "root angle of attack (°)", "CL", 420)
    for i, (lab, c) in enumerate(cases):
        r, col = c.full(), _col(i)
        cv, m = r["curve"], r["metrics"]
        f.add_trace(go.Scatter(x=cv["alpha_deg"], y=cv["CL"], mode="lines", name=lab, line=dict(color=col, width=2.5)))
        if "CL_max_first" in m:
            f.add_trace(go.Scatter(x=[m["alpha_stall_first_deg"]], y=[m["CL_max_first"]], mode="markers", marker=dict(symbol="triangle-down", size=12, color=col),
                                   name=f"{lab}: first strip stalls (η {m['eta_stall_first']:.2f})"))
            f.add_hline(y=m["CL_max_first"] / c.sf["lift"], line=dict(color=col, dash="dash", width=1.5), annotation_text=f"{lab}: usable CLmax with SF" if i == 0 else None,
                        annotation_position="bottom right")
        if r.get("design_ok"):
            pt = r["design_point"]
            f.add_trace(go.Scatter(x=[pt.alpha_deg], y=[pt.CL], mode="markers", marker=dict(symbol="circle", size=10, color=col, line=dict(color="white", width=1.5)), name=f"{lab}: design point"))
    f.add_annotation(xref="paper", yref="paper", x=1, y=0, xanchor="right", yanchor="bottom", text=_sf_note(cases), showarrow=False, font=dict(size=10, color=HZ["steel"]))
    return f


def wing_polar(cases) -> go.Figure:
    f = _fig("Drag polar", "CD", "CL", 420)
    for i, (lab, c) in enumerate(cases):
        r, col = c.full(), _col(i)
        cv = r["curve"]
        n = int(np.argmax(cv["CL"])) + 1
        f.add_trace(go.Scatter(x=cv["CD"][:n], y=cv["CL"][:n], mode="lines", name=lab, line=dict(color=col, width=2.5)))
        f.add_trace(go.Scatter(x=cv["CD"][:n] * c.sf["drag"], y=cv["CL"][:n], mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False))
        if r.get("design_ok"):
            pt = r["design_point"]
            f.add_trace(go.Scatter(x=[pt.CD], y=[pt.CL], mode="markers", marker=dict(size=10, color=col, line=dict(color="white", width=1.5)), name=f"{lab}: design point", showlegend=False))
    return f


def wing_drag_breakdown(case: Case) -> go.Figure:
    f = _fig("Drag breakdown", "CL", "CD", 420)
    cv = case.full()["curve"]
    n = int(np.argmax(cv["CL"])) + 1
    for k, nm, col in (("CDp", "profile (sections, strip by strip)", HZ["blue"]), ("CDi", "induced (lifting line)", HZ["orange"]), ("CDf", "fuselage", HZ["steel"])):
        f.add_trace(go.Scatter(x=cv["CL"][:n], y=cv[k][:n], mode="lines", name=nm, line=dict(color=col, width=2.5), stackgroup="one"))
    return f


def wing_ld(cases) -> go.Figure:
    f = _fig("Lift-to-drag against lift coefficient", "CL", "CL / CD", 420)
    for i, (lab, c) in enumerate(cases):
        r, col = c.full(), _col(i)
        cv = r["curve"]
        n = int(np.argmax(cv["CL"])) + 1
        ok = cv["CL"][:n] > 0
        f.add_trace(go.Scatter(x=cv["CL"][:n][ok], y=(cv["CL"] / cv["CD"])[:n][ok], mode="lines", name=lab, line=dict(color=col, width=2.5)))
        f.add_trace(go.Scatter(x=cv["CL"][:n][ok], y=(cv["CL"] / (cv["CD"] * c.sf["drag"]))[:n][ok], mode="lines", line=dict(color=col, width=1.5, dash="dash"), showlegend=False))
    return f


def wing_moment(cases) -> go.Figure:
    f = make_subplots(rows=1, cols=2, subplot_titles=("Pitching moment against lift", "Pitching moment against angle of attack"))
    for i, (lab, c) in enumerate(cases):
        r, col = c.full(), _col(i)
        cv, m = r["curve"], r["metrics"]
        n = int(np.argmax(cv["CL"])) + 1
        for key, dash, nm in (("Cm_ref", "solid", f"{lab}: about the reference point ({c.p['x_cg_mac'] * 100:.0f} % MAC)"), ("Cm_ac", "dot", f"{lab}: about the aerodynamic centre")):
            f.add_trace(go.Scatter(x=cv["CL"][:n], y=cv[key][:n], mode="lines", name=nm, line=dict(color=col, width=2.5 if dash == "solid" else 1.5, dash=dash), legendgroup=lab), row=1, col=1)
            f.add_trace(go.Scatter(x=cv["alpha_deg"][:n], y=cv[key][:n], mode="lines", showlegend=False, line=dict(color=col, width=2.5 if dash == "solid" else 1.5, dash=dash), legendgroup=lab), row=1, col=2)
        f.add_trace(go.Scatter(x=cv["CL"][:n], y=cv["Cm_ref"][:n] * c.sf["moment"], mode="lines", showlegend=False, line=dict(color=col, width=1.5, dash="dash"), legendgroup=lab), row=1, col=1)
        if "CL_max_first" in m:
            f.add_vline(x=m["CL_max_first"], line=dict(color=col, dash="dashdot", width=1), row=1, col=1)
        if r.get("design_ok"):
            f.add_trace(go.Scatter(x=[m["CL_design"]], y=[m["Cm_ref"]], mode="markers", marker=dict(size=10, color=col, line=dict(color="white", width=1.5)), showlegend=False), row=1, col=1)
    f.update_layout(height=470, template="plotly_white", margin=dict(l=10, r=10, t=50, b=70), legend=dict(orientation="h", y=-0.22))
    f.update_xaxes(title_text="CL", gridcolor=HZ["line"], row=1, col=1)
    f.update_xaxes(title_text="root angle of attack (°)", gridcolor=HZ["line"], row=1, col=2)
    f.update_yaxes(title_text="Cm", gridcolor=HZ["line"])
    f.add_annotation(xref="paper", yref="paper", x=0, y=-0.42, text="Dash-dot vertical: first strip stalls. " + _sf_note(cases), showarrow=False, font=dict(size=10, color=HZ["steel"]))
    return f


def _right(c: Case) -> np.ndarray:
    return c.ll.right


def spanwise_stall_map(case: Case) -> go.Figure:
    """Strip lift against the strip's own maximum along the span, at several wing lift coefficients up to the stall."""
    r = case.full()
    cv, m = r["curve"], r["metrics"]
    h = _right(case)
    eta = case.ll.eta[h]
    f = _fig("Section lift against section maximum along the span", "η = y / (b/2)", "section cl", 460)
    tab = case.tab
    f.add_trace(go.Scatter(x=eta, y=tab.cap[h], mode="lines", name="section maximum (cap)", line=dict(color=HZ["red"], width=3)))
    f.add_trace(go.Scatter(x=eta, y=tab.cap[h] / case.sf["lift"], mode="lines", name="allowed with the safety factor", line=dict(color=HZ["red"], width=2, dash="dash")))
    cls = m.get("CL_max_first", float(cv["CL"].max()))
    targets = [("0.4", 0.4), ("0.6", 0.6), ("0.8", 0.8), ("0.9", 0.9), ("1.0 (first strip stalls)", 1.0)]
    for k, (nm, fr) in enumerate(targets):
        i = int(np.argmin(np.abs(cv["CL"][: int(np.argmax(cv["CL"])) + 1] - fr * cls)))
        f.add_trace(go.Scatter(x=eta, y=cv["cl_j"][i][h], mode="lines", name=f"{nm} CLmax: CL {cv['CL'][i]:.2f}", line=dict(color=COLORS[(k + 1) % len(COLORS)], width=2)))
    if r.get("design_ok"):
        f.add_trace(go.Scatter(x=eta, y=r["design_point"].cl_j[h], mode="lines", name=f"design point: CL {m['CL_design']:.2f}", line=dict(color=HZ["navy"], width=3, dash="dot")))
    f.add_vline(x=m.get("eta_stall_first", 0), line=dict(color=HZ["amber"], dash="dot"), annotation_text="stall starts" if "eta_stall_first" in m else None)
    for nm, a_, b_ in (("aileron", case.p["aileron_eta_in"], case.p["aileron_eta_out"]) if case.p["aileron_on"] else (None, 0, 0),
                       ("flap", case.p["flap_eta_in"], case.p["flap_eta_out"]) if case.p["flap_on"] else (None, 0, 0)):
        if nm:
            f.add_vrect(x0=a_, x1=b_, fillcolor=HZ["amber"] if nm == "flap" else HZ["orange"], opacity=0.12, line_width=0, annotation_text=nm, annotation_position="top left")
    return f


def utilisation_heatmap(case: Case) -> go.Figure:
    r = case.full()
    cv = r["curve"]
    h = _right(case)
    n = int(np.argmax(cv["CL"])) + 1
    z = cv["util_j"][:n][:, h] * case.sf["lift"]
    f = go.Figure(go.Heatmap(x=case.ll.eta[h], y=cv["CL"][:n], z=np.minimum(z, 1.6), colorscale=[[0, "#F3F7FA"], [0.5, "#2C6596"], [0.62, "#E0A040"], [0.625, "#B23B3B"], [1, "#7a1f1f"]],
                             zmin=0, zmax=1.6, colorbar=dict(title="SF · cl / cl max")))
    f.update_layout(title="Strip utilisation (red: beyond the allowed lift with the safety factor)", xaxis_title="η = y / (b/2)", yaxis_title="wing CL", height=430,
                    template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    return f


def spanwise_state(case: Case) -> go.Figure:
    """At the design point: chord, Reynolds number, effective angle against the stall angle, and the loading."""
    r = case.full()
    h = _right(case)
    eta = case.ll.eta[h]
    f = make_subplots(rows=2, cols=2, subplot_titles=("Chord and Reynolds number", "Effective angle against stall angle", "Lift loading (cl·c / MAC)", "Section drag (cd)"))
    f.add_trace(go.Scatter(x=eta, y=case.ll.c[h] * 1000.0, name="chord (mm)", line=dict(color=HZ["navy"], width=2)), row=1, col=1)
    f.add_trace(go.Scatter(x=eta, y=case.tab.re[h] / 1e3, name="Re (thousand)", line=dict(color=HZ["orange"], width=2)), row=1, col=1)
    if r.get("design_ok"):
        pt = r["design_point"]
        f.add_trace(go.Scatter(x=eta, y=pt.aeff_deg_j[h], name="effective angle (°)", line=dict(color=HZ["blue"], width=2)), row=1, col=2)
        f.add_trace(go.Scatter(x=eta, y=case.tab.alpha_cap_deg[h], name="angle at the section maximum (°)", line=dict(color=HZ["red"], width=2, dash="dash")), row=1, col=2)
        f.add_trace(go.Scatter(x=eta, y=pt.cl_j[h] * case.ll.c[h] / case.wing.mac, name="cl·c/MAC", line=dict(color=HZ["green"], width=2)), row=2, col=1)
        f.add_trace(go.Scatter(x=eta, y=pt.cd_j[h], name="cd", line=dict(color=HZ["steel"], width=2)), row=2, col=2)
    f.update_layout(height=620, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.12))
    f.update_xaxes(title_text="η", gridcolor=HZ["line"])
    f.update_yaxes(gridcolor=HZ["line"])
    return f


def wing_control(cases) -> go.Figure:
    f = make_subplots(rows=1, cols=2, subplot_titles=("Roll power of the ailerons against lift", "Roll moment at full deflection against angle of attack"))
    any_ail = False
    for i, (lab, c) in enumerate(cases):
        if not c.p["aileron_on"]:
            continue
        any_ail = True
        r, col = c.full(), _col(i)
        cv, m = r["curve"], r["metrics"]
        n = int(np.argmax(cv["CL"])) + 1
        f.add_trace(go.Scatter(x=cv["CL"][:n], y=cv["ail_small"][:n], mode="lines", name=lab, line=dict(color=col, width=2.5), legendgroup=lab), row=1, col=1)
        f.add_trace(go.Scatter(x=cv["CL"][:n], y=cv["ail_small"][:n] / c.sf["control"], mode="lines", showlegend=False, line=dict(color=col, width=1.5, dash="dash"), legendgroup=lab), row=1, col=1)
        f.add_trace(go.Scatter(x=cv["alpha_deg"][:n], y=cv["ail_max"][:n], mode="lines", showlegend=False, line=dict(color=col, width=2.5), legendgroup=lab), row=1, col=2)
        f.add_trace(go.Scatter(x=cv["alpha_deg"][:n], y=cv["ail_max"][:n] / c.sf["control"], mode="lines", showlegend=False, line=dict(color=col, width=1.5, dash="dash"), legendgroup=lab), row=1, col=2)
        if "CL_max_first" in m:
            f.add_vline(x=m["CL_max_first"], line=dict(color=col, dash="dashdot", width=1), row=1, col=1)
        if not math.isnan(m.get("control_loss_CL", float("nan"))):
            f.add_vline(x=m["control_loss_CL"], line=dict(color=col, dash="dot", width=1), row=1, col=1, annotation_text="half the roll power lost")
    f.add_hline(y=0, line=dict(color=HZ["steel"], width=1), row=1, col=1)
    f.update_layout(height=430, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    f.update_xaxes(title_text="wing CL", gridcolor=HZ["line"], row=1, col=1)
    f.update_xaxes(title_text="root angle of attack (°)", gridcolor=HZ["line"], row=1, col=2)
    f.update_yaxes(title_text="Cl per rad of aileron", gridcolor=HZ["line"], row=1, col=1)
    f.update_yaxes(title_text="Cl (full deflection)", gridcolor=HZ["line"], row=1, col=2)
    if not any_ail:
        f.add_annotation(text="Switch the ailerons on (Inputs, group Ailerons) to see the roll control.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=14))
    return f


def wing_performance(cases) -> go.Figure:
    f = make_subplots(rows=1, cols=2, subplot_titles=("Power required against speed", "Lift-to-drag against speed"))
    for i, (lab, c) in enumerate(cases):
        r, col = c.full(), _col(i)
        pf, m = r.get("performance", {}), r["metrics"]
        if not len(pf.get("V", [])):
            continue
        f.add_trace(go.Scatter(x=pf["V"], y=pf["P"], mode="lines", name=lab, line=dict(color=col, width=2.5), legendgroup=lab), row=1, col=1)
        f.add_trace(go.Scatter(x=pf["V"], y=pf["P"] * c.sf["drag"], mode="lines", showlegend=False, line=dict(color=col, width=1.5, dash="dash"), legendgroup=lab), row=1, col=1)
        f.add_trace(go.Scatter(x=pf["V"], y=pf["LD"], mode="lines", showlegend=False, line=dict(color=col, width=2.5), legendgroup=lab), row=1, col=2)
        f.add_trace(go.Scatter(x=pf["V"], y=pf["LD"] / c.sf["drag"], mode="lines", showlegend=False, line=dict(color=col, width=1.5, dash="dash"), legendgroup=lab), row=1, col=2)
        if "V_stall_mps" in m:
            f.add_vline(x=m["V_stall_mps"], line=dict(color=col, dash="dashdot", width=1), row=1, col=1)
            f.add_vline(x=m["V_stall_mps"] * math.sqrt(c.sf["lift"]), line=dict(color=col, dash="dot", width=1), row=1, col=1)
        f.add_vline(x=m["V_design_mps"], line=dict(color=col, width=1), row=1, col=1)
    f.update_layout(height=470, template="plotly_white", margin=dict(l=10, r=10, t=50, b=70), legend=dict(orientation="h", y=-0.22))
    f.update_xaxes(title_text="speed (m/s)", gridcolor=HZ["line"])
    f.update_yaxes(title_text="aerodynamic power D·V (W)", gridcolor=HZ["line"], row=1, col=1)
    f.update_yaxes(title_text="CL / CD", gridcolor=HZ["line"], row=1, col=2)
    f.add_annotation(xref="paper", yref="paper", x=0, y=-0.42, text="Vertical lines: dash-dot stall speed, dotted stall speed with the safety factor, solid design speed. " + _sf_note(cases), showarrow=False, font=dict(size=10, color=HZ["steel"]))
    return f


def stall_speed_mass(cases) -> go.Figure:
    f = _fig("Stall speed against mass", "mass (kg)", "stall speed (m/s)", 400)
    for i, (lab, c) in enumerate(cases):
        m = c.full()["metrics"]
        if "CL_max_first" not in m:
            continue
        ms = np.linspace(0.5 * c.p["mass_kg"], 1.5 * c.p["mass_kg"], 30)
        v = np.sqrt(2.0 * c.n * ms * G * 1.0 / (c.air.density * c.wing.area * m["CL_max_first"]))
        f.add_trace(go.Scatter(x=ms, y=v, mode="lines", name=lab, line=dict(color=_col(i), width=2.5)))
        f.add_trace(go.Scatter(x=ms, y=v * math.sqrt(c.sf["lift"]), mode="lines", showlegend=False, line=dict(color=_col(i), width=1.5, dash="dash")))
        f.add_trace(go.Scatter(x=[c.p["mass_kg"]], y=[m["V_stall_mps"]], mode="markers", marker=dict(size=10, color=_col(i)), showlegend=False))
    return f


G = 9.80665


def flap_effect(case: Case) -> go.Figure:
    f = _fig("Effect of the flap", "root angle of attack (°)", "CL", 420)
    c = case
    if not c.p["flap_on"] or c.flap_defl <= 0:
        f.add_annotation(text="Switch the flaps on (Inputs, group Flaps) to see their effect.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=14))
        return f
    cv1 = c.full()["curve"]
    cv0 = c.curve(c.tables(c.V, 0.0), alpha_min=float(cv1['alpha_deg'][0]), with_roll=False)
    f.add_trace(go.Scatter(x=cv0["alpha_deg"], y=cv0["CL"], mode="lines", name="flap up", line=dict(color=HZ["steel"], width=2.5)))
    f.add_trace(go.Scatter(x=cv1["alpha_deg"], y=cv1["CL"], mode="lines", name=f"flap {c.flap_defl:.0f}° down", line=dict(color=HZ["orange"], width=2.5)))
    f.add_annotation(xref="paper", yref="paper", x=1, y=0, xanchor="right", yanchor="bottom", text="Flaps: thin-airfoil theory with viscous factors; lower confidence", showarrow=False, font=dict(size=10, color=HZ["steel"]))
    return f


def sensitivity_figure(rows: list[dict], metric_label: str) -> go.Figure:
    """Tornado chart: the change of a result when each uncertain input goes to the low and the high end of its range."""
    rows = sorted(rows, key=lambda r: abs(r["hi"] - r["lo"]))
    f = _fig(f"Sensitivity of {metric_label} to each uncertain input", "change of the result relative to the base case (%)", "", max(300, 40 * len(rows) + 120))
    f.add_trace(go.Bar(y=[r["label"] for r in rows], x=[r["lo"] for r in rows], orientation="h", name="input at its lower bound", marker_color=HZ["blue"]))
    f.add_trace(go.Bar(y=[r["label"] for r in rows], x=[r["hi"] for r in rows], orientation="h", name="input at its upper bound", marker_color=HZ["orange"]))
    f.update_layout(barmode="relative")
    return f


def optimiser_history(history: list[float]) -> go.Figure:
    f = _fig("Optimiser progress", "generation", "best weighted objective (lower is better)", 320)
    f.add_trace(go.Scatter(x=list(range(1, len(history) + 1)), y=history, mode="lines+markers", line=dict(color=HZ["orange"], width=2.5)))
    return f


# ------------------------------------------------------------------------------------------------ search through the ranges
def explore_scatter(sample, kx: str, ky: str, kc: str, labels: dict) -> go.Figure:
    """Every design of the sample as a point: two results on the axes, a third as colour; designs that break a limit are grey."""
    f = _fig("The sample", labels.get(kx, kx), labels.get(ky, ky), 480)
    ok = [r for r in sample.usable() if r["feasible"]]
    bad = [r for r in sample.usable() if not r["feasible"]]

    def hover(r):
        def show(v):
            if isinstance(v, bool):
                return "yes" if v else "no"
            return v if isinstance(v, str) else round(v, 3)
        return "<br>".join(f"{k}: {show(v)}" for k, v in r["inputs"].items())
    if bad:
        f.add_trace(go.Scatter(x=[r["m"][kx] for r in bad], y=[r["m"][ky] for r in bad], mode="markers", name="breaks a limit",
                               marker=dict(size=6, color="#B8C0CA"), text=[hover(r) for r in bad], hoverinfo="text"))
    if ok:
        f.add_trace(go.Scatter(x=[r["m"][kx] for r in ok], y=[r["m"][ky] for r in ok], mode="markers", name="meets every limit",
                               marker=dict(size=8, color=[r["m"][kc] for r in ok], colorscale="Viridis", showscale=True, colorbar=dict(title=labels.get(kc, kc)),
                                           line=dict(color="white", width=0.5)), text=[hover(r) for r in ok], hoverinfo="text"))
    return f


def effects_figure(rows: list[dict], metric_label: str) -> go.Figure:
    rows = sorted(rows, key=lambda r: abs(r["effect"]))
    f = _fig(f"Which input moves {metric_label}", "rank correlation (numbers, signed) or √ of the variance explained (airfoils)", "", max(300, 38 * len(rows) + 120))
    f.add_trace(go.Bar(y=[r["label"] for r in rows], x=[r["effect"] for r in rows], orientation="h",
                       marker_color=[HZ["orange"] if r["effect"] > 0 else HZ["blue"] for r in rows]))
    f.update_xaxes(range=[-1, 1])
    return f


def sweep1d_figure(sw: dict, metric_key: str, metric_label: str, input_label: str) -> go.Figure:
    f = _fig(f"{metric_label} against {input_label}", input_label, metric_label, 420)
    xs = [r["x"] for r in sw["rows"] if r["m"] is not None and r["m"].get("ok") and r["m"].get(metric_key) is not None]
    ys = [r["m"][metric_key] for r in sw["rows"] if r["m"] is not None and r["m"].get("ok") and r["m"].get(metric_key) is not None]
    f.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=HZ["navy"], width=2), name=metric_label))
    okx = [r["x"] for r in sw["rows"] if r["feasible"] and r["m"].get(metric_key) is not None]
    oky = [r["m"][metric_key] for r in sw["rows"] if r["feasible"] and r["m"].get(metric_key) is not None]
    badx = [r["x"] for r in sw["rows"] if r["m"] is not None and r["m"].get("ok") and not r["feasible"] and r["m"].get(metric_key) is not None]
    bady = [r["m"][metric_key] for r in sw["rows"] if r["m"] is not None and r["m"].get("ok") and not r["feasible"] and r["m"].get(metric_key) is not None]
    f.add_trace(go.Scatter(x=okx, y=oky, mode="markers", marker=dict(size=9, color=HZ["green"]), name="meets every limit"))
    f.add_trace(go.Scatter(x=badx, y=bady, mode="markers", marker=dict(size=9, color=HZ["red"], symbol="x"), name="breaks a limit"))
    return f


def sweep2d_figure(sw: dict, metric_key: str, metric_label: str, lx: str, ly: str) -> go.Figure:
    z = [[(c["m"].get(metric_key) if (c["m"] is not None and c["m"].get("ok")) else None) for c in row] for row in sw["grid"]]
    bad = [[0 if c["feasible"] else 1 for c in row] for row in sw["grid"]]
    f = go.Figure(go.Heatmap(x=sw["x"], y=sw["y"], z=z, colorscale="Viridis", colorbar=dict(title=metric_label)))
    f.add_trace(go.Heatmap(x=sw["x"], y=sw["y"], z=bad, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(184,192,202,0.82)"]], showscale=False, zmin=0, zmax=1, hoverinfo="skip"))
    f.update_layout(title=f"{metric_label}: grey where a limit is broken", xaxis_title=lx, yaxis_title=ly, height=480, template="plotly_white", margin=dict(l=10, r=10, t=50, b=10))
    return f


# ------------------------------------------------------------------------------------------------ valleys around a design (valleys.py)
VERDICT_COLORS = {"free": "#2E7D5B", "loose": "#7BAE7F", "tight": "#E0A040", "critical": "#B23B3B"}


def _runs(flags) -> list[tuple[int, int]]:
    """Index ranges (start, end) of the consecutive True values."""
    out, a = [], None
    for k, fl in enumerate(flags):
        if fl and a is None:
            a = k
        if (not fl) and a is not None:
            out.append((a, k - 1))
            a = None
    if a is not None:
        out.append((a, len(flags) - 1))
    return out


def valley_bars_figure(report) -> go.Figure:
    """The free range of every numeric input as a bar over the range the input allows (0 to 100 %), the optimum marked: a wide bar is an input to play with."""
    vs = [v for v in report.valleys if v.kind == "float"]
    vs = sorted(vs, key=lambda v: v.share)
    f = _fig(f"Free range of each input: the loss stays under {report.tol_pct:g} % and every limit is met", "position in the range the input allows (%)", "", max(320, 34 * len(vs) + 130))
    labels = [v.label for v in vs]
    for k, v in enumerate(vs):
        def pct(x, v=v):
            return 100.0 * (x - v.lo) / (v.hi - v.lo) if v.hi > v.lo else 0.0
        f.add_trace(go.Bar(y=[labels[k]], x=[100.0], orientation="h", marker_color="#E8EDF2", hoverinfo="skip", showlegend=False, base=0))
        txt = (f"{v.label}<br>optimum {v.x_star:.4g}{(' ' + v.unit) if v.unit else ''}<br>free from {v.free_lo:.4g} to {v.free_hi:.4g} ({100 * v.share:.0f} % of the range)<br>"
               f"lower end: {v.bind_lo or '-'}<br>upper end: {v.bind_hi or '-'}")
        f.add_trace(go.Bar(y=[labels[k]], x=[max(pct(v.free_hi) - pct(v.free_lo), 0.4)], base=[pct(v.free_lo)], orientation="h", marker_color=VERDICT_COLORS[v.verdict],
                           hovertext=[txt], hoverinfo="text", name=v.verdict, showlegend=False))
        f.add_trace(go.Scatter(y=[labels[k]], x=[pct(v.x_star)], mode="markers", marker=dict(symbol="diamond", size=11, color=HZ["navy"], line=dict(color="white", width=1)),
                               hovertext=[txt], hoverinfo="text", showlegend=False))
    f.update_layout(barmode="overlay", bargap=0.35)
    f.update_xaxes(range=[0, 100])
    for name, col in VERDICT_COLORS.items():                  # a legend for the colours
        f.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=11, color=col, symbol="square"), name=name))
    f.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=11, color=HZ["navy"], symbol="diamond"), name="optimum"))
    f.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), margin=dict(l=10, r=10, t=95, b=10))
    return f


def valley_slices_figure(report, ymax: float | None = None, cols: int = 3) -> go.Figure:
    """One small chart per numeric input: the loss of the weighted objective against the input (the others at the optimum). The valley is the floor under the
    tolerance line; red bands: a limit is broken (or there is no design solution); the diamond: the optimum."""
    vs = [v for v in report.valleys if v.kind == "float"]
    if not vs:
        return _fig("Slices", "", "", 200)
    rows = math.ceil(len(vs) / cols)
    ymax = ymax or max(5.0, 6.0 * report.tol_pct)
    f = make_subplots(rows=rows, cols=cols, subplot_titles=[v.label[:38] for v in vs], vertical_spacing=min(0.16, 0.5 / rows), horizontal_spacing=0.07)
    for k, v in enumerate(vs):
        r, c = k // cols + 1, k % cols + 1
        xs = np.array(v.xs)
        loss = np.array(v.loss, float)
        bad = np.array([(not ok) or (g > 1e-6) for ok, g in zip(v.ok, v.gmax)])
        f.add_trace(go.Scatter(x=xs, y=np.minimum(np.where(np.isfinite(loss), loss, ymax), ymax), mode="lines", line=dict(color=HZ["blue"], width=2), showlegend=False,
                               hovertemplate="%{x:.4g}: loss %{y:.2f} %<extra></extra>"), row=r, col=c)       # first: Plotly skips shapes on a subplot that has no trace yet
        for a, b in _runs(bad):
            x0 = 0.5 * (xs[a - 1] + xs[a]) if a > 0 else xs[a]
            x1 = 0.5 * (xs[b] + xs[b + 1]) if b < len(xs) - 1 else xs[b]
            f.add_vrect(x0=x0, x1=x1, fillcolor="rgba(178,59,59,0.22)", line_width=0, row=r, col=c)
        f.add_hline(y=report.tol_pct, line=dict(color=HZ["orange"], width=1, dash="dash"), row=r, col=c)
        f.add_vrect(x0=v.free_lo, x1=v.free_hi, fillcolor="rgba(46,125,91,0.14)", line_width=0, row=r, col=c)
        f.add_trace(go.Scatter(x=[v.x_star], y=[0.0], mode="markers", marker=dict(symbol="diamond", size=9, color=HZ["navy"]), showlegend=False, hoverinfo="skip"), row=r, col=c)
        f.update_yaxes(range=[min(-0.5, float(np.nanmin(loss)) - 0.3) if np.isfinite(loss).any() else -0.5, ymax], row=r, col=c)
        f.update_xaxes(range=[v.lo, v.hi], row=r, col=c)
    f.update_layout(title=f"Loss of the weighted objective (%) against each input: green = free range, red = a limit is broken, dashed = the {report.tol_pct:g} % tolerance",
                    height=max(320, 230 * rows + 80), template="plotly_white", margin=dict(l=10, r=10, t=80, b=10),
                    font=dict(family="Segoe UI, Arial, sans-serif", size=11, color=HZ["ink"]))
    f.update_xaxes(gridcolor=HZ["line"])
    f.update_yaxes(gridcolor=HZ["line"])
    return f


def valley_surface_figure(surf: dict, lx: str, ly: str, zmax: float | None = None) -> go.Figure:
    """The loss over two inputs as a 3D surface: the valley. Coloured: designs that meet every limit; grey: designs that break one (a wall); orange plane: the
    tolerance; the diamond: the optimum. Holes: no design solution."""
    tol = surf["tol_pct"]
    zmax = zmax or max(5.0, 6.0 * tol)
    loss = np.where(np.isfinite(surf["loss"]), surf["loss"], np.nan)
    z = np.minimum(loss, zmax)
    feas = surf["feasible"]
    z_ok = np.where(feas, z, np.nan)
    z_bad = np.where(~feas & surf["ok"], z, np.nan)
    f = go.Figure()
    f.add_trace(go.Surface(x=surf["x"], y=surf["y"], z=z_ok, colorscale=[[0, "#2E7D5B"], [0.25, "#9CCB86"], [0.55, "#E0A040"], [1, "#B23B3B"]], cmin=0, cmax=zmax,
                           colorbar=dict(title="loss %", len=0.6), hovertemplate=lx + " %{x:.4g}<br>" + ly + " %{y:.4g}<br>loss %{z:.2f} %<extra>meets every limit</extra>", name="meets every limit"))
    f.add_trace(go.Surface(x=surf["x"], y=surf["y"], z=z_bad, colorscale=[[0, "#B8C0CA"], [1, "#B8C0CA"]], showscale=False, opacity=0.55, customdata=surf["why"],
                           hovertemplate=lx + " %{x:.4g}<br>" + ly + " %{y:.4g}<br>loss %{z:.2f} %<br>%{customdata}<extra>breaks a limit</extra>", name="breaks a limit"))
    f.add_trace(go.Surface(x=[surf["x"][0], surf["x"][-1]], y=[surf["y"][0], surf["y"][-1]], z=[[tol, tol], [tol, tol]], colorscale=[[0, "#C2410C"], [1, "#C2410C"]],
                           showscale=False, opacity=0.18, hoverinfo="skip", name="tolerance"))
    f.add_trace(go.Scatter3d(x=[surf["x_star"]], y=[surf["y_star"]], z=[0.0], mode="markers", marker=dict(size=6, color=HZ["navy"], symbol="diamond"), name="optimum",
                             hovertemplate="optimum<extra></extra>"))
    f.update_layout(title=f"The valley: loss of the weighted objective (%) over {lx} and {ly}", height=680, margin=dict(l=0, r=0, t=50, b=0), template="plotly_white",
                    scene=dict(xaxis_title=lx, yaxis_title=ly, zaxis_title="loss (%)", zaxis=dict(range=[min(-0.5, float(np.nanmin(z)) - 0.2) if np.isfinite(z).any() else -0.5, zmax]),
                               camera=dict(eye=dict(x=1.25, y=-1.45, z=0.85), center=dict(x=0, y=0, z=-0.15)), aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=0.6)),
                    font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=HZ["ink"]), showlegend=False)
    return f


def valley_contour_figure(surf: dict, lx: str, ly: str) -> go.Figure:
    """The same valley seen from above: lines of equal loss (the tolerance in orange), grey where a limit is broken (or there is no design solution)."""
    tol = surf["tol_pct"]
    zmax = max(5.0, 6.0 * tol)
    loss = np.where(np.isfinite(surf["loss"]), surf["loss"], np.nan)
    zmin = min(0.0, float(np.nanmin(loss))) if np.isfinite(loss).any() else 0.0
    z_ok = np.where(surf["feasible"], np.minimum(loss, zmax), np.nan)
    bad = np.where(surf["feasible"], 0.0, 1.0)
    f = go.Figure()
    f.add_trace(go.Heatmap(x=surf["x"], y=surf["y"], z=bad, colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(184,192,202,0.95)"]], showscale=False, zmin=0, zmax=1, hoverinfo="skip"))
    f.add_trace(go.Contour(x=surf["x"], y=surf["y"], z=z_ok, colorscale=[[0, "#2E7D5B"], [0.25, "#9CCB86"], [0.55, "#E0A040"], [1, "#B23B3B"]], zmin=zmin, zmax=zmax,
                           contours=dict(start=zmin, end=zmax, size=(zmax - zmin) / 12.0), colorbar=dict(title="loss %"), line=dict(width=0.5, color="rgba(255,255,255,0.6)"),
                           connectgaps=False, hovertemplate=lx + " %{x:.4g}<br>" + ly + " %{y:.4g}<br>loss %{z:.2f} %<extra>meets every limit</extra>"))
    f.add_trace(go.Contour(x=surf["x"], y=surf["y"], z=np.where(np.isfinite(loss), np.minimum(loss, zmax), np.nan), contours=dict(start=tol, end=tol, size=1, coloring="none"),
                           line=dict(color=HZ["orange"], width=3), showscale=False, hoverinfo="skip", name="tolerance"))
    f.add_trace(go.Scatter(x=[surf["x_star"]], y=[surf["y_star"]], mode="markers", marker=dict(symbol="diamond", size=12, color=HZ["navy"], line=dict(color="white", width=1)), name="optimum"))
    f.update_layout(title=f"The valley from above: grey = a limit is broken, orange line = {tol:g} % loss", xaxis_title=lx, yaxis_title=ly, height=520, template="plotly_white",
                    margin=dict(l=10, r=10, t=50, b=10), font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=HZ["ink"]), showlegend=False)
    return f


def combination_radius_figure(cv: dict) -> go.Figure:
    """How far each free combination of inputs can move from the optimum (both senses), as a share of the ranges, before the objective loses the tolerance,
    a limit breaks or a range ends; the sloppiest combination on the left."""
    dirs = cv["directions"]
    x = [f"{i + 1}" for i in range(len(dirs))]
    f = _fig(f"How far each free combination can move: a {cv['tol_pct']:g} % loss, every limit met", "combination (the sloppiest first)", "distance it can move (% of the ranges)", 360)
    f.add_trace(go.Bar(name="in the + sense", x=x, y=[100.0 * d["r_plus"] for d in dirs], marker_color=HZ["blue"],
                       hovertext=[f"{d['text']}<br>ended by: {d['why_plus']}" for d in dirs], hoverinfo="text"))
    f.add_trace(go.Bar(name="in the - sense", x=x, y=[100.0 * d["r_minus"] for d in dirs], marker_color=HZ["orange"],
                       hovertext=[f"{d['text']}<br>ended by: {d['why_minus']}" for d in dirs], hoverinfo="text"))
    f.update_layout(barmode="group", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    f.update_yaxes(range=[0, 105])
    return f


def combination_loadings_figure(cv: dict) -> go.Figure:
    """What the combinations are made of: the weight of every input (scaled to its range) in each free combination (1, 2, ...; sloppiest first) and in the
    direction each active limit blocks (L1, L2, ...)."""
    dirs, act = cv["directions"], cv["active"]
    cols, z, tips = [], [], []
    for i, d in enumerate(dirs):
        cols.append(f"{i + 1}")
        z.append(d["vec"])
        tips.append("free combination")
    for j, a in enumerate(act):
        gr = a["grad"] / a["norm"] if a["norm"] > 1e-9 else a["grad"]
        cols.append(f"L{j + 1}")
        z.append(gr)
        tips.append("blocked by: " + a["name"])
    Z = np.array(z).T if z else np.zeros((len(cv["labels"]), 0))
    f = go.Figure(go.Heatmap(z=Z, x=cols, y=cv["labels"], colorscale="RdBu", zmid=0, zmin=-1, zmax=1, colorbar=dict(title="weight"), customdata=[tips] * len(cv["labels"]),
                             hovertemplate="%{x}: %{customdata}<br>%{y}: %{z:+.2f}<extra></extra>"))
    f.update_layout(title="What the combinations are made of (1, 2, ...: free, sloppiest first; L1, L2, ...: the direction an active limit blocks)", height=max(340, 28 * len(cv["labels"]) + 120),
                    template="plotly_white", margin=dict(l=10, r=10, t=60, b=10), font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=HZ["ink"]))
    f.update_yaxes(autorange="reversed")
    return f
