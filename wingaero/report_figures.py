"""The charts of the PDF report, drawn with matplotlib (no browser, no Plotly image export: they work on the PC and on the free online server).

Every function returns a PNG as bytes. The design is the first case of a list of (label, Case); a second case (the starting design) is drawn for
comparison. Solid lines are as calculated, dashed lines have the safety factor applied; the colours are those of the app.
"""
from __future__ import annotations

import io
import math
import textwrap

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from . import airfoil as AF

HZ = {"navy": "#1D3043", "ink": "#1F2933", "orange": "#C2410C", "blue": "#2C6596", "amber": "#E0A040", "steel": "#6B7A89", "green": "#2E7D5B", "red": "#B23B3B",
      "mist": "#F3F7FA", "line": "#D8DEE6"}
COLORS = [HZ["blue"], HZ["orange"], HZ["green"], HZ["amber"], HZ["navy"], HZ["steel"]]
WIDTH = 6.85                                                     # inches: the text width of an A4 page with 18 mm margins
RC = {"font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": "#9AA5B1",
      "axes.grid": True, "grid.color": HZ["line"], "grid.linewidth": 0.5, "axes.titleweight": "bold", "axes.titlecolor": HZ["navy"], "lines.linewidth": 1.8}


def _new(h: float, rows: int = 1, cols: int = 1, **kw):
    fig = Figure(figsize=(WIDTH, h), layout="constrained", facecolor="white")
    axes = fig.subplots(rows, cols, **kw)
    return fig, axes


def _png(fig, dpi: int = 170) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    return buf.getvalue()


def _note(fig, text: str) -> None:
    """A line of small print at the bottom of the figure, with room kept for it."""
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    fig.text(0.01, 0.012, text, fontsize=6.3, color=HZ["steel"], va="bottom", ha="left", wrap=True)


def _col(i: int) -> str:
    return COLORS[i % len(COLORS)]


def _style(ax) -> None:
    ax.tick_params(length=2.5, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _right(case) -> np.ndarray:
    return case.ll.right


def _upto_peak(cv: dict) -> int:
    return int(np.argmax(cv["CL"])) + 1


# ------------------------------------------------------------------------------------------------ drawings
def top_view(case) -> bytes:
    """Wing, fuselage, control surfaces, mean aerodynamic chord, aerodynamic centre and the moment reference point, nose up, with the main dimensions."""
    w, p = case.wing, case.p
    from matplotlib import rc_context
    with rc_context({**RC, "axes.grid": False}):
        fig, ax = _new(3.6)
        o = w.outline()
        half = 0.5 * w.span
        x_nose = -p["wing_x_fraction"] * case.fuse.length if case.fuse is not None else 0.0
        ys = np.concatenate([o["y"], o["y"][::-1]])
        xs = np.concatenate([o["x_le"], o["x_te"][::-1]])
        for sgn in (1, -1):
            ax.fill(sgn * ys, -xs, facecolor=HZ["blue"], alpha=0.28, edgecolor=HZ["navy"], linewidth=1.6, label="wing" if sgn == 1 else None, zorder=2)
            ax.plot(sgn * o["y"], -o["x_qc"], color=HZ["steel"], linewidth=0.8, linestyle="--", label="quarter-chord line" if sgn == 1 else None, zorder=3)
        if len(o["y"]) <= 4:
            for k in range(1, len(o["y"]) - 1):
                for sgn in (1, -1):
                    ax.plot([sgn * o["y"][k]] * 2, [-o["x_le"][k], -o["x_te"][k]], color=HZ["steel"], linewidth=0.8, linestyle=":", zorder=3)
        for on, a, b, cf, colr, nm in ((p["flap_on"], p["flap_eta_in"], p["flap_eta_out"], p["flap_chord"], HZ["amber"], "flap"),
                                       (p["aileron_on"], p["aileron_eta_in"], p["aileron_eta_out"], p["aileron_chord"], HZ["orange"], "aileron")):
            if not on:
                continue
            ya, yb = a * half, b * half
            px = [ya, yb, yb, ya]
            py = [-float(w.x_te(a)), -float(w.x_te(b)), -(float(w.x_te(b)) - cf * float(w.chord(b))), -(float(w.x_te(a)) - cf * float(w.chord(a)))]
            for sgn in (1, -1):
                ax.fill([sgn * x for x in px], py, facecolor=colr, alpha=0.75, edgecolor=colr, linewidth=0.8, label=nm if sgn == 1 else None, zorder=4)
        if case.fuse is not None:
            xf, hw = case.fuse.outline()
            xw = xf + x_nose
            ax.fill(np.concatenate([hw, -hw[::-1]]), np.concatenate([-xw, -xw[::-1]]), facecolor=HZ["navy"], alpha=0.55, edgecolor=HZ["navy"], linewidth=1.4, label="fuselage", zorder=5)
        for sgn in (1, -1):
            ax.plot([sgn * w.y_mac] * 2, [-w.x_le_mac, -(w.x_le_mac + w.mac)], color=HZ["orange"], linewidth=3.2, solid_capstyle="butt",
                    label="mean aerodynamic chord" if sgn == 1 else None, zorder=6)
        m = case.full()["metrics"]
        ax.plot([0], [-m["x_ac_m"]], marker="D", color=HZ["green"], markersize=6, markeredgecolor="white", linestyle="none", label="aerodynamic centre (wing alone)", zorder=7)
        ax.plot([0], [-case.x_ref], marker="x", color=HZ["red"], markersize=7, markeredgewidth=2, linestyle="none", label=f"moment reference ({p['x_cg_mac'] * 100:.0f} % MAC)", zorder=7)
        # dimensions
        y_low = min(float(np.min(-o["x_te"])), (-(x_nose + case.fuse.length) if case.fuse is not None else 0.0)) - 0.12 * w.root_chord
        ax.annotate("", xy=(-half, y_low), xytext=(half, y_low), arrowprops=dict(arrowstyle="<->", color=HZ["ink"], linewidth=0.8))
        ax.text(0.5 * half, y_low + 0.015 * w.span, f"span {w.span:.2f} m", ha="center", va="bottom", fontsize=8, color=HZ["ink"])
        ax.text(-half * 1.04, -0.5 * float(o["x_le"][0] + o["x_te"][0]), f"root chord\n{w.root_chord * 1000:.0f} mm", ha="right", va="center", fontsize=7, color=HZ["ink"])
        ax.text(half * 1.04, -0.5 * float(o["x_le"][-1] + o["x_te"][-1]), f"tip chord\n{w.tip_chord * 1000:.0f} mm", ha="left", va="center", fontsize=7, color=HZ["ink"])
        ax.set_aspect("equal")
        ax.set_xlim(-1.25 * half, 1.25 * half)
        ax.set_xlabel("span position (m)")
        ax.set_ylabel("along the fuselage, nose up (m)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
        _style(ax)
        return _png(fig)


def _station_foils(case) -> list:
    names = {"root": "Root", "1": "First break", "2": "Second break", "tip": "Tip"}
    keys = {2: ["root", "tip"], 3: ["root", "1", "tip"], 4: ["root", "1", "2", "tip"]}[len(case.foils)]
    return [(names[k], af, float(eta)) for k, af, eta in zip(keys, case.foils, case.foil_eta)]


def airfoils(case) -> bytes:
    """The section at every station on one chord with its camber line, and the thickness and camber distributions."""
    from matplotlib import rc_context
    with rc_context(RC):
        fig = Figure(figsize=(WIDTH, 5.2), layout="constrained", facecolor="white")
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.6, 1.1])
        ax = fig.add_subplot(gs[0, :])
        axl = fig.add_subplot(gs[1, :])
        at, ac = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
        for i, (st, af, eta) in enumerate(_station_foils(case)):
            sh = af.shape
            up, lo = AF.split_surfaces(af.coords)
            x = np.concatenate([up[:, 0], lo[::-1, 0]])
            y = np.concatenate([up[:, 1], lo[::-1, 1]])
            lab = f"{st} (η {eta:.2f}): {af.label}, t/c {af.thickness * 100:.1f} %, camber {sh['camber_max'] * 100:.1f} % at {sh['x_camber'] * 100:.0f} %"
            ax.plot(x, y, color=_col(i), linewidth=2.0, label=lab)
            if i == 0:
                ax.fill(x, y, color=_col(i), alpha=0.10)
            ax.plot(sh["x"], sh["camber"], color=_col(i), linewidth=0.9, linestyle=":")
            at.plot(sh["x"], sh["thickness"], color=_col(i), label=st)
            ac.plot(sh["x"], sh["camber"], color=_col(i), label=st)
        ax.set_aspect("equal")
        ax.set_title("Airfoil at each station (chord = 1), camber line dotted")
        ax.set_xlabel("x / c")
        ax.set_ylabel("y / c")
        h, lab = ax.get_legend_handles_labels()
        axl.axis("off")
        axl.legend(h, lab, loc="center", ncol=1, frameon=False)
        at.set_title("Thickness distribution")
        at.set_xlabel("x / c")
        at.set_ylabel("t / c")
        ac.set_title("Camber line")
        ac.set_xlabel("x / c")
        ac.set_ylabel("camber / c")
        for a in (ax, at, ac):
            _style(a)
        return _png(fig)


# ------------------------------------------------------------------------------------------------ wing charts
def _sf_note(cases) -> str:
    c = cases[0][1]
    f = c.sf
    return (f"Solid: as calculated. Dashed: safety factor {c.p['safety_factor']:.2f} applied (drag x{f['drag']:.2f}, lift /{f['lift']:.2f}, "
            f"moments x{f['moment']:.2f}, control /{f['control']:.2f}).")


def lift_polar(cases) -> bytes:
    """Wing lift curve, drag polar and lift-to-drag against lift."""
    from matplotlib import rc_context
    with rc_context(RC):
        fig, (a, b, c3) = _new(2.9, 1, 3)
        for i, (lab, c) in enumerate(cases):
            r, col = c.full(), _col(i)
            cv, m = r["curve"], r["metrics"]
            n = _upto_peak(cv)
            a.plot(cv["alpha_deg"], cv["CL"], color=col, label=lab)
            if "CL_max_first" in m:
                a.plot([m["alpha_stall_first_deg"]], [m["CL_max_first"]], marker="v", color=col, markersize=6, linestyle="none")
                a.axhline(m["CL_max_first"] / c.sf["lift"], color=col, linestyle="--", linewidth=1.0)
            if r.get("design_ok"):
                pt = r["design_point"]
                a.plot([pt.alpha_deg], [pt.CL], marker="o", color=col, markersize=6, markeredgecolor="white", linestyle="none")
                b.plot([pt.CD], [pt.CL], marker="o", color=col, markersize=6, markeredgecolor="white", linestyle="none")
                c3.plot([pt.CL], [pt.CL / pt.CD], marker="o", color=col, markersize=6, markeredgecolor="white", linestyle="none")
            b.plot(cv["CD"][:n], cv["CL"][:n], color=col)
            b.plot(cv["CD"][:n] * c.sf["drag"], cv["CL"][:n], color=col, linestyle="--", linewidth=1.0)
            ok = cv["CL"][:n] > 0
            c3.plot(cv["CL"][:n][ok], (cv["CL"] / cv["CD"])[:n][ok], color=col)
            c3.plot(cv["CL"][:n][ok], (cv["CL"] / (cv["CD"] * c.sf["drag"]))[:n][ok], color=col, linestyle="--", linewidth=1.0)
        a.set_title("Wing lift curve")
        a.set_xlabel("root angle of attack (°)")
        a.set_ylabel("CL")
        b.set_title("Drag polar")
        b.set_xlabel("CD")
        b.set_ylabel("CL")
        c3.set_title("Lift-to-drag")
        c3.set_xlabel("CL")
        c3.set_ylabel("CL / CD")
        if len(cases) > 1:
            a.legend(loc="lower right", frameon=False)
        for ax in (a, b, c3):
            _style(ax)
        _note(fig, _sf_note(cases) + "  Triangle: first strip stalls; circle: design point.")
        return _png(fig)


def drag_breakdown(case) -> bytes:
    from matplotlib import rc_context
    with rc_context(RC):
        fig, (a, b) = _new(2.8, 1, 2, gridspec_kw={"width_ratios": [2.2, 1]})
        r = case.full()
        cv, m = r["curve"], r["metrics"]
        n = _upto_peak(cv)
        parts = (("CDp", "profile (sections, strip by strip)", HZ["blue"]), ("CDi", "induced (lifting line)", HZ["orange"]), ("CDf", "fuselage", HZ["steel"]))
        a.stackplot(cv["CL"][:n], *[cv[k][:n] for k, _, _ in parts], labels=[nm for _, nm, _ in parts], colors=[c for _, _, c in parts], alpha=0.85)
        if "CL_design" in m and r.get("design_ok"):
            a.axvline(m["CL_design"], color=HZ["navy"], linewidth=1.0, linestyle=":")
            a.text(m["CL_design"], a.get_ylim()[1] * 0.97, " design point", rotation=90, va="top", ha="right", fontsize=7, color=HZ["navy"])
        a.set_title("Drag breakdown against lift")
        a.set_xlabel("CL")
        a.set_ylabel("CD")
        a.legend(loc="upper left", frameon=False)
        if r.get("design_ok"):
            vals = [m["CDp"], m["CDi"], m["CDf"]]
            tot = sum(vals)
            left = 0.0
            for (k, nm, col), v in zip(parts, vals):
                b.barh([0], [100 * v / tot], left=left, color=col, height=0.22)
                if v / tot > 0.07:
                    b.text(left + 50 * v / tot, 0, f"{100 * v / tot:.0f} %", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
                left += 100 * v / tot
            b.set_xlim(0, 100)
            b.set_ylim(-0.5, 0.5)
            b.set_yticks([])
            b.set_xlabel("share of CD at the design point (%)")
            b.set_title(f"CD = {tot:.4f}")
            b.grid(False)
        for ax in (a, b):
            _style(ax)
        return _png(fig)


def moment(cases) -> bytes:
    from matplotlib import rc_context
    with rc_context(RC):
        fig, (a, b) = _new(3.0, 1, 2)
        for i, (lab, c) in enumerate(cases):
            r, col = c.full(), _col(i)
            cv, m = r["curve"], r["metrics"]
            n = _upto_peak(cv)
            for key, ls, lw, nm in (("Cm_ref", "-", 1.9, f"{lab}: about the reference point ({c.p['x_cg_mac'] * 100:.0f} % MAC)"), ("Cm_ac", ":", 1.3, f"{lab}: about the aerodynamic centre")):
                a.plot(cv["CL"][:n], cv[key][:n], color=col, linestyle=ls, linewidth=lw, label=nm)
                b.plot(cv["alpha_deg"][:n], cv[key][:n], color=col, linestyle=ls, linewidth=lw)
            a.plot(cv["CL"][:n], cv["Cm_ref"][:n] * c.sf["moment"], color=col, linestyle="--", linewidth=1.0)
            if "CL_max_first" in m:
                a.axvline(m["CL_max_first"], color=col, linestyle="-.", linewidth=0.8)
            if r.get("design_ok"):
                a.plot([m["CL_design"]], [m["Cm_ref"]], marker="o", color=col, markersize=6, markeredgecolor="white", linestyle="none")
        a.set_title("Pitching moment against lift")
        a.set_xlabel("CL")
        a.set_ylabel("Cm")
        b.set_title("Pitching moment against angle of attack")
        b.set_xlabel("root angle of attack (°)")
        a.legend(loc="best", frameon=False, fontsize=6.3)
        for ax in (a, b):
            _style(ax)
        _note(fig, _sf_note(cases) + "  Dash-dot vertical: first strip stalls.")
        return _png(fig)


def span_state(case) -> bytes:
    """Section lift against the section maximum along the span (the stall map), and the state of the strips at the design point."""
    from matplotlib import rc_context
    with rc_context(RC):
        fig = Figure(figsize=(WIDTH, 6.0), layout="constrained", facecolor="white")
        gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])
        amap = fig.add_subplot(gs[0, :])
        a1, a2, a3 = (fig.add_subplot(gs[1, k]) for k in range(3))
        r = case.full()
        cv, m = r["curve"], r["metrics"]
        h = _right(case)
        eta = case.ll.eta[h]
        tab = case.tab
        amap.plot(eta, tab.cap[h], color=HZ["red"], linewidth=2.6, label="section maximum (cap)")
        amap.plot(eta, tab.cap[h] / case.sf["lift"], color=HZ["red"], linewidth=1.5, linestyle="--", label="allowed with the safety factor")
        cls = m.get("CL_max_first", float(cv["CL"].max()))
        pk = int(np.argmax(cv["CL"])) + 1
        for k, (nm, fr) in enumerate((("0.4", 0.4), ("0.6", 0.6), ("0.8", 0.8), ("0.9", 0.9), ("1.0 (first strip stalls)", 1.0))):
            i = int(np.argmin(np.abs(cv["CL"][:pk] - fr * cls)))
            amap.plot(eta, cv["cl_j"][i][h], color=COLORS[(k + 1) % len(COLORS)], label=f"{nm} CLmax: CL {cv['CL'][i]:.2f}")
        if r.get("design_ok"):
            amap.plot(eta, r["design_point"].cl_j[h], color=HZ["navy"], linewidth=2.4, linestyle=":", label=f"design point: CL {m['CL_design']:.2f}")
        if "eta_stall_first" in m:
            amap.axvline(m["eta_stall_first"], color=HZ["amber"], linestyle=":", linewidth=1.4)
            amap.text(m["eta_stall_first"], amap.get_ylim()[1] * 0.98, " stall starts", color=HZ["amber"], fontsize=7, va="top")
        for on, a_, b_, nm, colr in ((case.p["aileron_on"], case.p["aileron_eta_in"], case.p["aileron_eta_out"], "aileron", HZ["orange"]),
                                     (case.p["flap_on"], case.p["flap_eta_in"], case.p["flap_eta_out"], "flap", HZ["amber"])):
            if on:
                amap.axvspan(a_, b_, color=colr, alpha=0.12)
                amap.text(0.5 * (a_ + b_), amap.get_ylim()[0] + 0.02, nm, ha="center", va="bottom", fontsize=7, color=colr)
        amap.set_title("Section lift against the section maximum along the span")
        amap.set_xlabel("η = y / (b/2)")
        amap.set_ylabel("section cl")
        amap.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, fontsize=6.5)
        a1.plot(eta, case.ll.c[h] * 1000.0, color=HZ["navy"], label="chord (mm)")
        a1.set_ylabel("chord (mm)", color=HZ["navy"])
        a1b = a1.twinx()
        a1b.plot(eta, tab.re[h] / 1e3, color=HZ["orange"])
        a1b.set_ylabel("Reynolds number (thousand)", color=HZ["orange"])
        a1b.grid(False)
        a1.set_title("Chord and Reynolds number")
        if r.get("design_ok"):
            pt = r["design_point"]
            a2.plot(eta, pt.aeff_deg_j[h], color=HZ["blue"], label="effective angle")
            a2.plot(eta, tab.alpha_cap_deg[h], color=HZ["red"], linestyle="--", label="at the section maximum")
            a2.legend(loc="best", frameon=False, fontsize=6.3)
            a3.plot(eta, pt.cl_j[h] * case.ll.c[h] / case.wing.mac, color=HZ["green"])
            a3b = a3.twinx()
            a3b.plot(eta, pt.cd_j[h], color=HZ["steel"], linestyle="--")
            a3b.set_ylabel("section cd (dashed)", color=HZ["steel"])
            a3b.grid(False)
            a3.set_ylabel("cl·c / MAC", color=HZ["green"])
        a2.set_title("Effective and stall angle (°)")
        a3.set_title("Loading and section drag")
        for ax in (amap, a1, a2, a3):
            ax.set_xlabel("η")
            _style(ax)
        return _png(fig)


def utilisation(case) -> bytes:
    from matplotlib import rc_context
    with rc_context(RC):
        fig, ax = _new(3.3)
        r = case.full()
        cv, m = r["curve"], r["metrics"]
        h = _right(case)
        n = _upto_peak(cv)
        z = np.minimum(cv["util_j"][:n][:, h] * case.sf["lift"], 1.6)
        cmap = LinearSegmentedColormap.from_list("util", [(0, "#F3F7FA"), (0.5, "#2C6596"), (0.62, "#E0A040"), (0.625, "#B23B3B"), (1, "#7a1f1f")])
        order = np.argsort(cv["CL"][:n])
        mesh = ax.pcolormesh(case.ll.eta[h], cv["CL"][:n][order], z[order], cmap=cmap, vmin=0, vmax=1.6, shading="auto")
        cb = fig.colorbar(mesh, ax=ax, pad=0.01)
        cb.set_label("SF · cl / cl max")
        if r.get("design_ok"):
            ax.axhline(m["CL_design"], color="white", linestyle=":", linewidth=1.2)
            ax.text(0.99, m["CL_design"], "design point ", color="white", ha="right", va="bottom", fontsize=7)
        ax.set_title("Strip utilisation (red: beyond the allowed lift with the safety factor)")
        ax.set_xlabel("η = y / (b/2)")
        ax.set_ylabel("wing CL")
        ax.grid(False)
        _style(ax)
        return _png(fig)


def control(cases) -> bytes | None:
    """Roll power of the ailerons against lift; None when no case has ailerons."""
    cs = [(i, lab, c) for i, (lab, c) in enumerate(cases) if c.p["aileron_on"]]
    if not cs:
        return None
    from matplotlib import rc_context
    with rc_context(RC):
        fig, (a, b) = _new(2.9, 1, 2)
        for i, lab, c in cs:
            r, col = c.full(), _col(i)
            cv, m = r["curve"], r["metrics"]
            n = _upto_peak(cv)
            a.plot(cv["CL"][:n], cv["ail_small"][:n], color=col, label=lab)
            a.plot(cv["CL"][:n], cv["ail_small"][:n] / c.sf["control"], color=col, linestyle="--", linewidth=1.0)
            b.plot(cv["alpha_deg"][:n], cv["ail_max"][:n], color=col)
            b.plot(cv["alpha_deg"][:n], cv["ail_max"][:n] / c.sf["control"], color=col, linestyle="--", linewidth=1.0)
            if "CL_max_first" in m:
                a.axvline(m["CL_max_first"], color=col, linestyle="-.", linewidth=0.8)
            if not math.isnan(m.get("control_loss_CL", float("nan"))):
                a.axvline(m["control_loss_CL"], color=col, linestyle=":", linewidth=1.0)
        a.axhline(0, color=HZ["steel"], linewidth=0.7)
        a.set_title("Roll power of the ailerons against lift")
        a.set_xlabel("wing CL")
        a.set_ylabel("Cl per rad of aileron")
        b.set_title("Roll moment at full deflection")
        b.set_xlabel("root angle of attack (°)")
        b.set_ylabel("Cl (full deflection)")
        if len(cs) > 1:
            a.legend(loc="best", frameon=False)
        for ax in (a, b):
            _style(ax)
        _note(fig, _sf_note(cases) + "  Dash-dot: first strip stalls; dotted: half the roll power lost.")
        return _png(fig)


def performance(cases) -> bytes:
    from matplotlib import rc_context
    with rc_context(RC):
        fig, (a, b) = _new(3.0, 1, 2)
        for i, (lab, c) in enumerate(cases):
            r, col = c.full(), _col(i)
            pf, m = r.get("performance", {}), r["metrics"]
            if not len(pf.get("V", [])):
                continue
            a.plot(pf["V"], pf["P"], color=col, label=lab)
            a.plot(pf["V"], pf["P"] * c.sf["drag"], color=col, linestyle="--", linewidth=1.0)
            b.plot(pf["V"], pf["LD"], color=col)
            b.plot(pf["V"], pf["LD"] / c.sf["drag"], color=col, linestyle="--", linewidth=1.0)
            if "V_stall_mps" in m:
                a.axvline(m["V_stall_mps"], color=col, linestyle="-.", linewidth=0.8)
                a.axvline(m["V_stall_mps"] * math.sqrt(c.sf["lift"]), color=col, linestyle=":", linewidth=1.0)
            a.axvline(m["V_design_mps"], color=col, linewidth=0.9)
        a.set_title("Power required against speed")
        a.set_xlabel("speed (m/s)")
        a.set_ylabel("aerodynamic power D·V (W)")
        b.set_title("Lift-to-drag against speed")
        b.set_xlabel("speed (m/s)")
        b.set_ylabel("CL / CD")
        if len(cases) > 1:
            a.legend(loc="best", frameon=False)
        for ax in (a, b):
            _style(ax)
        _note(fig, _sf_note(cases) + "  Vertical: dash-dot stall speed, dotted stall speed with the safety factor, solid design speed.")
        return _png(fig)


def airfoil_sections(case) -> bytes:
    """Section data of every airfoil station at its Reynolds number at the design speed."""
    from matplotlib import rc_context
    with rc_context(RC):
        fig, axes = _new(5.6, 2, 2)
        (a, b), (c3, d) = axes
        sf = case.sf
        shown = set()
        for i, (st, af, eta) in enumerate(_station_foils(case)):
            re = float(case.air.density * case.V * case.wing.chord(eta) / case.air.viscosity)
            key = (af.name, round(re, -3))
            if key in shown:
                continue
            shown.add(key)
            col = _col(i)
            lab = f"{st}: {af.label}, Re {re / 1e3:,.0f} k"
            pol = af.polar(re)
            al, pre = pol["alpha_deg"], pol["pre_stall"]
            cl, cd, cm = pol["cl"], pol["cd"], pol["cm"]
            a.plot(al[pre], cl[pre], color=col, label=lab)
            a.plot(al[~pre], cl[~pre], color=col, linestyle=":", linewidth=1.2)
            a.plot(al, np.minimum(cl, pol["cl_max"] / sf["lift"]), color=col, linestyle="--", linewidth=0.9)
            b.plot(cd[pre], cl[pre], color=col)
            b.plot(cd[pre] * sf["drag"], cl[pre], color=col, linestyle="--", linewidth=0.9)
            c3.plot(al[pre], cl[pre] / cd[pre], color=col)
            c3.plot(al[pre], cl[pre] / (cd[pre] * sf["drag"]), color=col, linestyle="--", linewidth=0.9)
            d.plot(al[pre], cm[pre], color=col)
            d.plot(al[pre], cm[pre] * sf["moment"], color=col, linestyle="--", linewidth=0.9)
        a.set_title("Lift curve")
        a.set_xlabel("angle of attack (°)")
        a.set_ylabel("cl")
        b.set_title("Drag polar")
        b.set_xlabel("cd")
        b.set_ylabel("cl")
        c3.set_title("Section glide ratio")
        c3.set_xlabel("angle of attack (°)")
        c3.set_ylabel("cl / cd")
        d.set_title("Pitching moment about the quarter chord")
        d.set_xlabel("angle of attack (°)")
        d.set_ylabel("cm")
        a.legend(loc="lower right", frameon=False, fontsize=6.2)
        for ax in (a, b, c3, d):
            _style(ax)
        _note(fig, "NeuralFoil sections. Solid: as calculated (dotted beyond the stall); dashed: safety factor applied.")
        return _png(fig)


def flap_effect(case) -> bytes | None:
    c = case
    if not c.p["flap_on"] or c.flap_defl <= 0:
        return None
    from matplotlib import rc_context
    with rc_context(RC):
        fig, ax = _new(2.8)
        cv1 = c.full()["curve"]
        cv0 = c.curve(c.tables(c.V, 0.0), alpha_min=float(cv1["alpha_deg"][0]), with_roll=False)
        ax.plot(cv0["alpha_deg"], cv0["CL"], color=HZ["steel"], label="flap up")
        ax.plot(cv1["alpha_deg"], cv1["CL"], color=HZ["orange"], label=f"flap {c.flap_defl:.0f}° down")
        ax.set_title("Effect of the flap (thin-airfoil theory with viscous factors: lower confidence)")
        ax.set_xlabel("root angle of attack (°)")
        ax.set_ylabel("CL")
        ax.legend(frameon=False)
        _style(ax)
        return _png(fig)


# ------------------------------------------------------------------------------------------------ optimiser and valleys
def history(values: list) -> bytes:
    from matplotlib import rc_context
    with rc_context(RC):
        fig, ax = _new(2.4)
        ax.plot(range(1, len(values) + 1), values, color=HZ["orange"], marker="o", markersize=3)
        ax.set_title("Optimiser progress")
        ax.set_xlabel("generation")
        ax.set_ylabel("best weighted objective (lower is better)")
        _style(ax)
        return _png(fig)


VERDICT_COLORS = {"free": "#2E7D5B", "loose": "#7BAE7F", "tight": "#E0A040", "critical": "#B23B3B"}


def valley_bars(report) -> bytes | None:
    vs = sorted((v for v in report.valleys if v.kind == "float"), key=lambda v: v.share)
    if not vs:
        return None
    from matplotlib import rc_context
    with rc_context({**RC, "axes.grid": False}):
        fig, ax = _new(max(2.2, 0.36 * len(vs) + 1.3))
        for k, v in enumerate(vs):
            span = v.hi - v.lo
            lo = 100 * (v.free_lo - v.lo) / span if span > 0 else 0.0
            hi = 100 * (v.free_hi - v.lo) / span if span > 0 else 100.0
            ax.barh([k], [100], color="#E8EDF2", height=0.62)
            ax.barh([k], [max(hi - lo, 0.5)], left=lo, color=VERDICT_COLORS[v.verdict], height=0.62)
            ax.plot([100 * (v.x_star - v.lo) / span if span > 0 else 0], [k], marker="D", color=HZ["navy"], markersize=5, markeredgecolor="white")
        ax.set_yticks(range(len(vs)))
        ax.set_yticklabels([v.label[:42] for v in vs])
        ax.set_xlim(0, 100)
        ax.set_xlabel("position in the range the input allows (%)")
        ax.set_title(f"Free range of each input: the loss stays under {report.tol_pct:g} % and every limit is met")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=n) for n, c in VERDICT_COLORS.items()], loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
        _style(ax)
        return _png(fig)


def valley_slices(report, cols: int = 3) -> bytes | None:
    """One small chart per numeric input: the loss against the input with the others at the optimum; green = the free range, red = a limit is broken."""
    vs = [v for v in report.valleys if v.kind == "float"]
    if not vs:
        return None
    from matplotlib import rc_context
    rows = math.ceil(len(vs) / cols)
    ymax = max(5.0, 6.0 * report.tol_pct)
    with rc_context(RC):
        fig, axes = _new(1.75 * rows + 0.5, rows, cols, squeeze=False)
        for k, v in enumerate(vs):
            ax = axes[k // cols][k % cols]
            xs = np.array(v.xs)
            loss = np.array(v.loss, float)
            bad = np.array([(not ok) or (g > 1e-6) for ok, g in zip(v.ok, v.gmax)])
            ax.axvspan(v.free_lo, v.free_hi, color=HZ["green"], alpha=0.14, linewidth=0)
            run = None
            for j, fl in enumerate(list(bad) + [False]):
                if fl and run is None:
                    run = j
                if (not fl) and run is not None:
                    x0 = 0.5 * (xs[run - 1] + xs[run]) if run > 0 else xs[run]
                    x1 = 0.5 * (xs[j - 1] + xs[j]) if j < len(xs) else xs[j - 1]
                    ax.axvspan(x0, x1, color=HZ["red"], alpha=0.20, linewidth=0)
                    run = None
            ax.plot(xs, np.minimum(np.where(np.isfinite(loss), loss, ymax), ymax), color=HZ["blue"], linewidth=1.5)
            ax.axhline(report.tol_pct, color=HZ["orange"], linestyle="--", linewidth=0.9)
            ax.plot([v.x_star], [0], marker="D", color=HZ["navy"], markersize=4.5, linestyle="none")
            ax.set_xlim(v.lo, v.hi)
            lo_y = min(-0.5, float(np.nanmin(loss)) - 0.3) if np.isfinite(loss).any() else -0.5
            ax.set_ylim(lo_y, ymax)
            ax.set_title(v.label[:34], fontsize=7)
            _style(ax)
        for k in range(len(vs), rows * cols):
            axes[k // cols][k % cols].axis("off")
        _note(fig, f"Loss of the weighted objective (%) against each input: green = free range, red = a limit is broken, dashed = the {report.tol_pct:g} % tolerance.")
        return _png(fig)


# ------------------------------------------------------------------------------------------------ lift against weight
def lift_vs_weight(case, mission: list) -> bytes:
    """The lift the wing can carry against its weight along the speed range (the stall boundary of a V-n diagram), with the points of the mission marked, and for every point
    the lift it must generate against the maximum lift it has there."""
    from matplotlib import rc_context
    r = case.full()
    m = r["metrics"]
    W0, S = case.weight, case.wing.area
    sfl = case.sf["lift"]
    with rc_context(RC):
        fig, (a, b) = _new(3.3, 1, 2, gridspec_kw={"width_ratios": [1.45, 1]})
        vd = case.V
        v_stall_sf = m.get("V_stall_mps", 0.0) * math.sqrt(sfl)
        v = np.linspace(0.35 * min(vd, v_stall_sf or vd), max(2.0 * vd, 1.3 * v_stall_sf, 1.3 * max(q["V"] for q in mission)), 160)
        if "CL_max_first" in m:
            n_calc = 0.5 * case.air.density * v ** 2 * S * m["CL_max_first"] / W0
            a.plot(v, n_calc, color=HZ["blue"], label="maximum lift ÷ weight, as calculated")
            a.plot(v, n_calc / sfl, color=HZ["blue"], linestyle="--", label=f"with the safety factor ({sfl:.2f})")
            ymax = max(3.2, 1.15 * max(float(np.interp(1.6 * vd, v, n_calc / sfl)), max((q["L"] / W0 for q in mission), default=1.0)))
            a.fill_between(v, n_calc / sfl, ymax, color=HZ["red"], alpha=0.07, label="beyond the usable lift")
            a.set_ylim(0, ymax)
            vs, vs_sf = m["V_stall_mps"], m["V_stall_mps"] * math.sqrt(sfl)
            for x, lab in ((vs, "stall"), (vs_sf, "stall with SF")):
                a.axvline(x, color=HZ["steel"], linestyle=":", linewidth=1.0)
                a.text(x, ymax * 0.97, f" {lab} {x:.1f} m/s", rotation=90, va="top", ha="right", fontsize=6.5, color=HZ["steel"])
        a.axhline(1.0, color=HZ["navy"], linewidth=1.2)
        a.text(v[-1], 1.0, "weight (lift = weight) ", ha="right", va="bottom", fontsize=7, color=HZ["navy"])
        for i, q in enumerate(mission):
            a.plot([q["V"]], [q["L"] / W0], marker="o", color=_col(i + 1), markersize=7, markeredgecolor="white", linestyle="none")
            a.annotate(q["name"], (q["V"], q["L"] / W0), textcoords="offset points", xytext=(6, 5 + 4 * (i % 2)), fontsize=7, color=_col(i + 1))
        a.set_xlabel("speed (m/s)")
        a.set_ylabel("lift ÷ design weight (load factor)")
        a.set_title("Lift the wing can carry against its weight")
        a.legend(loc="lower right", frameon=False, fontsize=6.3)
        have = [q for q in mission if "Lmax_sf" in q]
        x = np.arange(len(have))
        w = 0.36
        b.bar(x - w / 2, [q["L"] for q in have], w, color=HZ["navy"], label="lift needed (n × weight)")
        b.bar(x + w / 2, [q["Lmax_sf"] for q in have], w, color=HZ["blue"], alpha=0.75, label="maximum lift, with SF")
        for xi, q in zip(x, have):
            b.text(xi + w / 2, q["Lmax_sf"], f"{q['Lmax_sf'] / q['W']:.1f}×", ha="center", va="bottom", fontsize=6.5, color=HZ["blue"])
            b.text(xi - w / 2, q["L"], f"{q['L'] / q['W']:.1f}×", ha="center", va="bottom", fontsize=6.5, color=HZ["navy"])
        b.set_xticks(x)
        b.set_xticklabels([textwrap.fill(q["name"], 11) for q in have], fontsize=6.5)
        b.set_ylabel("lift (N); labels: × the weight")
        b.set_title("At each point of the mission")
        b.legend(loc="upper left", frameon=False, fontsize=6.3)
        b.set_ylim(0, 1.25 * max([q["Lmax_sf"] for q in have] + [q["L"] for q in have] + [1.0]))        # room for the labels of the tallest bar (text is not clipped)
        for ax in (a, b):
            _style(ax)
        return _png(fig)
