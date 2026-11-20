"""Numbers for the lift-against-weight sections of the reports (PDF and Excel): the lift the wing generates and the most it could generate, at every
flight point and over speed, and the strip-by-strip working of the lifting line at the design point.

Everything here is read from the model (`Case`); nothing is re-solved except the stall of the other flight points. Plain Python and numpy: no PDF or
spreadsheet code, so both reports use the same numbers.
"""
from __future__ import annotations

import math

import numpy as np

from .points import cases_for, point_list


def mission_lift(case, points=None) -> list[dict]:
    """For every flight point: the lift the wing has to generate (L = n W) against its weight, and the most it could generate (q S CLmax, and with
    the safety factor on lift). Keys: name, label, V, rho, q, mass, W, n, L, S, CL_req, SFL and, when the wing stalls, CLmax, CLmax_sf, Lmax, Lmax_sf,
    nmax, nmax_sf (maximum lift / weight), use (L / Lmax_sf), V_stall, V_stall_sf, margin_sf, eta_stall."""
    out = []
    for i, (pt, cc) in enumerate(zip(point_list(points), cases_for(case, points))):
        S, V = cc.wing.area, cc.V
        q = 0.5 * cc.air.density * V ** 2
        W = cc.weight
        L = cc.n * W
        sfl = cc.sf["lift"]
        d = {"name": pt.name, "label": pt.label(), "V": V, "rho": cc.air.density, "q": q, "mass": cc.p["mass_kg"], "W": W, "n": cc.n, "L": L, "S": S,
             "CL_req": L / (q * S), "SFL": sfl, "altitude": cc.p["altitude_m"]}
        st = cc.full()["stall"] if i == 0 else cc.stall()
        if st is not None:
            cl_max = st["CL"]
            lmax = q * S * cl_max
            d.update({"CLmax": cl_max, "CLmax_sf": cl_max / sfl, "Lmax": lmax, "Lmax_sf": lmax / sfl, "nmax": lmax / W, "nmax_sf": lmax / sfl / W,
                      "use": L / (lmax / sfl), "V_stall": st["V_stall"], "V_stall_sf": st["V_stall"] * math.sqrt(sfl),
                      "margin_sf": V / (st["V_stall"] * math.sqrt(sfl)), "eta_stall": st["eta"]})
        out.append(d)
    return out


def speed_sweep(case, n: int = 22) -> list[dict]:
    """The most lift the wing can carry over speed (the stall boundary of a V-n diagram), for the weight of the design point: speed, dynamic pressure,
    maximum lift as calculated and with the safety factor, and each divided by the weight. CLmax is that of the stall speed (it rises a little with
    speed through the Reynolds number, so this is on the safe side)."""
    st = case.full()["stall"]
    if st is None:
        return []
    vs, sfl = st["V_stall"], case.sf["lift"]
    s, w = case.wing.area, case.weight
    v_hi = max(2.0 * case.V, 2.2 * vs)
    speeds = sorted(set(np.round(np.linspace(0.5 * vs, v_hi, n), 3).tolist()) | {round(vs, 3), round(vs * math.sqrt(sfl), 3), round(case.V, 3)})
    rows = []
    for v in speeds:
        q = 0.5 * case.air.density * v * v
        lmax = q * s * st["CL"]
        rows.append({"V": v, "q": q, "Lmax": lmax, "Lmax_sf": lmax / sfl, "nmax": lmax / w, "nmax_sf": lmax / sfl / w})
    return rows


def strip_data(case) -> dict | None:
    """The strips of the right half-wing (root to tip) at the design point: the working of the lifting line. None when the wing has no design point.

    Keys (arrays over the strips): eta, y, c, dy, w (strip share of the wing area), re, alpha_geo (deg: root angle + twist + flap), alpha_eff (deg),
    alpha_ind (deg: alpha_geo - alpha_eff), cl (the section lift coefficient at alpha_eff, capped), cl_unc, cap, util, cd, gamma (circulation / speed),
    lift (N, the lift of the strip: q c cl dy); and the scalars alpha_root_deg, q, S."""
    r = case.full()
    if not r.get("design_ok"):
        return None
    pt, ll, tab = r["design_point"], case.ll, case.tab
    h = ll.right
    a_geo = np.degrees(math.radians(pt.alpha_deg) + ll.theta + case.d_alpha_ctrl(0.0, None))
    q = 0.5 * case.air.density * case.V ** 2
    d = {"eta": ll.eta[h], "y": ll.ym[h], "c": ll.c[h], "dy": ll.dy[h], "w": ll.weight[h], "re": tab.re[h], "alpha_geo": a_geo[h], "alpha_eff": pt.aeff_deg_j[h],
         "cl": pt.cl_j[h], "cl_unc": pt.cl_unc_j[h], "cap": np.asarray(tab.cap)[h], "util": pt.util_j[h], "cd": pt.cd_j[h], "gamma": pt.gamma[h]}
    d["alpha_ind"] = d["alpha_geo"] - d["alpha_eff"]
    d["lift"] = q * d["c"] * d["cl"] * d["dy"]
    d.update({"alpha_root_deg": pt.alpha_deg, "q": q, "S": case.wing.area, "n_strips": int(ll.n)})
    return d
