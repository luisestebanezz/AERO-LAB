"""Wing analysis. A Case is one resolved input set in one flight condition; it builds the planform, the section tables of every
strip, the lifting line, and evaluates points, curves, the stall, the pitching moment, the roll control and the performance.

Everything is non-linear (section polars up to 22 degrees), incompressible, planar, steady. The tail, propeller and
other installation effects are not part of the lab; the fuselage enters only through its drag and the induced-drag
factor, both switchable.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from . import airfoil as AF
from .core import G0, bisect, golden_min, isa
from .geometry import broadcast_stations, build_fuselage, build_wing, wing_area
from .lifting_line import LiftingLine, SectionTables

TWO_PI = 2.0 * math.pi


# ------------------------------------------------------------------------------------------------ control surfaces
def plain_flap_theory(cf: float) -> tuple[float, float]:
    """Thin-airfoil plain flap of chord share cf: lift effectiveness d(alpha_0)/d(delta) and quarter-chord moment slope
    d(cm)/d(delta) per radian (Glauert; cos(theta_f) = 2 cf - 1, tau = [(pi - theta_f) + sin(theta_f)]/pi,
    cm_delta = -(1/2) sin(theta_f)(1 - cos(theta_f)))."""
    cf = min(max(cf, 0.01), 0.99)
    th = math.acos(2.0 * cf - 1.0)
    return ((math.pi - th) + math.sin(th)) / math.pi, -0.5 * math.sin(th) * (1.0 - math.cos(th))


def deflection_factor(delta_deg: float, lin_limit_deg: float, eff_25: float) -> float:
    """Loss of effectiveness of a deflected plain surface: 1 up to lin_limit, then linear to eff_25 at 25 degrees,
    and on at the same slope down to a floor of 0.4 (engineering judgement STB-38/39; not read from a chart)."""
    d = abs(delta_deg)
    if d <= lin_limit_deg:
        return 1.0
    slope = (1.0 - eff_25) / max(25.0 - lin_limit_deg, 1e-6)
    return max(1.0 - slope * (d - lin_limit_deg), 0.4)


def cf_turbulent(re: float) -> float:
    return 0.455 / (math.log10(max(re, 1e4)) ** 2.58)


def cf_laminar(re: float) -> float:
    return 1.328 / math.sqrt(max(re, 1e3))


@dataclass
class Point:
    alpha_deg: float
    CL: float
    CD: float
    cdp: float
    cdi: float
    cdf: float
    cm_ref: float
    cm_ac: float
    cl_roll: float
    cl_j: np.ndarray
    cl_unc_j: np.ndarray
    cd_j: np.ndarray
    aeff_deg_j: np.ndarray
    util_j: np.ndarray
    stalled_j: np.ndarray
    gamma: np.ndarray
    converged: bool

    @property
    def util_max(self) -> float:
        return float(np.max(self.util_j))

    @property
    def ld(self) -> float:
        return self.CL / self.CD


class Case:
    def __init__(self, p: dict, name: str = ""):
        self.p, self.name = p, name
        self.air = isa(p["altitude_m"], p["dT_K"])
        self.weight = p["mass_kg"] * G0
        self.wing = build_wing(p)
        self.fuse = build_fuselage(p)
        self.n = float(p["load_factor"])
        self.V = self._design_speed()
        self.ll = LiftingLine(self.wing)
        self._setup_foils()
        self.x_ref = self.wing.x_le_mac + p["x_cg_mac"] * self.wing.mac
        d = self.fuse.diameter if (self.fuse is not None and p["body_factor_on"]) else 0.0
        self.body_factor = 1.0 - 2.0 * (d / self.wing.span) ** 2
        self.sf = safety_factors(p)
        self.flap_mask = self.span_weight(p["flap_eta_in"], p["flap_eta_out"]) if p["flap_on"] else np.zeros(self.ll.n)
        self.ail_mask = self.span_weight(p["aileron_eta_in"], p["aileron_eta_out"]) if p["aileron_on"] else np.zeros(self.ll.n)
        self.flap_defl = float(p["flap_defl_deg"]) if p["flap_on"] else 0.0
        self._tabs: dict = {}
        self._full = None
        self.tab = self.tables(self.V)

    def _design_speed(self) -> float:
        """The design speed: the one given, or (with a design lift coefficient set) the one that carries n x the weight at that coefficient."""
        if self.p["design_cl"] is not None:
            return math.sqrt(2.0 * self.n * self.weight / (self.air.density * self.wing.area * float(self.p["design_cl"])))
        return float(self.p["speed_mps"])

    def variant(self, p_new: dict, name: str = "") -> "Case":
        """The same wing in another flight condition (speed, altitude, mass, load factor changed in p_new). The planform, the lifting line, the
        airfoils and the control surfaces are shared, so this costs a section-table build (a few ms) instead of a whole Case."""
        c = copy.copy(self)
        c.p, c.name = p_new, name
        c.air = isa(p_new["altitude_m"], p_new["dT_K"])
        c.weight = p_new["mass_kg"] * G0
        c.n = float(p_new["load_factor"])
        c.V = c._design_speed()
        c._tabs, c._full = {}, None
        c.tab = c.tables(c.V)
        return c

    def span_weight(self, eta_in: float, eta_out: float) -> np.ndarray:
        """Share (0 to 1) of every strip covered by a control surface between two span stations, smoothed over one local chord:
        a sharp end of a flap or an aileron puts a step in the circulation, and a lifting line answers a step with an almost
        singular induced velocity on the next strip; in the real flow the end effect is spread over about a chord."""
        half = 0.5 * self.wing.span
        y, c = np.abs(self.ll.ym), self.ll.c
        lo, hi = y - 0.5 * c, y + 0.5 * c
        ov = np.maximum(0.0, np.minimum(eta_out * half, hi) - np.maximum(eta_in * half, lo))
        return np.clip(ov / c, 0.0, 1.0)

    # ---- sections along the span
    def _setup_foils(self):
        p = self.p
        s = broadcast_stations(p)
        names = []
        for k in s["foil_keys"]:
            nm = p[f"foil_{k}"]
            if nm == AF.PARAM:
                nm = AF.parametric_name(p[f"naca_m_{k}"], p[f"naca_p_{k}"], p[f"naca_t_{k}"])
            names.append(nm)
        nc, xu, xl = round(p["n_crit"] * 2) / 2, round(p["xtr_upper"] * 20) / 20, round(p["xtr_lower"] * 20) / 20
        self.foil_names = names
        self.foils = [AF.get_airfoil(nm, nc, xu, xl) for nm in names]
        fe = s["foil_eta"]
        self.foil_eta = fe
        eta = self.ll.eta
        k = np.clip(np.searchsorted(fe, eta, side="right") - 1, 0, len(fe) - 2)
        t = np.clip((eta - fe[k]) / (fe[k + 1] - fe[k]), 0.0, 1.0)
        self.W = np.zeros((self.ll.n, len(fe)))
        self.W[np.arange(self.ll.n), k] = 1.0 - t
        self.W[np.arange(self.ll.n), k + 1] = t

    def re_strips(self, v: float) -> np.ndarray:
        return self.air.density * v * self.ll.c / self.air.viscosity

    def tables(self, v: float, flap_defl: float | None = None) -> SectionTables:
        """Section tables of every strip at speed v: airfoil blend, lift-slope cap, maximum-lift cap, flap effects."""
        fd = self.flap_defl if flap_defl is None else float(flap_defl)
        key = (round(v, 4), round(fd, 3))
        if key in self._tabs:
            return self._tabs[key]
        p, ll = self.p, self.ll
        n = ll.n
        re = self.re_strips(v)
        cl = np.zeros((n, len(AF.ALPHA_DEG)))
        cd, cm = np.zeros_like(cl), np.zeros_like(cl)
        conf = np.zeros(n)
        for j, af in enumerate(self.foils):
            wj = self.W[:, j][:, None]
            a, b, c = af.table_at_re(re)
            cl += wj * a
            cd += wj * b
            cm += wj * c
            conf += self.W[:, j] * af.per_re(af.conf_min, re)
        if fd > 0.0 and self.flap_mask.any():
            # a flap lifts the section's whole lift curve by 2 pi x (change of zero-lift angle), so the maximum lift rises by the
            # same amount, up to the flap's limit (Raymer's table gives about 0.9 for a plain flap); moment and drag as below
            cf = p["flap_chord"]
            tau_f, cm_d = plain_flap_theory(cf)
            kf = p["ctrl_visc"] * deflection_factor(fd, p["ctrl_lin_limit_deg"], p["ctrl_eff_25"])
            m = self.flap_mask
            cl += m[:, None] * min(TWO_PI * kf * tau_f * math.radians(fd), p["flap_dclmax"])
            cm += m[:, None] * (kf * cm_d * math.radians(fd))
            cd += m[:, None] * (p["flap_drag_f"] * cf * max(fd - 10.0, 0.0))
        lin = (AF.ALPHA_DEG >= -2.0) & (AF.ALPHA_DEG <= 5.0)
        x = np.radians(AF.ALPHA_DEG[lin])
        xm = x.mean()
        yl = cl[:, lin]
        slope = ((yl - yl.mean(1, keepdims=True)) * (x - xm)).sum(1) / ((x - xm) ** 2).sum()
        alpha0 = -(yl.mean(1) - slope * xm) / slope
        ar = np.radians(AF.ALPHA_DEG)[None, :]
        if p["cap_slope"]:
            line = TWO_PI * (ar - alpha0[:, None])
            cl_unc = np.where(ar >= alpha0[:, None], np.minimum(cl, line), np.maximum(cl, line))
            slope_k = np.minimum(slope, TWO_PI)
        else:
            cl_unc = cl.copy()
            slope_k = slope
        cap = p["clmax_factor"] * ll.cos_sweep * cl_unc[:, AF.ALPHA_DEG >= 0.0].max(1)
        cd = cd * p["cd_factor"]
        cl_cap = np.minimum(cl_unc, cap[:, None])
        # angle at which each strip first reaches its cap
        hit = (cl_unc >= cap[:, None]) & (AF.ALPHA_DEG[None, :] >= 0.0)
        idx = np.where(hit.any(1), hit.argmax(1), len(AF.ALPHA_DEG) - 1)
        i0 = np.maximum(idx - 1, 0)
        r = np.arange(n)
        c0, c1 = cl_unc[r, i0], cl_unc[r, idx]
        frac = np.where(c1 > c0, (cap - c0) / np.where(c1 > c0, c1 - c0, 1.0), 1.0)
        alpha_cap = AF.ALPHA_DEG[i0] + np.clip(frac, 0.0, 1.0) * AF.D_ALPHA
        tab = SectionTables(cl_cap, cl_unc, cd, cm, slope_k / TWO_PI, alpha0, cap, re, conf)
        tab.alpha_cap_deg, tab.V, tab.flap_defl, tab.slope = alpha_cap, v, fd, slope_k
        tab.lin = ll.linear(tab.k, tab.alpha0)
        tab.x_ac = ll.aerodynamic_centre(tab.lin)
        self._tabs[key] = tab
        return tab

    # ---- control surfaces
    def d_alpha_ctrl(self, ail_deg: float = 0.0, flap_defl: float | None = None) -> np.ndarray:
        """Change of the geometric angle of every strip from the aileron (antisymmetric: right edge up, left edge down for a
        positive deflection). The aileron is modelled as an angle change, which keeps the section maximum lift unchanged:
        conservative for the stall of the down-going side. (The flap is in the section tables.)"""
        p = self.p
        da = np.zeros(self.ll.n)
        if ail_deg != 0.0 and self.ail_mask.any():
            tau, _ = plain_flap_theory(p["aileron_chord"])
            k = p["ctrl_visc"] * tau * deflection_factor(ail_deg, p["ctrl_lin_limit_deg"], p["ctrl_eff_25"]) * math.radians(ail_deg)
            da -= self.ail_mask * np.sign(self.ll.ym) * k
        return da

    # ---- fuselage
    def fuselage_cd(self, v: float, alpha_fus_rad: float) -> float:
        f = self.fuse
        if f is None or not self.p["fuse_drag_on"]:
            return 0.0
        p = self.p
        re = self.air.density * v * f.length / self.air.viscosity
        cf = 0.1 * cf_laminar(re) + 0.9 * cf_turbulent(re)
        fine = max(f.fineness, 3.0)
        ff = 1.0 + 60.0 / fine ** 3 + fine / 400.0
        s_wet = math.pi * f.diameter * f.length * (1.0 - 2.0 / fine) ** (2.0 / 3.0) * (1.0 + 1.0 / fine ** 2)
        cross = p["crossflow_factor"] * f.plan_area * abs(math.sin(alpha_fus_rad)) ** 3
        return (cf * ff * s_wet + cross) * p["interference"] * (1.0 + p["excrescence"]) / self.wing.area

    # ---- a point of the polar
    def point(self, alpha_deg: float, tab: SectionTables | None = None, ail_deg: float = 0.0, flap_defl: float | None = None,
              gamma0=None) -> Point:
        tab = tab or self.tab
        ll = self.ll
        a_geo = math.radians(alpha_deg) + ll.theta + self.d_alpha_ctrl(ail_deg, flap_defl)
        # always from the linear solution, so that every point is on the attached-flow branch (a warm start from a neighbouring angle
        # can land on a second, nearby solution where a section is close to its negative stall); the warm start is the fallback
        gam, aeff, ok = ll.solve_nonlinear(a_geo, tab, None)
        if not ok and gamma0 is not None:
            gam, aeff, ok = ll.solve_nonlinear(a_geo, tab, gamma0)
        ad = np.degrees(aeff)
        lk = tab.lookup(ad)
        cl_tot = ll.lift_coeff(gam)
        cdp = (1.0 + self.p["excrescence"]) * float(np.sum(ll.weight * lk["cd"]))
        cdi = ll.induced_drag_coeff(gam) / self.body_factor
        cdf = self.fuselage_cd(tab.V, math.radians(alpha_deg - self.p["incidence_deg"]))
        cm_sec = ll.section_moment_coeff(lk["cm"])
        x_ac = tab.x_ac
        util = lk["cl_unc"] / tab.cap
        util = np.where(ad >= tab.alpha_cap_deg - 1e-9, np.maximum(util, 1.0), util)     # stays >= 1 beyond the stall
        if not ok:
            # sections with an abrupt stall give the non-linear system a limit point: beyond it there is no attached-flow solution, and the
            # solver cannot converge. A solution that cannot be found is a stalled wing.
            util = np.maximum(util, 1.0)
        return Point(alpha_deg, cl_tot, cdp + cdi + cdf, cdp, cdi, cdf, ll.lift_moment_coeff(gam, self.x_ref) + cm_sec,
                     ll.lift_moment_coeff(gam, x_ac) + cm_sec, ll.rolling_coeff(gam), lk["cl"], lk["cl_unc"], lk["cd"], ad, util,
                     ad >= tab.alpha_cap_deg - 1e-9, gam, ok)

    def at_cl(self, cl_target: float, tab: SectionTables | None = None, ail_deg: float = 0.0, flap_defl: float | None = None) -> Point | None:
        """The point at a wing lift coefficient (secant on the angle of attack); None when it cannot be reached."""
        tab = tab or self.tab
        lin = tab.lin
        a1 = math.degrees((cl_target - lin.cl0) / lin.cl_alpha)
        p1 = self.point(a1, tab, ail_deg, flap_defl)
        a2 = a1 + (0.5 if p1.CL < cl_target else -0.5)
        p2 = self.point(a2, tab, ail_deg, flap_defl, p1.gamma)
        for _ in range(25):
            if abs(p2.CL - cl_target) < 1e-7:
                return p2
            d = p2.CL - p1.CL
            if abs(d) < 1e-12:
                return None
            a3 = a2 - (p2.CL - cl_target) * (a2 - a1) / d
            if not (AF.ALPHA_DEG[0] + 0.5 < a3 < AF.ALPHA_DEG[-1] - 0.5):
                return None
            p1, a1 = p2, a2
            p2, a2 = self.point(a3, tab, ail_deg, flap_defl, p2.gamma), a3
        return p2 if abs(p2.CL - cl_target) < 1e-5 else None

    # ---- stall
    def first_stall(self, tab: SectionTables | None = None, flap_defl: float | None = None) -> dict | None:
        """Wing lift coefficient, root angle and span station where the first strip reaches its section maximum."""
        tab = tab or self.tab
        lin, ll = tab.lin, self.ll
        pos = lin.p > 1e-9
        crit = np.where(pos, (tab.cap - lin.q) / np.where(pos, lin.p, 1.0), np.inf)
        cl_est = float(np.min(crit))
        a_est = math.degrees((cl_est - lin.cl0) / lin.cl_alpha) if math.isfinite(cl_est) else 12.0

        def f(a):
            return self.point(a, tab, 0.0, flap_defl).util_max - 1.0
        lo, hi = a_est - 3.0, a_est + 3.0
        while f(lo) > 0.0 and lo > AF.ALPHA_DEG[0] + 1.0:
            lo -= 3.0
        while f(hi) < 0.0 and hi < AF.ALPHA_DEG[-1] - 1.0:
            hi += 3.0
        if f(lo) > 0.0 or f(hi) < 0.0:
            return None
        a = bisect(f, lo, hi, tol=2e-5)
        pt = self.point(a, tab, 0.0, flap_defl)
        j = int(np.argmax(pt.util_j))
        return {"alpha_deg": a, "CL": pt.CL, "eta": float(ll.eta[j]), "strip": j, "point": pt}

    def stall(self) -> dict | None:
        """First-strip stall at the Reynolds number of the stall speed: the stall speed is first estimated from the linear
        critical-section lift coefficient at the design Reynolds number, the section tables are rebuilt there, and the
        non-linear first stall is solved with them."""
        lin, tab = self.tab.lin, self.tab
        pos = lin.p > 1e-9
        cl_est = float(np.min(np.where(pos, (tab.cap - lin.q) / np.where(pos, lin.p, 1.0), np.inf)))
        if not math.isfinite(cl_est) or cl_est <= 0.2:
            cl_est = 1.2
        v_est = math.sqrt(2.0 * self.n * self.weight / (self.air.density * self.wing.area * cl_est))
        st = self.first_stall(self.tables(round(v_est, 2)))
        if st is None:
            return None
        st["V_stall"] = math.sqrt(2.0 * self.n * self.weight / (self.air.density * self.wing.area * st["CL"]))
        return st

    # ---- curves
    def curve(self, tab: SectionTables | None = None, alpha_min: float = -6.0, step: float = 0.5, with_roll: bool = True,
              alpha_max: float = 24.0) -> dict:
        """Everything against the root angle of attack, from alpha_min until the lift has fallen well past its maximum."""
        tab = tab or self.tab
        al, rows = [], []
        gam = None
        cl_best, past = -9.0, 0
        a = alpha_min
        d_small = math.radians(3.0)
        d_max = self.p["aileron_defl_deg"]
        while a <= alpha_max:
            pt = self.point(a, tab, 0.0, None, gam)
            gam = pt.gamma
            row = {"pt": pt}
            if with_roll and self.ail_mask.any():
                pu = self.point(a, tab, 3.0, None, gam)
                pd = self.point(a, tab, -3.0, None, gam)
                row["ail_small"] = (pu.cl_roll - pd.cl_roll) / (2.0 * d_small)
                row["ail_max"] = self.point(a, tab, d_max, None, gam).cl_roll - pt.cl_roll
            al.append(a)
            rows.append(row)
            if pt.CL > cl_best:
                cl_best, past = pt.CL, 0
            elif cl_best > 0.3 and pt.CL < 0.93 * cl_best:               # the lift has fallen past its peak (only once the peak is real)
                past += 1
                if past >= 3:
                    break
            a += step
        pts = [r["pt"] for r in rows]
        out = {"alpha_deg": np.array(al), "CL": np.array([q.CL for q in pts]), "CD": np.array([q.CD for q in pts]),
               "CDp": np.array([q.cdp for q in pts]), "CDi": np.array([q.cdi for q in pts]), "CDf": np.array([q.cdf for q in pts]),
               "Cm_ref": np.array([q.cm_ref for q in pts]), "Cm_ac": np.array([q.cm_ac for q in pts]),
               "Cl_roll": np.array([q.cl_roll for q in pts]), "cl_j": np.array([q.cl_j for q in pts]),
               "util_j": np.array([q.util_j for q in pts]), "aeff_j": np.array([q.aeff_deg_j for q in pts]),
               "stalled_j": np.array([q.stalled_j for q in pts]), "converged": np.array([q.converged for q in pts])}
        if with_roll and self.ail_mask.any():
            out["ail_small"] = np.array([r["ail_small"] for r in rows])
            out["ail_max"] = np.array([r["ail_max"] for r in rows])
        return out

    # ---- performance against speed
    def performance(self, n: int = 28) -> dict:
        st = self.stall()
        if st is None:
            return {}
        v_s = st["V_stall"]
        vs = np.geomspace(1.02 * v_s, max(2.8 * v_s, 1.5 * self.V), n)
        out = {"V": [], "CL": [], "CD": [], "LD": [], "P": []}
        for v in vs:
            tab = self.tables(float(v))
            cl = self.n * self.weight / (0.5 * self.air.density * v * v * self.wing.area)
            pt = self.at_cl(cl, tab)
            if pt is None:
                continue
            d = pt.CD * 0.5 * self.air.density * v * v * self.wing.area
            out["V"].append(float(v))
            out["CL"].append(pt.CL)
            out["CD"].append(pt.CD)
            out["LD"].append(pt.CL / pt.CD)
            out["P"].append(d * v)
        return {k: np.array(v) for k, v in out.items()}

    # ---- everything, once
    def full(self) -> dict:
        if self._full is None:
            self._full = self._compute_full()
        return self._full

    def _compute_full(self) -> dict:
        p, w, ll, tab = self.p, self.wing, self.ll, self.tab
        s, b = w.area, w.span
        q_d = 0.5 * self.air.density * self.V ** 2
        cl_d = self.n * self.weight / (q_d * s)
        lin = tab.lin
        st = self.stall()
        m: dict = {"span_m": b, "area_m2": s, "aspect_ratio": w.aspect_ratio, "root_chord_m": w.root_chord, "tip_chord_m": w.tip_chord,
                   "mac_m": w.mac, "y_mac_m": w.y_mac, "mean_sweep_deg": w.mean_sweep_deg, "wing_loading_Nm2": self.weight / s,
                   "mass_kg": p["mass_kg"], "V_design_mps": self.V, "q_design_Pa": q_d, "Re_mac_design": self.air.density * self.V * w.mac / self.air.viscosity,
                   "CL_design": cl_d, "air_density": self.air.density, "load_factor": self.n,
                   "CL_alpha_per_rad": lin.cl_alpha, "CL_alpha_per_deg": lin.cl_alpha * math.pi / 180.0, "CL0": lin.cl0,
                   "alpha_zero_lift_deg": math.degrees(-lin.cl0 / lin.cl_alpha), "e_inviscid": lin.e_inviscid * self.body_factor}
        x_ac = ll.aerodynamic_centre(lin)
        m.update({"x_ac_m": x_ac, "x_ac_mac": (x_ac - w.x_le_mac) / w.mac, "x_ref_m": self.x_ref, "x_ref_mac": p["x_cg_mac"], "x_le_mac_m": w.x_le_mac})
        out = {"metrics": m, "stall": st, "design_ok": False}
        if st is None:
            m["warning"] = "the wing does not stall within the table range"
            return out
        m.update({"CL_max_first": st["CL"], "alpha_stall_first_deg": st["alpha_deg"], "eta_stall_first": st["eta"], "V_stall_mps": st["V_stall"],
                  "speed_margin": self.V / st["V_stall"]})
        # the polar, moment and control curves
        a_zero = math.degrees(-lin.cl0 / lin.cl_alpha)
        a_start = max(-18.0, math.floor((a_zero - 2.5) * 2.0) / 2.0)                                     # from 2.5 degrees below zero lift
        cv = self.curve(alpha_min=a_start)
        out["curve"] = cv
        m["curve_points_not_converged"] = int(np.sum(~cv["converged"] & (cv["alpha_deg"] > a_zero)))      # below zero lift the sections are in their negative stall: not of interest
        i_pk = int(np.argmax(cv["CL"]))
        m["CL_max_nl"], m["alpha_stall_nl_deg"] = float(cv["CL"][i_pk]), float(cv["alpha_deg"][i_pk])
        m["post_stall_drop"] = float(cv["CL"][i_pk] - cv["CL"][-1])
        pt = self.at_cl(cl_d) if cl_d < st["CL"] else None
        if pt is None:
            m["warning"] = "the design lift coefficient is beyond the stall"
            return out
        out["design_ok"] = True
        out["design_point"] = pt
        m["design_converged"] = bool(pt.converged)
        ar = w.aspect_ratio
        d_n = pt.CD * q_d * s
        m.update({"alpha_design_deg": pt.alpha_deg, "alpha_fus_design_deg": pt.alpha_deg - p["incidence_deg"], "CD": pt.CD, "CDp": pt.cdp, "CDi": pt.cdi,
                  "CDf": pt.cdf, "LD": pt.CL / pt.CD, "e_design": pt.CL ** 2 / (math.pi * ar * pt.cdi) if pt.cdi > 0 else float("nan"),
                  "drag_N": d_n, "P_req_W": d_n * self.V, "Cm_ref": pt.cm_ref, "Cm_ac": pt.cm_ac,
                  "util_max": pt.util_max, "util_eta": float(ll.eta[int(np.argmax(pt.util_j))]),
                  "lift_centroid_eta": ll.lift_centroid(pt.gamma), "root_bending_coeff": ll.root_bending_coeff(pt.gamma)})
        # pitching moment slope and break
        cl_c, cm_c = cv["CL"][: i_pk + 1], cv["Cm_ref"][: i_pk + 1]
        i_d = int(np.argmin(np.abs(cl_c - cl_d)))
        i1, i2 = max(i_d - 1, 0), min(i_d + 1, len(cl_c) - 1)
        m["dCm_dCL_ref"] = float((cm_c[i2] - cm_c[i1]) / (cl_c[i2] - cl_c[i1])) if cl_c[i2] != cl_c[i1] else float("nan")
        m["dCm_dCL_ac"] = float((cv["Cm_ac"][i2] - cv["Cm_ac"][i1]) / (cl_c[i2] - cl_c[i1])) if cl_c[i2] != cl_c[i1] else float("nan")
        cls = st["CL"]

        def cm_at(cl_x):
            return float(np.interp(cl_x, cl_c, cm_c))
        s_lin = (cm_at(0.6 * cls) - cm_at(0.3 * cls)) / (0.3 * cls)
        cl_hi = float(cv["CL"][i_pk])
        # break: the moment the wing has at its lift peak minus what the linear slope (0.3 to 0.6 of the first-strip CLmax) would give, from 90 %
        # of the first-strip CLmax; it includes the load that moves inboard as outer strips stall (nose-up on a swept wing)
        m["pitch_break"] = cm_at(cl_hi) - (cm_at(0.9 * cls) + s_lin * (cl_hi - 0.9 * cls))
        m["pitch_break_first_stall"] = cm_at(cls) - (cm_at(0.9 * cls) + s_lin * 0.1 * cls)
        m["pitch_tendency"] = "nose-up" if m["pitch_break"] > 0.005 else ("nose-down" if m["pitch_break"] < -0.005 else "neutral")
        usable = cl_c <= st["CL"]
        zc = np.where(np.diff(np.sign(cm_c[usable])) != 0)[0]          # trim only below the stall
        if len(zc):
            i = int(zc[0])
            m["trim_CL_wing_alone"] = float(cl_c[i] + (0.0 - cm_c[i]) * (cl_c[i + 1] - cl_c[i]) / (cm_c[i + 1] - cm_c[i]))
        else:
            m["trim_CL_wing_alone"] = float("nan")
        # lateral derivatives (linear)
        m["Cl_p"] = ll.roll_damping(tab.k)
        m["Cl_beta_dihedral"] = ll.dihedral_effect(tab.k)
        m["Cl_beta_sweep"] = ll.sideslip_effect(lin, pt.CL)
        m["Cl_beta_total"] = m["Cl_beta_dihedral"] + m["Cl_beta_sweep"]
        # ailerons: linear roll power, roll power at the design point and near the stall
        if self.ail_mask.any():
            a_lin = ll.rolling_coeff(ll.solve_linear_dalpha(tab.k, self.d_alpha_ctrl(3.0) / math.radians(3.0)))
            m["Cl_da_linear_per_rad"] = a_lin

            def roll_power(alpha_deg):
                pu = self.point(alpha_deg, tab, 3.0)
                pd = self.point(alpha_deg, tab, -3.0)
                return (pu.cl_roll - pd.cl_roll) / (2.0 * math.radians(3.0))
            m["Cl_da_design_per_rad"] = roll_power(pt.alpha_deg)
            a09 = float(np.interp(0.9 * cls, cv["CL"][: i_pk + 1], cv["alpha_deg"][: i_pk + 1]))
            m["Cl_da_at_0p9_per_rad"] = roll_power(a09)
            m["roll_authority_0p9"] = m["Cl_da_at_0p9_per_rad"] / a_lin if a_lin else float("nan")
            m["roll_authority_design"] = m["Cl_da_design_per_rad"] / a_lin if a_lin else float("nan")
            m["roll_max_design"] = self.point(pt.alpha_deg, tab, p["aileron_defl_deg"]).cl_roll - pt.cl_roll
            m["roll_max_at_stall"] = float(self.point(st["alpha_deg"], tab, p["aileron_defl_deg"]).cl_roll - self.point(st["alpha_deg"], tab, 0.0).cl_roll)
            eff = cv["ail_small"][: i_pk + 1] / a_lin if a_lin else None
            cl_eff = cv["CL"][: i_pk + 1]
            lost = np.where((eff < 0.5) & (cl_eff >= cl_d))[0] if eff is not None else []      # from the design point upwards
            m["control_loss_CL"] = float(cl_eff[lost[0]]) if len(lost) else float("nan")
        # flap
        if self.flap_defl > 0.0 and self.flap_mask.any():
            t0 = self.tables(self.V, 0.0)
            p0a = self.point(pt.alpha_deg, t0, 0.0, 0.0)               # same angle of attack, flap up
            p0 = self.at_cl(pt.CL, t0, 0.0, 0.0)                       # same lift, flap up
            m["flap_dCL_same_alpha"] = pt.CL - p0a.CL
            if p0 is not None:
                m["flap_dAlpha_same_CL_deg"] = pt.alpha_deg - p0.alpha_deg
                m["flap_dCm_same_CL"] = pt.cm_ref - p0.cm_ref
                m["flap_dCD_same_CL"] = pt.CD - p0.CD
            s0 = self.first_stall(t0, 0.0)
            m["flap_dCLmax"] = (st["CL"] - s0["CL"]) if s0 else float("nan")
        # performance
        perf = self.performance()
        out["performance"] = perf
        if len(perf.get("V", [])) > 3:
            i = int(np.argmax(perf["LD"]))
            m.update({"LD_max": float(perf["LD"][i]), "CL_at_LD_max": float(perf["CL"][i]), "V_best_glide": float(perf["V"][i])})
            j = int(np.argmin(perf["P"]))
            m.update({"P_min_W": float(perf["P"][j]), "V_min_power": float(perf["V"][j])})
            endu = perf["CL"] ** 1.5 / perf["CD"]
            m["endurance_factor"] = float(endu.max())
        m["endurance_factor_design"] = pt.CL ** 1.5 / pt.CD
        m["confidence_min"] = float(np.min(tab.conf))
        m["stall_confidence"] = float(tab.conf[st["strip"]])
        return out

    # ---- speed-up for the optimiser: only what the objectives and the limits need
    def quick(self, need: set | None = None) -> dict:
        need = need or set()
        p, w, ll, tab = self.p, self.wing, self.ll, self.tab
        q_d = 0.5 * self.air.density * self.V ** 2
        cl_d = self.n * self.weight / (q_d * w.area)
        st = self.stall()
        bad = {"ok": False, "reason": "no stall found"}
        if st is None:
            return bad
        m = {"CL_max_first": st["CL"], "eta_stall_first": st["eta"], "V_stall_mps": st["V_stall"], "speed_margin": self.V / st["V_stall"],
             "wing_loading_Nm2": self.weight / w.area, "CL_design": cl_d, "V_design_mps": self.V, "area_m2": w.area}
        if cl_d >= st["CL"]:
            m.update({"ok": False, "reason": "the design lift coefficient is beyond the stall"})
            return m
        pt = self.at_cl(cl_d)
        if pt is None:
            m.update({"ok": False, "reason": "no solution at the design lift coefficient"})
            return m
        d_n = pt.CD * q_d * w.area
        m.update({"ok": True, "CD": pt.CD, "LD": pt.CL / pt.CD, "P_req_W": d_n * self.V, "Cm_ref": pt.cm_ref, "util_max": pt.util_max,
                  "endurance_factor_design": pt.CL ** 1.5 / pt.CD})
        if "LD_max" in need:
            def neg_ld(cl):
                q = self.at_cl(cl)
                return -(q.CL / q.CD) if q is not None else 0.0
            x, f = golden_min(neg_ld, 0.15, 0.97 * st["CL"], 2e-3)
            m["LD_max"] = -f
        if "roll" in need and self.ail_mask.any():
            a_lin = ll.rolling_coeff(ll.solve_linear_dalpha(tab.k, self.d_alpha_ctrl(3.0) / math.radians(3.0)))
            a09 = self.at_cl(0.9 * st["CL"])
            if a09 is not None and a_lin:
                pu, pd = self.point(a09.alpha_deg, tab, 3.0), self.point(a09.alpha_deg, tab, -3.0)
                m["roll_authority_0p9"] = (pu.cl_roll - pd.cl_roll) / (2.0 * math.radians(3.0)) / a_lin
                m["Cl_da_at_0p9_per_rad"] = m["roll_authority_0p9"] * a_lin
        return m


# ------------------------------------------------------------------------------------------------ safety factor
def safety_factors(p: dict) -> dict:
    """The factor of each category: the safety factor where its switch is on, 1 where it is off."""
    f = float(p["safety_factor"])
    return {"drag": f if p["sf_drag"] else 1.0, "lift": f if p["sf_lift"] else 1.0, "moment": f if p["sf_moment"] else 1.0,
            "control": f if p["sf_control"] else 1.0}


def with_safety(m: dict, f: dict) -> dict:
    """The results with the safety factor applied in the adverse direction (keys end in _sf)."""
    out = {}

    def put(key, val):
        out[key + "_sf"] = val

    def has(*ks):
        return all(k in m and m[k] is not None and not (isinstance(m[k], float) and math.isnan(m[k])) for k in ks)
    if has("CD"):
        put("CD", m["CD"] * f["drag"])
        put("LD", m["LD"] / f["drag"])
        put("P_req_W", m["P_req_W"] * f["drag"])
        put("drag_N", m["drag_N"] * f["drag"]) if has("drag_N") else None
    if has("LD_max"):
        put("LD_max", m["LD_max"] / f["drag"])
    if has("P_min_W"):
        put("P_min_W", m["P_min_W"] * f["drag"])
    if has("endurance_factor"):
        put("endurance_factor", m["endurance_factor"] / f["drag"])
    if has("endurance_factor_design"):
        put("endurance_factor_design", m["endurance_factor_design"] / f["drag"])
    if has("CL_max_first"):
        put("CL_max_first", m["CL_max_first"] / f["lift"])
        put("V_stall_mps", m["V_stall_mps"] * math.sqrt(f["lift"]))
        put("speed_margin", m["V_design_mps"] / (m["V_stall_mps"] * math.sqrt(f["lift"])))
    if has("util_max"):
        put("util_max", m["util_max"] * f["lift"])
    if has("Cm_ref"):
        put("Cm_ref", m["Cm_ref"] * f["moment"])
        put("Cm_ac", m["Cm_ac"] * f["moment"]) if has("Cm_ac") else None
    if has("pitch_break"):
        put("pitch_break", m["pitch_break"] * f["moment"])
    if has("root_bending_coeff"):
        put("root_bending_coeff", m["root_bending_coeff"] * f["moment"])
    for k in ("Cl_da_linear_per_rad", "Cl_da_design_per_rad", "Cl_da_at_0p9_per_rad", "roll_max_design", "roll_max_at_stall",
              "roll_authority_0p9", "roll_authority_design"):
        if has(k):
            put(k, m[k] / f["control"])
    return {k: v for k, v in out.items() if v is not None}
