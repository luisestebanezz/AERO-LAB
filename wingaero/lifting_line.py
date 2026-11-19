"""Vortex-lattice lifting line (Weissinger) for a planform of several panels, linear and non-linear.

Linear solution (the method of the configurator, physics-model.md section 3.3): one horseshoe vortex per spanwise strip
(bound segment on the quarter-chord line, trailing legs to downstream infinity), flow tangent at three-quarters of the
chord, and a lift slope per strip scaled by k = a0 / 2 pi:

    sum_j A_ij G_j = - k_i (alpha + theta_i - alpha0_i)               (G = circulation / speed)

Non-linear solution (this lab): the same lattice, but the section lift follows the airfoil table beyond the linear range,
up to and past the stall. The section's effective angle is the geometric angle minus the induced angle taken from the
lattice, with the 2-D self-induction of the strip removed:

    alpha_ind_i = - [ (A G)_i + G_i / (pi c_i) ] / k_i ,     G_i = (1/2) c_i cl_i(alpha_geo_i - alpha_ind_i)

(for a linear section the fixed point is exactly the linear solution above). It is solved with Newton's method.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .airfoil import ALPHA_DEG, D_ALPHA
from .geometry import Wing

N_HALF = 16


def _segment_w(px, py, x1, y1, x2, y2):
    """Vertical velocity at points (px, py) induced by unit-strength straight vortex segments 1 -> 2 (all in z = 0)."""
    r1x, r1y = px[:, None] - x1[None, :], py[:, None] - y1[None, :]
    r2x, r2y = px[:, None] - x2[None, :], py[:, None] - y2[None, :]
    cz = r1x * r2y - r1y * r2x
    r1, r2 = np.hypot(r1x, r1y), np.hypot(r2x, r2y)
    dot = (x2 - x1)[None, :] * (r1x / r1 - r2x / r2) + (y2 - y1)[None, :] * (r1y / r1 - r2y / r2)
    cz = np.where(np.abs(cz) < 1e-12, 1e-12, cz)
    return dot / (4.0 * math.pi * cz)


class SectionTables:
    """cl, cd and cm of every strip against the effective angle of attack, at the Reynolds number of the strip.

    cl is capped at the strip's maximum (plateau); cl_unc keeps the uncapped curve, so cl_unc / cap measures how close a
    strip is to its stall (1 = at the cap)."""

    def __init__(self, cl, cl_unc, cd, cm, k, alpha0, cap, re, conf=None):
        self.cl, self.cl_unc, self.cd, self.cm = cl, cl_unc, cd, cm
        self.k, self.alpha0, self.cap, self.re = k, alpha0, cap, re
        self.conf = conf
        self.rows = np.arange(len(k))

    def _idx(self, adeg):
        x = (np.clip(adeg, ALPHA_DEG[0], ALPHA_DEG[-1] - 1e-9) - ALPHA_DEG[0]) / D_ALPHA
        i = x.astype(int)
        return i, x - i

    def cl_slope(self, adeg):
        i, f = self._idx(adeg)
        v0, v1 = self.cl[self.rows, i], self.cl[self.rows, i + 1]
        return v0 + f * (v1 - v0), (v1 - v0) / D_ALPHA * (180.0 / math.pi)

    def lookup(self, adeg) -> dict:
        i, f = self._idx(adeg)
        out = {}
        for name in ("cl", "cl_unc", "cd", "cm"):
            t = getattr(self, name)
            v0, v1 = t[self.rows, i], t[self.rows, i + 1]
            out[name] = v0 + f * (v1 - v0)
        return out


@dataclass
class LinearSolution:
    g1: np.ndarray
    g0: np.ndarray
    cl_alpha: float          # per rad (root-chord angle of attack)
    cl0: float
    a2: float
    a1: float
    a0: float
    p: np.ndarray            # strip cl = p CL + q
    q: np.ndarray

    @property
    def e_inviscid(self) -> float:
        return 1.0 / (math.pi * self.aspect_ratio * self.a2)

    aspect_ratio: float = 1.0


class LiftingLine:
    def __init__(self, wing: Wing, n_half: int = N_HALF):
        self.wing = wing
        b, s = wing.span, wing.area
        th = np.linspace(0.0, math.pi, 2 * n_half + 1)
        y_edges = -0.5 * b * np.cos(th)
        ya, yb = y_edges[:-1], y_edges[1:]
        ym = -0.5 * b * np.cos(0.5 * (th[:-1] + th[1:]))     # control points at the cosine-angle midpoints
        dy = yb - ya
        eta = np.abs(ym) / (0.5 * b)
        c = wing.chord(eta)
        xa, xb = wing.shift_at(np.abs(ya) / (0.5 * b)), wing.shift_at(np.abs(yb) / (0.5 * b))
        xm = wing.shift_at(eta)
        xc = xm + 0.5 * c
        far = 1.0e3 * b
        self.aic = (_segment_w(xc, ym, xa + far, ya, xa, ya) + _segment_w(xc, ym, xa, ya, xb, yb)
                    + _segment_w(xc, ym, xb, yb, xb + far, yb))
        self.n = 2 * n_half
        self.ym, self.dy, self.eta, self.c = ym, dy, eta, c
        self.ya, self.yb, self.xa, self.xb = ya, yb, xa, xb
        self.xqc = 0.25 * wing.root_chord + xm              # quarter-chord point of each strip (behind the root leading edge)
        self.theta = wing.twist_rad(eta)
        self.dihedral = wing.dihedral_rad(eta)
        self.cos_sweep = wing.cos_sweep(eta)
        self.weight = c * dy / s                           # strip share of the wing area (sums to 1)
        self.tre = (1.0 / (2.0 * math.pi)) * (1.0 / (ym[:, None] - yb[None, :]) - 1.0 / (ym[:, None] - ya[None, :]))
        self.right = ym > 0
        self._ad = self.aic + np.diag(1.0 / (math.pi * c))

    # ---- integrated coefficients of a circulation distribution (per unit speed)
    def lift_coeff(self, gam) -> float:
        return 2.0 * float(np.sum(gam * self.dy)) / self.wing.area

    def rolling_coeff(self, gam) -> float:
        """Rolling-moment coefficient, positive right wing down."""
        return -2.0 * float(np.sum(gam * self.dy * self.ym)) / (self.wing.area * self.wing.span)

    def induced_drag_coeff(self, gam) -> float:
        return -float(np.sum(gam * (self.tre @ gam) * self.dy)) / self.wing.area

    def lift_moment_coeff(self, gam, x_ref: float) -> float:
        """Pitching-moment coefficient (nose up positive) of the lift of every strip acting at its quarter-chord point,
        about x_ref (m behind the root leading edge), on the mean chord."""
        w = self.wing
        return -float(np.sum(2.0 * gam * self.dy / w.area * (self.xqc - x_ref))) / w.mac

    def section_moment_coeff(self, cm_strip) -> float:
        w = self.wing
        return float(np.sum(cm_strip * self.c ** 2 * self.dy)) / (w.area * w.mac)

    def lift_centroid(self, gam) -> float:
        """Spanwise centroid of the lift of the right half-wing, as a share of the semi-span."""
        h = self.right
        return float(np.sum(gam[h] * self.dy[h] * self.eta[h]) / np.sum(gam[h] * self.dy[h]))

    def root_bending_coeff(self, gam) -> float:
        """Root bending moment of the lift of one half-wing per (q S b / 2): sum of strip lift times arm."""
        h = self.right
        return float(np.sum(2.0 * gam[h] * self.dy[h] / self.wing.area * self.eta[h]))

    # ---- linear solution
    def solve_linear_dalpha(self, k, dalpha):
        return np.linalg.solve(self.aic, -k * dalpha)

    def linear(self, k, alpha0) -> LinearSolution:
        k = np.asarray(k, float) * np.ones(self.n)
        a0 = np.asarray(alpha0, float) * np.ones(self.n)
        w, s, b = self.wing, self.wing.area, self.wing.span
        rhs = np.column_stack([-k, -k * (self.theta - a0)])
        g = np.linalg.solve(self.aic, rhs)
        g1, g0 = g[:, 0], g[:, 1]
        cl1, cl0 = self.lift_coeff(g1), self.lift_coeff(g0)

        def q(ga, gb):
            return -float(np.sum(ga * (self.tre @ gb) * self.dy)) / s
        a2, a1, ac = q(g1, g1), q(g1, g0) + q(g0, g1), q(g0, g0)
        p = 2.0 * g1 / (self.c * cl1)
        qq = 2.0 * (g0 - g1 * cl0 / cl1) / self.c
        return LinearSolution(g1, g0, cl1, cl0, a2 / cl1 ** 2, a1 / cl1 - 2.0 * a2 * cl0 / cl1 ** 2,
                              a2 * cl0 ** 2 / cl1 ** 2 - a1 * cl0 / cl1 + ac, p, qq, w.aspect_ratio)

    def aerodynamic_centre(self, lin: LinearSolution) -> float:
        """x of the wing aerodynamic centre behind the root leading edge: centroid of the additional loading."""
        wgt = lin.g1 * self.dy
        return float(np.sum(wgt * self.xqc) / np.sum(wgt))

    def sideslip_effect(self, lin: LinearSolution, cl: float) -> float:
        """Cl_beta per rad of the planar wing (sweep and taper, no dihedral) at wing lift coefficient cl."""
        s, b = self.wing.area, self.wing.span
        dx = self.xb - self.xa
        clb_alpha = -2.0 * float(np.sum(lin.g1 * dx * self.ym)) / (s * b)
        clb_zero = -2.0 * float(np.sum(lin.g0 * dx * self.ym)) / (s * b)
        alpha = (cl - lin.cl0) / lin.cl_alpha
        return clb_alpha * alpha + clb_zero

    def roll_damping(self, k) -> float:
        """Cl_p per unit p b / 2V (linear section slope)."""
        return self.rolling_coeff(self.solve_linear_dalpha(np.asarray(k) * np.ones(self.n), 2.0 * self.ym / self.wing.span))

    def dihedral_effect(self, k) -> float:
        """Cl_beta per rad of sideslip from the dihedral of every panel (the windward wing sees an extra angle)."""
        return self.rolling_coeff(self.solve_linear_dalpha(np.asarray(k) * np.ones(self.n), self.dihedral * np.sign(self.ym)))

    # ---- non-linear solution
    def solve_nonlinear(self, alpha_geo, tab: SectionTables, gamma0=None, tol: float = 1e-10, maxit: int = 80):
        """Circulation, effective angles (rad) and convergence flag for the geometric angle alpha_geo (rad, per strip).

        Newton's method first. Where a strip sits at the end of its lift plateau (the table's slope jumps between zero and a steep
        negative value) Newton can cycle between the two linearisations; then an under-relaxed fixed-point iteration, which is stable
        for negative slopes, finishes from the best Newton iterate."""
        c, k, n = self.c, tab.k, self.n
        m_ind = self._ad / k[:, None]                       # alpha_eff = alpha_geo + M G
        eye = np.eye(n)
        gam = np.linalg.solve(self.aic, -k * (alpha_geo - tab.alpha0)) if gamma0 is None else np.array(gamma0, float)
        # symmetric flight: keep the circulation symmetric (a post-stall solution can otherwise break the symmetry through round-off)
        sym = bool(np.allclose(alpha_geo, alpha_geo[::-1], atol=1e-12))
        if sym:
            gam = 0.5 * (gam + gam[::-1])
        ok = False
        best, best_res = gam.copy(), float("inf")
        for _ in range(maxit + 40):
            aeff = alpha_geo + m_ind @ gam
            cl, slope = tab.cl_slope(np.degrees(aeff))
            r = gam - 0.5 * c * cl
            res = float(np.max(np.abs(r)))
            if res < best_res:
                best, best_res = gam.copy(), res
            if res < tol:
                ok = True
                break
            jac = eye - (0.5 * c * slope)[:, None] * m_ind
            try:
                d = np.linalg.solve(jac, -r)
            except np.linalg.LinAlgError:
                d = -0.3 * r
            big = float(np.max(np.abs(m_ind @ d)))                    # trust region: no strip's effective angle moves by more than 1 degree per step,
            if big > 0.0175:                                          # so the iterate cannot jump over the narrow lift plateau into the falling branch
                d = d * (0.0175 / big)
            t = 1.0
            for _ in range(10):
                g2 = gam + t * d
                cl2, _ = tab.cl_slope(np.degrees(alpha_geo + m_ind @ g2))
                if float(np.max(np.abs(g2 - 0.5 * c * cl2))) < res * (1.0 - 1e-4 * t):
                    break
                t *= 0.5
            gam = 0.5 * (g2 + g2[::-1]) if sym else g2
        if not ok:
            gam = best
            for _ in range(400):
                cl, _ = tab.cl_slope(np.degrees(alpha_geo + m_ind @ gam))
                r = 0.5 * c * cl - gam
                if float(np.max(np.abs(r))) < 1e-9:
                    ok = True
                    break
                gam = gam + 0.12 * r
                if sym:
                    gam = 0.5 * (gam + gam[::-1])
        aeff = alpha_geo + m_ind @ gam
        return gam, aeff, ok
