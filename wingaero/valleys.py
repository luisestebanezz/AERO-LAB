"""Valleys around a design: which inputs can move a little and which cannot (Luis, 2026-11-19).

Around an optimum the objective forms a valley: along an input that hardly matters the floor is wide and flat, along an input that matters the walls
are close. This module measures the valley of every searched input and of every pair, with the same evaluation as the optimiser (optimise.Problem: the
weighted objective at every flight point, the limits of every point, the safety factor everywhere).

* **Loss.** The weighted objective J is measured as a loss, 100 (J - J*) / sum|w|: the change of the weighted objective as a percentage of the starting
  design's value (one objective of weight one: 1 % is 1 % of its starting value). Negative: a better design exists there.
* **Free range of an input** (`tolerances`): the interval around the optimum, with every other input held at the optimum, in which the loss stays below
  the tolerance **and** every limit is met. The end of the interval is found by bisection and said: the objective, a named limit, or the end of the range
  the input allows. An airfoil or a choice is tested option by option. The free range is a share of the allowed range: free, loose, tight or critical.
* **Surface** (`surface`): the loss over the grid of two inputs, with the designs that break a limit marked: a 3D valley.
* **Combinations** (`combinations`): valleys are often diagonal (two inputs that can move together but not alone). At an optimum some limits are
  active (the speed margin over the stall, typically): they block the directions along their gradient, and the inputs can only move in combinations that
  keep those limits. The free combinations are the eigenvectors of the Hessian of the loss reduced to the directions the active limits leave free, sloppiest
  first, with how far each can move before the objective loses the tolerance, another limit breaks, or a range ends. Finite differences in coordinates
  scaled to the range of each input; the prediction is checked against the real design.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .params import BY_KEY

KEY_RESULTS = ("LD_sf", "P_req_W_sf", "V_stall_mps_sf", "speed_margin_sf", "eta_stall_first", "Cm_ref_sf", "util_max_sf")
VERDICTS = (("free", 0.60), ("loose", 0.25), ("tight", 0.08), ("critical", 0.0))     # (name, least share of the allowed range)
G_TOL = 1e-6
WALL = 5.0                                                                            # objective added where a design cannot be evaluated (finite differences)


def verdict_of(share: float) -> str:
    for name, lo in VERDICTS:
        if share >= lo:
            return name
    return "critical"


def _point(problem, x) -> dict:
    """One design evaluated: the loss ingredients and the key results (design point)."""
    e = problem.evaluate(x)
    g = e.g or []
    imax = int(np.argmax(g)) if g else -1
    return {"ok": bool(e.ok), "J": float(e.J) if e.ok and e.J is not None else float("nan"),
            "gmax": float(max(g)) if g else (float("inf") if not e.ok else -1.0), "imax": imax, "g": g,
            "viol": e.violation, "m": {k: (e.m or {}).get(k) for k in KEY_RESULTS}}


def _norm(problem) -> float:
    return max(sum(abs(w) for w in problem.weights.values()), 1e-9)


def _status(pt: dict, loss: float, tol: float, names: list) -> tuple[bool, str]:
    """(acceptable, why not): acceptable = evaluable, within the tolerance and every limit met."""
    if not pt["ok"]:
        return False, "no design solution (the design lift coefficient is beyond the stall)"
    if pt["gmax"] > G_TOL:
        return False, names[pt["imax"]] if 0 <= pt["imax"] < len(names) else "a limit"
    if loss > tol:
        return False, f"the objective loses {tol:g} %"
    return True, ""


# ------------------------------------------------------------------------------------------------ the free range of every input
@dataclass
class VarValley:
    key: str
    label: str
    unit: str
    kind: str                       # float | airfoil | choice | bool
    x_star: object                  # the value at the optimum (a name for an airfoil or a choice)
    lo: float                       # the range the input allows (the range given, or everything it allows)
    hi: float
    free_lo: float                  # the free range around the optimum
    free_hi: float
    share: float                    # free range / allowed range (a share of the options for an airfoil or a choice)
    verdict: str
    bind_lo: str                    # what ends the free range on each side
    bind_hi: str
    xs: list = field(default_factory=list)          # the scan
    loss: list = field(default_factory=list)
    gmax: list = field(default_factory=list)
    ok: list = field(default_factory=list)
    effect: dict = field(default_factory=dict)      # key result -> (least, greatest) over the free range
    options: list = field(default_factory=list)     # airfoils and choices: one row per option

    @property
    def width(self) -> float:
        return self.free_hi - self.free_lo


@dataclass
class ValleyReport:
    problem: object
    x_star: np.ndarray
    J_star: float
    norm: float
    tol_pct: float
    valleys: list                   # VarValley, one per searched input, in the order of problem.vs
    warnings: list = field(default_factory=list)
    active: list = field(default_factory=list)      # the limits that are (nearly) active at the centre: {name, g}
    n_evals: int = 0
    seconds: float = 0.0

    def counts(self) -> dict:
        out = {v[0]: 0 for v in VERDICTS}
        for r in self.valleys:
            out[r.verdict] += 1
        return out


def _scan_float(problem, x_star, i, n, tol, norm, J_star, names) -> VarValley:
    var = problem.vs[i]
    xs = np.unique(np.concatenate([np.linspace(var.lo, var.hi, n), [x_star[i]]]))
    pts = []
    for xv in xs:
        x = np.array(x_star, float)
        x[i] = xv
        pts.append(_point(problem, x))
    loss = np.array([100.0 * (p["J"] - J_star) / norm if p["ok"] else float("nan") for p in pts])
    i0 = int(np.argmin(np.abs(xs - x_star[i])))
    acc = [_status(p, l if math.isfinite(l) else 0.0, tol, names) for p, l in zip(pts, loss)]

    def at(xv):
        x = np.array(x_star, float)
        x[i] = xv
        p = _point(problem, x)
        l = 100.0 * (p["J"] - J_star) / norm if p["ok"] else float("nan")
        return p, l, _status(p, l if math.isfinite(l) else 0.0, tol, names)

    inside = [pts[i0]]                                  # the designs inside the free range (for the effect on every result)

    def walk(step: int):
        k = i0
        while True:
            k2 = k + step
            if k2 < 0 or k2 >= len(xs):
                return float(xs[k]), "the end of the range the input allows"
            if acc[k2][0]:
                k = k2
                inside.append(pts[k])
                continue
            a, b = float(xs[k]), float(xs[k2])               # a acceptable, b not: bisection for the edge
            why = acc[k2][1]
            for _ in range(5):
                mid = 0.5 * (a + b)
                p, _, (okm, wm) = at(mid)
                if okm:
                    a = mid
                    inside.append(p)
                else:
                    b, why = mid, wm
            return a, why

    if not acc[i0][0]:
        free_lo = free_hi = float(x_star[i])
        bind_lo = bind_hi = acc[i0][1] or "the optimum itself breaks a limit"
    else:
        free_lo, bind_lo = walk(-1)
        free_hi, bind_hi = walk(+1)
    span = var.hi - var.lo
    share = (free_hi - free_lo) / span if span > 0 else 1.0
    effect = {}
    for k in KEY_RESULTS:
        vals = [p["m"][k] for p in inside if p["m"].get(k) is not None and math.isfinite(p["m"][k])]
        if vals:
            effect[k] = (float(min(vals)), float(max(vals)))
    pr = BY_KEY[var.key]
    return VarValley(var.key, pr.label.strip(), "" if pr.unit == "-" else pr.unit, "float", float(x_star[i]), float(var.lo), float(var.hi), free_lo, free_hi, share,
                     verdict_of(share), bind_lo, bind_hi, [float(v) for v in xs], [float(v) for v in loss], [p["gmax"] for p in pts], [p["ok"] for p in pts], effect)


def _scan_options(problem, x_star, i, tol, norm, J_star, names) -> VarValley:
    var = problem.vs[i]
    pr = BY_KEY[var.key]
    rows = []
    inside = []
    for q, opt in enumerate(var.pool):
        x = np.array(x_star, float)
        x[i] = q
        p = _point(problem, x)
        l = 100.0 * (p["J"] - J_star) / norm if p["ok"] else float("nan")
        okm, why = _status(p, l if math.isfinite(l) else 0.0, tol, names)
        rows.append({"value": opt, "loss": l, "ok": okm, "why": why, "m": p["m"], "current": q == int(round(x_star[i]))})
        if okm:
            inside.append(p)
    n_ok = sum(1 for r in rows if r["ok"])
    share = n_ok / len(rows) if rows else 1.0
    effect = {}
    for k in KEY_RESULTS:
        vals = [p["m"][k] for p in inside if p["m"].get(k) is not None and math.isfinite(p["m"][k])]
        if vals:
            effect[k] = (float(min(vals)), float(max(vals)))
    cur = var.pool[int(round(x_star[i]))]
    return VarValley(var.key, pr.label.strip(), "", var.kind, cur, 0.0, float(len(var.pool) - 1), 0.0, float(len(var.pool) - 1), share,
                     verdict_of(share), "", "", [], [], [], [], effect, rows)


def tolerances(problem, x_star, tol_pct: float = 1.0, n: int = 41, progress=None) -> ValleyReport:
    """The free range of every searched input around x_star (a point of the searched variables), with the others held there."""
    t0, n0 = time.time(), problem.n_evals
    x_star = np.asarray(x_star, float)
    norm = _norm(problem)
    e0 = problem.evaluate(x_star)
    if not e0.ok or e0.J is None:
        raise ValueError("the design at the centre of the valleys has no design solution: choose another centre")
    names = list(problem.names)
    out = ValleyReport(problem, x_star, float(e0.J), norm, float(tol_pct), [])
    if not e0.feasible:
        out.warnings.append("The design at the centre breaks a limit, so the free ranges are empty: the valley is outside the feasible region.")
    for i, var in enumerate(problem.vs):
        if var.kind == "float":
            out.valleys.append(_scan_float(problem, x_star, i, n, tol_pct, norm, e0.J, names))
        else:
            out.valleys.append(_scan_options(problem, x_star, i, tol_pct, norm, e0.J, names))
        if progress:
            progress(i + 1, len(problem.vs))
    better = []
    for v in out.valleys:
        if v.kind == "float":
            good = [l for l, ok, g in zip(v.loss, v.ok, v.gmax) if ok and g <= G_TOL and math.isfinite(l)]
            if good:
                better.append((min(good), v.label))
    if better and min(b[0] for b in better) < -0.2:
        gain, lab = min(better)
        out.warnings.append(f"A better design that meets every limit exists near the centre (moving {lab} gains {-gain:.1f} %): the search stopped short of the best "
                            "point; run the optimiser again with a higher effort or another seed.")
    for k, gk in enumerate(e0.g or []):
        if gk > -0.1:
            out.active.append({"name": names[k], "g": float(gk)})
    out.n_evals, out.seconds = problem.n_evals - n0, time.time() - t0
    return out


def ranges_from_report(report: ValleyReport, keep=("free", "loose")) -> dict:
    """What to do with every input after the valley analysis: an input whose free range is wide enough stays a range (its free range), the others
    become a value (the optimum). Returns key -> ('range', lo, hi) | ('list', [options]) | ('value', x)."""
    out = {}
    for v in report.valleys:
        if v.kind == "float":
            out[v.key] = ("range", v.free_lo, v.free_hi) if (v.verdict in keep and v.width > 0) else ("value", v.x_star)
        else:
            ok = [r["value"] for r in v.options if r["ok"]]
            out[v.key] = ("list", ok) if (v.verdict in keep and len(ok) > 1) else ("value", v.x_star)
    return out


# ------------------------------------------------------------------------------------------------ a 3D valley: two inputs
def surface(problem, x_star, i: int, j: int, n: int = 21, window: float = 1.0, tol_pct: float = 1.0, progress=None) -> dict:
    """The loss over a grid of two numeric inputs i and j (indices into problem.vs), the others at x_star. window: the share of the allowed range of each
    input that is covered, centred on the optimum and kept inside the range."""
    x_star = np.asarray(x_star, float)
    vi, vj = problem.vs[i], problem.vs[j]
    if vi.kind != "float" or vj.kind != "float":
        raise ValueError("a surface needs two numeric inputs")
    norm = _norm(problem)
    e0 = problem.evaluate(x_star)
    if not e0.ok or e0.J is None:
        raise ValueError("the design at the centre has no design solution")

    def axis(v, c):
        half = 0.5 * window * (v.hi - v.lo)
        lo, hi = max(v.lo, c - half), min(v.hi, c + half)
        if hi - lo < 1e-9 * max(1.0, abs(c)):
            lo, hi = v.lo, v.hi
        return np.linspace(lo, hi, n)
    xs, ys = axis(vi, x_star[i]), axis(vj, x_star[j])
    loss = np.full((n, n), np.nan)
    gmax = np.full((n, n), np.nan)
    why = [[""] * n for _ in range(n)]
    ok = np.zeros((n, n), bool)
    res = {k: np.full((n, n), np.nan) for k in KEY_RESULTS}
    for b, yv in enumerate(ys):
        for a, xv in enumerate(xs):
            x = x_star.copy()
            x[i], x[j] = xv, yv
            p = _point(problem, x)
            ok[b, a] = p["ok"]
            if p["ok"]:
                loss[b, a] = 100.0 * (p["J"] - e0.J) / norm
                gmax[b, a] = p["gmax"]
                if p["gmax"] > G_TOL and 0 <= p["imax"] < len(problem.names):
                    why[b][a] = problem.names[p["imax"]]
                for k in KEY_RESULTS:
                    if p["m"].get(k) is not None:
                        res[k][b, a] = p["m"][k]
        if progress:
            progress(b + 1, n)
    feasible = ok & (gmax <= G_TOL)
    return {"i": i, "j": j, "key_x": vi.key, "key_y": vj.key, "x": xs, "y": ys, "loss": loss, "gmax": gmax, "ok": ok, "feasible": feasible, "why": why,
            "results": res, "x_star": float(x_star[i]), "y_star": float(x_star[j]), "tol_pct": float(tol_pct)}


# ------------------------------------------------------------------------------------------------ combinations of inputs
def describe_direction(vec, labels: list, k: int = 3, floor: float = 0.12) -> str:
    """The inputs a direction is made of, largest first, as text: '+0.62 aspect ratio, -0.55 tip chord / root chord'."""
    order = np.argsort(-np.abs(vec))[:k]
    return ", ".join(f"{vec[a]:+.2f} {labels[a]}" for a in order if abs(vec[a]) > floor)


def combinations(problem, x_star, h_share: float = 0.04, tol_pct: float = 1.0, max_vars: int = 14, active_g: float = -0.1, progress=None) -> dict:
    """The valley as a whole: which combinations of inputs can move together, and what blocks the others.

    Coordinates are scaled to the range of each numeric input (u = x / range: a step of 0.04 is 4 % of the range). At the optimum, by central differences:
    the gradient of the loss and of every limit function g, and the second derivatives (Hessian) of the loss. The **active limits** (g close to zero) block
    the directions along their gradient: those are the stiff combinations. The remaining directions (the null space of the active gradients) are the
    **free combinations**: the eigenvectors of the Hessian of the loss reduced to them, sloppiest first. How far each can move, in either sense: the
    loss stays under the tolerance (a quadratic with the reduced gradient), every other limit stays met (linearised), and the range an input allows is not left.
    The loss alone is smooth, so this is robust where a penalised merit with its kinks is not; the prediction is checked against the real design."""
    x_star = np.asarray(x_star, float)
    idx = [i for i, v in enumerate(problem.vs) if v.kind == "float"][:max_vars]
    if len(idx) < 2:
        raise ValueError("the combinations need at least two numeric inputs")
    norm = _norm(problem)
    e0 = problem.evaluate(x_star)
    if not e0.ok or e0.J is None:
        raise ValueError("the design at the centre has no design solution")
    J_star = float(e0.J)
    n = len(idx)
    R = np.array([problem.vs[i].hi - problem.vs[i].lo for i in idx], float)
    lo = np.array([problem.vs[i].lo for i in idx], float)
    hi = np.array([problem.vs[i].hi for i in idx], float)
    h = h_share * R
    xc = x_star.copy()
    for a, i in enumerate(idx):
        xc[i] = min(max(x_star[i], lo[a] + h[a]), hi[a] - h[a])
    names = list(problem.names)
    n_g = len(names)
    total = 1 + 2 * n + 4 * (n * (n - 1) // 2)
    done = [0]

    def at(shifts: dict):
        x = xc.copy()
        for a, s in shifts.items():
            x[idx[a]] += s * h[a]
        done[0] += 1
        if progress and done[0] % 8 == 0:
            progress(min(done[0], total), total)
        e = problem.evaluate(x)
        if not e.ok or e.J is None or e.g is None:
            return J_star + WALL, [1.0] * n_g                # no design solution: a wall
        return float(e.J), list(e.g)
    J0, g0 = at({})
    Jp, gp, Jm, gm = [], [], [], []
    for a in range(n):
        j1, g1 = at({a: +1})
        j2, g2 = at({a: -1})
        Jp.append(j1), gp.append(g1), Jm.append(j2), gm.append(g2)
    s100 = 100.0 / norm
    grad_J = np.array([(Jp[a] - Jm[a]) / (2.0 * h_share) for a in range(n)]) * s100          # loss % per unit of u
    grad_g = np.array([[(gp[a][k] - gm[a][k]) / (2.0 * h_share) for a in range(n)] for k in range(n_g)]) if n_g else np.zeros((0, n))
    H = np.zeros((n, n))
    for a in range(n):
        H[a, a] = (Jp[a] - 2.0 * J0 + Jm[a]) / h_share ** 2
    for a in range(n):
        for b in range(a + 1, n):
            H[a, b] = H[b, a] = (at({a: +1, b: +1})[0] - at({a: +1, b: -1})[0] - at({a: -1, b: +1})[0] + at({a: -1, b: -1})[0]) / (4.0 * h_share ** 2)
    if progress:
        progress(total, total)
    H = 0.5 * (H + H.T) * s100                                                             # loss % per unit of u squared
    labels = [BY_KEY[problem.vs[i].key].label.strip() for i in idx]
    g_c = np.array(g0 if g0 else [], float)                                                # limit values at the centre (<= 0 is met)
    # ---- the active limits: they block the directions along their gradient
    act = [k for k in range(n_g) if g_c[k] > active_g]
    A = grad_g[act] if act else np.zeros((0, n))
    blocked = []
    for k in act:
        gr = grad_g[k]
        nr = float(np.linalg.norm(gr))
        step = (-g_c[k]) / nr if nr > 1e-9 else float("inf")                              # distance along its gradient to break it (negative: already broken)
        blocked.append({"name": names[k], "g": float(g_c[k]), "grad": gr, "norm": nr, "room": step, "text": describe_direction(gr / nr, labels) if nr > 1e-9 else "no input moves it"})
    # ---- the free directions: the null space of the active gradients
    if len(act):
        _, sv, Vt = np.linalg.svd(A, full_matrices=True)
        rank = int(np.sum(sv > 1e-6 * max(sv[0], 1e-12))) if len(sv) else 0
        Z = Vt[rank:].T
    else:
        Z = np.eye(n)
    dirs = []
    if Z.shape[1]:
        Hred = Z.T @ H @ Z
        Hred = 0.5 * (Hred + Hred.T)
        lam, W = np.linalg.eigh(Hred)
        for q in range(len(lam)):
            d = Z @ W[:, q]
            d = d / np.linalg.norm(d)
            if d[int(np.argmax(np.abs(d)))] < 0:
                d = -d
            lam_q = float(lam[q])
            b = float(grad_J @ d)                                                          # slope of the loss along d (about zero at the optimum of the free subspace)
            ends = {}
            for sgn, label in ((+1, "plus"), (-1, "minus")):
                bs = sgn * b
                if lam_q > 1e-9:
                    s_obj = (-bs + math.sqrt(bs * bs + 2.0 * lam_q * tol_pct)) / lam_q
                else:
                    s_obj = tol_pct / bs if bs > 1e-9 else float("inf")
                s_lim, why, lim_name = float("inf"), "the objective", ""
                for k in range(n_g):
                    slope = float(sgn * (grad_g[k] @ d))
                    if slope > 1e-9:
                        sk = max(-g_c[k], 0.0) / slope
                        if sk < s_lim:
                            s_lim, lim_name = sk, names[k]
                dd = sgn * d * R
                with np.errstate(divide="ignore", invalid="ignore"):
                    room = np.where(dd > 1e-12, (hi - xc[idx]) / dd, np.where(dd < -1e-12, (lo - xc[idx]) / dd, np.inf))
                s_box = float(np.min(room))
                s_best = min(s_obj, s_lim, s_box)
                if s_best == s_obj:
                    why = f"the objective loses {tol_pct:g} %"
                elif s_best == s_lim:
                    why = lim_name
                else:
                    why = "the end of a range"
                ends[label] = (float(min(s_best, 1.0)), why, float(s_best))
            dirs.append({"vec": d, "lambda": lam_q, "slope": b, "r_plus": ends["plus"][0], "r_minus": ends["minus"][0], "why_plus": ends["plus"][1], "why_minus": ends["minus"][1],
                         "text": describe_direction(d, labels), "move_plus": {labels[a]: float(ends["plus"][0] * d[a] * R[a]) for a in np.argsort(-np.abs(d))[:3] if abs(d[a]) > 0.12},
                         "move_minus": {labels[a]: float(-ends["minus"][0] * d[a] * R[a]) for a in np.argsort(-np.abs(d))[:3] if abs(d[a]) > 0.12}})
    # ---- check the prediction on the real design along the sloppiest free directions (half of the predicted distance, in the plus sense)
    checks = []
    for q, dq in enumerate(dirs[:3]):
        for sgn, r in ((+1, dq["r_plus"]), (-1, dq["r_minus"])):
            s = 0.5 * min(r, 0.5)
            if s < 1e-4:
                continue
            x = xc.copy()
            for a, i in enumerate(idx):
                x[i] = min(max(x[i] + sgn * s * dq["vec"][a] * R[a], lo[a]), hi[a])
            e = problem.evaluate(x)
            pred_loss = sgn * dq["slope"] * s + 0.5 * dq["lambda"] * s * s
            pred_g = [float(g_c[k] + sgn * s * (grad_g[k] @ dq["vec"])) for k in range(n_g)]
            act_loss = 100.0 * (e.J - J0) / norm if (e.ok and e.J is not None) else float("nan")
            checks.append({"direction": q + 1, "sign": sgn, "step": float(s), "predicted_loss": float(pred_loss), "actual_loss": float(act_loss),
                           "predicted_feasible": bool(all(v <= 1e-6 for v in pred_g)), "actual_feasible": bool(e.feasible)})
    return {"idx": idx, "keys": [problem.vs[i].key for i in idx], "labels": labels, "R": R, "centre": xc[idx], "lo": lo, "hi": hi, "h_share": h_share, "tol_pct": tol_pct,
            "grad_J": grad_J, "H": H, "g": g_c, "names": names, "active": blocked, "directions": dirs, "checks": checks, "n_evals": done[0]}


def pair_to_show(problem, cv: dict, valleys: list | None = None) -> tuple[int, int] | None:
    """The two numeric inputs worth a 3D surface (indices into problem.vs): the input that moves the most active limit the most (the wall of the valley)
    with the tightest other input; without an active limit, the two tightest inputs (by the width of their free ranges, when they are known)."""
    idx = cv["idx"]
    if len(idx) < 2:
        return None
    share = {i: valleys[i].share for i in idx if valleys and i < len(valleys)} if valleys else {}
    order = sorted(idx, key=lambda i: (share.get(i, 1.0), i))                 # the tightest first
    first = None
    if cv["active"]:
        gr = max(cv["active"], key=lambda a: a["g"])["grad"]
        first = idx[int(np.argmax(np.abs(gr)))]
    if first is None:
        first = order[0]
    return first, next(i for i in order if i != first)
