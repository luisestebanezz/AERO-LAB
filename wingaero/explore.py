"""Search through the ranges: a Sobol sample of the whole space the ranges span, sweeps of one or two inputs, and which inputs matter.

The optimiser (optimise.py) finds one good design inside the ranges; these tools show the space itself: what is feasible, how the results spread,
where the limits bite, and which input moves which result. Every result carries the safety factor (the *_sf values) and every design is tested
against the limits of the inputs (stall station, speed margin, strip utilisation, ...) at every flight point (points.py). The middle of every range
is the nominal design. The results shown are those of the design point (the first flight point).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .optimise import Problem
from .params import BY_KEY, RANGE, mode_of, resolve

# label -> key in the quick results (the *_sf keys have the safety factor applied)
METRICS = {
    "Lift-to-drag at the design point (SF)": "LD_sf", "Power at the design point, W (SF)": "P_req_W_sf", "Stall speed, m/s (SF)": "V_stall_mps_sf",
    "Usable maximum lift (SF)": "CL_max_first_sf", "Where the stall starts (share of the semi-span)": "eta_stall_first", "Cm at the design point (SF)": "Cm_ref_sf",
    "Design speed / stall speed (SF)": "speed_margin_sf", "Strip utilisation (SF)": "util_max_sf", "Wing loading, N/m²": "wing_loading_Nm2", "Wing area, m²": "area_m2",
    "Endurance factor CL^1.5/CD (SF)": "endurance_factor_design_sf", "Drag coefficient at the design point (SF)": "CD_sf",
}


@dataclass
class Sample:
    variables: list
    rows: list = field(default_factory=list)       # dicts: inputs, p, m (design point), ms (every flight point), g, ok, feasible, objective
    weights: dict = field(default_factory=dict)
    scale: list = field(default_factory=list)      # per flight point: objective key -> normalising value
    nominal: dict = field(default_factory=dict)    # results at the middle of every range (design point)
    problem: object = None

    def usable(self) -> list[dict]:
        return [r for r in self.rows if r["ok"]]

    def feasible(self) -> list[dict]:
        return [r for r in self.usable() if r["feasible"]]


def sample(values: dict, fixed: dict, bounds: dict, lists: dict, n: int = 256, seed: int = 1, include_uncertain: bool = True,
           weights: dict | None = None, progress=None, points=None) -> Sample:
    """n designs of a scrambled Sobol sequence over the ranges (numeric inputs uniformly, airfoils and choices from their lists); a design input that
    is auto is sampled over everything it allows. points: the flight points (None: the design point alone)."""
    from scipy.stats import qmc
    prob = Problem(values, fixed, bounds, lists, weights or {"ld": 1.0}, points, None, include_uncertain)
    vs = prob.vs
    if not vs:
        raise ValueError("nothing to search: switch 'range' on for the inputs to search, or leave a design input auto")
    out = Sample(vs, [], prob.weights, prob.scale, prob.baseline_ms[0], prob)
    u = qmc.Sobol(d=len(vs), scramble=True, seed=seed).random(n)
    for i, ui in enumerate(u):
        x = [(v.lo + t * (v.hi - v.lo)) if v.kind == "float" else min(int(t * len(v.pool)), len(v.pool) - 1) for v, t in zip(vs, ui)]
        e = prob.evaluate(x)
        out.rows.append({"inputs": prob.values_at(x), "p": e.p, "m": e.m, "ms": e.ms, "g": e.g, "ok": e.ok, "feasible": e.feasible,
                         "objective": e.J if e.feasible else None})
        if progress and (i + 1) % 16 == 0:
            progress(i + 1, n)
    prob._cache.clear()                                      # a sample is evaluated once: do not keep every design in memory
    return out


def effects(s: Sample, metric_key: str, only_feasible: bool = False) -> list[dict]:
    """How strongly each searched input is tied to a result over the sample: Spearman rank correlation for numeric inputs (signed), and the
    share of the variance explained (correlation ratio, unsigned) for airfoils. Sorted by strength."""
    from scipy.stats import spearmanr
    rows = s.feasible() if only_feasible else s.usable()
    rows = [r for r in rows if r["m"].get(metric_key) is not None and math.isfinite(r["m"][metric_key])]
    if len(rows) < 8:
        return []
    y = np.array([r["m"][metric_key] for r in rows])
    out = []
    for v in s.variables:
        col = [r["inputs"][v.key] for r in rows]
        if v.kind == "float":
            xs = np.array(col, float)
            if np.ptp(xs) < 1e-12 or np.ptp(y) < 1e-15:
                continue
            rho = float(spearmanr(xs, y).statistic)
            out.append({"key": v.key, "label": BY_KEY[v.key].label.strip(), "effect": rho, "kind": "rank correlation"})
        else:
            groups = {}
            for c, yi in zip(col, y):
                groups.setdefault(c, []).append(yi)
            ss_b = sum(len(g) * (np.mean(g) - y.mean()) ** 2 for g in groups.values())
            ss_t = float(np.sum((y - y.mean()) ** 2))
            if ss_t > 0:
                out.append({"key": v.key, "label": BY_KEY[v.key].label.strip(), "effect": math.sqrt(ss_b / ss_t), "kind": "variance explained (root)"})
    return sorted(out, key=lambda r: -abs(r["effect"]))


def bounds_of(fixed: dict, bounds: dict, key: str) -> tuple[float, float]:
    """The span a sweep covers for a numeric input: its range if it is given as one, else everything the input allows (auto)."""
    p = BY_KEY[key]
    return tuple(bounds[key]) if (mode_of(fixed, key) == RANGE and key in bounds) else (p.lo, p.hi)


def _grid_point(prob: Problem, assign: dict):
    v, f = dict(prob.values), dict(prob.fixed)
    for k, x in assign.items():
        v[k], f[k] = x, True
    return prob.evaluate_inputs(resolve(v, f, prob.bounds, prob.lists))


def _sweep_problem(values, fixed, bounds, lists, points) -> Problem:
    return Problem(values, fixed, bounds, lists, {"ld": 1.0}, points, vs=[])


def sweep1d(values: dict, fixed: dict, bounds: dict, lists: dict, key: str, n: int = 21, need: set | None = None, points=None) -> dict:
    """Every result against one input over its range, the other inputs at the middle of their ranges."""
    prob = _sweep_problem(values, fixed, bounds, lists, points)
    lo, hi = bounds_of(fixed, bounds, key)
    xs = np.linspace(lo, hi, n)
    rows = []
    for x in xs:
        e = _grid_point(prob, {key: float(x)})
        rows.append({"x": float(x), "m": e.m, "feasible": e.feasible, "g": e.g})
    return {"key": key, "x": xs, "rows": rows}


def sweep2d(values: dict, fixed: dict, bounds: dict, lists: dict, key_x: str, key_y: str, n: int = 13, need: set | None = None, points=None) -> dict:
    """Every result over a grid of two inputs, the others at the middle of their ranges."""
    prob = _sweep_problem(values, fixed, bounds, lists, points)
    xlo, xhi = bounds_of(fixed, bounds, key_x)
    ylo, yhi = bounds_of(fixed, bounds, key_y)
    xs, ys = np.linspace(xlo, xhi, n), np.linspace(ylo, yhi, n)
    grid = [[None] * n for _ in range(n)]
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            e = _grid_point(prob, {key_x: float(x), key_y: float(y)})
            grid[j][i] = {"m": e.m, "feasible": e.feasible, "g": e.g}
    return {"key_x": key_x, "key_y": key_y, "x": xs, "y": ys, "grid": grid}
