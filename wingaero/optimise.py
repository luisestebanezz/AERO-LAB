"""Optimiser: chooses the inputs the user gave as a range or left auto.

The design inputs given as ranges are searched within their bounds by differential evolution (airfoil stations choose from their list, a choice
from its options); an auto design input is searched over everything it allows (see free_variables). Several objectives can be combined with weights (each normalised by its
value for the starting design); the limits are constraints. Everything is evaluated with the safety factor applied.

Flight points (points.py): the objectives are evaluated at several flight conditions of the same wing and combined with the weights of the points; the limits
of every point are constraints. The Problem class holds one such problem (inputs searched, objectives, points, limits) and evaluates a design the same way for the
optimiser, the sample and sweeps (explore.py) and the valley analysis (valleys.py).

Evaluation of one candidate takes about 30 ms per flight point (the non-linear lifting line at the design point and the first stall).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import airfoil as AF
from .analysis import Case, safety_factors, with_safety
from .params import (AUTO, BY_KEY, DEFAULT_AIRFOIL_POOL, PARAMS, RANGE, STATIONS, VALUE, active, coerce, default_fixed, defaults, mode_of, options_of,
                     resolve, validate)

# key -> (label, metric key, +1 to minimise / -1 to maximise, extra work the metric needs)
OBJECTIVES = {
    "ld": ("Lift-to-drag at the design point", "LD_sf", -1, set()),
    "ldmax": ("Best lift-to-drag", "LD_max_sf", -1, {"LD_max"}),
    "power": ("Power required at the design point", "P_req_W_sf", +1, set()),
    "endurance": ("Endurance factor CL^1.5 / CD at the design point", "endurance_factor_design_sf", -1, set()),
    "stall": ("Stall speed", "V_stall_mps_sf", +1, set()),
    "clmax": ("Usable maximum lift coefficient", "CL_max_first_sf", -1, set()),
    "cm": ("Pitching moment about the reference point (magnitude)", "Cm_ref_sf", +1, set()),
    "roll": ("Roll power left near the stall", "roll_authority_0p9_sf", -1, {"roll"}),
}


@dataclass
class Var:
    key: str
    kind: str                 # float | airfoil | choice | bool (the last three: an index into pool)
    lo: float = 0.0
    hi: float = 1.0
    pool: list = field(default_factory=list)


def free_variables(values: dict, fixed: dict, bounds: dict | None = None, foil_pool: list | None = None, lists: dict | None = None,
                   include_uncertain: bool = False) -> list[Var]:
    """The inputs that are searched.

    * range (ranges in `bounds`; airfoil lists and lists of options in `lists`): searched if the input is optimisable (a design input);
      include_uncertain adds the rest (environment, calibration, model switches: never optimised; the sample and the sweeps may vary them);
    * auto: the numeric design inputs and the airfoils are searched over everything they allow (the full allowed span; the default airfoil pool);
      environment, calibration, model switches, structure choices (panels, chord distribution, twist) and overrides are not searched
      (they take the lab's default, an override is not used);
    * value: not searched.
    Only inputs that are used given the others are searched (a choice that is searched counts every option it may take)."""
    bounds = bounds or {}
    lists = lists or {}
    pool = list(foil_pool or DEFAULT_AIRFOIL_POOL)
    res = resolve(values, fixed, bounds, lists)

    def searched(p) -> bool:
        mode = mode_of(fixed, p.key)
        if mode == VALUE:
            return False
        if mode == AUTO:
            return p.optimisable and p.kind in ("float", "airfoil") and not p.override
        return p.optimisable or include_uncertain

    def span(p):
        return bounds.get(p.key, (p.lo, p.hi)) if mode_of(fixed, p.key) == RANGE else (p.lo, p.hi)

    def pool_of(p) -> list:
        return list(lists.get(p.key) or pool) if mode_of(fixed, p.key) == RANGE else list(pool)

    options = {}                                       # a searched choice: every option it may take
    for p in PARAMS:
        if p.kind in ("choice", "bool") and searched(p):
            allowed = [coerce(p, a) for a in (lists.get(p.key) or options_of(p))]
            if len(allowed) > 1:
                options[p.key] = allowed
    foil_param = {}
    for st in STATIONS:
        key = f"foil_{st}"
        foil_param[st] = (res[key] == AF.PARAM) or (mode_of(fixed, key) != VALUE and AF.PARAM in pool_of(BY_KEY[key]))
    area_driven = any(mode_of(fixed, k) != AUTO for k in ("area_m2", "wing_loading_Nm2"))     # the aspect ratio is then ignored
    vs: list[Var] = []
    for p in PARAMS:
        if not searched(p) or not active(p, res, options):
            continue
        if p.key == "aspect_ratio" and area_driven:
            continue
        if p.kind == "airfoil":
            pool_k = pool_of(p)
            if len(pool_k) > 1:
                vs.append(Var(p.key, "airfoil", 0, len(pool_k) - 1, pool_k))
        elif p.kind in ("choice", "bool"):
            if p.key in options:
                vs.append(Var(p.key, p.kind, 0, len(options[p.key]) - 1, options[p.key]))
        elif p.key.startswith("naca_"):
            if foil_param[p.key.rsplit("_", 1)[1]]:
                lo, hi = span(p)
                vs.append(Var(p.key, "float", lo, hi))
        else:
            lo, hi = span(p)
            if hi > lo:
                vs.append(Var(p.key, "float", lo, hi))
    return vs


def apply_x(values: dict, fixed: dict, vs: list[Var], x) -> tuple[dict, dict]:
    v, f = dict(values), dict(fixed)
    for var, xi in zip(vs, x):
        f[var.key] = True
        if var.kind == "float":
            v[var.key] = float(xi)
        else:                                                  # an airfoil, a choice or a yes/no: an index into its options
            v[var.key] = var.pool[int(round(xi))]
    return v, f


def metrics_sf(case: Case, need: set) -> dict:
    q = case.quick(need)
    q.update(with_safety(q, case.sf))
    return q


def objective_value(m: dict, key: str) -> float | None:
    mk = OBJECTIVES[key][1]
    v = m.get(mk)
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return abs(v) if key == "cm" else float(v)


def constraint_values(p: dict, m: dict, base: bool = True, margin_min: float | None = None) -> list[float]:
    """Constraint functions g <= 0, each scaled to be of order one. The pitching-moment, roll and wing-loading limits belong to the design point
    (base); margin_min is the least speed margin of a flight point (None: the limit of the inputs)."""
    g = []
    if p["con_stall_on"]:
        g.append((m.get("eta_stall_first", 1.0) - p["stall_onset_max"]) / 0.2)
    if p["con_margin_on"]:
        g.append(((p["speed_margin_min"] if margin_min is None else margin_min) - m.get("speed_margin_sf", 0.0)) / 0.2)
    if p["con_util_on"]:
        g.append((m.get("util_max_sf", 9.0) - 1.0) / 0.2)
    if not base:
        return g
    if p["con_cm_on"]:
        g.append((abs(m.get("Cm_ref_sf", 9.0)) - p["cm_abs_max"]) / 0.05)
    if p["con_roll_on"] and p["aileron_on"]:
        g.append((p["roll_authority_min"] - m.get("roll_authority_0p9_sf", 0.0)) / 0.2)
    if p["con_loading_on"]:
        g.append((m.get("wing_loading_Nm2", 1e3) - p["wing_loading_max"]) / 50.0)
    return g


def constraint_names(p: dict, base: bool = True) -> list[str]:
    """What each value of constraint_values means, in the same order."""
    n = []
    if p["con_stall_on"]:
        n.append("the stall starts too far outboard")
    if p["con_margin_on"]:
        n.append("the speed margin over the stall is too small")
    if p["con_util_on"]:
        n.append("a strip is beyond its allowed lift")
    if not base:
        return n
    if p["con_cm_on"]:
        n.append("the pitching moment is too large")
    if p["con_roll_on"] and p["aileron_on"]:
        n.append("too little roll power near the stall")
    if p["con_loading_on"]:
        n.append("the wing loading is too high")
    return n


# ------------------------------------------------------------------------------------------------ one problem: inputs, flight points, objectives, limits
@dataclass
class Eval:
    """One candidate design evaluated: its resolved inputs, the results at every flight point, the weighted objective J (to minimise) and the
    limit functions g (<= 0 is met). ms is None when the candidate cannot be evaluated; ok is False when a point has no design solution (the
    design lift coefficient is beyond the stall)."""
    p: dict | None
    ms: list | None
    ok: bool
    J: float | None
    g: list | None

    @property
    def m(self) -> dict | None:
        """The results at the design point (the first flight point)."""
        return self.ms[0] if self.ms else None

    @property
    def violation(self) -> float:
        return float(sum(max(0.0, x) for x in self.g)) if self.g is not None else float("inf")

    @property
    def feasible(self) -> bool:
        return bool(self.ok and self.g is not None and all(x <= 1e-6 for x in self.g))


class Problem:
    """A design problem: which inputs are searched (free_variables), the objectives and their weights, the flight points, and the limits.
    Used by the optimiser, the sample and sweeps, and the valley analysis, so that all of them evaluate a design the same way.

    The objective J is the weighted sum, over the flight points, of the weighted objectives of the Optimise page (each normalised by its value for
    the starting design at that point); the point weights are normalised to add up to one. The limits g <= 0 of every point are concatenated."""

    def __init__(self, values: dict, fixed: dict, bounds: dict | None = None, lists: dict | None = None, weights: dict | None = None,
                 points=None, foil_pool: list | None = None, include_uncertain: bool = False, vs: list | None = None):
        from .points import point_list
        self.values, self.fixed = dict(values), dict(fixed)
        self.bounds, self.lists = dict(bounds or {}), dict(lists or {})
        w = {k: float(x) for k, x in (weights or {"ld": 1.0}).items() if x and k in OBJECTIVES}
        if not w:
            raise ValueError("give at least one objective a weight above zero")
        self.weights = w
        self.points = point_list(points)
        pw = [max(float(pt.weight), 0.0) for pt in self.points]
        if sum(pw) <= 0.0:
            pw[0] = 1.0
        self.point_weights = [x / sum(pw) for x in pw]
        self.vs = list(vs) if vs is not None else free_variables(values, fixed, bounds, foil_pool, lists, include_uncertain)
        self.need = set().union(*(OBJECTIVES[k][3] for k in w))
        self.p0 = resolve(values, fixed, self.bounds, self.lists)
        if self.p0["con_roll_on"] and self.p0["aileron_on"]:
            self.need.add("roll")
        self.baseline_ms = self.metrics(self.p0)
        self.scale = []
        for m in self.baseline_ms:
            sc = {}
            for k in w:
                v = objective_value(m, k) if m.get("ok") else None
                sc[k] = max(abs(v), 0.02) if v is not None else 1.0
            self.scale.append(sc)
        multi = len(self.points) > 1
        self.names = []
        for i, pt in enumerate(self.points):
            for nm in constraint_names(self.p0, base=(i == 0)):
                self.names.append(f"[{pt.name}] {nm}" if multi else nm)
        self.n_con = len(self.names)
        self._cache: dict = {}
        self.n_evals = 0

    # ---- evaluation
    def metrics(self, p: dict) -> list[dict]:
        """The results (with the safety factor) at every flight point."""
        from .points import cases_for
        return [metrics_sf(c, self.need) for c in cases_for(Case(p), self.points)]

    def objective(self, ms: list) -> float | None:
        tot = 0.0
        for i, m in enumerate(ms):
            for k, wk in self.weights.items():
                val = objective_value(m, k)
                if val is None:
                    return None
                tot += self.point_weights[i] * wk * OBJECTIVES[k][2] * val / self.scale[i][k]
        return tot

    def constraints(self, p: dict, ms: list) -> list[float]:
        from .points import margin_limit
        g: list[float] = []
        for i, (pt, m) in enumerate(zip(self.points, ms)):
            g += constraint_values(p, m, base=(i == 0), margin_min=None if i == 0 else margin_limit(pt, p))
        return g

    def evaluate_inputs(self, p: dict) -> Eval:
        """Evaluate resolved inputs (a design given as inputs, not as a point x of the searched variables)."""
        if any("must" in msg or "needs" in msg for msg in validate(p)):
            return Eval(p, None, False, None, None)
        try:
            ms = self.metrics(p)
        except Exception:  # noqa: BLE001  (a candidate that breaks the model is just a bad candidate)
            return Eval(p, None, False, None, None)
        if not all(m.get("ok") for m in ms):
            return Eval(p, ms, False, None, None)
        J = self.objective(ms)
        if J is None:
            return Eval(p, ms, False, None, None)
        return Eval(p, ms, True, J, self.constraints(p, ms))

    def inputs_of(self, x) -> dict:
        """The resolved inputs of a point x of the searched variables."""
        v, f = apply_x(self.values, self.fixed, self.vs, x)
        return resolve(v, f, self.bounds, self.lists)

    def evaluate(self, x) -> Eval:
        key = tuple(np.round(np.asarray(x, float), 6))
        e = self._cache.get(key)
        if e is None:
            self.n_evals += 1
            try:
                e = self.evaluate_inputs(self.inputs_of(x))
            except Exception:  # noqa: BLE001
                e = Eval(None, None, False, None, None)
            self._cache[key] = e
        return e

    def nominal_x(self) -> np.ndarray:
        """The point of the searched variables that is the nominal design (the middle of a range, the default of an auto input, the value of the
        inputs): where the inputs stand when nothing is optimised."""
        out = []
        for var in self.vs:
            val = self.p0[var.key]
            if var.kind == "float":
                out.append(min(max(float(val), var.lo), var.hi))
            else:
                try:
                    out.append(float(var.pool.index(val)))
                except ValueError:
                    out.append(0.0)
        return np.array(out, float)

    def values_at(self, x) -> dict:
        """The searched inputs at x as {key: value} (an airfoil or a choice by name)."""
        return {var.key: (float(xi) if var.kind == "float" else var.pool[int(round(xi))]) for var, xi in zip(self.vs, x)}


@dataclass
class Result:
    values: dict
    fixed: dict
    variables: list
    x: np.ndarray
    changed: dict
    baseline: dict
    best: dict
    history: list
    n_evals: int
    seconds: float
    feasible: bool
    violations: list
    weights: dict
    message: str = ""
    points: list = field(default_factory=list)           # the flight points used
    baseline_points: list = field(default_factory=list)   # results at every point, starting design
    best_points: list = field(default_factory=list)       # results at every point, optimised design
    names: list = field(default_factory=list)             # the meaning of every limit function
    problem: object = None                                 # the Problem that was solved (for the valley analysis)


def optimise(values: dict, fixed: dict, weights: dict, bounds: dict | None = None, foil_pool: list | None = None,
             effort: str = "normal", seed: int = 1, progress=None, lists: dict | None = None, points=None) -> Result:
    """Run the optimiser. weights: objective key -> weight (0 = unused). points: the flight points (None: the design point alone).
    progress(generation, best_objective, n_evals)."""
    from scipy.optimize import NonlinearConstraint, differential_evolution
    t0 = time.time()
    prob = Problem(values, fixed, bounds, lists, weights, points, foil_pool)
    vs = prob.vs
    if not vs:
        raise ValueError("nothing to search: give at least one input as a range (range switch on) or leave a design input auto (on switch off)")
    n_con = prob.n_con

    def fun(x):
        e = prob.evaluate(x)
        if e.J is not None:
            return e.J
        return 50.0 if e.ms is not None else 1e3

    def con(x):
        e = prob.evaluate(x)
        return np.array(e.g if e.g is not None else [5.0] * n_con)
    bnds = [(v.lo, v.hi) if v.kind == "float" else (0, len(v.pool) - 1) for v in vs]
    integrality = [v.kind != "float" for v in vs]
    eff = {"test": (6, 12), "quick": (8, 25), "normal": (10, 45), "thorough": (14, 90)}[effort]
    history = []

    def cb(intermediate_result):
        history.append(float(intermediate_result.fun) if hasattr(intermediate_result, "fun") else float("nan"))
        if progress:
            progress(len(history), history[-1], prob.n_evals)
    cons = NonlinearConstraint(con, -np.inf, 0.0) if n_con else ()
    res = differential_evolution(fun, bnds, integrality=integrality, constraints=cons, maxiter=eff[1], popsize=eff[0],
                                 seed=seed, tol=1e-6, polish=False, updating="immediate", callback=cb)
    e = prob.evaluate(res.x)
    v_best, f_best = apply_x(values, fixed, vs, res.x)
    viol = [x for x in (e.g or []) if x > 1e-6]
    changed = {var.key: v_best[var.key] for var in vs}
    return Result(v_best, f_best, vs, np.asarray(res.x), changed, prob.baseline_ms[0], e.m or {}, history, prob.n_evals, time.time() - t0,
                  bool(e.ok) and not viol, viol, prob.weights, str(res.message), list(prob.points), prob.baseline_ms, e.ms or [], list(prob.names), prob)
