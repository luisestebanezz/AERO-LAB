"""The inputs of the lab. Every input has an **on / off** switch and a **range** switch (Luis, 2026-11-19):

* on, range off  -> a **value**: the lab uses the number you give;
* on, range on   -> a **range** (minimum and maximum; for an airfoil a list of airfoils, for a choice or a yes/no a list of allowed options): the
  lab searches inside it (optimiser, sample, sweeps) and, for a plain analysis, takes the middle of the range;
* off            -> **auto**: the lab decides. A design input (aspect ratio, taper, sweep, twist, airfoils, incidence, ...) is searched by the
  optimiser over everything it allows; an input that describes the environment, a calibration or a model switch takes the lab's default; an
  override (wing area, wing loading, design lift coefficient) is simply not used.

Four inputs are the **minimum inputs**: the problem statement without which nothing can be computed (mass, design speed, altitude, span). They are
always on (their on / off switch is locked); they can still be a value or a range. Everything else may be auto.

The state of an input is stored in `fixed[key]`: True = value, False = range, None = auto (an override that is missing counts as auto).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from .airfoil import PARAM

MINIMUM_GROUP = "Minimum inputs"
GROUPS = [MINIMUM_GROUP, "Wing planform", "Twist and incidence", "Airfoils", "Section physics", "Ailerons", "Flaps", "Fuselage", "Flight condition",
          "Safety factor", "Limits for the optimiser"]
STATIONS = ("root", "1", "2", "tip")


@dataclass(frozen=True)
class P:
    key: str
    label: str
    unit: str
    default: object
    lo: float | None = None
    hi: float | None = None
    group: str = ""
    kind: str = "float"           # float | int | choice | bool | airfoil
    step: float | None = None
    optimisable: bool = False
    override: bool = False        # active only when set
    choices: tuple = ()
    depends: tuple | None = None  # (key, value): shown and used only while that input has that value
    help: str = ""


def _f(key, label, unit, default, lo, hi, group, step=None, opt=False, help="", depends=None, override=False):
    return P(key, label, unit, default, lo, hi, group, "float", step, opt, override, (), depends, help)


PARAMS: list[P] = []


def _add(*ps):
    PARAMS.extend(ps)


g = "Wing planform"
_add(
    _f("span_m", "Wingspan", "m", 4.0, 1.0, 6.0, g, 0.05, True, "Tip to tip."),
    _f("aspect_ratio", "Aspect ratio", "-", 12.0, 5.0, 24.0, g, 0.25, True, "Span² / area. Ignored while the area or the wing loading is set."),
    _f("area_m2", "Wing area (overrides the aspect ratio)", "m²", 1.3, 0.2, 4.0, g, 0.01, True,
       "Switch it on to fix the area (the aspect ratio then follows from the span); as a range the area is searched. Off (auto): the aspect ratio decides.", override=True),
    _f("wing_loading_Nm2", "Wing loading (overrides the aspect ratio)", "N/m²", 150.0, 40.0, 400.0, g, 1.0, True,
       "Switch it on to fix the area from the mass: S = m g / (W/S); as a range the wing loading is searched. The area has priority over it. Off (auto): not used.", override=True),
    P("n_panels", "Number of panels", "-", 2, None, None, g, "choice", None, True, False, (1, 2, 3), None, "Trapezoidal panels with their own chord, sweep and dihedral."),
    P("chord_shape", "Chord distribution", "-", "panels", None, None, g, "choice", None, True, False, ("panels", "elliptic"), None,
      "'elliptic': c = c_root sqrt(1 - eta²), not smaller than the tip chord ratio (one sweep and one dihedral, from panel 1)."),
    _f("break_1", "First break (share of the semi-span)", "-", 0.40, 0.15, 0.85, g, 0.01, True, "Used with 2 or 3 panels.", ("n_panels", (2, 3))),
    _f("break_2", "Second break (share of the semi-span)", "-", 0.75, 0.30, 0.95, g, 0.01, True, "Used with 3 panels; at least 0.02 behind the first.", ("n_panels", (3,))),
    _f("chord_ratio_1", "Chord at the first break / root chord", "-", 0.80, 0.30, 1.20, g, 0.01, True, "", ("n_panels", (2, 3))),
    _f("chord_ratio_2", "Chord at the second break / root chord", "-", 0.65, 0.25, 1.10, g, 0.01, True, "", ("n_panels", (3,))),
    _f("taper_tip", "Tip chord / root chord", "-", 0.50, 0.15, 1.00, g, 0.01, True),
    _f("sweep_1", "Sweep of panel 1 (quarter chord)", "°", 0.0, -10.0, 35.0, g, 0.5, True, "Aft positive."),
    _f("sweep_2", "Sweep of panel 2", "°", 0.0, -10.0, 35.0, g, 0.5, True, "", ("n_panels", (2, 3))),
    _f("sweep_3", "Sweep of panel 3", "°", 0.0, -10.0, 35.0, g, 0.5, True, "", ("n_panels", (3,))),
    _f("dihedral_1", "Dihedral of panel 1", "°", 2.0, -5.0, 15.0, g, 0.5, False,
       "Changes the roll-sideslip coupling (Cl_beta); the lifting line itself is planar, so no objective or limit of the lab depends on it: not optimised."),
    _f("dihedral_2", "Dihedral of panel 2", "°", 2.0, -5.0, 15.0, g, 0.5, False, "", ("n_panels", (2, 3))),
    _f("dihedral_3", "Dihedral of panel 3", "°", 2.0, -5.0, 15.0, g, 0.5, False, "", ("n_panels", (3,))),
)
g = "Twist and incidence"
_add(
    P("twist_mode", "Twist", "-", "linear", None, None, g, "choice", None, True, False, ("linear", "stations"), None,
      "'linear': from zero at the root to the tip washout. 'stations': a value at each break."),
    _f("washout_tip", "Tip twist (positive = nose down)", "°", 3.0, -2.0, 8.0, g, 0.1, True, "Washout moves the stall inboard."),
    _f("twist_1", "Twist at the first break", "°", 1.0, -2.0, 8.0, g, 0.1, True, "", ("twist_mode", ("stations",))),
    _f("twist_2", "Twist at the second break", "°", 2.0, -2.0, 8.0, g, 0.1, True, "", ("twist_mode", ("stations",))),
    _f("incidence_deg", "Incidence of the root chord on the fuselage axis", "°", 2.0, -3.0, 8.0, g, 0.1, True, "Sets the fuselage attitude and its cross-flow drag."),
)
g = "Airfoils"
for st, lab in zip(STATIONS, ("root", "first break", "second break", "tip")):
    dep = None if st in ("root", "tip") else ("n_panels", (2, 3) if st == "1" else (3,))
    _add(
        P(f"foil_{st}", f"Airfoil at the {lab}", "-", "sd7032", None, None, g, "airfoil", None, True, False, (), dep,
          "A library section, or 'param' for a NACA 4-digit section from the three inputs below. Blended linearly between stations."),
        _f(f"naca_m_{st}", f"  camber at the {lab}", "-", 0.04, 0.0, 0.09, g, 0.01, True, "Share of the chord (whole percent steps).", dep),
        _f(f"naca_p_{st}", f"  camber position at the {lab}", "-", 0.4, 0.1, 0.8, g, 0.1, True, "Share of the chord (tenths).", dep),
        _f(f"naca_t_{st}", f"  thickness at the {lab}", "-", 0.12, 0.06, 0.21, g, 0.01, True, "Share of the chord (whole percent steps).", dep),
    )
g = "Section physics"
_add(
    _f("n_crit", "Transition: n_crit", "-", 9.0, 3.0, 12.0, g, 0.5, False, "9 = clean wind tunnel / smooth wing; 5-7 = turbulence, rough surface."),
    _f("xtr_upper", "Forced transition, upper surface (share of the chord)", "-", 1.0, 0.05, 1.0, g, 0.05, False, "1 = free transition; a turbulator at x/c = 0.3 forces it there."),
    _f("xtr_lower", "Forced transition, lower surface", "-", 1.0, 0.05, 1.0, g, 0.05, False),
    _f("cd_factor", "Factor on the airfoil drag (AER-01)", "-", 1.10, 1.0, 1.30, g, 0.01, False, "NeuralFoil / XFOIL are optimistic at low Reynolds number."),
    _f("clmax_factor", "Factor on the airfoil maximum lift (AER-02)", "-", 0.95, 0.85, 1.0, g, 0.01, False, "Same reason; the lift curve is capped at this share of the section maximum."),
    P("cap_slope", "Cap the section lift slope at 2π", "-", True, None, None, g, "bool", None, False, False, (), None, "Removes a laminar-bubble artefact (up to 25 % extra slope) of NeuralFoil."),
    _f("excrescence", "Excrescence drag (hinges, gaps) as a share of profile drag (AER-04)", "-", 0.08, 0.0, 0.15, g, 0.01, False),
)
g = "Ailerons"
_add(
    P("aileron_on", "Ailerons", "-", False, None, None, g, "bool", None, False, False, (), None, "Plain ailerons on both wings; roll control is studied up to and through the stall."),
    _f("aileron_eta_in", "Aileron inner end (share of the semi-span)", "-", 0.60, 0.2, 0.9, g, 0.01, True, "", ("aileron_on", (True,))),
    _f("aileron_eta_out", "Aileron outer end", "-", 0.95, 0.4, 1.0, g, 0.01, True, "", ("aileron_on", (True,))),
    _f("aileron_chord", "Aileron chord / local chord", "-", 0.25, 0.1, 0.4, g, 0.01, True, "", ("aileron_on", (True,))),
    _f("aileron_defl_deg", "Aileron deflection (each side)", "°", 10.0, 1.0, 25.0, g, 0.5, False, "", ("aileron_on", (True,))),
)
g = "Flaps"
_add(
    P("flap_on", "Flaps", "-", False, None, None, g, "bool", None, False, False, (), None,
      "Plain flaps, thin-airfoil theory with viscous factors: lower confidence than the rest of the lab (NeuralFoil cannot do deflected flaps)."),
    _f("flap_eta_in", "Flap inner end", "-", 0.08, 0.0, 0.6, g, 0.01, True, "", ("flap_on", (True,))),
    _f("flap_eta_out", "Flap outer end", "-", 0.55, 0.1, 0.9, g, 0.01, True, "", ("flap_on", (True,))),
    _f("flap_chord", "Flap chord / local chord", "-", 0.25, 0.1, 0.4, g, 0.01, True, "", ("flap_on", (True,))),
    _f("flap_defl_deg", "Flap deflection (down)", "°", 20.0, 0.0, 45.0, g, 1.0, False, "", ("flap_on", (True,))),
    _f("flap_dclmax", "Upper limit of the flap's section maximum-lift gain", "-", 0.9, 0.0, 1.5, g, 0.05, False,
       "The gain follows the shift of the lift curve (2π x change of angle) up to this limit; the limit is of the order of Raymer's table for plain flaps (engineering judgement).", ("flap_on", (True,))),
    _f("flap_drag_f", "Flap drag factor F (Raymer: 0.0144 plain)", "-", 0.0144, 0.005, 0.03, g, 0.0005, False,
       "dCD0 = F (c_f/c)(S_flapped/S)(delta - 10°).", ("flap_on", (True,))),
    _f("ctrl_visc", "Viscous factor on the thin-airfoil effectiveness (flaps and ailerons, STB-41)", "-", 0.85, 0.6, 1.0, g, 0.01, False),
    _f("ctrl_lin_limit_deg", "Deflection up to which the effectiveness stays linear (STB-38)", "°", 12.0, 5.0, 20.0, g, 0.5, False),
    _f("ctrl_eff_25", "Effectiveness at 25° relative to the linear value (STB-39)", "-", 0.75, 0.4, 1.0, g, 0.01, False),
)
g = "Fuselage"
_add(
    P("fuse_on", "Fuselage", "-", True, None, None, g, "bool", None, False, False, (), None, "Drawn in the top view; its drag and its effect on the wing are separate switches."),
    _f("fuse_length_m", "Length", "m", 1.6, 0.4, 3.5, g, 0.05, False, "", ("fuse_on", (True,))),
    _f("fuse_width_m", "Width", "m", 0.20, 0.05, 0.6, g, 0.01, False, "", ("fuse_on", (True,))),
    _f("fuse_height_m", "Height", "m", 0.20, 0.05, 0.6, g, 0.01, False, "", ("fuse_on", (True,))),
    _f("nose_fraction", "Nose length / length", "-", 0.25, 0.1, 0.5, g, 0.01, False, "", ("fuse_on", (True,))),
    _f("tail_fraction", "Tail cone length / length", "-", 0.35, 0.1, 0.7, g, 0.01, False, "", ("fuse_on", (True,))),
    _f("wing_x_fraction", "Root leading edge, share of the length from the nose", "-", 0.35, 0.1, 0.7, g, 0.01, False, "Top view only.", ("fuse_on", (True,))),
    P("fuse_drag_on", "Include the fuselage drag", "-", True, None, None, g, "bool", None, False, False, (), ("fuse_on", (True,)),
      "Friction and form drag (Raymer), and the cross-flow drag at the fuselage attitude."),
    P("body_factor_on", "Include the fuselage effect on the induced drag", "-", True, None, None, g, "bool", None, False, False, (), ("fuse_on", (True,)),
      "Span efficiency factor 1 - 2 (d/b)² (Nita and Scholz 2012)."),
    _f("interference", "Interference factor on the fuselage drag (AER-03)", "-", 1.08, 1.0, 1.25, g, 0.01, False, "", ("fuse_on", (True,))),
    _f("crossflow_factor", "Cross-flow drag factor of the body (AER-20 x AER-21)", "-", 0.72, 0.4, 0.9, g, 0.01, False,
       "Allen and Perkins NACA TR 1048: 1.2 x 0.6.", ("fuse_on", (True,))),
)
g = "Flight condition"
_add(
    _f("mass_kg", "Mass", "kg", 20.0, 1.0, 25.0, g, 0.1, False),
    _f("altitude_m", "Altitude", "m", 500.0, 0.0, 5000.0, g, 50.0, False),
    _f("dT_K", "Temperature offset from the standard atmosphere", "K", 0.0, -30.0, 30.0, g, 1.0, False),
    _f("speed_mps", "Design speed", "m/s", 22.0, 6.0, 50.0, g, 0.5, False, "Ignored while the design lift coefficient is set."),
    _f("design_cl", "Design lift coefficient (overrides the speed)", "-", 0.7, 0.1, 1.6, g, 0.01, False, "Set it to fly at this CL: the speed follows from the mass and the area.", override=True),
    _f("load_factor", "Load factor n", "-", 1.0, 1.0, 4.0, g, 0.1, False, "Lift = n m g (a turn or a gust)."),
    _f("x_cg_mac", "Moment reference point (share of the MAC from its leading edge)", "-", 0.30, 0.0, 0.8, g, 0.01, False,
       "Pitching moments are taken about this point (the centre of gravity of the aircraft)."),
)
g = "Safety factor"
_add(
    _f("safety_factor", "Safety factor", "-", 1.5, 1.0, 2.0, g, 0.05, False,
       "Every result is shown as calculated (SF = 1) and with the factor applied in its adverse direction."),
    P("sf_drag", "Apply to drag and power (CD x SF)", "-", True, None, None, g, "bool"),
    P("sf_lift", "Apply to lift capability (CLmax / SF, stall speed x √SF, strip utilisation x SF)", "-", True, None, None, g, "bool"),
    P("sf_moment", "Apply to moments and loads (|Cm| x SF, bending x SF)", "-", True, None, None, g, "bool"),
    P("sf_control", "Apply to control authority (roll power / SF, deflection needed x SF)", "-", True, None, None, g, "bool"),
)
g = "Limits for the optimiser"
_add(
    P("con_stall_on", "Limit where the stall starts", "-", True, None, None, g, "bool", None, False, False, (), None, "The first strip to stall must be inboard of the ailerons."),
    _f("stall_onset_max", "  furthest outboard (share of the semi-span)", "-", 0.60, 0.3, 0.95, g, 0.01, False, "", ("con_stall_on", (True,))),
    P("con_margin_on", "Limit the speed margin over the stall", "-", True, None, None, g, "bool"),
    _f("speed_margin_min", "  design speed / stall speed (with the safety factor), at least", "-", 1.20, 1.05, 1.6, g, 0.01, False, "", ("con_margin_on", (True,))),
    P("con_util_on", "Limit the strip utilisation at the design point", "-", True, None, None, g, "bool", None, False, False, (), None,
      "SF x cl / cl_max of every strip at the design lift must stay below 1."),
    P("con_cm_on", "Limit the pitching moment", "-", False, None, None, g, "bool"),
    _f("cm_abs_max", "  |Cm| about the reference point at the design point (with SF), at most", "-", 0.10, 0.0, 0.4, g, 0.005, False, "", ("con_cm_on", (True,))),
    P("con_roll_on", "Keep roll authority near the stall", "-", False, None, None, g, "bool", None, False, False, (), ("aileron_on", (True,)),
      "Needs the ailerons."),
    _f("roll_authority_min", "  share of the linear roll power left at 90 % of the maximum lift, at least", "-", 0.5, 0.1, 1.0, g, 0.05, False, "", ("con_roll_on", (True,))),
    P("con_loading_on", "Limit the wing loading", "-", False, None, None, g, "bool"),
    _f("wing_loading_max", "  wing loading, at most", "N/m²", 250.0, 60.0, 500.0, g, 5.0, False, "", ("con_loading_on", (True,))),
)

# ---- the minimum inputs: the problem statement, always on, shown first
MINIMUM_INPUTS = ("mass_kg", "speed_mps", "altitude_m", "span_m")
MINIMUM_WHY = {
    "mass_kg": "What the wing has to carry: the lift equals mass x g, which sets the wing loading, the Reynolds number and the stall speed.",
    "speed_mps": "How fast it flies at the design point: sets the lift coefficient needed and the Reynolds number of every strip.",
    "altitude_m": "The air it flies in: sets the density and the viscosity.",
    "span_m": "How big the wing is. Without a size the optimiser would simply stretch the wing to the longest span allowed.",
}
_order = {k: i for i, k in enumerate(MINIMUM_INPUTS)}
PARAMS[:] = sorted((replace(p, group=MINIMUM_GROUP) if p.key in _order else p for p in PARAMS), key=lambda p: _order.get(p.key, len(_order)))

BY_KEY = {p.key: p for p in PARAMS}
assert len(BY_KEY) == len(PARAMS), "duplicate input keys"
assert set(MINIMUM_INPUTS) <= set(BY_KEY), "unknown minimum input"

VALUE, RANGE, AUTO = "value", "range", "auto"


def initial_state(p: P):
    """True = value, False = range, None = auto. Everything starts as a value except the overrides, which start off (auto)."""
    return None if p.override else True


def mode_of(fixed: dict | None, key: str) -> str:
    """'value', 'range' or 'auto' for an input. A minimum input can never be auto (it counts as a value)."""
    f = (fixed or {}).get(key, initial_state(BY_KEY[key]))
    if f is None:
        return VALUE if key in MINIMUM_INPUTS else AUTO
    return VALUE if f else RANGE


def auto_kind(p: P) -> str:
    """What 'auto' means for an input: 'searched' (the optimiser searches everything the input allows), 'default' (the lab's default),
    'unused' (an override that is not applied) or 'locked' (a minimum input: it cannot be auto)."""
    if p.key in MINIMUM_INPUTS:
        return "locked"
    if p.override:
        return "unused"
    if p.optimisable and p.kind in ("float", "airfoil"):
        return "searched"
    return "default"


def defaults() -> dict:
    return {p.key: p.default for p in PARAMS}


def default_fixed() -> dict:
    """Every input a value, except the overrides (wing area, wing loading, design CL), which start off (auto)."""
    return {p.key: initial_state(p) for p in PARAMS}


def all_auto(fixed: dict | None = None) -> dict:
    """The state with every input auto except the minimum inputs, which keep their state (a value or a range)."""
    cur = fixed or default_fixed()
    out = {}
    for p in PARAMS:
        if p.key in MINIMUM_INPUTS:
            f = cur.get(p.key, True)
            out[p.key] = True if f is None else f
        else:
            out[p.key] = None
    return out


def auto_inputs(fixed: dict | None) -> list[P]:
    """The inputs that are auto (minimum inputs never are)."""
    return [p for p in PARAMS if mode_of(fixed, p.key) == AUTO]


DEFAULT_AIRFOIL_POOL = ["sd7032", "sd7037", "naca2412", "naca4412", "e387", "naca23012", "naca642415"]


def options_of(p: P) -> list:
    """Every option of a choice or a yes/no input."""
    return [False, True] if p.kind == "bool" else list(p.choices)


def default_ranges(values: dict | None = None) -> dict:
    """A range for every numeric input: around its value, plus and minus 25 % of the allowed span, clipped to the allowed span."""
    v = values or defaults()
    out = {}
    for p in PARAMS:
        if p.kind == "float":
            span = p.hi - p.lo
            c = float(v.get(p.key, p.default))
            out[p.key] = (round(max(p.lo, c - 0.25 * span), 6), round(min(p.hi, c + 0.25 * span), 6))
    return out


def default_lists() -> dict:
    """The options an airfoil station, a choice or a yes/no may take when it is given as a range: the airfoil pool, or every option."""
    out = {}
    for p in PARAMS:
        if p.kind == "airfoil":
            out[p.key] = list(DEFAULT_AIRFOIL_POOL)
        elif p.kind in ("choice", "bool"):
            out[p.key] = options_of(p)
    return out


def _nominal(p: P, v, ranges: dict | None, lists: dict | None):
    """The value of an input given as a range, for a plain analysis: the middle of a numeric range, the first airfoil of a list, the current
    value of a choice if it is allowed (else the first allowed option)."""
    if p.kind == "float":
        if ranges and p.key in ranges:
            lo, hi = ranges[p.key]
            return 0.5 * (float(lo) + float(hi))
        return float(v) if p.override else p.default
    allowed = (lists or {}).get(p.key) or []
    if p.kind == "airfoil":
        return allowed[0] if allowed else p.default
    cur = coerce(p, v)
    if allowed:
        allowed = [coerce(p, a) for a in allowed]
        return cur if cur in allowed else allowed[0]
    return cur


def resolve(values: dict, fixed: dict | None = None, ranges: dict | None = None, lists: dict | None = None) -> dict:
    """The numbers the model uses. Value: the number given. Range: the middle of the range (an airfoil list: its first airfoil; a choice: the
    current value if allowed). Auto: the lab's default; an override that is auto (or missing) is None."""
    out = {}
    for p in PARAMS:
        v = values.get(p.key, p.default)
        mode = mode_of(fixed, p.key)
        if mode == AUTO:
            out[p.key] = None if p.override else p.default
        elif mode == VALUE:
            out[p.key] = float(v) if p.override else coerce(p, v)
        else:
            out[p.key] = _nominal(p, v, ranges, lists)
    return out


def coerce(p: P, v):
    if p.kind == "bool":
        return bool(v)
    if p.kind in ("choice", "airfoil"):
        if p.kind == "choice" and p.choices and isinstance(p.choices[0], int):
            return int(v)
        return v
    if p.kind == "int":
        return int(round(float(v)))
    return float(v)


def active(p: P, values: dict, options: dict | None = None) -> bool:
    """Whether the input is used given the other inputs (its dependency). `options` maps an input to the options it may take (a searched choice): the
    input counts as used if any of them uses it."""
    if p.depends is None:
        return True
    k, allowed = p.depends
    allowed = allowed if isinstance(allowed, tuple) else (allowed,)
    possible = (options or {}).get(k) or [values.get(k, BY_KEY[k].default)]
    return any(x in allowed for x in possible)


def range_options(fixed: dict | None, lists: dict | None) -> dict:
    """The options of every choice or yes/no input that is given as a range (more than one option): used to decide which inputs are used."""
    out = {}
    for p in PARAMS:
        if p.kind in ("choice", "bool") and mode_of(fixed, p.key) == RANGE:
            allowed = [coerce(p, a) for a in ((lists or {}).get(p.key) or options_of(p))]
            if len(allowed) > 1:
                out[p.key] = allowed
    return out


def validate(p: dict) -> list[str]:
    """Problems that make the input set meaningless (the lab refuses to run) and warnings."""
    msgs = []
    if p["n_panels"] == 3 and p["break_2"] <= p["break_1"] + 0.019:
        msgs.append("The second break must be at least 0.02 of the semi-span behind the first (it is moved).")
    if p["aileron_on"] and p["aileron_eta_out"] <= p["aileron_eta_in"] + 0.05:
        msgs.append("The aileron needs a span of at least 0.05 of the semi-span.")
    if p["flap_on"] and p["flap_eta_out"] <= p["flap_eta_in"] + 0.05:
        msgs.append("The flap needs a span of at least 0.05 of the semi-span.")
    if p["aileron_on"] and p["flap_on"] and p["flap_eta_out"] > p["aileron_eta_in"] + 1e-9:
        msgs.append("The flap and the aileron overlap in span.")
    if p["fuse_on"] and p["fuse_width_m"] > 0.4 * p["span_m"]:
        msgs.append("The fuselage is wider than 40 % of the span.")
    if p["nose_fraction"] + p["tail_fraction"] > 0.95:
        msgs.append("Nose and tail cone are longer than the fuselage.")
    if p["xtr_upper"] < 1.0 or p["xtr_lower"] < 1.0:
        msgs.append("Forced transition is on: the polars are those of a tripped section.")
    return msgs


JSON_FORMAT = 2     # 2: fixed = true (value) / false (range) / null (auto). 1: an override that is not set was false


def to_json(values: dict, fixed: dict, ranges: dict | None = None, lists: dict | None = None, points: list | None = None) -> str:
    """The inputs as a file. points: the flight points (a list of dictionaries, points.to_dicts), saved with them."""
    d = {"format": JSON_FORMAT, "values": values, "fixed": fixed, "ranges": {k: list(v) for k, v in (ranges or {}).items()}, "lists": lists or {}}
    if points is not None:
        d["points"] = points
    return json.dumps(d, indent=1)


def from_json_full(text: str) -> tuple[dict, dict, dict, dict]:
    """(values, fixed, ranges, lists) from a saved file; inputs missing from the file keep their defaults."""
    d = json.loads(text)
    fmt = int(d.get("format", 1))
    vals, fx = defaults(), default_fixed()
    vals.update({k: v for k, v in d.get("values", {}).items() if k in BY_KEY})
    for k, v in d.get("fixed", {}).items():
        if k not in BY_KEY:
            continue
        p = BY_KEY[k]
        if fmt >= 2:
            fx[k] = None if v is None else bool(v)
        elif p.override:
            fx[k] = True if v else None                       # an older file: an override that was not set is auto
        elif p.kind in ("choice", "bool"):
            fx[k] = True                                      # an older file: yes/no inputs and choices were always values
        else:
            fx[k] = bool(v)
    for k in MINIMUM_INPUTS:
        if fx[k] is None:
            fx[k] = True
    rng = default_ranges(vals)
    rng.update({k: (float(v[0]), float(v[1])) for k, v in d.get("ranges", {}).items() if k in BY_KEY and BY_KEY[k].kind == "float"})
    lst = default_lists()
    for k, v in d.get("lists", {}).items():
        if k not in BY_KEY or not v:
            continue
        p = BY_KEY[k]
        if p.kind == "airfoil":
            lst[k] = list(v)
        elif p.kind in ("choice", "bool"):
            ok = [x for x in v if x in options_of(p)]
            if ok:
                lst[k] = ok
    return vals, fx, rng, lst


def from_json(text: str) -> tuple[dict, dict]:
    vals, fx, _, _ = from_json_full(text)
    return vals, fx
