"""Flight points: the same wing evaluated in several flight conditions at once (Luis, 2026-11-19: multi-point optimisation).

A *point* is the flight condition of the inputs changed by a few factors: speed (a share of the design speed), altitude (an offset), mass (a share),
load factor. The first point is always the design point of the inputs themselves (its factors are ignored). Every other point shares the wing
(planform, airfoils, section physics) of the first: only the speed, the air, the weight and the load factor change, so a point costs about 30 ms.

Each point has a **weight** in the objective (the objectives of the Optimise page are evaluated at every point and combined with these weights, which are
normalised to add up to one; a weight of zero makes it a point that only checks the limits) and its own **minimum speed margin** over the stall (with the
safety factor). Every point is tested against the stall-position limit and the strip-utilisation limit; the pitching-moment, roll and wing-loading limits
belong to the design point only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class FlightPoint:
    name: str
    on: bool = True
    speed_factor: float = 1.0               # x the design speed of the inputs
    altitude_delta_m: float = 0.0           # added to the altitude of the inputs
    mass_factor: float = 1.0                # x the mass of the inputs
    load_factor: float | None = None        # None: the load factor of the inputs
    weight: float = 1.0                     # share of the objective (0: the point only checks the limits)
    margin_min: float | None = None         # least design speed / stall speed with the safety factor; None: the limit of the inputs

    def label(self) -> str:
        bits = []
        if abs(self.speed_factor - 1.0) > 1e-9:
            bits.append(f"{self.speed_factor:g} x speed")
        if self.altitude_delta_m:
            bits.append(f"{self.altitude_delta_m:+g} m")
        if abs(self.mass_factor - 1.0) > 1e-9:
            bits.append(f"{self.mass_factor:g} x mass")
        if self.load_factor is not None:
            bits.append(f"n = {self.load_factor:g}")
        return f"{self.name} ({', '.join(bits)})" if bits else self.name


CRUISE = FlightPoint("Cruise", weight=1.0)
SLOW = FlightPoint("Slow flight", speed_factor=0.85, weight=0.5, margin_min=1.05)
MANOEUVRE = FlightPoint("Manoeuvre or gust", speed_factor=1.3, load_factor=2.5, weight=0.0, margin_min=1.0)
STANDARD_POINTS = (CRUISE, SLOW, MANOEUVRE)


def point_list(points) -> list[FlightPoint]:
    """The points that are used: the first (the design point of the inputs, always) and every other one that is on."""
    if not points:
        return [CRUISE]
    pts = list(points)
    return [pts[0]] + [p for p in pts[1:] if p.on]


def condition_inputs(p: dict, pt: FlightPoint, base_speed: float) -> dict:
    """The resolved inputs of a point: the same wing, another flight condition. base_speed is the design speed of the inputs (m/s)."""
    q = dict(p)
    q["design_cl"] = None                                  # the point sets the speed itself
    q["speed_mps"] = float(base_speed) * float(pt.speed_factor)
    q["altitude_m"] = max(0.0, float(p["altitude_m"]) + float(pt.altitude_delta_m))
    q["mass_kg"] = float(p["mass_kg"]) * float(pt.mass_factor)
    if pt.load_factor is not None:
        q["load_factor"] = float(pt.load_factor)
    return q


def cases_for(case, points) -> list:
    """One Case per point (the first is `case` itself), sharing the wing."""
    out = []
    for i, pt in enumerate(point_list(points)):
        if i == 0:
            out.append(case)
        else:
            out.append(case.variant(condition_inputs(case.p, pt, case.V), pt.name))
    return out


def margin_limit(pt: FlightPoint, p: dict) -> float:
    """The least speed margin of a point: its own, or the limit of the inputs."""
    return float(pt.margin_min) if pt.margin_min is not None else float(p["speed_margin_min"])


# ------------------------------------------------------------------------------------------------ files
def to_dicts(points) -> list[dict]:
    return [asdict(p) for p in (points or [])]


def from_dicts(rows) -> list[FlightPoint]:
    """Points from saved dictionaries; unknown keys are ignored, a bad entry is skipped."""
    names = {f.name for f in fields(FlightPoint)}
    out = []
    for r in rows or []:
        try:
            d = {k: v for k, v in r.items() if k in names}
            d.setdefault("name", f"Point {len(out) + 1}")
            d["on"] = bool(d.get("on", True))
            for k in ("speed_factor", "altitude_delta_m", "mass_factor", "weight"):
                if k in d:
                    d[k] = float(d[k])
            for k in ("load_factor", "margin_min"):
                if d.get(k) is not None:
                    d[k] = float(d[k])
            pt = FlightPoint(**d)
            if pt.speed_factor > 0 and pt.mass_factor > 0 and pt.weight >= 0:
                out.append(pt)
        except (TypeError, ValueError, AttributeError):
            continue
    return out
