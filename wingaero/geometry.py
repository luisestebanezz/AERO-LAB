"""Wing planform (up to three trapezoidal panels or an elliptical chord), twist table, dihedral per panel, and a simple
fuselage for the top view and its drag.

Conventions: x runs aft from the leading edge of the root chord, y to the right, angles in degrees at the interface and
radians inside. Twist is positive for a tip nose-down (washout); the sweep of each panel is that of its quarter-chord
line (aft positive); chords are given at the panel stations as a share of the root chord.
"""
from __future__ import annotations

import math

import numpy as np

from .core import G0


class Wing:
    def __init__(self, span: float, area: float, eta, ratio, sweep_deg, dihedral_deg, twist_deg):
        self.span, self.area = float(span), float(area)
        self.eta, self.ratio = np.asarray(eta, float), np.asarray(ratio, float)
        n = len(self.eta) - 1
        self.sweep_deg = np.broadcast_to(np.asarray(sweep_deg, float), (n,)).copy()
        self.dihedral_deg = np.broadcast_to(np.asarray(dihedral_deg, float), (n,)).copy()
        self.twist_deg = np.asarray(twist_deg, float)
        if n < 1 or self.eta[0] != 0.0 or self.eta[-1] != 1.0 or np.any(np.diff(self.eta) <= 0):
            raise ValueError("wing stations must increase from 0 to 1")
        if len(self.ratio) != n + 1 or len(self.twist_deg) != n + 1:
            raise ValueError("one chord ratio and one twist per station")
        d_eta = np.diff(self.eta)
        r0, r1 = self.ratio[:-1], self.ratio[1:]
        self.root_chord = self.area / (self.span * float(np.sum(0.5 * (r0 + r1) * d_eta)))
        self.aspect_ratio = self.span ** 2 / self.area
        self.tip_chord = self.root_chord * float(self.ratio[-1])
        self.shift = np.concatenate([[0.0], np.cumsum(0.5 * self.span * d_eta * np.tan(np.radians(self.sweep_deg)))])
        # mean aerodynamic chord and its spanwise position (exact for piecewise-linear chords)
        i_r2 = float(np.sum(d_eta * (r0 ** 2 + r0 * r1 + r1 ** 2) / 3.0))
        m = (r1 - r0) / d_eta
        i_re = float(np.sum(r0 * (self.eta[1:] ** 2 - self.eta[:-1] ** 2) / 2.0
                            + m * ((self.eta[1:] ** 3 - self.eta[:-1] ** 3) / 3.0 - self.eta[:-1] * (self.eta[1:] ** 2 - self.eta[:-1] ** 2) / 2.0)))
        self.mac = self.span * self.root_chord ** 2 / self.area * i_r2
        self.y_mac = self.span ** 2 * self.root_chord / (2.0 * self.area) * i_re
        self.eta_mac = self.y_mac / (0.5 * self.span)
        self.x_le_mac = float(self.x_le(self.eta_mac))
        self.n_panels = n
        s_panel = self.span * self.root_chord * 0.5 * (r0 + r1) * d_eta        # area of both halves of each panel
        self.panel_area = s_panel
        self.mean_sweep_deg = float(np.sum(self.sweep_deg * s_panel) / np.sum(s_panel))

    # ---- distributions (eta = y / (b/2), scalar or array, symmetric)
    def chord(self, eta):
        return self.root_chord * np.interp(np.abs(eta), self.eta, self.ratio)

    def shift_at(self, eta):
        """Quarter-chord line behind the root quarter-chord point (m)."""
        return np.interp(np.abs(eta), self.eta, self.shift)

    def x_qc(self, eta):
        return 0.25 * self.root_chord + self.shift_at(eta)

    def x_le(self, eta):
        return self.x_qc(eta) - 0.25 * self.chord(eta)

    def x_te(self, eta):
        return self.x_le(eta) + self.chord(eta)

    def twist_rad(self, eta):
        """Geometric angle of the section against the root chord (radians, negative for washout)."""
        return -np.radians(np.interp(np.abs(eta), self.eta, self.twist_deg))

    def panel_index(self, eta):
        e = np.abs(np.atleast_1d(eta))
        return np.clip(np.searchsorted(self.eta, e, side="right") - 1, 0, self.n_panels - 1)

    def dihedral_rad(self, eta):
        return np.radians(self.dihedral_deg[self.panel_index(eta)])

    def cos_sweep(self, eta):
        return np.cos(np.radians(self.sweep_deg[self.panel_index(eta)]))

    @property
    def sweep_le_deg(self) -> float:
        return math.degrees(math.atan((self.x_le(1.0) - self.x_le(0.0)) / (0.5 * self.span)))

    def outline(self) -> dict:
        """Points of the right half-planform at the stations (for the top view)."""
        e = self.eta
        return {"y": e * 0.5 * self.span, "x_le": self.x_le(e), "x_te": self.x_te(e), "x_qc": self.x_qc(e), "chord": self.chord(e)}


def broadcast_stations(p: dict) -> dict:
    """Station arrays of a wing from the resolved inputs: chord ratios, sweep, dihedral and twist at the panel stations."""
    n_panels = int(p["n_panels"])
    b1, b2 = float(p["break_1"]), float(p["break_2"])
    if p["chord_shape"] == "elliptic":
        eta = np.linspace(0.0, 1.0, 25)
        ratio = np.maximum(np.sqrt(np.maximum(1.0 - eta ** 2, 0.0)), float(p["taper_tip"]))
        return {"eta": eta, "ratio": ratio, "sweep": np.full(24, float(p["sweep_1"])), "dihedral": np.full(24, float(p["dihedral_1"])),
                "twist": float(p["washout_tip"]) * eta, "foil_eta": np.array([0.0, 1.0]), "foil_keys": ["root", "tip"]}
    if n_panels == 1:
        eta, ratio = [0.0, 1.0], [1.0, p["taper_tip"]]
        twist = [0.0, p["washout_tip"]]
        foil_keys = ["root", "tip"]
    elif n_panels == 2:
        eta, ratio = [0.0, b1, 1.0], [1.0, p["chord_ratio_1"], p["taper_tip"]]
        twist = [0.0, p["twist_1"], p["washout_tip"]]
        foil_keys = ["root", "1", "tip"]
    else:
        if b2 <= b1 + 0.02:
            b2 = min(b1 + 0.02, 0.99)
        eta, ratio = [0.0, b1, b2, 1.0], [1.0, p["chord_ratio_1"], p["chord_ratio_2"], p["taper_tip"]]
        twist = [0.0, p["twist_1"], p["twist_2"], p["washout_tip"]]
        foil_keys = ["root", "1", "2", "tip"]
    eta = np.asarray(eta, float)
    if p["twist_mode"] == "linear":
        twist = float(p["washout_tip"]) * eta
    sweep = [p["sweep_1"], p["sweep_2"], p["sweep_3"]][:n_panels]
    dih = [p["dihedral_1"], p["dihedral_2"], p["dihedral_3"]][:n_panels]
    return {"eta": eta, "ratio": np.asarray(ratio, float), "sweep": np.asarray(sweep, float), "dihedral": np.asarray(dih, float),
            "twist": np.asarray(twist, float), "foil_eta": eta.copy(), "foil_keys": foil_keys}


def wing_area(p: dict) -> tuple[float, str]:
    """Wing area and how it was fixed: the area itself, the wing loading with the mass, or span and aspect ratio."""
    if p.get("area_m2") is not None:
        return float(p["area_m2"]), "area"
    if p.get("wing_loading_Nm2") is not None:
        return float(p["mass_kg"]) * G0 / float(p["wing_loading_Nm2"]), "wing loading"
    return float(p["span_m"]) ** 2 / float(p["aspect_ratio"]), "span and aspect ratio"


def build_wing(p: dict) -> Wing:
    s = broadcast_stations(p)
    area, _ = wing_area(p)
    return Wing(p["span_m"], area, s["eta"], s["ratio"], s["sweep"], s["dihedral"], s["twist"])


class Fuselage:
    """Streamlined body: an elliptical nose, a constant section and a tail cone (top view and drag)."""

    def __init__(self, length: float, width: float, height: float, nose_fraction: float, tail_fraction: float):
        self.length, self.width, self.height = float(length), float(width), float(height)
        self.nose_fraction = float(min(max(nose_fraction, 0.05), 0.6))
        self.tail_fraction = float(min(max(tail_fraction, 0.05), 0.9 - self.nose_fraction))
        self.diameter = math.sqrt(self.width * self.height)          # equivalent circular section
        self.fineness = self.length / self.diameter
        self.plan_area = math.pi / 4.0 * self.length * self.diameter

    def outline(self, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
        """(x, half-width) along the body, nose at x = 0."""
        ln, lt = self.nose_fraction * self.length, self.tail_fraction * self.length
        xn = np.linspace(0.0, ln, n)
        hn = 0.5 * self.width * np.sqrt(np.clip(1.0 - ((ln - xn) / ln) ** 2, 0.0, 1.0))
        xc = np.array([ln, self.length - lt])
        hc = np.full(2, 0.5 * self.width)
        xt = np.linspace(self.length - lt, self.length, n)
        ht = 0.5 * self.width * (1.0 - 0.92 * (xt - (self.length - lt)) / lt)
        return np.concatenate([xn, xc, xt]), np.concatenate([hn, hc, ht])


def build_fuselage(p: dict) -> Fuselage | None:
    if not p.get("fuse_on"):
        return None
    return Fuselage(p["fuse_length_m"], p["fuse_width_m"], p["fuse_height_m"], p["nose_fraction"], p["tail_fraction"])
