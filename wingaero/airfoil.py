"""Airfoil sections: coordinates, a NACA 4-digit generator, and NeuralFoil polars computed on demand.

NeuralFoil (Sharpe and Hansman 2025, a neural-network model trained on XFOIL) takes about 50 ms for ten Reynolds
numbers, so the polars are not stored: transition (n_crit, forced transition on each surface) is an input of the lab and
every section is computed when first needed and cached. The raw tables against angle of attack (including the stall and
the region beyond it, up to 25 degrees) feed the non-linear lifting line; the lift-curve slope, zero-lift angle, maximum
lift and moment at zero lift are derived from them for each Reynolds number.

Valid range: Reynolds number 6e4 to 2e6 (clamped outside), angle of attack -20 to 25 degrees (sections of very high camber have zero-lift angles near -10), incompressible.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from .core import DATA

ALPHA_DEG = np.arange(-20.0, 25.001, 0.5)
D_ALPHA = 0.5
REYNOLDS = np.array([6e4, 1e5, 1.5e5, 2e5, 3e5, 4e5, 6e5, 8e5, 1.2e6, 2e6])
LOG_RE = np.log(REYNOLDS)
MODEL_SIZE = "large"
PARAM = "param"          # station source meaning 'NACA 4-digit with the station's own m, p, t'
_META = json.loads((DATA / "airfoils" / "polars_meta.json").read_text(encoding="utf-8"))["airfoils"]
_CUSTOM: dict[str, tuple[np.ndarray, str]] = {}


# ------------------------------------------------------------------------------------------------ coordinates
def naca4_coordinates(m: float, p: float, t: float, n: int = 100) -> np.ndarray:
    """NACA 4-digit section (camber m, its position p, thickness t, all as fractions of the chord): closed trailing
    edge, cosine spacing, Selig order (upper surface from the trailing edge to the leading edge, then the lower)."""
    beta = np.linspace(0.0, math.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m <= 0.0 or p <= 0.0:
        yc, dyc = np.zeros_like(x), np.zeros_like(x)
    else:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.vstack([np.column_stack([xu[::-1], yu[::-1]]), np.column_stack([xl[1:], yl[1:]])])


def split_surfaces(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(upper, lower) from the leading edge to the trailing edge, for Selig-ordered coordinates."""
    i = int(np.argmin(c[:, 0]))
    return c[: i + 1][::-1], c[i:]


def shape_numbers(c: np.ndarray) -> dict:
    """Thickness and camber distributions on a common chordwise grid, and their maxima."""
    up, lo = split_surfaces(c)
    x = np.linspace(0.0, 1.0, 201)
    yu, yl = np.interp(x, up[:, 0], up[:, 1]), np.interp(x, lo[:, 0], lo[:, 1])
    t, cam = yu - yl, 0.5 * (yu + yl)
    k = int(np.argmax(np.abs(cam)))
    return {"x": x, "yu": yu, "yl": yl, "thickness": t, "camber": cam, "t_max": float(t.max()), "x_t": float(x[int(np.argmax(t))]),
            "camber_max": float(cam[k]), "x_camber": float(x[k])}


def parametric_name(m: float, p: float, t: float) -> str:
    """Name of the NACA 4-digit section on the lab's coarse grid (m and t in whole percent, p in tenths of the chord);
    the grid keeps the number of distinct sections, and so of NeuralFoil runs, small during an optimisation."""
    mi, pi_, ti = int(round(m * 100)), int(round(p * 10)), int(round(t * 100))
    mi, pi_, ti = min(max(mi, 0), 9), min(max(pi_, 1), 8), min(max(ti, 5), 24)
    return f"naca4_m{mi}_p{pi_}_t{ti}"


def parse_parametric(name: str) -> tuple[float, float, float] | None:
    if not name.startswith("naca4_m"):
        return None
    try:
        a, b, c = name[len("naca4_"):].split("_")
        return int(a[1:]) / 100.0, int(b[1:]) / 10.0, int(c[1:]) / 100.0
    except (ValueError, IndexError):
        return None


def display_name(name: str) -> str:
    par = parse_parametric(name)
    if par is not None:
        return f"NACA {int(round(par[0] * 100))}{int(round(par[1] * 10))}{int(round(par[2] * 100)):02d} (parametric)"
    if name in _CUSTOM:
        return _CUSTOM[name][1]
    n = name.lower()
    if n.startswith("naca"):
        code = n[4:]
        if len(code) == 6 and code[:2] in ("63", "64", "65", "66") and code.isdigit():
            return f"NACA {code[:2]}{'₀₁₂₃₄₅₆₇₈₉'[int(code[2])]}-{code[3:]}"
        return f"NACA {code.upper()}"
    return name.upper()


def library_names(include_reflex: bool = False) -> list[str]:
    return sorted(n for n, m in _META.items() if include_reflex or m["family"] != "reflex") + sorted(_CUSTOM)


def library_info(name: str) -> dict:
    par = parse_parametric(name)
    if par is not None:
        return {"family": "conventional", "series": "NACA 4-digit (parametric)", "why": "Built from camber, camber position and thickness"}
    if name in _CUSTOM:
        return {"family": "conventional", "series": "User coordinates", "why": "Added by the user"}
    m = _META.get(name, {})
    return {"family": m.get("family", "conventional"), "series": m.get("series", ""), "why": m.get("why", "")}


def register_custom(name: str, coords: np.ndarray, label: str) -> None:
    """Add a section from Selig-ordered coordinates (trailing edge, upper surface, leading edge, lower surface)."""
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or len(c) < 20:
        raise ValueError("coordinates must be an N x 2 table with at least 20 points")
    if not (0.98 <= c[:, 0].max() <= 1.02 and -0.02 <= c[:, 0].min() <= 0.02):
        raise ValueError("chordwise coordinates must run from 0 to 1")
    _CUSTOM[name] = (c, label)
    get_airfoil.cache_clear()


def coordinates_for(name: str) -> np.ndarray:
    par = parse_parametric(name)
    if par is not None:
        return naca4_coordinates(*par)
    if name in _CUSTOM:
        return _CUSTOM[name][0]
    return np.loadtxt(DATA / "airfoils" / f"{name}.dat")


# ------------------------------------------------------------------------------------------------ NeuralFoil tables
def compute_tables(coords: np.ndarray, n_crit: float = 9.0, xtr_upper: float = 1.0, xtr_lower: float = 1.0):
    """cl, cd, cm and NeuralFoil's analysis confidence on the (Reynolds, angle of attack) grid."""
    import neuralfoil as nf
    shape = (len(REYNOLDS), len(ALPHA_DEG))
    cl, cd, cm, conf = (np.zeros(shape) for _ in range(4))
    for i, re in enumerate(REYNOLDS):
        r = nf.get_aero_from_coordinates(coords, alpha=ALPHA_DEG, Re=float(re), n_crit=float(n_crit), xtr_upper=float(xtr_upper),
                                         xtr_lower=float(xtr_lower), model_size=MODEL_SIZE)
        cl[i], cd[i], cm[i], conf[i] = r["CL"], r["CD"], r["CM"], r["analysis_confidence"]
    return cl, cd, cm, conf


class Airfoil:
    """One section with its tables at ten Reynolds numbers and the numbers derived from them."""

    def __init__(self, name: str, coords: np.ndarray, cl, cd, cm, conf, n_crit: float, xtr_upper: float, xtr_lower: float):
        self.name, self.coords = name, coords
        self.label = display_name(name)
        info = library_info(name)
        self.family, self.series, self.why = info["family"], info["series"], info["why"]
        self.n_crit, self.xtr_upper, self.xtr_lower = n_crit, xtr_upper, xtr_lower
        self.cl, self.cd, self.cm, self.conf = cl, cd, cm, conf
        self.shape = shape_numbers(coords)
        self.thickness = self.shape["t_max"]
        n = len(REYNOLDS)
        pos = ALPHA_DEG >= 0.0
        lin = (ALPHA_DEG >= -2.0) & (ALPHA_DEG <= 5.0)
        self.clmax, self.alpha_stall, self.slope, self.alpha0, self.cm0 = (np.zeros(n) for _ in range(5))
        self.cdmin, self.conf_min = np.zeros(n), np.zeros(n)
        for i in range(n):
            ap = ALPHA_DEG[pos]
            j = int(np.argmax(cl[i, pos]))
            self.clmax[i], self.alpha_stall[i] = cl[i, pos][j], ap[j]
            s, b = np.polyfit(np.radians(ALPHA_DEG[lin]), cl[i, lin], 1)
            self.slope[i], self.alpha0[i] = s, -b / s
            self.cm0[i] = float(np.interp(math.degrees(-b / s), ALPHA_DEG, cm[i]))
            rng = (ALPHA_DEG >= -2.0) & (ALPHA_DEG <= self.alpha_stall[i])
            self.cdmin[i] = float(cd[i, rng].min())
            self.conf_min[i] = float(conf[i, rng].min())

    # ---- Reynolds interpolation (linear in log Re, clamped to the table)
    @staticmethod
    def _bracket(re):
        x = np.log(np.clip(np.atleast_1d(np.asarray(re, dtype=float)), REYNOLDS[0], REYNOLDS[-1]))
        i = np.clip(np.searchsorted(LOG_RE, x) - 1, 0, len(LOG_RE) - 2)
        return i, (x - LOG_RE[i]) / (LOG_RE[i + 1] - LOG_RE[i])

    def table_at_re(self, re):
        """(cl, cd, cm), each of shape (len(re), len(ALPHA_DEG)), at the given Reynolds numbers."""
        i, t = self._bracket(re)
        w = t[:, None]
        return tuple((1 - w) * a[i] + w * a[i + 1] for a in (self.cl, self.cd, self.cm))

    def per_re(self, values: np.ndarray, re):
        i, t = self._bracket(re)
        return (1 - t) * values[i] + t * values[i + 1]

    def cl_max(self, re):
        return self.per_re(self.clmax, re)

    def lift_slope(self, re):
        return self.per_re(self.slope, re)

    def zero_lift_angle(self, re):
        return self.per_re(self.alpha0, re)

    def polar(self, re: float) -> dict:
        """Everything for the plots at one Reynolds number: the whole table and the derived numbers."""
        cl, cd, cm = (a[0] for a in self.table_at_re(re))
        conf = self.per_re_conf(re)
        pre = ALPHA_DEG <= float(self.per_re(self.alpha_stall, re)[0]) + 1e-9
        return {"alpha_deg": ALPHA_DEG, "cl": cl, "cd": cd, "cm": cm, "confidence": conf, "pre_stall": pre,
                "cl_max": float(self.cl_max(re)[0]), "alpha_stall_deg": float(self.per_re(self.alpha_stall, re)[0]),
                "cd_min": float(self.per_re(self.cdmin, re)[0]), "alpha0_deg": math.degrees(float(self.zero_lift_angle(re)[0])),
                "cm0": float(self.per_re(self.cm0, re)[0]), "lift_slope_per_deg": float(self.lift_slope(re)[0]) * math.pi / 180.0,
                "ld_max": float(np.max(cl[pre & (cl > 0)] / cd[pre & (cl > 0)])), "confidence_min": float(self.per_re(self.conf_min, re)[0])}

    def per_re_conf(self, re):
        i, t = self._bracket(re)
        return (1 - t[0]) * self.conf[i[0]] + t[0] * self.conf[i[0] + 1]


@lru_cache(maxsize=1024)
def get_airfoil(name: str, n_crit: float = 9.0, xtr_upper: float = 1.0, xtr_lower: float = 1.0) -> Airfoil:
    coords = coordinates_for(name)
    cl, cd, cm, conf = compute_tables(coords, n_crit, xtr_upper, xtr_lower)
    return Airfoil(name, coords, cl, cd, cm, conf, n_crit, xtr_upper, xtr_lower)
