"""Constants, the standard atmosphere and small numeric helpers (SI units; angles in degrees at the interfaces,
radians inside)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
G0 = 9.80665
RHO0 = 1.225
T0, P0, LAPSE, R_AIR, GAMMA = 288.15, 101325.0, 0.0065, 287.053, 1.4


@dataclass(frozen=True)
class Air:
    altitude_m: float
    temperature_K: float
    pressure_Pa: float
    density: float
    viscosity: float
    speed_of_sound: float

    @property
    def temperature_C(self) -> float:
        return self.temperature_K - 273.15

    @property
    def sigma(self) -> float:
        return self.density / RHO0


def isa(altitude_m: float, offset_K: float = 0.0) -> Air:
    """International Standard Atmosphere, troposphere (ICAO Doc 7488), with a temperature offset; Sutherland viscosity."""
    h = max(0.0, min(float(altitude_m), 11000.0))
    t_std = T0 - LAPSE * h
    p = P0 * (t_std / T0) ** (G0 / (R_AIR * LAPSE))
    t = t_std + offset_K
    rho = p / (R_AIR * t)
    mu = 1.458e-6 * t ** 1.5 / (t + 110.4)
    return Air(h, t, p, rho, mu, math.sqrt(GAMMA * R_AIR * t))


def bisect(f, lo: float, hi: float, tol: float = 1e-8, itmax: int = 100):
    """Root of f on [lo, hi] by bisection; None without a sign change."""
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if flo * fhi > 0:
        return None
    for _ in range(itmax):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(hi - lo) < tol * max(1.0, abs(mid)):
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def golden_min(f, lo: float, hi: float, tol: float = 1e-4):
    """Minimum of a unimodal function on [lo, hi] (golden section). Returns (x, f(x))."""
    gr = (math.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = f(c), f(d)
    while abs(hi - lo) > tol * max(1.0, abs(hi)):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = f(d)
    x = 0.5 * (lo + hi)
    return x, f(x)
