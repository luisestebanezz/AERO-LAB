"""HORYZN wing aerodynamics lab: a stand-alone wing study derived from the scout configurator (2026-11-19).

Wing only (plus a simple fuselage for the top view and, optionally, its drag): multi-panel planform, twist table,
a different airfoil at each station, non-linear lifting line with the section polars (stall, pitching moment, roll
control near the stall), flaps and ailerons, a safety factor from 1 to 2 on every result, and an optimiser over the
inputs the user leaves free. Equations: docs/physics-lab.md.
"""
__version__ = "1.0.0"
