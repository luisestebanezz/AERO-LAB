# HORYZN wing aerodynamics lab

A stand-alone study of **the wing** (and a simple fuselage), derived from the scout configurator. It runs locally in your browser. No tail, propeller or installation effects.

## Use

1. The configurator's Python environment must exist (`..\configurator\setup.bat`, once). The lab uses it; nothing else to install.
2. Double-click **`run.bat`**. Your browser opens at http://localhost:8518. Close the black window to stop.
3. **`run_tests.bat`** runs all tests (physics, optimiser, every page).

## Online copy (a link for a friend)

`python tools/make_online_copy.py` builds `06-tools/aero-lab-online/` (app, model, airfoil data; 0.4 MB) for Streamlit Community Cloud; `tools/online_smoke.py` tests it in a clean environment.
The steps for GitHub and Streamlit (the parts that need your accounts) are in `06-tools/HORYZN_aero-lab_publish-online_guide.md`.

## Pages

| Page | What it does |
|---|---|
| Inputs | About 100 inputs in 11 tabs, **the first being the four minimum inputs** (mass, design speed, altitude, span: always on) with the button **Set every other input to auto**; then the planform of up to three panels, twist, an airfoil per station, section physics, ailerons, flaps, fuselage, flight condition, safety factor, limits. **Every input has an *on* switch (off = auto: the lab decides) and a *range* switch** (minimum and maximum; an airfoil station or a choice: a list of options). Live top view and key numbers at the right (the middle of every range, the default for every auto input). |
| Results | In this order: **1 airfoils** (shape at each station, thickness and camber, section numbers), **2 top view** with the fuselage, **3 charts**: airfoil (cl, polar, cl/cd, cm, Reynolds sensitivity, confidence), wing lift, drag and L/D, **pitching moment (Cm)**, **stall and span** (strip lift against its maximum, utilisation map), **roll control**, performance, flap, and every number in a table. If you ran the optimiser, the optimised wing is shown next to your inputs and in the same charts. |
| Optimise | Weights on eight objectives, the **flight points** (the wing in cruise, slow flight and a manoeuvre or gust at once, each with its weight and its least speed margin), the effort; searches inside the ranges of the Inputs page and over everything an auto design input allows; shows what changed, where each result sits in its range (and warns when it is at the edge), the improvement, and can copy the result into the inputs. |
| Explore | **Search through the ranges**: a sample of 64 to 1,024 designs of the whole space (how many meet every limit, which limit bites, a scatter of any two results coloured by a third, the best designs, which input moves a result), and sweeps of one or two inputs over their range with the limits marked. A design can be used as the inputs or shown in Results. |
| Valleys | **What to play with.** Around the optimum (or the nominal design): the free range of every input (how far it can move before the objective loses 1 % or a limit breaks), the slice of every input, a **3D valley surface** of any two inputs with the limits as a wall, and the combinations of inputs that can move together; a button turns the free ranges into the new ranges. |
| Sensitivity | Tornado chart: which calibration or environment input moves a result most. |
| About the model | Short description; the equations are in `docs/physics-lab.md`. |

The **safety factor** (1 to 2, tab Safety factor of the Inputs page, default 1.5; the sidebar shows it) is applied to every result in its adverse direction; each result is also shown as calculated. Four families (drag and power,
lift capability, moments and loads, control authority) can be switched off separately.

Inputs can be saved and loaded (JSON) from the sidebar, and the wing of the configurator's recommended design can be loaded with one button.

## Files

```
wingaero/     the model: core, airfoil, geometry, lifting_line, analysis, params, optimise, figures
app/app.py    the interface
tests/        113 tests
tools/        sanity_sweep.py (random inputs, invariants), hand_check.py (hand calculation)
data/airfoils the 34 sections (coordinates)
docs/         physics-lab.md (equations, validation, limits)
```

Results, the plan and the decisions are in `05-systems-architecture/sizing/aero-lab/` of the HORYZN folder.
