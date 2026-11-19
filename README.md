# HORYZN wing aerodynamics lab

A stand-alone study of **the wing** (and a simple fuselage), derived from the HORYZN scout configurator: an airfoil per station, a non-linear lifting line (stall, pitching
moment, roll control), a safety factor, an optimiser with several flight points, search through ranges, and the valleys around an optimum. No tail, propeller or installation
effects. Built with Streamlit.

## Pages

| Page | What it does |
|---|---|
| Inputs | About 100 inputs in 11 tabs, **the first being the four minimum inputs** (mass, design speed, altitude, span: always on) with the button **Set every other input to auto**; then the planform of up to three panels, twist, an airfoil per station, section physics, ailerons, flaps, fuselage, flight condition, safety factor, limits. **Every input has an *on* switch (off = auto: the lab decides) and a *range* switch** (minimum and maximum; an airfoil station or a choice: a list of options). Live top view and key numbers at the right (the middle of every range, the default for every auto input). |
| Results | In this order: **1 airfoils** (shape at each station, thickness and camber, section numbers), **2 top view** with the fuselage, **3 charts**: airfoil (cl, polar, cl/cd, cm, Reynolds sensitivity, confidence), wing lift, drag and L/D, **pitching moment (Cm)**, **stall and span** (strip lift against its maximum, utilisation map), **roll control**, performance, flap, and every number in a table. If you ran the optimiser, the optimised wing is shown next to your inputs and in the same charts. A **PDF report** button (here and on the Optimise page) writes a document of the design: the numbers, **how each one was calculated** (formula, numbers put in, result), the drawings of the wing and its airfoils, the charts, and for an optimised design how the optimum was found and the valleys around it. |
| Optimise | Weights on eight objectives, the **flight points** (the wing in cruise, slow flight and a manoeuvre or gust at once, each with its weight and its least speed margin), the effort; searches inside the ranges of the Inputs page and over everything an auto design input allows; shows what changed, where each result sits in its range (and warns when it is at the edge), the improvement, and can copy the result into the inputs. |
| Explore | **Search through the ranges**: a sample of 64 to 1,024 designs of the whole space (how many meet every limit, which limit bites, a scatter of any two results coloured by a third, the best designs, which input moves a result), and sweeps of one or two inputs over their range with the limits marked. A design can be used as the inputs or shown in Results. |
| Valleys | **What to play with.** Around the optimum (or the nominal design): the free range of every input (how far it can move before the objective loses 1 % or a limit breaks), the slice of every input, a **3D valley surface** of any two inputs with the limits as a wall, and the combinations of inputs that can move together; a button turns the free ranges into the new ranges. |
| Sensitivity | Tornado chart: which calibration or environment input moves a result most. |
| About the model | What the model does and does not do. |

The **safety factor** (1 to 2, tab Safety factor of the Inputs page, default 1.5; the sidebar shows it) is applied to every result in its adverse direction; each result is also shown as calculated. Four families (drag and power,
lift capability, moments and loads, control authority) can be switched off separately.

## Run it on your own computer

```
pip install -r requirements.txt
streamlit run app/app.py
```

## Files

```
wingaero/        the model: core, airfoil, geometry, lifting_line, analysis, params, points, optimise, explore, valleys, figures
app/app.py       the interface
data/airfoils    the 34 sections (coordinates)
docs/about.md    what the model does and does not do
```
