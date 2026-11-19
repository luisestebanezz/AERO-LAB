## About the model

This lab studies **the wing** (with a simple fuselage for the top view and, if you want, its drag). It was derived from the scout configurator
(`06-tools/configurator`), reusing its lifting line and its airfoil polars, and extended.

**What it computes.** A vortex-lattice lifting line (Weissinger) of up to three panels with a twist table and a different airfoil at each station. The section
polars (NeuralFoil, a neural-network model of XFOIL) are followed up to and past the stall, so the lift curve, the drag, the pitching moment, the stall and the
aileron roll power come from a **non-linear** solution (Newton's method), not from a straight-line estimate.

**The safety factor.** One factor from 1 to 2 (Inputs page, tab Safety factor; the sidebar shows it). Every result is shown as calculated and with the factor applied in its adverse direction:
drag and power × SF, usable maximum lift ÷ SF (stall speed × √SF, strip utilisation × SF), moments and loads × SF, roll power ÷ SF. Each family can be switched off.

**Every input has two switches: on and range.**
- **on, range off: a value.** The lab uses your number.
- **on, range on: a range.** A **minimum and a maximum** (an airfoil station: a **list of allowed airfoils**; a choice or a yes/no: a **list of allowed options**). The lab searches inside it: the
  optimiser (Optimise page) finds a good design, the sample and the sweeps (Explore page) show the whole space the ranges span, which limits bite, and which input moves which result.
  A plain analysis takes the middle of the range.
- **on off: auto.** The lab decides. A *design* input (aspect ratio, taper, sweep, twist, incidence, the breaks and chords of the panels, the airfoils) is searched by the optimiser over everything
  it allows. An input that describes the environment, a calibration or a model switch (transition, calibration factors, fuselage dimensions, safety-factor families, limits, flap and aileron
  settings...) takes the lab's default, and so does the structure of the wing (two panels, linear twist, no ailerons or flaps). A wing area, wing loading or design lift coefficient that is auto is not used.

**Minimum inputs (first tab).** Four numbers state the problem and cannot be auto (their *on* switch is locked; each can still be a range): **mass** (what the wing carries), **design speed**,
**altitude** (the air) and **span** (how big the wing is: without a size the optimiser would simply stretch the wing to the longest span allowed). Press **Set every other input to auto** and the lab
searches the 11 design inputs for you from these four numbers alone; the tab lists what is searched and what takes the default.

**Flight points (Optimise page).** The optimiser evaluates the same wing in several flight conditions at once: by default **cruise** (the design point of the inputs), **slow flight** (0.85 x the design
speed, weight 0.5, least speed margin 1.05) and a **manoeuvre or gust** (1.3 x the speed at load factor 2.5, weight 0: it only checks the limits). Each point has a speed factor, an altitude offset, a mass factor, a
load factor, a weight in the objective (normalised to add up to one) and its own least margin over the stall. The objectives are evaluated at every point and combined with the weights; every point is tested
against the stall-position and strip-utilisation limits and its own margin; the pitching-moment, roll and wing-loading limits belong to the design point. Points can be added, removed, switched off and saved with the inputs.

**Valleys (page Valleys).** Around an optimum the weighted objective forms a valley: wide and flat along an input that hardly matters, narrow along one that matters. The page maps it, with the same flight points,
limits and safety factor as the optimiser: the **free range of every input** (how far it can move, with the others held, before the objective loses the tolerance, 1 % by default, a limit breaks, or its range ends),
tested option by option for airfoils and choices; a **slice** chart per input; a **3D surface** of any two inputs, with the designs that break a limit as a wall; and the **combinations** that can move together when a limit
is active (the free combinations, sloppiest first, and how far each can move). A button turns the free and loose inputs into ranges and the tight ones into values, so that the next optimisation moves only what can move.

**What it does not do.** No tail, propeller, nacelles, camera or other installation effects; no compressibility; no post-stall redistribution beyond what the section
tables give (a swept wing's tip-stall pitch-up is therefore only partly captured); steady, planar, symmetric flight. The flap and aileron models are thin-airfoil
theory with viscous factors and are the least certain part. With no wind-tunnel or flight data, treat the absolute drag and the stall speed as ±10 %: look at the
differences between designs, and use the safety factor and the sensitivity page to see how much the calibration matters.

Equations, assumptions, validation: `docs/physics-lab.md`.


## Using the online copy

This copy runs on a free shared server (Streamlit Community Cloud), not on a PC.

- **Your inputs live in this browser tab only.** Closing or reloading the tab loses them. Use **Save the inputs and ranges (JSON)** in the sidebar before you leave, and
  **Load inputs (JSON)** to come back.
- **The server is small and shared.** A search of many inputs with three flight points takes minutes, and other visitors share the same processor. Use the **quick** effort,
  switch flight points off (Optimise page, *Cruise only*) or give fewer inputs as ranges. The thorough effort is not offered here.
- **The server sleeps** after about 12 hours without visitors (Streamlit's rule at the time of writing); the first visitor waits about a minute while it wakes up.
- The button that loads the wing of the scout configurator does nothing here (that file lives on the PC).
