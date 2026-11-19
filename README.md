# HORYZN wing aerodynamics lab (online copy)

A stand-alone study of the wing (and a simple fuselage): airfoil per station, non-linear lifting line, stall, pitching moment, roll control, safety factor,
optimiser with flight points, sample and sweeps through ranges, and the valleys around an optimum. Built with Streamlit; the model is in `wingaero/`,
the airfoil data in `data/`, the interface in `app/app.py`.

Online notes: this runs on a free shared server. Your inputs live only in your browser tab (save them with the button in the sidebar), and long
optimisations are slow there. The master copy lives in the HORYZN project folder (06-tools/aero-lab); this folder is generated from it by
tools/make_online_copy.py.
