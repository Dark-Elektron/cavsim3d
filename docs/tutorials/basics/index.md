# Basics

Three lessons on a single rectangular waveguide, from the first frequency sweep to its
resonant fields. Each one takes a few minutes and ends with a comparison against the exact
solution.

1. **[Your first simulation](first_simulation.ipynb)**: create a project, add a
   waveguide, look at its mesh and ports, sweep 0.5 to 5 GHz, and plot the S- and
   Z-parameters against the closed-form solution. Reopen the saved project.
2. **[Build a reduced-order model](reduced_order_model.ipynb)**: reduce the 46-sample
   sweep to a model with a few dozen unknowns, sweep 2000 frequencies with it, measure its
   error, and see what the reduction tolerance does.
3. **[Find resonances and look at fields](resonances_and_fields.ipynb)**: compute the
   resonant frequencies, find them as peaks in the impedance, and draw a resonant mode, a
   driven field and a port mode. See which resonances of a reduced model can be trusted.

After these three, continue with [Building models](../models/index.md) to bring in your
own geometry, or with [Models of several parts](../multi_part/index.md).
