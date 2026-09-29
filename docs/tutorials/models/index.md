# Building models

Where models come from, and what they are made of. The basics tutorials used a waveguide
primitive; these four build cavities from their parameters, bring in a CAD file, cut a
model into domains that are solved separately, and fill a model with lossy material. Each
result is checked against a closed-form solution.

1. **[Build cavities from parameters](parametric_geometries.ipynb)**: build elliptical
   cavities, a pillbox, an RF gun and beam-line elements from their cavsim2d parameters,
   pass the parameters as config dictionaries, and check a meshed pillbox's volume.
2. **[Import a CAD model](cad_import.ipynb)**: import a circular waveguide from a STEP file,
   mesh it, check its circular ports, and match its S- and Z-parameters to the exact TE11
   solution.
3. **[Cut a model into domains](splitting_cad.ipynb)**: cut the same guide into three domains
   with two planes, solve and reduce each domain on its own, and join the reduced models
   into a model of the whole guide.
4. **[Fill a waveguide with lossy material](materials_and_losses.ipynb)**: give a waveguide a
   dielectric filling, watch its cutoff move, add a loss tangent and measure the absorbed
   power.

Next: [Models of several parts](../multi_part/index.md) chains separate parts into one
model.
