# Beams

A beam passing through a structure leaves energy behind and is kicked by the fields it
excites. These five lessons add a beam to a model and compute its longitudinal
impedance next to the S-parameters: in one solve, for a structure solved earlier, with
lossy material, for a chain built from parts solved one by one, and across the spectrum
of a 9-cell cavity.

1. **[Compute a beam impedance](collimator_impedance.ipynb)**: a beam on the axis of a
   collimator; the generalised scattering matrix, the beam impedance below the cut-off,
   and Yokoya's estimate for gentle tapers.
2. **[Add a beam to a solved cavity](beam_on_solved_cavity.ipynb)**: a pillbox solved
   without a beam gets one; only the beam's columns are solved, and R/Q read from the pole
   of the beam impedance matches the eigenmode's.
3. **[Beam past a lossy absorber](lossy_absorber.ipynb)**: a ceramic ring in the wall of a
   pipe; the real part of the beam impedance is the power the ring absorbs.
4. **[Join parts with a beam](beam_through_parts.ipynb)**: a pillbox cell solved once and
   repeated three times, joined with the beam and checked against the cells in one piece;
   the same chain from a cell solved in another project without a beam.
5. **[Beam impedance of the TESLA cavity](tesla_beam_impedance.ipynb)**: the 9-cell
   cavity from 1.0 to 2.9 GHz; the π-mode and two higher-order modes as poles of the
   beam impedance, with frequency and R/Q read from each pole and checked against
   cavsim2d's eigenmodes.

Background: [Beam excitation](../../theory/beam.md) derives the formulation; the
[how-to guide](../../how-to/beam_impedance.md) collects the options.
