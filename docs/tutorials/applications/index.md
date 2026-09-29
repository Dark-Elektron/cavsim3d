# Applications

Complete studies on real accelerator structures, using everything from the earlier
sections: imported CAD, repeated parts, reduced and joined models, and comparison with CST
Studio Suite. They take longer to run than the lessons (minutes to tens of minutes).

- **[TESLA 9-cell cavity chain](../tesla_cavity_chain.ipynb)**: three superconducting 9-cell
  cavities, one solved and reduced, joined into a chain; the chain's S-parameters,
  accelerating passband, resonant modes and field on axis.
- **[C3794 two-cavity module vs CST](../benchmarks/c3794_cavity_module.ipynb)**: a module of
  two cavities with couplers, built from one reduced cavity and benchmarked against CST
  Studio Suite band by band.
- **[Cavity figures of merit: validation](../benchmarks/figures_of_merit.ipynb)**: R/Q,
  geometry factor, wall and dielectric Q, peak fields, transverse kick and cell coupling,
  checked against closed-form pillbox modes, the Lorentz force, one-piece solutions and
  cavsim2d.
