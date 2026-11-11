# Applications

Studies of accelerator cavities, using everything from the earlier
sections: imported CAD, repeated parts, reduced and joined models, and comparison with CST
Studio Suite. They take longer to run than the lessons (minutes to tens of minutes).

- **[A chain of three TESLA cavities](../tesla_cavity_chain.ipynb)**: the TESLA 9-cell
  cavity imported from a STEP file, solved and reduced once and joined three times; the
  nine modes of its accelerating passband as triplets of the chain, and the π-mode along
  the beam axis.
- **[C3794 two-cavity module vs CST](../benchmarks/c3794_cavity_module.ipynb)**: a module of
  two cavities with couplers, built from one reduced cavity and compared with CST Studio
  Suite band by band.
- **[Figures of merit of a pillbox cavity](../benchmarks/figures_of_merit.ipynb)**: R/Q,
  geometry factor, Q and peak fields of the accelerating mode, checked against the closed
  form; the dielectric Q of a lossy filling, and the coupling of two cells.
