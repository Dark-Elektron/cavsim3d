# Explanation

Background reading: why the code is built the way it is, what its results mean, and where
its limits are. These pages are for understanding, not for following step by step; the
[tutorials](../tutorials/index.md) and [how-to guides](../how-to/index.md) link to them where
the background matters.

- **[How a model is solved in pieces](architecture.md)**: the staged pipeline
  (full-order model, reduced model, concatenation) and the three ways of solving a model
  of several parts.
- **[How model order reduction works](model_reduction.md)**: snapshots, the reduced basis,
  what the tolerance controls, and where a reduced model can be trusted.
- **[Ports and port modes](ports.md)**: modes, normalisation, reference impedances, and why
  port faces are magnetic walls when resonances are computed.
- **[Parts, joins and netlists](parts_and_joins.md)**: the list of parts, glued and coupled
  parts, imported projects, and what is checked at every join.
- **[The project folder](project_layout.md)**: what is saved where, and when saved results
  are reused or recomputed.

The equations behind these pages -- the variational form, the port eigenproblems, the
Z-to-S conversion, the reduction, the join and the resonances -- are derived in
[Mathematical Theory](../theory.md).
