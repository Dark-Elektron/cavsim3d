# Models of several parts

Most real structures are chains: couplers, cavities, bellows, tapers, windows. These three
lessons build models from several parts, solve each part on its own, and join them. They
also show why that is worth doing: a repeated part is solved once, and a part solved in an
earlier project is not solved again.

1. **[Join parts into one model](combine_parts.ipynb)**: chain an air section, a dielectric
   slab and another air section; solve, reduce and join the parts; check the slab's
   reflection against the closed-form solution and against the same model solved in one
   piece.
2. **[Repeat a section](repeated_sections.ipynb)**: build a guide from four copies of one
   section, solved once; compute the chain's resonances and draw one over the whole chain.
3. **[Reuse a solved project](reuse_projects.ipynb)**: use a section solved in another
   project twice, next to a new part; read the solve plan; turn the reference into a copy;
   see what happens when the imported section does not cover the new band.

Background: [How a model is solved in pieces](../../explanation/architecture.md) and
[Parts, joins and netlists](../../explanation/parts_and_joins.md).
