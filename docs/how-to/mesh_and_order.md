# How to choose the mesh size and element order

The accuracy of a full-order solve is set by the mesh (`maxh`) and the polynomial order of
the finite elements (`order`). This page shows how to set both, refine the mesh locally, and
check that the result has converged.

## Set the mesh size

`maxh` is the largest element edge, **in metres**, like every length in the code:

```python
proj.create_primitive("rwg", name="guide", a=0.1, b=0.05, L=0.2, maxh=0.02)   # primitive
proj.generate_mesh(maxh=0.005)                                                  # any model
```

A good starting point is a fifth of the shortest wavelength in the model, at the top of the
band, with the default third-order elements. Inside a dielectric the wavelength is shorter by
$\sqrt{\varepsilon_r}$.

!!! warning "maxh is in metres"
    `maxh=5` means five metres, not five millimetres. If `maxh` is larger than the model, it
    does not constrain the mesh at all and the code warns:
    `maxh=5 m is larger than the geometry (largest extent 0.15 m) and will not constrain the
    mesh. maxh is in metres -- did you mean 0.005 (i.e. 5 mm)?`

`proj.generate_mesh()` replaces the mesh and discards results computed on the old one; it
asks first if results exist (pass `force=True` to skip the question in a script).

Curved surfaces are meshed with curved elements, to order 3 by default (`curve_order`). On
some imported CAD edges, curving to order 3 fails
(`GeomAPI_ProjectPointOnCurve::NearestPoint`); the mesh is then curved to the highest lower
order that works, and the code warns. `geo.curve_order` holds the order used. Pass
`curve_order=2` to ask for it directly.

## Refine the mesh locally

Give faces or solids that need smaller elements their own `maxh`, by name, before meshing.
Wildcards `*` and alternatives `|` are allowed:

```python
geo = proj.geo
geo.set_local_mesh_refinement("port*", 0.002)            # all port faces
geo.set_local_mesh_refinement("port*|ceramic", 0.002)    # ports and the 'ceramic' solid
proj.generate_mesh(maxh=0.01)                            # the global maxh caps the rest
```

For a model of several parts, call it on the project's geometry (the assembly), not on one
part: the assembly meshes its own copy.

## Set the element order

`order` is a `solve()` option (default 3):

```python
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30, order=2)
```

Higher order is more accurate per element and costs more unknowns. Order 2 with a finer mesh,
or order 3 with a coarser one, are both common choices.

## Choose the element kind

`nedelec='first'` (the default) uses first-kind Nédélec elements, `'second'` the full
polynomial space:

```python
proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30, nedelec="second")
```

Both approximate the field's curl equally well; `'first'` has about a third fewer unknowns
at order 2. A project saved without the setting was computed with `'second'` and keeps it.
Changing the order or the kind is a different request, so `solve()` recomputes.

## Check convergence

Solve the same model with two mesh sizes (or orders) and compare. Only the quantity you need
has to converge:

```python
import numpy as np

proj.generate_mesh(maxh=0.02, force=True)
S_coarse = proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30)["S"]

proj.generate_mesh(maxh=0.01, force=True)
S_fine = proj.fds.solve(fmin=1.0, fmax=3.0, nsamples=30)["S"]

print("largest change in S21:", np.max(np.abs(S_fine[:, 1, 0] - S_coarse[:, 1, 0])))
```

If the change is below what you need, the coarser mesh is good enough. For a first check
of any new model, compare with a case whose answer is known (a uniform waveguide section of
the same cross-section, a closed-form resonance).

**See also:** [solve() options](../reference/solve_options.md);
[How to run large models](large_models.md).
