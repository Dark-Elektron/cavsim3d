---
name: Bug report
about: Something computes wrong results, fails, or behaves unexpectedly
labels: bug
---

**What happened**
A clear description of the problem.

**How to reproduce**
The smallest script that shows it (a small waveguide or primitive, a few
frequency samples), or the steps in a notebook:

```python
from cavsim3d import EMProject

proj = EMProject(name="bug", base_dir="./simulations", overwrite=True)
...
```

**What you expected**
The result you expected instead (and, for a wrong number, where the reference
value comes from: analytic formula, CST, measurement).

**Output**
The full error message or traceback, or the wrong values/plot.

**Environment**
- OS:
- Python version:
- cavsim3d version or commit (`python -c "import cavsim3d; print(cavsim3d.__version__)"`):
- ngsolve version (`python -c "import ngsolve; print(ngsolve.__version__)"`):
