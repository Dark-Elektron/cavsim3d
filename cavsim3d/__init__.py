"""cavsim3d - 3D RF structure analysis.

>>> from cavsim3d import EMProject
>>> proj = EMProject(name="my_project", base_dir="./simulations")
"""

__version__ = "0.1.0"

__all__ = ["EMProject", "__version__"]


def __getattr__(name):
    # Imported on first use, so ``import cavsim3d`` does not load the solver
    # stack (NGSolve, pythonocc).
    if name == "EMProject":
        from cavsim3d.core.em_project import EMProject
        return EMProject
    raise AttributeError(f"module 'cavsim3d' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"EMProject"})
