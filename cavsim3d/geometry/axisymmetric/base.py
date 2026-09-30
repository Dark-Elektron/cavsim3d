"""Base class of the bodies of revolution (the cavsim2d models in 3D)."""

import functools
import inspect
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import BaseGeometry
from .profile import Profile, revolve
from cavsim3d.utils.names import is_port_name

#: Length units a model's dimensions may be given in, as metres per unit.
UNITS = {'m': 1.0, 'cm': 1e-2, 'mm': 1e-3, 'um': 1e-6}


def _jsonable(value):
    """*value* with numpy arrays/scalars and tuples turned into JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _beampipe_option(beampipe) -> str:
    bp = str(beampipe).lower()
    if bp not in ('none', 'left', 'right', 'both'):
        raise ValueError(f"beampipe must be 'none', 'left', 'right' or 'both', got {beampipe!r}.")
    return bp


def _with_config(init):
    """Let *init* take its arguments from a ``config`` dict as well.

    ``Model(config=cfg)`` is ``Model(**cfg)``; arguments given explicitly
    (positionally or by keyword) take precedence over the dict. The wrapper
    consumes ``config``: *init* declares it (so it shows in signatures and
    IDEs) but always receives ``None``.
    """
    names = list(inspect.signature(init).parameters)[1:]     # without self

    @functools.wraps(init)
    def wrapper(self, *args, config: Optional[dict] = None, **kwargs):
        if config is not None:
            if not isinstance(config, dict):
                raise TypeError(f'config must be a dict, got {type(config).__name__}.')
            given = set(names[:len(args)]) | set(kwargs)
            kwargs = {**{k: v for k, v in config.items() if k not in given}, **kwargs}
        return init(self, *args, **kwargs)
    return wrapper


class AxisymmetricGeometry(BaseGeometry):
    """A body of revolution: a meridian :class:`Profile` swept about the Z axis.

    Subclasses define the meridian in :meth:`profile` (metres) and call
    :meth:`_finish_init` at the end of their constructor. The beam axis is Z;
    each beam aperture is a port (``port1`` at the low-z end), and the rest of
    the surface is the PEC wall ``'default'``.

    Every constructor:

    - takes its dimensions in ``unit`` (default ``'mm'``, as in cavsim2d;
      also ``'m'``, ``'cm'``, ``'um'``). ``maxh`` is always in metres;
    - accepts its arguments as a dict, ``Model(config={...})``; explicit
      arguments override the dict;
    - builds the solid only. The mesh is made by :meth:`generate_mesh`, or by
      the first solve if it was never called, with the ``maxh`` given here.
    """

    #: The solver meshes this geometry itself (with its own ``maxh``) when a
    #: solve starts and no mesh has been generated.
    mesh_on_demand = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__init__' in cls.__dict__:
            cls.__init__ = _with_config(cls.__dict__['__init__'])

    def __init__(self):
        super().__init__()
        self.maxh = None
        self.unit = 'mm'
        self.parameters: Dict[str, Any] = {}
        self._init_params: Dict[str, Any] = {}
        self._mesh_params: Dict[str, Any] = {}

    def _require(self, **named) -> None:
        """Raise if a required argument was given neither directly nor in ``config``."""
        missing = [k for k, v in named.items() if v is None]
        if missing:
            raise TypeError(f"{type(self).__name__} needs {', '.join(missing)}: give "
                            f"{'it' if len(missing) == 1 else 'them'} as arguments or "
                            "in config.")

    def _set_unit(self, unit: str) -> None:
        """Set the length unit of the dimensions (``self._s`` = metres per unit)."""
        key = str(unit).lower()
        if key not in UNITS:
            raise ValueError(f"unit must be one of {sorted(UNITS)}, got {unit!r}.")
        self.unit = key
        self._s = UNITS[key]

    def _finish_init(self, maxh, init_params: Dict[str, Any]) -> None:
        """Build the solid and record the constructor arguments (for reload)."""
        self.maxh = maxh
        self._mesh_params = {'maxh': maxh, 'curve_order': 3, 'curvaturesafety': 2}
        self._init_params = _jsonable({**init_params, 'unit': self.unit})
        self.build()
        self._record('__init__', **self._init_params)

    @abstractmethod
    def profile(self) -> Profile:
        """The meridian contour in metres (``'AXI'`` / ``'PEC'`` / ``'PMC'``)."""

    def _chained(self, prof: Profile) -> Profile:
        """Apply the ``chain`` / ``spacing`` arguments (spacing in ``unit``)."""
        chain = int(getattr(self, 'chain', 1) or 1)
        spacing = getattr(self, 'spacing', None)
        if spacing is not None:
            spacing = ([float(s) * self._s for s in spacing]
                       if isinstance(spacing, (list, tuple, np.ndarray))
                       else float(spacing) * self._s)
        return prof.chained(chain, spacing=spacing)

    def build(self) -> None:
        """Revolve :meth:`profile` into the solid and name its faces."""
        self.geo = revolve(self.profile())
        self.bc = 'default'
        self._bc_explicitly_set = True
        self._ports = None
        self._boundaries = None
        self.invalidate_tag()

    def generate_mesh(self, maxh=None, curve_order=None, curvaturesafety=None):
        """Mesh the solid. Arguments left out keep their previous values.

        ``maxh`` defaults to the one given to the constructor (metres). netgen
        also refines to the surface curvature: roughly
        ``radius of curvature / curvaturesafety`` (default 2), so small corner
        radii refine the mesh below ``maxh``.
        """
        params = dict(self._mesh_params)
        for key, val in (('maxh', maxh), ('curve_order', curve_order),
                         ('curvaturesafety', curvaturesafety)):
            if val is not None:
                params[key] = val
        self._mesh_params = params
        self.maxh = params['maxh']
        return super().generate_mesh(**params)

    @property
    def ports(self) -> List[str]:
        """Port names, from the mesh if there is one, else from the solid."""
        if self.mesh is None:
            return sorted({f.name for f in self.geo.faces if is_port_name(f.name)})
        return super().ports

    @property
    def boundaries(self) -> List[str]:
        """Boundary names, from the mesh if there is one, else from the solid."""
        if self.mesh is None:
            return sorted({f.name for f in self.geo.faces if f.name})
        return super().boundaries

    def get_dimensions(self) -> dict:
        return dict(self.parameters)

    def plot_profile(self, ax=None, **kwargs):
        """Plot the meridian contour (z, r) in millimetres; returns the axes.

        The ports (the beam apertures) are drawn in red and material regions
        (e.g. an absorber ring) are shaded.
        """
        import matplotlib.pyplot as plt
        prof = self.profile()
        pts = prof.contour_points() * 1e3
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 3))
        closed = np.vstack([pts, pts[:1]])
        ax.plot(closed[:, 0], closed[:, 1], **kwargs)
        for z_ap, r_ap in prof.apertures():
            ax.plot([z_ap * 1e3] * 2, [0, r_ap * 1e3], color='r', lw=2)
        for reg in prof.regions():
            zr = np.asarray(reg.points) * 1e3
            ax.fill(zr[:, 0], zr[:, 1], alpha=0.35, label=reg.material)
        if prof.regions():
            ax.legend(loc='center')
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('r [mm]')
        ax.set_aspect('equal')
        return ax

    # -- tagging / persistence ---------------------------------------------

    def _get_geometry_params(self) -> Dict[str, Any]:
        return {'class': type(self).__name__,
                **{k: v for k, v in self._init_params.items() if k != 'maxh'},
                'bc': self.bc}

    def _get_mesh_params(self) -> Dict[str, Any]:
        return {**self._mesh_params, 'nv': self.mesh.nv if self.mesh else None}

    @classmethod
    def _rebuild_from_history(cls, history: List[dict], project_path: Path,
                              source_file=None) -> 'AxisymmetricGeometry':
        """Rebuild from the recorded constructor arguments, then later ops.

        A recorded ``generate_mesh`` only sets the mesh parameters: the mesh
        itself is reloaded by the project (or regenerated on demand).
        """
        ops = [e.get('op') for e in history]
        if '__init__' not in ops:
            raise ValueError(f"{cls.__name__} history has no '__init__' entry.")
        start = ops.index('__init__')
        params = {k: v for k, v in history[start].items() if k not in ('op', 'timestamp')}
        obj = cls(**params)
        for entry in history[start + 1:]:
            if entry.get('op') == 'generate_mesh':
                obj._mesh_params.update({k: entry[k] for k in
                                         ('maxh', 'curve_order', 'curvaturesafety')
                                         if k in entry})
                obj.maxh = obj._mesh_params.get('maxh')
            else:
                obj._replay_common_op(entry)
        return obj

    def save_geometry(self, project_path) -> None:
        """Save the solid as STEP plus the history that rebuilds it."""
        geo_dir = Path(project_path) / 'geometry'
        geo_dir.mkdir(parents=True, exist_ok=True)
        if self.geo is not None:
            try:
                self.geo.WriteStep(str(geo_dir / 'cavsim3d.geometry.step'))
            except Exception:
                pass
        meta = {
            'type': type(self).__name__,
            'module': type(self).__module__,
            'source_link': None,
            'source_filename': 'cavsim3d.geometry.step',
            'source_hash': None,
            'history': self._history,
        }
        with open(geo_dir / 'history.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
