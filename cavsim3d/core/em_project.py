from __future__ import annotations
import os
from cavsim3d.core.persistence import ProjectManager
from pathlib import Path
from typing import Optional, Union
import json
from datetime import datetime
import shutil
from cavsim3d.utils.io_utils import get_user_confirmation
import cavsim3d.utils.printing as pr
from cavsim3d.geometry.assembly import Assembly
from cavsim3d.geometry.importers import OCCImporter
import cavsim3d.geometry.primitives as primitives
from cavsim3d.geometry.base import BaseGeometry
from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
from ngsolve import Mesh  # type: ignore


def _default_part_name(geometry) -> str:
    """Name for a part that was set without one (file name or class name)."""
    fp = getattr(geometry, 'filepath', None)
    if fp:
        return Path(str(fp)).stem
    return type(geometry).__name__.lower()


# What a project folder holds at its top level (current and older layouts),
# plus files an operating system drops into any folder.
_PROJECT_ENTRIES = {
    'project.json', 'timing.json', 'geometry', 'mesh', 'fds',
    'fom', 'foms', 'roms', 'eigenmode', 'port_modes', 'matrices.h5', 'snapshots.h5',
    '.DS_Store', 'Thumbs.db', 'desktop.ini',
}


def _check_project_name(name) -> str:
    """A project name is one folder name: not empty, no path, no '.'/'..'."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Project name must be a non-empty string, got {name!r}.")
    bad = set('<>:"|?*\\/') & set(name)
    if (name in ('.', '..') or bad or name != name.strip() or name.endswith('.')
            or any(ord(c) < 32 for c in name)):
        raise ValueError(
            f"Invalid project name {name!r}: it becomes the project's folder inside "
            f"base_dir, so it must be a plain folder name (no path separators, no "
            f"'.' or '..', none of <>:\"|?*, no leading/trailing space or trailing dot).")
    return name


def _looks_like_project(path: Path) -> bool:
    """True if ``path`` holds only what a cavsim3d project writes (or is empty)."""
    if (path / 'project.json').is_file():
        return True
    try:
        return all(entry.name in _PROJECT_ENTRIES for entry in path.iterdir())
    except OSError:
        return False


class EMProject:
    """
    Central class for managing electromagnetic simulation projects.

    Responsibility:
    - Manage the project directory structure.
    - Orchestrate saving and loading of geometry, mesh, and solvers.
    - Provide a unified entry point for simulation.

    Parameters
    ----------
    name : str
        Project name; the project lives in the folder ``base_dir / name``.
        An existing project of that name is opened.
    base_dir : str or Path, optional
        Folder holding the project folder (default: the current directory).
    geometry : BaseGeometry, optional
        Geometry of a new project.
    bc : str, optional
        Boundary condition pattern (default: the geometry's).
    overwrite : bool
        Delete an existing project of that name and start afresh.  A folder
        that is not a cavsim3d project is never deleted.
    """
    
    def create_assembly(self, main_axis: Optional[str] = None, force: bool = False) -> 'Assembly':
        """
        Start the project's part list afresh as an empty assembly.

        Usually not needed: ``import_geometry``, ``create_primitive``,
        ``import_project`` and ``add`` build the list on their own.  Use this to
        discard the current parts, or to choose the axis up front.

        Parameters
        ----------
        main_axis : str, optional
            Axis along which the parts are chained ('X', 'Y' or 'Z'); defaults
            to the project's ``main_axis`` (Z unless set)
        force : bool
            Replace an existing mesh / results without asking.

        Returns
        -------
        Assembly
            The new assembly instance (or the current geometry if replacing
            existing results was declined)
        """
        if (self.has_mesh() or self.has_results()) and not force:
            if not get_user_confirmation(
                "\nWARNING: A new assembly will invalidate the current mesh and simulation results.\n"
                "Do you want to continue and delete existing results?"
            ):
                pr.info("Keeping the existing geometry and results.")
                return self.geometry
            self.invalidate_mesh()
        elif force and (self.has_mesh() or self.has_results()):
            self.invalidate_mesh()

        if main_axis is not None:
            self._main_axis = str(main_axis).upper()
        self._part_name = None
        self.geometry = Assembly(main_axis=self._main_axis)  # the setter saves
        return self.geometry

    def __init__(
        self,
        name: str,
        base_dir: Optional[Union[str, Path]] = None,
        geometry: Optional[BaseGeometry] = None,
        bc: Optional[str] = None,
        overwrite: bool = False,
        *,
        _read_only: bool = False,
        _announce: bool = True,
    ):
        # _read_only: open another project to read its results (an imported
        # part): no questions about its CAD file, and save() writes nothing.
        # _announce=False: an internal (scratch) project, created and opened
        # without messages or a notebook banner.
        self._read_only = bool(_read_only)
        self._announce = bool(_announce)
        if self._read_only and overwrite:
            raise ValueError("a project opened read-only cannot be overwritten")
        self.name = _check_project_name(name)
        # Use current directory if base_dir is not provided
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.project_path = self.base_dir / self.name
        if self.project_path.exists() and not self.project_path.is_dir():
            raise FileExistsError(
                f"{self.project_path} exists and is a file, not a project folder.")

        # Overwrite protection: if overwrite=True, delete existing project folder
        if overwrite and self.project_path.exists():
            if not _looks_like_project(self.project_path):
                raise FileExistsError(
                    f"{self.project_path} exists but is not a cavsim3d project (no "
                    f"project.json, and it holds other files), so overwrite=True does "
                    f"not delete it. Choose another name, or remove the folder yourself.")
            pr.info(f"Project '{self.name}' already exists and overwrite=True. Deleting old project...")
            self._force_rmtree(self.project_path)
            if self.project_path.exists():
                raise RuntimeError(
                    f"Could not fully delete existing project at {self.project_path} "
                    f"(a file may be locked by another program or a still-open file "
                    f"handle). Close anything using it and retry, or delete it manually."
                )

        # A fresh project must also start from a clean in-memory state: the
        # solution cache and timing registry are module-level singletons that
        # otherwise survive across project re-creations in the same kernel and
        # can return stale results.  NOTE: this does NOT reload edited library
        # code — in Jupyter, changes to cavsim3d source only take effect after a
        # kernel restart or `%autoreload 2`.
        if overwrite:
            try:
                from cavsim3d.geometry.component_registry import get_global_cache
                get_global_cache().clear()
            except Exception:
                pass
            from cavsim3d.utils.timing import get_timing_registry
            get_timing_registry().clear()
            
        self._geometry = geometry
        self.bc = bc
        self._mesh: Optional[Mesh] = None
        self._fds: Optional[FrequencyDomainSolver] = None
        self._order = 3
        self._n_port_modes = 1
        self._loading = False  # Guard flag to prevent save() during _initial_load()
        # Axis along which parts are chained (None = not chosen -> Z) and the
        # name of the part while the project holds a single geometry.
        self._main_axis: Optional[str] = None
        self._part_name: Optional[str] = None
        # Beams along the main axis (solvers/beam.py); saved in project.json
        self._beam = None
        
        # Automatic Loading or Creation
        say = pr.milestone if self._announce else pr.debug
        if self.project_path.exists():
            say(f"Project '{self.name}' exists. Loading...")
            self._initial_load()
            say(f"Project '{self.name}' loaded.")
        else:
            # In Jupyter, a new project opens with the banner (a reopened one has none)
            if self._announce:
                self._show_welcome_banner()
            say(f"Creating new project '{self.name}' at {self.project_path}")
            self.project_path.mkdir(parents=True, exist_ok=True)
            self.geometry_path.mkdir(parents=True, exist_ok=True)
            self.mesh_path.mkdir(parents=True, exist_ok=True)
            self.fds_path.mkdir(parents=True, exist_ok=True)
            # Automatic save on creation if geometry is provided
            if self.geometry:
                self.save()

    def _welcome_banner_html(self) -> Optional[str]:
        """The notebook banner: logo, name, version and project name."""
        import base64
        from cavsim3d import __version__
        # shipped with the package (assets/), so installed copies show it
        logo_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "assets", "cavsim3d_logo_square.svg"
        )
        if not os.path.exists(logo_path):
            return None
        with open(logo_path, 'rb') as fh:
            logo = base64.b64encode(fh.read()).decode()
        return (
            '<div style="display: flex; align-items: center; gap: 6px; margin: 2px 0;">'
            f'<img src="data:image/svg+xml;base64,{logo}" style="height: 18px;">'
            '<span style="font-size: 12px; font-weight: bold; color: #e66433;">CAVSIM-3D</span>'
            f'<span style="font-size: 11px; color: #888;">v{__version__} &mdash; {self.name}</span>'
            '</div>')

    def _show_welcome_banner(self):
        """Display the banner in Jupyter notebooks."""
        try:
            from IPython import get_ipython
            if get_ipython() is None:
                return  # Not in IPython/Jupyter
            html = self._welcome_banner_html()
            if html:
                from IPython.display import display, HTML
                display(HTML(html))
        except Exception:
            pass  # The banner is cosmetic; never let it break project creation

    def _initial_load(self):
        """Internal helper for automatic loading during instantiation."""
        metadata_file = self.project_path / "project.json"
        if not metadata_file.exists():
            return

        self._loading = True  # Prevent save() from being triggered during load
        try:
            self._load_saved_state(metadata_file)
        finally:
            self._loading = False  # Re-enable save()

    def _load_saved_state(self, metadata_file: Path) -> None:
        """Geometry, mesh and solver of a saved project (see _initial_load).

        A part that cannot be read (a missing or damaged file) is reported and
        skipped, so the project still opens and can be solved again.
        """
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        self.bc = metadata.get("bc", self.bc)
        self._order = metadata.get("order", self._order)
        self._n_port_modes = metadata.get("n_port_modes", self._n_port_modes)
        self._main_axis = metadata.get("main_axis")
        self._part_name = metadata.get("part_name")
        if metadata.get("beam"):
            from cavsim3d.solvers.beam import BeamSetup
            self._beam = BeamSetup.from_dict(metadata["beam"])

        # 1. Load Geometry FIRST
        has_geo = metadata.get("has_geometry", False)
        # Self-healing: check if geometry folder has contents even if flag is false
        if not has_geo and self.geometry_path.exists() and any(self.geometry_path.iterdir()):
            has_geo = True

        if has_geo:
            try:
                self.geometry = BaseGeometry.load_geometry(
                    self.project_path, check_source=not self._read_only)
            except Exception as e:
                pr.warning(f"Could not load geometry: {e}")

        # 2. Load Mesh SECOND (needed for FDS port modes)
        has_mesh = metadata.get("has_mesh", False)
        # Self-healing: check if mesh folder has contents even if flag is false
        if not has_mesh and self.mesh_path.exists() and (self.mesh_path / "mesh.pkl").exists():
            has_mesh = True

        if has_mesh:
            try:
                pm = ProjectManager(self.base_dir)
                self.mesh = pm.load_ngs_mesh(self.mesh_path)
            except Exception as e:
                pr.warning(f"Could not load the saved mesh: {e}. generate_mesh() "
                           "makes a new one.")
            # A reloaded chain has no mesh of its own until meshed again; give
            # it the project's, so proj.geo.show('mesh') works after reopening.
            if (self.mesh is not None and self.geometry is not None
                    and getattr(self.geometry, 'mesh', None) is None):
                self.geometry.mesh = self.mesh

        # 3. Load Solver (FDS) LAST - needs mesh for port mode reconstruction
        if metadata.get("has_fds"):
            if not (self.fds_path / "config.json").exists():
                pr.info("No saved solver state in fds/: the solver starts afresh.")
            else:
                try:
                    # Pass mesh to load method so port modes can be reconstructed
                    self._fds = FrequencyDomainSolver.load_from_path(
                        self.fds_path,
                        geometry=self.geometry,
                        mesh=self.mesh,  # Pass mesh here
                        order=self._order,
                        bc=self.bc
                    )
                except Exception as e:
                    self._fds = None
                    pr.warning(f"Could not load the saved solver state: {e}. The "
                               "solver starts afresh: solve() again to recompute.")

            if self._fds:
                self._fds._project_path = self.project_path
                self._fds._project_name = self.name
                self._fds._project_ref = self
                # load_from_path received the mesh; FE spaces are rebuilt on it
                # (a pickled FES would carry its own, separate mesh copy).
                if self.mesh and self._fds.mesh is None:
                    self._fds.mesh = self.mesh

        # Referenced parts: say so if a source moved or was re-solved since.
        try:
            from cavsim3d.solvers.netlist_persistence import check_references
            for msg in check_references(self.project_path):
                pr.warning(f"{msg}. The saved results may be out of date: solve() "
                           "again to refresh them (or localize() to keep copies).")
        except Exception as e:
            pr.warning(f"Could not check referenced projects: {e}")

    @property
    def geo(self) -> Optional[BaseGeometry]:
        """Shortcut for geometry."""
        return self.geometry

    @property
    def fds(self) -> FrequencyDomainSolver:
        """Lazy initialization of the FrequencyDomainSolver."""
        if self._fds is None:
            if self.geometry is None:
                raise RuntimeError("Cannot initialize solver without geometry.")
            from cavsim3d.solvers.frequency_domain import FrequencyDomainSolver
            self._fds = FrequencyDomainSolver(
                geometry=self.geometry,
                order=self.order,
                bc=self.bc
            )
            self._fds._project_path = self.project_path
            self._fds._project_name = self.name
            self._fds._project_ref = self
        return self._fds

    @fds.setter
    def fds(self, value: Optional[FrequencyDomainSolver]):
        self._fds = value
        if value:
            value._project_path = self.project_path
            value._project_name = self.name
            value._project_ref = self
            value.order = self._order

    # =========================================================================
    # Parts: the project's geometry list
    # =========================================================================
    #
    # A project holds a list of parts.  With one part the project's geometry is
    # that part itself; adding a second part turns the geometry into an
    # Assembly holding both, chained in list order along ``main_axis``.
    # Adding a part under a name the project already has REPLACES that part
    # (re-running a notebook cell does not double the model); use ``n=`` to
    # repeat a part.

    @property
    def main_axis(self) -> str:
        """Axis along which the parts are chained (Z unless set)."""
        return self._main_axis or 'Z'

    @main_axis.setter
    def main_axis(self, axis: str) -> None:
        axis = str(axis).upper()
        if axis not in ('X', 'Y', 'Z'):
            raise ValueError(f"main axis must be 'X', 'Y' or 'Z', got {axis!r}")
        if self._beam is not None and self._beam.paths and self._beam.axis != axis:
            raise ValueError(
                f"The beams run along {self._beam.axis}: remove them "
                "(remove_beam / remove_beam_path) before changing the main axis.")
        self._main_axis = axis
        if isinstance(self.geometry, Assembly):
            self.geometry.set_main_axis(axis)
        if not self._loading:
            self.save()

    # =========================================================================
    # Beams: lines parallel to the main axis (docs/theory/beam.md)
    # =========================================================================
    #
    # A beam is a line current of 1 A travelling along +main_axis at the speed
    # of light; the solve adds one column per beam and one row per voltage
    # path to the generalised matrices s_tilde / z_tilde.  Every beam is also a
    # path; add_beam_path adds paths without current.  The same name replaces.

    def _beam_point(self, where: str, **position) -> tuple:
        from cavsim3d.solvers.beam import transverse_names
        axis = self.main_axis
        allowed = transverse_names(axis)
        unknown = sorted(set(position) - set(allowed))
        if unknown:
            raise TypeError(
                f"{where}: unknown position keyword(s) {unknown}; with main_axis={axis!r} "
                f"the beam's position is given by {allowed[0]}= and {allowed[1]}= (metres).")
        point = [0.0, 0.0, 0.0]
        for name in allowed:
            point['xyz'.index(name)] = float(position.get(name, 0.0))
        return tuple(point)

    def _beam_definition(self):
        from cavsim3d.solvers.beam import BeamSetup
        if self._beam is None or not self._beam.paths:
            self._beam = BeamSetup(axis=self.main_axis)
        return self._beam

    def add_beam(self, name: str = 'beam', *, beta: float = 1.0, **position):
        """Add a beam: a line current of 1 A along ``main_axis`` at the speed of light.

        The position is given by the two coordinates across the main axis, in
        metres (``x=``, ``y=`` for the default main axis Z); left out, they
        are 0 (the axis).  A beam of the same name is replaced.  ``beta`` must
        be 1 for now.  The next ``proj.fds.solve()`` adds the beam's column
        and its voltage row to ``s_tilde`` / ``z_tilde`` (labels ``b(1)``,
        ...); ``fom.beam_impedance(name)`` gives Z_par.  Returns the beam.

        >>> proj.add_beam('beam')                 # on the axis
        >>> proj.add_beam('offset', x=2e-3)       # a second beam, 2 mm off
        """
        from cavsim3d.solvers.beam import BeamLine
        line = BeamLine(name=str(name), point=self._beam_point('add_beam', **position),
                        beta=float(beta), current=True)
        setup = self._beam_definition()
        if any(l.name == line.name and not l.current for l in setup.paths):
            raise ValueError(f"{name!r} is a beam path; remove_beam_path({name!r}) first.")
        setup.add(line)
        self.save()
        return line

    def add_beam_path(self, name: str, *, beta: float = 1.0, **position):
        """Add a voltage path: a line along ``main_axis`` without current.

        Its voltage v = int E_a exp(j k_b s) ds adds a row to ``s_tilde`` /
        ``z_tilde`` (e.g. to read the field of a beam away from its own
        line).  Position keywords as in :meth:`add_beam`.  Returns the path.
        """
        from cavsim3d.solvers.beam import BeamLine
        line = BeamLine(name=str(name), point=self._beam_point('add_beam_path', **position),
                        beta=float(beta), current=False)
        setup = self._beam_definition()
        if any(l.name == line.name and l.current for l in setup.paths):
            raise ValueError(f"{name!r} is a beam; remove_beam({name!r}) first.")
        setup.add(line)
        self.save()
        return line

    def remove_beam(self, name: str) -> None:
        """Remove a beam (its own voltage path goes with it)."""
        if self._beam is None:
            raise KeyError(f"no beam named {name!r}")
        self._beam.remove(name, current=True)
        self.save()

    def remove_beam_path(self, name: str) -> None:
        """Remove a voltage path added with :meth:`add_beam_path`."""
        if self._beam is None:
            raise KeyError(f"no beam path named {name!r}")
        self._beam.remove(name, current=False)
        self.save()

    @property
    def beams(self) -> dict:
        """The beams, ``{name: BeamLine}`` in label order (``b(1)``, ...)."""
        return {l.name: l for l in self._beam.sources} if self._beam is not None else {}

    @property
    def beam_paths(self) -> dict:
        """Every voltage path, ``{name: BeamLine}``: the beams' own lines first,
        then the paths without current, in label order."""
        return {l.name: l for l in self._beam.paths} if self._beam is not None else {}

    @property
    def beam_setup(self):
        """The beam definition the solvers read (a ``BeamSetup``), or None."""
        return self._beam if self._beam is not None and self._beam.paths else None

    @property
    def parts(self) -> dict:
        """The project's parts in chain order, ``{name: part}``."""
        g = self.geometry
        if g is None:
            return {}
        if isinstance(g, Assembly):
            return {k: g._components[k].geometry for k in g._component_order}
        return {self._part_name or _default_part_name(g): g}

    def import_geometry(self, filepath: Union[str, Path], name: Optional[str] = None, *,
                        n: int = 1, flip: bool = False, after: Optional[str] = None,
                        before: Optional[str] = None, force: bool = False,
                        **kwargs) -> BaseGeometry:
        """Import a CAD file (STEP, IGES, BREP) as a part of this project.

        The first part becomes the project's geometry; further parts are
        appended and chained along ``main_axis``.  A part with the same
        ``name`` (default: the file name) is replaced.  ``**kwargs`` go to the
        importer (``unit=``, ``auto_build=`` ...).  Returns the part.
        """
        part = OCCImporter(str(filepath), **kwargs)
        name = name or Path(filepath).stem
        return self._add_part(name, part, n=n, flip=flip, after=after,
                              before=before, force=force)

    def create_primitive(self, primitive_type: str, name: Optional[str] = None, *,
                         n: int = 1, flip: bool = False, after: Optional[str] = None,
                         before: Optional[str] = None, force: bool = False,
                         **kwargs) -> BaseGeometry:
        """Create a primitive as a part of this project.

        ``primitive_type`` is one of (case and underscores ignored, so the
        class name works too):

        - ``'rectangular_waveguide'`` / ``'rwg'``, ``'circular_waveguide'`` /
          ``'cwg'`` (dimensions in metres);
        - the bodies of revolution ported from cavsim2d, with the cavsim2d
          constructor arguments (dimensions in mm unless ``unit=`` says
          otherwise): ``'elliptical_cavity'``, ``'elliptical_cavity_flattop'``
          / ``'flattop'``, ``'rfgun'``, ``'pillbox'``, ``'spline_cavity'``,
          ``'beampipe'``, ``'bla'``, ``'bellows'``, ``'taper'``.  These are
          built without a mesh: :meth:`generate_mesh` makes it (or the first
          solve, with the part's ``maxh``).

        ``**kwargs`` go to the class (``maxh`` in metres); the bodies of
        revolution also take them as one dict, ``config={...}``.  Parts are
        handled as in :meth:`import_geometry`.  Returns the part.

        >>> tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
        >>> proj.create_primitive('elliptical_cavity', name='tesla', n_cells=9,
        ...                       mid_cell=tesla, beampipe='both', maxh=0.02)
        """
        from cavsim3d.geometry import axisymmetric as axi
        mapping = {
            'rectangularwaveguide': primitives.RectangularWaveguide,
            'circularwaveguide': primitives.CircularWaveguide,
            'rwg': primitives.RectangularWaveguide,
            'cwg': primitives.CircularWaveguide,
            'ellipticalcavity': axi.EllipticalCavity,
            'ellipticalcavityflattop': axi.EllipticalCavityFlatTop,
            'flattop': axi.EllipticalCavityFlatTop,
            'rfgun': axi.RFGun,
            'pillbox': axi.Pillbox,
            'splinecavity': axi.SplineCavity,
            'beampipe': axi.Beampipe,
            'bla': axi.BLA,
            'bellows': axi.Bellows,
            'taper': axi.Taper,
        }
        key = primitive_type.lower().replace('_', '').replace(' ', '')
        cls = mapping.get(key)
        if not cls:
            raise ValueError(f"Unknown primitive type: {primitive_type!r}. Known: "
                             f"{sorted(mapping)}")
        return self._add_part(name or primitive_type.lower(), cls(**kwargs), n=n,
                              flip=flip, after=after, before=before, force=force)

    def import_project(self, project_path: Union[str, Path], name: Optional[str] = None, *,
                       mode: str = 'reference', n: int = 1, after: Optional[str] = None,
                       before: Optional[str] = None, force: bool = False):
        """Use another (solved) project as a part of this one.

        ``mode='reference'`` (default) reads the source's saved results where
        they are; ``mode='copy'`` copies them into this project so it stands
        alone (:meth:`localize` converts references later).  The source is
        never written to.  An imported project is always coupled to the other
        parts through its port modes.  Returns the imported-project handle.
        """
        from cavsim3d.core.reuse import ImportedModel
        handle = ImportedModel(project_path, mode=mode)
        name = name or Path(project_path).name
        return self._add_part(name, handle, n=n, after=after, before=before,
                              force=force)

    def add(self, name: str, part, *, n: int = 1, flip: bool = False,
            after: Optional[str] = None, before: Optional[str] = None,
            force: bool = False):
        """Add any geometry, sub-assembly or imported project as a named part."""
        return self._add_part(name, part, n=n, flip=flip, after=after,
                              before=before, force=force)

    def localize(self) -> int:
        """Copy every referenced project into this one, so it stands alone.

        Run it before sharing or archiving a project that uses
        ``import_project(..., mode='reference')``.  Already-solved results are
        copied now; parts not solved yet are copied at the next ``solve()``.
        Returns the number of parts converted.
        """
        from cavsim3d.solvers import netlist_persistence as npz
        g = self.geometry
        if not isinstance(g, Assembly):
            return 0
        refs = [e for e in g._components.values()
                if getattr(e.geometry, 'mode', None) == 'reference']
        if not refs:
            return 0

        roms_dir = self.project_path / "fds" / "foms" / "roms"
        flat = roms_dir / "structures.json"
        entries = json.loads(flat.read_text())["structures"] if flat.exists() else None
        localized = set()
        for e in refs:
            e.geometry.mode = 'copy'
            base = e.base_name
            if base in localized:
                continue
            localized.add(base)
            src = Path(e.geometry.project_path)
            if not src.exists():
                raise FileNotFoundError(
                    f"Cannot localize part '{base}': its project {src} is gone.")
            if entries is not None and any(x.get("domain") == base and
                                           x.get("source_rom_dir") for x in entries):
                npz.stage_fom(src, base, self.project_path)
                new = npz.stage_rom(src, base, self.project_path)
                entries = [new if x.get("domain") == base else x for x in entries]
        # The recorded import mode must follow, so a reopened project copies too.
        for h in g._history:
            if h.get('op') in ('add', 'replace') and h.get('geometry_type') == 'ImportedModel':
                for gh in h.get('geometry_history') or []:
                    if isinstance(gh, dict) and gh.get('op') == 'import_model':
                        gh['mode'] = 'copy'
        if entries is not None:
            npz.write_flat_structures(self.project_path, entries)
        nl = getattr(self._fds, '_netlist_foms', None) if self._fds else None
        for base, rec in (getattr(nl, '_components', {}) or {}).items():
            if base in localized:
                rec['mode'] = 'copy'
        imports = npz.read_imports(self.project_path)
        for e in refs:
            r = imports.setdefault(e.base_name, {"source": str(e.geometry.project_path),
                                                 "fingerprint": e.geometry.fingerprint()})
            r["mode"] = "copy"
        npz.write_imports(self.project_path, imports)
        self.save()
        pr.milestone(f"Localized {len(localized)} part(s): {sorted(localized)}")
        return len(localized)

    def create_importer(self, filepath: Union[str, Path], **kwargs) -> 'OCCImporter':
        """Deprecated: use :meth:`import_geometry`, which adds the part to the project."""
        import warnings
        warnings.warn(
            "create_importer() is deprecated: use proj.import_geometry(path, name=...), "
            "which adds the part to the project (a second part is chained after the "
            "first).", DeprecationWarning, stacklevel=2)
        return OCCImporter(str(filepath), **kwargs)

    def _add_part(self, name, part, n=1, flip=False, after=None, before=None,
                  force=False):
        from cavsim3d.geometry.assembly import _is_netlist_ref
        g = self.geometry
        simple = (n == 1 and not flip and after is None and before is None
                  and not _is_netlist_ref(part))

        if g is not None and not self._confirm_geometry_change(force):
            return self.geometry

        if g is None or (not isinstance(g, Assembly)
                         and name == (self._part_name or _default_part_name(g))):
            # First part, or the single part replaced.
            if simple:
                self._part_name = name
                self._setting_part = True
                try:
                    self.geometry = part                   # the setter saves
                finally:
                    self._setting_part = False
                return part
            asm = Assembly(main_axis=self._main_axis)
            asm.add(name, part, n=n, flip=flip)
            self._part_name = None
            self.geometry = asm
            return part

        if not isinstance(g, Assembly):
            # A second part: the geometry becomes a chain of both.
            asm = Assembly(main_axis=self._main_axis)
            asm.add(self._part_name or _default_part_name(g), g)
            asm.add(name, part, n=n, flip=flip, after=after, before=before)
            self._part_name = None
            self.geometry = asm
            return part

        if name in g._components or name in g._base_name_groups:
            if after is not None or before is not None:
                pr.info(f"Part '{name}' replaced in place; after=/before= ignored.")
            g.replace(name, part, n=n, flip=flip)
        else:
            g.add(name, part, n=n, flip=flip, after=after, before=before)
        self.save()
        return part

    def _confirm_geometry_change(self, force: bool) -> bool:
        """Changing the parts invalidates a mesh / results: ask unless forced."""
        if not (self.has_mesh() or self.has_results()):
            return True
        if not force and not get_user_confirmation(
            "\nWARNING: Changing the parts will invalidate the current mesh and "
            "simulation results.\nDo you want to continue and delete existing results?"
        ):
            pr.info("Keeping the existing parts and results.")
            return False
        self.invalidate_mesh()
        return True

    def generate_mesh(self, force: bool = False, **kwargs) -> Mesh:
        """
        Generate mesh from current geometry.

        Automatically invalidates existing simulation results if the mesh changes.
        With a beam defined (:meth:`add_beam`), the mesh is curved to order 4
        unless ``curve_order`` is given: the beam impedance is sensitive to
        how closely the mesh follows curved walls.
        """
        if self.geometry is None:
            raise RuntimeError("Cannot generate mesh without geometry.")
        if self.beam_setup is not None and 'curve_order' not in kwargs:
            from cavsim3d.solvers.beam import BEAM_CURVE_ORDER
            kwargs['curve_order'] = BEAM_CURVE_ORDER

        if self.has_results():
            # force only skips the question: results of the old mesh must go
            # either way, or the next solve() would return them unchanged.
            if not force and not get_user_confirmation(
                "\nWARNING: Re-generating the mesh will invalidate existing simulation results.\n"
                "Do you want to continue and delete existing results?"
            ):
                pr.info("Aborting mesh generation.")
                return self.mesh

            self.invalidate_results()

        self.mesh = self.geometry.generate_mesh(**kwargs)  # the setter saves
        if self.mesh is None:
            # A coupled chain has no project-level mesh (each part is meshed
            # when solved); still save, so changes made to the chain persist.
            self.save()
        return self.mesh
    
    def draw_material_cf(self, which: str = 'eps'):
        """Draw the relative permittivity ('eps') or permeability ('mu') map."""
        self.fds.draw_material_cf(which)

    def has_mesh(self) -> bool:
        """Check if mesh exists (either in memory or on disk)."""
        if self.mesh is not None:
            return True
        return (self.mesh_path / "mesh.pkl").exists()

    def has_results(self) -> bool:
        """Check if any simulation results exist (either in memory or on disk)."""
        # Check in memory
        if self._fds and (getattr(self._fds, '_fom_cache', None) or getattr(self._fds, '_resonant_mode_cache', None)):
            return True
        
        # Check on disk: the solver's saved state says whether it holds
        # results (a netlist's fds/config.json holds its solve settings only)
        config = self.fds_path / "config.json"
        if not config.exists():
            return False
        try:
            saved = json.loads(config.read_text())
        except (OSError, ValueError):
            return True                     # unreadable: treat as results
        return bool(saved.get("has_results", True))

    def invalidate_mesh(self) -> None:
        """Invalidate the mesh and all downstream results (fom, rom, etc.)."""
        pr.info(f"Invalidating mesh for project '{self.name}'...")
        
        # 1. Physical Cleanup (Content only, preserve directory)
        if self.mesh_path.exists():
            for item in self.mesh_path.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        
        # 2. In-Memory Reset
        self.mesh = None
        if self.geometry:
            self.geometry.mesh = None # Sync with geometry object

        # 3. Propagate Downstream
        self.invalidate_results()

    def invalidate_results(self) -> None:
        """Invalidate all simulation results (fom, rom, concat, etc.)."""
        pr.info(f"Invalidating simulation results for project '{self.name}'...")
        
        # 1. Physical Cleanup (Content only, preserve directories)
        for path in [self.fds_path, self.fom_path, self.foms_path, self.eigenmode_path]:
            if path.exists():
                for item in path.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        
        # 2. In-Memory Reset
        if self._fds:
            # We don't delete self._fds object, but we clear its entire state.
            # Use full_reset to ensure matrices, FES, and flags are cleared.
            self._fds.full_reset()
            self._fds.mesh = self.mesh # Ensure solver still has access to new mesh if it exists

        # 3. project.json must describe what is left on disk, or reopening
        # would look for the deleted solver state.
        if not self._loading:
            self.save()

    @property
    def geometry(self) -> Optional[BaseGeometry]:
        """Current project geometry."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: Optional[BaseGeometry]):
        # A geometry assigned directly has no part name (it gets a default);
        # _add_part sets the name itself.
        if not getattr(self, '_loading', False) and not getattr(self, '_setting_part', False):
            self._part_name = None
        self._geometry = value
        # Sync solver with new geometry object
        if self._fds:
            self._fds.geometry = value

        # Persist like the mesh setter does (but NOT during _initial_load), so
        # `proj.geometry = g` survives a restart.
        if value is not None and not getattr(self, '_loading', False):
            self.save()

    @property
    def mesh(self) -> Optional[Mesh]:
        """Project mesh (NGSolve Mesh object)."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Optional[Mesh]):
        self._mesh = value
        if self._fds:
            self._fds.mesh = value
        
        # Auto-save mesh to disk if it exists (but NOT during _initial_load)
        if value is not None and not getattr(self, '_loading', False):
            self.save()

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, value: int):
        self._order = value
        if self._fds:
            self._fds.order = value

    @property
    def n_port_modes(self) -> int:
        """Deprecated: has no effect.  Set the port modes per solve with
        ``proj.fds.solve(..., nportmodes=...)``."""
        return self._n_port_modes

    @n_port_modes.setter
    def n_port_modes(self, value: int):
        import warnings
        warnings.warn(
            "proj.n_port_modes has no effect: give the number of port modes to the "
            "solve, proj.fds.solve(..., nportmodes=N) (an int, a list per port, or "
            "a dict {port: N}).", DeprecationWarning, stacklevel=2)
        self._n_port_modes = value

    @property
    def mesh_path(self) -> Path:
        return self.project_path / "mesh"

    @property
    def fds_path(self) -> Path:
        return self.project_path / "fds"

    @property
    def geometry_path(self) -> Path:
        return self.project_path / "geometry"

    @property
    def fom_path(self) -> Path:
        return self.project_path / "fom"

    @property
    def foms_path(self) -> Path:
        return self.project_path / "foms"

    @property
    def eigenmode_path(self) -> Path:
        return self.project_path / "eigenmode"

    def save(self):
        """Save the entire project with the new folder structure."""
        if self._read_only:
            pr.debug(f"Project {self.project_path} is open read-only: not saved.")
            return
        pr.info(f"Saving project to {self.project_path}")
        
        # 1. Save Geometry
        if self.geometry:
            self.geometry.save_geometry(self.project_path)
            
        # 2. Save Mesh
        # We prefer to save the mesh that is currently in use by the solver
        current_mesh = self.mesh
        if self._fds and self._fds.mesh:
            current_mesh = self._fds.mesh
            
        if current_mesh:
            self.mesh_path.mkdir(parents=True, exist_ok=True)
            pm = ProjectManager(self.base_dir)
            pm.save_ngs_mesh(self.mesh_path, current_mesh)
            
            # Also save global FES if available from solver
            if self._fds and hasattr(self._fds, '_fes_global') and self._fds._fes_global:
                pm.save_ngs_fes(self.mesh_path, self._fds._fes_global)

        # 3. Save FDS Results
        if self._fds:
            self._fds._project_path = self.project_path
            self.fds_path.mkdir(parents=True, exist_ok=True)
            self._fds.save(path=self.fds_path)

        # 4. Global Metadata
        metadata = {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "has_geometry": self.geometry is not None,
            "has_mesh": current_mesh is not None,
            "has_fds": self._fds is not None,
            "order": self._order,
            "n_port_modes": self._n_port_modes,
            "bc": self.bc,
            "main_axis": self._main_axis,
            "part_name": self._part_name,
            "beam": self._beam.to_dict() if self._beam is not None and self._beam.paths else None,
        }
        ProjectManager.save_json(self.project_path, metadata, filename="project.json")

        # 5. Save timing analysis
        self.save_timing()

    def save_timing(self) -> None:
        """Write the timing analysis (``timing.json``).  A reduced or joined
        solve saves its own results and this, not the whole project."""
        if self._read_only:
            return
        try:
            from cavsim3d.utils.timing import get_timing_registry
            reg = get_timing_registry()
            if reg.entries:
                reg.save(self.project_path / "timing.json")
        except Exception:
            pass

    @property
    def timing(self):
        """Shared timing registry (FOM / ROM / concat records)."""
        from cavsim3d.utils.timing import get_timing_registry
        return get_timing_registry()

    def timing_summary(self, save: bool = True) -> str:
        """Return (and print) a comparison table of FOM / ROM / concat timing.

        Includes per-stage wall-clock times, the achieved model-order
        reduction (full -> reduced DOFs, % compression) and the per-sample
        speed-up of the ROM / concatenated solves over the full-order solve.

        Parameters
        ----------
        save : bool
            Also write ``timing.json`` to the project directory.
        """
        from cavsim3d.utils.timing import get_timing_registry
        reg = get_timing_registry()
        text = reg.summary(title=f"TIMING SUMMARY - {self.name}")
        print(text)
        if save and reg.entries:
            try:
                reg.save(self.project_path / "timing.json")
            except Exception:
                pass
        return text

    @staticmethod
    def _force_rmtree(path: Path) -> None:
        """Robustly delete a directory tree, even on Windows.

        Closes any open log-file handles first (an open handle locks the file
        and makes deletion fail on Windows), clears read-only bits via an
        error handler, and retries briefly to ride out transient locks.
        """
        import time
        import stat

        # Release our own file handles (e.g. solve.log) so they aren't locked.
        try:
            pr.close_all_file_logs()
        except Exception:
            pass

        def _on_error(func, p, exc_info):
            # Clear read-only attribute and retry the operation once.
            try:
                os.chmod(p, stat.S_IWRITE)
                func(p)
            except Exception:
                pass

        for attempt in range(5):
            if not path.exists():
                return
            try:
                # onexc (3.12+) / onerror (older) — pass whichever is supported.
                try:
                    shutil.rmtree(path, onexc=lambda f, p, e: _on_error(f, p, e))
                except TypeError:
                    shutil.rmtree(path, onerror=lambda f, p, e: _on_error(f, p, e))
            except Exception:
                pass
            if not path.exists():
                return
            time.sleep(0.3)

    @classmethod
    def load(cls, name: str, base_dir: Optional[Union[str, Path]] = None,
             overwrite: bool = False) -> EMProject:
        """Load a project from disk (``EMProject(name, base_dir)`` does the same).

        ``overwrite`` is accepted for compatibility only: ``overwrite=True``
        would delete the project being loaded, so it is refused.
        """
        if overwrite:
            raise ValueError(
                "EMProject.load() opens an existing project; overwrite=True would "
                "delete it. To start afresh use EMProject(name, base_dir, overwrite=True).")
        base_dir = Path(base_dir) if base_dir else Path.cwd()
        if not (base_dir / _check_project_name(name)).is_dir():
            raise FileNotFoundError(f"No project '{name}' in {base_dir}.")
        # The __init__ already handles searching and automatic loading
        return cls(name=name, base_dir=base_dir)

    def __repr__(self) -> str:
        return f"EMProject({self.name}, path={self.project_path})"
