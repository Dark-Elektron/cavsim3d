"""Keep netgen's OpenCASCADE apart from pythonocc's (Linux).

pythonocc-core runs on conda-forge's OpenCASCADE, and NGSolve's netgen loads its own
copy (the ``netgen-occt`` wheel) with ``RTLD_GLOBAL``. On Linux, a pythonocc module
imported after that binds to netgen's copy instead of its own, so a shape built by one
copy is handed to the other: reading a STEP file then fails with
``Standard_NoSuchObject`` or "Wrong number or type of arguments". Loaded with
``RTLD_LOCAL`` instead, each copy is used only by the libraries linked to it, as on
macOS, whatever the import order.
"""
import ctypes
import importlib.abc
import importlib.machinery
import sys


class _LocalOCCTLoader(importlib.abc.Loader):
    """netgen's own loader, run with ``ctypes.RTLD_GLOBAL`` meaning local."""

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        shared = ctypes.RTLD_GLOBAL
        ctypes.RTLD_GLOBAL = ctypes.RTLD_LOCAL      # read by netgen's load_occ_libs()
        try:
            self._loader.exec_module(module)
        finally:
            ctypes.RTLD_GLOBAL = shared

    def __getattr__(self, name):        # get_filename(), get_data(), ... of netgen's loader
        try:
            loader = self.__dict__["_loader"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(loader, name)


class _NetgenFinder(importlib.abc.MetaPathFinder):
    """Hands the first ``import netgen`` to :class:`_LocalOCCTLoader`, then leaves."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "netgen":
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _LocalOCCTLoader(spec.loader)
        return spec


def keep_netgen_occt_private() -> None:
    """Make the coming ``import netgen`` load its OpenCASCADE locally (Linux only)."""
    if not sys.platform.startswith("linux") or "netgen" in sys.modules:
        return
    if not any(isinstance(finder, _NetgenFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _NetgenFinder())
