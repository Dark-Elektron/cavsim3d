"""Options accepted by the solve() methods, and checks of the requested sweep.

A misspelt option (``n_port_modes=2`` for ``nportmodes=2``) must not run with
defaults, so every ``solve()`` rejects the names it does not know.  A config
dict written for the full-order solve may be reused for the reduced and joined
models: the full-order options are accepted there and have no effect.
"""

import difflib
from numbers import Integral, Real
from typing import Iterable

import numpy as np

# fds.solve(): full-order sweep (docs/reference/solve_options.md)
FOM_SOLVE_OPTIONS = frozenset({
    "fmin", "fmax", "nsamples", "order", "nedelec", "nportmodes",
    "store_snapshots", "compute_s_params", "per_domain", "global_method",
    "solver_type", "iterative_opts", "rerun", "verbose", "impedance_reference",
    "mode_source", "mode_source_internal", "qtem_ports", "qtem_conductor_bbnd",
    "qtem_voltage_path",
})

# rom.solve() / concat.solve(): sweep of a reduced or joined model
REDUCED_SOLVE_OPTIONS = frozenset({
    "fmin", "fmax", "nsamples", "solver_type", "rerun", "verbose",
    "compute_s_params",
})


def check_solve_options(cfg: dict, allowed: Iterable[str],
                        also_accepted: Iterable[str] = (),
                        where: str = "solve()") -> None:
    """Raise TypeError naming every key of ``cfg`` that is not an option.

    ``also_accepted`` lists names that are tolerated without effect (the
    full-order options, when a config dict is reused for a reduced model).
    """
    allowed = set(allowed)
    known = allowed | set(also_accepted)
    unknown = sorted(k for k in cfg if k not in known)
    if not unknown:
        return
    hints = []
    for k in unknown:
        close = difflib.get_close_matches(str(k), sorted(known), n=1, cutoff=0.6)
        if not close:
            close = [a for a in sorted(known)
                     if str(k).replace('_', '').lower() == a.replace('_', '').lower()]
        hints.append(f"'{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
    raise TypeError(
        f"{where} got unknown option(s): {', '.join(hints)}. "
        f"Valid options: {', '.join(sorted(allowed))}.")


def validate_sweep(fmin, fmax, nsamples) -> int:
    """Check a requested sweep (GHz) and return ``nsamples`` as an int.

    ``fmin`` must be > 0 (at f = 0 the curl-curl system is singular), both
    ends finite, ``fmax >= fmin``, and ``nsamples`` a whole number >= 1.
    """
    for name, val in (("fmin", fmin), ("fmax", fmax)):
        if isinstance(val, bool) or not isinstance(val, Real) or not np.isfinite(val):
            raise ValueError(f"{name} must be a finite number in GHz (got {val!r}).")
    if fmin <= 0:
        raise ValueError(
            f"fmin must be > 0 GHz (got {fmin}). At f = 0 the curl-curl "
            f"system is singular (every gradient field is a solution).")
    if fmax < fmin:
        raise ValueError(f"fmax ({fmax}) must be >= fmin ({fmin}).")
    if isinstance(nsamples, bool) or not isinstance(nsamples, Real):
        raise ValueError(f"nsamples must be a whole number >= 1 (got {nsamples!r}).")
    if not isinstance(nsamples, Integral):
        if not float(nsamples).is_integer():
            raise ValueError(f"nsamples must be a whole number >= 1 (got {nsamples!r}).")
    n = int(nsamples)
    if n < 1:
        raise ValueError(f"nsamples must be >= 1 (got {nsamples}).")
    return n
