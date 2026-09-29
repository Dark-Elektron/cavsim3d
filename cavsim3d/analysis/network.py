"""Compare the network parameters (S, Z) of several models.

Every function takes, for each model, one of

- a solved result object with ``S_dict`` / ``Z_dict`` (FOM, ROM or
  concatenated system, or a :class:`~cavsim3d.analytical.cst_result.CSTResult`),
- a parameter dictionary keyed ``'<port>(<mode>)<port>(<mode>)'``, or
- an array ``[frequency, response, excitation]``.

Port modes are named ``'<port>(<mode>)'``, e.g. ``'2(1)'``.  The code keys
its dictionaries excitation first; CST keys them response first (``S_ij`` is
the response at ``i`` to an excitation at ``j``).  :func:`network_matrix`
reads both and always returns ``[frequency, response, excitation]``.

Examples
--------
>>> S, Z = keep_port_modes(concat, ["1(1)", "2(1)", "3(1)"])
>>> S_cst = network_matrix(cst, ["1(1)", "2(1)", "3(1)"])
>>> plot_matrix({"CST": S_cst, "FEM": S}, freq=f_ghz)
>>> band_difference(S, S_cst, f_ghz, edges=[0.1, 0.5, 1.0])
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from cavsim3d.utils.plot_mixin import PlotMixin

_PORT_MODE = re.compile(r"\d+\(\d+\)")

Model = Union[np.ndarray, dict, object]


# ---------------------------------------------------------------------------
# Parameter matrices
# ---------------------------------------------------------------------------

def _param_dict(model, kind: str) -> dict:
    if isinstance(model, dict):
        return model
    d = getattr(model, f"{kind.upper()}_dict", None)
    if d is None:
        raise ValueError(f"{type(model).__name__} has no {kind.upper()}-parameters; "
                         f"solve it first.")
    return d


def _is_cst(model) -> bool:
    from cavsim3d.analytical.cst_result import CSTResult
    return isinstance(model, CSTResult)


def _mode_key(label: str) -> Tuple[int, ...]:
    return tuple(int(v) for v in re.findall(r"\d+", label))


def port_mode_labels(model, kind: str = "S") -> List[str]:
    """Port modes of a model, as ``'port(mode)'`` labels in port, then mode order."""
    labels = {m for k in _param_dict(model, kind) if k != "frequencies"
              for m in _PORT_MODE.findall(k)}
    return sorted(labels, key=_mode_key)


def _block(d: dict, rows: Sequence[str], cols: Sequence[str],
           excitation_first: bool) -> np.ndarray:
    def key(r, c):
        return c + r if excitation_first else r + c

    missing = [key(r, c) for r in rows for c in cols if key(r, c) not in d]
    if missing:
        raise KeyError(f"{len(missing)} entries missing, e.g. {missing[:3]}; "
                       f"available port modes: {sorted({m for k in d for m in _PORT_MODE.findall(k)}, key=_mode_key)}")
    return np.stack([np.stack([np.asarray(d[key(r, c)]) for c in cols], -1)
                     for r in rows], -2)


def network_matrix(model, labels: Optional[Sequence[str]] = None, kind: str = "S",
                   excitation_first: Optional[bool] = None) -> np.ndarray:
    """S- or Z-parameters between the given port modes as one array.

    Parameters
    ----------
    model : result object or dict
        Anything with ``S_dict`` / ``Z_dict``, or such a dictionary.
    labels : list of str, optional
        Port modes ``'port(mode)'`` in the order wanted.  Default: all of the
        model's port modes (:func:`port_mode_labels`).
    kind : {'S', 'Z'}
    excitation_first : bool, optional
        Key convention of the dictionary.  Default: ``False`` for a
        ``CSTResult`` (CST names the response first), ``True`` otherwise.

    Returns
    -------
    ndarray, shape (n_freq, n, n)
        Indexed ``[frequency, response, excitation]``.
    """
    d = _param_dict(model, kind)
    if labels is None:
        labels = port_mode_labels(d, kind)
    if excitation_first is None:
        excitation_first = not _is_cst(model)
    return _block(d, labels, labels, excitation_first)


def keep_port_modes(model, keep: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """S and Z on the port modes ``keep``; every other port mode is open-circuited.

    A port carries no current in a mode it does not define (``I = 0``: an open
    circuit).  To compare with a model that defines fewer port modes, the extra
    modes are closed the same way.  With pseudo-wave S-parameters an open
    circuit reflects with Gamma = +1, so for kept modes ``k`` and closed
    modes ``d``::

        S' = S_kk + S_kd (I - S_dd)^-1 S_dk,    Z' = Z_kk

    Parameters
    ----------
    model : result object
        Solved model with ``S_dict`` and ``Z_dict``.
    keep : list of str
        Port modes ``'port(mode)'`` to keep, in the order wanted.

    Returns
    -------
    S, Z : ndarray, shape (n_freq, len(keep), len(keep))
        Indexed ``[frequency, response, excitation]``.
    """
    keep = list(keep)
    s_dict = _param_dict(model, "S")
    exc_first = not _is_cst(model)
    drop = [m for m in port_mode_labels(s_dict) if m not in keep]

    def block(rows, cols):
        return _block(s_dict, rows, cols, exc_first)

    S = block(keep, keep)
    if drop:
        S = S + block(keep, drop) @ np.linalg.solve(
            np.eye(len(drop)) - block(drop, drop), block(drop, keep))
    Z = _block(_param_dict(model, "Z"), keep, keep, exc_first)
    return S, Z


def _as_array(model, kind: str, freq) -> Tuple[np.ndarray, np.ndarray]:
    """(frequency in GHz, [frequency, response, excitation]) of one model."""
    if isinstance(model, np.ndarray):
        if freq is None:
            raise ValueError("freq (GHz) is required when models are given as arrays.")
        return np.asarray(freq), model
    f = getattr(model, "frequencies", None)
    if f is None and isinstance(model, dict):
        f = model.get("frequencies")
    f = np.asarray(f) / 1e9 if f is not None else freq
    if f is None:
        raise ValueError("freq (GHz) is required for a model without frequencies.")
    return np.asarray(f), network_matrix(model, kind=kind)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _styles(n: int) -> List[dict]:
    # the first model is the reference: a wide line the others are drawn over
    return [dict(lw=2.6 if k == 0 else 1.2, zorder=1 + k) for k in range(n)]


def _entry_title(kind: str, labels: Sequence[str], i: int, j: int) -> str:
    li, lj = str(labels[i]), str(labels[j])
    sep = "" if len(li) == len(lj) == 1 else ","
    return f"${kind}_{{{li}{sep}{lj}}}$"


def plot_matrix(models: Dict[str, Model], freq=None, ports: Optional[Sequence[int]] = None,
                labels: Optional[Sequence[str]] = None, kind: str = "S",
                plot_type: str = "db", title: Optional[str] = None,
                figsize: Optional[Tuple[float, float]] = None, show: bool = False):
    """Grid of panels, one per matrix entry, with every model overlaid.

    Parameters
    ----------
    models : dict
        ``{legend label: model}``; each model is a result object, a parameter
        dict or an array ``[frequency, response, excitation]``.  All models
        must share one port-mode order.  The first is the reference, drawn with
        a wide line.
    freq : array, optional
        Frequencies in GHz of the array models (result objects carry their own).
    ports : list of int, optional
        Matrix rows/columns to show (default: all).
    labels : list of str, optional
        Names of all rows/columns, used in the panel titles (default 1, 2, ...).
    kind : {'S', 'Z'}
        Parameter symbol; also which parameters are read from result objects.
    plot_type : {'db', 'mag', 'phase', 're', 'im'}
    title : str, optional
        Figure title.
    figsize : tuple, optional
    show : bool
        Call ``plt.show()``.

    Returns
    -------
    fig, axs
    """
    data = {name: _as_array(m, kind, freq) for name, m in models.items()}
    n_all = next(iter(data.values()))[1].shape[1]
    ports = list(range(n_all)) if ports is None else list(ports)
    labels = [str(k + 1) for k in range(n_all)] if labels is None else list(labels)
    n = len(ports)
    fig, axs = plt.subplots(n, n, figsize=figsize or (2.7 * n, 2.2 * n),
                            sharex=True, squeeze=False)
    styles = _styles(len(data))
    for r, i in enumerate(ports):
        for c, j in enumerate(ports):
            ax = axs[r, c]
            for (name, (f, A)), st in zip(data.items(), styles):
                ylabel = PlotMixin._apply_data(ax, f, A[:, i, j], plot_type, name, st)
            ax.set_title(_entry_title(kind, labels, i, j), fontsize=11)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if r == n - 1:
                ax.set_xlabel("f (GHz)", fontsize=9)
    axs[0, -1].legend(loc="best", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


def plot_entries(models: Dict[str, Model], entries: Iterable[tuple], freq=None,
                 kind: str = "S", plot_type: str = "db",
                 figsize: Optional[Tuple[float, float]] = None, show: bool = False):
    """A row of panels, one per chosen matrix entry, with every model overlaid.

    Parameters
    ----------
    models : dict
        As in :func:`plot_matrix`; the first model is the reference.
    entries : list of tuple
        ``(i, j)`` or ``(i, j, title)`` -- response row ``i``, excitation
        column ``j`` (0-based).
    freq, kind, plot_type, figsize, show
        As in :func:`plot_matrix`.

    Returns
    -------
    fig, axs
    """
    entries = [tuple(e) for e in entries]
    data = {name: _as_array(m, kind, freq) for name, m in models.items()}
    fig, axs = plt.subplots(1, len(entries), figsize=figsize or (5.3 * len(entries), 4.2),
                            squeeze=False)
    axs = axs[0]
    styles = _styles(len(data))
    for ax, entry in zip(axs, entries):
        i, j = entry[:2]
        for (name, (f, A)), st in zip(data.items(), styles):
            ylabel = PlotMixin._apply_data(ax, f, A[:, i, j], plot_type, name, st)
        ax.set_title(entry[2] if len(entry) > 2
                     else _entry_title(kind, [str(k + 1) for k in range(max(i, j) + 1)], i, j),
                     fontsize=11)
        ax.set_xlabel("f (GHz)")
        ax.set_ylabel(f"|{kind}| ({ylabel})" if plot_type in ("db", "mag") else ylabel)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=9)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axs


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def band_difference(model: Model, reference: Model, freq=None, edges: Sequence[float] = (),
                    ports: Optional[Sequence[int]] = None, kind: str = "S") -> Dict[str, float]:
    """Mean magnitude difference to a reference, per frequency band.

    The mean of ``| |A| - |A_ref| |`` over the frequencies in each band and
    over the matrix entries between ``ports``.

    Parameters
    ----------
    model, reference : result object, dict or array
        On the same frequency grid and port-mode order.
    freq : array, optional
        Frequencies in GHz of array models.
    edges : list of float
        Band edges in GHz; band ``k`` is ``edges[k] <= f <= edges[k+1]``.
    ports : list of int, optional
        Matrix rows/columns included (default: all).
    kind : {'S', 'Z'}

    Returns
    -------
    dict
        ``{'<lo>-<hi>': mean difference}``, ready for ``pandas.DataFrame``.
    """
    f, A = _as_array(model, kind, freq)
    f_ref, A_ref = _as_array(reference, kind, freq)
    if A.shape != A_ref.shape or not np.allclose(f, f_ref):
        raise ValueError("model and reference must share the frequency grid and port modes; "
                         f"got {A.shape} and {A_ref.shape}.")
    if ports is not None:
        idx = np.asarray(ports)
        A, A_ref = A[:, idx][:, :, idx], A_ref[:, idx][:, :, idx]
    diff = np.abs(np.abs(A) - np.abs(A_ref))
    return {f"{lo:.3f}-{hi:.3f}": float(diff[(f >= lo) & (f <= hi)].mean())
            for lo, hi in zip(edges[:-1], edges[1:])}
