"""Per-sample checkpoint of a frequency sweep, so an interrupted ``solve()`` resumes.

Every finished sample is written to ``<project>/fds/checkpoint/<key>/`` as one
file (a netlist section solved in a scratch project writes to
``<importing project>/fds/checkpoint/sections/<section>/<key>/``).  A later ``solve()`` of the same system at the same frequencies reads
those samples back instead of recomputing them.  The folder is removed once
the complete results are saved.

A sample file is written under a temporary name and then renamed, so an
interrupt while writing never leaves a damaged sample behind.

A sample is reused only for the same sweep: the same frequencies, FE space
and materials (``fingerprint``, exact) and the same port excitations -- the
right-hand sides, which depend on the mesh, the order and the port modes.
Those are compared numerically (``rhs_signature``, to 1e-8), because
re-assembling them after reopening a project can change the last bits.  The
solver type is left out: direct and iterative samples agree to the solver
tolerance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import cavsim3d.utils.printing as pr


def sweep_fingerprint(frequencies, ndof: int, n_free: int, n_rhs: int, materials,
                      beam: Optional[str] = None) -> str:
    """Exact part of a sweep's identity (``beam``: the beam definition's
    fingerprint, for a sweep with beam columns)."""
    payload = {
        "frequencies_hz": [round(float(f)) for f in frequencies],
        "ndof": int(ndof), "n_free": int(n_free), "n_rhs": int(n_rhs),
        "materials": repr(materials),
    }
    if beam is not None:
        payload["beam"] = str(beam)
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def rhs_signature(rhs: np.ndarray) -> np.ndarray:
    """Column norms and projections onto a fixed random vector of the RHS matrix."""
    rhs = np.asarray(rhs)
    w = np.random.default_rng(20260926).standard_normal(rhs.shape[0])
    return np.concatenate([np.linalg.norm(rhs, axis=0), w @ rhs])


class SweepCheckpoint:
    """Samples of one sweep (``key``: ``'global'`` or a domain name) on disk.

    ``folder=None`` (a solver without a project) disables checkpointing.
    """

    def __init__(self, folder: Optional[Path], key: str, fingerprint: str = "",
                 signature: Optional[np.ndarray] = None):
        self.folder = Path(folder) / key if folder is not None else None
        self.fingerprint = fingerprint
        self.signature = (np.asarray(signature) if signature is not None
                          else np.zeros(0))

    def _file(self, k: int) -> Path:
        return self.folder / f"sample_{k:05d}.npz"

    def _matches(self, d) -> bool:
        sig = d["rhs_signature"]
        return (str(d["fingerprint"]) == self.fingerprint
                and sig.shape == self.signature.shape
                and np.allclose(sig, self.signature, rtol=1e-8,
                                atol=1e-12 * (np.abs(self.signature).max(initial=0) + 1e-300)))

    def load(self) -> Dict[int, dict]:
        """Samples already computed for this sweep, ``{index: record}``.

        Samples of a different sweep are deleted (they will be recomputed).
        """
        if self.folder is None or not self.folder.is_dir():
            return {}
        done, stale = {}, []
        for f in sorted(self.folder.glob("sample_*.npz")):
            if f.name.endswith(".tmp.npz"):
                continue
            try:
                with np.load(f, allow_pickle=False) as d:
                    if not self._matches(d):
                        stale.append(f)
                        continue
                    rec = {
                        "Z": d["Z"], "iters": d["iters"], "res": d["res"],
                        "time": float(d["time"]),
                        "x": d["x"] if "x" in d.files else None,
                    }
                    for name in d.files:                 # beam outputs, if any
                        if name.startswith("beam_"):
                            rec[name] = d[name]
                    done[int(d["index"])] = rec
            except Exception as e:                      # unreadable: recompute it
                pr.warning(f"Ignoring unreadable checkpoint sample {f.name}: {e}")
                stale.append(f)
        if stale:
            pr.info(f"  {len(stale)} checkpoint sample(s) in {self.folder} belong to a "
                    f"different sweep; they are recomputed.")
            for f in stale:
                f.unlink(missing_ok=True)
        return done

    def write(self, k: int, Z: np.ndarray, x: Optional[np.ndarray], iters, res,
              time_s: float, extra: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Store sample ``k`` (``x``: its solutions, one column per excitation;
        ``extra``: further arrays, e.g. the beam outputs ``beam_*``)."""
        if self.folder is None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / f"sample_{k:05d}.tmp.npz"
        arrays = dict(fingerprint=np.array(self.fingerprint),
                      rhs_signature=self.signature, index=np.array(k),
                      Z=np.asarray(Z), iters=np.asarray(iters),
                      res=np.asarray(res), time=np.array(time_s))
        if x is not None:
            arrays["x"] = np.asarray(x)
        for name, value in (extra or {}).items():
            arrays[name] = np.asarray(value)
        np.savez(tmp, **arrays)
        os.replace(tmp, self._file(k))


def clear_checkpoints(folder: Optional[Path]) -> None:
    """Remove a checkpoint folder (after the results it protected are saved)."""
    if folder is not None and Path(folder).exists():
        shutil.rmtree(folder, ignore_errors=True)
