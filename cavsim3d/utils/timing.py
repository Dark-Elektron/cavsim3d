"""Lightweight timing registry for FOM / ROM / concatenation comparison.

A single :class:`TimingRegistry` collects named, categorised timing records
(plus metadata such as DOF counts and sample counts) produced while solving.
It can render a human-readable summary table that compares the full-order
model, the reduced-order model and the concatenated system — including how
much the system was reduced and the resulting speed-up — and can be persisted
to / loaded from the project directory.

Usage
-----
>>> reg = get_timing_registry()
>>> with reg.measure("coupled solve", category="FOM", n_samples=10, n_dofs=1_581_915):
...     ...                      # the timed work
>>> reg.record("reduction", 5.2, category="ROM",
...            full_dofs=1_581_915, reduced_dofs=24)
>>> print(reg.summary())
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Canonical category order for reporting.
_CATEGORY_ORDER = ["FOM", "ROM", "CONCAT", "GENERAL"]


class TimingRegistry:
    """Collects timing records and renders a comparison summary."""

    def __init__(self, enabled: bool = True):
        self._entries: List[Dict[str, Any]] = []
        self.enabled: bool = enabled

    # ------------------------------------------------------------------ record
    def record(
        self,
        stage: str,
        seconds: float,
        category: str = "GENERAL",
        **metadata: Any,
    ) -> None:
        """Add a timing record.

        Parameters
        ----------
        stage : str
            Human label for the timed operation (e.g. ``"coupled solve"``).
        seconds : float
            Wall-clock duration.
        category : str
            Grouping bucket — typically ``"FOM"``, ``"ROM"`` or ``"CONCAT"``.
        **metadata
            Extra fields stored alongside the record, e.g. ``n_samples``,
            ``n_dofs``, ``full_dofs``, ``reduced_dofs``.
        """
        if not self.enabled:
            return
        entry = {
            "stage": stage,
            "category": category.upper(),
            "seconds": float(seconds),
        }
        entry.update({k: v for k, v in metadata.items() if v is not None})
        self._entries.append(entry)

    @contextmanager
    def measure(self, stage: str, category: str = "GENERAL", **metadata: Any):
        """Context manager that records the wall-clock time of the block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - t0, category, **metadata)

    # ------------------------------------------------------------------ access
    @property
    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def total(self, category: Optional[str] = None) -> float:
        """Total seconds, optionally restricted to one category."""
        cat = category.upper() if category else None
        return sum(e["seconds"] for e in self._entries
                   if cat is None or e["category"] == cat)

    def _latest_meta(self, category: str, *keys: str) -> Optional[Any]:
        """Return the first present metadata value (latest entry first)."""
        for e in reversed(self._entries):
            if e["category"] != category.upper():
                continue
            for k in keys:
                if k in e:
                    return e[k]
        return None

    # ----------------------------------------------------------------- summary
    def reduction_info(self) -> Optional[Dict[str, float]]:
        """Full vs reduced DOF counts and compression, if recorded."""
        full = self._latest_meta("ROM", "full_dofs", "n_dofs")
        reduced = self._latest_meta("ROM", "reduced_dofs")
        if full and reduced:
            comp = 100.0 * (1.0 - reduced / full)
            return {"full_dofs": full, "reduced_dofs": reduced, "compression_pct": comp}
        return None

    def summary(self, title: str = "TIMING SUMMARY") -> str:
        """Build a readable comparison table of all recorded stages."""
        if not self._entries:
            return "(no timing data recorded)"

        # Order entries by canonical category then insertion order.
        def _cat_key(e):
            cat = e["category"]
            return _CATEGORY_ORDER.index(cat) if cat in _CATEGORY_ORDER else len(_CATEGORY_ORDER)

        ordered = sorted(enumerate(self._entries), key=lambda x: (_cat_key(x[1]), x[0]))

        rows = []
        for _, e in ordered:
            n = e.get("n_samples")
            secs = e["seconds"]
            per = f"{secs / n:.4g}" if n else "-"
            rows.append((
                e["category"],
                e["stage"],
                f"{secs:.3f}",
                str(n) if n else "-",
                per,
            ))

        headers = ("Category", "Stage", "Time [s]", "Samples", "s/sample")
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        sep = "-" * len(line)
        out = [f"{'=' * len(line)}", title, "=" * len(line), line, sep]
        for r in rows:
            out.append("  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
        out.append(sep)

        # Totals per category.
        for cat in _CATEGORY_ORDER:
            t = self.total(cat)
            if t > 0:
                out.append(f"{cat} total: {t:.3f} s")

        # Reduction info.
        red = self.reduction_info()
        if red:
            out.append(
                f"Reduction: {int(red['full_dofs']):,} -> {int(red['reduced_dofs']):,} DOFs "
                f"({red['compression_pct']:.1f}% compression)"
            )

        # Speed-up: ROM/CONCAT solve per-sample vs FOM solve per-sample.
        fom_ps = self._per_sample_solve("FOM")
        for cat in ("ROM", "CONCAT"):
            ps = self._per_sample_solve(cat)
            if fom_ps and ps and ps > 0:
                out.append(f"Speed-up ({cat} vs FOM, per sample): {fom_ps / ps:,.0f}x")

        out.append("=" * len(line))
        return "\n".join(out)

    def _per_sample_solve(self, category: str) -> Optional[float]:
        """Per-sample time of the 'solve' stage in a category, if available."""
        for e in reversed(self._entries):
            if e["category"] == category.upper() and "solve" in e["stage"].lower():
                n = e.get("n_samples")
                if n:
                    return e["seconds"] / n
        return None

    def print_summary(self, title: str = "TIMING SUMMARY") -> None:
        print(self.summary(title))

    # ------------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"entries": self._entries}, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TimingRegistry":
        path = Path(path)
        reg = cls()
        if path.exists():
            try:
                with open(path) as f:
                    reg._entries = json.load(f).get("entries", [])
            except Exception:
                pass
        return reg


# Module-level shared registry (mirrors the solution-cache pattern) so the
# solvers can record without threading an object through every call.
_registry: Optional[TimingRegistry] = None


def get_timing_registry() -> TimingRegistry:
    """Return the shared timing registry, creating it on first use."""
    global _registry
    if _registry is None:
        _registry = TimingRegistry()
    return _registry


def set_timing_registry(registry: Optional[TimingRegistry]) -> None:
    """Replace the shared timing registry (use ``None`` to reset)."""
    global _registry
    _registry = registry
