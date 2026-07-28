"""Eigenfrequency-spectrum comparison (cluster-matched).

Compare a computed spectrum (e.g. cavsim3d chain eigenfrequencies) against a
reference spectrum (CST, analytical, or another cavsim3d run) in a way that is
robust to near-degenerate clusters:

  1. group each spectrum into frequency clusters (gap-based),
  2. match clusters by mean frequency and CHECK the per-cluster counts,
  3. only where counts agree, compare the individual modes and report the
     (relative) error — so you never compare mismatched modes.

This is the reusable core behind the eigenmode comparison used in the demo
scripts; it takes plain arrays so it works for any reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def cluster_frequencies(freqs, tol: float) -> List[List[float]]:
    """Group a frequency list into clusters separated by gaps greater than *tol*.

    Parameters
    ----------
    freqs : array-like
        Frequencies (any unit; *tol* must be in the same unit).
    tol : float
        Two consecutive frequencies belong to the same cluster if their gap is
        <= tol.

    Returns
    -------
    list[list[float]]
        Clusters (sorted), each a list of the frequencies it contains.
    """
    fs = np.sort(np.asarray(freqs, dtype=float))
    clusters: List[List[float]] = []
    cur: List[float] = []
    for f in fs:
        if cur and (f - cur[-1]) > tol:
            clusters.append(cur)
            cur = []
        cur.append(float(f))
    if cur:
        clusters.append(cur)
    return clusters


@dataclass
class SpectrumComparison:
    """Result of :func:`compare_spectra`.

    Attributes
    ----------
    clusters_a, clusters_b : list[list[float]]
        Clusters of the computed (a) and reference (b) spectra.
    cluster_rows : list[dict]
        Per computed-cluster: mean_a, n_a, mean_b, n_b, counts_ok, dmean.
    matched : np.ndarray, shape (m, 2)
        (computed, reference) frequency pairs from count-equal clusters only.
    n_matched : int
    mean_abs, max_abs : float
        Absolute error stats over matched modes.
    mean_rel, max_rel : float
        Relative error stats over matched modes.
    band_stats : dict[str, dict]
        Optional per-band {'n','mean_rel','max_rel','mean_abs','max_abs','range'}.
    """
    clusters_a: List[List[float]] = field(default_factory=list)
    clusters_b: List[List[float]] = field(default_factory=list)
    cluster_rows: List[Dict] = field(default_factory=list)
    matched: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    n_matched: int = 0
    mean_abs: float = float("nan")
    max_abs: float = float("nan")
    mean_rel: float = float("nan")
    max_rel: float = float("nan")
    band_stats: Dict[str, Dict] = field(default_factory=dict)

    # -- convenience --------------------------------------------------------
    @property
    def abs_err(self) -> np.ndarray:
        if self.matched.size == 0:
            return np.empty(0)
        return np.abs(self.matched[:, 0] - self.matched[:, 1])

    @property
    def rel_err(self) -> np.ndarray:
        if self.matched.size == 0:
            return np.empty(0)
        return self.abs_err / np.abs(self.matched[:, 1])

    def matched_in_band(self, lo: float, hi: float) -> np.ndarray:
        """Matched pairs whose computed frequency is in [lo, hi]."""
        if self.matched.size == 0:
            return np.empty((0, 2))
        sel = (self.matched[:, 0] >= lo) & (self.matched[:, 0] <= hi)
        return self.matched[sel]


def compare_spectra(
    computed,
    reference,
    tol: float,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> SpectrumComparison:
    """Cluster-matched comparison of a computed vs a reference spectrum.

    Parameters
    ----------
    computed, reference : array-like
        Frequencies to compare (same unit; ``tol`` in the same unit).
    tol : float
        Cluster gap tolerance.
    fmin, fmax : float, optional
        Restrict both spectra to this window before comparing.
    bands : dict[name -> (lo, hi)], optional
        If given, per-band error stats over the matched modes are computed
        (used e.g. for "fundamental" and "first dipole" passbands).

    Returns
    -------
    SpectrumComparison
    """
    a = np.sort(np.asarray(computed, dtype=float))
    b = np.sort(np.asarray(reference, dtype=float))
    if fmin is not None:
        a = a[a >= fmin]; b = b[b >= fmin]
    if fmax is not None:
        a = a[a <= fmax]; b = b[b <= fmax]

    ca = cluster_frequencies(a, tol)
    cb = cluster_frequencies(b, tol)
    res = SpectrumComparison(clusters_a=ca, clusters_b=cb)
    if not ca or not cb:
        return res

    b_means = np.array([np.mean(c) for c in cb])
    pairs: List[Tuple[float, float]] = []
    for cc in ca:
        mean_a, n_a = float(np.mean(cc)), len(cc)
        j = int(np.argmin(np.abs(b_means - mean_a)))
        cj, mean_b, n_b = cb[j], float(b_means[j]), len(cb[j])
        counts_ok = (n_a == n_b)
        res.cluster_rows.append(dict(
            mean_a=mean_a, n_a=n_a, mean_b=mean_b, n_b=n_b,
            counts_ok=counts_ok, dmean=abs(mean_a - mean_b)))
        if counts_ok:                       # compare like-with-like only
            for x, y in zip(sorted(cc), sorted(cj)):
                pairs.append((x, y))

    if pairs:
        mp = np.array(pairs)
        res.matched = mp
        res.n_matched = len(mp)
        ae = np.abs(mp[:, 0] - mp[:, 1])
        re = ae / np.abs(mp[:, 1])
        res.mean_abs, res.max_abs = float(ae.mean()), float(ae.max())
        res.mean_rel, res.max_rel = float(re.mean()), float(re.max())
        if bands:
            for name, (lo, hi) in bands.items():
                sel = (mp[:, 0] >= lo) & (mp[:, 0] <= hi)
                if not np.any(sel):
                    continue
                aeb = ae[sel]; reb = re[sel]
                res.band_stats[name] = dict(
                    n=int(sel.sum()), range=(lo, hi),
                    mean_abs=float(aeb.mean()), max_abs=float(aeb.max()),
                    mean_rel=float(reb.mean()), max_rel=float(reb.max()))
    return res
