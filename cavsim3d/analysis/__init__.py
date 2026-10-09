"""Post-processing / comparison utilities for cavsim3d results."""

from cavsim3d.analysis.beam import bunch_train_spectrum
from cavsim3d.analysis.eigenmodes import (
    cluster_frequencies,
    compare_spectra,
    SpectrumComparison,
)
from cavsim3d.analysis.network import (
    band_difference,
    keep_port_modes,
    network_matrix,
    plot_entries,
    plot_matrix,
    port_mode_labels,
)

__all__ = [
    "bunch_train_spectrum",
    "cluster_frequencies", "compare_spectra", "SpectrumComparison",
    "band_difference", "keep_port_modes", "network_matrix",
    "plot_entries", "plot_matrix", "port_mode_labels",
]
