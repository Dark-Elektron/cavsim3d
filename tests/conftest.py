"""Pytest configuration and shared fixtures."""

import logging

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/tests


@pytest.fixture(autouse=True)
def _cavsim3d_logs_to_caplog(request):
    """Hand cavsim3d's log records to ``caplog`` in the tests that use it.

    The ``cavsim3d`` logger does not propagate to the root logger (where
    ``caplog`` listens), so its capture handler is attached directly.
    """
    if "caplog" not in request.fixturenames:
        yield
        return
    caplog = request.getfixturevalue("caplog")
    logger = logging.getLogger("cavsim3d")
    logger.addHandler(caplog.handler)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


@pytest.fixture(scope="session")
def tolerance():
    """Default numerical tolerance."""
    return 1e-10


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
