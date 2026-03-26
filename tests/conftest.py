"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI/tests


@pytest.fixture(scope="session")
def tolerance():
    """Default numerical tolerance."""
    return 1e-10


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42
