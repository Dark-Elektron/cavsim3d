"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def tolerance():
    """Default numerical tolerance."""
    return 1e-10


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42