"""Shared fixtures for BRW-FPT tests.


These fixtures create a small, well-behaved Gaussian-jump BRW in R^3 and a
supercritical offspring law with finite second moment. We keep seeds fixed for
reproducibility. All code is designed to be NumPy-only.
"""

from __future__ import annotations


from typing import Iterator
import numpy as np
import pytest


from brwfpt.brw_fpt import GaussianJump, OffspringLaw


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Global RNG for deterministic tests."""
    return np.random.default_rng(20250927)


@pytest.fixture(scope="session")
def jumps() -> GaussianJump:
    """3D centered Gaussian with identity covariance (non-lattice, PD)."""
    return GaussianJump.from_params(mu=[0.0, 0.0, 0.0], Sigma=np.eye(3))


@pytest.fixture(scope="session")
def offspring() -> OffspringLaw:
    """Supercritical GW with p0=0.2, p1=0.3, p2=0.5 (rho = 1.3)."""
    return OffspringLaw.from_pmf([0.2, 0.3, 0.5])


@pytest.fixture(scope="session")
def u() -> np.ndarray:
    """Unit direction e1 in R^3."""
    v = np.array([1.0, 0.0, 0.0], dtype=float)
    v /= np.linalg.norm(v)
    return v
