from __future__ import annotations

import math
import numpy as np

from brwfpt.brw_fpt import GaussianJump, OffspringLaw


def test_gaussian_rate_function_closed_form(jumps: GaussianJump) -> None:
    """I(v) = 0.5 * (v - mu)^T Sigma^{-1} (v - mu)."""
    v = np.array([0.3, -0.1, 0.2])
    dv = v - jumps.mu
    expected = 0.5 * float(dv @ (jumps.Sigma_inv @ dv))
    assert math.isclose(jumps.rate_I(v), expected, rel_tol=1e-12, abs_tol=0.0)


def test_exponential_tilting_mean_shift(jumps: GaussianJump) -> None:
    """Under tilt λ, samples are N(mu + Sigma λ, Sigma). Empirical mean matches shift."""
    lam = np.array([0.7, -0.4, 0.2])
    target_mean = jumps.mu + jumps.Sigma @ lam
    rng = np.random.default_rng(123)
    x = jumps.sample_tilted(lam, size=12000, rng=rng)
    emp = x.mean(axis=0)
    # Loose tolerance due to sampling error
    assert np.allclose(emp, target_mean, atol=1.5e-2)


def test_offspring_supercritical_and_moments(offspring: OffspringLaw) -> None:
    """Basic properties: rho>1, finite second moment, p0+p1>0."""
    assert offspring.rho > 1.0
    assert offspring.second_moment > offspring.rho  # sanity
    assert (offspring.p[0] + offspring.p[1]) > 0.0


def test_extinction_rate_gamma_finite(offspring: OffspringLaw) -> None:
    """γ = -log E[ζ q^{ζ-1}] must be finite positive for this law."""
    gamma = offspring.log_extinction_rate_gamma()
    assert math.isfinite(gamma) and gamma > 0.0
