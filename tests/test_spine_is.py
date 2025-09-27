from __future__ import annotations

import math
import numpy as np

from brwfpt.brw_fpt import BRWSpeed, SpineISConfig, SpineISEstimator


def test_spine_is_estimator_returns_finite(u, jumps, offspring) -> None:
    """
    Spine-IS returns a finite probability estimate with a sensible stderr.

    We use modest x and replication budget to keep the test fast.
    """
    c1 = BRWSpeed.for_gaussian_along_u(offspring, jumps, u).c1
    chat = c1 + 0.6
    x = 40.0

    cfg = SpineISConfig(K_bias=4, L=4, m_power=1/3,
                        max_replications=400,
                        rng=np.random.default_rng(9))
    est = SpineISEstimator(offspring, jumps, chat, u, r=1.0, config=cfg)
    out = est.estimate_cdf(x)

    assert math.isfinite(out["prob_hat"]) and out["prob_hat"] >= 0.0
    assert math.isfinite(out["stderr"]) and out["stderr"] >= 0.0
    assert out["n_eff"] == cfg.max_replications * cfg.K_bias


def test_spine_is_window_pmf_sums_reasonably(u, jumps, offspring) -> None:
    """
    Window PMF masses should sum close to the CDF estimate (same window).

    This is a heuristic diagnostic, not exact equality. We allow a generous tolerance.
    """
    c1 = BRWSpeed.for_gaussian_along_u(offspring, jumps, u).c1
    chat = c1 + 0.5
    x = 40.0

    cfg = SpineISConfig(K_bias=5, L=5, m_power=1/3,
                        max_replications=200,
                        rng=np.random.default_rng(11))
    est = SpineISEstimator(offspring, jumps, chat, u, r=1.0, config=cfg)

    cdf = est.estimate_cdf(x)
    pmf = est.estimate_window_pmf(x)
    pmf_sum = sum(v[0] for v in pmf.values())

    # Same order of magnitude as the CDF estimate.
    assert pmf_sum > 0.0
    ratio = pmf_sum / max(1e-16, cdf["prob_hat"]) if cdf["prob_hat"] > 0 else 1.0
    assert 0.2 <= ratio <= 5.0
