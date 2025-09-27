from __future__ import annotations

import numpy as np

from brwfpt.brw_fpt import BRWSimulator


def test_simulator_runs_and_may_hit(u, jumps, offspring) -> None:
    """Smoke: simulate a few generations and check FPT API contracts."""
    rng = np.random.default_rng(77)
    sim = BRWSimulator(offspring, jumps, rng)

    # Tiny distance and small n to keep test fast and sometimes hit
    x = 6.0
    n_max = 5
    positions, tau = sim.simulate_until(n_max=n_max, x_fpt=x, u_fpt=u, r_fpt=1.0)

    # Contract checks
    assert isinstance(positions, list)
    assert positions[0].shape[1] == jumps.d
    if tau is not None:
        assert isinstance(tau, int)
        assert 0 <= tau <= n_max
