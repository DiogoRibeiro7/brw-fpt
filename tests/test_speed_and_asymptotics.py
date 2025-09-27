from __future__ import annotations

import math
import numpy as np
import pytest

from brwfpt.brw_fpt import BRWSpeed, lower_tail_asymptotics, upper_tail_T_value


def test_frontier_speed_solves_rate(u, jumps, offspring) -> None:
    """c1 along u solves I(c1 u) = log rho (within tolerance)."""
    sp = BRWSpeed.for_gaussian_along_u(offspring, jumps, u)
    c1 = sp.c1
    lhs = jumps.rate_I(c1 * u)
    rhs = math.log(offspring.rho)
    assert math.isclose(lhs, rhs, rel_tol=1e-12)


def test_lower_tail_asymptotics_monotone_in_x(u, jumps, offspring) -> None:
    """Asymptotic probability decays as distance x grows when ĉ>c1."""
    c1 = BRWSpeed.for_gaussian_along_u(offspring, jumps, u).c1
    chat = c1 + 0.5
    p1 = lower_tail_asymptotics(40.0, chat, offspring, jumps, u)
    p2 = lower_tail_asymptotics(80.0, chat, offspring, jumps, u)
    assert p2 < p1


@pytest.mark.parametrize("delta", [0.2, 0.6])
def test_upper_tail_T_value_positive(u, jumps, offspring, delta: float) -> None:
    """Upper-tail T(ĉ) > 0 for ĉ<c1 (basic sanity)."""
    c1 = BRWSpeed.for_gaussian_along_u(offspring, jumps, u).c1
    chat1 = c1 - delta
    T1 = upper_tail_T_value(chat1, offspring, jumps, u, grid_points=160)
    assert T1 > 0.0 and math.isfinite(T1)
