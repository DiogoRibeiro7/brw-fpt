"""
brwfpt: Large Deviations and Spine Importance Sampling for Branching Random Walks

This package implements:
- Gaussian jump distributions with exact rate functions
- Offspring distributions for Galton–Watson branching
- Frontier speed calculation
- Asymptotic probability estimates (lower and upper tail of FPT)
- Branching random walk simulator
- Spine-decomposition importance sampling estimator for rare events

Main entrypoint for end-users: import from `brwfpt` directly.
"""

from .brw_fpt import (
    GaussianJump,
    OffspringLaw,
    BRWSpeed,
    BRWSimulator,
    SpineISConfig,
    SpineISEstimator,
    lower_tail_asymptotics,
    upper_tail_T_value,
    fpt_to_ball,
    local_clt_density_gaussian,
    bernstein_ci,
    plan_replications_for_relative_error,
    benchmark_small_x,
    sanity_compare_asymptotic,
)

__all__ = [
    "GaussianJump",
    "OffspringLaw",
    "BRWSpeed",
    "BRWSimulator",
    "SpineISConfig",
    "SpineISEstimator",
    "lower_tail_asymptotics",
    "upper_tail_T_value",
    "fpt_to_ball",
    "local_clt_density_gaussian",
    "bernstein_ci",
    "plan_replications_for_relative_error",
    "benchmark_small_x",
    "sanity_compare_asymptotic",
]

__version__ = "0.1.0"
