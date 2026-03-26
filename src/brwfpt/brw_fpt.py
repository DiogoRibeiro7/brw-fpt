# brw_fpt_ld_extended.py
# =============================================================================
# Large Deviations of First Passage Times of Branching Random Walks in R^d
# Asymptotics and Algorithms
#
# Additions over the base module:
#   1) General FPT geometry: ball B_r(x * u) with unit direction u ∈ R^d and radius r > 0
#   2) Bernstein confidence intervals + replication planning for target relative error
#   3) Windowed exact-time PMF estimation (per-t masses) for τ_x = n0 - t
#   4) Local-CLT density helpers (exact for Gaussian; shell for non-Gaussian extension)
#   5) Antithetic option for Gaussian tilting to reduce variance
#   6) Micro-bench utilities: small-x comparisons and sanity checks vs asymptotics
#
# Dependencies: numpy (required), typing, dataclasses, math, time
# =============================================================================
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

ArrayLike = np.ndarray | list[float] | tuple[float, ...]

# -----------------------------------------------------------------------------#
# Basic utilities
# -----------------------------------------------------------------------------#

def _as_1d(v: ArrayLike) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != 1:
        raise ValueError("Expected a 1D array-like.")
    return a

def _normal_quantile(p: float) -> float:
    """Inverse standard-normal CDF via Acklam's rational approximation.

    Accurate to ~1.15e-9 for 0 < p < 1.  Used instead of scipy so the
    package stays NumPy-only.
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def _ensure_positive(x: float, name: str) -> None:
    if not (x > 0):
        raise ValueError(f"{name} must be > 0.")

def _unit_e1(d: int) -> np.ndarray:
    e = np.zeros(d, dtype=np.float64)
    e[0] = 1.0
    return e

def _ensure_unit_vector(u: ArrayLike, tol: float = 1e-9) -> np.ndarray:
    u = _as_1d(u)
    nrm = float(np.linalg.norm(u))
    if nrm <= 0:
        raise ValueError("Direction vector u must be non-zero.")
    u = u / nrm
    if abs(np.linalg.norm(u) - 1.0) > tol:
        # re-normalize defensively, and warn via exception to be explicit
        raise ValueError("u must be unit-norm. Provide a unit vector.")
    return u

# -----------------------------------------------------------------------------#
# Gaussian jumps (unchanged core, with exact tilting)
# -----------------------------------------------------------------------------#

@dataclass(frozen=True)
class GaussianJump:
    mu: np.ndarray           # (d,)
    Sigma: np.ndarray        # (d, d) PD
    Sigma_inv: np.ndarray    # (d, d)
    d: int

    @staticmethod
    def from_params(mu: ArrayLike, Sigma: ArrayLike) -> "GaussianJump":
        mu = _as_1d(mu)
        Sigma = np.asarray(Sigma, dtype=np.float64)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1] or Sigma.shape[0] != mu.shape[0]:
            raise ValueError("Sigma must be square dxd and match mu dimension.")
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError as e:
            raise ValueError("Sigma must be symmetric positive definite.") from e
        Sigma_inv = np.linalg.inv(Sigma)
        return GaussianJump(mu=mu, Sigma=Sigma, Sigma_inv=Sigma_inv, d=mu.shape[0])

    def log_mgf(self, lam: ArrayLike) -> float:
        lam = _as_1d(lam)
        return float(lam @ self.mu + 0.5 * lam @ (self.Sigma @ lam))

    def grad_log_mgf(self, lam: ArrayLike) -> np.ndarray:
        lam = _as_1d(lam)
        return self.mu + self.Sigma @ lam

    def rate_I(self, v: ArrayLike) -> float:
        v = _as_1d(v)
        dv = v - self.mu
        return 0.5 * float(dv @ (self.Sigma_inv @ dv))

    def argmax_lambda_for_v(self, v: ArrayLike) -> np.ndarray:
        v = _as_1d(v)
        return self.Sigma_inv @ (v - self.mu)

    def sample_nominal(self, size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.multivariate_normal(self.mu, self.Sigma, size=size)

    def sample_tilted(self, lam: ArrayLike, size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        lam = _as_1d(lam)
        rng = rng or np.random.default_rng()
        mean_tilt = self.mu + self.Sigma @ lam
        return rng.multivariate_normal(mean_tilt, self.Sigma, size=size)

# -----------------------------------------------------------------------------#
# Offspring law (Galton–Watson)
# -----------------------------------------------------------------------------#

@dataclass(frozen=True)
class OffspringLaw:
    p: np.ndarray
    support: np.ndarray
    rho: float
    second_moment: float

    @staticmethod
    def from_pmf(p: ArrayLike) -> "OffspringLaw":
        p = np.asarray(p, dtype=np.float64)
        if p.ndim != 1:
            raise ValueError("PMF must be 1D.")
        if not np.isclose(p.sum(), 1.0):
            raise ValueError("PMF must sum to 1.")
        supp = np.arange(p.shape[0])
        rho = float(np.dot(supp, p))
        if rho <= 1.0:
            raise ValueError("Require supercritical offspring: E[ζ]=rho>1.")
        k2 = float(np.dot(supp**2, p))
        if not (p[0] + p[1] > 0):
            raise ValueError("Need p0 + p1 > 0.")
        return OffspringLaw(p=p, support=supp, rho=rho, second_moment=k2)

    def sample(self, size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.choice(self.support, size=size, p=self.p)

    def log_extinction_rate_gamma(self, max_iter: int = 10000, tol: float = 1e-12) -> float:
        """γ = -log E[ ζ q^{ζ-1} ], q is the extinction prob (fixed-point of pgf)."""
        def f(q: float) -> float:
            return float(np.dot(self.p, (q ** self.support)))
        q = 1.0
        converged = False
        for _ in range(max_iter):
            q_new = f(q)
            if abs(q_new - q) < tol:
                converged = True
                break
            q = q_new
        if not converged:
            warnings.warn(
                f"Extinction probability fixed-point did not converge after {max_iter} iterations.",
                stacklevel=2,
            )
        q = float(np.clip(q, 0.0, 1.0))
        vals = self.support * self.p * (q ** np.maximum(self.support - 1, 0))
        m = float(np.sum(vals))
        if m <= 0:
            return float("inf")
        return -math.log(m)

# -----------------------------------------------------------------------------#
# Typical frontier speed c1 along any unit direction u
# For Gaussian jumps, along u, I(c1 u) = log rho -> scalar quadratic in c1.
# -----------------------------------------------------------------------------#

@dataclass(frozen=True)
class BRWSpeed:
    c1: float
    lam_star_for_c1_u: np.ndarray  # λ such that mean under tilt has component along u equal to c1

    @staticmethod
    def for_gaussian_along_u(offspring: OffspringLaw, jumps: GaussianJump, u: ArrayLike) -> "BRWSpeed":
        u = _ensure_unit_vector(u)
        # Along u: v = c1 u. I(v) = 0.5 (v - mu)^T Σ^{-1} (v - mu) = log rho
        # Solve for c1 by minimizing distance in Sigma^{-1}-metric along u.
        # Equivalent scalar form using projection:
        # Let a = u^T Σ^{-1} u, b = u^T Σ^{-1} mu. Then I(c u) = 0.5 (c^2 a - 2 c b + mu^T Σ^{-1} mu).
        # Solve 0.5 (c^2 a - 2 c b + c0) = log rho for the larger root.
        a = float(u @ (jumps.Sigma_inv @ u))
        b = float(u @ (jumps.Sigma_inv @ jumps.mu))
        c0 = float(jumps.mu @ (jumps.Sigma_inv @ jumps.mu))
        target = math.log(offspring.rho)
        # Quadratic: (a/2) c^2 - b c + (c0/2 - log rho) = 0
        qa = 0.5 * a
        qb = -b
        qc = 0.5 * c0 - target
        disc = qb*qb - 4*qa*qc
        tol = 1e-10 * max(abs(qb*qb), abs(4*qa*qc), 1e-30)
        if disc < -tol:
            raise RuntimeError(
                f"Negative discriminant ({disc:.2e}) while solving for c1; "
                "log(rho) may be too large for the given Sigma and direction u."
            )
        disc = max(disc, 0.0)
        root1 = (-qb + math.sqrt(disc)) / (2 * qa)
        root2 = (-qb - math.sqrt(disc)) / (2 * qa)
        c1 = max(root1, root2)
        # λ* s.t. grad log mgf(λ) = μ + Σ λ has projection along u equal to c1
        denom = float(u @ (jumps.Sigma @ u))
        if denom <= 0:
            raise ValueError(f"Invalid direction u: u^T Sigma u = {denom} <= 0.")
        alpha = (c1 - float(jumps.mu @ u)) / denom
        lam = alpha * u
        return BRWSpeed(c1=c1, lam_star_for_c1_u=lam)

# -----------------------------------------------------------------------------#
# FPT to a general ball B_r(x * u)
# -----------------------------------------------------------------------------#

def fpt_to_ball(positions_by_gen: list[np.ndarray], x: float, u: ArrayLike, r: float) -> int | None:
    """
    First passage time τ_x to the ball B_r(center), center = x * u (u is unit vector).

    Returns n if any particle in generation n is inside, else None (if not hit in provided gens).
    """
    _ensure_positive(x, "x")
    _ensure_positive(r, "r")
    u = _ensure_unit_vector(u)
    center = u * x
    r2 = r * r
    for n, pos in enumerate(positions_by_gen):
        dif = pos - center
        if np.any(np.einsum("ij,ij->i", dif, dif) <= r2):
            return n
    return None

# -----------------------------------------------------------------------------#
# Plain BRW simulator (nominal)
# -----------------------------------------------------------------------------#

@dataclass
class BRWSimulator:
    offspring: OffspringLaw
    jumps: GaussianJump
    rng: np.random.Generator

    def simulate_until(
        self,
        n_max: int,
        x_fpt: float | None = None,
        u_fpt: ArrayLike | None = None,
        r_fpt: float = 1.0,
        max_population: int = 1_000_000,
    ) -> tuple[list[np.ndarray], int | None]:
        """
        Simulate up to generation n_max, optionally stopping when hitting B_r(x*u).

        Args:
            n_max: max generation (inclusive)
            x_fpt, u_fpt, r_fpt: optional FPT target (if provided)
            max_population: stop early if total children in a generation exceeds this

        Returns:
            (positions_by_gen, tau_x) where tau_x is None if no hit before/at n_max.
        """
        if n_max < 0:
            raise ValueError("n_max must be >= 0.")
        d = self.jumps.d
        positions_by_gen: list[np.ndarray] = [np.zeros((1, d), dtype=np.float64)]

        tau_x = None
        if x_fpt is not None and u_fpt is not None:
            tau_x = fpt_to_ball(positions_by_gen, x_fpt, _ensure_unit_vector(u_fpt), r_fpt)
            if tau_x is not None:
                return positions_by_gen, tau_x

        for n in range(n_max):
            parents = positions_by_gen[-1]
            K = self.offspring.sample(size=parents.shape[0], rng=self.rng)
            total_children = int(np.sum(K))
            if total_children == 0:
                positions_by_gen.append(np.zeros((0, d), dtype=np.float64))
                break
            if total_children > max_population:
                warnings.warn(
                    f"Population ({total_children}) exceeded max_population "
                    f"({max_population}) at generation {n + 1}; stopping early.",
                    stacklevel=2,
                )
                break
            idxs = np.repeat(np.arange(parents.shape[0]), K)
            pre = parents[idxs]
            jumps = self.jumps.sample_nominal(size=total_children, rng=self.rng)
            next_pos = pre + jumps
            positions_by_gen.append(next_pos)

            if x_fpt is not None and u_fpt is not None:
                tau_x = fpt_to_ball([next_pos], x_fpt, u_fpt, r_fpt)
                if tau_x is not None:
                    tau_x = n + 1
                    return positions_by_gen, tau_x

        return positions_by_gen, tau_x

# -----------------------------------------------------------------------------#
# Lower-tail asymptotics (Theorem 1), generalized to direction u and radius r
# For radius r ≠ 1, leading exponential rate is unchanged; polynomial prefactor
# scaling is direction-invariant (keep x^{-d/2}). We expose r for completeness.
# -----------------------------------------------------------------------------#

def lower_tail_asymptotics(
    x: float,
    chat: float,
    offspring: OffspringLaw,
    jumps: GaussianJump,
    u: ArrayLike,
    r: float = 1.0,
) -> float:
    """
    Asymptotic P(τ_x < x/chat | S) ~ x^{-d/2} exp( - (x/chat) * ( I(chat * u) - log rho ) )

    Args:
        x: center distance along direction u
        chat: ĉ1 > c1 (fast frontier)
        offspring, jumps: model params
        u: unit direction for the target center
        r: ball radius (does not change the leading exponential term)

    Returns:
        Asymptotic estimate as float (for large x).
    """
    _ensure_positive(x, "x")
    _ensure_positive(chat, "chat")
    _ensure_positive(r, "r")
    u = _ensure_unit_vector(u)
    d = jumps.d
    I_val = jumps.rate_I(chat * u)
    rate = (x / chat) * (I_val - math.log(offspring.rho))
    return (x ** (-d / 2.0)) * math.exp(-rate)

# -----------------------------------------------------------------------------#
# Upper-tail optimization T-value (Theorem 2), along u
# We retain a scalar search using boundary feasibility along u for Gaussian.
# -----------------------------------------------------------------------------#

def upper_tail_T_value(
    chat: float,
    offspring: OffspringLaw,
    jumps: GaussianJump,
    u: ArrayLike,
    grid_alpha: Iterable[float] = (),
    grid_points: int = 200,
) -> float:
    """
    Compute T = inf_{α ∈ (0, 1/chat]} ( γ α + α I( y/α ) ), subject to
        I( (u - y) / (1/chat - α) ) >= log rho,
    restricting y parallel to u (tight for spherical/symmetric Gaussian).

    Returns:
        T value (float), so that - (1/x) log P(τ_x >= x/chat | S) → T.
    """
    u = _ensure_unit_vector(u)
    c1_obj = BRWSpeed.for_gaussian_along_u(offspring, jumps, u).c1
    if not (chat < c1_obj):
        raise ValueError("Upper-tail regime requires chat < c1.")
    rho = offspring.rho
    gamma = offspring.log_extinction_rate_gamma()

    if not grid_alpha:
        ub = 1.0 / chat
        alpha_grid = np.linspace(1e-4 * ub, 0.999 * ub, grid_points)
    else:
        alpha_grid = np.array(list(grid_alpha), dtype=np.float64)

    # Boundary I(z u) = log rho -> z solutions along u:
    # For Gaussian: 0.5 (z u - mu)^T Σ^{-1} (z u - mu) = log rho.
    # Let a = u^T Σ^{-1} u, b = u^T Σ^{-1} mu, c0 = mu^T Σ^{-1} mu
    a = float(u @ (jumps.Sigma_inv @ u))
    b = float(u @ (jumps.Sigma_inv @ jumps.mu))
    c0 = float(jumps.mu @ (jumps.Sigma_inv @ jumps.mu))
    target = math.log(rho)
    qa = 0.5 * a
    qb = -b
    qc = 0.5 * c0 - target
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        raise RuntimeError("Feasibility boundary not found; check inputs.")
    z_roots = [(-qb + math.sqrt(disc)) / (2 * qa), (-qb - math.sqrt(disc)) / (2 * qa)]

    best = float("inf")
    for alpha in alpha_grid:
        denom = (1.0 / chat) - alpha
        if denom <= 0:
            continue
        for z in z_roots:
            y1 = 1.0 - z * denom  # scalar along u
            y_vec = y1 * u
            I_y_over_alpha = jumps.rate_I(y_vec / alpha)
            obj = gamma * alpha + alpha * I_y_over_alpha
            if obj < best:
                best = obj
    return best

# -----------------------------------------------------------------------------#
# Local-CLT density helpers (exact for Gaussian)
# -----------------------------------------------------------------------------#

def local_clt_density_gaussian(n: int, x_vec: ArrayLike, jumps: GaussianJump) -> float:
    """
    Exact density of S_n for Gaussian jumps: S_n ~ N(n μ, n Σ).

    Args:
        n: number of steps
        x_vec: evaluation point in R^d
        jumps: GaussianJump

    Returns:
        PDF value at x_vec.
    """
    if n <= 0:
        raise ValueError("n must be >= 1 for sum of jumps.")
    x_vec = _as_1d(x_vec)
    d = jumps.d
    mean = n * jumps.mu
    cov = n * jumps.Sigma
    cov_inv = jumps.Sigma_inv / n
    sign, logdet_cov = np.linalg.slogdet(cov)
    if sign <= 0:
        raise RuntimeError("Covariance determinant non-positive.")
    diff = x_vec - mean
    quad = float(diff @ (cov_inv @ diff))
    log_norm = -0.5 * (d * math.log(2 * math.pi) + float(logdet_cov))
    return math.exp(log_norm - 0.5 * quad)

# Placeholder for non-Gaussian extension (plug-in variance-cov, Edgeworth, etc.)
def local_clt_density_shell(*args, **kwargs) -> float:
    """
    Shell function for a future non-Gaussian local-CLT approximation. Not implemented here.
    """
    raise NotImplementedError("Non-Gaussian local CLT not implemented in this module.")

# -----------------------------------------------------------------------------#
# Spine-decomposition IS (lower-tail) with antithetic option
# -----------------------------------------------------------------------------#

@dataclass
class SpineISConfig:
    K_bias: int = 8           # number of exact-time masses to aggregate
    L: int = 8                # last-L strict control steps
    m_power: float = 1.0/3.0  # window size m = n^{m_power}
    max_replications: int = 2000
    antithetic: bool = False  # use antithetic pairs for tilted jumps along the spine
    rng: np.random.Generator | None = None

@dataclass
class SpineISEstimator:
    offspring: OffspringLaw
    jumps: GaussianJump
    chat: float          # ĉ1 > c1
    u: np.ndarray        # unit vector
    r: float = 1.0       # radius of the target ball
    config: SpineISConfig = SpineISConfig()

    def __post_init__(self):
        _ensure_positive(self.chat, "chat")
        _ensure_positive(self.r, "r")
        self.u = _ensure_unit_vector(self.u)
        self._speed = BRWSpeed.for_gaussian_along_u(self.offspring, self.jumps, self.u)
        if not (self.chat > self._speed.c1):
            raise ValueError("Lower-tail IS requires chat > c1.")
        self.lam_hat = self._speed.lam_star_for_c1_u
        self.rng = self.config.rng or np.random.default_rng()

        # Precompute Cholesky factor for tilted sampling (avoids recomputing per step)
        self._cholesky_L = np.linalg.cholesky(self.jumps.Sigma)

        # Build size-biased offspring CDF once
        sb = self.offspring.support * self.offspring.p
        tot = sb.sum()
        if tot <= 0:
            raise RuntimeError("Invalid offspring for size-biasing.")
        self._sb_cum = np.cumsum(sb / tot)

    # ---------- Public API ---------- #

    def estimate_cdf(self, x: float) -> dict[str, float]:
        """
        Estimate P( τ_x <= x/chat ) for B_r(x u) via windowed exact-time masses and IS.
        """
        _ensure_positive(x, "x")
        n0 = int(math.floor(x / self.chat))
        estimates: list[float] = []
        for t in range(self.config.K_bias):
            n_target = max(n0 - t, 0)
            for _ in range(self.config.max_replications):
                if self.config.antithetic:
                    z1 = self._single_replication_Z_eq_time(x, n_target, negate_z=False)
                    z2 = self._single_replication_Z_eq_time(x, n_target, negate_z=True)
                    estimates.append(0.5 * (z1 + z2))
                else:
                    estimates.append(self._single_replication_Z_eq_time(x, n_target))
        arr = np.asarray(estimates, dtype=np.float64)
        prob_hat = float(np.mean(arr))
        stderr = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")
        return {"prob_hat": prob_hat, "stderr": stderr, "n_eff": int(arr.size)}

    def estimate_window_pmf(self, x: float) -> dict[int, tuple[float, float, int]]:
        """
        Estimate each P(τ_x = n0 - t) separately across t=0..K_bias-1.

        Returns:
            {t: (p_hat_t, stderr_t, N_t)} mapping for diagnostics.
        """
        _ensure_positive(x, "x")
        n0 = int(math.floor(x / self.chat))
        out: dict[int, tuple[float, float, int]] = {}
        for t in range(self.config.K_bias):
            n_target = max(n0 - t, 0)
            if self.config.antithetic:
                vals = [
                    0.5 * (self._single_replication_Z_eq_time(x, n_target, negate_z=False)
                           + self._single_replication_Z_eq_time(x, n_target, negate_z=True))
                    for _ in range(self.config.max_replications)
                ]
            else:
                vals = [self._single_replication_Z_eq_time(x, n_target) for _ in range(self.config.max_replications)]
            arr = np.asarray(vals, dtype=np.float64)
            p_hat = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")
            out[t] = (p_hat, se, int(arr.size))
        return out

    # ---------- Internals ---------- #

    def _size_biased_offspring(self) -> int:
        p = self.rng.random()
        k = int(np.searchsorted(self._sb_cum, p, side="right"))
        return max(1, k)

    def _spine_child_index(self, child_jumps_tilted: np.ndarray) -> int:
        # weights proportional to exp(λ̂·ξ)
        w = np.exp(child_jumps_tilted @ self.lam_hat)
        w_cum = np.cumsum(w / np.sum(w))
        p = self.rng.random()
        return int(np.searchsorted(w_cum, p, side="right"))

    def _is_hit(self, positions: np.ndarray, x: float) -> bool:
        center = self.u * x
        dif = positions - center
        return bool(np.any(np.einsum("ij,ij->i", dif, dif) <= self.r * self.r))

    def _single_replication_Z_eq_time(self, x: float, n: int, negate_z: bool = False) -> float:
        """
        One IS replication for exact-time event {τ_x = n} under the Q measure.

        Args:
            negate_z: If True, negate the standard-normal innovations (antithetic pair).

        Returns:
            Non-negative estimator Z with E_Q[Z] = P(τ_x = n).
        """
        d = self.jumps.d
        rho = self.offspring.rho
        if n == 0:
            # hitting at time 0 is possible only if origin inside B_r(x u) (rare / off by design for x>r)
            return 1.0 if x <= self.r else 0.0

        # Radon–Nikodym: W_n = sum_{u ∈ V_n} exp(λ̂·η_u - n ψ(λ̂)), ψ(λ̂)=log ρ + log mgf(λ̂)
        psi = math.log(rho) + self.jumps.log_mgf(self.lam_hat)

        # Spine positions
        S_prev = np.zeros(d, dtype=np.float64)  # S_0
        S_k = S_prev.copy()

        L = self.config.L
        m = max(1, int(round(n ** self.config.m_power)))

        hit_before = False

        # Forward simulate spine generations 1..n
        for k in range(1, n + 1):
            # Size-biased offspring at spine
            spine_children = self._size_biased_offspring()

            # Tilted jumps for children
            if not negate_z:
                child_jumps = self.jumps.sample_tilted(self.lam_hat, size=spine_children, rng=self.rng)
            else:
                # Antithetic: negate the standard-normal innovations
                mu_tilt = self.jumps.mu + self.jumps.Sigma @ self.lam_hat
                z = self.rng.standard_normal((spine_children, d))
                z = (-z) @ self._cholesky_L.T
                child_jumps = mu_tilt + z

            spine_idx = self._spine_child_index(child_jumps)
            spine_jump = child_jumps[spine_idx]
            S_prev = S_k
            S_k = S_prev + spine_jump

            # Prevent early hits via shallow exploration of siblings in last L steps
            if k < n and k >= n - max(L, m):
                off_idx = [i for i in range(spine_children) if i != spine_idx]
                if off_idx:
                    seeds = (S_prev + child_jumps[off_idx])  # absolute positions of siblings at gen k
                    # grow up to depth dmax generations nominally; check hits
                    dmax = min(L, n - k)
                    frontier = seeds
                    for _depth in range(dmax):
                        if frontier.shape[0] == 0:
                            break
                        K = self.offspring.sample(size=frontier.shape[0], rng=self.rng)
                        total_children = int(np.sum(K))
                        if total_children == 0:
                            frontier = np.zeros((0, d), dtype=np.float64)
                            break
                        idxs = np.repeat(np.arange(frontier.shape[0]), K)
                        pre = frontier[idxs]
                        jumps_nom = self.jumps.sample_nominal(size=total_children, rng=self.rng)
                        frontier = pre + jumps_nom
                        if self._is_hit(frontier, x):
                            hit_before = True
                            break
                    if hit_before:
                        break

        if hit_before:
            return 0.0

        # Enforce final hit at time n by spine (dominant in lower-tail)
        if not self._is_hit(S_k.reshape(1, -1), x):
            return 0.0

        # Approximate W_n by spine term + small sibling correction at time n
        exp_lam_eta_spine_n = math.exp(self.lam_hat @ S_k)
        Wn_approx = exp_lam_eta_spine_n * math.exp(-n * psi)

        pilot = 4
        sib_jumps = self.jumps.sample_nominal(size=pilot, rng=self.rng)
        sib_pos = S_prev + sib_jumps
        sib_exp = float(np.sum(np.exp(sib_pos @ self.lam_hat)))
        Wn_approx += (self.offspring.rho / pilot) * sib_exp * math.exp(-n * psi)

        if Wn_approx <= 0.0 or not np.isfinite(Wn_approx):
            return 0.0
        LR = 1.0 / Wn_approx
        return LR

# -----------------------------------------------------------------------------#
# Bernstein confidence intervals and replication planning
# -----------------------------------------------------------------------------#

def bernstein_ci(samples: ArrayLike, alpha: float = 0.05, value_range: tuple[float, float] | None = None) -> tuple[float, float]:
    """
    Bernstein-style CI for bounded variables (or IS estimates with known bounds).
    If value_range is None, falls back to normal CI using sample variance.

    Args:
        samples: 1D array of i.i.d. observations
        alpha: 1 - confidence level
        value_range: (a, b) known almost-sure bounds, optional

    Returns:
        (lo, hi) confidence interval for the mean.
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n < 2:
        m = float(np.mean(x)) if n == 1 else 0.0
        return (m, m)

    m = float(np.mean(x))
    v = float(np.var(x, ddof=1))
    z = math.log(2.0 / max(1e-16, alpha))
    if value_range is None:
        # Two-sided z-quantile: Phi^{-1}(1 - alpha/2)
        z_val = _normal_quantile(1.0 - alpha / 2.0)
        half = z_val * math.sqrt(v / n)
        return (m - half, m + half)
    a, b = value_range
    rng = max(1e-16, b - a)
    # Bernstein bound: |mean - m| <= sqrt(2 v log(2/α) / n) + (3 rng log(2/α)) / (n)
    half = math.sqrt(2 * v * z / n) + (3 * rng * z) / n
    return (m - half, m + half)

def plan_replications_for_relative_error(
    p_hat: float,
    target_rel_err: float = 0.2,
    alpha: float = 0.05,
    variance_proxy: float | None = None,
) -> int:
    """
    Suggest N so that stderr / p_hat <= target_rel_err at (approx.) 1-α confidence.

    We use normal approximation: stderr ≈ sqrt(Var / N).
    If variance_proxy is None, fall back to conservative Var ≈ p_hat (for [0,1] variables),
    else use provided variance estimate.

    Returns:
        N (int), at least 1.
    """
    if p_hat <= 0:
        return 1
    if variance_proxy is None:
        variance_proxy = max(1e-16, p_hat * (1 - min(1.0, p_hat)))  # crude
    z = _normal_quantile(1.0 - alpha / 2.0)
    target_stderr = target_rel_err * p_hat / z
    N = int(math.ceil(variance_proxy / max(1e-18, target_stderr ** 2)))
    return max(1, N)

# -----------------------------------------------------------------------------#
# Micro-bench utilities
# -----------------------------------------------------------------------------#

def benchmark_small_x(
    offspring: OffspringLaw,
    jumps: GaussianJump,
    x: float,
    chat: float,
    u: ArrayLike,
    r: float = 1.0,
    mc_trials: int = 300,
    is_trials: int = 1500,
    seed: int = 7,
) -> dict[str, dict[str, float]]:
    """
    Compare Spine-IS vs crude MC for small x:
      MC estimates P(τ_x <= floor(x/chat)) brute-force (dangerous for large x).
      Spine-IS uses windowed exact-time masses.

    Returns:
        {"mc": {...}, "is": {...}} with point estimates, stderr, time.
    """
    u = _ensure_unit_vector(u)
    rng = np.random.default_rng(seed)
    sim = BRWSimulator(offspring, jumps, rng)
    n0 = int(math.floor(x / chat))

    # MC (crude, small x only)
    t0 = time.time()
    hits = 0
    for _ in range(mc_trials):
        _, tau = sim.simulate_until(n_max=n0, x_fpt=x, u_fpt=u, r_fpt=r)
        hits += int(tau is not None and tau <= n0)
    mc_est = hits / mc_trials
    mc_se = math.sqrt(mc_est * (1 - mc_est) / max(1, mc_trials))
    t_mc = time.time() - t0

    # IS
    t0 = time.time()
    cfg = SpineISConfig(K_bias=6, L=6, m_power=1/3, max_replications=is_trials, rng=rng)
    est = SpineISEstimator(offspring=offspring, jumps=jumps, chat=chat, u=u, r=r, config=cfg)
    out = est.estimate_cdf(x)
    t_is = time.time() - t0

    return {
        "mc": {"p": mc_est, "stderr": mc_se, "N": mc_trials, "time_sec": t_mc},
        "is": {"p": out["prob_hat"], "stderr": out["stderr"], "N": out["n_eff"], "time_sec": t_is},
    }

def sanity_compare_asymptotic(
    offspring: OffspringLaw,
    jumps: GaussianJump,
    x: float,
    chat: float,
    u: ArrayLike,
    r: float = 1.0,
    is_trials: int = 3000,
    seed: int = 123,
) -> dict[str, float]:
    """
    Compare Spine-IS estimate vs Theorem 1 asymptotic at moderately large x.
    """
    u = _ensure_unit_vector(u)
    rng = np.random.default_rng(seed)
    cfg = SpineISConfig(K_bias=8, L=8, m_power=1/3, max_replications=is_trials, rng=rng)
    est = SpineISEstimator(offspring=offspring, jumps=jumps, chat=chat, u=u, r=r, config=cfg)
    out = est.estimate_cdf(x)
    asymp = lower_tail_asymptotics(x, chat, offspring, jumps, u=u, r=r)
    return {"is_p": out["prob_hat"], "is_se": out["stderr"], "asymp": asymp}

# -----------------------------------------------------------------------------#
# Example usage (safe defaults)
# -----------------------------------------------------------------------------#

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d = 3
    mu = np.zeros(d)
    Sigma = np.eye(d)
    jumps = GaussianJump.from_params(mu, Sigma)
    offspring = OffspringLaw.from_pmf([0.2, 0.3, 0.5])  # rho = 1.3
    u = np.array([1.0, 0.0, 0.0])  # direction

    speed = BRWSpeed.for_gaussian_along_u(offspring, jumps, u)
    c1 = speed.c1
    print(f"[INFO] c1 ≈ {c1:.4f}")

    x = 80.0
    chat_fast = c1 + 0.6
    p_asymp = lower_tail_asymptotics(x, chat_fast, offspring, jumps, u)
    print(f"[Thm1] Asymptotic P(τ_x < x/chat) ≈ {p_asymp:.3e}")

    chat_slow = c1 - 0.4
    T_val = upper_tail_T_value(chat_slow, offspring, jumps, u, grid_points=300)
    print(f"[Thm2] T(chat={chat_slow:.3f}) ≈ {T_val:.6f} (log prob rate)")

    # Spine-IS estimate
    cfg = SpineISConfig(K_bias=6, L=6, m_power=1/3, max_replications=800, rng=rng, antithetic=True)
    is_est = SpineISEstimator(offspring, jumps, chat_fast, u, 1.0, cfg)
    out = is_est.estimate_cdf(x)
    print(f"[Spine-IS] P(τ_x <= x/chat) ≈ {out['prob_hat']:.3e}  (stderr {out['stderr']:.1e}, N={out['n_eff']})")

    # Small-x benchmark (safe)
    bench = benchmark_small_x(offspring, jumps, x=8.0, chat=c1+0.8, u=u, r=1.0, mc_trials=200, is_trials=800)
    print(f"[Bench small-x] MC: {bench['mc']},  IS: {bench['is']}")

    # Sanity vs asymptotics
    sanity = sanity_compare_asymptotic(offspring, jumps, x=60.0, chat=c1+0.5, u=u, r=1.0, is_trials=2000)
    print(f"[Sanity asymptotic] {sanity}")
