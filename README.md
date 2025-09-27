# BRW‑FPT: Large Deviations & Spine IS for Branching Random Walks

A production‑grade Python implementation of **large‑deviation asymptotics** and a **spine‑decomposition importance sampling** (IS) estimator for first‑passage times (FPT) in supercritical branching random walks (BRW) in (\mathbb R^d).

> Focus: probability of **rare, unusually fast arrivals** (lower‑tail) and the **log‑rate** for unusually slow arrivals (upper‑tail). Includes a NumPy‑only simulator, closed‑form Gaussian rate function, and polynomial‑time IS.

---

## Highlights

* **Exact Gaussian rate function** (I(v) = \tfrac12 (v-\mu)^\top \Sigma^{-1}(v-\mu)) and **frontier speed** (c_1) along any unit direction.
* **Lower‑tail asymptotics (Theorem 1)** and **upper‑tail rate (Theorem 2)** along a chosen direction.
* **Spine IS estimator** for (\mathbb P(\tau_x \le x/\hat c_1)) with size‑biased offspring, exponential tilting, and controlled decorations (polynomial runtime).
* **General FPT geometry**: target ball (B_r(x u)) with radius `r` and unit direction `u`.
* **Utilities**: Bernstein CIs, replication planning, local‑CLT helper (exact for Gaussian), micro‑benchmarks vs. crude MC.
* **NumPy‑only**. Strong typing, inline docs, runtime checks.

---

## Repository name & layout

**Suggested repo name:** `brw-fpt`
Short, searchable, and neutral. Alternatives: `branching-fpt`, `spine-brw-fpt`.

**Python package name:** `brwfpt`
Avoids hyphens on import; keeps the brand compact.

**Recommended layout**

```
brw-fpt/
├─ pyproject.toml            # Poetry config
├─ README.md                 # (this file)
├─ LICENSE                   # MIT (suggested)
├─ src/
│  └─ brwfpt/
│     ├─ __init__.py
│     └─ brw_fpt.py          # Main module (full implementation)
├─ tests/
│  └─ test_smoke.py
└─ examples/
   └─ quickstart.ipynb
```

If you prefer a single‑file drop‑in for notebooks, keep `brw_fpt.py` at project root.

**File naming**
Use `brw_fpt.py` for the consolidated implementation, or `brw_fpt_extended.py` if you keep both base and extended variants.

---

## Installation

We recommend **Poetry**.

```bash
# inside brw-fpt/
poetry init -n
poetry add numpy
poetry install
# activate env
poetry shell
```

> Python >= 3.9. No external native deps beyond NumPy.

---

## Quickstart

```python
import numpy as np
from brwfpt.brw_fpt import (
    GaussianJump, OffspringLaw, BRWSpeed,
    lower_tail_asymptotics, upper_tail_T_value,
    SpineISConfig, SpineISEstimator,
)

# Model: 3D Gaussian jumps, supercritical offspring
jumps = GaussianJump.from_params(mu=[0.0, 0.0, 0.0], Sigma=np.eye(3))
offspring = OffspringLaw.from_pmf([0.2, 0.3, 0.5])  # rho=1.3
u = np.array([1.0, 0.0, 0.0])  # direction; must be unit

# Frontier speed c1 along u
c1 = BRWSpeed.for_gaussian_along_u(offspring, jumps, u).c1

# Lower‑tail asymptotic at distance x
x = 80.0
chat = c1 + 0.6
p_asymp = lower_tail_asymptotics(x, chat, offspring, jumps, u)
print("asymptotic P(τ_x < x/ĉ) ≈", p_asymp)

# Upper‑tail log‑rate
chat_slow = c1 - 0.4
T = upper_tail_T_value(chat_slow, offspring, jumps, u)
print("upper‑tail rate T(ĉ) =", T)

# Spine IS estimate (polynomial‑time)
cfg = SpineISConfig(K_bias=6, L=6, m_power=1/3, max_replications=2000)
est = SpineISEstimator(offspring, jumps, chat, u, r=1.0, config=cfg)
out = est.estimate_cdf(x)
print(out)  # {'prob_hat': ..., 'stderr': ..., 'n_eff': ...}
```

---

## API overview

### Distributions & speed

* `GaussianJump.from_params(mu, Sigma)`
  Multivariate Gaussian with exact tilting, MGF, and rate (I).
* `OffspringLaw.from_pmf(p)`
  Supercritical GW offspring (checks (\rho>1), (p_0+p_1>0)).
* `BRWSpeed.for_gaussian_along_u(offspring, jumps, u)` → `BRWSpeed(c1, lam_star_for_c1_u)`

### Asymptotics & rates

* `lower_tail_asymptotics(x, chat, offspring, jumps, u, r=1.0)`
  Returns (x^{-d/2}\exp{-(x/\hat c_1)(I(\hat c_1 u)-\log\rho)}).
* `upper_tail_T_value(chat, offspring, jumps, u, grid_points=200)`
  Returns (T) with (-\log P(\tau_x\ge x/\hat c_1)/x\to T).

### Simulation & IS

* `BRWSimulator(offspring, jumps, rng)` with `.simulate_until(n_max, x_fpt, u_fpt, r_fpt)`
* `SpineISEstimator(offspring, jumps, chat, u, r, config)` with:

  * `.estimate_cdf(x)` → point estimate & stderr for (\mathbb P(\tau_x \le x/\hat c_1))
  * `.estimate_window_pmf(x)` → per‑t masses within the right‑edge window

### Statistics utilities

* `bernstein_ci(samples, alpha=0.05, value_range=(0.0, 1.0))`
  Conservative CI for bounded means (falls back to normal if no bounds).
* `plan_replications_for_relative_error(p_hat, target_rel_err=0.2)`
  Back‑of‑envelope replication count.

---

## Mathematical model (brief)

* **Branching**: Galton–Watson with offspring (\zeta), mean (\rho=\mathbb E\zeta>1) and finite (\mathbb E\zeta^2).
* **Motion**: i.i.d. jumps (\xi\in\mathbb R^d), here Gaussian (\mathcal N(\mu,\Sigma)) (non‑lattice).
* **Target**: ball (B_r(x u)), unit direction (u), radius (r>0).
* **FPT**: (\tau_x = \inf{n\ge0: \exists\text{ particle in }B_r(xu) }).
* **Front speed**: along (u), (c_1) solves (I(c_1 u)=\log\rho).

---

## Reproducibility

* **Deterministic RNG** via `numpy.random.Generator` seeds in examples.
* Pure NumPy, no hidden state, explicit configs.

---

## Testing

Create a minimal smoke test (pytest):

```python
# tests/test_smoke.py
import numpy as np
from brwfpt.brw_fpt import GaussianJump, OffspringLaw, BRWSpeed

def test_speed_positive():
    jumps = GaussianJump.from_params([0,0,0], np.eye(3))
    off = OffspringLaw.from_pmf([0.2,0.3,0.5])
    u = np.array([1.0,0.0,0.0])
    c1 = BRWSpeed.for_gaussian_along_u(off, jumps, u).c1
    assert c1 > 0
```

Run:

```bash
poetry run pytest -q
```

---

## Roadmap

* Non‑Gaussian jumps: saddlepoint tilting + Edgeworth local‑CLT.
* Upper‑tail **simulation** (not just rate) with alternative change‑of‑measure.
* Vectorized multi‑target evaluation for grid scans over (u) and (x).
* Optional JAX backend for GPU acceleration (keeping NumPy API).

---

## License

MIT. See `LICENSE`.

---

## Citation

If you use this code in academic work, please cite this repository and the original paper on **large deviations of first‑passage times in BRW**.

```
@software{brw_fpt_2025,
  title   = {BRW-FPT: Large Deviations & Spine Importance Sampling for Branching Random Walks},
  author  = {Ribeiro, Diogo and contributors},
  year    = {2025},
  url     = {https://github.com/diogoribeiro7/brw-fpt},
  license = {MIT}
}
```
