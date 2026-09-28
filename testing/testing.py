from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import time

from src.cpu_mc import mc_european_cpu
from src.analytics import bs_price

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20

Ns = [
    1_000,
    3_000,
    10_000,
    30_000,
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000
]

exact = bs_price(S0, K, T, r, sigma)

prices = []
errors = []
standard_errors = []
times = []

for N in Ns:

    start = time.perf_counter()

    price, se = mc_european_cpu(
        S0,
        K,
        T,
        r,
        sigma,
        N,
        seed=42
    )

    elapsed = time.perf_counter() - start

    prices.append(price)
    errors.append(abs(price - exact))
    standard_errors.append(se)
    times.append(elapsed)

    print(
        f"N={N:>10,} | "
        f"price={price:.6f} | "
        f"SE={se:.6f} | "
        f"error={abs(price-exact):.6f} | "
        f"time={elapsed:.3f}s"
    )

plt.figure(figsize=(8, 6))

plt.loglog(
    Ns,
    errors,
    "o-",
    label="Observed absolute error"
)

plt.loglog(
    Ns,
    standard_errors,
    "s--",
    label="Standard error"
)

reference = (
    errors[0]
    * (np.array(Ns) / Ns[0]) ** (-0.5)
)

plt.loglog(
    Ns,
    reference,
    ":",
    label=r"$N^{-1/2}$ reference"
)

plt.xlabel("Number of Monte Carlo paths (N)")
plt.ylabel("Error")
plt.title("Monte Carlo Convergence")

plt.grid(
    True,
    which="both",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "convergence.png",
    dpi=150
)

plt.show()

log_N = np.log(np.array(Ns))
log_error = np.log(np.array(errors))

slope, intercept = np.polyfit(
    log_N,
    log_error,
    1
)

print(f"Estimated convergence slope: {slope:.4f}")

N = 100_000
exact = bs_price(S0, K, T, r, sigma)

inside = 0
experiments = 100

for seed in range(experiments):

    price, se = mc_european_cpu(
        S0,
        K,
        T,
        r,
        sigma,
        N,
        seed=seed
    )

    lower = price - 1.96 * se
    upper = price + 1.96 * se

    if lower <= exact <= upper:
        inside += 1

coverage = inside / experiments

print(f"Coverage: {coverage:.2%}")