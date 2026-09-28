from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cpu_mc import mc_european_cpu
from src.analytics import bs_price, bs_delta, bs_vega

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20

call = bs_price(S0, K, T, r, sigma, "call")
put = bs_price(S0, K, T, r, sigma, "put")

delta = bs_delta(S0, K, T, r, sigma)
vega = bs_vega(S0, K, T, r, sigma)

print(f"Call : {call:.6f}")
print(f"Put  : {put:.6f}")
print(f"Delta: {delta:.6f}")
print(f"Vega : {vega:.6f}")

print("-"*60)
print("\nMonte Carlo Simulation Results:")
print("-" * 60)

for N in [
    10_000,
    100_000,
    1_000_000,
    10_000_000
]:

    price, se = mc_european_cpu(
        S0,
        K,
        T,
        r,
        sigma,
        N
    )

    exact = bs_price(
        S0,
        K,
        T,
        r,
        sigma
    )

    z = (price - exact) / se

    print(
        f"N={N:>10,} | "
        f"MC={price:.6f} | "
        f"SE={se:.6f} | "
        f"z={z:+.2f}"
    )