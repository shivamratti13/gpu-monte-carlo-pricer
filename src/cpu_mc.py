import math
import numpy as np

def mc_european_cpu(
        S0,
        K,
        T,
        r,
        sigma,
        n_paths,
        kind="call",
        seed=42,
        batch_size=2_000_000,
):
    """
    Monte Carlo simulation for European option pricing.

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - n_paths: Number of Monte Carlo paths to simulate
    - kind: Type of option ('call' or 'put')
    - seed: Random seed for reproducibility
    - batch_size: Number of paths to simulate in each batch

    Returns:
    - Estimated option price
    """

    rng = np.random.default_rng(seed)

    drift = (r - 0.5 * sigma ** 2) * T
    vol = sigma * math.sqrt(T)

    total_sum = 0.0
    total_sumsq = 0.0

    done = 0
    while done < n_paths:
        m = min(batch_size, n_paths - done)
        Z = rng.standard_normal(m)
        ST = S0 * np.exp(drift + vol * Z)

        if kind == "call":
            payoff = np.maximum(ST - K, 0.0)
        elif kind == "put":
            payoff = np.maximum(K - ST, 0.0)
        else:
            raise ValueError("kind must be 'call' or 'put'")

        total_sum += payoff.sum(dtype=np.float64)
        total_sumsq += np.square(payoff).sum(dtype=np.float64)

        done += m

    mean = total_sum / n_paths

    variance = (total_sumsq / n_paths) - mean ** 2
    variance = max(variance, 0.0)

    discount = math.exp(-r * T)
    price = discount * mean

    standard_error = discount * math.sqrt(variance / n_paths)

    return price, standard_error