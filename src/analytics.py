import math
from scipy.stats import norm

def bs_price(S0,K,T,r,sigma,kind='call'):
    """
    Calculate the Black-Scholes price of a European option.

    Parameters:
    S0 : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    kind : str
        Type of the option ('call' or 'put')

    Returns:
    float
        The price of the option
    """

    if T <= 0:
        if kind == "call":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    if sigma <= 0:
        forward = S0 * math.exp(r * T)

        if kind == "call":
            return math.exp(-r * T) * max(forward - K, 0.0)
        else:
            return math.exp(-r * T) * max(K - forward, 0.0)

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if kind == 'call':
        price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif kind == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("kind must be 'call' or 'put'")

    return price


def bs_delta(S0,K,T,r,sigma,kind='call'):
    """
    Calculate the Black-Scholes delta of a European option.

    Parameters:
    S0 : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying asset (annualized)
    kind : str
        Type of the option ('call' or 'put')

    Returns:
    float
        The delta of the option
    """

    if T <= 0:
        if kind == "call":
            return 1.0 if S0 > K else 0.0
        else:
            return -1.0 if S0 < K else 0.0


    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    if kind == 'call':
        delta = norm.cdf(d1)
    elif kind == 'put':
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("kind must be 'call' or 'put'")

    return delta

def bs_vega(S0, K, T, r, sigma):
    """
    Black-Scholes Vega.

    Returns change in option price per unit
    change in volatility.
    """

    sqrt_T = math.sqrt(T)

    d1 = (
        math.log(S0 / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * sqrt_T)

    return S0 * sqrt_T * norm.pdf(d1)


# print(bs_price(100, 110, 2, 0.05, 0.2, kind='call'))   # Example usage
# print(bs_delta(100, 110, 2, 0.05, 0.2, kind='call'))   # Example usage
# print(bs_vega(100, 110, 2, 0.05, 0.2))                 # Example usage

# print(bs_price(100, 90, 1.5, 0.05, 0.2, kind='put'))   # Example usage
# print(bs_delta(100, 90, 1.5, 0.05, 0.2, kind='put'))   # Example usage
# print(bs_vega(100, 90, 1.5, 0.05, 0.2))                # Example usage