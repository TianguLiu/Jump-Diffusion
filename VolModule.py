import numpy as np
import scipy.stats as stats
import math
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq

# Black-Scholes formula for European call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

# Black-Scholes formula for European put option
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

# Merton Jump Diffusion call pricing function
def merton_jump_diffusion_call(S, K, T, r, sigma, lamb, sigma_j, N=50):    
    k = 0.0
    lambda_p = lamb * (1 + k)
    price = 0.0

    total_prob = 0.0
    for n in range(N):
        poisson_prob = np.exp(-lambda_p * T) * (lambda_p * T)**n / math.factorial(n)
        sigma_n_squared = sigma**2 + n * sigma_j**2 / T
        sigma_n = np.sqrt(sigma_n_squared)
        r_n = r - lamb * k + n * np.log(1 + k) / T
        bs_price = black_scholes_call(S, K, T, r_n, sigma_n)
        price += poisson_prob * bs_price
        total_prob+=poisson_prob
        
    #print("total prob" ,total_prob)
    return price

# Merton Jump Diffusion put pricing function
def merton_jump_diffusion_put(S, K, T, r, sigma, lamb, sigma_j, N=50):    
    k = 0.0
    lambda_p = lamb * (1 + k)
    price = 0.0
    for n in range(N):
        poisson_prob = np.exp(-lambda_p * T) * (lambda_p * T)**n / math.factorial(n)
        sigma_n_squared = sigma**2 + n * sigma_j**2 / T
        sigma_n = np.sqrt(sigma_n_squared)
        r_n = r - lamb * k + n * np.log(1 + k) / T
        bs_price = black_scholes_put(S, K, T, r_n, sigma_n)
        price += poisson_prob * bs_price
    return price


# Monte Carlo simulation
def monte_carlo_jump_diffusion_option_price(
    S0=95, K=95, T=1.0, r=0.05,
    mu=0.05, sigma=0.2,
    lamb=0.75, mu_j=-0.5 * 0.3**2, sigma_j=0.3,
    N=252, M=1500, seed=42
):
    #np.random.seed(seed)
    dt = T / N
    call_payoff_sum = 0.0
    put_payoff_sum = 0.0
    sum_J = 0.0
    for _ in range(M):
        S = S0
        for _ in range(N):
            dW = np.random.normal(0, np.sqrt(dt))
            jump = np.random.rand() < lamb * dt
            J = np.random.lognormal(mean=mu_j, sigma=sigma_j) if jump else 1.0
            dS = mu * S * dt + sigma * S * dW + (J - 1) * S
            S += dS
            sum_J += J;
        call_payoff_sum += max(S - K, 0)
        put_payoff_sum += max(K - S, 0)

    call_price = np.exp(-r * T) * (call_payoff_sum / M)
    put_price = np.exp(-r * T) * (put_payoff_sum / M)
    #print(sum_J)
    return call_price, put_price

    
# Function to compute implied volatility given a market price
def compute_call_implied_volatility(S, K, T, r, market_price):
    # Define the error function between BS price and market price
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price

    # Use Brent's method to find the root (i.e., implied volatility)
    return brentq(objective, 0.01, 2.0)

# Function to compute implied volatility from put option market price
def compute_put_implied_volatility(S, K, T, r, market_price):
    def objective(sigma):
        return black_scholes_put(S, K, T, r, sigma) - market_price

    # Brent's method to find root in volatility range [0.0001, 5.0]
    return brentq(objective, 0.01, 2.0)

# ---------------------------
# Example usage
# ---------------------------

# Market and option parameters
S = 95         # Spot price
K = 95         # Strike price
T = 1           # Time to maturity (1 year)
r = 0.05        # Risk-free rate
sigma = 0.2     # Diffusion volatility
lamb = 0.75     # Jump intensity (lambda)
sigma_log_j = 0.3   # Std deviation of log jump size

# Black-Scholes
call_bs = black_scholes_call(S, K, T, r, sigma)
put_bs = black_scholes_put(S, K, T, r, sigma)

# Merton Model
call_merton_jump = merton_jump_diffusion_call(S, K, T, r, sigma, lamb, sigma_log_j)
put_merton_jump = merton_jump_diffusion_put(S, K, T, r, sigma, lamb, sigma_log_j)



# Monte Carlo
call_mc_jump, put_mc_jump = monte_carlo_jump_diffusion_option_price()


# Calculate implied volatility
call_bs_implied_vol = compute_call_implied_volatility(S, K, T, r, call_bs)
call_merton_jump_implied_vol = compute_call_implied_volatility(S, K, T, r, call_merton_jump)
call_mc_jump_implied_vol = compute_call_implied_volatility(S, K, T, r, call_mc_jump)

# Calculate implied volatility
put_bs_implied_vol = compute_put_implied_volatility(S, K, T, r, put_bs)
put_merton_jump_implied_vol = compute_put_implied_volatility(S, K, T, r, put_merton_jump)
put_mc_jump_implied_vol = compute_put_implied_volatility(S, K, T, r, put_mc_jump)

# Results
print(f"Black-Scholes Call:               {call_bs:.4f}")
print(f"Merton Call (Jump Diffusion):     {call_merton_jump:.4f}")
print(f"Monte Carlo Call (Jump):          {call_mc_jump:.4f}")
print()
print(f"Black-Scholes Put:                {put_bs:.4f}")
print(f"Merton Put  (Jump Diffusion):     {put_merton_jump:.4f}")
print(f"Monte Carlo Put  (Jump):          {put_mc_jump:.4f}")

print()
print(f"Call BS Implied Volatility: {call_bs_implied_vol:.2%}")
print(f"Call Merton Jump Implied Volatility: {call_merton_jump_implied_vol:.2%}")
print(f"Call MC Jump Volatility: {call_mc_jump_implied_vol:.2%}")
print()
print(f"Put BS Implied Volatility: {put_bs_implied_vol:.2%}")
print(f"Put Merton Jump Implied Volatility: {put_merton_jump_implied_vol:.2%}")
print(f"Put MC Jump Volatility: {put_mc_jump_implied_vol:.2%}")

