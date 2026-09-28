"""The one-factor short-rate family from the README, as a single simulator.

Every model discussed is a special case of

    dr = [K0 + K1 r + K2 r log r] dt + [H0 + H1 r]^v dB^Q,

so a model is just a set of coefficients. Presets for the lognormal models live
here; Vasicek and CIR (which also have closed-form bond prices) are in affine.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMES = ("euler", "euler_floor", "full_truncation", "log_euler")


@dataclass(frozen=True)
class OneFactorModel:
    K0: float = 0.0
    K1: float = 0.0
    K2: float = 0.0
    H0: float = 0.0
    H1: float = 0.0
    v: float = 1.0
    name: str = "one-factor"

    def drift(self, r):
        r = np.asarray(r, dtype=float)
        out = self.K0 + self.K1 * r
        if self.K2:
            rp = np.maximum(r, 1e-12)
            out = out + self.K2 * rp * np.log(rp)
        return out

    def diffusion(self, r):
        base = self.H0 + self.H1 * np.asarray(r, dtype=float)
        if self.v == 1.0:
            return base
        return np.maximum(base, 0.0) ** self.v

    @property
    def is_lognormal(self):
        """H0 = 0, v = 1: volatility proportional to r, so log r is Gaussian-driven."""
        return self.H0 == 0.0 and self.v == 1.0 and self.H1 != 0.0

    def simulate(self, r0, T, dt, n_paths, seed=None, scheme="euler", antithetic=True):
        """Simulate short-rate paths; returns (times, paths) with paths shaped (n_steps + 1, n_paths).

        Schemes
        - "euler": plain Euler-Maruyama. Right for Gaussian models (Vasicek), where negative rates
          are part of the model.
        - "euler_floor": Euler, then r = max(r, 0). This is what the original R code did.
          It biases rates upward whenever the floor binds.
        - "full_truncation": Euler with coefficients evaluated at max(r, 0) (Lord et al., 2010);
          the reported rate is max(r, 0). Standard for square-root models.
        - "log_euler": Euler on log r via Ito, for lognormal models (Dothan, Black-Karasinski);
          rates stay positive by construction.

        With antithetic=True the second half of the paths reuses the negated shocks of the first half.
        """
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}")
        if scheme == "log_euler" and not self.is_lognormal:
            raise ValueError("log_euler requires a lognormal model (H0 = 0, v = 1)")
        n_steps = int(round(T / dt))
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        sqdt = np.sqrt(dt)

        paths = np.empty((n_steps + 1, 2 * half if antithetic else n_paths))
        paths[0] = r0
        state = np.full(paths.shape[1], float(r0))
        if scheme == "log_euler":
            state = np.log(state)
        for i in range(1, n_steps + 1):
            z = rng.standard_normal(half)
            dW = (np.concatenate([z, -z]) if antithetic else z) * sqdt
            if scheme == "log_euler":
                r = np.exp(state)
                state = state + (self.drift(r) / r - 0.5 * self.H1**2) * dt + self.H1 * dW
                paths[i] = np.exp(state)
            elif scheme == "full_truncation":
                rp = np.maximum(state, 0.0)
                state = state + self.drift(rp) * dt + self.diffusion(rp) * dW
                paths[i] = np.maximum(state, 0.0)
            else:
                state = state + self.drift(state) * dt + self.diffusion(state) * dW
                if scheme == "euler_floor":
                    state = np.maximum(state, 0.0)
                paths[i] = state
        return np.linspace(0.0, n_steps * dt, n_steps + 1), paths


def dothan(mu, sigma):
    """Dothan: dr = mu r dt + sigma r dB (geometric Brownian motion short rate)."""
    return OneFactorModel(K1=mu, H1=sigma, v=1.0, name="Dothan")


def black_karasinski(kappa, theta_log, sigma):
    """Black-Karasinski: d log r = kappa (theta_log - log r) dt + sigma dB.

    By Ito, in r: dr = r [kappa theta_log + sigma^2 / 2 - kappa log r] dt + sigma r dB,
    i.e. K1 = kappa theta_log + sigma^2 / 2, K2 = -kappa, H1 = sigma, v = 1.
    """
    return OneFactorModel(K1=kappa * theta_log + 0.5 * sigma**2, K2=-kappa, H1=sigma, v=1.0,
                          name="Black-Karasinski")


def mc_bond_price(times, paths, antithetic=True):
    """Monte Carlo zero-coupon bond price P(0, T) = E^Q[exp(-int_0^T r dt)] for T = times[-1].

    The integral uses the trapezoid rule. Returns (price, standard_error). With antithetic
    paths the standard error is computed on pair averages, since the pairs are correlated.
    """
    integral = np.trapezoid(paths, times, axis=0)
    discount = np.exp(-integral)
    if antithetic:
        half = discount.size // 2
        discount = 0.5 * (discount[:half] + discount[half:])
    return float(discount.mean()), float(discount.std(ddof=1) / np.sqrt(discount.size))


def mc_bond_curve(times, paths, maturities, antithetic=True):
    """MC bond prices and standard errors for several maturities from one set of paths."""
    out = []
    for T in maturities:
        k = int(np.searchsorted(times, T - 1e-12))
        out.append(mc_bond_price(times[: k + 1], paths[: k + 1], antithetic))
    prices, errors = map(np.array, zip(*out))
    return prices, errors
