"""Affine short-rate models with closed-form bond prices: Vasicek and CIR.

Both give P(t, T) = A(tau) exp(-B(tau) r_t) with tau = T - t, so zero yields are
affine in the short rate: y(tau) = (B(tau) r - log A(tau)) / tau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .short_rate import OneFactorModel


class _Affine:
    def bond_price(self, r, tau):
        tau = np.asarray(tau, dtype=float)
        return np.exp(self.log_A(tau) - self.B(tau) * r)

    def zero_yield(self, r, tau):
        tau = np.asarray(tau, dtype=float)
        return (self.B(tau) * r - self.log_A(tau)) / tau


@dataclass(frozen=True)
class Vasicek(_Affine):
    """dr = kappa (theta - r) dt + sigma dB."""

    kappa: float
    theta: float
    sigma: float

    def B(self, tau):
        return (1.0 - np.exp(-self.kappa * tau)) / self.kappa

    def log_A(self, tau):
        k, th, s = self.kappa, self.theta, self.sigma
        B = self.B(tau)
        return (th - s**2 / (2 * k**2)) * (B - tau) - s**2 * B**2 / (4 * k)

    @property
    def long_yield(self):
        """Limit of the zero yield as tau -> infinity."""
        return self.theta - self.sigma**2 / (2 * self.kappa**2)

    def general(self):
        return OneFactorModel(K0=self.kappa * self.theta, K1=-self.kappa, H0=self.sigma, v=1.0,
                              name="Vasicek")

    def transition_moments(self, r, dt):
        """Exact conditional mean and variance of r_{t+dt} given r_t = r."""
        e = np.exp(-self.kappa * dt)
        return self.theta + (r - self.theta) * e, self.sigma**2 * (1 - e**2) / (2 * self.kappa)

    def simulate_exact(self, r0, T, dt, n_paths, seed=None, antithetic=True):
        """Exact Gaussian transition: no discretisation error in r at the grid points."""
        n_steps = int(round(T / dt))
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        paths = np.empty((n_steps + 1, 2 * half if antithetic else n_paths))
        paths[0] = r0
        _, var = self.transition_moments(0.0, dt)
        sd = np.sqrt(var)
        for i in range(1, n_steps + 1):
            z = rng.standard_normal(half)
            z = np.concatenate([z, -z]) if antithetic else z
            mean, _ = self.transition_moments(paths[i - 1], dt)
            paths[i] = mean + sd * z
        return np.linspace(0.0, n_steps * dt, n_steps + 1), paths


@dataclass(frozen=True)
class CIR(_Affine):
    """dr = kappa (theta - r) dt + sigma sqrt(r) dB."""

    kappa: float
    theta: float
    sigma: float

    @property
    def gamma(self):
        return np.sqrt(self.kappa**2 + 2 * self.sigma**2)

    def _denominator(self, tau):
        g = self.gamma
        return (g + self.kappa) * np.expm1(g * tau) + 2 * g

    def B(self, tau):
        return 2 * np.expm1(self.gamma * tau) / self._denominator(tau)

    def log_A(self, tau):
        g, k = self.gamma, self.kappa
        return (2 * k * self.theta / self.sigma**2) * (
            np.log(2 * g) + 0.5 * (k + g) * tau - np.log(self._denominator(tau))
        )

    @property
    def feller(self):
        """2 kappa theta >= sigma^2 keeps r strictly positive."""
        return 2 * self.kappa * self.theta >= self.sigma**2

    @property
    def long_yield(self):
        return 2 * self.kappa * self.theta / (self.kappa + self.gamma)

    def general(self):
        return OneFactorModel(K0=self.kappa * self.theta, K1=-self.kappa, H1=self.sigma**2, v=0.5,
                              name="CIR")

    def simulate_exact(self, r0, T, dt, n_paths, seed=None):
        """Exact noncentral chi-square transition (no negative rates, no discretisation bias)."""
        n_steps = int(round(T / dt))
        rng = np.random.default_rng(seed)
        e = np.exp(-self.kappa * dt)
        c = self.sigma**2 * (1 - e) / (4 * self.kappa)
        df = 4 * self.kappa * self.theta / self.sigma**2
        paths = np.empty((n_steps + 1, n_paths))
        paths[0] = r0
        for i in range(1, n_steps + 1):
            paths[i] = c * rng.noncentral_chisquare(df, paths[i - 1] * e / c)
        return np.linspace(0.0, n_steps * dt, n_steps + 1), paths
