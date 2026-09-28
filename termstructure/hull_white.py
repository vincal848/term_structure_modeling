"""Hull-White: Vasicek with a time-varying target, fitted to today's curve.

    dr = (theta(t) - a r) dt + sigma dB^Q

Choosing theta(t) from the initial instantaneous forward curve f(0, t) makes model
bond prices equal market discount factors at every maturity:

    theta(t) = df(0,t)/dt + a f(0,t) + sigma^2 / (2a) (1 - exp(-2at)).

Equivalently r(t) = x(t) + alpha(t), where x is a zero-mean Ornstein-Uhlenbeck
process and alpha(t) = f(0,t) + sigma^2 / (2a^2) (1 - exp(-at))^2. That split gives
an exact simulation scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import NSFit


@dataclass(frozen=True)
class HullWhite:
    a: float
    sigma: float
    curve: NSFit  # initial term structure; supplies f(0, t), its slope, and P(0, T)

    def theta(self, t):
        t = np.asarray(t, dtype=float)
        a, s = self.a, self.sigma
        return self.curve.forward_slope(t) + a * self.curve.forward(t) + s**2 / (2 * a) * (1 - np.exp(-2 * a * t))

    def alpha(self, t):
        t = np.asarray(t, dtype=float)
        return self.curve.forward(t) + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t)) ** 2

    def B(self, tau):
        return (1 - np.exp(-self.a * np.asarray(tau, dtype=float))) / self.a

    def bond_price(self, t, T, r_t):
        """P(t, T) given r(t). At t = 0 with r(0) = f(0, 0) this returns the input curve exactly."""
        t, T = np.asarray(t, dtype=float), np.asarray(T, dtype=float)
        B = self.B(T - t)
        log_A = (np.log(self.curve.discount(T)) - np.log(self.curve.discount(t)) + B * self.curve.forward(t)
                 - self.sigma**2 / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * B**2)
        return np.exp(log_A - B * r_t)

    @property
    def r0(self):
        return float(self.curve.forward(0.0))

    def simulate_exact(self, T, dt, n_paths, seed=None, antithetic=True):
        """Exact paths r = x + alpha, with x an OU process started at 0."""
        n_steps = int(round(T / dt))
        times = np.linspace(0.0, n_steps * dt, n_steps + 1)
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        e = np.exp(-self.a * dt)
        sd = self.sigma * np.sqrt((1 - e**2) / (2 * self.a))
        x = np.zeros((n_steps + 1, 2 * half if antithetic else n_paths))
        for i in range(1, n_steps + 1):
            z = rng.standard_normal(half)
            x[i] = x[i - 1] * e + sd * (np.concatenate([z, -z]) if antithetic else z)
        return times, x + self.alpha(times)[:, None]
