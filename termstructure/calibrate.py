"""Calibrating Vasicek and CIR two ways.

- Time series (physical measure P): fit the short-rate dynamics to the history of
  the 3-month T-bill. Tells you how rates actually move.
- Cross section (pricing measure Q): choose parameters so model yields match one
  day's zero curve. Tells you what the curve prices in.

The two generally disagree; the gap is the market price of interest-rate risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.stats import ncx2

from .affine import CIR, Vasicek


def tbill_to_continuous(discount_yield, days=91):
    """Convert a T-bill bank-discount yield (FRED DTB3, decimal) to a continuous rate."""
    price = 1.0 - np.asarray(discount_yield) * days / 360.0
    return -np.log(price) / (days / 365.0)


def weekly(rates: pd.Series) -> pd.Series:
    """Friday-close weekly sampling: daily bill data are noisy and the model is not about microstructure."""
    return rates.resample("W-FRI").last().dropna()


@dataclass(frozen=True)
class Calibration:
    model: Vasicek | CIR
    std_errors: dict
    n_obs: int
    rmse_bp: float | None = None  # cross-sectional fits only
    extras: dict = field(default_factory=dict)

    @property
    def half_life(self):
        """Years for an expected deviation from theta to halve: ln 2 / kappa."""
        return np.log(2) / self.model.kappa


def _numerical_hessian(f, x, rel_step=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    h = rel_step * np.maximum(np.abs(x), 1e-6)
    H = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h[i], np.eye(n)[j] * h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def fit_vasicek_ar1(rates: pd.Series, dt=1 / 52) -> Calibration:
    """Vasicek under P from its exact discretisation, an AR(1):

        r_{t+dt} = a + b r_t + eps,  b = exp(-kappa dt),  a = theta (1 - b),
        Var(eps) = sigma^2 (1 - b^2) / (2 kappa).

    OLS gives the exact Gaussian MLE of (a, b, Var eps); standard errors of
    (kappa, theta, sigma) follow by the delta method.
    """
    r = np.asarray(rates, dtype=float)
    x, y = r[:-1], r[1:]
    X = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ np.array([a, b])
    n = len(y)
    s = np.sqrt(resid @ resid / (n - 2))
    if not 0 < b < 1:
        raise ValueError(f"AR(1) coefficient {b:.4f} is not in (0, 1): no mean reversion in this sample")

    def to_params(p):
        a_, b_, s_ = p
        kappa = -np.log(b_) / dt
        return np.array([kappa, a_ / (1 - b_), s_ * np.sqrt(2 * kappa / (1 - b_**2))])

    kappa, theta, sigma = to_params([a, b, s])
    cov_ab = s**2 * np.linalg.inv(X.T @ X)
    cov = np.zeros((3, 3))
    cov[:2, :2] = cov_ab
    cov[2, 2] = s**2 / (2 * n)  # asymptotic variance of the residual std dev
    eps = 1e-7
    J = np.column_stack([(to_params(np.array([a, b, s]) + eps * np.eye(3)[k]) -
                          to_params(np.array([a, b, s]) - eps * np.eye(3)[k])) / (2 * eps) for k in range(3)])
    se = np.sqrt(np.diag(J @ cov @ J.T))
    return Calibration(Vasicek(kappa, theta, sigma), dict(zip(("kappa", "theta", "sigma"), se)), n,
                       extras={"ar1_b": b})


def cir_log_likelihood(params, rates, dt):
    """Exact CIR log-likelihood: r_{t+dt} / c is noncentral chi-square."""
    kappa, theta, sigma = params
    e = np.exp(-kappa * dt)
    c = sigma**2 * (1 - e) / (4 * kappa)
    df = 4 * kappa * theta / sigma**2
    x, y = rates[:-1], rates[1:]
    return float(np.sum(ncx2.logpdf(y / c, df, x * e / c) - np.log(c)))


def fit_cir_mle(rates: pd.Series, dt=1 / 52, floor=1e-4) -> Calibration:
    """CIR under P by exact maximum likelihood.

    CIR cannot produce zero or negative rates, so observations are floored at
    `floor` (1 bp): the 2009-15 and 2020-21 bills traded at or just below zero.
    Standard errors come from the inverse numerical Hessian of the log-likelihood.
    """
    r = np.maximum(np.asarray(rates, dtype=float), floor)
    start = fit_vasicek_ar1(pd.Series(r), dt).model
    x0 = np.log([start.kappa, max(start.theta, floor), start.sigma / np.sqrt(max(r.mean(), floor))])

    def nll(log_p):
        val = -cir_log_likelihood(np.exp(log_p), r, dt)
        return val if np.isfinite(val) else 1e10

    res = minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
    params = np.exp(res.x)
    H = _numerical_hessian(lambda p: -cir_log_likelihood(p, r, dt), params)
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    return Calibration(CIR(*params), dict(zip(("kappa", "theta", "sigma"), se)), len(r) - 1,
                       extras={"log_likelihood": -res.fun})


def fit_to_curve(model_cls, taus, zero_rates, r0, sigma) -> Calibration:
    """Cross-sectional (risk-neutral) fit of (kappa, theta) to one day's zero curve.

    sigma is held at its time-series estimate: by Girsanov, changing from P to Q
    changes only the drift, so the diffusion coefficient is the same under both
    measures. (Leaving sigma free lets the optimiser use absurd volatilities, e.g.
    20%+, to bend long yields through convexity.) r0 is held at the observed short
    rate, so the model must explain the whole curve from today's short rate and a
    constant drift.
    """
    taus = np.asarray(taus, dtype=float)
    z = np.asarray(zero_rates, dtype=float)
    if model_cls is CIR:
        r0 = max(r0, 1e-4)
        lower, upper = [1e-3, 1e-4], [5.0, 0.3]
    else:
        lower, upper = [1e-3, -0.1], [5.0, 0.3]

    def resid(p):
        return model_cls(p[0], p[1], sigma).zero_yield(r0, taus) - z

    best = None
    for k0 in (0.05, 0.2, 0.5, 1.0, 2.0):  # multistart: kappa and theta trade off along a ridge
        res = least_squares(resid, np.clip([k0, z[-1]], lower, upper), bounds=(lower, upper))
        if best is None or res.cost < best.cost:
            best = res
    rmse = float(np.sqrt(np.mean(best.fun**2)))
    at_bound = bool(np.any(np.isclose(best.x, lower) | np.isclose(best.x, upper)))
    return Calibration(model_cls(best.x[0], best.x[1], sigma), {}, len(taus), rmse_bp=rmse * 1e4,
                       extras={"r0": r0, "at_bound": at_bound})
