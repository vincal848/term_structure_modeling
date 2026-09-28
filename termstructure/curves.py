"""Yield-curve construction: bootstrapping zero rates and Nelson-Siegel fitting.

Conventions: maturities in years; par yields are bond-equivalent (semiannual);
zero rates are continuously compounded, so P(tau) = exp(-z * tau).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

COUPON_FREQ = 2  # Treasury notes and bonds pay semiannually


# --- Bootstrapping -----------------------------------------------------------


def bootstrap_zeros(maturities, par_yields, max_maturity=None):
    """Bootstrap continuously compounded zero rates from a par curve.

    Maturities up to one coupon period carry no coupon, so the quoted yield is
    already a (semiannually compounded) zero rate. Beyond that, par yields are
    interpolated onto the semiannual coupon grid and each discount factor is
    solved from the par-bond condition

        c/2 * sum_{i<n} P_i + (1 + c/2) * P_n = 1.

    Interpolation is monotone piecewise-cubic (PCHIP): a natural cubic spline
    overshoots between the sparse long maturities (10y/20y/30y) and creates
    humps and dips in the zero curve that are not in the data.

    Returns (taus, zero_rates) covering the sub-coupon quotes plus the coupon grid.
    """
    m = np.asarray(maturities, dtype=float)
    y = np.asarray(par_yields, dtype=float)
    order = np.argsort(m)
    m, y = m[order], y[order]
    step = 1.0 / COUPON_FREQ
    max_maturity = m[-1] if max_maturity is None else max_maturity

    short = m < step
    short_taus = m[short]
    short_zeros = COUPON_FREQ * np.log1p(y[short] / COUPON_FREQ)

    grid = np.arange(step, max_maturity + 1e-9, step)
    par_on_grid = PchipInterpolator(m, y)(grid)

    discounts = np.empty_like(grid)
    coupon_annuity = 0.0  # running sum of earlier discount factors
    for n, c in enumerate(par_on_grid):
        discounts[n] = (1.0 - c / COUPON_FREQ * coupon_annuity) / (1.0 + c / COUPON_FREQ)
        coupon_annuity += discounts[n]

    grid_zeros = -np.log(discounts) / grid
    return np.concatenate([short_taus, grid]), np.concatenate([short_zeros, grid_zeros])


def par_from_zeros(taus, zero_rates):
    """Inverse of the bootstrap on a semiannual grid: par yields from zero rates."""
    taus = np.asarray(taus, dtype=float)
    discounts = np.exp(-np.asarray(zero_rates, dtype=float) * taus)
    return COUPON_FREQ * (1.0 - discounts) / np.cumsum(discounts)


# --- Nelson-Siegel -----------------------------------------------------------


def _ns_loadings(tau, lam):
    tau = np.asarray(tau, dtype=float)
    x = lam * tau
    safe_x = np.where(x < 1e-8, 1.0, x)
    slope = np.where(x < 1e-8, 1.0 - x / 2, (1.0 - np.exp(-safe_x)) / safe_x)
    curvature = slope - np.exp(-x)
    return slope, curvature


def nelson_siegel(tau, b0, b1, b2, lam):
    """Nelson-Siegel zero rate: level b0, slope b1, curvature b2, decay lam (per year)."""
    slope, curvature = _ns_loadings(tau, lam)
    return b0 + b1 * slope + b2 * curvature


@dataclass(frozen=True)
class NSFit:
    b0: float
    b1: float
    b2: float
    lam: float
    rmse: float  # in-sample fit error, same units as the input yields

    @property
    def params(self):
        return self.b0, self.b1, self.b2, self.lam

    def zero(self, tau):
        return nelson_siegel(tau, *self.params)

    def forward(self, tau):
        """Instantaneous forward rate f(0, tau) = d/dtau [tau * z(tau)]."""
        tau = np.asarray(tau, dtype=float)
        decay = np.exp(-self.lam * tau)
        return self.b0 + self.b1 * decay + self.b2 * self.lam * tau * decay

    def forward_slope(self, tau):
        """d f(0, tau) / d tau, needed for the Hull-White drift."""
        tau = np.asarray(tau, dtype=float)
        decay = np.exp(-self.lam * tau)
        return -self.lam * self.b1 * decay + self.b2 * self.lam * decay * (1.0 - self.lam * tau)

    def discount(self, tau):
        return np.exp(-self.zero(tau) * np.asarray(tau, dtype=float))


def fit_nelson_siegel(taus, zero_rates, lam_grid=None, refine=True) -> NSFit:
    """Fit Nelson-Siegel to zero rates.

    For fixed lam the model is linear in (b0, b1, b2), so a grid search over lam
    with least squares for the betas finds the global basin without starting
    values; a joint nonlinear refinement then polishes all four parameters.
    """
    taus = np.asarray(taus, dtype=float)
    y = np.asarray(zero_rates, dtype=float)
    lam_grid = np.linspace(0.05, 3.0, 300) if lam_grid is None else lam_grid

    best = None
    for lam in lam_grid:
        slope, curvature = _ns_loadings(taus, lam)
        X = np.column_stack([np.ones_like(taus), slope, curvature])
        betas, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((X @ betas - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, *betas, lam)
    params = np.array(best[1:])

    if refine:
        res = least_squares(
            lambda p: nelson_siegel(taus, *p) - y,
            params,
            bounds=([-np.inf, -np.inf, -np.inf, 1e-3], [np.inf, np.inf, np.inf, 10.0]),
        )
        params = res.x

    rmse = float(np.sqrt(np.mean((nelson_siegel(taus, *params) - y) ** 2)))
    return NSFit(*map(float, params), rmse=rmse)
