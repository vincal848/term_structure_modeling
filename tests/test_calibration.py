import numpy as np
import pandas as pd
import pytest

from termstructure.affine import CIR, Vasicek
from termstructure.calibrate import (
    fit_cir_mle,
    fit_to_curve,
    fit_vasicek_ar1,
    tbill_to_continuous,
)
from termstructure.curves import NSFit
from termstructure.hull_white import HullWhite
from termstructure.short_rate import mc_bond_curve

DT = 1 / 52


def test_tbill_conversion():
    # 5% discount yield on a 91-day bill: price 0.987361, continuous rate ~5.1%
    r = tbill_to_continuous(0.05)
    assert r == pytest.approx(-np.log(1 - 0.05 * 91 / 360) * 365 / 91)
    assert r > 0.05


def test_vasicek_ar1_recovers_parameters():
    true = Vasicek(kappa=0.5, theta=0.04, sigma=0.01)
    _, paths = true.simulate_exact(0.04, T=100.0, dt=DT, n_paths=1, seed=2, antithetic=False)
    cal = fit_vasicek_ar1(pd.Series(paths[:, 0]), DT)
    for name in ("kappa", "theta", "sigma"):
        est, se = getattr(cal.model, name), cal.std_errors[name]
        assert abs(est - getattr(true, name)) < 3 * se, name


def test_cir_mle_recovers_parameters():
    true = CIR(kappa=0.5, theta=0.04, sigma=0.05)
    _, paths = true.simulate_exact(0.04, T=100.0, dt=DT, n_paths=1, seed=4)
    cal = fit_cir_mle(pd.Series(paths[:, 0]), DT)
    for name in ("kappa", "theta", "sigma"):
        est, se = getattr(cal.model, name), cal.std_errors[name]
        assert abs(est - getattr(true, name)) < 3 * se, name


@pytest.mark.parametrize("cls", [Vasicek, CIR])
def test_curve_fit_recovers_model_curve(cls):
    true = cls(0.4, 0.045, 0.02 if cls is Vasicek else 0.06)
    taus = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    cal = fit_to_curve(cls, taus, true.zero_yield(0.02, taus), r0=0.02, sigma=true.sigma)
    assert cal.rmse_bp < 0.01
    assert cal.model.kappa == pytest.approx(true.kappa, rel=1e-4)
    assert cal.model.theta == pytest.approx(true.theta, rel=1e-4)


CURVE = NSFit(0.04, -0.02, 0.015, 0.6, rmse=0.0)
HW = HullWhite(a=0.1, sigma=0.01, curve=CURVE)


def test_hull_white_reproduces_initial_curve():
    T = np.linspace(0.25, 30, 50)
    np.testing.assert_allclose(HW.bond_price(0.0, T, HW.r0), CURVE.discount(T), rtol=1e-12)


def test_hull_white_alpha_solves_theta_ode():
    # alpha(t) is the mean of r(t); it must satisfy d alpha/dt = theta(t) - a alpha(t)
    t, h = np.linspace(0.5, 25, 20), 1e-5
    dalpha = (HW.alpha(t + h) - HW.alpha(t - h)) / (2 * h)
    np.testing.assert_allclose(dalpha, HW.theta(t) - HW.a * HW.alpha(t), atol=1e-8)


def test_hull_white_mc_matches_curve():
    maturities = np.array([1.0, 5.0, 10.0, 20.0])
    times, paths = HW.simulate_exact(20.0, DT, 20_000, seed=9)
    prices, se = mc_bond_curve(times, paths, maturities)
    assert np.all(np.abs(prices - CURVE.discount(maturities)) < 3 * se + 1e-5)
