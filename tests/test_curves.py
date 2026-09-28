import numpy as np
import pytest

from termstructure.curves import (
    NSFit,
    bootstrap_zeros,
    fit_nelson_siegel,
    nelson_siegel,
    par_from_zeros,
)


def test_bootstrap_round_trip():
    grid = np.arange(0.5, 30.01, 0.5)
    zeros = nelson_siegel(grid, 0.045, -0.02, 0.01, 0.6)
    par = par_from_zeros(grid, zeros)
    taus, recovered = bootstrap_zeros(grid, par)
    np.testing.assert_allclose(taus, grid)
    np.testing.assert_allclose(recovered, zeros, atol=1e-8)


def test_flat_par_curve_gives_flat_zeros():
    maturities = [1, 2, 5, 10, 30]
    taus, zeros = bootstrap_zeros(maturities, [0.04] * 5)
    # 4% semiannual par everywhere <=> continuous zero 2*ln(1.02) everywhere
    np.testing.assert_allclose(zeros, 2 * np.log(1.02), atol=1e-12)


def test_sub_coupon_quotes_are_zero_rates():
    taus, zeros = bootstrap_zeros([1 / 12, 0.25, 1, 2], [0.05, 0.05, 0.05, 0.05])
    assert taus[0] == pytest.approx(1 / 12)
    assert zeros[0] == pytest.approx(2 * np.log(1.025))


def test_nelson_siegel_recovers_parameters():
    true = (0.04, -0.015, 0.02, 0.7)
    taus = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    fit = fit_nelson_siegel(taus, nelson_siegel(taus, *true))
    np.testing.assert_allclose(fit.params, true, rtol=1e-5)
    assert fit.rmse < 1e-10


def test_ns_short_end_limit():
    # As tau -> 0 the NS zero rate tends to b0 + b1 (the instantaneous short rate)
    assert nelson_siegel(0.0, 0.04, -0.01, 0.02, 0.5) == pytest.approx(0.03)


def test_ns_forward_is_derivative_of_tau_times_zero():
    fit = NSFit(0.04, -0.015, 0.02, 0.7, rmse=0.0)
    tau, h = np.linspace(0.5, 20, 15), 1e-6
    numeric = ((tau + h) * fit.zero(tau + h) - (tau - h) * fit.zero(tau - h)) / (2 * h)
    np.testing.assert_allclose(fit.forward(tau), numeric, atol=1e-8)
    numeric_slope = (fit.forward(tau + h) - fit.forward(tau - h)) / (2 * h)
    np.testing.assert_allclose(fit.forward_slope(tau), numeric_slope, atol=1e-7)


def test_discount_matches_zero():
    fit = NSFit(0.04, -0.015, 0.02, 0.7, rmse=0.0)
    tau = np.array([1.0, 5.0, 10.0])
    np.testing.assert_allclose(fit.discount(tau), np.exp(-fit.zero(tau) * tau))
