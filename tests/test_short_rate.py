import numpy as np
import pytest

from termstructure.affine import CIR, Vasicek
from termstructure.short_rate import (
    OneFactorModel,
    black_karasinski,
    mc_bond_curve,
    mc_bond_price,
)

VAS = Vasicek(kappa=0.3, theta=0.04, sigma=0.01)
CIR_MODEL = CIR(kappa=0.3, theta=0.04, sigma=0.05)


def test_presets_match_general_form():
    g = VAS.general()
    r = np.array([0.01, 0.05])
    np.testing.assert_allclose(g.drift(r), 0.3 * (0.04 - r))
    np.testing.assert_allclose(g.diffusion(r), 0.01)
    c = CIR_MODEL.general()
    np.testing.assert_allclose(c.diffusion(r), 0.05 * np.sqrt(r))
    bk = black_karasinski(0.1, np.log(0.05), 0.2)
    # d log r drift implied by the r-form coefficients: drift/r - sigma^2/2
    np.testing.assert_allclose(bk.drift(r) / r - 0.5 * 0.2**2, 0.1 * (np.log(0.05) - np.log(r)),
                               atol=1e-14)


def test_bond_price_boundaries():
    for m in (VAS, CIR_MODEL):
        assert m.bond_price(0.03, 1e-12) == pytest.approx(1.0)
        assert m.zero_yield(0.03, 1e-6) == pytest.approx(0.03, abs=1e-6)
        assert m.zero_yield(0.03, 500.0) == pytest.approx(m.long_yield, abs=2e-4)


def test_vasicek_exact_step_moments():
    _, paths = VAS.simulate_exact(0.02, T=2.0, dt=2.0, n_paths=400_000, seed=1)
    mean, var = VAS.transition_moments(0.02, 2.0)
    assert paths[-1].mean() == pytest.approx(mean, abs=4 * np.sqrt(var / 400_000))
    assert paths[-1].var() == pytest.approx(var, rel=0.01)


@pytest.mark.parametrize("model", [VAS, CIR_MODEL], ids=["vasicek", "cir"])
def test_mc_bond_prices_match_closed_form(model):
    r0, maturities = 0.02, np.array([1.0, 5.0, 10.0])
    if isinstance(model, Vasicek):
        times, paths = model.simulate_exact(r0, 10.0, 1 / 52, 20_000, seed=7)
        prices, se = mc_bond_curve(times, paths, maturities)
    else:
        times, paths = model.simulate_exact(r0, 10.0, 1 / 52, 20_000, seed=7)
        prices, se = mc_bond_curve(times, paths, maturities, antithetic=False)
    exact = model.bond_price(r0, maturities)
    assert np.all(np.abs(prices - exact) < 3 * se + 1e-5)


def test_cir_exact_and_full_truncation_stay_nonnegative():
    no_feller = CIR(kappa=0.3, theta=0.04, sigma=0.25)
    assert not no_feller.feller
    _, exact = no_feller.simulate_exact(0.01, 5.0, 1 / 52, 2_000, seed=3)
    _, ft = no_feller.general().simulate(0.01, 5.0, 1 / 52, 2_000, seed=3, scheme="full_truncation")
    assert exact.min() >= 0.0 and ft.min() >= 0.0


def test_log_euler_keeps_lognormal_rates_positive():
    _, paths = black_karasinski(0.1, np.log(0.05), 0.5).simulate(
        0.03, 5.0, 1 / 52, 2_000, seed=0, scheme="log_euler")
    assert paths.min() > 0.0


def test_log_euler_rejects_non_lognormal():
    with pytest.raises(ValueError):
        VAS.general().simulate(0.03, 1.0, 0.1, 10, scheme="log_euler")


def test_antithetic_standard_error_uses_pairs():
    times = np.linspace(0, 1, 3)
    paths = np.tile(np.array([[0.0], [0.01], [0.02]]), (1, 4))  # identical paths -> zero variance
    price, se = mc_bond_price(times, paths)
    assert price == pytest.approx(np.exp(-0.01)) and se == 0.0


def test_euler_floor_matches_original_r_scheme():
    m = OneFactorModel(K0=0.0, K1=0.0, H0=1.0, v=1.0)  # pure Brownian motion, floored at zero
    _, paths = m.simulate(0.0, 1.0, 0.01, 1_000, seed=0, scheme="euler_floor")
    assert paths.min() == 0.0 and paths.max() > 0.0


def test_plain_euler_allows_negative_vasicek_rates():
    _, paths = Vasicek(kappa=0.1, theta=0.0, sigma=0.05).general().simulate(0.0, 1.0, 0.01, 1_000, seed=0)
    assert paths.min() < 0.0
