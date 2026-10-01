# Closed-form zero-coupon bond prices against Monte Carlo from the simulators,
# and the Hull-White curve-fitting guarantee.

test_that("vasicek monte carlo bond price matches the closed form within 3 standard errors", {
  r0 <- 0.03; a <- 0.5; b <- 0.04; sigma <- 0.015; Tt <- 3; dt <- 1 / 52
  paths <- simulate_vasicek(r0, a, b, sigma, Tt, dt, n_paths = 20000, seed = 10)
  mc <- mc_bond_price(paths)
  closed <- vasicek_bond_price(r0, a, b, sigma, Tt)
  expect_lt(abs(mc$price - closed), 3 * mc$se)
})

test_that("cir monte carlo bond price matches the closed form within 3 standard errors", {
  r0 <- 0.03; a <- 0.8; b <- 0.05; sigma <- 0.12; Tt <- 3; dt <- 1 / 52
  paths <- simulate_cir(r0, a, b, sigma, Tt, dt, n_paths = 20000, seed = 11, method = "exact")
  mc <- mc_bond_price(paths)
  closed <- cir_bond_price(r0, a, b, sigma, Tt)
  expect_lt(abs(mc$price - closed), 3 * mc$se)
})

test_that("vasicek and cir bond prices reject a non-positive time to maturity", {
  expect_error(vasicek_bond_price(0.03, 0.5, 0.04, 0.015, tau = 0))
  expect_error(cir_bond_price(0.03, 0.8, 0.05, 0.12, tau = -1))
})

test_that("hull-white reproduces the input discount curve exactly at t = 0", {
  maturities <- c(0.5, 1, 2, 3, 5, 7, 10)
  zero_rates <- c(0.03, 0.032, 0.035, 0.037, 0.04, 0.041, 0.042)
  curve <- discount_curve_from_zero_rates(maturities, zero_rates)
  for (Tt in c(1, 2, 5, 10)) {
    got <- hull_white_bond_price(r = 0.03, a = 0.5, sigma = 0.01, t = 0, Tt = Tt,
                                  discount_curve = curve)
    expect_equal(got, curve(Tt), tolerance = 1e-10)
  }
})

test_that("hull-white monte carlo bond price matches the curve-fitted closed form", {
  maturities <- c(0.5, 1, 2, 3, 5, 7, 10)
  zero_rates <- c(0.03, 0.032, 0.035, 0.037, 0.04, 0.041, 0.042)
  curve <- discount_curve_from_zero_rates(maturities, zero_rates)
  r0 <- forward_rate_from_discount(curve)(0)
  a <- 0.5; sigma <- 0.01; Tt <- 3; dt <- 1 / 52

  paths <- simulate_hull_white(r0, a, sigma, Tt, dt, curve, n_paths = 20000, seed = 20)
  mc <- mc_bond_price(paths)
  closed <- hull_white_bond_price(r0, a, sigma, t = 0, Tt = Tt, discount_curve = curve)
  expect_lt(abs(mc$price - closed), 3 * mc$se)
})

test_that("hull-white bond price rejects a maturity at or before the valuation time", {
  curve <- discount_curve_from_zero_rates(c(1, 2, 5), c(0.03, 0.035, 0.04))
  expect_error(hull_white_bond_price(0.03, 0.5, 0.01, t = 2, Tt = 2, discount_curve = curve))
})

test_that("mc_bond_price errors on a matrix without a times attribute", {
  expect_error(mc_bond_price(matrix(0.03, nrow = 10, ncol = 5)))
})
