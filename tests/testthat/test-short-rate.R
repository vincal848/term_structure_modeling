# Short rate simulators against their closed-form moments, positivity, and
# input validation.

test_that("vasicek monte carlo mean and variance match the closed form within standard error bands", {
  r0 <- 0.03; a <- 0.5; b <- 0.04; sigma <- 0.015; Tt <- 2; dt <- 1 / 252
  paths <- simulate_vasicek(r0, a, b, sigma, Tt, dt, n_paths = 20000, seed = 1)
  rT <- paths[nrow(paths), ]

  mean_true <- r0 * exp(-a * Tt) + b * (1 - exp(-a * Tt))
  var_true <- sigma^2 / (2 * a) * (1 - exp(-2 * a * Tt))
  se_mean <- sqrt(var_true / length(rT))
  se_var <- var_true * sqrt(2 / (length(rT) - 1))

  expect_lt(abs(mean(rT) - mean_true), 4 * se_mean)
  expect_lt(abs(stats::var(rT) - var_true), 4 * se_var)
})

test_that("cir exact scheme never produces a negative rate, under and over the feller condition", {
  under <- simulate_cir(0.03, 0.1, 0.01, 0.5, Tt = 2, dt = 1 / 252, n_paths = 2000,
                         seed = 2, method = "exact")
  over <- simulate_cir(0.03, 0.8, 0.05, 0.1, Tt = 2, dt = 1 / 252, n_paths = 2000,
                        seed = 3, method = "exact")
  expect_false(feller_condition(0.1, 0.01, 0.5))
  expect_true(feller_condition(0.8, 0.05, 0.1))
  expect_true(all(under >= 0))
  expect_true(all(over >= 0))
})

test_that("cir full truncation euler also stays non-negative at a moderate time step", {
  paths <- simulate_cir(0.03, 0.8, 0.05, 0.1, Tt = 2, dt = 1 / 252, n_paths = 2000,
                         seed = 4, method = "full_truncation")
  expect_true(all(paths >= 0))
})

test_that("black-karasinski is strictly positive for every path and step", {
  paths <- simulate_bk(0.03, 0.5, 0.04, 0.25, Tt = 2, dt = 1 / 252, n_paths = 2000, seed = 5)
  expect_true(all(paths > 0))
})

test_that("simulate_vasicek rejects a horizon that is not an integer number of steps", {
  expect_error(simulate_vasicek(0.03, 0.5, 0.04, 0.015, Tt = 1, dt = 0.3, n_paths = 1))
})

test_that("simulate_vasicek rejects a non-positive mean reversion speed", {
  expect_error(simulate_vasicek(0.03, 0, 0.04, 0.015, Tt = 1, dt = 1 / 252, n_paths = 1))
})

test_that("simulate_vasicek rejects a non-positive sigma", {
  expect_error(simulate_vasicek(0.03, 0.5, 0.04, 0, Tt = 1, dt = 1 / 252, n_paths = 1))
})

test_that("simulate_cir rejects a negative initial rate", {
  expect_error(simulate_cir(-0.01, 0.5, 0.04, 0.1, Tt = 1, dt = 1 / 252, n_paths = 1))
})

test_that("simulate_bk rejects a non-positive initial rate or theta", {
  expect_error(simulate_bk(-0.01, 0.5, 0.04, 0.1, Tt = 1, dt = 1 / 252, n_paths = 1))
  expect_error(simulate_bk(0.03, 0.5, -0.04, 0.1, Tt = 1, dt = 1 / 252, n_paths = 1))
})
