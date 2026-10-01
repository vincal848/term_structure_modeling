# Nelson-Siegel limits, parameter recovery, and the fit against the cached
# FRED curve.

test_that("nelson_siegel's long-maturity limit is beta0", {
  v <- nelson_siegel(1e6, beta0 = 0.04, beta1 = -0.01, beta2 = 0.02, lambda = 0.5)
  expect_lt(abs(v - 0.04), 1e-6)
})

test_that("nelson_siegel's zero-maturity limit is beta0 + beta1", {
  v0 <- nelson_siegel(0, beta0 = 0.04, beta1 = -0.01, beta2 = 0.02, lambda = 0.5)
  expect_equal(v0, 0.03)
  v_small <- nelson_siegel(1e-6, beta0 = 0.04, beta1 = -0.01, beta2 = 0.02, lambda = 0.5)
  expect_lt(abs(v_small - 0.03), 1e-6)
})

test_that("fit_ns recovers known parameters from synthetic noiseless data", {
  true <- list(beta0 = 0.04, beta1 = -0.015, beta2 = 0.01, lambda = 0.6)
  maturity <- c(0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)
  yield <- nelson_siegel(maturity, true$beta0, true$beta1, true$beta2, true$lambda)

  fit <- fit_ns(maturity, yield)
  expect_equal(fit$params$beta0, true$beta0, tolerance = 1e-4)
  expect_equal(fit$params$beta1, true$beta1, tolerance = 1e-4)
  expect_equal(fit$params$beta2, true$beta2, tolerance = 1e-4)
  expect_equal(fit$params$lambda, true$lambda, tolerance = 1e-4)
  expect_lt(fit$rmse, 1e-6)
})

test_that("fit_ns has small rmse on the cached FRED curve", {
  curve <- load_treasury_curve()
  fit <- fit_ns(curve$maturity_years, curve$yield_pct / 100)
  expect_lt(fit$rmse, 0.01)
})

test_that("nelson_siegel rejects a non-positive lambda", {
  expect_error(nelson_siegel(1, beta0 = 0.04, beta1 = -0.01, beta2 = 0.02, lambda = 0))
})

test_that("nelson_siegel rejects a negative maturity", {
  expect_error(nelson_siegel(-1, beta0 = 0.04, beta1 = -0.01, beta2 = 0.02, lambda = 0.5))
})

test_that("fit_ns rejects mismatched maturity and yield lengths", {
  expect_error(fit_ns(c(1, 2, 3), c(0.01, 0.02)))
})

test_that("fit_ns rejects fewer than four points", {
  expect_error(fit_ns(c(1, 2, 3), c(0.01, 0.02, 0.03)))
})
