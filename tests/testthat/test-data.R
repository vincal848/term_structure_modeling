# The cached Treasury curve, loaded with no network access.

test_that("load_treasury_curve returns the committed cache without touching the network", {
  curve <- load_treasury_curve()
  expect_true(all(c("date", "series", "maturity_years", "yield_pct") %in% names(curve)))
  expect_gte(nrow(curve), 10)
  expect_true(all(diff(curve$maturity_years) > 0))
})

test_that("load_treasury_curve errors on a missing cache path", {
  expect_error(load_treasury_curve(path = tempfile(fileext = ".csv")))
})

test_that("treasury_curve_as_zero_rates converts percent to decimal", {
  curve <- load_treasury_curve()
  z <- treasury_curve_as_zero_rates(curve)
  expect_equal(z$zero_rate, curve$yield_pct / 100)
})
