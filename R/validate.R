# Regenerates docs/VALIDATION.md and docs/img/*.png from the code in R/. Called
# by `Rscript run.R validate`. Every number in README.md and docs/VALIDATION.md
# comes from running `run_validate()`, so the documentation cannot drift away
# from the code.

VASICEK_PARAMS <- list(r0 = 0.03, a = 0.5, b = 0.04, sigma = 0.015)
CIR_PARAMS <- list(r0 = 0.03, a = 0.8, b = 0.05, sigma = 0.12)
BK_PARAMS <- list(r0 = 0.03, a = 0.5, theta = 0.04, sigma = 0.25)
HW_PARAMS <- list(a = 0.5, sigma = 0.01)
NS_TRUE <- list(beta0 = 0.04, beta1 = -0.015, beta2 = 0.01, lambda = 0.6)

.section_moments <- function(con) {
  writeLines(c("## Short rate moments: simulation vs closed form", ""), con)
  writeLines(c(
    "20,000 paths, 2-year horizon, dt = 1/252. Vasicek has an exact Gaussian",
    "transition so the Euler scheme is exact on this grid; CIR uses the exact",
    "noncentral chi-square scheme, so neither comparison has discretisation error",
    "of its own -- only Monte Carlo sampling error, which is what the SE column is.",
    ""
  ), con)
  writeLines(c("| Model | Quantity | Simulated | Closed form | diff | SE |",
               "|---|---|---|---|---|---|"), con)

  Tt <- 2; dt <- 1 / 252; n_paths <- 20000
  with(VASICEK_PARAMS, {
    paths <- simulate_vasicek(r0, a, b, sigma, Tt, dt, n_paths = n_paths, seed = 101)
    rT <- paths[nrow(paths), ]
    mean_true <- r0 * exp(-a * Tt) + b * (1 - exp(-a * Tt))
    var_true <- sigma^2 / (2 * a) * (1 - exp(-2 * a * Tt))
    se_mean <- sqrt(var_true / n_paths)
    writeLines(sprintf("| Vasicek | E[r_T] | %.6f | %.6f | %.2e | %.2e |",
                        mean(rT), mean_true, mean(rT) - mean_true, se_mean), con)
    writeLines(sprintf("| Vasicek | Var[r_T] | %.6e | %.6e | %.2e | - |",
                        stats::var(rT), var_true, stats::var(rT) - var_true), con)
  })
  with(CIR_PARAMS, {
    paths <- simulate_cir(r0, a, b, sigma, Tt, dt, n_paths = n_paths, seed = 102, method = "exact")
    rT <- paths[nrow(paths), ]
    mean_true <- r0 * exp(-a * Tt) + b * (1 - exp(-a * Tt))
    var_true <- r0 * sigma^2 / a * (exp(-a * Tt) - exp(-2 * a * Tt)) +
      b * sigma^2 / (2 * a) * (1 - exp(-a * Tt))^2
    se_mean <- sqrt(var_true / n_paths)
    writeLines(sprintf("| CIR | E[r_T] | %.6f | %.6f | %.2e | %.2e |",
                        mean(rT), mean_true, mean(rT) - mean_true, se_mean), con)
    writeLines(sprintf("| CIR | Var[r_T] | %.6e | %.6e | %.2e | - |",
                        stats::var(rT), var_true, stats::var(rT) - var_true), con)
    writeLines(sprintf("\nCIR Feller condition `2ab >= sigma^2`: %.4f >= %.4f -- %s.\n",
                        2 * a * b, sigma^2, if (feller_condition(a, b, sigma)) "satisfied" else "violated"), con)
  })
  writeLines(sprintf(
    "\nBlack-Karasinski (a=%.2f, theta=%.2f, sigma=%.2f): minimum simulated rate over %d paths is %.6f (strictly positive by construction, since the state is exp() of a Gaussian).\n",
    BK_PARAMS$a, BK_PARAMS$theta, BK_PARAMS$sigma, n_paths,
    min(simulate_bk(BK_PARAMS$r0, BK_PARAMS$a, BK_PARAMS$theta, BK_PARAMS$sigma, Tt, dt, n_paths = n_paths, seed = 103))
  ), con)
}

.section_bonds <- function(con) {
  writeLines(c("## Zero-coupon bond prices: Monte Carlo vs closed form", ""), con)
  writeLines(c(
    "Monte Carlo price is `mean(exp(-integral r dt))` along each path (trapezoid",
    "rule), 5,000 paths, dt = 1/52 so the 10-year maturity stays cheap to simulate.",
    ""
  ), con)
  writeLines(c("| Model | Maturity | MC price | Closed form | diff | 3 x SE |",
               "|---|---|---|---|---|---|"), con)
  maturities <- c(0.5, 1, 2, 3, 5, 7, 10)
  dt <- 1 / 52; n_paths <- 5000
  rows <- list()
  for (Tt in maturities) {
    with(VASICEK_PARAMS, {
      paths <- simulate_vasicek(r0, a, b, sigma, Tt, dt, n_paths = n_paths, seed = 201)
      mc <- mc_bond_price(paths)
      closed <- vasicek_bond_price(r0, a, b, sigma, Tt)
      writeLines(sprintf("| Vasicek | %.1fy | %.6f | %.6f | %.2e | %.2e |",
                          Tt, mc$price, closed, mc$price - closed, 3 * mc$se), con)
      rows[[length(rows) + 1]] <<- data.frame(model = "Vasicek", maturity = Tt, mc = mc$price, closed = closed)
    })
    with(CIR_PARAMS, {
      paths <- simulate_cir(r0, a, b, sigma, Tt, dt, n_paths = n_paths, seed = 202, method = "exact")
      mc <- mc_bond_price(paths)
      closed <- cir_bond_price(r0, a, b, sigma, Tt)
      writeLines(sprintf("| CIR | %.1fy | %.6f | %.6f | %.2e | %.2e |",
                          Tt, mc$price, closed, mc$price - closed, 3 * mc$se), con)
      rows[[length(rows) + 1]] <<- data.frame(model = "CIR", maturity = Tt, mc = mc$price, closed = closed)
    })
  }
  writeLines("", con)
  do.call(rbind, rows)
}

.section_hull_white <- function(con, ns_fred) {
  writeLines(c("## Hull-White: fitting today's curve", ""), con)
  curve <- discount_curve_from_zero_rates(ns_fred$maturity, ns_fred$fitted)
  f0 <- forward_rate_from_discount(curve)(0)
  writeLines(c(
    "The discount curve is built from the Nelson-Siegel fit to the cached FRED",
    "curve below, so theta(t) in the Hull-White SDE is time-dependent and the",
    "model reproduces that curve by construction rather than by calibration error",
    "being small.", ""
  ), con)
  writeLines(c("| Maturity | P(0,T) from curve | Hull-White closed form | diff |",
               "|---|---|---|---|"), con)
  for (Tt in c(1, 2, 5, 10, 20)) {
    hw <- hull_white_bond_price(f0, HW_PARAMS$a, HW_PARAMS$sigma, t = 0, Tt = Tt, discount_curve = curve)
    writeLines(sprintf("| %gy | %.6f | %.6f | %.2e |", Tt, curve(Tt), hw, hw - curve(Tt)), con)
  }
  Tt <- 5; dt <- 1 / 52; n_paths <- 5000
  paths <- simulate_hull_white(f0, HW_PARAMS$a, HW_PARAMS$sigma, Tt, dt, curve, n_paths = n_paths, seed = 301)
  mc <- mc_bond_price(paths)
  closed <- hull_white_bond_price(f0, HW_PARAMS$a, HW_PARAMS$sigma, t = 0, Tt = Tt, discount_curve = curve)
  writeLines(sprintf(
    "\nSimulated short rate, 5y bond, %d paths: MC price %.6f vs closed form %.6f, diff %.2e, 3 x SE %.2e.\n",
    n_paths, mc$price, closed, mc$price - closed, 3 * mc$se
  ), con)
}

.section_nelson_siegel <- function(con) {
  writeLines(c("## Nelson-Siegel curve fitting", ""), con)
  maturity <- c(0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)
  yield <- nelson_siegel(maturity, NS_TRUE$beta0, NS_TRUE$beta1, NS_TRUE$beta2, NS_TRUE$lambda)
  synth <- fit_ns(maturity, yield)
  writeLines(c(
    "**Synthetic, noiseless data** -- recovering known parameters:", "",
    "| Parameter | True | Fitted |", "|---|---|---|"
  ), con)
  for (p in names(NS_TRUE)) {
    writeLines(sprintf("| %s | %.6f | %.6f |", p, NS_TRUE[[p]], synth$params[[p]]), con)
  }
  writeLines(sprintf("\nRMSE: %.2e (engine: %s)\n", synth$rmse, synth$engine), con)

  fred <- load_treasury_curve()
  fit <- fit_ns(fred$maturity_years, fred$yield_pct / 100)
  writeLines(c(sprintf("**Cached FRED Treasury curve, %s**:", fred$date[1]), "",
               "| Parameter | Fitted |", "|---|---|")
, con)
  for (p in names(fit$params)) {
    writeLines(sprintf("| %s | %.6f |", p, fit$params[[p]]), con)
  }
  writeLines(sprintf("\nRMSE: %.4f (%.1f bp), engine: %s\n", fit$rmse, fit$rmse * 1e4, fit$engine), con)
  list(maturity = fred$maturity_years, yield = fred$yield_pct / 100, fitted = fit$fitted,
       params = fit$params, date = fred$date[1])
}

.figures <- function(img_dir, bond_rows, ns_fred) {
  dir.create(img_dir, recursive = TRUE, showWarnings = FALSE)

  tau <- seq(0.01, 30, length.out = 300)
  fitted_curve <- nelson_siegel(tau, ns_fred$params$beta0, ns_fred$params$beta1,
                                 ns_fred$params$beta2, ns_fred$params$lambda)
  p1 <- ggplot2::ggplot() +
    ggplot2::geom_line(data = data.frame(tau = tau, yield = fitted_curve * 100),
                        ggplot2::aes(x = tau, y = yield), color = "steelblue", linewidth = 1) +
    ggplot2::geom_point(data = data.frame(maturity = ns_fred$maturity, yield = ns_fred$yield * 100),
                         ggplot2::aes(x = maturity, y = yield), color = "firebrick", size = 2) +
    ggplot2::labs(title = sprintf("Nelson-Siegel fit to the Treasury curve, %s", ns_fred$date),
                  x = "maturity (years)", y = "yield (%)") +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(img_dir, "ns_fit.png"), p1, width = 6.5, height = 4, dpi = 130)

  p2 <- ggplot2::ggplot(bond_rows, ggplot2::aes(x = maturity, y = closed, color = model)) +
    ggplot2::geom_line() +
    ggplot2::geom_point(ggplot2::aes(y = mc), shape = 1, size = 2.5) +
    ggplot2::labs(title = "Zero-coupon bond price: closed form (line) vs Monte Carlo (points)",
                  x = "maturity (years)", y = "P(0,T)", color = "model") +
    ggplot2::theme_minimal()
  ggplot2::ggsave(file.path(img_dir, "bond_prices.png"), p2, width = 6.5, height = 4, dpi = 130)
}

#' Regenerate docs/VALIDATION.md and docs/img/*.png.
#'
#' @param root repository root; defaults to the result of `.find_repo_root()`
run_validate <- function(root = .find_repo_root()) {
  docs_dir <- file.path(root, "docs")
  img_dir <- file.path(docs_dir, "img")
  dir.create(docs_dir, showWarnings = FALSE)
  md_path <- file.path(docs_dir, "VALIDATION.md")

  con <- file(md_path, open = "wt")
  on.exit(close(con))
  writeLines(c("# Validation", "", "Generated by `run.R validate`. Do not edit by hand.", ""), con)

  .section_moments(con)
  bond_rows <- .section_bonds(con)
  ns_fred <- .section_nelson_siegel(con)
  .section_hull_white(con, ns_fred)

  .figures(img_dir, bond_rows, ns_fred)
  cat(sprintf("wrote %s\n", md_path))
  cat(sprintf("wrote figures to %s\n", img_dir))
  invisible(NULL)
}
