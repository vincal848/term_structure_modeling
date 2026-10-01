# Nelson-Siegel (and optional Svensson) yield curve fitting.

#' Nelson-Siegel yield curve, vectorised over maturity.
#'
#' `nelson_siegel(tau, b0, b1, b2, lambda)`:
#' `beta0 + beta1 * (1 - exp(-lambda tau)) / (lambda tau) + beta2 * ((1 - exp(-lambda tau)) / (lambda tau) - exp(-lambda tau))`
#'
#' Limits (both by L'Hopital on `(1 - exp(-x)) / x` as `x -> 0`):
#' `tau -> 0`: yield -> `beta0 + beta1` (the instantaneous short rate).
#' `tau -> Inf`: yield -> `beta0` (the long-run level).
#' `tau = 0` is handled by that limit directly rather than by dividing by zero.
#'
#' @param tau maturity in years, >= 0, vector or scalar
#' @param beta0 long-run level
#' @param beta1 short-term component
#' @param beta2 medium-term (hump/trough) component
#' @param lambda decay rate (> 0); controls where the hump sits
nelson_siegel <- function(tau, beta0, beta1, beta2, lambda) {
  if (any(tau < 0)) stop("tau must be non-negative")
  if (lambda <= 0) stop("lambda must be positive")
  x <- lambda * tau
  # (1 - exp(-x)) / x -> 1 as x -> 0; fill that in rather than dividing by zero.
  decay <- ifelse(x == 0, 1, (1 - exp(-x)) / x)
  beta0 + beta1 * decay + beta2 * (decay - exp(-x))
}

#' Svensson extension: Nelson-Siegel plus a second hump term.
#'
#' Adds `beta3 * ((1 - exp(-lambda2 tau)) / (lambda2 tau) - exp(-lambda2 tau))`,
#' which vanishes at both `tau -> 0` and `tau -> Inf`, so the same limits as
#' `nelson_siegel()` hold.
#'
#' @inheritParams nelson_siegel
#' @param beta3 second hump/trough component
#' @param lambda2 decay rate for the second hump (> 0)
nelson_siegel_svensson <- function(tau, beta0, beta1, beta2, beta3, lambda, lambda2) {
  if (lambda2 <= 0) stop("lambda2 must be positive")
  x2 <- lambda2 * tau
  decay2 <- ifelse(x2 == 0, 1, (1 - exp(-x2)) / x2)
  nelson_siegel(tau, beta0, beta1, beta2, lambda) + beta3 * (decay2 - exp(-x2))
}

#' Fit Nelson-Siegel to maturity/yield pairs by nonlinear least squares.
#'
#' Tries `minpack.lm::nlsLM` first (Levenberg-Marquardt, the more robust choice
#' for this model because the hump term makes the surface flat in `lambda` when
#' the start value is far off); falls back to `optim()` on the sum of squared
#' errors if `nlsLM` does not converge.
#'
#' @param maturity maturities in years
#' @param yield observed yields, same units as the fitted curve will be returned in
#' @param start named list of starting values for beta0, beta1, beta2, lambda;
#'   defaults to the long end as beta0, the short-minus-long spread as beta1,
#'   zero curvature, and a lambda putting the hump near the middle of the data
#' @return list(params, fitted, resid, rmse, model) -- `model` is the nls object
#'   (engine = "nlsLM") or the optim result (engine = "optim")
fit_ns <- function(maturity, yield, start = NULL) {
  if (length(maturity) != length(yield)) stop("maturity and yield must have the same length")
  if (length(maturity) < 4) stop("need at least 4 points to fit 4 parameters")

  if (is.null(start)) {
    start <- list(
      beta0 = tail(yield, 1),
      beta1 = yield[1] - tail(yield, 1),
      beta2 = 0,
      lambda = 1 / stats::median(maturity)
    )
  }

  fit <- tryCatch(
    minpack.lm::nlsLM(
      yield ~ nelson_siegel(maturity, beta0, beta1, beta2, lambda),
      start = start
    ),
    error = function(e) NULL
  )

  if (!is.null(fit)) {
    params <- as.list(stats::coef(fit))
    fitted <- stats::predict(fit)
    engine <- "nlsLM"
    model <- fit
  } else {
    sse <- function(p) {
      sum((yield - nelson_siegel(maturity, p[1], p[2], p[3], abs(p[4])))^2)
    }
    opt <- stats::optim(
      c(start$beta0, start$beta1, start$beta2, start$lambda),
      sse, method = "Nelder-Mead"
    )
    params <- list(beta0 = opt$par[1], beta1 = opt$par[2], beta2 = opt$par[3],
                    lambda = abs(opt$par[4]))
    fitted <- nelson_siegel(maturity, params$beta0, params$beta1, params$beta2, params$lambda)
    engine <- "optim"
    model <- opt
  }

  resid <- yield - fitted
  list(params = params, fitted = fitted, resid = resid,
       rmse = sqrt(mean(resid^2)), engine = engine, model = model)
}
