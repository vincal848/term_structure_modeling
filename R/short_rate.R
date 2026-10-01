# Simulation of one-factor short rate models by Euler discretisation (Vasicek,
# Hull-White, Black-Karasinski) or exact transition density (CIR).
#
# Every simulate_* function returns an (n_steps + 1) x n_paths matrix, time down
# the rows, path across the columns, with the time grid attached as attr(,"times").
# All are vectorised over paths: the inner loop is over time steps only, and each
# step updates every path at once.

#' Validate the discretisation arguments shared by every simulate_* function.
#'
#' Centralised here so every model raises the same error for the same mistake,
#' and so "invalid inputs error" only needs to be pinned once per argument.
.check_sim_inputs <- function(Tt, dt, n_paths, sigma) {
  if (!is.numeric(Tt) || length(Tt) != 1 || Tt <= 0) {
    stop("Tt (horizon, in years) must be a single positive number")
  }
  if (!is.numeric(dt) || length(dt) != 1 || dt <= 0) {
    stop("dt (time step, in years) must be a single positive number")
  }
  if (!is.numeric(n_paths) || length(n_paths) != 1 || n_paths < 1) {
    stop("n_paths must be a single positive integer")
  }
  if (!is.numeric(sigma) || length(sigma) != 1 || sigma <= 0) {
    stop("sigma must be a single positive number")
  }
  n_steps <- Tt / dt
  # Tolerance wide enough to absorb a dt typed as a truncated decimal (e.g. "1/252"
  # entered as 0.003968254 on the command line, which is off from the exact
  # fraction by about 2e-6 once multiplied back up by n_steps), tight enough to
  # still catch a genuine mismatch like dt = 0.3 at Tt = 1.
  if (abs(n_steps - round(n_steps)) > 1e-4) {
    stop("Tt / dt must be (nearly) an integer number of steps")
  }
  as.integer(round(n_steps))
}

#' Attach a time grid to a simulated path matrix.
.with_times <- function(r, Tt, n_steps) {
  structure(r, times = seq(0, Tt, length.out = n_steps + 1L))
}

#' Simulate the Vasicek short rate: dr = a(b - r) dt + sigma dW.
#'
#' @param r0 initial short rate
#' @param a mean-reversion speed (> 0)
#' @param b long-run mean level
#' @param sigma volatility (> 0)
#' @param Tt horizon in years
#' @param dt time step in years; Tt / dt must be (nearly) an integer
#' @param n_paths number of independent paths to simulate
#' @param seed optional RNG seed, set once before simulation for reproducibility
#' @return (n_steps + 1) x n_paths matrix with attr(,"times")
simulate_vasicek <- function(r0, a, b, sigma, Tt, dt, n_paths = 1, seed = NULL) {
  if (a <= 0) stop("a (mean-reversion speed) must be positive")
  n_steps <- .check_sim_inputs(Tt, dt, n_paths, sigma)
  if (!is.null(seed)) set.seed(seed)

  r <- matrix(NA_real_, nrow = n_steps + 1L, ncol = n_paths)
  r[1, ] <- r0
  dW <- matrix(rnorm(n_steps * n_paths, sd = sqrt(dt)), nrow = n_steps, ncol = n_paths)
  for (i in seq_len(n_steps)) {
    r[i + 1, ] <- r[i, ] + a * (b - r[i, ]) * dt + sigma * dW[i, ]
  }
  .with_times(r, Tt, n_steps)
}

#' Simulate the CIR short rate: dr = a(b - r) dt + sigma sqrt(r) dW.
#'
#' The legacy implementation this replaces drew a Vasicek-style Gaussian step and
#' then set `short_rate[i] <- max(short_rate[i], 0)`. That clipping is biased: it
#' acts as a reflecting barrier at zero, which injects extra probability mass right
#' above zero on every step the Gaussian step would have gone negative, pushing the
#' simulated mean above the true CIR mean -- worst when the Feller condition
#' `2ab >= sigma^2` is violated and the process spends real time near the boundary.
#'
#' Two bias-free alternatives are offered instead:
#'
#' - `method = "exact"` (default): the CIR transition density is a scaled
#'   noncentral chi-square (Cox, Ingersoll & Ross, 1985), so sampling from it
#'   directly has no discretisation error at all and is non-negative by
#'   construction -- there is nothing to clip.
#' - `method = "full_truncation"`: an Euler scheme that uses the positive part of
#'   the current rate, `r+ = max(r, 0)`, inside the drift and diffusion
#'   coefficients for the *next* step, but never clips the simulated state itself.
#'   This is the full truncation scheme of Lord, Koekkoek & Van Dijk (2010), shown
#'   there to have the smallest bias among the common Euler fixes. It can still
#'   produce a transiently negative state on a single step when dt is large
#'   relative to sigma^2 / a near r = 0; the exact scheme has no such artifact and
#'   is the one `bond_prices.R` and the tests rely on for positivity.
#'
#' @inheritParams simulate_vasicek
#' @param b long-run mean level
#' @param method "exact" (default) or "full_truncation"
simulate_cir <- function(r0, a, b, sigma, Tt, dt, n_paths = 1, seed = NULL,
                          method = c("exact", "full_truncation")) {
  method <- match.arg(method)
  if (a <= 0) stop("a (mean-reversion speed) must be positive")
  if (r0 < 0) stop("r0 must be non-negative for CIR")
  n_steps <- .check_sim_inputs(Tt, dt, n_paths, sigma)
  if (!is.null(seed)) set.seed(seed)

  r <- matrix(NA_real_, nrow = n_steps + 1L, ncol = n_paths)
  r[1, ] <- r0

  if (method == "exact") {
    df <- 4 * a * b / sigma^2
    scale <- sigma^2 * (1 - exp(-a * dt)) / (4 * a)
    for (i in seq_len(n_steps)) {
      ncp <- 4 * a * r[i, ] * exp(-a * dt) / (sigma^2 * (1 - exp(-a * dt)))
      r[i + 1, ] <- scale * rchisq(n_paths, df = df, ncp = ncp)
    }
  } else {
    dW <- matrix(rnorm(n_steps * n_paths, sd = sqrt(dt)), nrow = n_steps, ncol = n_paths)
    for (i in seq_len(n_steps)) {
      r_pos <- pmax(r[i, ], 0)
      r[i + 1, ] <- r[i, ] + a * (b - r_pos) * dt + sigma * sqrt(r_pos) * dW[i, ]
    }
  }
  .with_times(r, Tt, n_steps)
}

#' Is the Feller condition `2ab >= sigma^2` satisfied?
#'
#' When it is, the CIR process a.s. never reaches zero. When it is not, zero is
#' attainable and the noncentral chi-square degrees of freedom `4ab/sigma^2` drop
#' below 1 -- the exact scheme still samples correctly, but a positivity test on a
#' CIR path with violated Feller is a weaker claim than one that stays strictly
#' positive the whole time.
feller_condition <- function(a, b, sigma) 2 * a * b >= sigma^2

#' Simulate Hull-White with a time-dependent theta(t) fitted to today's curve:
#' dr = (theta(t) - a r) dt + sigma dW.
#'
#' theta(t) is pinned down by requiring the model reproduce the instantaneous
#' forward rate f(0,t) implied by `discount_curve` (Brigo & Mercurio, 2006, eq.
#' 3.34): `theta(t) = df/dt(0,t) + a f(0,t) + sigma^2/(2a) (1 - exp(-2 a t))`.
#' f and its derivative are obtained by numerically differentiating
#' `log(discount_curve(t))`, since the curve is built from a handful of market
#' points via `discount_curve_from_zero_rates()` (see R/bond_prices.R) rather than
#' handed to us in closed form.
#'
#' @inheritParams simulate_vasicek
#' @param discount_curve function(T) -> P(0, T), vectorised, from
#'   `discount_curve_from_zero_rates()`
#' @param h step used for the central-difference derivatives of the forward curve
simulate_hull_white <- function(r0, a, sigma, Tt, dt, discount_curve, n_paths = 1,
                                 seed = NULL, h = 1e-4) {
  if (a <= 0) stop("a (mean-reversion speed) must be positive")
  n_steps <- .check_sim_inputs(Tt, dt, n_paths, sigma)
  if (!is.null(seed)) set.seed(seed)

  forward_rate <- forward_rate_from_discount(discount_curve, h = h)
  times <- seq(0, Tt, length.out = n_steps + 1L)
  f0 <- forward_rate(times)
  dfdt <- .deriv0(forward_rate, times, h)
  theta <- dfdt + a * f0 + sigma^2 / (2 * a) * (1 - exp(-2 * a * times))

  r <- matrix(NA_real_, nrow = n_steps + 1L, ncol = n_paths)
  r[1, ] <- r0
  dW <- matrix(rnorm(n_steps * n_paths, sd = sqrt(dt)), nrow = n_steps, ncol = n_paths)
  for (i in seq_len(n_steps)) {
    r[i + 1, ] <- r[i, ] + (theta[i] - a * r[i, ]) * dt + sigma * dW[i, ]
  }
  .with_times(r, Tt, n_steps)
}

#' Simulate Black-Karasinski in log-rate: d(log r) = a(log theta - log r) dt + sigma dW.
#'
#' The short rate itself is `exp()` of an Ornstein-Uhlenbeck process, so it is
#' strictly positive for every path and every step by construction -- there is no
#' clipping to get right here, unlike CIR. `theta` and `sigma` are taken constant,
#' matching the one-factor BK model this repository compares against the others.
#'
#' @inheritParams simulate_vasicek
#' @param theta long-run mean level of the short rate (not its log)
simulate_bk <- function(r0, a, theta, sigma, Tt, dt, n_paths = 1, seed = NULL) {
  if (a <= 0) stop("a (mean-reversion speed) must be positive")
  if (r0 <= 0 || theta <= 0) stop("r0 and theta must be positive for Black-Karasinski")
  n_steps <- .check_sim_inputs(Tt, dt, n_paths, sigma)
  if (!is.null(seed)) set.seed(seed)

  x <- matrix(NA_real_, nrow = n_steps + 1L, ncol = n_paths)
  x[1, ] <- log(r0)
  log_theta <- log(theta)
  dW <- matrix(rnorm(n_steps * n_paths, sd = sqrt(dt)), nrow = n_steps, ncol = n_paths)
  for (i in seq_len(n_steps)) {
    x[i + 1, ] <- x[i, ] + a * (log_theta - x[i, ]) * dt + sigma * dW[i, ]
  }
  .with_times(exp(x), Tt, n_steps)
}
