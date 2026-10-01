# Zero-coupon bond prices: closed form for the affine models (Vasicek, CIR),
# exact-by-construction for Hull-White, and a Monte Carlo estimator built from
# any simulate_* path matrix in R/short_rate.R so the closed forms can be checked
# against the simulations that are supposed to agree with them.

#' Vasicek zero-coupon bond price, P(t,T) = A(t,T) exp(-B(t,T) r).
#'
#' Standard affine solution of the Vasicek PDE (Brigo & Mercurio, 2006, eq. 3.9).
#'
#' @param r short rate at time t
#' @param a mean-reversion speed
#' @param b long-run mean level
#' @param sigma volatility
#' @param tau time to maturity, T - t, in years (> 0)
vasicek_bond_price <- function(r, a, b, sigma, tau) {
  if (any(tau <= 0)) stop("tau (time to maturity) must be positive")
  B <- (1 - exp(-a * tau)) / a
  A <- exp((b - sigma^2 / (2 * a^2)) * (B - tau) - sigma^2 / (4 * a) * B^2)
  A * exp(-B * r)
}

#' CIR zero-coupon bond price, P(t,T) = A(t,T) exp(-B(t,T) r).
#'
#' Standard affine solution (Cox, Ingersoll & Ross, 1985; Brigo & Mercurio, 2006,
#' eq. 3.25).
#'
#' @inheritParams vasicek_bond_price
cir_bond_price <- function(r, a, b, sigma, tau) {
  if (any(tau <= 0)) stop("tau (time to maturity) must be positive")
  gamma <- sqrt(a^2 + 2 * sigma^2)
  egt <- exp(gamma * tau)
  denom <- (gamma + a) * (egt - 1) + 2 * gamma
  B <- 2 * (egt - 1) / denom
  A <- (2 * gamma * exp((a + gamma) * tau / 2) / denom)^(2 * a * b / sigma^2)
  A * exp(-B * r)
}

#' Build a continuous discount curve P(0, T) from a handful of zero (spot) rates.
#'
#' Fits a natural cubic spline through the continuously-compounded zero rates
#' against maturity, anchored at `P(0, 0) = 1`. "Natural" means zero second
#' derivative at the two end knots, which is also what makes extrapolation
#' outside the quoted maturities well-behaved: it continues linearly from the
#' boundary slope rather than flattening abruptly. Flattening abruptly was tried
#' first (clamping T into `[min(maturities), max(maturities)]` before evaluating
#' the spline) and rejected -- it creates a kink in the zero-rate curve exactly at
#' the shortest quoted maturity, and `forward_rate_from_discount()` differentiates
#' that kink into a derivative spike (theta(t) reached 150 at one time step in a
#' 2-4% curve), which blew up every Hull-White path by an amount shared across all
#' of them since theta(t) does not depend on the path. This is the curve both
#' `simulate_hull_white()` and `hull_white_bond_price()` treat as "today's curve";
#' build it once from `load_treasury_curve()` and pass the same object to both so
#' the simulated short rate and the closed-form price agree on what they are
#' fitting.
#'
#' @param maturities maturities in years, strictly positive, increasing
#' @param zero_rates continuously-compounded zero rates at those maturities
#'   (decimal, not percent)
#' @return a vectorised function T -> P(0, T)
discount_curve_from_zero_rates <- function(maturities, zero_rates) {
  if (length(maturities) != length(zero_rates)) {
    stop("maturities and zero_rates must have the same length")
  }
  if (any(maturities <= 0)) stop("maturities must be positive")
  z <- stats::splinefun(maturities, zero_rates, method = "natural")
  function(T) {
    if (any(T < 0)) stop("T must be non-negative")
    exp(-z(T) * T)
  }
}

#' Central difference that falls back to a one-sided difference at a `t = 0`
#' boundary, so it never has to evaluate `f` to the left of zero.
#'
#' A naive `(f(t+h) - f(t-h)) / (2h)` evaluated at `t = 0` needs `f(-h)`. If `f`
#' itself clamps negative inputs (as `discount_curve_from_zero_rates()` used to),
#' two different negative arguments can clamp to the *same* floored value, and
#' the derivative silently comes back as a spurious large number instead of an
#' error -- `theta(t)` hit 150 at `t = 0` on a 2-4% curve this way before this
#' helper existed. Refusing negative arguments and shrinking the step near the
#' boundary instead turns that into a visibly smaller (first- rather than
#' second-order accurate) estimate at `t = 0`, with no silent blow-up.
#'
#' @param f vectorised function to differentiate
#' @param t point(s) to differentiate at, >= 0
#' @param h nominal step size
.deriv0 <- function(f, t, h) {
  if (any(t < 0)) stop("t must be non-negative")
  t_lo <- pmax(t - h, 0)
  t_hi <- t + h
  (f(t_hi) - f(t_lo)) / (t_hi - t_lo)
}

#' Instantaneous forward rate f(0,t) = -d/dt log P(0,t).
#'
#' @param discount_curve function(T) -> P(0,T), vectorised
#' @param h nominal step for the finite difference (see `.deriv0()`)
forward_rate_from_discount <- function(discount_curve, h = 1e-4) {
  log_p <- function(t) log(discount_curve(t))
  function(t) -.deriv0(log_p, t, h)
}

#' Hull-White zero-coupon bond price, fitted to `discount_curve` by construction.
#'
#' Extended-Vasicek closed form (Brigo & Mercurio, 2006, eq. 3.39):
#' `P(t,T) = (P_M(0,T) / P_M(0,t)) * exp(B(t,T) f_M(0,t) - sigma^2/(4a) (1 - exp(-2at)) B(t,T)^2)`
#' where `P_M` is the market discount curve and `f_M(0,t)` its instantaneous
#' forward rate. At `t = 0` this collapses to `P(0,T) = P_M(0,T)` exactly, which is
#' the point of allowing theta(t) to vary with time: the model reproduces the
#' input curve rather than approximating it. `test_hull_white_reproduces_the_input_discount_curve`
#' pins this at several maturities.
#'
#' @param r short rate at time t
#' @param a mean-reversion speed
#' @param sigma volatility
#' @param t valuation time, in years (>= 0)
#' @param Tt maturity, in years (> t)
#' @param discount_curve function(T) -> P(0,T), from `discount_curve_from_zero_rates()`
hull_white_bond_price <- function(r, a, sigma, t, Tt, discount_curve) {
  if (any(Tt <= t)) stop("Tt (maturity) must be greater than t")
  if (t == 0) return(discount_curve(Tt))
  forward_rate <- forward_rate_from_discount(discount_curve)
  B <- (1 - exp(-a * (Tt - t))) / a
  f_t <- forward_rate(t)
  (discount_curve(Tt) / discount_curve(t)) *
    exp(B * f_t - sigma^2 / (4 * a) * (1 - exp(-2 * a * t)) * B^2)
}

#' Monte Carlo zero-coupon bond price from a simulated short rate path matrix.
#'
#' `P(0,T) = E[exp(-integral_0^T r dt)]`, with the integral along each path done
#' by the trapezoid rule on the simulated grid and the expectation taken as the
#' sample mean across paths.
#'
#' @param r_paths matrix returned by a simulate_* function, with attr(,"times")
#' @return list(price, se) -- the mean discount factor and its standard error
mc_bond_price <- function(r_paths) {
  times <- attr(r_paths, "times")
  if (is.null(times)) stop("r_paths must carry attr(,\"times\") from a simulate_* function")
  n_steps <- nrow(r_paths) - 1L
  dt <- diff(times)
  # Trapezoid weights: half-weight on the two endpoints, full weight in between.
  w <- c(dt[1] / 2, (head(dt, -1) + tail(dt, -1)) / 2, dt[n_steps] / 2)
  integral <- as.numeric(w %*% r_paths)
  discount <- exp(-integral)
  list(price = mean(discount), se = stats::sd(discount) / sqrt(ncol(r_paths)))
}
