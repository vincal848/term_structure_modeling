# Treasury par yield curve from FRED, with a small committed cache so tests and
# CI never touch the network. Only `fetch_treasury_curve()` calls out; everything
# else reads the cached CSV.

.ts_series <- c("DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5",
                 "DGS7", "DGS10", "DGS20", "DGS30")
.ts_maturity_years <- c(1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30)

#' Find the repository root by walking up from `start` until DESCRIPTION is found.
#'
#' Lets `load_treasury_curve()` find `data/treasury_curve.csv` whether it is run
#' from the repo root (`Rscript run.R ...`) or from `tests/testthat` (`testthat::test_dir`).
.find_repo_root <- function(start = getwd()) {
  dir <- normalizePath(start, mustWork = FALSE)
  for (i in 1:10) {
    if (file.exists(file.path(dir, "DESCRIPTION"))) return(dir)
    parent <- dirname(dir)
    if (identical(parent, dir)) break
    dir <- parent
  }
  start
}

#' Download today's Treasury par yield curve from FRED and overwrite the cache.
#'
#' Requires network access; not called by tests or by `load_treasury_curve()`
#' unless `refresh = TRUE`. The curve is the most recent complete row (all eleven
#' maturities non-NA) rather than literally today's date, since short maturities
#' sometimes post a day or two ahead of the long end.
#'
#' @param path CSV path to write
fetch_treasury_curve <- function(path) {
  env <- new.env()
  quantmod::getSymbols(.ts_series, src = "FRED", env = env)
  dat <- do.call(merge, lapply(.ts_series, function(s) get(s, envir = env)))
  dat <- stats::na.omit(dat)
  row <- utils::tail(dat, 1)
  d <- zoo::index(row)
  out <- data.frame(
    date = rep(as.character(d), length(.ts_series)),
    series = .ts_series,
    maturity_years = .ts_maturity_years,
    yield_pct = as.numeric(row[1, ])
  )
  utils::write.csv(out, path, row.names = FALSE)
  out
}

#' Load the Treasury par yield curve, from the committed cache by default.
#'
#' The cache (`data/treasury_curve.csv`) holds one date's curve across the eleven
#' CMT maturities FRED publishes (1 month through 30 years). Par yields are used
#' here as a stand-in for continuously-compounded zero rates, which is the usual
#' simplification for a single-date illustrative curve -- par and zero are close
#' at the short end and diverge more at the long end as coupon effects compound,
#' which is also why `fit_ns`'s RMSE on this curve, reported in docs/VALIDATION.md,
#' is not driven to zero.
#'
#' @param path CSV path; defaults to `data/treasury_curve.csv` under the repo root
#' @param refresh if TRUE, call `fetch_treasury_curve()` first (needs network)
#' @return data.frame(date, series, maturity_years, yield_pct), sorted by maturity
load_treasury_curve <- function(path = NULL, refresh = FALSE) {
  if (is.null(path)) path <- file.path(.find_repo_root(), "data", "treasury_curve.csv")
  if (refresh) fetch_treasury_curve(path)
  if (!file.exists(path)) {
    stop("no cached treasury curve at ", path, " -- call with refresh = TRUE (needs network)")
  }
  dat <- utils::read.csv(path, stringsAsFactors = FALSE)
  dat[order(dat$maturity_years), ]
}

#' Pull (maturity, decimal zero rate) vectors out of `load_treasury_curve()`'s output.
#'
#' @param curve data.frame from `load_treasury_curve()`
treasury_curve_as_zero_rates <- function(curve) {
  list(maturity = curve$maturity_years, zero_rate = curve$yield_pct / 100)
}
