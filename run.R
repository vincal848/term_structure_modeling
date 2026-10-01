#!/usr/bin/env Rscript
# Term structure modeling CLI.
#
#   Rscript run.R simulate --model vasicek --r0 0.03 --a 0.5 --b 0.04 --sigma 0.015 --T 1 --dt 0.0039682539683 --paths 5 --seed 1
#   Rscript run.R simulate --model cir   --r0 0.03 --a 0.8 --b 0.05 --sigma 0.12 --T 1 --dt 0.0039682539683 --paths 5 --seed 1
#   Rscript run.R simulate --model bk    --r0 0.03 --a 0.5 --theta 0.04 --sigma 0.25 --T 1 --dt 0.0039682539683 --paths 5 --seed 1
#   Rscript run.R bonds --model vasicek --r0 0.03 --a 0.5 --b 0.04 --sigma 0.015 --T 5 --paths 5000 --seed 1
#   Rscript run.R bonds --model cir     --r0 0.03 --a 0.8 --b 0.05 --sigma 0.12 --T 5 --paths 5000 --seed 1
#   Rscript run.R fit-ns --fred
#   Rscript run.R validate
#
# Run from the repository root (it sources R/*.R by relative path).

.script_path <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0) return(getwd())
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}
ROOT <- .script_path()
for (f in sort(list.files(file.path(ROOT, "R"), pattern = "\\.R$", full.names = TRUE))) {
  source(f)
}

#' Parse `--key value` pairs off a command-line argument vector into a named list.
#' Values that parse as numeric are converted; everything else stays a string.
.parse_flags <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    val <- if (i < length(args)) args[i + 1] else NA
    num <- suppressWarnings(as.numeric(val))
    out[[key]] <- if (!is.na(num)) num else val
    i <- i + 2
  }
  out
}

cmd_simulate <- function(flags) {
  model <- flags$model
  if (is.null(model)) stop("simulate requires --model (vasicek|cir|hw|bk)")
  Tt <- flags$T %||% 1
  dt <- flags$dt %||% (1 / 252)
  n_paths <- as.integer(flags$paths %||% 1)
  seed <- if (is.null(flags$seed)) NULL else as.integer(flags$seed)

  paths <- switch(model,
    vasicek = simulate_vasicek(flags$r0 %||% 0.03, flags$a %||% 0.5, flags$b %||% 0.04,
                                flags$sigma %||% 0.015, Tt, dt, n_paths, seed),
    cir = simulate_cir(flags$r0 %||% 0.03, flags$a %||% 0.8, flags$b %||% 0.05,
                        flags$sigma %||% 0.12, Tt, dt, n_paths, seed),
    bk = simulate_bk(flags$r0 %||% 0.03, flags$a %||% 0.5, flags$theta %||% 0.04,
                      flags$sigma %||% 0.25, Tt, dt, n_paths, seed),
    hw = {
      curve <- treasury_curve_as_zero_rates(load_treasury_curve())
      dc <- discount_curve_from_zero_rates(curve$maturity, curve$zero_rate)
      simulate_hull_white(forward_rate_from_discount(dc)(0), flags$a %||% 0.5,
                           flags$sigma %||% 0.01, Tt, dt, dc, n_paths, seed)
    },
    stop("unknown --model: ", model)
  )
  cat(sprintf("%s: %d steps x %d paths, terminal rate mean %.6f sd %.6f\n",
              model, nrow(paths) - 1L, ncol(paths),
              mean(paths[nrow(paths), ]), stats::sd(paths[nrow(paths), ])))
  if (!is.null(flags$out)) {
    utils::write.csv(paths, flags$out, row.names = FALSE)
    cat(sprintf("wrote %s\n", flags$out))
  }
}

cmd_bonds <- function(flags) {
  model <- flags$model
  if (is.null(model)) stop("bonds requires --model (vasicek|cir|hw)")
  Tt <- flags$T %||% 5
  dt <- flags$dt %||% (1 / 52)
  n_paths <- as.integer(flags$paths %||% 5000)
  seed <- if (is.null(flags$seed)) NULL else as.integer(flags$seed)

  if (model == "hw") {
    curve <- treasury_curve_as_zero_rates(load_treasury_curve())
    dc <- discount_curve_from_zero_rates(curve$maturity, curve$zero_rate)
    r0 <- forward_rate_from_discount(dc)(0)
    a <- flags$a %||% 0.5; sigma <- flags$sigma %||% 0.01
    paths <- simulate_hull_white(r0, a, sigma, Tt, dt, dc, n_paths, seed)
    mc <- mc_bond_price(paths)
    closed <- hull_white_bond_price(r0, a, sigma, t = 0, Tt = Tt, discount_curve = dc)
  } else {
    r0 <- flags$r0 %||% 0.03; a <- flags$a %||% 0.5
    b <- flags$b %||% (if (model == "vasicek") 0.04 else 0.05)
    sigma <- flags$sigma %||% (if (model == "vasicek") 0.015 else 0.12)
    paths <- switch(model,
      vasicek = simulate_vasicek(r0, a, b, sigma, Tt, dt, n_paths, seed),
      cir = simulate_cir(r0, a, b, sigma, Tt, dt, n_paths, seed),
      stop("unknown --model: ", model))
    mc <- mc_bond_price(paths)
    closed <- switch(model,
      vasicek = vasicek_bond_price(r0, a, b, sigma, Tt),
      cir = cir_bond_price(r0, a, b, sigma, Tt))
  }
  cat(sprintf("%s bond, T=%g, %d paths: MC %.6f (SE %.2e) vs closed form %.6f, diff %.2e\n",
              model, Tt, n_paths, mc$price, mc$se, closed, mc$price - closed))
}

cmd_fit_ns <- function(flags) {
  if (isTRUE(flags$fred) || !is.null(flags$fred)) {
    curve <- load_treasury_curve(refresh = isTRUE(flags$refresh))
    maturity <- curve$maturity_years
    yield <- curve$yield_pct / 100
  } else if (!is.null(flags$maturity) && !is.null(flags$yield)) {
    maturity <- as.numeric(strsplit(flags$maturity, ",")[[1]])
    yield <- as.numeric(strsplit(flags$yield, ",")[[1]])
  } else {
    stop("fit-ns requires --fred, or --maturity and --yield as comma-separated lists")
  }
  fit <- fit_ns(maturity, yield)
  cat(sprintf("beta0=%.6f beta1=%.6f beta2=%.6f lambda=%.6f  RMSE=%.6f (%s)\n",
              fit$params$beta0, fit$params$beta1, fit$params$beta2, fit$params$lambda,
              fit$rmse, fit$engine))
}

cmd_validate <- function(flags) {
  run_validate(ROOT)
}

`%||%` <- function(x, y) if (is.null(x)) y else x

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 0) {
    cat("usage: Rscript run.R <simulate|bonds|fit-ns|validate> [--flag value ...]\n")
    return(invisible(NULL))
  }
  sub <- args[1]
  flags <- .parse_flags(args[-1])
  switch(sub,
    simulate = cmd_simulate(flags),
    bonds = cmd_bonds(flags),
    `fit-ns` = cmd_fit_ns(flags),
    validate = cmd_validate(flags),
    stop("unknown subcommand: ", sub)
  )
}

if (identical(environment(), globalenv())) {
  main()
}
