# The run.R command line entry point runs and prints numbers consistent with
# the library it wraps.

.rscript_bin <- function() {
  file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
}

invoke <- function(...) {
  root <- .helper_root
  system2(.rscript_bin(), c(file.path(root, "run.R"), ...), stdout = TRUE, stderr = TRUE)
}

test_that("run.R simulate prints terminal rate statistics for vasicek", {
  out <- invoke("simulate", "--model", "vasicek", "--paths", "10", "--T", "1",
                "--dt", "0.019230769", "--seed", "1")
  expect_true(any(grepl("terminal rate mean", out)))
})

test_that("run.R bonds compares monte carlo to the closed form for cir", {
  out <- invoke("bonds", "--model", "cir", "--T", "2", "--paths", "500", "--seed", "1")
  expect_true(any(grepl("closed form", out)))
})

test_that("run.R fit-ns reports fitted parameters on the cached fred curve", {
  out <- invoke("fit-ns", "--fred")
  expect_true(any(grepl("RMSE", out)))
})

test_that("run.R rejects an unknown subcommand", {
  out <- invoke("nonexistent")
  status <- attr(out, "status")
  expect_true(!is.null(status) && status != 0)
})
