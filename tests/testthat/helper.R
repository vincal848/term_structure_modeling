# Source R/*.R before running any test in this directory. Works whether
# testthat::test_dir() is invoked from the repo root or from tests/testthat.
.helper_root <- getwd()
if (!dir.exists(file.path(.helper_root, "R"))) {
  .helper_root <- normalizePath(file.path(.helper_root, "..", ".."))
}
for (.f in sort(list.files(file.path(.helper_root, "R"), pattern = "\\.R$", full.names = TRUE))) {
  source(.f)
}
rm(.f)
