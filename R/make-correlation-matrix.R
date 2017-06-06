make_correlation_matrix <- function (d, eta=1) {
  eta_vec <- rep(0, choose(d, 2))
  idx_eta <- 0
  for (j in 1:d) {
    for (k in 1:(d-j)) {
      eta_vec[idx_eta + k] <- eta + (d - 1 - j) / 2
    }
    idx_eta <- idx_eta + d-j
  }
  xi <- rep(0, choose(d, 2))
  for (i in 1:choose(d, 2)) {
    xi[i] <- 2 * rbeta(1, eta_vec[i], eta_vec[i]) - 1
  }
  R <- makeLKJ(xi, d, cholesky = TRUE)
  return(list(xi=xi, R=R))
}
