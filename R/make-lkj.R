###############################################################################
################ helper function for creation of LKJ Cholesky #################
###############################################################################

makeUpperLKJ <- function(xi, d) {
  out <- matrix(0, d, d)
  idx <- 1
  for(j in 1:d) {
    for(i in 1:d) {
      if (i < j) {
        out[i, j] <- xi[idx]
        idx <- idx + 1
      }
    }
  }
  return(out)
}

###############################################################################
#################### function for creation of LKJ Cholesky ####################
###############################################################################

makeLKJ <- function(xi, d, cholesky=TRUE) {
  R <- matrix(0, nrow=d, ncol=d)
  z <- makeUpperLKJ(xi, d)
  for (i in 1:d) {
    for (j in 1:d) {
      if (i == j & i == 1) {
        R[1, 1] <- 1.0
      } else if (i < j & i == 1) {
        R[1, j] <- z[1, j]
      } else if (i == j) {
        ones <- rep(1, i-1)
        R[i, j] <- prod(sqrt(ones - z[1:(i-1), j]^2.0))
      } else if (i < j) {
        ones <- rep(1, i-1)
        R[i, j] <- z[i, j] * prod(sqrt(ones - z[1:(i-1), j]^2.0))
      }
    }
  }
  if(cholesky) {
    return(R)
  } else {
    return(t(R) %*% R)
  }
}