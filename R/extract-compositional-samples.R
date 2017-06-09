extract_compositional_samples <- function(out) {
  ## likelihood is the likelihood type. Options are::
  ##    - "gaussian"
  ##    - "multi-logit"
  ##    - "dirichlet-multinomial"
  ## function_type is the form of the latent functional. Options are:
  ##    - "basis"
  ##    - "gaussian-process"
  ## additive_correlation: include an additive correlated error in the "multi-logit" or "dirichlet-multinomial" likelihood options
  ## multiplicative_correlation: include a multiplicative correlation structure in the latent functional

  ## number of mcmc chains
  n_chains <- length(out)

  ## number of mcmc samples
  n_save <- dim(out[[1]][[1]])[1]
  n_samples <- n_save * n_chains

  ## variable names, we assume these are identical across all chains
  var_names <- names(out[[1]])

  samples <- list()
  for (i in var_names) {
    dim_params <- dim(out[[1]][[i]])
    dim_params[1] <- n_samples
    samples[[i]] <- array(0, dim_params)

    ## extract values for parameter var_names == i
    for (ii in 1:n_chains) {
      if (length(dim_params) == 1) {
        ## scalar parameter
        samples[[i]][1:n_save + (ii-1)*n_save] <- out[[ii]][[i]]
      } else if (length(dim_params) == 2) {
        ## vector parameter
        samples[[i]][1:n_save + (ii-1)*n_save, ] <- out[[ii]][[i]]
      } else if (length(dim_params) == 3) {
        ## matrix parameter
        samples[[i]][1:n_save + (ii-1)*n_save, , ] <- out[[ii]][[i]]
      } else if (length(dim_params) == 4) {
        ## 4d array parameter
        samples[[i]][1:n_save + (ii-1)*n_save, , , ] <- out[[ii]][[i]]
      } else if (length(dim_params) == 4) {
        ## 5d array parameter
        samples[[i]][1:n_save + (ii-1)*n_save, , , , ] <- out[[ii]][[i]]
      } else {
        stop("extract parameters can only handle up to 5-dimensional paramters")
      }
    }

  }
  return(samples)
}

