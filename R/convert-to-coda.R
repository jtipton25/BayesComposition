##
## script to process custom MCMC output from Rcpp into a format similar to R coda/JAGS/NIMBLE/STAN output
##

## Assumes a list where each element of the list is an MCMC chain
## Assumes the first dimension of each parameter in the MCMC chain is the number of MCMC iterations

convert_to_coda <- function(out) {
  n_chains <- length(out)
  
  process_chain <- function(j) {
    ## j is the chain number
    ## number of mcmc iterations (first dimesion of the chain)
    n_mcmc <- ifelse(is.null(dim(out[[j]][[1]])), length(out[[j]][[1]]), dim(out[[j]][[1]])[1])
    ## variable names, we assume these are identical across all chains
    var_names <- names(out[[j]])
    ## accumlate dimension and create a storage matrix
    total_variables <- 0
    for (i in var_names) {
      params <- eval(parse(text=paste("out[[", j, "]]$", i, sep="")))
      dim_params <- dim(params)    
      params_size <- dim_params[2:length(dim_params)]
      total_variables <- total_variables + prod(params_size)
    }
    ## create out matrix
    out_matrix <- matrix(0, n_mcmc, total_variables)
    ## index for looping and keeping track of location in out matrix
    out_idx <- 1
    ## name vector
    name_vector <- rep("NA", total_variables)
    for (i in var_names) {
      params <- eval(parse(text=paste("out[[", j, "]]$", i, sep="")) )
      if(is.null(dim(params))) {
        ## parameter is a vector
        names(params) <- paste(i, "[1]", sep="")
        out_matrix[, out_idx] <- params
        name_vector[out_idx] <- paste(i, "[1]", sep="")
        out_idx <- out_idx + 1
      } else {
        dim_params <- dim(params)
        params_size <- dim_params[2:length(dim_params)]
        
        ## Macro for processing, could extend this for very large dimensional arrays of parameters
        if (length(params_size) == 1) {
          ## vector of parameters
          for (kk in 1:params_size) {
            out_matrix[, out_idx] <- params[, kk]
            name_vector[out_idx] <- paste(i, "[", kk, "]", sep="")
            out_idx <- out_idx + 1
          }
        } else if (length(params_size) == 2) {
          ## matrix of parameters
          for (kk in 1:params_size[1]) {
            for (ll in 1:params_size[2]) {
              out_matrix[, out_idx] <- params[, kk, ll]
              name_vector[out_idx] <- paste(i, "[", kk, ", ", ll, "]", sep="")
              out_idx <- out_idx + 1
            }
          }
        } else if (length(params_size) == 3) {
          ## matrix of parameters
          for (kk in 1:params_size[1]) {
            for (ll in 1:params_size[2]) {
              for (mm in 1:params_size[3]) {
                out_matrix[, out_idx] <- params[, kk, ll, mm]
                name_vector[out_idx] <- paste(i, "[", kk, ", ", ll, ", ", mm, "]", sep="")
                out_idx <- out_idx + 1
              }
            }
          }
        } else if (length(params_size) == 4) {
          ## matrix of parameters
          for (kk in 1:params_size[1]) {
            for (ll in 1:params_size[2]) {
              for (mm in 1:params_size[3]) {
                for (nn in 1:params_size[3]) {
                  out_matrix[, out_idx] <- params[, kk, ll, mm, nn]
                  name_vector[out_idx] <- paste(i, "[", kk, ", ", ll, ", ", mm, ", ", nn, "]", sep="")
                  out_idx <- out_idx + 1
                }
              }
            }
          }
        }
      }
    }
    colnames(out_matrix) <- name_vector
    return(out_matrix)
  }
 
  out_mat <- lapply(1:n_chains, process_chain)
  # out_mat <- do.call(rbind, out_mat)
  return(out_mat)
}
