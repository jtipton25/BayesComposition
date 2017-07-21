fit_compositional_data <- function(
  y,
  X,
  params,
  progress_directory     = "./",
  progress_file         = "progress.txt",
  save_directory        = "./",
  save_file             = "out.RData"
) {
  ## y is an N by d matrix of observations
  ## N is the sample size
  ## d is the number of "species" in the composition
  ## X is an N-vector of covariates
  ## params is a control vector of MCMC parameters
  ## likelihood is the likelihood type. Options are::
  ##    - "gaussian"
  ##    - "multi-logit"
  ##    - "dirichlet-multinomial"
  ## function_type is the form of the latent functional. Options are:
  ##    - "basis"
  ##    - "gaussian-process"
  ## df is the basis degrees of freedom
  ## degree is the B-spline basis polynomial degree
  ## n_knots is the number of predictive process knots
  ## corelation_function is the Gaussian process correlation functional form. Options are:
  ##     - NULL if function_type = "basis"
  ##     - if function_type = "gaussian-process", options are:
  ##         -  "gaussian"
  ##         -  "exponential"
  ## additive_correlation: include an additive correlated error in the "multi-logit" or "dirichlet-multinomial" likelihood options
  ## multiplicative_correlation: include a multiplicative correlation structure in the latent functional
  ## n_chains is the number of parallel chains fit.
  ## n_cores is the number of cores (parallel processes) to be used
  ## progress_directory is the directory where the MCMC monitoring file will be saved. must end in "/"
  ## progress_file is the name of the progress file. this file ends in ".txt"
  ## save_directory is the directory where the MCMC samples will be saved. must end in "/"
  ## save_file is the name of the MCMC samples file. this file ends in ".RData"

  library(snowfall)
  ## make sure y is a matrix
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }

  ## create directory for progress file
  if (!dir.exists(progress_directory)) {
    dir.create(progress_directory)
  }
  ## create progress file
  file.create(paste0(progress_directory, progress_file))
  ## create directory for MCMC output file
  if (!dir.exists(save_directory)) {
    dir.create(save_directory)
  }

  ## if not given, set default parameter for number of adaptation iterations
  if(is.null(params$n_adapt)) {
    params$n_adapt <- 500
    params$n_adapt <<- 500
  }

  ## if not given, set default parameter for number of mcmc iterations
  if(is.null(params$n_mcmc)) {
    params$n_mcmc <- 500
    params$n_mcmc <<- 500
  }

  ## if not given, set default parameter of mcmc iterations to thin
  if(is.null(params$n_thin)) {
    params$n_thin <- 1
    params$n_thin <<- 1
  }

  ## if not given, set default parameter of mcmc iteration to output file
  if(is.null(params$message)) {
    params$message <- 1000
    params$message <<- 1000
  }

  ## if not given, set default B-spline degrees of freedom
  if(is.null(params$df)) {
    params$df <- 6
    params$df <<- 6
  }

  ## if not given, set default B-spline polynomial degree to cubic
  if(is.null(params$degree)) {
    params$degree <- 3
    params$degree <<- 3
  }

  ## if not given, set default predictive proocess knots to 30
  if(is.null(params$n_knots)) {
    params$n_knots <- 30
    params$n_knots <<- 30
  }

  ## if not given, make X_knots locations for predictive process
  if (is.null(params$X_knots) & params$function_type == "gaussian-process") {
    params$X_knots <- seq(min(X)-1.25*sd(X), max(X)+1.25*sd(X),
                          length=params$n_knots)
    params$X_knots <<- seq(min(X)-1.25*sd(X), max(X)+1.25*sd(X),
                           length=params$n_knots)
  }


  ## if not given, set additive_correlation to FALSE
  if(is.null(params$additive_correlation)) {
    params$additive_correlation <- FALSE
    params$additive_correlation <<- FALSE
  }

  ## if not given, set multiplicative_correlation to FALSE
  if(is.null(params$multiplicative_correlation)) {
    params$multiplicative_correlation <- FALSE
    params$multiplicative_correlation <<- FALSE
  }

  ## if not given, set n_chains to 4
  if(is.null(params$n_chains)) {
    params$n_chains <- 4
    params$n_chains <<- 4
  }
  ## if not given, set n_cors to 1
  if(is.null(params$n_cores)) {
    params$n_cores <- 1
    params$n_cores <<- 1
  }

  ## parallel wrapper for code
  parallel_chains <- function(n_chains) {
    # , y=y, X=X, params=params,
    #                           progress_directory=progress_directory,
    #                           progress_file=progress_file) {
    if (params$likelihood == "gaussian") {
      if (params$function_type == "basis") {
        ## gaussian basis mcmc
        out <- coda::mcmc(mcmcRcppBasis(y, X, params, n_chain=n_chains,
                                        file_name=paste0(progress_directory,
                                                         progress_file)))
      } else if (params$function_type== "gaussian-process") {
        ## gaussian mvgp mcmc
        out <- coda::mcmc(mcmcRcppMVGP(y, X, params, n_chain=n_chains,
                                       file_name=paste0(progress_directory,
                                                        progress_file)))
      } else {
        ## error if function_type argument is incorrect
        stop('only valid options for function_type are "basis" and "gaussian-process"')
      }
    } else if (params$likelihood == "multi-logit") {
      if (params$function_type == "basis") {
        ## multi-logit basis mcmc
        out <- coda::mcmc(mcmcRcppMLBasis(y, X, params, n_chain=n_chains,
                                          file_name=paste0(progress_directory,
                                                           progress_file)))
      } else if (params$function_type == "gaussian-process") {
        ## multi-logit mvgp mcmc
        out <- coda::mcmc(mcmcRcppMLMVGP(y, X, params, n_chain=n_chains,
                                         file_name=paste0(progress_directory,
                                                          progress_file)))
      } else {
        ## error if function_type argument is incorrect
        stop('only valid options for function_type are "basis" and "gaussian-process"')
      }
    } else if (params$likelihood == "dirichlet-multinomial") {
      if (params$function_type == "basis") {
        ## dirichlet-multinomial basis mcmc
        if (params$multiplicative == FALSE) {
          if (params$additive == FALSE) {
            out <- coda::mcmc(mcmcRcppDMBasis(y, X, params, n_chain=n_chains,
                                              file_name=paste0(progress_directory,
                                                               progress_file)))
          } else {
            out <- coda::mcmc(mcmcRcppDMBasisAdditive(y, X, params, n_chain=n_chains,
                                                      file_name=paste0(progress_directory,
                                                                       progress_file)))
          }
        } else {
          if (params$additive == FALSE) {
            out <- coda::mcmc(mcmcRcppDMBasisMultiplicative(y, X, params, n_chain=n_chains,
                                                            file_name=paste0(progress_directory,
                                                                             progress_file)))
          } else {
            out <- coda::mcmc(mcmcRcppDMBasisMultiplicativeAdditive(y, X, params, n_chain=n_chains,
                                                                    file_name=paste0(progress_directory,
                                                                                     progress_file)))
          }
        }
      } else if (params$function_type == "gaussian-process") {
        ## dirichlet-multinomial mvgp mcmc
        if (params$multiplicative == FALSE) {
          if (params$additive == FALSE) {
            out <- coda::mcmc(mcmcRcppDMMVGP(y, X, params, n_chain=n_chains,
                                             file_name=paste0(progress_directory,
                                                              progress_file)))
          } else {
            out <- coda::mcmc(mcmcRcppDMMVGPAdditive(y, X, params, n_chain=n_chains,
                                                     file_name=paste0(progress_directory,
                                                                      progress_file)))
          }
        } else {
          if (params$additive == FALSE) {
            out <- coda::mcmc(mcmcRcppDMMVGPMultiplicative(y, X, params, n_chain=n_chains,
                                                           file_name=paste0(progress_directory,
                                                                            progress_file)))
          } else {
            out <- coda::mcmc(mcmcRcppDMMVGPMultiplicativeAdditive(y, X, params, n_chain=n_chains,
                                                                   file_name=paste0(progress_directory,
                                                                                    progress_file)))
          }
        }
      } else {
        ## error if function_type argument is incorrect
        stop('only valid options for function_type are "basis" and "gaussian-process"')
      }
    } else {
      ## error if likelihood argument is incorrect
      stop('only valid options for likelihood are "gaussian", "multi-logit", or "dirichlet-multinomial"')
    }
  }

  ## initialize storage list
  out <- list()

  if (params$n_cores == 1) {
    ## run chains sequentially
    out <- lapply(1:params$n_chains, parallel_chains)
    # , y=y, X=X, params=params,
    #               progress_directory=progress_directory,
    #               progress_file=progress_file)
  } else {
    ## run chains in parallel

    ## Initalize cluster
    sfInit(parallel=TRUE, cpus=params$n_cores)
    sfClusterSetupRNG()
    sfExport("y", "X", "params",
             "progress_directory", "progress_file")
    sfLibrary(BayesComposition)
    sfLibrary(coda)


    out <- sfLapply(1:params$n_chains, parallel_chains)
    # , y=y, X=X, params=params,
    #                 progress_directory=progress_directory,
    #                 progress_file=progress_file)

    ## stop cluster
    sfStop()
  }
  ## save the output file
  save(out, params, file=paste0(save_directory, save_file))

  return(out)

  ##
  ## To do
  ## update mcmc files
  ##    - fit and prediction
  ## add in covariance options
  ##
  ## extract parameters
  ## make Rhat diagnostics
  ##

}
