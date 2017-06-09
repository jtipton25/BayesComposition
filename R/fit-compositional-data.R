fit_compositional_data <- function(
  y,
  X,
  params                = list("n_adapt"                     = 500,
                               "n_mcmc"                      = 500,
                               "n_thin"                      = 1,
                               df                            = 6,
                               degree                        = 3,
                               n_knots                       = 30,
                               X_knots                       = NULL,
                               correlation_function          = NULL,
                               additive_correlation          = TRUE,
                               multiplicative_correlation    = TRUE),
  likelihood            = "multi-logit",
  function_type         = "basis",

  n_chains              = 4,
  n_cores               = 1,
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

  ## make X_knots locations for predictive process

  if (is.null(params$X_knots) & function_type == "gaussian-process") {
    params$X_knots <- seq(min(X)-1.25*sd(X), max(X)+1.25*sd(X), length=n_knots)
  }


  ## parallel wrapper for code
  parallel_chains <- function(n_chains) {
    # , y=y, X=X, params=params,
    #                           progress_directory=progress_directory,
    #                           progress_file=progress_file) {
    if (likelihood == "gaussian") {
      if (function_type == "basis") {
        ## gaussian basis mcmc
        out <- coda::mcmc(mcmcRcppBasis(y, X, params, n_chain=n_chains,
                                        file_name=paste0(progress_directory,
                                                         progress_file)))
      } else if (function_type == "gaussian-process") {
        ## gaussian mvgp mcmc
        out <- coda::mcmc(mcmcRcppMVGP(y, X, params, n_chain=n_chains,
                                       file_name=paste0(progress_directory,
                                                        progress_file)))
      } else {
        ## error if function_type argument is incorrect
        stop('only valid options for function_type are "basis" and "gaussian-process"')
      }
    } else if (likelihood == "multi-logit") {
      if (function_type == "basis") {
        ## multi-logit basis mcmc
        out <- coda::mcmc(mcmcRcppMLBasis(y, X, params, n_chain=n_chains,
                                          file_name=paste0(progress_directory,
                                                           progress_file)))
      } else if (function_type == "gaussian-process") {
        ## multi-logit mvgp mcmc
        out <- coda::mcmc(mcmcRcppMLMVGP(y, X, params, n_chain=n_chains,
                                         file_name=paste0(progress_directory,
                                                          progress_file)))
      } else {
        ## error if function_type argument is incorrect
        stop('only valid options for function_type are "basis" and "gaussian-process"')
      }
    } else if (likelihood == "dirichlet-multinomial") {
      if (function_type == "basis") {
        ## dirichlet-multinomial basis mcmc
        out <- coda::mcmc(mcmcRcppDMBasis(y, X, params, n_chain=n_chains,
                                          file_name=paste0(progress_directory,
                                                           progress_file)))
      } else if (function_type == "gaussian-process") {
        ## dirichlet-multinomial mvgp mcmc
        out <- coda::mcmc(mcmcRcppDMMVGP(y, X, params, n_chain=n_chains,
                                         file_name=paste0(progress_directory,
                                                          progress_file)))
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

  if (n_cores == 1) {
    ## run chains sequentially
    out <- lapply(1:n_chains, parallel_chains)
    # , y=y, X=X, params=params,
    #               progress_directory=progress_directory,
    #               progress_file=progress_file)
  } else {
    ## run chains in parallel

    ## Initalize cluster
    sfInit(parallel=TRUE, cpus=4)
    sfClusterSetupRNG()
    sfExport("y", "X", "params", "likelihood", "function_type",
             "progress_directory", "progress_file")
    sfLibrary(BayesComposition)
    sfLibrary(coda)


    out <- sfLapply(1:n_chains, parallel_chains)
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
