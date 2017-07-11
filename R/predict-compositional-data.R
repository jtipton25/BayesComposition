predict_compositional_data <- function(
  y_reconstruct,
  X_calibrate,
  params,
  samples,
  progress_directory,
  progress_file
) {
  ## y is an N_pred by d matrix of observations where inverse prediction is desired
  ## N is the sample size
  ## d is the number of "species" in the composition
  ## X is an N-vector of covariates used in calibration
  ## params is the control vector of MCMC parameters used in fitting
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
  ## samples are the posterior model samples

  ## make sure y is a matrix
  if (!is.matrix(y_reconstruct)) {
    y_reconstruct <- as.matrix(y_reconstruct)
  }
  ## calculate prior from calibration data
  mu_X = mean(X_calibrate)
  s2_X <- var(X_calibrate)
  min_X <- min(X_calibrate)
  max_X <- max(X_calibrate)


  ## predictions
  if (params$likelihood == "gaussian") {
    if (params$function_type == "basis") {
      ## gaussian basis mcmc
      preds <- predictRcppBasis(y_reconstruct, mu_X, s2_X, min_X, max_X,
                                params, samples,
                                paste0(progress_directory, progress_file))
    } else if (params$function_type== "gaussian-process") {
      ## gaussian mvgp mcmc
      preds <- predictRcppMVGP(y_reconstruct, mu_X, s2_X, min_X, max_X, params,
                               samples, paste0(progress_directory, progress_file))
    } else {
      ## error if function_type argument is incorrect
      stop('only valid options for function_type are "basis" and "gaussian-process"')
    }
  } else if (params$likelihood == "multi-logit") {
    if (params$function_type == "basis") {
      ## multi-logit basis mcmc
      preds <- predictRcppMLBasis(y_reconstruct, mu_X, s2_X, min_X, max_X,
                                  params, samples,
                                  paste0(progress_directory, progress_file))
    } else if (params$function_type == "gaussian-process") {
      ## multi-logit mvgp mcmc
      preds <- predictRcppMLMVGP(y_reconstruct, mu_X, s2_X, min_X, max_X,
                                 params, samples,
                                 paste0(progress_directory, progress_file))
    } else {
      ## error if function_type argument is incorrect
      stop('only valid options for function_type are "basis" and "gaussian-process"')
    }
  } else if (params$likelihood == "dirichlet-multinomial") {
    if (params$function_type == "basis") {
      ## dirichlet-multinomial basis mcmc
      preds <- predictRcppDMBasis(y_reconstruct, mu_X, s2_X, min_X, max_X,
                                  params, samples,
                                  paste0(progress_directory, progress_file))
    } else if (params$function_type == "gaussian-process") {
      ## dirichlet-multinomial mvgp mcmc
      preds <- predictRcppDMMVGP(y_reconstruct, mu_X, s2_X, min_X, max_X,
                                 params, samples,
                                 paste0(progress_directory, progress_file))
    } else {
      ## error if function_type argument is incorrect
      stop('only valid options for function_type are "basis" and "gaussian-process"')
    }
  } else {
    ## error if likelihood argument is incorrect
    stop('only valid options for likelihood are "gaussian", "multi-logit", or "dirichlet-multinomial"')
  }



  return(preds)

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
