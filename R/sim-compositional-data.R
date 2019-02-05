sim_compositional_data <- function(
  N                           = 500,
  N_pred                      = 25,
  d                           = 8,
  likelihood                  = "dirichlet-multinomial",
  expected_counts             = 100,
  function_type               = "basis",
  df                          = 6,
  degree                      = 3,
  predictive_process          = TRUE,
  n_knots                     = 30,
  phi                         = 5,
  correlation_function        = NULL,
  additive_correlation        = FALSE,
  multiplicative_correlation  = FALSE) {
  ## N is the sample size
  ## d is the number of "species" in the composition
  ## likelihood is the likelihood type. Options are::
  ##    - "gaussian"
  ##    - "multi-logit"
  ##    - "dirichlet-multinomial"
  ## expected_counts are the expected number of compositional counts
  ## function_type is the form of the latent functional. Options are:
  ##    - "basis"
  ##    - "gaussian-process"
  ## predictive_process if TRUE, simulate from the reduced-rank preditive process model
  ## n_knots is the number of knots used in the predictive-process simulation
  ## phi is the Gaussian process range parameter
  ## corelation_function is the Gaussian process correlation functional form. Options are:
  ##     - NULL if function_type = "basis"
  ##     - if function_type = "gaussian-process", options are:
  ##         -  "gaussian"
  ##         -  "exponential"
  ## additive_correlation: include an additive correlated error in the "multi-logit" or "dirichlet-multinomial" likelihood options
  ## multiplicative_correlation: include a multiplicative correlation structure in the latent functional

  library(fields)
  library(splines)

  # mu <- rnorm(d, 0, 0.25)
  mu <- rnorm(d)
  if (multiplicative_correlation) {
    ## select the dth component as the reference category
    mu[d] <-0
  }
  X <- rnorm(N, 0, 2)
  X_pred <- rnorm(N_pred, 0, 2)
  ## initialized values in global environment before conditional statements
  zeta <- matrix(0, N, d)
  zeta_pred <- matrix(0, N_pred, d)
  eta_star <- matrix(0, n_knots, d)
  beta <- matrix(0, df-1, d)
  epsilon <- matrix(0, N, d)
  epsilon_pred <- matrix(0, N_pred, d)
  y <- matrix(0, N, d)
  y_pred <- matrix(0, N_pred, d)



  if (function_type=="basis") {
    knots <- seq(min(X), max(X),
                 length=df-degree-1+2)[-c(1, df-degree-1+2)]
    Xbs <- BayesComposition::bs_cpp(X, df, knots, degree, FALSE, c(min(X), max(X)))
    Xbs_pred <- BayesComposition::bs_cpp(X_pred, df, knots, degree, FALSE, c(min(X), max(X)))
    beta <- matrix(rnorm((df-1)*d), df-1, d)
    if (likelihood=="multi-logit") {
      beta[, d] <- rep(0, df-1)
    }
    zeta <- Xbs %*% beta
    zeta_pred <- Xbs_pred %*% beta
  } else if (function_type=="gaussian-process") {
    phi <- 5
    if (predictive_process) {
      ## simulate from the predictive process model
      n_knots <- 30
      X_knots <- seq(min(X)-1.25*sd(X), max(X)+1.25*sd(X), length=n_knots)
      D_knots <- as.matrix(dist(X_knots))                     ## distance among knots
      D_interpolate <- as.matrix(rdist(X, X_knots))           ## distance from observed locations to knots
      D_interpolate_pred <- as.matrix(rdist(X_pred, X_knots)) ## distance from unobserved locations to knots
      if (correlation_function == "gaussian") {
        C_knots <- exp(- D_knots^2 / phi) #+ diag(n_knots)* 1e-12
        c_knots <- exp( - D_interpolate^2 / phi)
        c_knots_pred <- exp( - D_interpolate_pred^2 / phi)
      } else if (correlation_function == "exponential") {
        C_knots <- exp(- D_knots / phi) #+ diag(n_knots) * 1e-12
        c_knots <- exp( - D_interpolate / phi)
        c_knots_pred <- exp( - D_interpolate_pred / phi)
      } else {
        stop ('Only "gaussian" and "exponential" covariance functions are supported')
      }

      C_knots_inv <- solve(C_knots)
      Z_knots <- c_knots %*% C_knots_inv
      Z_knots_pred <- c_knots_pred %*% C_knots_inv
      eta_star <- t(mvnfast::rmvn(d, rep(0, n_knots), chol(C_knots),
                                  isChol=TRUE))
      zeta <- Z_knots %*% eta_star
      zeta_pred <- Z_knots_pred %*% eta_star
      if (likelihood=="multi-logit") {
        eta_star[, d] <- rep(0, n_knots)
        zeta[, d] <- rep(0, N)
        zeta_pred[, d] <- rep(0, N_pred)
      }
    } else {
      ## simulate from the full-rank model
      D <- as.matrix(rdist(X))
      D_pred <- as.matrix(rdist(X_pred))
      D_full <- as.matrix(rdist(c(X, X_pred)))
      if (correlation_function == "gaussian") {
        C_full <- exp( - D_full^2 / phi) #+ diag(n_knots)* 1e-12
        # C <- exp(- D^2 / phi) #+ diag(n_knots)* 1e-12
        # C_pred <- exp(- D_pred^2 / phi) #+ diag(n_knots)* 1e-12
      } else if (correlation_function== "exponential") {
        C_full <- exp( - D_full / phi) #+ diag(n_knots)* 1e-12
        # C <- exp(- D / phi) #+ diag(n_knots) * 1e-12
        # C_pred <- exp(- D_pred / phi) #+ diag(n_knots) * 1e-12
      } else {
        stop ('Only "gaussian" and "exponential" covariance functions are currently supported')
      }
      zeta_full <- t(mvnfast::rmvn(d, rep(0, N+N_pred), chol(C_full), isChol=TRUE))
      zeta <- zeta_full[1:N, ]
      zeta_pred <- zeta_full[(N + 1):(N + N_pred), ]
      # zeta <- t(mvnfast::rmvn(d, rep(0, N), chol(C), isChol=TRUE))
      # zeta_pred <- t(mvnfast::rmvn(d, rep(0, N_pred), chol(C_pred), isChol=TRUE))
      if (likelihood=="multi-logit") {
        ## fix the dth category as a reference category
        zeta[, d] <- rep(0, N)
        zeta_pred[, d] <- rep(0, N_pred)
      }
    }
  }
   else if (function_type == "bummer") {
    a <- rnorm(d, 0, 0.75)
    b <- rnorm(d, 0, 1)
    c <- rgamma(d, 20, 2.5)
    mu <- a
    for (j in 1:d) {
      zeta[, j] <-  - (b[j] - X)^2 / c[j]
      zeta_pred[, j] <- - (b[j] - X_pred)^2 / c[j]
    }
   } else {
     # stop('only options for function_type are "basis" and "gaussian-process"')
     stop('only options for function_type are "basis", "bummer" and "gaussian-process"')
  }
  ## initialize additive components
  R_additive <- diag(d)
  xi_additive <-rep(0, choose(d, 2))
  tau_additive <- rep(1, d)
  if (likelihood=="multi-logit") {
    ## select the dth component as the reference category
    R_additive <- diag(d-1)
    xi_additive <-rep(0, choose(d-1, 2))
    tau_additive <- rep(1, d-1)
  }
  R_tau_additive <- R_additive %*% diag(tau_additive)

  if (additive_correlation) {
    if (likelihood=="multi-logit") {
      ## select the dth component as the reference category
      out_additive <- make_correlation_matrix(d-1, eta=1)
      R_additive <- out_additive$R
      xi_additive <- out_additive$xi
      tau_additive <- rgamma(d-1, 2, 5)
      R_tau_additive <- R_additive %*% diag(tau_additive)
      epsilon[, 1:(d-1)] <- mvnfast::rmvn(N, rep(0, d-1), R_tau_additive, isChol=TRUE)
      epsilon_pred[, 1:(d-1)] <- mvnfast::rmvn(N_pred, rep(0, d-1), R_tau_additive, isChol=TRUE)
    } else {
      out_additive <- make_correlation_matrix(d, eta=1)
      R_additive <- out_additive$R
      xi_additive <- out_additive$xi
      tau_additive <- rgamma(d, 2, 5)
      R_tau_additive <- R_additive %*% diag(tau_additive)
      epsilon <- mvnfast::rmvn(N, rep(0, d), R_tau_additive, isChol=TRUE)
      epsilon_pred <- mvnfast::rmvn(N_pred, rep(0, d), R_tau_additive, isChol=TRUE)
    }
  }

  ## initialize multiplicative components
  R_multiplicative <- diag(d)
  xi_multiplicative <-rep(0, choose(d, 2))
  tau_multiplicative <- rep(1, d)
  if (likelihood=="multi-logit") {
    ## select the dth component as the reference category
    R_multiplicative <- diag(d-1)
    xi_multiplicative <-rep(0, choose(d-1, 2))
    tau_multiplicative <- rep(1, d-1)
  }
  R_tau_multiplicative <- R_multiplicative %*% diag(tau_multiplicative)

  if (multiplicative_correlation) {
    if (likelihood=="multi-logit") {
      ## select the dth component as the reference category
      out_multiplicative <- make_correlation_matrix(d-1, eta=1)
      R_multiplicative <- out_multiplicative$R
      xi_multiplicative <- out_multiplicative$xi
      # tau_multiplicative <- rgamma(d-1, 5, 2)
      tau_multiplicative <- rgamma(d-1, 5, 5)
      R_tau_multiplicative <- R_multiplicative %*% diag(tau_multiplicative)
      zeta[, 1:(d-1)] <- zeta[, 1:(d-1)] %*% R_tau_multiplicative
      zeta_pred[, 1:(d-1)] <- zeta_pred[, 1:(d-1)] %*% R_tau_multiplicative
    } else {
      out_multiplicative <- make_correlation_matrix(d, eta=1)
      R_multiplicative <- out_multiplicative$R
      xi_multiplicative <- out_multiplicative$xi
      # tau_multiplicative <- rgamma(d, 5, 2)
      tau_multiplicative <- rgamma(d, 5, 5)
      R_tau_multiplicative <- R_multiplicative %*% diag(tau_multiplicative)
      zeta <- zeta %*% R_tau_multiplicative
      zeta_pred <- zeta_pred %*% R_tau_multiplicative
    }
  }

  if (likelihood=="gaussian") {
    if (additive_correlation) {
      alpha <- t(mu + t(zeta))
      alpha_pred <- t(mu + t(zeta_pred))
      y <- t(mu + t(zeta) + t(epsilon))
      y_pred <- t(mu + t(zeta_pred) + t(epsilon_pred))
      if (function_type=="basis") {
      return(list(y=y, y_pred=y_pred, X=X, X_pred=X_pred,
                  alpha=alpha, alpha_pred=alpha_pred, mu=mu, beta=beta,
                  zeta=zeta, zeta_pred=zeta_pred, epsilon=epsilon,
                  epsilon_pred=epsilon_pred, R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
      } else {
        return(list(y=y, y_pred=y_pred, X=X, X_pred=X_pred,
                    alpha=alpha, alpha_pred=alpha_pred, mu=mu, phi=phi,
                    eta_star=eta_star, zeta=zeta, zeta_pred=zeta_pred,
                    epsilon=epsilon, epsilon_pred=epsilon_pred,
                    R_additive=R_additive, tau_additive=tau_additive,
                    R_multiplicative=R_multiplicative,
                    tau_multiplicative=tau_multiplicative))
      }
    } else {
      stop('If using a Gaussian likelihood, set additive_correlation="TRUE"')
    }
  } else if (likelihood=="multi-logit") {
    alpha <- matrix(0, N, d)
    alpha_pred <- matrix(0, N_pred, d)
    N_i <- rpois(N, expected_counts)
    N_i_pred <- rpois(N_pred, expected_counts)
    for (i in 1:N) {
      tmp <- exp(mu + zeta[i, ] + epsilon[i, ])
      alpha[i, ] <- tmp / sum(tmp)
      y[i, ] <- rmultinom(1, N_i[i], alpha[i, ])
    }
    for (i in 1:N_pred) {
      tmp <- exp(mu + zeta_pred[i, ] + epsilon_pred[i, ])
      alpha_pred[i, ] <- tmp / sum(tmp)
      y_pred[i, ] <- rmultinom(1, N_i_pred[i], alpha_pred[i, ])
    }
    if (function_type=="basis") {
      return(list(y=y, y_pred=y_pred, X=X, X_pred=X_pred, alpha=alpha,
                  alpha_pred=alpha_pred, N_i=N_i, N_i_pred=N_i_pred,
                  mu=mu, beta=beta, zeta=zeta, zeta_pred=zeta_pred,
                  epsilon=epsilon, epsilon_pred=epsilon_pred,
                  R_additive=R_additive, tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    } else {
      return(list(y=y, y_pred=y_pred, X=X, X_pred=X_pred, alpha=alpha,
                  alpha_pred=alpha_pred, N_i=N_i, N_i_pred=N_i_pred,
                  mu=mu, phi=phi, eta_star=eta_star,
                  zeta=zeta, zeta_pred=zeta_pred,
                  epsilon=epsilon, epsilon_pred=epsilon_pred,
                  R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    }
  } else if (likelihood=="dirichlet-multinomial") {
    alpha <- exp(t(mu + t(zeta) + t(epsilon)))
    alpha_pred <- exp(t(mu + t(zeta_pred) + t(epsilon_pred)))
    N_i <- rpois(N, expected_counts)
    N_i_pred <- rpois(N_pred, expected_counts)
    for (i in 1:N) {
      tmp <- rgamma(d, alpha[i, ], 1)
      p <- tmp / sum(tmp)
      y[i, ] <- rmultinom(1, N_i[i], p)
    }
    for (i in 1:N_pred) {
      tmp <- rgamma(d, alpha_pred[i, ], 1)
      p <- tmp / sum(tmp)
      y_pred[i, ] <- rmultinom(1, N_i_pred[i], p)
    }
    if (function_type=="basis") {
      return(list(y=y, y_pred=y_pred, X=X, X_pred=X_pred, alpha=alpha,
                  alpha_pred=alpha_pred, N_i=N_i, N_i_pred=N_i_pred,
                  mu=mu, beta=beta,
                  zeta=zeta, zeta_pred=zeta_pred,
                  epsilon=epsilon, epsilon_pred=epsilon_pred,
                  R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    } else if (function_type == "bummer") {
      return(list(y=y, y_pred=y_pred, X=X, X_pred=X_pred, alpha=alpha,
                  alpha_pred=alpha_pred, zeta=zeta, N_i=N_i, N_i_pred=N_i_pred,
                  a=a, b=b, c=c))

    } else {
      return(list(y=y, y_pred=y_pred, X=X, X_pred=X_pred, alpha=alpha,
                  alpha_pred=alpha_pred, N_i=N_i, N_i_pred=N_i_pred,
                  mu=mu, phi=phi, eta_star=eta_star,
                  zeta=zeta, zeta_pred=zeta_pred,
                  epsilon=epsilon, epsilon_pred=epsilon_pred,
                  R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    }
  } else {
    stop('Likelihood must be either "gaussian", "multi-logit", or "dirichelt-multinomial"')
  }
}
