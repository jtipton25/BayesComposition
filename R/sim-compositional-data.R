sim_compositional_data <- function(
  N                           = 500,
  d                           = 8,
  likelihood                  = "dirichlet-multinomial",
  expected_counts             = 100,
  function_type               = "basis",
  df                          =6,
  degree                      =3,
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

  mu <- rnorm(d, 0, 0.25)
  if (multiplicative_correlation) {
    ## select the dth component as the reference category
    mu[d] <-0
  }
  X <- rnorm(N, 0, 4)
  ## initialized values in global environment before conditional statements
  zeta <- matrix(0, N, d)
  eta_star <- matrix(0, n_knots, d)
  beta <- matrix(0, df-1, d)
  epsilon <- matrix(0, N, d)
  y <- matrix(0, N, d)



  if (function_type=="basis") {
    knots <- seq(min(X), max(X),
                 length=df-degree-1+2)[-c(1, df-degree-1+2)]
    Xbs <- bs_cpp(X, df, knots, degree, FALSE, c(min(X), max(X)))
    beta <- matrix(rnorm((df-1)*d), df-1, d)
    if (likelihood=="multi-logit") {
      beta[, d] <- rep(0, df-1)
    }
    zeta <- Xbs %*% beta
  } else if (function_type=="gaussian-process") {
    phi <- 5
    if (predictive_process) {
      ## simulate from the predictive process model
      n_knots <- 30
      X_knots <- seq(min(X)-1.25*sd(X), max(X)+1.25*sd(X), length=n_knots)
      D_knots <- as.matrix(dist(X_knots))               ## distance among knots
      D_interpolate <- as.matrix(rdist(X, X_knots))     ## distance from observed
      ## locations to knots
      if (correlation_function == "gaussian") {
        C_knots <- exp(- D_knots^2 / phi) #+ diag(n_knots)* 1e-12
        c_knots <- exp( - D_interpolate^2 / phi)
      } else if (correlation_function == "exponential") {
        C_knots <- exp(- D_knots / phi) #+ diag(n_knots) * 1e-12
        c_knots <- exp( - D_interpolate / phi)
      } else {
        stop ('Only "gaussian" and "exponential" covariance functions are supported')
      }

      C_knots_inv <- solve(C_knots)
      Z_knots <- c_knots %*% C_knots_inv
      eta_star <- t(mvnfast::rmvn(d, rep(0, n_knots), chol(C_knots),
                                  isChol=TRUE))
      zeta <- Z_knots %*% eta_star
      if (likelihood=="multi-logit") {
        eta_star[, d] <- rep(0, n_knots)
        zeta[, d] <- rep(0, N)
      }
    } else {
      ## simulate from the full-rank model
      D <- as.matrix(rdist(X))
      if (correlation_fun == "gaussian") {
        C <- exp(- D^2 / phi) #+ diag(n_knots)* 1e-12
      } else if (correlation_fun == "exponential") {
        C <- exp(- D / phi) #+ diag(n_knots) * 1e-12
      } else {
        stop ('Only "gaussian" and "exponential" covariance functions are supported')
      }
      zeta <- t(mvnfast::rmvn(d, rep(0, N), chol(C), isChol=TRUE))
      if (likelihood=="multi-logit") {
        zeta[, d] <- rep(0, N)
      }
    }
  } else {
    stop('only options for function_type are "basis" and "gaussian-process"')
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
      tau_additive <- rgamma(d-1, 5, 5)
      R_tau_additive <- R_additive %*% diag(tau_additive)
      epsilon[, 1:(d-1)] <- mvnfast::rmvn(N, rep(0, d-1), R_tau_additive, isChol=TRUE)
    } else {
      out_additive <- make_correlation_matrix(d, eta=1)
      R_additive <- out_additive$R
      xi_additive <- out_additive$xi
      tau_additive <- rgamma(d, 5, 5)
      R_tau_additive <- R_additive %*% diag(tau_additive)
      epsilon <- mvnfast::rmvn(N, rep(0, d), R_tau_additive, isChol=TRUE)
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
      tau_multiplicative <- rgamma(d-1, 5, 5)
      R_tau_multiplicative <- R_multiplicative %*% diag(tau_multiplicative)
      zeta[, 1:(d-1)] <- zeta[, 1:(d-1)] %*% R_tau_multiplicative
    } else {
      out_multiplicative <- make_correlation_matrix(d, eta=1)
      R_multiplicative <- out_multiplicative$R
      xi_multiplicative <- out_multiplicative$xi
      tau_multiplicative <- rgamma(d, 5, 5)
      R_tau_multiplicative <- R_multiplicative %*% diag(tau_multiplicative)
      zeta <- zeta %*% R_tau_multiplicative
    }
  }

  if (likelihood=="gaussian") {
    if (additive_correlation) {
      alpha <- t(mu + t(zeta))
      y <- t(mu + t(zeta) + t(epsilon))
      if (function_type=="basis") {
      return(list(y=y, X=X, alpha=alpha, mu=mu, beta=beta, zeta=zeta,
                  epsilon=epsilon, R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
      } else {
        return(list(y=y, X=X, mu=mu, phi=phi, eta_star=eta_star, zeta=zeta,
                    epsilon=epsilon, R_additive=R_additive,
                    tau_additive=tau_additive,
                    R_multiplicative=R_multiplicative,
                    tau_multiplicative=tau_multiplicative))
      }
    } else {
      stop('If using a Gaussian likelihood, set additive_correlation="TRUE"')
    }
  } else if (likelihood=="multi-logit") {
    alpha <- matrix(0, N, d)
    N_i <- rpois(N, expected_counts)
    for (i in 1:N) {
      tmp <- exp(mu + zeta[i, ] + epsilon[i, ])
      alpha[i, ] <- tmp / sum(tmp)
      y[i, ] <- rmultinom(1, N_i[i], alpha[i, ])
    }
    if (function_type=="basis") {
      return(list(y=y, X=X, alpha=alpha, N_i=N_i, mu=mu, beta=beta,
                  zeta=zeta, epsilon=epsilon, R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    } else {
      return(list(y=y, X=X, alpha=alpha, N_i=N_i,  mu=mu, phi=phi,
                  eta_star=eta_star, zeta=zeta, epsilon=epsilon,
                  R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    }
  } else if (likelihood=="dirichlet-multinomial") {
    alpha <- exp(t(mu + t(zeta) + t(epsilon)))
    N_i <- rpois(N, expected_counts)
    for (i in 1:N) {
      tmp <- rgamma(d, alpha[i, ], 1)
      p <- tmp / sum(tmp)
      y[i, ] <- rmultinom(1, N_i[i], p)
    }
    if (function_type=="basis") {
      return(list(y=y, X=X, alpha=alpha, N_i=N_i, mu=mu, beta=beta,
                  zeta=zeta, epsilon=epsilon, R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    } else {
      return(list(y=y, X=X, alpha=alpha, N_i=N_i,  mu=mu, phi=phi,
                  eta_star=eta_star, zeta=zeta, epsilon=epsilon,
                  R_additive=R_additive,
                  tau_additive=tau_additive,
                  R_multiplicative=R_multiplicative,
                  tau_multiplicative=tau_multiplicative))
    }
  } else {
    stop('Likelihood must be either "gaussian", "multi-logit", or "dirichelt-multinomial"')
  }
}
