//
// To do
// - add in correlation R_tau
// - add in ESS for X
// - block sampler for efficiency
// - on/off swtich for sampler diagnostics
//
// General to do
// - jacobian for log and logit propsal

// #define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include "BayesComp.h"
#include <iostream>  // I/O
#include <fstream>   // file I/O
#include <iomanip>   // format manipulation
#include <string>

using namespace Rcpp;
using namespace arma;

// Inverse Gaussian Process predcition with splines MCMC

// Author: John Tipton
//
// Created 12.10.2016
// Last updated 04.29.2017

///////////////////////////////////////////////////////////////////////////////
////////////////////////// Gaussian Log-Likelihood ///////////////////////////
///////////////////////////////////////////////////////////////////////////////

double LL (const arma::mat& alpha, const arma::mat& Y, const double& N,
           const double& sigma, const double& d) {
  // alpha is an N by d matrix of transformed parameters
  // Y is a N by d dimensional vector of species counts
  double log_like = 0.0;
  for (int i=0; i<N; i++){
    for (int j=0; j<d; j++){
      log_like += R::dnorm(Y(i, j), alpha(i, j), sigma, true);
    }
  }
  return(log_like);
}

///////////////////////////////////////////////////////////////////////////////
///////////// Elliptical Slice Sampler for unobserved covariate X /////////////
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess_X (const double& X_current, const double& X_prior,
                  const double& mu_X,
                  const arma::mat& beta_current,
                  const arma::rowvec& alpha_row, const arma::rowvec& Y_row,
                  const double& sigma_current, const arma::rowvec& Xbs_current,
                  const double& d, const arma::vec& knots,
                  const int& df, const int& degree, const arma::vec& rangeX,
                  const std::string& file_name, const int& n_chain) {
  // X_current is the current value of the parameter
  // X_prior is a sample from the prior

  // calculate log likelihood of current value
  double current_log_like = 0.0;
  for (int j=0; j<d; j++) {
    current_log_like += R::dnorm(Y_row(j), alpha_row(j), sigma_current, true);
  }

  double hh = log(R::runif(0.0, 1.0)) + current_log_like;

  // Setup a bracket and pick a first proposal
  // Bracket whole ellipse with both edges at first proposed point
  double phi_angle = R::runif(0.0, 1.0) * 2.0 * arma::datum::pi;
  double phi_angle_min = phi_angle - 2.0 * arma::datum::pi;
  double phi_angle_max = phi_angle;

  double X_ess = X_current;
  arma::rowvec Xbs_ess = Xbs_current;
  arma::rowvec alpha_ess = alpha_row;

  bool test = true;

  // Slice sampling loop
  while (test) {
    Rcpp::checkUserInterrupt();
    // compute proposal for angle difference and check to see if it is on the slice
    double X_proposal = X_current * cos(phi_angle) + X_prior * sin(phi_angle);
    // adjust for non-zero mean
    arma::vec X_tilde(1);
    X_tilde(0) = X_proposal + mu_X;
    arma::rowvec Xbs_proposal = bs_cpp(X_tilde, df, knots, degree, true,
                                       rangeX);
    arma::rowvec alpha_proposal = (Xbs_proposal * beta_current);

    // calculate log likelihood of proposed value
    double proposal_log_like = 0.0;
    for (int j=0; j<d; j++) {
      proposal_log_like += R::dnorm(Y_row(j), alpha_proposal(j), sigma_current,
                                    true);
    }


    if (proposal_log_like > hh) {
      // proposal is on the slice
      X_ess = X_proposal;
      Xbs_ess = Xbs_proposal;
      alpha_ess = alpha_proposal;
      test = false;
    } else if (phi_angle > 0.0) {
      phi_angle_max = phi_angle;
    } else if (phi_angle < 0.0) {
      phi_angle_min = phi_angle;
    } else {
      Rprintf("Bug detected - ESS for X shrunk to current position and still not acceptable");
      // set up output messages
      std::ofstream file_out;
      file_out.open(file_name, std::ios_base::app);
      file_out << "Bug - ESS for X shrunk to current position on chain " << n_chain << "\n";
      // close output file
      file_out.close();
    }
    // Propose new angle difference
    phi_angle = R::runif(0.0, 1.0) * (phi_angle_max - phi_angle_min) + phi_angle_min;
  }
  return(Rcpp::List::create(
      _["X"] = X_ess,
      _["Xbs"] = Xbs_ess,
      _["alpha"] = alpha_ess));
}

///////////////////////////////////////////////////////////////////////////////
/////////////// Metropolis-Hastings for unobserved covariate X ////////////////
///////////////////////////////////////////////////////////////////////////////

double mhX (const double& X_mh, const double& mu_X, const double& s2_X,
            const arma::rowvec& alpha, const arma::rowvec& Y_row,
            const double& sigma, const double& d) {
  // mu is a d-vector of regression intercepts
  // beta is a d-vector of regression parameters
  // X_mh are unobserved covariate values
  double logDensity = - 0.5 * pow(X_mh - mu_X, 2.0) / s2_X;
  for (int j=0; j<d; j++) {
    logDensity += R::dnorm(Y_row(j), alpha(j), sigma, true);
  }
  return(logDensity);
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppGAM (const arma::mat& Y, const arma::vec& X_input, List params,
                  int n_chain=1, std::string file_name="gam") {

  // Load parameters
  int n_adapt = as<int>(params["n_adapt"]);
  int n_mcmc = as<int>(params["n_mcmc"]);
  int n_thin = as<int>(params["n_thin"]);
  // defaults b-spline order
  int degree = 3;
  if (params.containsElementNamed("degree")) {
    degree = as<int>(params["degree"]);
  }
  // default b-spline degrees of freedom
  int df = 6;
  if (params.containsElementNamed("df")) {
    df = as<int>(params["df"]);
  }
  int N_obs = as<int>(params["N_obs"]);

  // set up dimensions
  double N = Y.n_rows;
  double d = Y.n_cols;
  // double B = Rf_choose(d, 2);
  arma::mat I_N(N, N, arma::fill::eye);
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_N(N, arma::fill::ones);
  arma::vec zeros_d(d, arma::fill::zeros);
  arma::vec zeros_df(df, arma::fill::zeros);

  // default to message output every 5000 iterations
  int message = 5000;
  if (params.containsElementNamed("message")) {
    message = as<int>(params["message"]);
  }

  // default half cauchy scale for error variance sigma2
  double s2_sigma2 = 25.0;
  if (params.containsElementNamed("s2_sigma2")) {
    s2_sigma2 = as<double>(params["s2_sigma2"]);
  }

  // default sigma2 tuning parameter standard deviation of 0.25
  double sigma2_tune = 0.25;
  if (params.containsElementNamed("sigma2_tune")) {
    sigma2_tune = as<double>(params["sigma2_tune"]);
  }

  // default beta tuning parameter
  double lambda_beta_tune_tmp = 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_beta_tune")) {
    lambda_beta_tune_tmp = as<double>(params["lambda_beta_tune"]);
  }
  arma::vec lambda_beta_tune(d, arma::fill::ones);
  lambda_beta_tune *= lambda_beta_tune_tmp;

  // default to centered missing covariate
  double mu_X = arma::mean(X_input.subvec(0, N_obs-1));
  // if (params.containsElementNamed("mu_X")) {
  //   mu_X = as<double>(params["mu_X"]);
  // }
  // default to scaled missing covariate
  double s2_X = arma::var(X_input.subvec(0, N_obs-1));
  // if (params.containsElementNamed("s2_X")) {
  //   s2_X = as<double>(params["s2_X"]);
  // }
  double s_X = sqrt(s2_X);
  // default to X tuning parameter standard deviation of 0.25
  double X_tune_tmp = 0.25;
  if (params.containsElementNamed("X_tune")) {
    X_tune_tmp = as<double>(params["X_tune"]);
  }

  //
  // Turn on/off sampler for X
  //

  bool sample_X = true;
  if (params.containsElementNamed("sample_X")) {
    sample_X = as<bool>(params["sample_X"]);
  }
  // default elliptical slice sampler for X
  bool sample_X_mh = false;
  if (params.containsElementNamed("sample_X_mh")) {
    sample_X_mh = as<bool>(params["sample_X_mh"]);
  }

  arma::vec X = X_input;
  for (int i=N_obs; i<N; i++) {
    X(i) = R::rnorm(0.0, s_X);
  }
  double minX = min(X_input);
  if (params.containsElementNamed("minX")) {
    minX = as<double>(params["minX"]);
  }
  double maxX = max(X_input);
  if (params.containsElementNamed("maxX")) {
    maxX = as<double>(params["maxX"]);
  }
  arma::vec rangeX(2);
  rangeX(0)=minX;
  rangeX(1)=maxX;
  // rangeX(0)=minX-1*s_X;   // Assuming X is mean 0 and sd 1, this gives 3 sds beyond
  // rangeX(1)=maxX+1*s_X;   // buffer for the basis beyond the range of the
  // observational data
  arma::vec knots = linspace(rangeX(0), rangeX(1), df-degree-1+2);
  knots = knots.subvec(1, df-degree-1);
  arma::mat Xbs = bs_cpp(X, df, knots, degree, true, rangeX);

  arma::vec count(N);
  for (int i=0; i<N; i++) {
    count(i) = sum(Y.row(i));
  }
  // default mu_beta prior
  arma::vec mu_beta(df, arma::fill::zeros);
  if (params.containsElementNamed("mu_beta")) {
    mu_beta = as<vec>(params["mu_beta"]);
  }
  // default Sigma_beta prior
  arma::mat Sigma_beta(df, df, arma::fill::eye);
  Sigma_beta *= 5.0;
  // Sigma_beta *= 100.0;
  if (params.containsElementNamed("Sigma_beta")) {
    Sigma_beta = as<mat>(params["Sigma_beta"]);
  }

  // initialize values
  arma::mat beta(df, d);
  for (int j=0; j<d; j++) {
    beta.col(j) = mvrnormArmaVec(mu_beta, Sigma_beta);
  }
  if (params.containsElementNamed("beta")) {
    beta = as<mat>(params["beta"]);
  }
  bool sample_beta = true;
  if (params.containsElementNamed("sample_beta")) {
    sample_beta = as<bool>(params["sample_beta"]);
  }

  arma::mat Sigma_beta_chol = chol(Sigma_beta);
  arma::mat Sigma_beta_inv = inv_sympd(Sigma_beta);
  arma::mat alpha(N, d, arma::fill::zeros);
  for (int i=0; i<N; i++) {
    alpha.row(i) = (Xbs.row(i) * beta);
  }

  //
  // Default value for sigma2
  //

  double lambda_sigma2 = R::rgamma(0.5, 1.0 / s2_sigma2);
  double sigma2 = std::min(R::rgamma(0.5, 1.0 / lambda_sigma2), 5.0);
  if (params.containsElementNamed("sigma2")) {
    sigma2 = as<double>(params["sigma2"]);
  }
  bool sample_sigma2 = true;
  if (params.containsElementNamed("sample_sigma2")) {
    sample_sigma2 = as<bool>(params["sample_sigma2"]);
  }
  double sigma = sqrt(sigma2);

  // setup save variables
  int n_save = n_mcmc / n_thin;
  arma::cube alpha_save(n_save, N, d, arma::fill::zeros);
  arma::cube beta_save(n_save, df, d, arma::fill::zeros);
  arma::vec sigma2_save(n_save, arma::fill::zeros);
  arma::mat X_save(n_save, N-N_obs, arma::fill::zeros);
  // arma::cube Xbs_save(n_save, N, df, arma::fill::zeros);

  // Initialize tuning
  // arma::mat beta_tune(df, d, arma::fill::ones);
  // beta_tune *= beta_tune_tmp;
  // arma::mat beta_accept(df, d, arma::fill::zeros);
  arma::vec beta_accept(d, arma::fill::zeros);
  arma::vec beta_accept_batch(d, arma::fill::zeros);
  arma::cube beta_batch(50, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune(df, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune_chol(df, df, d, arma::fill::zeros);
  for(int j=0; j<d; j++) {
    Sigma_beta_tune.slice(j).eye();
    Sigma_beta_tune_chol.slice(j) = chol(Sigma_beta_tune.slice(j));
  }

  double sigma2_accept = 0.0;
  double sigma2_accept_batch = 0.0;
  arma::vec X_tune(N-N_obs, arma::fill::ones);
  X_tune *= X_tune_tmp;
  arma::vec X_accept(N-N_obs, arma::fill::zeros);

  Rprintf("Starting MCMC Adaptive Tuning, running for %d iterations \n", n_adapt);
  // set up output messages
  std::ofstream file_out;
  file_out.open(file_name, std::ios_base::app);
  file_out << "Starting MCMC adaptation for chain " << n_chain <<
    ", running for " << n_adapt << " iterations \n";
  // close output file
  file_out.close();

  //
  // Start MCMC chain
  //

  //
  // Adaptive Phase
  //

  for (int k = 0; k < n_adapt; k++) {
    if ((k + 1) % message == 0) {
      Rprintf("Adaptation Iteration %d\n", k+1);
      // set up output messages
      std::ofstream file_out;
      file_out.open(file_name, std::ios_base::app);
      file_out << "MCMC Adaptive Iteration " << k+1 << " for chain " <<
        n_chain << "\n";
      // close output file
      file_out.close();
    }

    Rcpp::checkUserInterrupt();

    //
    // Sample beta - block MH
    //

    if (sample_beta) {
      for (int j=0; j<d; j++) {
        arma::mat beta_star = beta;
        beta_star.col(j) +=
          mvrnormArmaVecChol(zeros_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));

        arma::mat alpha_star = alpha;
        for (int i=0; i<N; i++) {
          alpha_star.row(i) = (Xbs.row(i) * beta_star);
        }
        arma::vec devs_star = beta_star.col(j) - mu_beta;
        arma::vec devs = beta.col(j) - mu_beta;
        double mh1 = LL(alpha_star, Y, N_obs, sigma, d) -
          as_scalar(0.5 * devs_star.t() * Sigma_beta_inv * devs_star);
        double mh2 = LL(alpha, Y, N_obs, sigma, d) -
          as_scalar(0.5 * devs.t() * Sigma_beta_inv * devs);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha = alpha_star;
          beta_accept_batch(j) += 1.0 / 50.0;
        }
      }
      beta_batch.subcube(k % 50, 0, 0, k % 50, df-1, d-1) = beta;
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuningMVMat(k, beta_accept_batch, lambda_beta_tune,
                          beta_batch, Sigma_beta_tune,
                          Sigma_beta_tune_chol);
      }
    }

    //
    // sample sigma2
    //

    if (sample_sigma2) {
      double sigma2_star = sigma2 + R::rnorm(0.0, sigma2_tune);
      if (sigma2_star > 0.0) {
        double sigma_star = sqrt(sigma2_star);
        double mh1 = R::dgamma(sigma2_star, 0.5, 1.0 / lambda_sigma2, 1);
        double mh2 = R::dgamma(sigma2, 0.5, 1.0 / lambda_sigma2, 1);
        for (int i=0; i<N; i++) {
          for (int j=0; j<d; j++) {
            mh1 += R::dnorm(Y(i, j), alpha(i, j), sigma_star, true);
            mh2 += R::dnorm(Y(i, j), alpha(i, j), sigma, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          sigma2 = sigma2_star;
          sigma = sigma_star;
          sigma2_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, sigma2_accept_batch, sigma2_tune);
      }
    }

    //
    // sample lambda_sigma2
    //

    lambda_sigma2 = R::rgamma(1.0, 1.0 / (s2_sigma2 + sigma2));



    //
    // sample X-MH
    //

    if (sample_X) {
      if (sample_X_mh) {
        for (int i=N_obs; i<N; i++) {
          arma::vec X_star = X;
          arma::mat alpha_star = alpha;
          X_star(i) = R::rnorm(X(i), X_tune(i-N_obs));
          arma::mat Xbs_star = Xbs;
          Xbs_star.row(i) = bs_cpp(X_star.row(i), df, knots, degree, true, rangeX);
          // arma::mat Xbs_star = bs_cpp(X_star, df, knots, degree, true, rangeX);
          alpha_star.row(i) = (Xbs_star.row(i) * beta);
          arma::rowvec Y_row = Y.row(i);
          double mh1 = mhX(X_star(i), mu_X, s2_X, alpha_star.row(i), Y_row, sigma, d);
          double mh2 = mhX(X(i), mu_X, s2_X, alpha.row(i), Y_row, sigma, d);
          double mh = exp(mh1 - mh2);
          if (mh > R::runif(0, 1)) {
            X = X_star;
            Xbs = Xbs_star;
            alpha = alpha_star;
            X_accept(i-N_obs) += 1.0 / 50.0;
          }
        }
        if((k+1) % 50 == 0){
          updateTuningVec(k, X_accept, X_tune);
        }
      } else {
        // elliptical slice sampler
        for (int i=N_obs; i<N; i++) {
          double X_prior = X(i);
          X_prior = R::rnorm(0.0, s_X);
          Rcpp::List X_ess = ess_X(X(i), X_prior, mu_X, beta,
                                   alpha.row(i), Y.row(i), sigma, Xbs.row(i),
                                   d, knots, df, degree, rangeX, file_name,
                                   n_chain);
          X(i) = as<double>(X_ess["X"]);
          Xbs.row(i) = as<rowvec>(X_ess["Xbs"]);
          alpha.row(i) = as<rowvec>(X_ess["alpha"]);
        }
      }
    }

  }

  //
  // Fitting Phase
  //

  Rprintf("Starting MCMC fitting, running for %d iterations \n", n_mcmc);
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  file_out << "Starting MCMC fit for chain " << n_chain <<
    ", running for " << n_mcmc << " iterations \n";
  // close output file
  file_out.close();

  for (int k = 0; k < n_mcmc; k++) {
    if ((k + 1) % message == 0) {
      Rprintf("MCMC Fitting Iteration %d\n", k+1);
      // set up output messages
      std::ofstream file_out;
      file_out.open(file_name, std::ios_base::app);
      file_out << "MCMC Fitting Iteration " << k+1 << " for chain " <<
        n_chain << "\n";
      // close output file
      file_out.close();
    }

    Rcpp::checkUserInterrupt();

    //
    // Sample beta - block MH
    //

    if (sample_beta) {
      for (int j=0; j<d; j++) {
        arma::mat beta_star = beta;
        beta_star.col(j) +=
          mvrnormArmaVecChol(zeros_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));

        arma::mat alpha_star = alpha;
        for (int i=0; i<N; i++) {
          alpha_star.row(i) = (Xbs.row(i) * beta_star);
        }
        arma::vec devs_star = beta_star.col(j) - mu_beta;
        arma::vec devs = beta.col(j) - mu_beta;
        double mh1 = LL(alpha_star, Y, N_obs, sigma, d) -
          as_scalar(0.5 * devs_star.t() * Sigma_beta_inv * devs_star);
        double mh2 = LL(alpha, Y, N_obs, sigma, d) -
          as_scalar(0.5 * devs.t() * Sigma_beta_inv * devs);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha = alpha_star;
          beta_accept(j) += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample sigma2
    //

    if (sample_sigma2) {
      double sigma2_star = sigma2 + R::rnorm(0.0, sigma2_tune);
      if (sigma2_star > 0.0) {
        double sigma_star = sqrt(sigma2_star);
        double mh1 = R::dgamma(sigma2_star, 0.5, 1.0 / lambda_sigma2, 1);
        double mh2 = R::dgamma(sigma2, 0.5, 1.0 / lambda_sigma2, 1);
        for (int i=0; i<N; i++) {
          for (int j=0; j<d; j++) {
            mh1 += R::dnorm(Y(i, j), alpha(i, j), sigma_star, true);
            mh2 += R::dnorm(Y(i, j), alpha(i, j), sigma, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          sigma2 = sigma2_star;
          sigma = sigma_star;
          sigma2_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample lambda_sigma2
    //

    lambda_sigma2 = R::rgamma(1.0, 1.0 / (s2_sigma2 + sigma2));

    // //
    // // sample X - ESS
    // //
    //


    if (sample_X) {
      if (sample_X_mh) {
        for (int i=N_obs; i<N; i++) {
          arma::vec X_star = X;
          arma::mat alpha_star = alpha;
          X_star(i) = R::rnorm(X(i), X_tune(i-N_obs));
          arma::mat Xbs_star = Xbs;
          Xbs_star.row(i) = bs_cpp(X_star.row(i), df, knots, degree, true, rangeX);
          // arma::mat Xbs_star = bs_cpp(X_star, df, knots, degree, true, rangeX);
          alpha_star.row(i) = (Xbs_star.row(i) * beta);
          arma::rowvec Y_row = Y.row(i);
          double mh1 = mhX(X_star(i), mu_X, s2_X, alpha_star.row(i), Y_row, sigma, d);
          double mh2 = mhX(X(i), mu_X, s2_X, alpha.row(i), Y_row, sigma, d);
          double mh = exp(mh1 - mh2);
          if (mh > R::runif(0, 1)) {
            X = X_star;
            Xbs = Xbs_star;
            alpha = alpha_star;
            X_accept(i-N_obs) += 1.0 / n_mcmc;
          }
        }
        if((k+1) % 50 == 0){
          updateTuningVec(k, X_accept, X_tune);
        }
      } else {
        for (int i=N_obs; i<N; i++) {
          double X_prior = X(i);
          X_prior = R::rnorm(0.0, s_X);
          Rcpp::List X_ess = ess_X(X(i), X_prior, mu_X, beta,
                                   alpha.row(i), Y.row(i), sigma, Xbs.row(i),
                                   d, knots, df, degree, rangeX, file_name,
                                   n_chain);
          X(i) = as<double>(X_ess["X"]);
          Xbs.row(i) = as<rowvec>(X_ess["Xbs"]);
          alpha.row(i) = as<rowvec>(X_ess["alpha"]);
        }
      }
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      alpha_save(save_idx, 0, 0, size(1, N, d)) = alpha;
      beta_save(save_idx, 0, 0, size(1, df, d)) = beta;
      sigma2_save(save_idx) = sigma2;
      X_save.submat(save_idx, 0, size(1, N-N_obs))= X(span(N_obs, N-1)).t() + mu_X;
      // Xbs_save.subcube(save_idx, 0, 0, size(1, N, df)) = Xbs;
    }
  }

  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  file_out << "Average acceptance rate for beta  = " << mean(beta_accept) <<
    " for chain " << n_chain << "\n";
  file_out << "Average acceptance rate for X  = " << mean(X_accept) <<
  " for chain " << n_chain << "\n";
  // file_out << "Average acceptance rate for xi  = " << mean(xi_accept) <<
  //   " for chain " << n_chain << "\n";
  // file_out << "Average acceptance rate for tau2  = " << mean(tau2_accept) <<
  //   " for chain " << n_chain << "\n";
  // close output file
  file_out.close();

  return Rcpp::List::create(
    _["alpha"] = alpha_save,
    _["beta"] = beta_save,
    // _["Xbs"] = Xbs_save,
    _["sigma2"] = sigma2_save,
    _["knots"] = knots,
    _["X"] = X_save);
}

