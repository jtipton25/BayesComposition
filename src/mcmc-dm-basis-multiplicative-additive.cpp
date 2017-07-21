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

// Multinomial Multi-logit multivariate Basis regression for Inverse Inference

// Author: John Tipton
//
// Created 05.15.2017
// Last updated 07.19.2017

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppDMBasisMultiplicativeAdditive (const arma::mat& Y, const arma::vec& X,
                                            List params,
                                            int n_chain=1, bool pool_s2_tau2=true,
                                            std::string file_name="DM-fit",
                                            std::string corr_function="exponential") {

  // Load parameters
  int n_adapt = as<int>(params["n_adapt"]);
  int n_mcmc = as<int>(params["n_mcmc"]);
  int n_thin = as<int>(params["n_thin"]);
  // default to message output every 500 iterations
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
  // default output message iteration
  int message = 500;
  if (params.containsElementNamed("message")) {
    message = as<int>(params["message"]);
  }


  // set up dimensions
  double N = Y.n_rows;
  double d = Y.n_cols;

  // // set up dimensions
  // double N_pred = Y_pred.n_rows;

  // double B = Rf_choose(d, 2);
  double B = Rf_choose(d, 2);

  // count - sum of counts at each site
  arma::vec count(N);
  for (int i=0; i<N; i++) {
    count(i) = sum(Y.row(i));
  }

  // constant vectors
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec ones_B(B, arma::fill::ones);
  arma::vec zero_df(df, arma::fill::zeros);
  arma::vec zero_d(d, arma::fill::zeros);

  // default normal prior for regression coefficients
  arma::vec mu_beta(df, arma::fill::zeros);
  if (params.containsElementNamed("mu_beta")) {
    mu_beta = as<vec>(params["mu_beta"]);
  }
  // default prior for overall mean mu
  arma::mat Sigma_beta(df, df, arma::fill::eye);
  Sigma_beta *= 1.0;
  if (params.containsElementNamed("Sigma_beta")) {
    Sigma_beta = as<mat>(params["Sigma_beta"]);
  }
  arma::mat Sigma_beta_inv = inv_sympd(Sigma_beta);
  arma::mat Sigma_beta_chol = chol(Sigma_beta);

  // default half cauchy scale for Covariance diagonal variance tau2
  double s2_tau2 = 1.0;
  if (params.containsElementNamed("s2_tau2")) {
    s2_tau2 = as<double>(params["s2_tau2"]);
  }
  // default half cauchy scale for Covariance diagonal variance tau2
  double A_s2 = 1.0;
  if (params.containsElementNamed("A_s2")) {
    A_s2 = as<double>(params["A_s2"]);
  }
  // default half cauchy scale for additive covariance
  double s2_tau2_additive = 1.0;
  if (params.containsElementNamed("s2_tau2_additive")) {
    s2_tau2_additive = as<double>(params["s2_tau2_additive"]);
  }
  // default half cauchy scale for additive covariance
  double A_s2_additive = 1.0;
  if (params.containsElementNamed("A_s2_additive")) {
    A_s2_additive = as<double>(params["A_s2_additive"]);
  }
  // default xi LKJ concentation parameter of 1
  double eta = 1.0;
  if (params.containsElementNamed("eta")) {
    eta = as<double>(params["eta"]);
  }
  // default xi_additive LKJ concentation parameter of 1
  double eta_additive = 1.0;
  if (params.containsElementNamed("eta_additive")) {
    eta_additive = as<double>(params["eta_additive"]);
  }

  // default beta tuning parameter
  arma::vec lambda_beta_tune(d, arma::fill::ones);
  lambda_beta_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_beta_tune")) {
    lambda_beta_tune = as<vec>(params["lambda_beta_tune"]);
  }
  // default lambda_epsilon tuning parameter standard deviation of 0.25
  double lambda_epsilon_tune_tmp = 0.25;
  if (params.containsElementNamed("lambda_epsilon_tune")) {
    lambda_epsilon_tune_tmp = as<double>(params["lambda_epsilon_tune"]);
  }
  arma::vec lambda_epsilon_tune(N, arma::fill::ones);
  lambda_epsilon_tune *= lambda_epsilon_tune_tmp;

  // default tau2 tuning parameter
  double lambda_tau2_tune = 0.25;
  if (params.containsElementNamed("lambda_tau2_tune")) {
    lambda_tau2_tune = as<double>(params["lambda_tau2_tune"]);
  }
  // default xi tuning parameter
  double lambda_xi_tune = 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_xi_tune")) {
    lambda_xi_tune = as<double>(params["lambda_xi_tune"]);
  }
  // default tau2_additive tuning parameter
  double lambda_tau2_additive_tune = 0.25;
  if (params.containsElementNamed("lambda_tau2_additive_tune")) {
    lambda_tau2_additive_tune = as<double>(params["lambda_tau2_additive_tune"]);
  }

  // default xi_additive tuning parameter
  double lambda_xi_additive_tune = 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_xi_additive_tune")) {
    lambda_xi_additive_tune = as<double>(params["lambda_xi_additive_tune"]);
  }

  double minX = min(X);
  if (params.containsElementNamed("minX")) {
    minX = as<double>(params["minX"]);
  }
  double maxX = max(X);
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

  //
  // initialize values
  //

  //
  // set default, fixed parameters and turn on/off samplers for testing
  //

  //
  // Default for beta
  //

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

  //
  // Variance parameter tau2 and hyperprior lambda_tau2
  //

  arma::vec lambda_tau2(d);
  arma::vec tau2(d);
  for (int j=0; j<d; j++) {
    lambda_tau2(j) = R::rgamma(0.5, 1.0 / s2_tau2);
    tau2(j) = std::max(std::min(R::rgamma(0.5, 1.0 / lambda_tau2(j)), 5.0), 1.0);
  }
  arma::vec tau = sqrt(tau2);
  if (params.containsElementNamed("tau2")) {
    tau2 = as<vec>(params["tau2"]);
    tau = sqrt(tau2);
  }
  bool sample_tau2 = true;
  if (params.containsElementNamed("sample_tau2")) {
    sample_tau2 = as<bool>(params["sample_tau2"]);
  }

  // additive variance
  arma::vec lambda_tau2_additive(d);
  arma::vec tau2_additive(d);
  for (int j=0; j<d; j++) {
    lambda_tau2_additive(j) = R::rgamma(0.5, 1.0 / s2_tau2_additive);
    tau2_additive(j) = std::max(std::min(R::rgamma(0.5, 1.0 / lambda_tau2_additive(j)), 5.0), 1.0);
  }
  arma::vec tau_additive = sqrt(tau2_additive);
  if (params.containsElementNamed("tau2_additive")) {
    tau2_additive = as<vec>(params["tau2_additive"]);
    tau_additive = sqrt(tau2_additive);
  }
  bool sample_tau2_additive = true;
  if (params.containsElementNamed("sample_tau2_additive")) {
    sample_tau2_additive = as<bool>(params["sample_tau2_additive"]);
  }

  //
  // Default LKJ hyperparameter xi
  //

  arma::vec eta_vec(B);
  int idx_eta = 0;
  for (int j=0; j<d; j++) {
    for (int k=0; k<(d-j-1); k++) {
      eta_vec(idx_eta+k) = eta + (d - 2.0 - j) / 2.0;
    }
    idx_eta += d-j-1;
  }

  arma::vec xi(B);
  for (int b=0; b<B; b++) {
    xi(b) = 2.0 * R::rbeta(eta_vec(b), eta_vec(b)) - 1.0;
  }
  arma::vec xi_tilde(B);
  if (params.containsElementNamed("xi")) {
    xi = as<vec>(params["xi"]);
  }
  for (int b=0; b<B; b++) {
    xi_tilde(b) = 0.5 * (xi(b) + 1.0);
  }
  bool sample_xi = true;
  if (params.containsElementNamed("sample_xi")) {
    sample_xi = as<bool>(params["sample_xi"]);
  }
  Rcpp::List R_out = makeRLKJ(xi, d, true, true);
  double log_jacobian = as<double>(R_out["log_jacobian"]);
  arma::mat R = as<mat>(R_out["R"]);
  arma::mat R_tau = R * diagmat(tau);

  //
  // Default LKJ hyperparameter xi_additive
  //

  arma::vec eta_vec_additive(B);
  int idx_eta_additive = 0;
  for (int j=0; j<(d-1); j++) {
    for (int k=0; k<(d-j-1); k++) {
      eta_vec_additive(idx_eta_additive+k) = eta_additive + (d - 2.0 - j) / 2.0;
    }
    idx_eta_additive += d-j-1;
  }
  arma::vec xi_additive(B);
  for (int b=0; b<B; b++) {
    xi_additive(b) = 2.0 * R::rbeta(eta_vec_additive(b), eta_vec_additive(b)) - 1.0;
  }
  arma::vec xi_tilde_additive(B);
  if (params.containsElementNamed("xi_additive")) {
    xi_additive = as<vec>(params["xi_additive"]);
  }
  for (int b=0; b<B; b++) {
    xi_tilde_additive(b) = 0.5 * (xi_additive(b) + 1.0);
  }
  bool sample_xi_additive = true;
  if (params.containsElementNamed("sample_xi_additive")) {
    sample_xi_additive = as<bool>(params["sample_xi_additive"]);
  }
  Rcpp::List R_out_additive = makeRLKJ(xi_additive, d, true, true);
  double log_jacobian_additive = as<double>(R_out_additive["log_jacobian"]);
  arma::mat R_additive = as<mat>(R_out_additive["R"]);
  arma::mat R_tau_additive = R_additive * diagmat(tau_additive);

  // sample random additive noise
  arma::mat epsilon(N, d, arma::fill::randn);
  bool sample_epsilon = false;
  if (params.containsElementNamed("sample_epsilon")) {
    sample_epsilon = as<bool>(params["sample_epsilon"]);
  }

  arma::mat alpha = exp(Xbs * beta * R_tau + epsilon);

  // setup save variables
  int n_save = n_mcmc / n_thin;
  arma::cube alpha_save(n_save, N, d, arma::fill::zeros);
  arma::cube beta_save(n_save, df, d, arma::fill::zeros);
  arma::cube epsilon_save(n_save, N, d, arma::fill::zeros);
  arma::mat tau2_save(n_save, d, arma::fill::zeros);
  arma::mat lambda_tau2_save(n_save, d, arma::fill::zeros);
  arma::vec s2_tau2_save(n_save, arma::fill::zeros);
  arma::mat tau2_additive_save(n_save, d, arma::fill::zeros);
  arma::vec s2_tau2_additive_save(n_save, arma::fill::zeros);
  arma::cube R_save(n_save, d, d, arma::fill::zeros);
  arma::cube R_tau_save(n_save, d, d, arma::fill::zeros);
  arma::mat xi_save(n_save, B, arma::fill::zeros);
  arma::cube R_additive_save(n_save, d, d, arma::fill::zeros);
  arma::cube R_tau_additive_save(n_save, d, d, arma::fill::zeros);
  arma::mat xi_additive_save(n_save, B, arma::fill::zeros);

  // initialize tuning

  double s2_tau2_accept = 0.0;
  double s2_tau2_accept_batch = 0.0;
  double s2_tau2_tune = 1.0;
  double s2_tau2_additive_accept = 0.0;
  double s2_tau2_additive_accept_batch = 0.0;
  double s2_tau2_additive_tune = 1.0;
  double tau2_accept = 0.0;
  double tau2_accept_batch = 0.0;
  arma::mat tau2_batch(50, d, arma::fill::zeros);
  arma::mat Sigma_tau2_tune(d, d, arma::fill::eye);
  arma::mat Sigma_tau2_tune_chol = chol(Sigma_tau2_tune);
  double tau2_additive_accept = 0.0;
  double tau2_additive_accept_batch = 0.0;
  arma::mat tau2_additive_batch(50, d, arma::fill::zeros);
  arma::mat Sigma_tau2_additive_tune(d, d, arma::fill::eye);
  arma::mat Sigma_tau2_additive_tune_chol = chol(Sigma_tau2_additive_tune);
  double xi_accept = 0.0;
  double xi_accept_batch = 0.0;
  arma::mat xi_batch(50, B, arma::fill::zeros);
  arma::mat Sigma_xi_tune(B, B, arma::fill::eye);
  arma::mat Sigma_xi_tune_chol = chol(Sigma_xi_tune);
  double xi_additive_accept = 0.0;
  double xi_additive_accept_batch = 0.0;
  arma::mat xi_additive_batch(50, B, arma::fill::zeros);
  arma::mat Sigma_xi_additive_tune(B, B, arma::fill::eye);
  arma::mat Sigma_xi_additive_tune_chol = chol(Sigma_xi_additive_tune);
  arma::vec beta_accept(d, arma::fill::zeros);
  arma::vec beta_accept_batch(d, arma::fill::zeros);
  arma::cube beta_batch(50, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune(df, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune_chol(df, df, d, arma::fill::zeros);
  for(int j=0; j<d; j++) {
    Sigma_beta_tune.slice(j).eye();
    Sigma_beta_tune_chol.slice(j) = chol(Sigma_beta_tune.slice(j));
  }
  arma::vec epsilon_accept(N, arma::fill::zeros);
  arma::vec epsilon_accept_batch(N, arma::fill::zeros);
  arma::cube epsilon_batch(50, d, N, arma::fill::zeros);
  arma::cube Sigma_epsilon_tune(d, d, N, arma::fill::zeros);
  arma::cube Sigma_epsilon_tune_chol(d, d, N, arma::fill::zeros);
  for(int i=0; i<N; i++) {
    Sigma_epsilon_tune.slice(i).eye();
    Sigma_epsilon_tune_chol.slice(i) = chol(Sigma_epsilon_tune.slice(i));
  }

  Rprintf("Starting MCMC adaptation for chain %d, running for %d iterations \n",
          n_chain, n_adapt);
  // set up output messages
  std::ofstream file_out;
  file_out.open(file_name, std::ios_base::app);
  file_out << "Starting MCMC adaptation for chain " << n_chain <<
    ", running for " << n_adapt << " iterations \n";
  // close output file
  file_out.close();
  // }

  // Start MCMC chain
  for (int k=0; k<n_adapt; k++) {
    if ((k+1) % message == 0) {
      Rprintf("MCMC Adaptive Iteration %d for chain %d\n", k+1, n_chain);
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
          mvrnormArmaVecChol(zero_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));
        arma::mat alpha_star = exp(Xbs * beta_star * R_tau + epsilon);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          dMVN(beta_star.col(j), mu_beta, Sigma_beta_chol);
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          dMVN(beta.col(j), mu_beta, Sigma_beta_chol);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha = alpha_star;
          beta_accept_batch(j) += 1.0 / 50.0;
        }
      }
      // update tuning
      beta_batch.subcube(k % 50, 0, 0, k % 50, df-1, d-1) = beta;
      if ((k+1) % 50 == 0){
        updateTuningMVMat(k, beta_accept_batch, lambda_beta_tune,
                          beta_batch, Sigma_beta_tune,
                          Sigma_beta_tune_chol);
      }
    }

    //
    // sample epsilon
    //

    if (sample_epsilon) {
      // Metroplois-Hastings
      for (int i=0; i<N; i++) {
        arma::mat epsilon_star = epsilon;
        epsilon_star.col(i) += mvrnormArmaVecChol(zero_d,
                         lambda_epsilon_tune(i) * Sigma_epsilon_tune_chol.slice(i));;
        arma::mat alpha_star = exp(Xbs * beta * R_tau + epsilon_star);
        double mh1 = dMVN(epsilon_star.row(i).t(), zero_d, R_tau_additive, true) +
          LL_DM_row(alpha_star.row(i), Y.row(i), d, count(i));
        double mh2 = dMVN(epsilon.row(i).t(), zero_d, R_tau_additive, true) +
          LL_DM_row(alpha.row(i), Y.row(i),  d, count(i));
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          epsilon = epsilon_star;
          alpha = alpha_star;
          epsilon_accept_batch(i) += 1.0 / 50.0;
        }
      }
      // update tuning
      epsilon_batch.subcube(k % 50, 0, 0, k % 50, N-1, d-1) = epsilon;
      if ((k+1) % 50 == 0){
        updateTuningMVMat(k, epsilon_accept_batch, lambda_epsilon_tune,
                          epsilon_batch, Sigma_epsilon_tune,
                          Sigma_epsilon_tune_chol);
      }
    }

    //
    // sample tau2
    //

    if (sample_tau2) {
      arma::vec log_tau2_star = log(tau2);
      log_tau2_star =
        mvrnormArmaVecChol(log(tau2),
                           lambda_tau2_tune * Sigma_tau2_tune_chol);
      arma::vec tau2_star = exp(log_tau2_star);
      if (all(tau2_star > 0.0)) {
        arma::vec tau_star = sqrt(tau2_star);
        arma::mat R_tau_star = R * diagmat(tau_star);
        arma::mat alpha_star = exp(Xbs * beta * R_tau_star + epsilon);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          // jacobian of log-scale proposal
          sum(log_tau2_star);
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          // jacobian of log-scale proposal
          sum(log(tau2));
        // prior
        for (int j=0; j<(d-1); j++) {
          mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2 = tau2_star;
          tau = tau_star;
          R_tau = R_tau_star;
          alpha = alpha_star;
          tau2_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      tau2_batch.row(k % 50) = log(tau2).t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, tau2_accept_batch, lambda_tau2_tune, tau2_batch,
                       Sigma_tau2_tune, Sigma_tau2_tune_chol);
      }
    }

    //
    // sample lambda_tau2
    //

    for (int j=0; j<(d-1); j++) {
      lambda_tau2(j) = R::rgamma(1.0, 1.0 / (s2_tau2 + tau2(j)));
    }

    //
    // sample s2_tau2
    //

    if (pool_s2_tau2) {
      double s2_tau2_star = s2_tau2 + R::rnorm(0, s2_tau2_tune);
      if (s2_tau2_star > 0 && s2_tau2_star < A_s2) {
        double mh1 = 0.0;
        double mh2 = 0.0;
        for (int j=0; j<(d-1); j++) {
          mh1 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2_star, true);
          mh2 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          s2_tau2 = s2_tau2_star;
          s2_tau2_accept += 1.0 / n_mcmc;
          s2_tau2_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, s2_tau2_accept_batch, s2_tau2_tune);
      }
    }

    //
    // sample xi - MH
    //

    if (sample_xi) {
      arma::vec logit_xi_tilde_star = mvrnormArmaVecChol(logit(xi_tilde),
                                                         lambda_xi_tune * Sigma_xi_tune_chol);
      arma::vec xi_tilde_star = expit(logit_xi_tilde_star);
      arma::vec xi_star = 2.0 * xi_tilde_star - 1.0;
      if (all(xi_star > -1.0) && all(xi_star < 1.0)) {
        Rcpp::List R_out = makeRLKJ(xi_star, d-1, true, true);
        arma::mat R_star = as<mat>(R_out["R"]);
        arma::mat R_tau_star = R_star * diagmat(tau);
        arma::mat alpha_star = exp(Xbs * beta * R_tau_star + epsilon);

        double log_jacobian_star = as<double>(R_out["log_jacobian"]);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          // Jacobian adjustment
          sum(log(xi_tilde_star) + log(ones_B - xi_tilde_star));
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          // Jacobian adjustment
          sum(log(xi_tilde) + log(ones_B - xi_tilde));
        for (int b=0; b<B; b++) {
          mh1 += R::dbeta(0.5 * (xi_star(b) + 1.0), eta_vec(b), eta_vec(b), true);
          mh2 += R::dbeta(0.5 * (xi(b) + 1.0), eta_vec(b), eta_vec(b), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          xi_tilde = xi_tilde_star;
          xi = xi_star;
          R = R_star;
          R_tau = R_tau_star;
          log_jacobian = log_jacobian_star;
          alpha = alpha_star;
          xi_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      xi_batch.row(k % 50) = logit(xi_tilde).t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, xi_accept_batch, lambda_xi_tune, xi_batch,
                       Sigma_xi_tune, Sigma_xi_tune_chol);
      }
    }

    //
    // sample tau2_additive
    //

    if (sample_tau2_additive) {
      arma::vec log_tau2_additive_star = log(tau2_additive);
      log_tau2_additive_star = mvrnormArmaVecChol(log(tau2_additive),
                                                  lambda_tau2_additive_tune * Sigma_tau2_additive_tune_chol);
      arma::vec tau2_additive_star = exp(log_tau2_additive_star);
      if (all(tau2_additive_star > 0.0)) {
        arma::vec tau_additive_star = sqrt(tau2_additive_star);
        arma::mat R_tau_additive_star = R_additive * diagmat(tau_additive_star);
        double mh1 = sum(log_tau2_additive_star);  // jacobian of log-scale proposal
        double mh2 = sum(log(tau2_additive));      // jacobian of log-scale proposal
        for (int i=0; i<N; i++) {
          mh1 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive_star, true);
          mh2 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += R::dgamma(tau2_additive_star(j), 0.5, 1.0 / lambda_tau2_additive(j), true);
          mh2 += R::dgamma(tau2_additive(j), 0.5, 1.0 / lambda_tau2_additive(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2_additive = tau2_additive_star;
          tau_additive = tau_additive_star;
          R_tau_additive = R_tau_additive_star;
          tau2_additive_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      tau2_additive_batch.row(k % 50) = log(tau2_additive).t();
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuningMV(k, tau2_additive_accept_batch, lambda_tau2_additive_tune, tau2_additive_batch,
                       Sigma_tau2_additive_tune, Sigma_tau2_additive_tune_chol);
      }

    }

    //
    // sample lambda_tau2_additive
    //

    for (int j=0; j<d; j++) {
      lambda_tau2_additive(j) = R::rgamma(1.0, 1.0 / (s2_tau2_additive + tau2_additive(j)));
    }

    //
    // sample s2_tau2_additive
    //

    double s2_tau2_additive_star = s2_tau2_additive + R::rnorm(0, s2_tau2_additive_tune);
    if (s2_tau2_additive_star > 0 && s2_tau2_additive_star < A_s2_additive) {
      double mh1 = 0.0;
      double mh2 = 0.0;
      for (int j=0; j<d; j++){
        mh1 += R::dgamma(lambda_tau2_additive(j), 0.5, 1.0 / s2_tau2_additive_star, true);
        mh2 += R::dgamma(lambda_tau2_additive(j), 0.5, 1.0 / s2_tau2_additive, true);
      }
      double mh = exp(mh1-mh2);
      if (mh > R::runif(0.0, 1.0)) {
        s2_tau2_additive = s2_tau2_additive_star;
        s2_tau2_additive_accept_batch += 1.0 / 50.0;
      }
    }

    //
    // sample xi_additive - MH
    //

    if (sample_xi_additive) {
      arma::vec logit_xi_tilde_additive_star = mvrnormArmaVecChol(logit(xi_tilde_additive),
                                                                  lambda_xi_additive_tune * Sigma_xi_additive_tune_chol);
      arma::vec xi_tilde_additive_star = expit(logit_xi_tilde_additive_star);
      arma::vec xi_additive_star = 2.0 * xi_tilde_additive_star - 1.0;
      // arma::vec xi_star =  mvrnormArmaVecChol(xi, lambda_xi_tune * Sigma_xi_tune_chol);
      if (all(xi_additive_star > -1.0) && all(xi_additive_star < 1.0)) {
        Rcpp::List R_out_additive = makeRLKJ(xi_additive_star, d, true, true);
        arma::mat R_additive_star = as<mat>(R_out_additive["R"]);
        arma::mat R_tau_additive_star = R_additive_star * diagmat(tau_additive);
        double log_jacobian_additive_star = as<double>(R_out_additive["log_jacobian_additive"]);
        double mh1 = sum(log(xi_tilde_additive_star) + log(ones_B - xi_tilde_additive_star));
        // Jacobian adjustment
        double mh2 = sum(log(xi_tilde_additive) + log(ones_B - xi_tilde_additive));
        // Jacobian adjustment
        for (int b=0; b<B; b++) {
          mh1 += R::dbeta(0.5 * (xi_additive_star(b) + 1.0), eta_vec_additive(b), eta_vec_additive(b), true);
          mh2 += R::dbeta(0.5 * (xi_additive(b) + 1.0), eta_vec_additive(b), eta_vec_additive(b), true);
        }
        for (int i=0; i<N; i++) {
          mh1 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive_star, true);
          mh2 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          xi_tilde_additive = xi_tilde_additive_star;
          xi_additive = xi_additive_star;
          R_additive = R_additive_star;
          R_tau_additive = R_tau_additive_star;
          log_jacobian_additive = log_jacobian_additive_star;
          xi_additive_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      xi_additive_batch.row(k % 50) = logit(xi_tilde_additive).t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, xi_additive_accept_batch, lambda_xi_additive_tune, xi_additive_batch,
                       Sigma_xi_additive_tune, Sigma_xi_additive_tune_chol);
      }
    }
    // end of adaptation loop
  }

  Rprintf("Starting MCMC fit for chain %d, running for %d iterations \n",
          n_chain, n_mcmc);
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  file_out << "Starting MCMC fit for chain " << n_chain <<
    ", running for " << n_mcmc << " iterations \n";
  // close output file
  file_out.close();

  // Start MCMC fitting phase
  for (int k=0; k<n_mcmc; k++) {
    if ((k+1) % message == 0) {
      Rprintf("MCMC Fitting Iteration %d for chain %d\n", k+1, n_chain);
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
          mvrnormArmaVecChol(zero_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));

        arma::mat alpha_star = exp(Xbs * beta_star * R_tau + epsilon);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          dMVN(beta_star.col(j), mu_beta, Sigma_beta_chol);
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          dMVN(beta.col(j), mu_beta, Sigma_beta_chol);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha = alpha_star;
          beta_accept(j) += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample epsilon
    //

    if (sample_epsilon) {
      // Metroplois-Hastings
      for (int i=0; i<N; i++) {
        arma::mat epsilon_star = epsilon;
        epsilon_star.col(i) += mvrnormArmaVecChol(zero_d,
                         lambda_epsilon_tune(i) * Sigma_epsilon_tune_chol.slice(i));;
        arma::mat alpha_star = exp(Xbs * beta * R_tau + epsilon_star);
        double mh1 = dMVN(epsilon_star.row(i).t(), zero_d, R_tau_additive, true) +
          LL_DM_row(alpha_star.row(i), Y.row(i), d, count(i));
        double mh2 = dMVN(epsilon.row(i).t(), zero_d, R_tau_additive, true) +
          LL_DM_row(alpha.row(i), Y.row(i),  d, count(i));
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          epsilon = epsilon_star;
          alpha = alpha_star;
          epsilon_accept(i) += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample tau2
    //

    if (sample_tau2) {
      arma::vec log_tau2_star = log(tau2);
      log_tau2_star =
        mvrnormArmaVecChol(log(tau2),
                           lambda_tau2_tune * Sigma_tau2_tune_chol);
      arma::vec tau2_star = exp(log_tau2_star);
      if (all(tau2_star > 0.0)) {
        arma::vec tau_star = sqrt(tau2_star);
        arma::mat R_tau_star = R * diagmat(tau_star);
        arma::mat alpha_star = exp(Xbs * beta * R_tau_star + epsilon);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          // jacobian of log-scale proposal
          sum(log_tau2_star);
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          // jacobian of log-scale proposal
          sum(log(tau2));
        // prior
        for (int j=0; j<(d-1); j++) {
          mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2 = tau2_star;
          tau = tau_star;
          R_tau = R_tau_star;
          alpha = alpha_star;
          tau2_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample lambda_tau2
    //

    for (int j=0; j<(d-1); j++) {
      lambda_tau2(j) = R::rgamma(1.0, 1.0 / (s2_tau2 + tau2(j)));
    }

    //
    // sample s2_tau2
    //

    if (pool_s2_tau2) {

      double s2_tau2_star = s2_tau2 + R::rnorm(0, s2_tau2_tune);
      if (s2_tau2_star > 0 && s2_tau2_star < A_s2) {
        double mh1 = 0.0;
        double mh2 = 0.0;
        for (int j=0; j<(d-1); j++) {
          mh1 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2_star, true);
          mh2 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          s2_tau2 = s2_tau2_star;
          s2_tau2_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample xi - MH
    //

    if (sample_xi) {
      arma::vec logit_xi_tilde_star = mvrnormArmaVecChol(logit(xi_tilde),
                                                         lambda_xi_tune * Sigma_xi_tune_chol);
      arma::vec xi_tilde_star = expit(logit_xi_tilde_star);
      arma::vec xi_star = 2.0 * xi_tilde_star - 1.0;
      if (all(xi_star > -1.0) && all(xi_star < 1.0)) {
        Rcpp::List R_out = makeRLKJ(xi_star, d-1, true, true);
        arma::mat R_star = as<mat>(R_out["R"]);
        arma::mat R_tau_star = R_star * diagmat(tau);
        arma::mat alpha_star = exp(Xbs * beta * R_tau + epsilon);
        double log_jacobian_star = as<double>(R_out["log_jacobian"]);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          // Jacobian adjustment
          sum(log(xi_tilde_star) + log(ones_B - xi_tilde_star));
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          // Jacobian adjustment
          sum(log(xi_tilde) + log(ones_B - xi_tilde));
        for (int b=0; b<B; b++) {
          mh1 += R::dbeta(0.5 * (xi_star(b) + 1.0), eta_vec(b), eta_vec(b), true);
          mh2 += R::dbeta(0.5 * (xi(b) + 1.0), eta_vec(b), eta_vec(b), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          xi_tilde = xi_tilde_star;
          xi = xi_star;
          R = R_star;
          R_tau = R_tau_star;
          log_jacobian = log_jacobian_star;
          alpha = alpha_star;
          xi_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample tau2_additive
    //

    if (sample_tau2_additive) {
      arma::vec log_tau2_additive_star = log(tau2_additive);
      log_tau2_additive_star = mvrnormArmaVecChol(log(tau2_additive),
                                                  lambda_tau2_additive_tune * Sigma_tau2_additive_tune_chol);
      arma::vec tau2_additive_star = exp(log_tau2_additive_star);
      if (all(tau2_additive_star > 0.0)) {
        arma::vec tau_additive_star = sqrt(tau2_additive_star);
        arma::mat R_tau_additive_star = R_additive * diagmat(tau_additive_star);
        double mh1 = sum(log_tau2_additive_star);  // jacobian of log-scale proposal
        double mh2 = sum(log(tau2_additive));      // jacobian of log-scale proposal
        for (int i=0; i<N; i++) {
          mh1 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive_star, true);
          mh2 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += R::dgamma(tau2_additive_star(j), 0.5, 1.0 / lambda_tau2_additive(j), true);
          mh2 += R::dgamma(tau2_additive(j), 0.5, 1.0 / lambda_tau2_additive(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2_additive = tau2_additive_star;
          tau_additive = tau_additive_star;
          R_tau_additive = R_tau_additive_star;
          tau2_additive_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample lambda_tau2_additive
    //

    for (int j=0; j<d; j++) {
      lambda_tau2_additive(j) = R::rgamma(1.0, 1.0 / (s2_tau2_additive + tau2_additive(j)));
    }

    //
    // sample s2_tau2_additive
    //

    double s2_tau2_additive_star = s2_tau2_additive + R::rnorm(0, s2_tau2_additive_tune);
    if (s2_tau2_additive_star > 0 && s2_tau2_additive_star < A_s2_additive) {
      double mh1 = 0.0;
      double mh2 = 0.0;
      for (int j=0; j<d; j++){
        mh1 += R::dgamma(lambda_tau2_additive(j), 0.5, 1.0 / s2_tau2_additive_star, true);
        mh2 += R::dgamma(lambda_tau2_additive(j), 0.5, 1.0 / s2_tau2_additive, true);
      }
      double mh = exp(mh1-mh2);
      if (mh > R::runif(0.0, 1.0)) {
        s2_tau2_additive = s2_tau2_additive_star;
        s2_tau2_additive_accept += 1.0 / n_mcmc;
      }
    }

    //
    // sample xi_additive - MH
    //

    if (sample_xi_additive) {
      arma::vec logit_xi_tilde_additive_star = mvrnormArmaVecChol(logit(xi_tilde_additive),
                                                                  lambda_xi_additive_tune * Sigma_xi_additive_tune_chol);
      arma::vec xi_tilde_additive_star = expit(logit_xi_tilde_additive_star);
      arma::vec xi_additive_star = 2.0 * xi_tilde_additive_star - 1.0;
      // arma::vec xi_star =  mvrnormArmaVecChol(xi, lambda_xi_tune * Sigma_xi_tune_chol);
      if (all(xi_additive_star > -1.0) && all(xi_additive_star < 1.0)) {
        Rcpp::List R_out_additive = makeRLKJ(xi_additive_star, d, true, true);
        arma::mat R_additive_star = as<mat>(R_out_additive["R"]);
        arma::mat R_tau_additive_star = R_additive_star * diagmat(tau_additive);
        double log_jacobian_additive_star = as<double>(R_out_additive["log_jacobian_additive"]);
        double mh1 = sum(log(xi_tilde_additive_star) + log(ones_B - xi_tilde_additive_star));
        // Jacobian adjustment
        double mh2 = sum(log(xi_tilde_additive) + log(ones_B - xi_tilde_additive));
        // Jacobian adjustment
        for (int b=0; b<B; b++) {
          mh1 += R::dbeta(0.5 * (xi_additive_star(b) + 1.0), eta_vec_additive(b), eta_vec_additive(b), true);
          mh2 += R::dbeta(0.5 * (xi_additive(b) + 1.0), eta_vec_additive(b), eta_vec_additive(b), true);
        }
        for (int i=0; i<N; i++) {
          mh1 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive_star, true);
          mh2 += dMVN(epsilon.row(i).t(), zero_d, R_tau_additive, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          xi_tilde_additive = xi_tilde_additive_star;
          xi_additive = xi_additive_star;
          R_additive = R_additive_star;
          R_tau_additive = R_tau_additive_star;
          log_jacobian_additive = log_jacobian_additive_star;
          xi_additive_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      alpha_save.subcube(span(save_idx), span(), span()) = alpha;
      beta_save.subcube(span(save_idx), span(), span()) = beta;
      tau2_save.row(save_idx) = tau2.t();
      lambda_tau2_save.row(save_idx) = lambda_tau2.t();
      s2_tau2_save(save_idx) = s2_tau2;
      R_save.subcube(span(save_idx), span(), span()) = R;
      R_tau_save.subcube(span(save_idx), span(), span()) = R_tau;
      xi_save.row(save_idx) = xi.t();
      tau2_additive_save.row(save_idx) = tau2_additive.t();
      s2_tau2_additive_save(save_idx) = s2_tau2_additive;
      R_additive_save.subcube(span(save_idx), span(), span()) = R_additive;
      R_tau_additive_save.subcube(span(save_idx), span(), span()) = R_tau_additive;
      xi_additive_save.row(save_idx) = xi_additive.t();
    }

    // end mcmc fitting loop
  }

  // print acceptance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  if (sample_beta) {
    file_out << "Average acceptance rate for beta  = " << mean(beta_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_xi) {
    file_out << "Average acceptance rate for xi  = " << mean(xi_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_tau2) {
    file_out << "Average acceptance rate for tau2  = " << mean(tau2_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_xi_additive) {
    file_out << "Average acceptance rate for xi_additive  = " << mean(xi_additive_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_tau2_additive) {
    file_out << "Average acceptance rate for tau2_additive  = " << mean(tau2_additive_accept) <<
      " for chain " << n_chain << "\n";
  }
  // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    _["alpha"] = alpha_save,
    _["beta"] = beta_save,
    _["tau2"] = tau2_save,
    _["lambda_tau2"] = lambda_tau2_save,
    _["s2_tau2"] = s2_tau2_save,
    _["R"] = R_save,
    _["R_tau"] = R_tau_save,
    _["xi"] = xi_save,
    _["tau2_additive"] = tau2_additive_save,
    _["s2_tau2_additive"] = s2_tau2_additive_save,
    _["R_additive"] = R_additive_save,
    _["R_tau_additive"] = R_tau_additive_save,
    _["xi_additive"] = xi_additive_save);

}
