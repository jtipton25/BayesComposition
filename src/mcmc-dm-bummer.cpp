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

// Multinomial Multi-logit multivariate Bummer regression for Inverse Inference

// Author: John Tipton
//
// Created 01.31.2019
// Last updated 01.31.2019

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppDMBummer (const arma::mat& Y,
                       const arma::vec& X,
                       List params,
                       int n_chain           = 1,
                       std::string file_name = "DM-fit") {

  // Load parameters
  int n_adapt = as<int>(params["n_adapt"]);
  int n_mcmc = as<int>(params["n_mcmc"]);
  int n_thin = as<int>(params["n_thin"]);
  // default to message output every 500 iterations
  // default output message iteration
  int message = 500;
  if (params.containsElementNamed("message")) {
    message = as<int>(params["message"]);
  }

  // set up dimensions
  double N = Y.n_rows;
  double d = Y.n_cols;

  // count - sum of counts at each site
  arma::vec count(N);
  for (int i=0; i<N; i++) {
    count(i) = sum(Y.row(i));
  }

  // constant vectors
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec zeros_d(d, arma::fill::ones);

  // default normal prior for mean parameter a
  arma::vec mu_a(d, arma::fill::zeros);
  if (params.containsElementNamed("mu_a")) {
    mu_a = as<vec>(params["mu_a"]);
  }
  // default normal prior for mean parameter a
  arma::mat Sigma_a(d, d, arma::fill::eye);
  Sigma_a *= 1.0;
  if (params.containsElementNamed("Sigma_a")) {
    Sigma_a = as<mat>(params["Sigma_a"]);
  }
  arma::mat Sigma_a_inv = inv_sympd(Sigma_a);
  arma::mat Sigma_a_chol = chol(Sigma_a);

  // default normal prior for mean parameter b
  arma::vec mu_b(d, arma::fill::zeros);
  if (params.containsElementNamed("mu_b")) {
    mu_b = as<vec>(params["mu_b"]);
  }
  // default normal prior for mean parameter b
  arma::mat Sigma_b(d, d, arma::fill::eye);
  Sigma_b *= 1.0;
  if (params.containsElementNamed("Sigma_b")) {
    Sigma_b = as<mat>(params["Sigma_b"]);
  }
  arma::mat Sigma_b_inv = inv_sympd(Sigma_b);
  arma::mat Sigma_b_chol = chol(Sigma_b);

  // default prior for variance parameter c
  double sigma_c = 2.0;
  if (params.containsElementNamed("sigma_c")) {
    sigma_c = as<double>(params["sigma_c"]);
  }

  // default a tuning parameter
  double lambda_a_tune = 1.0;

  lambda_a_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_a_tune")) {
    lambda_a_tune = as<double>(params["lambda_a_tune"]);
  }

  // default b tuning parameter
  double lambda_b_tune = 1.0;

  lambda_b_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_b_tune")) {
    lambda_b_tune = as<double>(params["lambda_b_tune"]);
  }

  // default c tuning parameter
  double lambda_c_tune = 1.0;

  lambda_c_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_c_tune")) {
    lambda_c_tune = as<double>(params["lambda_c_tune"]);
  }

   //
  // initialize values
  //

  //
  // set default, fixed parameters and turn on/off samplers for testing
  //

  //
  // Default for a
  //

  arma::vec a = mvrnormArmaVecChol(mu_a, Sigma_a_chol);
    if (params.containsElementNamed("a")) {
    a = as<vec>(params["a"]);
  }
  bool sample_a = true;
  if (params.containsElementNamed("sample_a")) {
    sample_a = as<bool>(params["sample_a"]);
  }

  //
  // Default for b
  //

  arma::vec b = mvrnormArmaVecChol(mu_b, Sigma_b_chol);
  if (params.containsElementNamed("b")) {
    b = as<vec>(params["b"]);
  }
  bool sample_b = true;
  if (params.containsElementNamed("sample_b")) {
    sample_b = as<bool>(params["sample_b"]);
  }

  //
  // Default for c
  //

  arma::vec c = exp(rnorm(d, 0.0, sigma_c));
  if (params.containsElementNamed("c")) {
    c = as<vec>(params["c"]);
  }
  arma::vec log_c = log(c);
  bool sample_c = true;
  if (params.containsElementNamed("sample_c")) {
    sample_c = as<bool>(params["sample_c"]);
  }

  arma::mat alpha(N, d);
  for (int i=0; i<N; i++) {
    for (int j=0; j<d; j++) {
      alpha(i, j) = exp(a(j) - pow(X(i) - b(j), 2.0) / c(j));
    }
  }

  // setup save variables
  int n_save = n_mcmc / n_thin;
  // arma::mat mu_save(n_save, d, arma::fill::zeros);
  arma::cube alpha_save(n_save, N, d, arma::fill::zeros);
  arma::mat a_save(n_save, d, arma::fill::zeros);
  arma::mat b_save(n_save, d, arma::fill::zeros);
  arma::mat c_save(n_save, d, arma::fill::zeros);

  // initialize tuning

  double a_accept = 0.0;
  double a_accept_batch = 0.0;
  double b_accept = 0.0;
  double b_accept_batch = 0.0;
  double c_accept = 0.0;
  double c_accept_batch = 0.0;
  arma::mat a_batch(50, d, arma::fill::zeros);
  arma::mat b_batch(50, d, arma::fill::zeros);
  arma::mat c_batch(50, d, arma::fill::zeros);
  arma::mat Sigma_a_tune(d, d, arma::fill::eye);
  arma::mat Sigma_a_tune_chol = chol(Sigma_a_tune);
  arma::mat Sigma_b_tune(d, d, arma::fill::eye);
  arma::mat Sigma_b_tune_chol = chol(Sigma_b_tune);
  arma::mat Sigma_c_tune(d, d, arma::fill::eye);
  arma::mat Sigma_c_tune_chol = chol(Sigma_c_tune);

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
    // Sample a - block MH
    //

    if (sample_a) {
      arma::vec a_star = a +
        mvrnormArmaVecChol(zeros_d, lambda_a_tune * Sigma_a_tune_chol);
      arma::mat alpha_star(N, d);
      for (int i=0; i<N; i++) {
        for (int j=0; j<d; j++) {
          alpha_star(i, j) = exp(a_star(j) - pow(X(i) - b(j), 2.0) / c(j));
        }
      }
      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        dMVN(a_star, mu_a, Sigma_a_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        dMVN(a, mu_a, Sigma_a_chol);
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0.0, 1.0)) {
        a = a_star;
        alpha = alpha_star;
        a_accept_batch += 1.0 / 50.0;
      }
      // update tuning
      a_batch.row(k % 50) = a.t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, a_accept_batch, lambda_a_tune,
                       a_batch, Sigma_a_tune,
                       Sigma_a_tune_chol);
      }
    }

    //
    // Sample b - block MH
    //

    if (sample_b) {
      arma::vec b_star = b +
        mvrnormArmaVecChol(zeros_d, lambda_b_tune * Sigma_b_tune_chol);
      arma::mat alpha_star(N, d);
      for (int i=0; i<N; i++) {
        for (int j=0; j<d; j++) {
          alpha_star(i, j) = exp(a(j) - pow(X(i) - b_star(j), 2.0) / c(j));
        }
      }
      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        dMVN(b_star, mu_b, Sigma_b_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        dMVN(b, mu_b, Sigma_b_chol);
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0.0, 1.0)) {
        b = b_star;
        alpha = alpha_star;
        b_accept_batch += 1.0 / 50.0;
      }
      // update tuning
      b_batch.row(k % 50) = b.t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, b_accept_batch, lambda_b_tune,
                       b_batch, Sigma_b_tune,
                       Sigma_b_tune_chol);
      }
    }

    //
    // Sample c - block MH
    //

    if (sample_c) {
      arma::vec log_c_star = log_c + mvrnormArmaVecChol(zeros_d, lambda_b_tune * Sigma_b_tune_chol);
      arma::vec c_star = exp(log_c_star);
      arma::mat alpha_star(N, d);
      for (int i=0; i<N; i++) {
        for (int j=0; j<d; j++) {
          alpha_star(i, j) = exp(a(j) - pow(X(i) - b(j), 2.0) / c_star(j));
        }
      }
      double mh1 = LL_DM(alpha_star, Y, N, d, count) + sum(log_c_star);  // jacobian of log-scale proposal;
      double mh2 = LL_DM(alpha, Y, N, d, count) + sum(log_c);            // jacobian of log-scale proposal;
      for (int j=0; j<d; j++) {
        mh1 += d_half_cauchy(c_star(j), sigma_c, true);
        mh2 += d_half_cauchy(c(j), sigma_c, true);
      }
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0.0, 1.0)) {
        log_c = log_c_star;
        c = c_star;
        alpha = alpha_star;
        c_accept_batch += 1.0 / 50.0;
      }
      // update tuning
      c_batch.row(k % 50) = log_c.t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, c_accept_batch, lambda_c_tune,
                       c_batch, Sigma_c_tune,
                       Sigma_c_tune_chol);
      }
    }

    // end of MCMC iteration
  }
  // end of MCMC adaptation

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
    // Sample a - block MH
    //

    if (sample_a) {
      arma::vec a_star = a +
        mvrnormArmaVecChol(zeros_d, lambda_a_tune * Sigma_a_tune_chol);
      arma::mat alpha_star(N, d);
      for (int i=0; i<N; i++) {
        for (int j=0; j<d; j++) {
          alpha_star(i, j) = exp(a_star(j) - pow(X(i) - b(j), 2.0) / c(j));
        }
      }
      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        dMVN(a_star, mu_a, Sigma_a_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        dMVN(a, mu_a, Sigma_a_chol);
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0.0, 1.0)) {
        a = a_star;
        alpha = alpha_star;
        a_accept += 1.0 / n_mcmc;
      }
    }

    //
    // Sample b - block MH
    //

    if (sample_b) {
      arma::vec b_star = b +
        mvrnormArmaVecChol(zeros_d, lambda_b_tune * Sigma_b_tune_chol);
      arma::mat alpha_star(N, d);
      for (int i=0; i<N; i++) {
        for (int j=0; j<d; j++) {
          alpha_star(i, j) = exp(a(j) - pow(X(i) - b_star(j), 2.0) / c(j));
        }
      }
      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        dMVN(b_star, mu_b, Sigma_b_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        dMVN(b, mu_b, Sigma_b_chol);
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0.0, 1.0)) {
        b = b_star;
        alpha = alpha_star;
        b_accept += 1.0 / n_mcmc;
      }
    }

    //
    // Sample c - block MH
    //

    if (sample_c) {
      arma::vec log_c_star = log_c + mvrnormArmaVecChol(zeros_d, lambda_b_tune * Sigma_b_tune_chol);
      arma::vec c_star = exp(log_c_star);
      arma::mat alpha_star(N, d);
      for (int i=0; i<N; i++) {
        for (int j=0; j<d; j++) {
          alpha_star(i, j) = exp(a(j) - pow(X(i) - b(j), 2.0) / c_star(j));
        }
      }
      double mh1 = LL_DM(alpha_star, Y, N, d, count) + sum(log_c_star);  // jacobian of log-scale proposal;
      double mh2 = LL_DM(alpha, Y, N, d, count) + sum(log_c);            // jacobian of log-scale proposal;
      for (int j=0; j<d; j++) {
        mh1 += d_half_cauchy(c_star(j), sigma_c, true);
        mh2 += d_half_cauchy(c(j), sigma_c, true);
      }
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0.0, 1.0)) {
        log_c = log_c_star;
        c = c_star;
        alpha = alpha_star;
        c_accept += 1.0 / n_mcmc;
      }
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      a_save.row(save_idx) = a.t();
      b_save.row(save_idx) = b.t();
      c_save.row(save_idx) = c.t();
      alpha_save.subcube(span(save_idx), span(), span()) = alpha;
    }

    // end of MCMC iteration
  }
  // end of MCMC fitting


  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);

  if (sample_a) {
    file_out << "Average acceptance rate for a  = " << mean(a_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_b) {
    file_out << "Average acceptance rate for b  = " << mean(b_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_c) {
    file_out << "Average acceptance rate for c  = " << mean(c_accept) <<
      " for chain " << n_chain << "\n";
  }

  // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    // _["mu"] = mu_save,
    _["alpha"] = alpha_save,
    _["a"] = a_save,
    _["b"] = b_save,
    _["c"] = c_save);
}
