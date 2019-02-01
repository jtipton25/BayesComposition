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
// Last updated 09.14.2017

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppDMBasis (const arma::mat& Y, const arma::vec& X,
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

  // // default normal prior for mean of overall means
  // double mu_mu = 0.0;
  // if (params.containsElementNamed("mu_mu")) {
  //   mu_mu = as<double>(params["mu_mu"]);
  // }
  // // default prior for standard deviation of overall means
  // double sigma_mu = 5.0;
  // if (params.containsElementNamed("sigma_mu")) {
  //   mu_mu = as<double>(params["sigma_mu"]);
  // }

  // default normal prior pooling of regression coefficients
  arma::vec mu_beta_0(df, arma::fill::zeros);
  if (params.containsElementNamed("mu_beta_0")) {
    mu_beta_0 = as<vec>(params["mu_beta_0"]);
  }
  // default prior for pooling of regression coefficient covariance
  arma::mat Sigma_beta_0(df, df, arma::fill::eye);
  Sigma_beta_0 *= 1.0;
  if (params.containsElementNamed("Sigma_beta_0")) {
    Sigma_beta_0 = as<mat>(params["Sigma_beta_0"]);
  }
  arma::mat Sigma_beta_0_inv = inv_sympd(Sigma_beta_0);
  // default prior inverse Wishart degrees of freedom
  double nu = df + 2.0;
  if (params.containsElementNamed("nu")) {
    nu = as<int>(params["nu"]);
  }
  // default prior matrix for inverse Wishart
  arma::mat S(df, df, arma::fill::eye);
  S *= 5.0;
  if (params.containsElementNamed("S")) {
    S = as<mat>(params["S"]);
  }
  arma::mat S_inv = inv_sympd(S);

  // initialize hierarchical pooling mean of beta
  arma::vec mu_beta = mvrnormArmaVec(mu_beta_0, Sigma_beta_0);
  if (params.containsElementNamed("mu_beta")) {
    mu_beta = as<vec>(params["mu_beta"]);
  }
  bool sample_mu_beta = true;
  if (params.containsElementNamed("sample_mu_beta")) {
    sample_mu_beta = as<bool>(params["sample_mu_beta"]);
  }

  // initialize hierarchical pooling covariance of beta
  arma::mat Sigma_beta = rIWishartArmaMat(nu, S);
  if (params.containsElementNamed("Sigma_beta")) {
    Sigma_beta = as<mat>(params["Sigma_beta"]);
  }
  arma::mat Sigma_beta_inv = inv_sympd(Sigma_beta);
  arma::mat Sigma_beta_chol = chol(Sigma_beta);

  bool sample_Sigma_beta = true;
  if (params.containsElementNamed("sample_Sigma_beta")) {
    sample_Sigma_beta = as<bool>(params["sample_Sigma_beta"]);
  }

  // // default mu tuning parameter
  // double mu_tune_tmp = 1.0;
  // if (params.containsElementNamed("mu_tune")) {
  //   mu_tune_tmp = as<double>(params["mu_tune"]);
  // }
  // default beta tuning parameter
  arma::vec lambda_beta_tune(d, arma::fill::ones);

  lambda_beta_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_beta_tune")) {
    lambda_beta_tune = as<vec>(params["lambda_beta_tune"]);
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
  knots = knots.subvec(1, df-degree);
  // knots = knots.subvec(1, df-degree-1);
  arma::mat Xbs = bs_cpp(X, df, knots, degree, false, rangeX);

  //
  // initialize values
  //

  //
  // set default, fixed parameters and turn on/off samplers for testing
  //

  //
  // Default for mu
  //

  // arma::vec mu(d);
  // for (int j=0; j<d; j++) {
  //   mu(j) = R::rnorm(mu_mu, sigma_mu);
  // }
  // if (params.containsElementNamed("mu")) {
  //   mu = as<vec>(params["mu"]);
  // }
  // bool sample_mu = true;
  // if (params.containsElementNamed("sample_mu")) {
  //   sample_mu = as<bool>(params["sample_mu"]);
  // }
  // arma::mat mu_mat(N, d, arma::fill::zeros);
  // for (int i=0; i<N; i++) {
  //   mu_mat.row(i) = mu.t();
  // }

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

  // arma::mat alpha = exp(mu_mat + Xbs * beta);
  arma::mat alpha = exp(Xbs * beta);

  // setup save variables
  int n_save = n_mcmc / n_thin;
  // arma::mat mu_save(n_save, d, arma::fill::zeros);
  arma::cube alpha_save(n_save, N, d, arma::fill::zeros);
  arma::cube beta_save(n_save, df, d, arma::fill::zeros);
  arma::mat mu_beta_save(n_save, df, arma::fill::zeros);
  arma::cube Sigma_beta_save(n_save, df, df, arma::fill::zeros);

  // initialize tuning

  // arma::vec mu_accept(d, arma::fill::zeros);
  // arma::vec mu_accept_batch(d, arma::fill::zeros);
  // arma::vec mu_tune(d, arma::fill::ones);
  // mu_tune *= mu_tune_tmp;
  arma::vec beta_accept(d, arma::fill::zeros);
  arma::vec beta_accept_batch(d, arma::fill::zeros);
  arma::cube beta_batch(50, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune(df, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune_chol(df, df, d, arma::fill::zeros);
  for(int j=0; j<d; j++) {
    Sigma_beta_tune.slice(j).eye();
    Sigma_beta_tune_chol.slice(j) = chol(Sigma_beta_tune.slice(j));
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

    // //
    // // Sample mu - MH
    // //
    //
    // if (sample_mu) {
    //   for (int j=0; j<d; j++) {
    //     arma::vec mu_star = mu;
    //     mu_star(j) += R::rnorm(0.0, mu_tune(j));
    //     arma::mat mu_mat_star = mu_mat;
    //     for (int i=0; i<N; i++) {
    //       mu_mat_star.row(i) = mu_star.t();
    //     }
    //     arma::mat alpha_star = exp(mu_mat_star + Xbs * beta);
    //     double mh1 = LL_DM(alpha_star, Y, N, d, count) +
    //       R::dnorm4(mu_star(j), mu_mu, sigma_mu, true);
    //     double mh2 = LL_DM(alpha, Y, N, d, count) +
    //       R::dnorm4(mu(j), mu_mu, sigma_mu, true);
    //     double mh = exp(mh1 - mh2);
    //     if (mh > R::runif(0, 1.0)) {
    //       mu = mu_star;
    //       mu_mat = mu_mat_star;
    //       alpha = alpha_star;
    //       mu_accept_batch(j) += 1.0 / 50.0;
    //     }
    //   }
    //   // update tuning
    //   if ((k+1) % 50 == 0){
    //     updateTuningVec(k, mu_accept_batch, mu_tune);
    //   }
    // }

    //
    // Sample beta - block MH
    //

    if (sample_beta) {
      for (int j=0; j<d; j++) {
        arma::mat beta_star = beta;
        beta_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));
        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha_star = exp(Xbs * beta_star);
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
    // sample mu_beta
    //

    if (sample_mu_beta) {
      arma::mat A = Sigma_beta_0_inv + d * Sigma_beta_inv;
      arma::vec b = Sigma_beta_0_inv * mu_beta_0;
      for (int j=0; j<d; j++) {
        b += Sigma_beta_inv * beta.col(j);
      }
      mu_beta = rMVNArma(A, b);
    }

    //
    // sample Sigma_beta
    //

    if (sample_Sigma_beta) {
      arma::mat tmp(df, df, arma::fill::zeros);
      for (int j=0; j<d; j++) {
        tmp += (beta.col(j) - mu_beta) * (beta.col(j) - mu_beta).t();
      }
      Sigma_beta_inv = rWishartArmaMat(d + nu, inv_sympd(tmp + nu * S));
      Sigma_beta = inv_sympd(Sigma_beta_inv);
      Sigma_beta_chol = chol(Sigma_beta);
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

    // //
    // // Sample mu - MH
    // //
    //
    // if (sample_mu) {
    //   for (int j=0; j<d; j++) {
    //     arma::vec mu_star = mu;
    //     mu_star(j) += R::rnorm(0.0, mu_tune(j));
    //     arma::mat mu_mat_star = mu_mat;
    //     for (int i=0; i<N; i++) {
    //       mu_mat_star.row(i) = mu_star.t();
    //     }
    //     arma::mat alpha_star = exp(mu_mat_star + Xbs * beta);
    //     double mh1 = LL_DM(alpha_star, Y, N, d, count) +
    //       R::dnorm4(mu_star(j), mu_mu, sigma_mu, true);
    //     double mh2 = LL_DM(alpha, Y, N, d, count) +
    //       R::dnorm4(mu(j), mu_mu, sigma_mu, true);
    //     double mh = exp(mh1 - mh2);
    //     if (mh > R::runif(0, 1.0)) {
    //       mu = mu_star;
    //       mu_mat = mu_mat_star;
    //       alpha = alpha_star;
    //       mu_accept(j) += 1.0 / n_mcmc;
    //     }
    //   }
    // }

    //
    // Sample beta - block MH
    //

    if (sample_beta) {
      for (int j=0; j<d; j++) {
        arma::mat beta_star = beta;
        beta_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));

        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha_star = exp(Xbs * beta_star);
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
    // sample mu_beta
    //

    if (sample_mu_beta) {
      arma::mat A = Sigma_beta_0_inv + d * Sigma_beta_inv;
      arma::vec b = Sigma_beta_0_inv * mu_beta_0;
      for (int j=0; j<d; j++) {
        b += Sigma_beta_inv * beta.col(j);
      }
      mu_beta = rMVNArma(A, b);
    }

    //
    // sample Sigma_beta
    //

    if (sample_Sigma_beta) {
      arma::mat tmp(df, df, arma::fill::zeros);
      for (int j=0; j<d; j++) {
        tmp += (beta.col(j) - mu_beta) * (beta.col(j) - mu_beta).t();
      }
      Sigma_beta_inv = rWishartArmaMat(d + nu, inv_sympd(tmp + nu * S));
      Sigma_beta = inv_sympd(Sigma_beta_inv);
      Sigma_beta_chol = chol(Sigma_beta);
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      // mu_save.row(save_idx) = mu.t();
      alpha_save.subcube(span(save_idx), span(), span()) = alpha;
      beta_save.subcube(span(save_idx), span(), span()) = beta;
      mu_beta_save.row(save_idx) = mu_beta.t();
      Sigma_beta_save.subcube(span(save_idx), span(), span()) = Sigma_beta;
    }

    // end of MCMC iteration
  }
  // end of MCMC fitting


  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  // if (sample_mu) {
  //   file_out << "Average acceptance rate for mu  = " << mean(mu_accept) <<
  //     " for chain " << n_chain << "\n";
  // }

    if (sample_beta) {
    file_out << "Average acceptance rate for beta  = " << mean(beta_accept) <<
      " for chain " << n_chain << "\n";
  }

  // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    // _["mu"] = mu_save,
    _["alpha"] = alpha_save,
    _["beta"] = beta_save,
    _["mu_beta"] = mu_beta_save,
    _["Sigma_beta"] = Sigma_beta_save);

}
