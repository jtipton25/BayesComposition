// #define ARMA_64BIT_WORD
#include <RcppArmadilloExtensions/sample.h>
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
// To do - allow for more than 2 mixture components

// Author: John Tipton
//
// Created 05.15.2017
// Last updated 09.14.2017

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

//
// Replicate R sample in c++
//

// [[Rcpp::export]]
arma::vec csample(arma::vec x, int size, bool replace,
                  NumericVector prob) {
  // int n = x.n_elem;
  // Rcpp::RcppArmadillo::FixProb(prob, n, true);
  arma::vec ret = Rcpp::RcppArmadillo::sample(x, size, replace, prob);
  return ret;
}


///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppDMBasisMixture (const arma::mat& Y,
                             const arma::vec& X,
                             List params,
                             int n_chain=1,
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

  // prior for mixing proportions
  arma::vec phi(2);
  phi(0) = 0.5;
  phi(1) = 0.5;
  if (params.containsElementNamed("phi")) {
    phi = as<vec>(params["phi"]);
  }

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
  arma::vec mu_beta1 = mvrnormArmaVec(mu_beta_0, Sigma_beta_0);
  arma::vec mu_beta2 = mvrnormArmaVec(mu_beta_0, Sigma_beta_0);
  if (params.containsElementNamed("mu_beta1")) {
    mu_beta1 = as<vec>(params["mu_beta1"]);
  }
  if (params.containsElementNamed("mu_beta2")) {
    mu_beta2 = as<vec>(params["mu_beta2"]);
  }
  bool sample_mu_beta = true;
  if (params.containsElementNamed("sample_mu_beta")) {
    sample_mu_beta = as<bool>(params["sample_mu_beta"]);
  }

  // initialize hierarchical pooling covariance of beta
  arma::mat Sigma_beta1 = rIWishartArmaMat(nu, S);
  arma::mat Sigma_beta2 = rIWishartArmaMat(nu, S);
  if (params.containsElementNamed("Sigma_beta1")) {
    Sigma_beta1 = as<mat>(params["Sigma_beta1"]);
  }
  if (params.containsElementNamed("Sigma_beta2")) {
    Sigma_beta2 = as<mat>(params["Sigma_beta2"]);
  }
  arma::mat Sigma_beta1_inv = inv_sympd(Sigma_beta1);
  arma::mat Sigma_beta1_chol = chol(Sigma_beta1);
  arma::mat Sigma_beta2_inv = inv_sympd(Sigma_beta2);
  arma::mat Sigma_beta2_chol = chol(Sigma_beta2);

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
  arma::vec lambda_beta1_tune(d, arma::fill::ones);
  arma::vec lambda_beta2_tune(d, arma::fill::ones);

  lambda_beta1_tune *= 1.0 / pow(3.0, 0.8);
  lambda_beta2_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_beta1_tune")) {
    lambda_beta1_tune = as<mat>(params["lambda_beta1_tune"]);
  }
  if (params.containsElementNamed("lambda_beta2_tune")) {
    lambda_beta2_tune = as<mat>(params["lambda_beta2_tune"]);
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

  arma::mat beta1(df, d);
  arma::mat beta2(df, d);
  for (int j=0; j<d; j++) {
    beta1.col(j) = mvrnormArmaVec(mu_beta1, Sigma_beta1);
    beta2.col(j) = mvrnormArmaVec(mu_beta2, Sigma_beta2);
  }
  if (params.containsElementNamed("beta1")) {
    beta1 = as<mat>(params["beta1"]);
  }
  if (params.containsElementNamed("beta2")) {
    beta1 = as<mat>(params["beta2"]);
  }
  bool sample_beta = true;
  if (params.containsElementNamed("sample_beta")) {
    sample_beta = as<bool>(params["sample_beta"]);
  }

  // arma::mat alpha = exp(mu_mat + Xbs * beta);
  arma::mat alpha1 = exp(Xbs * beta1);
  arma::mat alpha2 = exp(Xbs * beta2);

  //
  // Default for mixture component z
  //

  // initialize mixing parameter
  arma::vec one_to_2(2);
  for (int j=0; j<2; j++) {
    one_to_2(j) = j;
  }
  arma::vec z(N);

  for (int i=0; i<N; i++) {
    arma::vec log_p = log(phi);
    log_p(0) += LL_DM_row(alpha1.row(i), Y.row(i), d, count(i));
    log_p(1) += LL_DM_row(alpha2.row(i), Y.row(i), d, count(i));
    double a = log_p.max();
    arma::vec prob_sample = 1.0 / (exp(a + log(sum(exp(log_p - a))) - log_p));
    z(i) = as_scalar(csample(one_to_2, 1, false, as<NumericVector>(wrap(prob_sample))));
  }

  bool sample_z = true;
  if (params.containsElementNamed("sample_z")) {
    sample_z = as<bool>(params["sample_z"]);
  }

  // setup save variables
  int n_save = n_mcmc / n_thin;
  // arma::mat mu_save(n_save, d, arma::fill::zeros);
  arma::cube alpha1_save(n_save, N, d, arma::fill::zeros);
  arma::cube alpha2_save(n_save, N, d, arma::fill::zeros);
  arma::cube beta1_save(n_save, df, d, arma::fill::zeros);
  arma::cube beta2_save(n_save, df, d, arma::fill::zeros);
  arma::mat mu_beta1_save(n_save, df, arma::fill::zeros);
  arma::mat mu_beta2_save(n_save, df, arma::fill::zeros);
  arma::cube Sigma_beta1_save(n_save, df, df, arma::fill::zeros);
  arma::cube Sigma_beta2_save(n_save, df, df, arma::fill::zeros);
  arma::mat z_save(n_save, N, arma::fill::zeros);

  // initialize tuning

  // arma::vec mu_accept(d, arma::fill::zeros);
  // arma::vec mu_accept_batch(d, arma::fill::zeros);
  // arma::vec mu_tune(d, arma::fill::ones);
  // mu_tune *= mu_tune_tmp;

  arma::vec beta1_accept(d, arma::fill::zeros);
  arma::vec beta2_accept(d, arma::fill::zeros);
  arma::vec beta1_accept_batch(d, arma::fill::zeros);
  arma::vec beta2_accept_batch(d, arma::fill::zeros);
  arma::cube beta1_batch(50, df, d, arma::fill::zeros);
  arma::cube beta2_batch(50, df, d, arma::fill::zeros);
  arma::cube Sigma_beta1_tune(df, df, d, arma::fill::zeros);
  arma::cube Sigma_beta2_tune(df, df, d, arma::fill::zeros);
  arma::cube Sigma_beta1_tune_chol(df, df, d, arma::fill::zeros);
  arma::cube Sigma_beta2_tune_chol(df, df, d, arma::fill::zeros);
  for(int j=0; j<d; j++) {
    Sigma_beta1_tune.slice(j).eye();
    Sigma_beta2_tune.slice(j).eye();
    Sigma_beta1_tune_chol.slice(j) = chol(Sigma_beta1_tune.slice(j));
    Sigma_beta2_tune_chol.slice(j) = chol(Sigma_beta2_tune.slice(j));
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
      // sample beta1
      for (int j=0; j<d; j++) {
        arma::mat beta1_star = beta1;
        beta1_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta1_tune(j) * Sigma_beta1_tune_chol.slice(j));
        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha1_star = exp(Xbs * beta1_star);
        double mh1 = dMVN(beta1_star.col(j), mu_beta1, Sigma_beta1_chol);
        double mh2 = dMVN(beta1.col(j), mu_beta1, Sigma_beta1_chol);

        for (int i=0; i<N; i++) {
          if (z(i) == 0) {
            mh1 += LL_DM_row(alpha1_star.row(i), Y.row(i), d, count(i));
            mh2 += LL_DM_row(alpha1.row(i), Y.row(i), d, count(i));
          }

        }
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta1 = beta1_star;
          alpha1 = alpha1_star;
          beta1_accept_batch(j) += 1.0 / 50.0;
        }
      }
      // update tuning
      beta1_batch.subcube(k % 50, 0, 0, k % 50, df-1, d-1) = beta1;
      if ((k+1) % 50 == 0){
        updateTuningMVMat(k, beta1_accept_batch, lambda_beta1_tune,
                          beta1_batch, Sigma_beta1_tune,
                          Sigma_beta1_tune_chol);
      }
      // sample beta2
      for (int j=0; j<d; j++) {
        arma::mat beta2_star = beta2;
        beta2_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta2_tune(j) * Sigma_beta2_tune_chol.slice(j));
        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha2_star = exp(Xbs * beta2_star);
        double mh1 = dMVN(beta2_star.col(j), mu_beta2, Sigma_beta2_chol);
        double mh2 = dMVN(beta2.col(j), mu_beta2, Sigma_beta2_chol);

        for (int i=0; i<N; i++) {
          if (z(i) == 1) {
            mh1 += LL_DM_row(alpha2_star.row(i), Y.row(i), d, count(i));
            mh2 += LL_DM_row(alpha2.row(i), Y.row(i), d, count(i));
          }

        }
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta2 = beta2_star;
          alpha2 = alpha2_star;
          beta2_accept_batch(j) += 1.0 / 50.0;
        }
      }
      // update tuning
      beta2_batch.subcube(k % 50, 0, 0, k % 50, df-1, d-1) = beta2;
      if ((k+1) % 50 == 0){
        updateTuningMVMat(k, beta2_accept_batch, lambda_beta2_tune,
                          beta2_batch, Sigma_beta2_tune,
                          Sigma_beta2_tune_chol);
      }

    }

    //
    // sample mu_beta
    //

    if (sample_mu_beta) {
      // mu_beta_1
      arma::mat A1 = Sigma_beta_0_inv + d * Sigma_beta1_inv;
      arma::vec b1 = Sigma_beta_0_inv * mu_beta_0;
      for (int j=0; j<d; j++) {
        b1 += Sigma_beta1_inv * beta1.col(j);
      }
      mu_beta1= rMVNArma(A1, b1);
      // mu_beta_2
      arma::mat A2 = Sigma_beta_0_inv + d * Sigma_beta2_inv;
      arma::vec b2 = Sigma_beta_0_inv * mu_beta_0;
      for (int j=0; j<d; j++) {
        b2 += Sigma_beta2_inv * beta2.col(j);
      }
      mu_beta2= rMVNArma(A2, b2);
    }

    //
    // sample Sigma_beta
    //

    if (sample_Sigma_beta) {
      // Sigma_beta1
      arma::mat tmp1(df, df, arma::fill::zeros);
      for (int j=0; j<d; j++) {
        tmp1 += (beta1.col(j) - mu_beta1) * (beta1.col(j) - mu_beta1).t();
      }
      Sigma_beta1_inv = rWishartArmaMat(d + nu, inv_sympd(tmp1 + nu * S));
      Sigma_beta1 = inv_sympd(Sigma_beta1_inv);
      Sigma_beta1_chol = chol(Sigma_beta1);
      // Sigma_beta2
      arma::mat tmp2(df, df, arma::fill::zeros);
      for (int j=0; j<d; j++) {
        tmp2 += (beta2.col(j) - mu_beta2) * (beta2.col(j) - mu_beta2).t();
      }
      Sigma_beta2_inv = rWishartArmaMat(d + nu, inv_sympd(tmp2 + nu * S));
      Sigma_beta2 = inv_sympd(Sigma_beta2_inv);
      Sigma_beta2_chol = chol(Sigma_beta2);
    }

    //
    // sample mixing indicator z
    //

    if (sample_z) {
      for (int i=0; i<N; i++) {
        arma::vec log_p = log(phi);
        log_p(0) += LL_DM_row(alpha1.row(i), Y.row(i), d, count(i));
        log_p(1) += LL_DM_row(alpha2.row(i), Y.row(i), d, count(i));
        double a = log_p.max();
        arma::vec prob_sample = 1.0 / (exp(a + log(sum(exp(log_p - a))) - log_p));
        z(i) = as_scalar(csample(one_to_2, 1, false, as<NumericVector>(wrap(prob_sample))));
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
      // sample beta1
      for (int j=0; j<d; j++) {
        arma::mat beta1_star = beta1;
        beta1_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta1_tune(j) * Sigma_beta1_tune_chol.slice(j));
        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha1_star = exp(Xbs * beta1_star);
        double mh1 = dMVN(beta1_star.col(j), mu_beta1, Sigma_beta1_chol);
        double mh2 = dMVN(beta1.col(j), mu_beta1, Sigma_beta1_chol);

        for (int i=0; i<N; i++) {
          if (z(i) == 0) {
            mh1 += LL_DM_row(alpha1_star.row(i), Y.row(i), d, count(i));
            mh2 += LL_DM_row(alpha1.row(i), Y.row(i), d, count(i));
          }

        }
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta1 = beta1_star;
          alpha1 = alpha1_star;
          beta1_accept(j) += 1.0 / n_mcmc;
        }
      }

      // sample beta2
      for (int j=0; j<d; j++) {
        arma::mat beta2_star = beta2;
        beta2_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta2_tune(j) * Sigma_beta2_tune_chol.slice(j));
        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha2_star = exp(Xbs * beta2_star);
        double mh1 = dMVN(beta2_star.col(j), mu_beta2, Sigma_beta2_chol);
        double mh2 = dMVN(beta2.col(j), mu_beta2, Sigma_beta2_chol);

        for (int i=0; i<N; i++) {
          if (z(i) == 1) {
            mh1 += LL_DM_row(alpha2_star.row(i), Y.row(i), d, count(i));
            mh2 += LL_DM_row(alpha2.row(i), Y.row(i), d, count(i));
          }

        }
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta2 = beta2_star;
          alpha2 = alpha2_star;
          beta2_accept(j) += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample mu_beta
    //

    if (sample_mu_beta) {
      // mu_beta_1
      arma::mat A1 = Sigma_beta_0_inv + d * Sigma_beta1_inv;
      arma::vec b1 = Sigma_beta_0_inv * mu_beta_0;
      for (int j=0; j<d; j++) {
        b1 += Sigma_beta1_inv * beta1.col(j);
      }
      mu_beta1= rMVNArma(A1, b1);
      // mu_beta_2
      arma::mat A2 = Sigma_beta_0_inv + d * Sigma_beta2_inv;
      arma::vec b2 = Sigma_beta_0_inv * mu_beta_0;
      for (int j=0; j<d; j++) {
        b2 += Sigma_beta2_inv * beta2.col(j);
      }
      mu_beta2= rMVNArma(A2, b2);
    }

    //
    // sample Sigma_beta
    //

    if (sample_Sigma_beta) {
      // Sigma_beta1
      arma::mat tmp1(df, df, arma::fill::zeros);
      for (int j=0; j<d; j++) {
        tmp1 += (beta1.col(j) - mu_beta1) * (beta1.col(j) - mu_beta1).t();
      }
      Sigma_beta1_inv = rWishartArmaMat(d + nu, inv_sympd(tmp1 + nu * S));
      Sigma_beta1 = inv_sympd(Sigma_beta1_inv);
      Sigma_beta1_chol = chol(Sigma_beta1);
      // Sigma_beta2
      arma::mat tmp2(df, df, arma::fill::zeros);
      for (int j=0; j<d; j++) {
        tmp2 += (beta2.col(j) - mu_beta2) * (beta2.col(j) - mu_beta2).t();
      }
      Sigma_beta2_inv = rWishartArmaMat(d + nu, inv_sympd(tmp2 + nu * S));
      Sigma_beta2 = inv_sympd(Sigma_beta2_inv);
      Sigma_beta2_chol = chol(Sigma_beta2);
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      // mu_save.row(save_idx) = mu.t();
      alpha1_save.subcube(span(save_idx), span(), span()) = alpha1;
      alpha2_save.subcube(span(save_idx), span(), span()) = alpha2;
      beta1_save.subcube(span(save_idx), span(), span()) = beta1;
      beta2_save.subcube(span(save_idx), span(), span()) = beta2;
      mu_beta1_save.row(save_idx) = mu_beta1.t();
      mu_beta2_save.row(save_idx) = mu_beta2.t();
      Sigma_beta1_save.subcube(span(save_idx), span(), span()) = Sigma_beta1;
      Sigma_beta2_save.subcube(span(save_idx), span(), span()) = Sigma_beta2;
      z_save.row(save_idx) = z.t();
    }

    //
    // sample mixing indicator z
    //

    if (sample_z) {
      for (int i=0; i<N; i++) {
        arma::vec log_p = log(phi);
        log_p(0) += LL_DM_row(alpha1.row(i), Y.row(i), d, count(i));
        log_p(1) += LL_DM_row(alpha2.row(i), Y.row(i), d, count(i));
        double a = log_p.max();
        arma::vec prob_sample = 1.0 / (exp(a + log(sum(exp(log_p - a))) - log_p));
        z(i) = as_scalar(csample(one_to_2, 1, false, as<NumericVector>(wrap(prob_sample))));
      }
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
    file_out << "Average acceptance rate for beta1  = " << mean(beta1_accept) <<
      " for chain " << n_chain << "\n";
      file_out << "Average acceptance rate for beta2  = " << mean(beta2_accept) <<
        " for chain " << n_chain << "\n";
    }

  // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    _["z"] = z_save,
    _["alpha1"] = alpha1_save,
    _["alpha2"] = alpha2_save,
    _["beta1"] = beta1_save,
    _["beta2"] = beta2_save,
    _["mu_beta1"] = mu_beta1_save,
    _["mu_beta2"] = mu_beta2_save,
    _["Sigma_beta1"] = Sigma_beta1_save,
    _["Sigma_beta2"] = Sigma_beta2_save);

}
