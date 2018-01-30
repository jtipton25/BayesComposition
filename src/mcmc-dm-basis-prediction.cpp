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
// Last updated 05.15.2017

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
////////////////////////// Elliptical Slice Samplers //////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
///////////// Elliptical Slice Sampler for unobserved covariate X /////////////
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess_X (const double& X_current, const double& X_prior,
                  const double& mu_X,
                  const arma::rowvec& mu_current,
                  const arma::mat& beta_current,
                  const arma::rowvec& alpha_current,
                  const arma::rowvec& y_current,
                  const arma::rowvec& Xbs_current,
                  const arma::vec& knots,
                  const double& d, const int& degree, const int& df,
                  const arma::vec& rangeX, const double& count_double,
                  const std::string& file_name) {
  // const arma::mat& R_tau_current,
  //
  //
  //
  //

  // calculate log likelihood of current value
  double current_log_like = LL_DM_row(alpha_current, y_current, d, count_double);
  double hh = log(R::runif(0.0, 1.0)) + current_log_like;

  // Setup a bracket and pick a first proposal
  // Bracket whole ellipse with both edges at first proposed point
  double phi_angle = R::runif(0.0, 1.0) * 2.0 * arma::datum::pi;
  double phi_angle_min = phi_angle - 2.0 * arma::datum::pi;
  double phi_angle_max = phi_angle;

  // set up save variables
  double X_ess = X_current;
  arma::rowvec Xbs_ess = Xbs_current;
  arma::rowvec alpha_ess = alpha_current;
  bool test = true;


  // Slice sampling loop
  while (test) {
    // compute proposal for angle difference and check to see if it is on the slice
    double X_proposal = X_current * cos(phi_angle) + X_prior * sin(phi_angle);
    // adjust for non-zero mean
    arma::vec X_tilde(1);
    X_tilde(0) = X_proposal + mu_X;
    arma::rowvec Xbs_proposal = bs_cpp(X_tilde, df, knots, degree, true,
                                       rangeX);
    arma::rowvec alpha_proposal = exp(mu_current + Xbs_proposal * beta_current);

    // calculate log likelihood of proposed value
    double proposal_log_like = LL_DM_row(alpha_proposal, y_current, d,
                                         count_double);
    // control to limit alpha from getting unreasonably large
    if (alpha_proposal.max() > pow(10, 10) ) {
      if (phi_angle > 0.0) {
        phi_angle_max = phi_angle;
      } else if (phi_angle < 0.0) {
        phi_angle_min = phi_angle;
      } else {
        Rprintf("Bug - ESS for X shrunk to current position with large alpha \n");
        // set up output messages
        std::ofstream file_out;
        file_out.open(file_name, std::ios_base::app);
        file_out << "Bug - ESS for X shrunk to current position with large alpha \n";
        // close output file
        file_out.close();
        test = false;
      }
    } else {
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
        Rprintf("Bug - ESS for X shrunk to current position \n");
        // set up output messages
        std::ofstream file_out;
        file_out.open(file_name, std::ios_base::app);
        file_out << "Bug - ESS for X shrunk to current position \n";
        // close output file
        file_out.close();
        test = false;
      }
    }
    // Propose new angle difference
    phi_angle = R::runif(0.0, 1.0) * (phi_angle_max - phi_angle_min) +
      phi_angle_min;
  }
  return(Rcpp::List::create(
      _["X"] = X_ess,
      _["Xbs"] = Xbs_ess,
      _["alpha"] = alpha_ess));
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List predictRcppDMBasis (const arma::mat& Y_pred, const double mu_X,
                         const double s2_X, const double min_X,
                         const double max_X,
                         List params,
                         List samples,
                         std::string file_name="DM-predict") {

  // List mcmcRcppDMBasis (const arma::mat& Y, const arma::vec& X,
  //                     const arma::mat& Y_pred,
  //                     List params,
  //                     int n_chain=1, bool pool_s2_tau2=true,
  //                     std::string file_name="DM-fit",
  //                     std::string corr_function="exponential") {


  // Load parameters
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
  // default number of iterations to "convergence
  int n_rep = 10;
  if (params.containsElementNamed("n_rep")) {
    n_rep = as<int>(params["n_rep"]);
  }

  // add in option for reference category for mu
  // this option fixes the first mean parameter at 0
  // to allow for model identifiability
  bool mu_reference_category = true;
  if (params.containsElementNamed("mu_reference_category")) {
    mu_reference_category = as<bool>(params["mu_reference_category"]);
  }
  // add in option for reference category for Sigma
  // this option fixes the first variance parameter tau at 1
  // to allow for model identifiability
  bool Sigma_reference_category = true;
  if (params.containsElementNamed("Sigma_reference_category")) {
    Sigma_reference_category = as<bool>(params["Sigma_reference_category"]);
  }

  // set up dimensions
  double N_pred = Y_pred.n_rows;
  double d = Y_pred.n_cols;

  // Load MCMC estimated parameters
  // arma::cube alpha = as<arma::cube>(samples["alpha"]);
  arma::mat mu_fit = as<arma::mat>(samples["mu"]);
  arma::cube beta_fit = as<arma::cube>(samples["beta"]);
  int n_samples = beta_fit.n_rows;
  if (beta_fit.n_cols != df) {
    stop("df for posterior estimates for beta not equal to df for prediction");
  }

  // count - sum of counts at each site
  arma::vec count_pred(N_pred);
  for (int i=0; i<N_pred; i++) {
    count_pred(i) = sum(Y_pred.row(i));
  }

  // constant vectors
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec zero_df(df, arma::fill::zeros);

  // default X tuning parameter standard deviation of 0.25
  double X_tune_tmp = 0.5;
  if (params.containsElementNamed("X_tune")) {
    X_tune_tmp = as<double>(params["X_tune"]);
  }

  //
  // Turn on/off sampler for X
  //

  double s_X = sqrt(s2_X);

  bool sample_X = true;
  if (params.containsElementNamed("sample_X")) {
    sample_X = as<bool>(params["sample_X"]);
  }
  // default elliptical slice sampler for X
  bool sample_X_mh = false;
  if (params.containsElementNamed("sample_X_mh")) {
    sample_X_mh = as<bool>(params["sample_X_mh"]);
  }
  arma::vec X_pred(N_pred, arma::fill::zeros);
  for (int i=0; i<N_pred; i++) {
    X_pred(i) = R::rnorm(0.0, s_X);
  }
  // double minX = as<double>(params["minX"]);
  // double maxX = as<double>(params["maxX"]);
  arma::vec rangeX(2);
  rangeX(0)=min_X;
  rangeX(1)=max_X;
  // rangeX(0)=minX-1*s_X;   // Assuming X is mean 0 and sd 1, this gives 3 sds beyond
  // rangeX(1)=maxX+1*s_X;   // buffer for the basis beyond the range of the
  // observational data
  arma::vec knots = linspace(rangeX(0), rangeX(1), df-degree-1+2);
  knots = knots.subvec(1, df-degree-1);

  arma::mat Xbs_pred = bs_cpp(X_pred, df, knots, degree, true, rangeX);

  //
  // initialize values
  //

  //
  // set default, fixed parameters and turn on/off samplers for testing
  //

  arma::vec mu = mu_fit.row(0).t();
  arma::mat mu_mat(N_pred, d, arma::fill::zeros);
  for (int i=0; i<N_pred; i++) {
    mu_mat.row(i) = mu.t();
  }
  arma::mat beta = beta_fit.subcube(0, 0, 0, 0, df-1, d-1);
  arma::mat alpha_pred = exp(mu_mat + Xbs_pred * beta);

  // setup save variables
  arma::cube alpha_pred_save(n_samples, N_pred, d, arma::fill::zeros);
  arma::mat X_save(n_samples, N_pred, arma::fill::zeros);

  // initialize tuning

  arma::vec X_tune(N_pred, arma::fill::ones);
  X_tune *= X_tune_tmp;
  arma::vec X_accept(N_pred, arma::fill::zeros);
  arma::vec X_accept_batch(N_pred, arma::fill::zeros);

  Rprintf("Starting predictions, running for %d iterations \n",
          n_samples);
  // set up output messages
  std::ofstream file_out;
  file_out.open(file_name, std::ios_base::app);
  file_out << "Starting predictions, running for " << n_samples <<
    " iterations \n";
  // close output file
  file_out.close();
  // }

  // Start predictions
  for (int k=0; k<n_samples; k++) {
    if ((k+1) % message == 0) {
      Rprintf("Prediction iteration %d\n", k+1);
      // set up output messages
      std::ofstream file_out;
      file_out.open(file_name, std::ios_base::app);
      file_out << "Prediction iteration " << k+1 << "\n";
      // close output file
      file_out.close();
    }

    Rcpp::checkUserInterrupt();

    //
    // sample X - ESS
    //

    // run for 10 iterations per posterior sample to "guarantee" convergence
    for (int j=0; j<n_rep; j++) {
      arma::vec mu = mu_fit.row(0).t();
      arma::mat mu_mat(N_pred, d, arma::fill::zeros);
      for (int i=0; i<N_pred; i++) {
        mu_mat.row(i) = mu.t();
      }
      beta = beta_fit.subcube(k, 0, 0, k, df-1, d-1);
      arma::mat alpha_pred = exp(mu_mat + Xbs_pred * beta);
      if (sample_X) {
        for (int i=0; i<N_pred; i++) {
          double X_prior = R::rnorm(0.0, s_X);
          Rcpp::List ess_out = ess_X(X_pred(i), X_prior, mu_X, mu.t(), beta, alpha_pred.row(i),
                                     Y_pred.row(i), Xbs_pred.row(i), knots, d,
                                     degree, df, rangeX, count_pred(i), file_name);
          X_pred(i) = as<double>(ess_out["X"]);
          Xbs_pred.row(i) = as<rowvec>(ess_out["Xbs"]);
          alpha_pred.row(i) = as<rowvec>(ess_out["alpha"]);
        }
      }
    }

    //
    // save variables
    //

    alpha_pred_save.subcube(span(k), span(), span()) = alpha_pred;
    X_save.row(k) = X_pred.t() + mu_X;

  }

  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  // if (sample_X) {
  //   file_out << "Average acceptance rate for X  = " << mean(X_accept) <<
  //     " for chain " << n_chain << "\n";
  // }
  // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    _["alpha_pred"] = alpha_pred_save,
    _["X"] = X_save);

}
