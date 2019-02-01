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
// Created 02.01.2019
// Last updated 02.01.2019

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
////////////////////////// Elliptical Slice Samplers //////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
///////////// Elliptical Slice Sampler for unobserved covariate X /////////////
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess_X (const double& X_current,
                  const double& X_prior,
                  const double& mu_X,
                  const arma::rowvec& a_current,
                  const arma::rowvec& b_current,
                  const arma::rowvec& c_current,
                  const arma::rowvec& y_current,
                  const double& d, const double& count_double,
                  const std::string& file_name) {

  // adjust for non-zero mean
  arma::rowvec alpha_current(d);
  for (int jj=0; jj<d; jj++) {
    alpha_current(jj) = exp(a_current(jj) - pow(b_current(jj) - X_current - mu_X, 2.0) / c_current(jj));
  }

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
  arma::rowvec alpha_ess = alpha_current;
  bool test = true;


  // Slice sampling loop
  while (test) {
    // compute proposal for angle difference and check to see if it is on the slice
    double X_proposal = X_current * cos(phi_angle) + X_prior * sin(phi_angle);

    // adjust for non-zero mean
    arma::rowvec alpha_proposal(d);
    for (int jj=0; jj<d; jj++) {
      alpha_proposal(jj) = exp(a_current(jj) - pow(b_current(jj) - X_proposal - mu_X, 2.0) / c_current(jj));
    }

    // calculate log likelihood of proposed value
    double proposal_log_like = LL_DM_row(alpha_proposal, y_current, d,
                                         count_double);
    // control to limit alpha from getting unreasonably large
    if (alpha_proposal.max() > pow(10.0, 10) ) {
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
      // control to limit alpha from getting unreasonably small
    } else if (alpha_proposal.min() < pow(10.0, -8) ) {
      if (phi_angle > 0.0) {
        phi_angle_max = phi_angle;
      } else if (phi_angle < 0.0) {
        phi_angle_min = phi_angle;
      } else {
        Rprintf("Bug - ESS for X shrunk to current position with small alpha \n");
        // set up output messages
        std::ofstream file_out;
        file_out.open(file_name, std::ios_base::app);
        file_out << "Bug - ESS for X shrunk to current position with small alpha \n";
        // close output file
        file_out.close();
        test = false;
      }
    } else {
      if (proposal_log_like > hh) {
        // proposal is on the slice
        X_ess = X_proposal;
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
      _["alpha"] = alpha_ess));
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List predictRcppDMBummer (const arma::mat& Y_pred, const double mu_X,
                          const double s2_X, const double min_X,
                          const double max_X,
                          List params,
                          List samples,
                          std::string file_name="DM-predict") {


  // Load parameters

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

  // set up dimensions
  double N_pred = Y_pred.n_rows;
  double d = Y_pred.n_cols;

  // Load MCMC estimated parameters
  arma::mat a_fit = as<arma::mat>(samples["a"]);
  arma::mat b_fit = as<arma::mat>(samples["b"]);
  arma::mat c_fit = as<arma::mat>(samples["c"]);

  int n_samples = a_fit.n_rows;

  // count - sum of counts at each site
  arma::vec count_pred(N_pred);
  for (int i=0; i<N_pred; i++) {
    count_pred(i) = sum(Y_pred.row(i));
  }

  // constant vectors
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec zeros_d(d, arma::fill::zeros);

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
  //
  // initialize values
  //

  arma::rowvec a = a_fit.row(0);
  arma::rowvec b = b_fit.row(0);
  arma::rowvec c = c_fit.row(0);

  arma::mat alpha_pred(N_pred, d);
  for (int i=0; i<N_pred; i++) {
    for (int j=0; j<d; j++) {
      alpha_pred(i, j) = exp(a(j) - pow(b(j) - X_pred(i) - mu_X, 2.0) / c(j));
      if(alpha_pred(i, j) > pow(10.0, 10)) {
        alpha_pred(i, j) = pow(10.0, 10);
      }
      if(alpha_pred(i, j) < pow(10.0, -8)) {
        alpha_pred(i, j) = pow(10.0, -8);
      }
    }
  }

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

    a = a_fit.row(k);
    b = b_fit.row(k);
    c = c_fit.row(k);

    // run for 10 iterations per posterior sample to "guarantee" convergence
    for (int j=0; j<n_rep; j++) {

      if (sample_X) {
        for (int i=0; i<N_pred; i++) {
          double X_prior = R::rnorm(0.0, s_X);
          Rcpp::List ess_out = ess_X(X_pred(i), X_prior, mu_X, a, b, c,
                                     Y_pred.row(i), d,
                                     count_pred(i), file_name);
          X_pred(i) = as<double>(ess_out["X"]);
          alpha_pred.row(i) = as<rowvec>(ess_out["alpha"]);
        }
      }
    }

    //
    // save variables
    //

    alpha_pred_save.subcube(span(k), span(), span()) = alpha_pred;
    // add the mean back in
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
