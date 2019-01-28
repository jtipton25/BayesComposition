// #define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include "BayesComp.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Multivariate Gaussian Process for Inverse Inference

//
// Author: John Tipton
//
// Created 11.29.2016
// Last updated 06.09.2017

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
///////////// Elliptical Slice Sampler for unobserved covariate X /////////////
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess_X (const double& X_current,
                  const double& X_prior,
                  const double& mu_X,
                  const arma::vec& X_knots,
                  const arma::rowvec& y_current,
                  const arma::rowvec& Xbs_current,
                  const arma::rowvec& alpha_current,
                  const arma::mat& beta_current,
                  const double& sigma_current,
                  const arma::vec& knots,
                  const double& d,
                  const int& degree,
                  const int& df,
                  const arma::vec& rangeX,
                  const std::string& file_name) {
  // eta_star_current is the current value of the joint multivariate predictive process
  // prior_sample is a sample from the prior joing multivariate predictive process
  // R_tau is the current value of the Cholskey decomposition for  predictive process linear interpolator
  // Z_current is the current predictive process linear

  // calculate log likelihood of current value
  double current_log_like = 0.0;

  for (int j=0; j<d; j++) {
    current_log_like += R::dnorm(y_current(j),  alpha_current(j),
                             sigma_current, true);
  }


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
    arma::vec X_tilde(1);
    X_tilde(0) = X_proposal + mu_X;
    arma::rowvec Xbs_proposal = bs_cpp(X_tilde, df, knots, degree, true,
                                       rangeX);
    arma::rowvec alpha_proposal = exp(Xbs_proposal * beta_current);

    // calculate log likelihood of proposed value
    double proposal_log_like = 0.0;

    for (int j=0; j<d; j++) {
      proposal_log_like += R::dnorm(y_current(j),  alpha_proposal(j),
                                   sigma_current, true);
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
      Rprintf("Bug detected - ESS for X shrunk to current position and still not acceptable \n");
      // set up output messages
      std::ofstream file_out;
      file_out.open(file_name, std::ios_base::app);
      file_out << "Bug - ESS for X shrunk to current position and still not acceptable \n";
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

// [[Rcpp::export]]
List predictRcppBasis (const arma::mat& Y_pred,
                       const double mu_X,
                       const double s2_X,
                       const double min_X,
                       const double max_X,
                       List params,
                       List samples,
                       std::string file_name="mvgp-predict") {
  // arma::mat& R, arma::vec& tau2, double& phi, double& sigma2,
  // arma::mat& eta_star,

  // Load parameters

  // set up dimensions
  double N_pred = Y_pred.n_rows;
  double d = Y_pred.n_cols;
  double B = Rf_choose(d, 2);
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec ones_B(B, arma::fill::ones);


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

  // enable the sampler
  bool sample_X = true;
  if (params.containsElementNamed("sample_X")) {
    sample_X = as<bool>(params["sample_X"]);
  }

  // standard deviation of covariate
  double s_X = sqrt(s2_X);

  arma::vec X_pred(N_pred, arma::fill::zeros);
  for (int i=0; i<N_pred; i++) {
    X_pred(i) = R::rnorm(0.0, s_X);
  }

  arma::vec rangeX(2);
  rangeX(0)=min_X;
  rangeX(1)=max_X;
  // observational data
  arma::vec knots = linspace(rangeX(0), rangeX(1), df-degree-1+2);
  // knots = knots.subvec(1, df-degree);
  knots = knots.subvec(1, df-degree-1);

  arma::mat Xbs_pred = bs_cpp(X_pred, df, knots, degree, true, rangeX);
  // predictive process knots
  arma::vec X_knots = as<vec>(params["X_knots"]);
  double N_knots = X_knots.n_elem;

  // Load MCMC estimated parameters

  arma::cube beta_fit = as<arma::cube>(samples["beta"]);
  int n_samples = beta_fit.n_rows;
  if (beta_fit.n_cols != df) {
    stop("df for posterior estimates for beta not equal to df for prediction");
  }

  arma::mat beta = beta_fit.subcube(0, 0, 0, 0, N_knots-1, d-1);
  arma::vec sigma2_fit = as<vec>(samples["sigma2"]);
  double sigma = sqrt(sigma2_fit(0));

  arma::mat alpha_pred = exp(Xbs_pred * beta);

  // setup save variables
  arma::cube alpha_pred_save(n_samples, N_pred, d, arma::fill::zeros);
  arma::mat X_save(n_samples, N_pred, arma::fill::zeros);


  //   // default X tuning parameter standard deviation of 0.25
  // double X_tune_tmp = 2.5;
  // if (params.containsElementNamed("X_tune")) {
  //   X_tune_tmp = as<double>(params["X_tune"]);
  // }
  //
  // // Default sampling of missing covariate
  // bool sample_X = true;
  // if (params.containsElementNamed("sample_X")) {
  //   sample_X = as<bool>(params["sample_X"]);
  // }
  // // Default sampling of missing covariate using ESS
  // bool sample_X_mh = false;
  // if (params.containsElementNamed("sample_X_mh")) {
  //   sample_X_mh = as<bool>(params["sample_X_mh"]);
  // }
  // arma::vec X_pred(N_pred, arma::fill::zeros);
  // for (int i=0; i<N_pred; i++) {
  //   X_pred(i) = R::rnorm(0.0, s_X);
  // }

  // default correlation functions

  //

  Rprintf("Starting predictions, running for %d iterations \n",
          n_samples);
  // set up output messages
  std::ofstream file_out;
  file_out.open(file_name, std::ios_base::app);
  file_out << "Starting predictions, running for " << n_samples <<
    " iterations \n";
  // close output file
  file_out.close();


  // Start MCMC chain
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
    // sample X
    //

    // run for 10 iterations per posterior sample to "guarantee" convergence
    for (int j=0; j<n_rep; j++) {
      // Load MCMC estimated parameters
      beta = beta_fit.subcube(k, 0, 0, k, df-1, d-1);
      sigma = sqrt(sigma2_fit(k));
      // arma::mat alpha_pred = exp(mu_mat + Xbs_pred * beta);
      alpha_pred = exp(Xbs_pred * beta);
      if (sample_X) {
        for (int i=0; i<N_pred; i++) {
          double X_prior = R::rnorm(0.0, s_X);
          Rcpp::List ess_out = ess_X(X_pred(i), X_prior, mu_X,
                                     X_knots, Y_pred.row(i), Xbs_pred.row(i),
                                     alpha_pred.row(i), beta, sigma,
                                     knots, d, degree, df, rangeX, file_name);
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
  // close output file
  file_out.close();

  return Rcpp::List::create(
    _["alpha_pred"] = alpha_pred_save,
    _["X"] = X_save);
}
