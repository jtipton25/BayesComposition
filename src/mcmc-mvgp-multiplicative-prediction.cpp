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

Rcpp::List ess_X_multiplicative (const double& X_current, const double& X_prior,
                  const double& mu_X, const arma::vec& X_knots,
                  const arma::rowvec& y_current,
                  const arma::vec& mu_current,
                  const arma::mat& eta_star_current,
                  const arma::rowvec& zeta_current,
                  const arma::rowvec& D_current,
                  const arma::rowvec& c_current,
                  const arma::mat& R_tau_current,
                  const arma::rowvec& Z_current, const double& phi_current,
                  const double& sigma_current, const arma::mat C_inv_current,
                  const int& N_pred, const int& d,
                  const std::string& file_name,
                  const std::string& corr_function) {
  // eta_star_current is the current value of the joint multivariate predictive process
  // prior_sample is a sample from the prior joing multivariate predictive process
  // R_tau is the current value of the Cholskey decomposition for  predictive process linear interpolator
  // Z_current is the current predictive process linear

  // calculate log likelihood of current value
  double current_log_like = 0.0;
  for (int j=0; j<d; j++) {
    current_log_like += R::dnorm(y_current(j), mu_current(j) + zeta_current(j),
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
  arma::rowvec D_ess = D_current;
  arma::rowvec c_ess = c_current;
  arma::rowvec Z_ess = Z_current;
  arma::rowvec zeta_ess = zeta_current;
  bool test = true;

  // Slice sampling loop
  while (test) {
    // compute proposal for angle difference and check to see if it is on the slice
    double X_proposal = X_current * cos(phi_angle) + X_prior * sin(phi_angle);
    double X_tilde = X_proposal + mu_X;
    arma::rowvec D_proposal = sqrt(pow(X_tilde - X_knots, 2)).t();
    if (corr_function == "gaussian") {
      D_proposal = pow(D_proposal, 2.0);
    }
    arma::rowvec c_proposal = exp( - D_proposal / phi_current);
    arma::rowvec Z_proposal = c_proposal * C_inv_current;
    arma::rowvec zeta_proposal = Z_proposal * eta_star_current * R_tau_current;

    // calculate log likelihood of proposed value
    double proposal_log_like = 0.0;
    for (int j=0; j<d; j++) {
      proposal_log_like += R::dnorm(y_current(j), mu_current(j) + zeta_proposal(j),
                                    sigma_current, true);
    }
    if (proposal_log_like > hh) {
      // proposal is on the slice
      X_ess = X_proposal;
      D_ess = D_proposal;
      c_ess = c_proposal;
      Z_ess = Z_proposal;
      zeta_ess = zeta_proposal;
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
      _["D"] = D_ess,
      _["c"] = c_ess,
      _["Z"] = Z_ess,
      _["zeta"] = zeta_ess));
}

// [[Rcpp::export]]
List predictRcppMVGPMultiplicative (const arma::mat& Y_pred, const double mu_X,
                                    const double s2_X, const double min_X,
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

  // standard deviation of covariate
  double s_X = sqrt(s2_X);
  // predictive process knots
  arma::vec X_knots = as<vec>(params["X_knots"]);
  double N_knots = X_knots.n_elem;

  // Load MCMC estimated parameters
  arma::mat mu_fit = as<mat>(samples["mu"]);
  int n_samples = mu_fit.n_rows;
  arma::vec mu = mu_fit.row(0).t();
  arma::mat mu_mat(N_pred, d);
  for (int i=0; i<N_pred; i++) {
    mu_mat.row(i) = mu.t();
  }
  arma::vec phi_fit = as<vec>(samples["phi"]);
  double phi = phi_fit(0);
  arma::cube eta_star_fit = as<cube>(samples["eta_star"]);
  arma::mat eta_star = eta_star_fit.subcube(0, 0, 0, 0, N_knots-1, d-1);
  arma::mat xi_fit = as<mat>(samples["xi"]);
  arma::vec xi = xi_fit.row(0).t();
  arma::mat tau2_fit = as<mat>(samples["tau2"]);
  arma::vec tau2 = tau2_fit.row(0).t();
  arma::cube R_fit = as<cube>(samples["R"]);
  arma::mat R = R_fit.subcube(0, 0, 0, 0, d-1, d-1);
  arma::cube R_tau_fit = as<cube>(samples["R_tau"]);
  arma::mat R_tau = R_tau_fit.subcube(0, 0, 0, 0, d-1, d-1);
  arma::vec sigma2_fit = as<vec>(samples["sigma2"]);
  double sigma2 = sigma2_fit(0);
  double sigma = sqrt(sigma2);

  // default to message output every 5000 iterations
  int message = 5000;
  if (params.containsElementNamed("message")) {
    message = as<int>(params["message"]);
  }
  // default X tuning parameter standard deviation of 0.25
  double X_tune_tmp = 2.5;
  if (params.containsElementNamed("X_tune")) {
    X_tune_tmp = as<double>(params["X_tune"]);
  }

  // Default sampling of missing covariate
  bool sample_X = true;
  if (params.containsElementNamed("sample_X")) {
    sample_X = as<bool>(params["sample_X"]);
  }
  // Default sampling of missing covariate using ESS
  bool sample_X_mh = false;
  if (params.containsElementNamed("sample_X_mh")) {
    sample_X_mh = as<bool>(params["sample_X_mh"]);
  }
  arma::vec X_pred(N_pred, arma::fill::zeros);
  for (int i=0; i<N_pred; i++) {
    X_pred(i) = R::rnorm(0.0, s_X);
  }

  // default correlation functions
  std::string corr_function="exponential";
  if (params.containsElementNamed("corr_function")) {
    corr_function = as<std::string>(params["corr_function"]);
  }

  arma::mat D = makeDistARMA(X_pred, X_knots);
  arma::mat D_knots = makeDistARMA(X_knots, X_knots);
  if (corr_function == "gaussian") {
    D = pow(D, 2.0);
    D_knots = pow(D_knots, 2.0);
  } else if (corr_function != "exponential") {
    stop ("the only valid correlation functions are exponential and gaussian");
  }

  //
  // initialize values
  //

  //
  // set default, fixed parameters and turn on/off samplers for testing
  //

  //
  // Construct Gaussian Process Correlation matrices
  //

  arma::mat C = exp(- D_knots / phi);
  arma::mat C_chol = chol(C);
  arma::mat C_inv = inv_sympd(C);
  arma::mat c = exp( - D / phi);
  arma::mat Z = c * C_inv;

  arma::mat zeta_pred = Z * eta_star * R_tau;

  // Initialize constant vectors

  arma::vec zero_knots(N_knots, arma::fill::zeros);
  arma::vec zero_knots_d(N_knots*d, arma::fill::zeros);

  // setup save variables
  arma::cube zeta_pred_save(n_samples, N_pred, d, arma::fill::zeros);
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

    // Load MCMC estimated parameters
    mu = mu_fit.row(k).t();
    phi = phi_fit(k);
    C = exp(- D_knots / phi);
    C_chol = chol(C);
    C_inv = inv_sympd(C);
    c = exp( - D / phi);
    Z = c * C_inv;
    eta_star = eta_star_fit.subcube(k, 0, 0, k, N_knots-1, d-1);
    xi = xi_fit.row(k).t();
    tau2 = tau2_fit.row(k).t();
    R = R_fit.subcube(k, 0, 0, k, d-1, d-1);
    R_tau = R_tau_fit.subcube(k, 0, 0, k, d-1, d-1);
    sigma2 = sigma2_fit(k);
    zeta_pred = Z * eta_star * R_tau;


    if (sample_X) {
      if (sample_X_mh) {
        // sample using Metropolis-Hastings
        for (int i=0; i<N_pred; i++) {
          arma::vec X_star = X_pred;
          X_star(i) += R::rnorm(0.0, X_tune(i));
          // add in prior mean here
          arma::rowvec D_proposal = sqrt(pow(X_star(i) + mu_X - X_knots, 2)).t();
          if (corr_function == "gaussian") {
            D_proposal = pow(D_proposal, 2.0);
          }
          // arma::rowvec D_proposal = sqrt(pow(X_star(i) - X_knots, 2)).t();
          arma::rowvec c_proposal = exp( - D_proposal / phi);
          arma::rowvec Z_proposal = c_proposal * C_inv;
          arma::rowvec zeta_proposal = Z_proposal * eta_star * R_tau;
          double mh1 = R::dnorm(X_star(i), 0.0, s_X, true);
          double mh2 = R::dnorm(X_pred(i), 0.0, s_X, true);
          // double mh1 = R::dnorm(X_star(i), mu_X, s_X, true);
          // double mh2 = R::dnorm(X(i), mu_X, s_X, true);
          for (int j=0; j<d; j++) {
            mh1 += R::dnorm(Y_pred(i, j), mu(j) + zeta_proposal(j), sigma, true);
            mh2 += R::dnorm(Y_pred(i, j), mu(j) + zeta_pred(i, j), sigma, true);
          }
          double mh = exp(mh1-mh2);
          if (mh > R::runif(0.0, 1.0)) {
            X_pred = X_star;
            D.row(i) = D_proposal;
            c.row(i) = c_proposal;
            Z.row(i) = Z_proposal;
            zeta_pred.row(i) = zeta_proposal;
            X_accept_batch(i) += 1.0 / 50.0;
          }
        }
        // update tuning
        if ((k+1) % 50 == 0){
          updateTuningVec(k, X_accept_batch, X_tune);
        }
      } else {
        // sample using ESS
        for (int i=0; i<N_pred; i++) {
          double X_prior = R::rnorm(0.0, s_X);
          Rcpp::List ess_out = ess_X_multiplicative(X_pred(i), X_prior, mu_X, X_knots, Y_pred.row(i),
                                     mu, eta_star, zeta_pred.row(i), D.row(i), c.row(i),
                                     R_tau, Z.row(i), phi, sigma,
                                     C_inv, N_pred, d, file_name,
                                     corr_function);
          X_pred(i) = as<double>(ess_out["X"]);
          D.row(i) = as<rowvec>(ess_out["D"]);
          c.row(i) = as<rowvec>(ess_out["c"]);
          Z.row(i) = as<rowvec>(ess_out["Z"]);
          zeta_pred.row(i) = as<rowvec>(ess_out["zeta"]);
        }
      }
    }

    //
    // save variables
    //

    zeta_pred_save.subcube(span(k), span(), span()) = zeta_pred;
    X_save.row(k) = X_pred.t() + mu_X;
  }

  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  // close output file
  file_out.close();

  return Rcpp::List::create(
    _["zeta_pred"] = zeta_pred_save,
    _["X"] = X_save);
}
