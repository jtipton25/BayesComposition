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

// Dirichlet-Multinomial Multivariate Gaussian Process for Inverse Inference

// Author: John Tipton
//
// Created 07.11.2016
// Last updated 07.19.2017

///////////////////////////////////////////////////////////////////////////////
/////////////////////////// Functions for sampling ////////////////////////////
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
////////////////////////// Elliptical Slice Samplers //////////////////////////
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
///////////// Elliptical Slice Sampler for random effect eta_star /////////////
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess_multiplicative (const arma::mat& eta_star_current,
                const arma::vec& prior_sample,
                const arma::mat& alpha_current,
                const arma::mat& mu_mat_current,
                const arma::mat& zeta_current,
                const arma::mat& R_tau_current,
                const arma::mat& epsilon_current,
                const arma::mat& Z_current, const arma::mat& y,
                const int& N, const int& d, const int& j,
                const arma::vec& count,
                const std::string& file_name, const int& n_chain) {
  // eta_star_current is the current value of the joint multivariate predictive process
  // prior_sample is a sample from the prior joing multivariate predictive process
  // R_tau is the current value of the Cholskey decomposition for  predictive process linear interpolator
  // Z_current is the current predictive process linear

  // calculate log likelihood of current value
  double current_log_like = LL_DM(alpha_current, y, N, d, count);
  double hh = log(R::runif(0.0, 1.0)) + current_log_like;


  // Setup a bracket and pick a first proposal
  // Bracket whole ellipse with both edges at first proposed point
  double phi_angle = R::runif(0.0, 1.0) * 2.0 * arma::datum::pi;
  double phi_angle_min = phi_angle - 2.0 * arma::datum::pi;
  double phi_angle_max = phi_angle;

  arma::mat eta_star_ess = eta_star_current;
  arma::mat eta_star_proposal = eta_star_current;
  arma::mat zeta_ess = zeta_current;
  arma::mat alpha_ess = alpha_current;
  bool test = true;

  // Slice sampling loop
  while (test) {
    // compute proposal for angle difference and check to see if it is on the slice
    eta_star_proposal.col(j) = eta_star_current.col(j) * cos(phi_angle) +
      prior_sample * sin(phi_angle);
    arma::mat zeta_proposal = Z_current * eta_star_proposal * R_tau_current;
    arma::mat alpha_proposal = exp(mu_mat_current + zeta_proposal + epsilon_current);
    // calculate log likelihood of proposed value
    double proposal_log_like = LL_DM(alpha_proposal, y, N, d, count);
    // control to limit alpha from getting unreasonably large
    if (alpha_proposal.max() > pow(10.0, 10.0) ) {
      // Rprintf("Bug - alpha (eta_star) is to large for LL to be stable \n");
      // // set up output messages
      // std::ofstream file_out;
      // file_out.open(file_name, std::ios_base::app);
      // file_out << "Bug - alpha (eta_star) is to large for LL to be stable on chain " << n_chain << "\n";
      // // close output file
      // file_out.close();
      if (phi_angle > 0.0) {
        phi_angle_max = phi_angle;
      } else if (phi_angle < 0.0) {
        phi_angle_min = phi_angle;
      } else {
        Rprintf("Bug - ESS for eta_star shrunk to current position with large alpha \n");
        // set up output messages
        std::ofstream file_out;
        file_out.open(file_name, std::ios_base::app);
        file_out << "Bug - ESS for eta_star shrunk to current position with large alpha on chain " << n_chain << "\n";
        // close output file
        file_out.close();
        // proposal failed and don't update the chain
        // eta_star_ess = eta_star_proposal;
        // zeta_ess = zeta_proposal;
        // alpha_ess = alpha_proposal;
        test = false;
      }
    } else {
      if (proposal_log_like > hh) {
        // proposal is on the slice
        eta_star_ess = eta_star_proposal;
        zeta_ess = zeta_proposal;
        alpha_ess = alpha_proposal;
        test = false;
      } else if (phi_angle > 0.0) {
        phi_angle_max = phi_angle;
      } else if (phi_angle < 0.0) {
        phi_angle_min = phi_angle;
      } else {
        Rprintf("Bug - ESS for eta_star shrunk to current position \n");
        // set up output messages
        std::ofstream file_out;
        file_out.open(file_name, std::ios_base::app);
        file_out << "Bug - ESS for eta_star shrunk to current position on chain " << n_chain << "\n";
        // close output file
        file_out.close();
        // proposal failed and don't update the chain
        // eta_star_ess = eta_star_proposal;
        // zeta_ess = zeta_proposal;
        // alpha_ess = alpha_proposal;
        test = false;
      }
    }
    // Propose new angle difference
    phi_angle = R::runif(0.0, 1.0) * (phi_angle_max - phi_angle_min) + phi_angle_min;
  }
  return(Rcpp::List::create(
      _["eta_star"] = eta_star_ess,
      _["zeta"] = zeta_ess,
      _["alpha"] = alpha_ess));
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppDMMVGPAdditive (const arma::mat& Y, const arma::vec& X,
                             List params,
                             int n_chain=1,
                             std::string file_name="DM-fit",
                             std::string corr_function="exponential") {

  // Load parameters
  int n_adapt = as<int>(params["n_adapt"]);
  int n_mcmc = as<int>(params["n_mcmc"]);
  int n_thin = as<int>(params["n_thin"]);

  // set up dimensions
  double N = Y.n_rows;
  double d = Y.n_cols;
  double B = Rf_choose(d, 2);

  // count - sum of counts at each site
  arma::vec count(N);
  for (int i=0; i<N; i++) {
    count(i) = sum(Y.row(i));
  }
  // add in option for reference category for Sigma
  bool Sigma_reference_category = false;
  if (params.containsElementNamed("Sigma_reference_category")) {
    Sigma_reference_category = as<bool>(params["Sigma_reference_category"]);
  }

  // predictive process knots
  arma::vec X_knots = as<vec>(params["X_knots"]);
  double N_knots = X_knots.n_elem;

  // constant vectors
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec ones_B(B, arma::fill::ones);
  arma::vec zero_knots(N_knots, arma::fill::zeros);
  arma::vec zero_d(d, arma::fill::zeros);

  // default normal prior for overall mean mu
  arma::vec mu_mu(d, arma::fill::zeros);
  if (params.containsElementNamed("mu_mu")) {
    mu_mu = as<vec>(params["mu_mu"]);
  }
  // default prior for overall mean mu
  arma::mat Sigma_mu(d, d, arma::fill::eye);
  Sigma_mu *= 5.0;
  if (params.containsElementNamed("Sigma_mu")) {
    Sigma_mu = as<double>(params["Sigma_mu"]);
  }
  arma::mat Sigma_mu_inv = inv_sympd(Sigma_mu);
  arma::mat Sigma_mu_chol = chol(Sigma_mu);

  // default uniform prior for Gaussian Process range
  double phi_L = 0.0001;
  if (params.containsElementNamed("phi_L")) {
    phi_L = as<double>(params["phi_L"]);
  }
  // default uniform prior for Gaussian Process range
  double phi_U = 1000.0;
  if (params.containsElementNamed("phi_U")) {
    phi_U = as<double>(params["phi_U"]);
  }

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

  // default xi_additive LKJ concentation parameter of 1
  double eta_additive = 1.0;
  if (params.containsElementNamed("eta_additive")) {
    eta_additive = as<double>(params["eta_additive"]);
  }


  // default to message output every 500 iterations
  int message = 500;
  if (params.containsElementNamed("message")) {
    message = as<int>(params["message"]);
  }

  // default phi tuning parameter standard deviation of 0.25
  double phi_tune = 0.25;
  if (params.containsElementNamed("phi_tune")) {
    phi_tune = as<double>(params["phi_tune"]);
  }

  // default mu tuning parameter
  double lambda_mu_tune = 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_mu_tune")) {
    lambda_mu_tune = as<double>(params["lambda_mu_tune"]);
  }

  // default lambda_eta_star tuning parameter standard deviation of 0.25
  double lambda_eta_star_tune_tmp = 0.25;
  if (params.containsElementNamed("lambda_eta_star_tune")) {
    lambda_eta_star_tune_tmp = as<double>(params["lambda_eta_star_tune"]);
  }
  arma::vec lambda_eta_star_tune(d, arma::fill::ones);
  lambda_eta_star_tune *= lambda_eta_star_tune_tmp;

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

  // Gaussian process distance matrix

  arma::mat D = makeDistARMA(X, X_knots);
  arma::mat D_knots = makeDistARMA(X_knots, X_knots);
  if (corr_function == "gaussian") {
    D = pow(D, 2.0);
    D_knots = pow(D_knots, 2.0);
  } else if (corr_function != "exponential") {
    stop ("the only valid correlation functions are exponential and gaussian");
  }

  //
  // Default for mu
  //

  arma::vec mu(d, arma::fill::randn);
  if (params.containsElementNamed("mu")) {
    mu = as<vec>(params["mu"]);
  }
  bool sample_mu = true;
  if (params.containsElementNamed("sample_mu")) {
    sample_mu = as<bool>(params["sample_mu"]);
  }
  arma::mat mu_mat(N, d);
  for (int i=0; i<N; i++) {
    mu_mat.row(i) = mu.t();
  }

  //
  // Default for Gaussian process range parameter phi
  //

  double phi = std::min(R::runif(phi_L, phi_U), 5.0);
  if (params.containsElementNamed("phi")) {
    phi = as<double>(params["phi"]);
  }
  bool sample_phi = true;
  if (params.containsElementNamed("sample_phi")) {
    sample_phi = as<bool>(params["sample_phi"]);
  }

  //
  // Gaussian process sill parameter tau2 and hyperprior lambda_tau2
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
  if (Sigma_reference_category) {
    tau2(d-1) = 1.0;
    tau(d-1) = 1.0;
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
  // Construct Gaussian Process Correlation matrices
  //

  arma::mat C = exp(- D_knots / phi);
  arma::mat C_chol = chol(C);
  arma::mat C_inv = inv_sympd(C);
  arma::mat c = exp( - D / phi);
  arma::mat Z = c * C_inv;

  //
  // Default predictive process random effect eta_star
  //

  arma::mat eta_star = mvrnormArmaChol(d, zero_knots, C_chol).t();
  if (params.containsElementNamed("eta_star")) {
    eta_star = as<mat>(params["eta_star"]);
  }
  bool sample_eta_star = true;
  if (params.containsElementNamed("sample_eta_star")) {
    sample_eta_star = as<bool>(params["sample_eta_star"]);
  }
  bool sample_eta_star_mh = false;
  if (params.containsElementNamed("sample_eta_star_mh")) {
    sample_eta_star_mh = as<bool>(params["sample_eta_star_mh"]);
  }

  arma::mat R(d, d, arma::fill::eye);
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

  arma::mat zeta = Z * eta_star * R_tau;
  arma::mat alpha = exp(mu_mat + zeta + epsilon);

  // setup save variables
  int n_save = n_mcmc / n_thin;
  arma::cube alpha_save(n_save, N, d, arma::fill::zeros);
  arma::cube zeta_save(n_save, N, d, arma::fill::zeros);
  arma::cube epsilon_save(n_save, N, d, arma::fill::zeros);
  arma::mat mu_save(n_save, d, arma::fill::zeros);
  arma::mat tau2_save(n_save, d, arma::fill::zeros);
  arma::vec s2_tau2_save(n_save, arma::fill::zeros);
  arma::mat tau2_additive_save(n_save, d, arma::fill::zeros);
  arma::vec s2_tau2_additive_save(n_save, arma::fill::zeros);
  arma::vec phi_save(n_save, arma::fill::zeros);
  arma::cube eta_star_save(n_save, N_knots, d, arma::fill::zeros);
  arma::cube R_save(n_save, d, d, arma::fill::zeros);
  arma::cube R_tau_save(n_save, d, d, arma::fill::zeros);
  arma::cube R_additive_save(n_save, d, d, arma::fill::zeros);
  arma::cube R_tau_additive_save(n_save, d, d, arma::fill::zeros);
  arma::mat xi_additive_save(n_save, B, arma::fill::zeros);

  // initialize tuning
  double phi_accept = 0.0;
  double phi_accept_batch = 0.0;
  double s2_tau2_accept = 0.0;
  double s2_tau2_accept_batch = 0.0;
  double s2_tau2_tune = 1.0;
  double s2_tau2_additive_accept = 0.0;
  double s2_tau2_additive_accept_batch = 0.0;
  double s2_tau2_additive_tune = 1.0;
  double mu_accept = 0.0;
  double mu_accept_batch = 0.0;
  arma::mat mu_batch(50, d, arma::fill::zeros);
  arma::mat Sigma_mu_tune(d, d, arma::fill::eye);
  arma::mat Sigma_mu_tune_chol = chol(Sigma_mu_tune);
  double tau2_accept = 0.0;
  double tau2_accept_batch = 0.0;
  arma::mat tau2_batch(50, d, arma::fill::zeros);
  arma::mat Sigma_tau2_tune(d, d, arma::fill::eye);
  arma::mat Sigma_tau2_tune_chol = chol(Sigma_tau2_tune);
  if (Sigma_reference_category) {
    Sigma_tau2_tune.shed_col(d-1);
    Sigma_tau2_tune.shed_row(d-1);
    Sigma_tau2_tune_chol.shed_col(d-1);
    Sigma_tau2_tune_chol.shed_row(d-1);
    tau2_batch.shed_col(d-1);
  }
  double tau2_additive_accept = 0.0;
  double tau2_additive_accept_batch = 0.0;
  arma::mat tau2_additive_batch(50, d, arma::fill::zeros);
  arma::mat Sigma_tau2_additive_tune(d, d, arma::fill::eye);
  arma::mat Sigma_tau2_additive_tune_chol = chol(Sigma_tau2_additive_tune);
  double xi_additive_accept = 0.0;
  double xi_additive_accept_batch = 0.0;
  arma::mat xi_additive_batch(50, B, arma::fill::zeros);
  arma::mat Sigma_xi_additive_tune(B, B, arma::fill::eye);
  arma::mat Sigma_xi_additive_tune_chol = chol(Sigma_xi_additive_tune);
  arma::vec eta_star_accept(d, arma::fill::zeros);
  arma::vec eta_star_accept_batch(d, arma::fill::zeros);
  arma::cube eta_star_batch(50, N_knots, d, arma::fill::zeros);
  arma::cube Sigma_eta_star_tune(N_knots, N_knots, d, arma::fill::zeros);
  arma::cube Sigma_eta_star_tune_chol(N_knots, N_knots, d, arma::fill::zeros);
  for(int j=0; j<d; j++) {
    Sigma_eta_star_tune.slice(j).eye();
    Sigma_eta_star_tune_chol.slice(j) = chol(Sigma_eta_star_tune.slice(j));
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
    // sample mu
    //

    if (sample_mu) {
      // sample using MH
      arma::vec mu_star = mvrnormArmaVecChol(mu, lambda_mu_tune * Sigma_mu_tune_chol);
      arma::mat mu_mat_star(N, d);
      for (int i=0; i<N; i++) {
        mu_mat_star.row(i) = mu_star.t();
      }
      arma::mat alpha_star = exp(mu_mat_star + zeta + epsilon);
      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        // dMVNChol(mu_star, mu_mu, Sigma_mu_chol);
        dMVN(mu_star, mu_mu, Sigma_mu_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        // dMVNChol(mu, mu_mu, Sigma_mu_chol);
        dMVN(mu, mu_mu, Sigma_mu_chol);
      double mh = exp(mh1-mh2);
      if (mh > R::runif(0.0, 1.0)) {
        mu = mu_star;
        mu_mat = mu_mat_star;
        alpha = alpha_star;
        mu_accept_batch += 1.0 / 50;
      }
      // update tuning
      mu_batch.row(k % 50) = mu.t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, mu_accept_batch, lambda_mu_tune, mu_batch,
                       Sigma_mu_tune, Sigma_mu_tune_chol);
      }
    }

    //
    // sample phi
    //

    if (sample_phi) {
      double phi_star = phi + R::rnorm(0.0, phi_tune);
      if (phi_star > phi_L && phi_star < phi_U) {
        arma::mat C_star = exp(- D_knots / phi_star);
        arma::mat C_chol_star = chol(C_star);
        arma::mat C_inv_star = inv_sympd(C_star);
        arma::mat c_star = exp(- D / phi_star);
        arma::mat Z_star = c_star * C_inv_star;
        arma::mat zeta_star = Z_star * eta_star * R_tau;
        arma::mat alpha_star = exp(mu_mat + zeta_star + epsilon);
        double mh1 = 0.0 + // uniform prior
          LL_DM(alpha_star, Y, N, d, count);
        double mh2 = 0.0 + // uniform prior
          LL_DM(alpha, Y, N, d, count);
        for (int j=0; j<d; j++) {
          mh1 += dMVN(eta_star.col(j), zero_knots, C_chol_star, true);
          mh2 += dMVN(eta_star.col(j), zero_knots, C_chol, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          phi = phi_star;
          C = C_star;
          C_chol = C_chol_star;
          C_inv = C_inv_star;
          c = c_star;
          Z = Z_star;
          zeta = zeta_star;
          alpha = alpha_star;
          phi_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, phi_accept_batch, phi_tune);
      }
    }

    //
    // sample eta_star
    //

    if (sample_eta_star) {
      if (sample_eta_star_mh) {
        // Metroplois-Hastings
        for (int j=0; j<d; j++) {
          arma::mat eta_star_star = eta_star;
          eta_star_star.col(j) += mvrnormArmaVecChol(zero_knots,
                            lambda_eta_star_tune(j) * Sigma_eta_star_tune_chol.slice(j));
          arma::mat zeta_star = Z * eta_star_star * R_tau;
          arma::mat alpha_star = exp(mu_mat + zeta_star + epsilon);
          double mh1 = dMVN(eta_star_star.col(j), zero_knots, C_chol, true) +
            // double mh1 = dMVNChol(eta_star_star.col(j), zero_knots, C_chol, true) -
            LL_DM(alpha_star, Y, N, d, count);
          double mh2 = dMVN(eta_star.col(j), zero_knots, C_chol, true) +
            // double mh2 = dMVNChol(eta_star.col(j), zero_knots, C_chol, true) -
            LL_DM(alpha, Y, N, d, count);
          double mh = exp(mh1-mh2);
          if (mh > R::runif(0.0, 1.0)) {
            eta_star = eta_star_star;
            zeta = zeta_star;
            alpha = alpha_star;
            eta_star_accept_batch(j) += 1.0 / 50.0;
          }
        }
        // update tuning
        eta_star_batch.subcube(k % 50, 0, 0, k % 50, N_knots-1, d-1) = eta_star;
        if ((k+1) % 50 == 0){
          updateTuningMVMat(k, eta_star_accept_batch, lambda_eta_star_tune,
                            eta_star_batch, Sigma_eta_star_tune,
                            Sigma_eta_star_tune_chol);
        }
      } else {
        // elliptical slice sampler
        for (int j=0; j<d; j++) {
          arma::vec eta_star_prior = mvrnormArmaVecChol(zero_knots, C_chol);
          Rcpp::List ess_eta_star_out = ess_multiplicative(eta_star, eta_star_prior, alpha,
                                            mu_mat, zeta, R_tau, epsilon, Z, Y, N, d,
                                            j, count, file_name, n_chain);
          eta_star = as<mat>(ess_eta_star_out["eta_star"]);
          zeta = as<mat>(ess_eta_star_out["zeta"]);
          alpha = as<mat>(ess_eta_star_out["alpha"]);
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
        epsilon_star.col(i) += mvrnormArmaVecChol(zero_knots,
                         lambda_epsilon_tune(i) * Sigma_epsilon_tune_chol.slice(i));;
          arma::mat alpha_star = exp(mu_mat + zeta + epsilon_star);
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
      if (Sigma_reference_category) {
        // last element is fixed at one
        log_tau2_star.subvec(0, d-2) = mvrnormArmaVecChol(log(tau2.subvec(0, d-2)),
                             lambda_tau2_tune * Sigma_tau2_tune_chol);
      } else {
        log_tau2_star = mvrnormArmaVecChol(log(tau2),
                                           lambda_tau2_tune * Sigma_tau2_tune_chol);
      }
      arma::vec tau2_star = exp(log_tau2_star);
      if (all(tau2_star > 0.0)) {
        arma::vec tau_star = sqrt(tau2_star);
        arma::mat R_tau_star = R * diagmat(tau_star);
        arma::mat zeta_star = Z * eta_star * R_tau_star;
        arma::mat alpha_star = exp(mu_mat + zeta_star + epsilon);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) + sum(log_tau2_star);  // jacobian of log-scale proposal
        double mh2 = LL_DM(alpha, Y, N, d, count) + sum(log(tau2));      // jacobian of log-scale proposal
        for (int j=0; j<d; j++) {
          mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2 = tau2_star;
          tau = tau_star;
          R_tau = R_tau_star;
          zeta = zeta_star;
          alpha = alpha_star;
          tau2_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if (Sigma_reference_category) {
        tau2_batch.row(k % 50) = log(tau2.subvec(0, d-2)).t();
      } else {
        tau2_batch.row(k % 50) = log(tau2).t();
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuningMV(k, tau2_accept_batch, lambda_tau2_tune, tau2_batch,
                       Sigma_tau2_tune, Sigma_tau2_tune_chol);
      }
    }

    //
    // sample lambda_tau2
    //

    for (int j=0; j<d; j++) {
      lambda_tau2(j) = R::rgamma(1.0, 1.0 / (s2_tau2 + tau2(j)));
    }

    //
    // sample s2_tau2
    //

      double s2_tau2_star = s2_tau2 + R::rnorm(0, s2_tau2_tune);
      if (s2_tau2_star > 0 && s2_tau2_star < A_s2) {
        double mh1 = 0.0;
        double mh2 = 0.0;
        for (int j=0; j<d; j++){
          mh1 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2_star, true);
          mh2 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          s2_tau2 = s2_tau2_star;
          s2_tau2_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, s2_tau2_accept_batch, s2_tau2_tune);
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
    // sample mu
    //

    if (sample_mu) {
      // sample using MH
      arma::vec mu_star = mvrnormArmaVecChol(mu, lambda_mu_tune * Sigma_mu_tune_chol);
      arma::mat mu_mat_star(N, d);
      for (int i=0; i<N; i++) {
        mu_mat_star.row(i) = mu_star.t();
      }
      arma::mat alpha_star = exp(mu_mat_star + zeta + epsilon);
      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        // dMVNChol(mu_star, mu_mu, Sigma_mu_chol);
        dMVN(mu_star, mu_mu, Sigma_mu_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        // dMVNChol(mu, mu_mu, Sigma_mu_chol);
        dMVN(mu, mu_mu, Sigma_mu_chol);
      double mh = exp(mh1-mh2);
      if (mh > R::runif(0.0, 1.0)) {
        mu = mu_star;
        mu_mat = mu_mat_star;
        alpha = alpha_star;
        mu_accept += 1.0 / n_mcmc;
      }
    }

    //
    // sample phi
    //

    if (sample_phi) {
      double phi_star = phi + R::rnorm(0.0, phi_tune);
      if (phi_star > phi_L && phi_star < phi_U) {
        arma::mat C_star = exp(- D_knots / phi_star);
        arma::mat C_chol_star = chol(C_star);
        arma::mat C_inv_star = inv_sympd(C_star);
        arma::mat c_star = exp(- D / phi_star);
        arma::mat Z_star = c_star * C_inv_star;
        arma::mat zeta_star = Z_star * eta_star * R_tau;
        arma::mat alpha_star = exp(mu_mat + zeta_star + epsilon);
        double mh1 = 0.0 + // uniform prior
          LL_DM(alpha_star, Y, N, d, count);
        double mh2 = 0.0 + // uniform prior
          LL_DM(alpha, Y, N, d, count);
        for (int j=0; j<d; j++) {
          mh1 += dMVN(eta_star.col(j), zero_knots, C_chol_star, true);
          mh2 += dMVN(eta_star.col(j), zero_knots, C_chol, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          phi = phi_star;
          C = C_star;
          C_chol = C_chol_star;
          C_inv = C_inv_star;
          c = c_star;
          Z = Z_star;
          zeta = zeta_star;
          alpha = alpha_star;
          phi_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample eta_star
    //

    if (sample_eta_star) {
      if (sample_eta_star_mh) {
        // Metroplois-Hastings
        for (int j=0; j<d; j++) {
          arma::mat eta_star_star = eta_star;
          eta_star_star.col(j) += mvrnormArmaVecChol(zero_knots,
                            lambda_eta_star_tune(j) * Sigma_eta_star_tune_chol.slice(j));
          arma::mat zeta_star = Z * eta_star_star * R_tau;
          arma::mat alpha_star = exp(mu_mat + zeta_star + epsilon);
          double mh1 = dMVN(eta_star_star.col(j), zero_knots, C_chol, true) +
            // double mh1 = dMVNChol(eta_star_star.col(j), zero_knots, C_chol, true) -
            LL_DM(alpha_star, Y, N, d, count);
          double mh2 = dMVN(eta_star.col(j), zero_knots, C_chol, true) +
            // double mh2 = dMVNChol(eta_star.col(j), zero_knots, C_chol, true) -
            LL_DM(alpha, Y, N, d, count);
          double mh = exp(mh1-mh2);
          if (mh > R::runif(0.0, 1.0)) {
            eta_star = eta_star_star;
            zeta = zeta_star;
            alpha = alpha_star;
            eta_star_accept(j) += 1.0 / n_mcmc;
          }
        }
      } else {
        // elliptical slice sampler
        for (int j=0; j<d; j++) {
          arma::vec eta_star_prior = mvrnormArmaVecChol(zero_knots, C_chol);
          Rcpp::List ess_eta_star_out = ess_multiplicative(eta_star, eta_star_prior, alpha,
                                            mu_mat, zeta, R_tau, epsilon, Z, Y, N, d,
                                            j, count, file_name, n_chain);
          eta_star = as<mat>(ess_eta_star_out["eta_star"]);
          zeta = as<mat>(ess_eta_star_out["zeta"]);
          alpha = as<mat>(ess_eta_star_out["alpha"]);
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
        epsilon_star.col(i) += mvrnormArmaVecChol(zero_knots,
                  lambda_epsilon_tune(i) * Sigma_epsilon_tune_chol.slice(i));;
        arma::mat alpha_star = exp(mu_mat + zeta + epsilon_star);
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
      if (Sigma_reference_category) {
        // last element is fixed at one
        log_tau2_star.subvec(0, d-2) = mvrnormArmaVecChol(log(tau2.subvec(0, d-2)),
                             lambda_tau2_tune * Sigma_tau2_tune_chol);
      } else {
        log_tau2_star = mvrnormArmaVecChol(log(tau2),
                                           lambda_tau2_tune * Sigma_tau2_tune_chol);
      }
      arma::vec tau2_star = exp(log_tau2_star);
      if (all(tau2_star > 0.0)) {
        arma::vec tau_star = sqrt(tau2_star);
        arma::mat R_tau_star = R * diagmat(tau_star);
        arma::mat zeta_star = Z * eta_star * R_tau_star;
        arma::mat alpha_star = exp(mu_mat + zeta_star + epsilon);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) + sum(log_tau2_star);  // jacobian of log-scale proposal
        double mh2 = LL_DM(alpha, Y, N, d, count) + sum(log(tau2));      // jacobian of log-scale proposal
        for (int j=0; j<d; j++) {
          mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2 = tau2_star;
          tau = tau_star;
          R_tau = R_tau_star;
          zeta = zeta_star;
          alpha = alpha_star;
          tau2_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample lambda_tau2
    //

    for (int j=0; j<d; j++) {
      lambda_tau2(j) = R::rgamma(1.0, 1.0 / (s2_tau2 + tau2(j)));
    }

    //
    // sample s2_tau2
    //

      double s2_tau2_star = s2_tau2 + R::rnorm(0, s2_tau2_tune);
      if (s2_tau2_star > 0 && s2_tau2_star < A_s2) {
        double mh1 = 0.0;
        double mh2 = 0.0;
        for (int j=0; j<d; j++){
          mh1 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2_star, true);
          mh2 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          s2_tau2 = s2_tau2_star;
          s2_tau2_accept += 1.0 / n_mcmc;
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
      zeta_save.subcube(span(save_idx), span(), span()) = zeta;
      epsilon_save.subcube(span(save_idx), span(), span()) = epsilon;
      phi_save(save_idx) = phi;
      mu_save.row(save_idx) = mu.t();
      tau2_save.row(save_idx) = tau2.t();
      tau2_additive_save.row(save_idx) = tau2_additive.t();
      eta_star_save.subcube(span(save_idx), span(), span()) = eta_star;
      R_save.subcube(span(save_idx), span(), span()) = R;
      R_tau_save.subcube(span(save_idx), span(), span()) = R_tau;
      R_additive_save.subcube(span(save_idx), span(), span()) = R_additive;
      R_tau_additive_save.subcube(span(save_idx), span(), span()) = R_tau_additive;
      xi_additive_save.row(save_idx) = xi_additive.t();
    }
    // end of MCMC fitting loop
  }

  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  if (sample_mu) {
    file_out << "Average acceptance rate for mu  = " << mean(mu_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_eta_star) {
    if (sample_eta_star_mh) {
      file_out << "Average acceptance rate for eta_star  = " << mean(eta_star_accept) <<
        " for chain " << n_chain << "\n";
    }
  }
  if (sample_epsilon) {
      file_out << "Average acceptance rate for epsilon  = " << mean(epsilon_accept) <<
        " for chain " << n_chain << "\n";
  }
  if (sample_phi) {
    file_out << "Average acceptance rate for phi  = " << mean(phi_accept) <<
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
    file_out << "Average acceptance rate for tau2  = " << mean(tau2_additive_accept) <<
      " for chain " << n_chain << "\n";
  }

    // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    _["mu"] = mu_save,
    _["eta_star"] = eta_star_save,
    _["zeta"] = zeta_save,
    _["alpha"] = alpha_save,
    _["epsilon"] = epsilon_save,
    _["phi"] = phi_save,
    _["tau2"] = tau2_save,
    _["R"] = R_save,
    _["R_tau"] = R_tau_save,
    _["tau2_additive"] = tau2_additive_save,
    _["R_additive"] = R_additive_save,
    _["R_tau_additive"] = R_tau_additive_save,
    _["xi_additive"] = xi_additive_save);
}
