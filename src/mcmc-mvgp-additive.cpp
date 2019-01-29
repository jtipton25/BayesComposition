// #define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include "BayesComp.h"
// // [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Multivariate Gaussian Process for Inverse Inference
// Author: John Tipton
//
// Created 11.29.2016
// Last updated 06.09.2017

//
// Functions for sampling
//

///////////////////////////////////////////////////////////////////////////////
//// Elliptical Slice Sampler for predictive process random effect eta_star ///
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess_eta_star (const arma::mat& eta_star_current,
                         const arma::vec& eta_star_prior,
                         const arma::mat& y_current,
                         const arma::vec& mu_current,
                         const arma::mat& zeta_current,
                         const arma::mat& R_tau_current,
                         const arma::mat& Z_current,
                         const arma::mat& R_tau_epsilon_current,
                         const int& N,
                         const int& d,
                         const int& j,
                         const std::string& file_name,
                         const int& n_chain) {
  // eta_star_current is the current value of the joint multivariate predictive process
  // prior_sample is a sample from the prior joing multivariate predictive process
  // R_tau is the current value of the Cholskey decomposition for  predictive process linear interpolator
  // Z_current is the current predictive process linear

  // calculate log likelihood of current value
  double current_log_like = 0.0;

  for (int i=0; i<N; i++) {
    current_log_like += dMVN(
      y_current.row(i).t(),
      mu_current + zeta_current.row(i).t(),
      R_tau_epsilon_current,
      true);
  }

  double hh = log(R::runif(0.0, 1.0)) + current_log_like;

  // Setup a bracket and pick a first proposal
  // Bracket whole ellipse with both edges at first proposed point
  double phi_angle = R::runif(0.0, 1.0) * 2.0 * arma::datum::pi;
  double phi_angle_min = phi_angle - 2.0 * arma::datum::pi;
  double phi_angle_max = phi_angle;

  // set up save variables
  arma::mat eta_star_ess = eta_star_current;
  arma::mat eta_star_proposal = eta_star_current;
  arma::mat zeta_ess = zeta_current;
  bool test = true;

  // Slice sampling loop
  while (test) {
    // compute proposal for angle difference and check to see if it is on the slice
    arma::vec eta_star_proposal_col =
      eta_star_current.col(j) * cos(phi_angle) +
      eta_star_prior * sin(phi_angle);
    eta_star_proposal.col(j) = eta_star_proposal_col;
    arma::mat zeta_proposal = Z_current * eta_star_proposal * R_tau_current;

    // calculate log likelihood of proposed value
    double proposal_log_like = 0.0;
    for (int i=0; i<N; i++) {
      proposal_log_like += dMVN(
        y_current.row(i).t(),
        mu_current + zeta_proposal.row(i).t(),
        R_tau_epsilon_current,
        true);
    }

    if (proposal_log_like > hh) {
      // proposal is on the slice
      eta_star_ess = eta_star_proposal;
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
      file_out << "Bug - ESS for X shrunk to current position on chain " << n_chain << "\n";
      // close output file
      file_out.close();
    }
    // Propose new angle difference
    phi_angle = R::runif(0.0, 1.0) * (phi_angle_max - phi_angle_min) + phi_angle_min;
  }
  return(Rcpp::List::create(
      _["eta_star"] = eta_star_ess,
      _["zeta"] = zeta_ess));
}

// [[Rcpp::export]]
List mcmcRcppMVGP (const arma::mat& Y,
                   const arma::vec& X,
                   List params,
                   int n_chain = 1,
                   std::string file_name = "sim-fit") {
  // arma::mat& R, arma::vec& tau2, double& phi, double& sigma2,
  // arma::mat& eta_star,

  // Load parameters
  int n_adapt = as<int>(params["n_adapt"]);
  int n_mcmc = as<int>(params["n_mcmc"]);
  int n_thin = as<int>(params["n_thin"]);

  // set up dimensions
  double N = Y.n_rows;
  double d = Y.n_cols;
  double B = Rf_choose(d, 2);
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec ones_B(B, arma::fill::ones);

  // default normal prior for overall mean mu
  arma::vec mu_mu(d, arma::fill::zeros);
  if (params.containsElementNamed("mu_mu")) {
    mu_mu = as<vec>(params["mu_mu"]);
  }
  // default prior for overall mean mu
  double s2_mu = 100.0;
  if (params.containsElementNamed("s2_mu")) {
    s2_mu = as<double>(params["s2_mu"]);
  }
  arma::mat Sigma_mu(d, d, arma::fill::eye);
  Sigma_mu *= s2_mu;
  arma::mat Sigma_mu_inv = inv_sympd(Sigma_mu);

  double s_mu = sqrt(s2_mu);
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
  // default half cauchy scale for generalized Wishart Gaussian Process nugget
  // double s2_sigma2 = 5.0;
  // if (params.containsElementNamed("s2_sigma2")) {
  //   s2_sigma2 = as<double>(params["s2_sigma2"]);
  // }
  // default half cauchy scale for Covariance diagonal variance tau2
  double A_s2 = 25.0;
  if (params.containsElementNamed("A_s2")) {
    A_s2 = as<double>(params["A_s2"]);
  }
  // default half cauchy scale for generalized Wishart Gaussian Process sill
  double s2_tau2 = 1.0;
  if (params.containsElementNamed("s2_tau2")) {
    s2_tau2 = as<double>(params["s2_tau2"]);
  }
  // default xi LKJ concentation parameter of 1
  double eta = 4.0;
  if (params.containsElementNamed("eta")) {
    eta = as<double>(params["eta"]);
  }

  // default to message output every 5000 iterations
  int message = 5000;
  if (params.containsElementNamed("message")) {
    message = as<int>(params["message"]);
  }
  // default phi tuning parameter standard deviation of 0.25
  double phi_tune = 0.25;
  if (params.containsElementNamed("phi_tune")) {
    phi_tune = as<double>(params["phi_tune"]);
  }
  // // default sigma2 tuning parameter standard deviation of 0.25
  // double sigma2_tune = 0.25;
  // if (params.containsElementNamed("sigma2_tune")) {
  //   sigma2_tune = as<double>(params["sigma2_tune"]);
  // }
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

  // // default lambda_eta_star tuning parameter standard deviation of 0.25
  // double lambda_eta_star_tune = 0.25;
  // if (params.containsElementNamed("lambda_eta_star_tune")) {
  //   lambda_eta_star_tune = as<double>(params["lambda_eta_star_tune"]);
  // }
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
  // default tau2 tuning parameter
  double lambda_tau2_epsilon_tune = 0.25;
  if (params.containsElementNamed("lambda_tau2_epsilon_tune")) {
    lambda_tau2_epsilon_tune = as<double>(params["lambda_tau2_epsilon_tune"]);
  }
  // default xi tuning parameter
  double lambda_xi_epsilon_tune = 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_xi_epsilon_tune")) {
    lambda_xi_epsilon_tune = as<double>(params["lambda_xi_epsilon_tune"]);
  }

  // predictive process knots
  arma::vec X_knots = as<vec>(params["X_knots"]);
  double N_knots = X_knots.n_elem;

  // default correlation functions
  std::string corr_function="exponential";
  if (params.containsElementNamed("corr_function")) {
    corr_function = as<std::string>(params["corr_function"]);
  }

  arma::mat D = makeDistARMA(X, X_knots);
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
  bool sample_mu_mh = false;
  if (params.containsElementNamed("sample_mu_mh")) {
    sample_mu_mh = as<bool>(params["sample_mu_mh"]);
  }
  // arma::mat mu_mat(N, d);
  // for (int i=0; i<N; i++) {
  //   mu_mat.row(i) = mu.t();
  // }

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
  // Obeservation error parameters
  //

  arma::vec lambda_tau2_epsilon(d);
  arma::vec tau2_epsilon(d);
  for (int j=0; j<d; j++) {
    tau2_epsilon(j) = std::max(std::min(R::rgamma(0.5, 1.0), 5.0), 1.0);
  }
  arma::vec tau_epsilon = sqrt(tau2_epsilon);
  if (params.containsElementNamed("tau2_epsilon")) {
    tau2_epsilon = as<vec>(params["tau2_epsilon"]);
    tau_epsilon= sqrt(tau2_epsilon);
  }
  bool sample_tau2_epsilon = true;
  if (params.containsElementNamed("sample_tau2_epsilon")) {
    sample_tau2_epsilon = as<bool>(params["sample_tau2_epsilon"]);
  }

  //
  // Default LKJ hyperparameter xi_epsilon for observation error covariance
  //

  arma::vec eta_vec(B);
  int idx_eta = 0;
  for (int j=0; j<(d-1); j++) {
    for (int k=0; k<(d-j-1); k++) {
      eta_vec(idx_eta+k) = eta + (d - 2.0 - j) / 2.0;
    }
    idx_eta += d-j-1;
  }
  arma::vec xi_epsilon(B);
  for (int b=0; b<B; b++) {
    xi_epsilon(b) = 2.0 * R::rbeta(eta_vec(b), eta_vec(b)) - 1.0;
  }
  arma::vec xi_epsilon_tilde(B);
  if (params.containsElementNamed("xi_epsilon")) {
    xi_epsilon = as<vec>(params["xi_epsilon"]);
  }
  for (int b=0; b<B; b++) {
    xi_epsilon_tilde(b) = 0.5 * (xi_epsilon(b) + 1.0);
  }
  bool sample_xi_epsilon = true;
  if (params.containsElementNamed("sample_xi_epsilon")) {
    sample_xi_epsilon = as<bool>(params["sample_xi_epsilon"]);
  }

  Rcpp::List R_epsilon_out = makeRLKJ(xi_epsilon, d, true, true);
  double log_jacobian_epsilon = as<double>(R_epsilon_out["log_jacobian"]);
  arma::mat R_epsilon = as<mat>(R_epsilon_out["R"]);
  arma::mat R_tau_epsilon = R_epsilon * diagmat(tau_epsilon);
  //
  // Gaussian process sill parameter tau2 and hyperprior lambda_tau2
  //

  arma::vec lambda_tau2(d);
  arma::vec tau2(d);
  for (int j=0; j<d; j++) {
    // lambda_tau2(j) = R::rgamma(0.5, 1.0 / s2_tau2);
    // tau2(j) = std::max(std::min(R::rgamma(0.5, 1.0 / lambda_tau2(j)), 5.0), 1.0);
    tau2(j) = std::max(std::min(R::rgamma(0.5, 1.0), 5.0), 1.0);
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

  //
  // Construct Gaussian Process Correlation matrices
  //

  arma::mat C = exp(- D_knots / phi);
  arma::mat C_chol = chol(C);
  arma::mat C_inv = inv_sympd(C);
  arma::mat c = exp( - D / phi);
  arma::mat Z = c * C_inv;

  // Initialize constant vectors

  arma::vec zero_knots(N_knots, arma::fill::zeros);
  arma::vec zero_knots_d(N_knots*d, arma::fill::zeros);

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

  //
  // Default LKJ hyperparameter xi
  //

  // arma::vec eta_vec(B);
  // int idx_eta = 0;
  // for (int j=0; j<(d-1); j++) {
  //   for (int k=0; k<(d-j-1); k++) {
  //     eta_vec(idx_eta+k) = eta + (d - 2.0 - j) / 2.0;
  //   }
  //   idx_eta += d-j-1;
  // }
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
  arma::mat zeta = Z * eta_star * R_tau;

  // setup save variables
  int n_save = n_mcmc / n_thin;
  arma::mat mu_save(n_save, d, arma::fill::zeros);
  arma::cube zeta_save(n_save, N, d, arma::fill::zeros);
  arma::cube eta_star_save(n_save, N_knots, d, arma::fill::zeros);
  arma::cube Omega_save(n_save, d, d, arma::fill::zeros);
  arma::cube C_save(n_save, N_knots, N_knots, arma::fill::zeros);
  arma::cube c_save(n_save, N, N_knots, arma::fill::zeros);
  arma::cube C_inv_save(n_save, N_knots, N_knots, arma::fill::zeros);
  arma::cube Z_save(n_save, N, N_knots, arma::fill::zeros);
  arma::cube R_save(n_save, d, d, arma::fill::zeros);
  arma::cube R_tau_save(n_save, d, d, arma::fill::zeros);
  arma::mat tau2_save(n_save, d, arma::fill::zeros);

  arma::cube R_epsilon_save(n_save, d, d, arma::fill::zeros);
  arma::cube R_tau_epsilon_save(n_save, d, d, arma::fill::zeros);
  arma::mat tau2_epsilon_save(n_save, d, arma::fill::zeros);
  arma::mat xi_epsilon_save(n_save, B, arma::fill::zeros);

  // arma::mat lambda_tau2_save(n_save, d, arma::fill::zeros);
  // arma::vec s2_tau2_save(n_save, arma::fill::zeros);
  arma::vec phi_save(n_save, arma::fill::zeros);
  arma::mat xi_save(n_save, B, arma::fill::zeros);

  // initialize tuning
  double phi_accept = 0.0;
  double phi_accept_batch = 0.0;
  // double s2_tau2_accept = 0.0;
  // double s2_tau2_accept_batch = 0.0;
  // double s2_tau2_tune = 1.0;
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

  double tau2_epsilon_accept = 0.0;
  double tau2_epsilon_accept_batch = 0.0;
  arma::mat tau2_epsilon_batch(50, d, arma::fill::zeros);
  arma::mat Sigma_tau2_epsilon_tune(d, d, arma::fill::eye);
  arma::mat Sigma_tau2_epsilon_tune_chol = chol(Sigma_tau2_epsilon_tune);

  double xi_accept = 0.0;
  double xi_accept_batch = 0.0;
  arma::mat xi_batch(50, B, arma::fill::zeros);
  arma::mat Sigma_xi_tune(B, B, arma::fill::eye);
  arma::mat Sigma_xi_tune_chol = chol(Sigma_xi_tune);

  double xi_epsilon_accept = 0.0;
  double xi_epsilon_accept_batch = 0.0;
  arma::mat xi_epsilon_batch(50, B, arma::fill::zeros);
  arma::mat Sigma_xi_epsilon_tune(B, B, arma::fill::eye);
  arma::mat Sigma_xi_epsilon_tune_chol = chol(Sigma_xi_epsilon_tune);
  arma::vec eta_star_accept(d, arma::fill::zeros);
  arma::vec eta_star_accept_batch(d, arma::fill::zeros);
  arma::cube eta_star_batch(50, N_knots, d, arma::fill::zeros);
  arma::cube Sigma_eta_star_tune(N_knots, N_knots, d, arma::fill::zeros);
  arma::cube Sigma_eta_star_tune_chol(N_knots, N_knots, d, arma::fill::zeros);
  for(int j=0; j<d; j++) {
    Sigma_eta_star_tune.slice(j).eye();
    Sigma_eta_star_tune_chol.slice(j) = chol(Sigma_eta_star_tune.slice(j));
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
      Rprintf("MCMC Adaptive Iteration %d \n", k+1);
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
      if (sample_mu_mh) {
        // sample using MH
        arma::vec mu_star = mvrnormArmaVecChol(mu, lambda_mu_tune * Sigma_mu_tune_chol);
        // arma::mat mu_mat_star(N, d);
        // for (int i=0; i<N; i++) {
        //   mu_mat_star.row(i) = mu_star.t();
        // }
        double mh1 = 0.0;
        double mh2 = 0.0;
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu_star + zeta.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += R::dnorm(mu_star(j), mu_mu(j), s_mu, true);
          mh2 += R::dnorm(mu(j), mu_mu(j), s_mu, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          mu = mu_star;
          // mu_mat = mu_mat_star;
          mu_accept_batch += 1.0 / 50;
        }
        mu_batch.row(k % 50) = mu.t();
        // update tuning
        if ((k+1) % 50 == 0){
          updateTuningMV(k, mu_accept_batch, lambda_mu_tune, mu_batch,
                         Sigma_mu_tune, Sigma_mu_tune_chol);
        }
      } else {
        arma::mat Sigma_epsilon = R_tau_epsilon.t() * R_tau_epsilon;
        arma::mat Sigma_epsilon_inv = inv_sympd(Sigma_epsilon);
        // Fix this later if needed...
        // // sample mu using Gibbs
        arma::mat A = N * Sigma_epsilon_inv + Sigma_mu_inv;
        arma::vec b = colSums((Y - zeta) * Sigma_epsilon_inv) + Sigma_mu_inv * mu_mu;
        mu = rMVNArma(A, b);
        // for (int i=0; i<N; i++) {
        //   mu_mat.row(i) = mu.t();
        // }
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
        double mh1 = 0.0;  // uniform prior
        double mh2 = 0.0;  // uniform prior
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += dMVN(eta_star.col(j), zero_knots, C_chol_star, true);
          mh2 += dMVN(eta_star.col(j), zero_knots, C_chol, true);
          // mh1 += dMVNChol(eta_star.col(j), zero_knots, C_chol_star, true);
          // mh2 += dMVNChol(eta_star.col(j), zero_knots, C_chol, true);
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
          phi_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, phi_accept_batch, phi_tune);
      }
    }

    //
    // sample eta_star - MH
    //

    if (sample_eta_star) {
      if (sample_eta_star_mh) {
        for (int j=0; j<d; j++) {
          arma::mat eta_star_star = eta_star;
          eta_star_star.col(j) +=
            mvrnormArmaVecChol(zero_knots,
                               lambda_eta_star_tune(j) * Sigma_eta_star_tune_chol.slice(j));
          arma::mat zeta_star = Z * eta_star_star * R_tau;
          // double mh1 = dMVNChol(eta_star_star.col(j), zero_knots, C_chol, true) -
          double mh1 = dMVN(eta_star_star.col(j), zero_knots, C_chol, true);
          double mh2 = dMVN(eta_star.col(j), zero_knots, C_chol, true);
          for (int i=0; i<N; i++) {
            mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
            mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
          }
          double mh = exp(mh1-mh2);
          if (mh > R::runif(0.0, 1.0)) {
            eta_star = eta_star_star;
            zeta = zeta_star;
            eta_star_accept_batch(j) += 1.0 / 50;
          }
        }
        // update tuning
        eta_star_batch.subcube(k % 50, 0, 0, k % 50, N_knots-1, d-1) = eta_star;
        // update tuning
        if ((k+1) % 50 == 0){
          updateTuningMVMat(k, eta_star_accept_batch, lambda_eta_star_tune,
                            eta_star_batch, Sigma_eta_star_tune,
                            Sigma_eta_star_tune_chol);
        }
      } else {
        // elliptical slice sampler
        for (int j=0; j<d; j++) {
          arma::vec eta_star_prior = mvrnormArmaVecChol(zero_knots, C_chol);
          Rcpp::List ess_eta_star_out = ess_eta_star(eta_star,  eta_star_prior,
                                                     Y, mu, zeta, R_tau, Z,
                                                     R_tau_epsilon, N, d, j,
                                                     file_name, n_chain);
          eta_star = as<mat>(ess_eta_star_out["eta_star"]);
          zeta = as<mat>(ess_eta_star_out["zeta"]);
        }
      }
    }


    //
    // sample tau2
    //

    if (sample_tau2) {
      arma::vec log_tau2_star = mvrnormArmaVecChol(log(tau2),
                                                   lambda_tau2_tune * Sigma_tau2_tune_chol);
      arma::vec tau2_star = exp(log_tau2_star);
      if (all(tau2_star > 0.0)) {
        arma::vec tau_star = sqrt(tau2_star);
        arma::mat R_tau_star = R * diagmat(tau_star);
        arma::mat zeta_star = Z * eta_star * R_tau_star;
        double mh1 = sum(log(tau2_star));      // jacobian of log-scale proposal
        double mh2 = sum(log(tau2));           // jacobian of log-scale proposal
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += d_half_cauchy(tau2_star(j), s2_tau2, true);
          mh2 += d_half_cauchy(tau2(j), s2_tau2, true);
          // mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          // mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2 = tau2_star;
          tau = tau_star;
          R_tau = R_tau_star;
          zeta = zeta_star;
          tau2_accept_batch += 1.0 / 50.0;
        }
      }

      tau2_batch.row(k % 50) = log(tau2).t();
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuningMV(k, tau2_accept_batch, lambda_tau2_tune, tau2_batch,
                       Sigma_tau2_tune, Sigma_tau2_tune_chol);
      }
    }

    // //
    // // sample lambda_tau2
    // //
    //
    // for (int j=0; j<d; j++) {
    //   lambda_tau2(j) = R::rgamma(1.0, 1.0 / (s2_tau2 + tau2(j)));
    // }
    //
    // //
    // // sample s2_tau2
    // //
    //
    //   double s2_tau2_star = s2_tau2 + R::rnorm(0.0, s2_tau2_tune);
    //   if (s2_tau2_star > 0.0 && s2_tau2_star < A_s2) {
    //     double mh1 = 0.0;
    //     double mh2 = 0.0;
    //     for (int j=0; j<d; j++){
    //       mh1 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2_star, true);
    //       mh2 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2, true);
    //     }
    //     double mh = exp(mh1-mh2);
    //     if (mh > R::runif(0.0, 1.0)) {
    //       s2_tau2 = s2_tau2_star;
    //       s2_tau2_accept_batch += 1.0 / 50.0;
    //     }
    //   }
    // // update tuning
    // if ((k+1) % 50 == 0){
    //   updateTuning(k, s2_tau2_accept_batch, s2_tau2_tune);
    // }

    //
    // sample xi - MH
    //

    if (sample_xi) {
      arma::vec logit_xi_tilde_star = mvrnormArmaVecChol(logit(xi_tilde),
                                                         lambda_xi_tune * Sigma_xi_tune_chol);
      arma::vec xi_tilde_star = expit(logit_xi_tilde_star);
      arma::vec xi_star = 2.0 * xi_tilde_star - 1.0;
      // arma::vec xi_star =  mvrnormArmaVecChol(xi, lambda_xi_tune * Sigma_xi_tune_chol);
      if (all(xi_star > -1.0) && all(xi_star < 1.0)) {
        Rcpp::List R_out = makeRLKJ(xi_star, d, true, true);
        arma::mat R_star = as<mat>(R_out["R"]);
        arma::mat R_tau_star = R_star * diagmat(tau);
        arma::mat zeta_star = Z * eta_star * R_tau_star;

        double log_jacobian_star = as<double>(R_out["log_jacobian"]);
        double mh1 = sum(log(xi_tilde_star) + log(ones_B - xi_tilde_star)); // Jacobian adjustment
        double mh2 = sum(log(xi_tilde) + log(ones_B - xi_tilde));          // Jacobian adjustment
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
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
          zeta = zeta_star;
          xi_accept_batch += 1.0 / 50.0;
        }
      }

      // xi_batch.row(k % 50) = xi.t();
      xi_batch.row(k % 50) = logit(xi_tilde).t();

      // update tuning
      if ((k+1) % 50 == 0){
        updateTuningMV(k, xi_accept_batch, lambda_xi_tune, xi_batch,
                       Sigma_xi_tune, Sigma_xi_tune_chol);
      }
    }

    //
    // sample tau2_epsilon
    //

    if (sample_tau2_epsilon) {
      arma::vec log_tau2_epsilon_star = mvrnormArmaVecChol(log(tau2_epsilon),
                                                           lambda_tau2_epsilon_tune * Sigma_tau2_epsilon_tune_chol);
      arma::vec tau2_epsilon_star = exp(log_tau2_epsilon_star);
      if (all(tau2_epsilon_star > 0.0)) {
        arma::vec tau_epsilon_star = sqrt(tau2_epsilon_star);
        arma::mat R_tau_epsilon_star = R_epsilon * diagmat(tau_epsilon_star);
        double mh1 = sum(log(tau2_epsilon_star));      // jacobian of log-scale proposal
        double mh2 = sum(log(tau2_epsilon));           // jacobian of log-scale proposal
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon_star, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += d_half_cauchy(tau2_epsilon_star(j), s2_tau2, true);
          mh2 += d_half_cauchy(tau2_epsilon(j), s2_tau2, true);
          // mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          // mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2_epsilon = tau2_epsilon_star;
          tau_epsilon = tau_epsilon_star;
          R_tau_epsilon = R_tau_epsilon_star;
          tau2_epsilon_accept_batch += 1.0 / 50.0;
        }
      }

      tau2_epsilon_batch.row(k % 50) = log(tau2_epsilon).t();
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuningMV(k, tau2_epsilon_accept_batch, lambda_tau2_epsilon_tune, tau2_epsilon_batch,
                       Sigma_tau2_epsilon_tune, Sigma_tau2_epsilon_tune_chol);
      }
    }


    //
    // sample xi_epsilon - MH
    //

    if (sample_xi_epsilon) {
      arma::vec logit_xi_epsilon_tilde_star = mvrnormArmaVecChol(logit(xi_epsilon_tilde),
                                                                 lambda_xi_epsilon_tune * Sigma_xi_epsilon_tune_chol);
      arma::vec xi_epsilon_tilde_star = expit(logit_xi_epsilon_tilde_star);
      arma::vec xi_epsilon_star = 2.0 * xi_epsilon_tilde_star - 1.0;
      // arma::vec xi_star =  mvrnormArmaVecChol(xi, lambda_xi_tune * Sigma_xi_tune_chol);
      if (all(xi_epsilon_star > -1.0) && all(xi_epsilon_star < 1.0)) {
        Rcpp::List R_epsilon_out = makeRLKJ(xi_epsilon_star, d, true, true);
        arma::mat R_epsilon_star = as<mat>(R_epsilon_out["R"]);
        arma::mat R_tau_epsilon_star = R_epsilon_star * diagmat(tau_epsilon);
        double log_jacobian_epsilon_star = as<double>(R_epsilon_out["log_jacobian"]);
        double mh1 = sum(log(xi_epsilon_tilde_star) + log(ones_B - xi_epsilon_tilde_star)); // Jacobian adjustment

        double mh2 = sum(log(xi_epsilon_tilde) + log(ones_B - xi_epsilon_tilde));          // Jacobian adjustment
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon_star, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int b=0; b<B; b++) {
          mh1 += R::dbeta(0.5 * (xi_epsilon_star(b) + 1.0), eta_vec(b), eta_vec(b), true);
          mh2 += R::dbeta(0.5 * (xi_epsilon(b) + 1.0), eta_vec(b), eta_vec(b), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          xi_epsilon_tilde = xi_epsilon_tilde_star;
          xi_epsilon = xi_epsilon_star;
          R_epsilon = R_epsilon_star;
          R_tau_epsilon = R_tau_epsilon_star;
          log_jacobian_epsilon = log_jacobian_epsilon_star;
          xi_epsilon_accept_batch += 1.0 / 50.0;
        }
      }
      // xi_batch.row(k % 50) = xi.t();
      xi_epsilon_batch.row(k % 50) = logit(xi_epsilon_tilde).t();

      // update tuning
      if ((k+1) % 50 == 0){
        updateTuningMV(k, xi_epsilon_accept_batch, lambda_xi_epsilon_tune, xi_epsilon_batch,
                       Sigma_xi_epsilon_tune, Sigma_xi_epsilon_tune_chol);
      }
    }


  }

  Rprintf("Starting MCMC fit for chain %d, running for %d iterations \n",
          n_chain, n_mcmc);
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  file_out << "Starting MCMC fit for chain " << n_chain <<
    ", running for " << n_mcmc << " iterations \n";
  // close output file
  file_out.close();

  // Start MCMC chain
  for (int k=0; k<n_mcmc; k++) {
    if ((k+1) % message == 0) {
      Rprintf("MCMC Fitting Iteration %d \n", k+1);
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
      if (sample_mu_mh) {
        // sample using MH
        arma::vec mu_star = mvrnormArmaVecChol(mu, lambda_mu_tune * Sigma_mu_tune_chol);
        // arma::mat mu_mat_star(N, d);
        // for (int i=0; i<N; i++) {
        //   mu_mat_star.row(i) = mu_star.t();
        // }
        double mh1 = 0.0;
        double mh2 = 0.0;
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu_star + zeta.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += R::dnorm(mu_star(j), mu_mu(j), s_mu, true);
          mh2 += R::dnorm(mu(j), mu_mu(j), s_mu, true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          mu = mu_star;
          // mu_mat = mu_mat_star;
          mu_accept += 1.0 / n_mcmc;
        }
      } else {
        arma::mat Sigma_epsilon = R_tau_epsilon.t() * R_tau_epsilon;
        arma::mat Sigma_epsilon_inv = inv_sympd(Sigma_epsilon);
        // Fix this later if needed...
        // // sample mu using Gibbs
        arma::mat A = N * Sigma_epsilon_inv + Sigma_mu_inv;
        arma::vec b = colSums((Y - zeta) * Sigma_epsilon_inv) + Sigma_mu_inv * mu_mu;
        mu = rMVNArma(A, b);
        // for (int i=0; i<N; i++) {
        //   mu_mat.row(i) = mu.t();
        // }
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
        double mh1 = 0.0;  // uniform prior
        double mh2 = 0.0;  // uniform prior
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += dMVN(eta_star.col(j), zero_knots, C_chol_star, true);
          mh2 += dMVN(eta_star.col(j), zero_knots, C_chol, true);
          // mh1 += dMVNChol(eta_star.col(j), zero_knots, C_chol_star, true);
          // mh2 += dMVNChol(eta_star.col(j), zero_knots, C_chol, true);
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
          phi_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample eta_star - MH
    //

    if (sample_eta_star) {
      if (sample_eta_star_mh) {
        for (int j=0; j<d; j++) {
          arma::mat eta_star_star = eta_star;
          eta_star_star.col(j) +=
            mvrnormArmaVecChol(zero_knots,
                               lambda_eta_star_tune(j) * Sigma_eta_star_tune_chol.slice(j));
          arma::mat zeta_star = Z * eta_star_star * R_tau;
          // double mh1 = dMVNChol(eta_star_star.col(j), zero_knots, C_chol, true) -
          double mh1 = dMVN(eta_star_star.col(j), zero_knots, C_chol, true);
          double mh2 = dMVN(eta_star.col(j), zero_knots, C_chol, true);
          for (int i=0; i<N; i++) {
            mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
            mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
          }
          double mh = exp(mh1-mh2);
          if (mh > R::runif(0.0, 1.0)) {
            eta_star = eta_star_star;
            zeta = zeta_star;
            eta_star_accept(j) += 1.0 / n_mcmc;
          }
        }
      } else {
        // elliptical slice sampler
        for (int j=0; j<d; j++) {
          arma::vec eta_star_prior = mvrnormArmaVecChol(zero_knots, C_chol);
          Rcpp::List ess_eta_star_out = ess_eta_star(eta_star,  eta_star_prior,
                                                     Y, mu, zeta, R_tau, Z,
                                                     R_tau_epsilon, N, d, j,
                                                     file_name, n_chain);
          eta_star = as<mat>(ess_eta_star_out["eta_star"]);
          zeta = as<mat>(ess_eta_star_out["zeta"]);
        }
      }
    }

    //
    // sample tau2
    //

    if (sample_tau2) {
      arma::vec log_tau2_star = mvrnormArmaVecChol(log(tau2),
                                                   lambda_tau2_tune * Sigma_tau2_tune_chol);
      arma::vec tau2_star = exp(log_tau2_star);
      if (all(tau2_star > 0.0)) {
        arma::vec tau_star = sqrt(tau2_star);
        arma::mat R_tau_star = R * diagmat(tau_star);
        arma::mat zeta_star = Z * eta_star * R_tau_star;
        double mh1 = sum(log(tau2_star));      // jacobian of log-scale proposal
        double mh2 = sum(log(tau2));           // jacobian of log-scale proposal
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += d_half_cauchy(tau2_star(j), s2_tau2, true);
          mh2 += d_half_cauchy(tau2(j), s2_tau2, true);
          // mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          // mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2 = tau2_star;
          tau = tau_star;
          R_tau = R_tau_star;
          zeta = zeta_star;
          tau2_accept += 1.0 / n_mcmc;
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
      // arma::vec xi_star =  mvrnormArmaVecChol(xi, lambda_xi_tune * Sigma_xi_tune_chol);
      if (all(xi_star > -1.0) && all(xi_star < 1.0)) {
        Rcpp::List R_out = makeRLKJ(xi_star, d, true, true);
        arma::mat R_star = as<mat>(R_out["R"]);
        arma::mat R_tau_star = R_star * diagmat(tau);
        arma::mat zeta_star = Z * eta_star * R_tau_star;

        double log_jacobian_star = as<double>(R_out["log_jacobian"]);
        double mh1 = sum(log(xi_tilde_star) + log(ones_B - xi_tilde_star)); // Jacobian adjustment
        double mh2 = sum(log(xi_tilde) + log(ones_B - xi_tilde));          // Jacobian adjustment
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta_star.row(i).t(), R_tau_epsilon, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
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
          zeta = zeta_star;
          log_jacobian = log_jacobian_star;
          xi_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample tau2_epsilon
    //

    if (sample_tau2_epsilon) {
      arma::vec log_tau2_epsilon_star = mvrnormArmaVecChol(log(tau2_epsilon),
                                                           lambda_tau2_epsilon_tune * Sigma_tau2_epsilon_tune_chol);
      arma::vec tau2_epsilon_star = exp(log_tau2_epsilon_star);
      if (all(tau2_epsilon_star > 0.0)) {
        arma::vec tau_epsilon_star = sqrt(tau2_epsilon_star);
        arma::mat R_tau_epsilon_star = R_epsilon * diagmat(tau_epsilon_star);
        double mh1 = sum(log(tau2_epsilon_star));      // jacobian of log-scale proposal
        double mh2 = sum(log(tau2_epsilon));           // jacobian of log-scale proposal
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon_star, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int j=0; j<d; j++) {
          mh1 += d_half_cauchy(tau2_epsilon_star(j), s2_tau2, true);
          mh2 += d_half_cauchy(tau2_epsilon(j), s2_tau2, true);
          // mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
          // mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau2_epsilon = tau2_epsilon_star;
          tau_epsilon = tau_epsilon_star;
          R_tau_epsilon = R_tau_epsilon_star;
          tau2_epsilon_accept += 1.0 / n_mcmc;
        }
      }
    }


    //
    // sample xi_epsilon - MH
    //

    if (sample_xi_epsilon) {
      arma::vec logit_xi_epsilon_tilde_star = mvrnormArmaVecChol(logit(xi_epsilon_tilde),
                                                                 lambda_xi_epsilon_tune * Sigma_xi_epsilon_tune_chol);
      arma::vec xi_epsilon_tilde_star = expit(logit_xi_epsilon_tilde_star);
      arma::vec xi_epsilon_star = 2.0 * xi_epsilon_tilde_star - 1.0;
      // arma::vec xi_star =  mvrnormArmaVecChol(xi, lambda_xi_tune * Sigma_xi_tune_chol);
      if (all(xi_epsilon_star > -1.0) && all(xi_epsilon_star < 1.0)) {
        Rcpp::List R_epsilon_out = makeRLKJ(xi_epsilon_star, d, true, true);
        arma::mat R_epsilon_star = as<mat>(R_epsilon_out["R"]);
        arma::mat R_tau_epsilon_star = R_epsilon_star * diagmat(tau_epsilon);
        double log_jacobian_epsilon_star = as<double>(R_epsilon_out["log_jacobian"]);
        double mh1 = sum(log(xi_epsilon_tilde_star) + log(ones_B - xi_epsilon_tilde_star)); // Jacobian adjustment
        double mh2 = sum(log(xi_epsilon_tilde) + log(ones_B - xi_epsilon_tilde));          // Jacobian adjustment
        for (int i=0; i<N; i++) {
          mh1 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon_star, true);
          mh2 += dMVN(Y.row(i).t(), mu + zeta.row(i).t(), R_tau_epsilon, true);
        }
        for (int b=0; b<B; b++) {
          mh1 += R::dbeta(0.5 * (xi_epsilon_star(b) + 1.0), eta_vec(b), eta_vec(b), true);
          mh2 += R::dbeta(0.5 * (xi_epsilon(b) + 1.0), eta_vec(b), eta_vec(b), true);
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          xi_epsilon_tilde = xi_epsilon_tilde_star;
          xi_epsilon = xi_epsilon_star;
          R_epsilon = R_epsilon_star;
          R_tau_epsilon = R_tau_epsilon_star;
          log_jacobian_epsilon = log_jacobian_epsilon_star;
          xi_epsilon_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      mu_save.row(save_idx) = mu.t();
      eta_star_save.subcube(span(save_idx), span(), span()) = eta_star;
      zeta_save.subcube(span(save_idx), span(), span()) = zeta;
      Omega_save.subcube(span(save_idx), span(), span()) = R.t() * R;
      phi_save(save_idx) = phi;
      tau2_save.row(save_idx) = tau2.t();
      tau2_epsilon_save.row(save_idx) = tau2_epsilon.t();
      // lambda_tau2_save.row(save_idx) = lambda_tau2.t();
      // s2_tau2_save(save_idx) = s2_tau2;
      c_save.subcube(span(save_idx), span(), span()) = c;
      C_save.subcube(span(save_idx), span(), span()) = C;
      C_inv_save.subcube(span(save_idx), span(), span()) = C_inv;
      Z_save.subcube(span(save_idx), span(), span()) = Z;
      R_save.subcube(span(save_idx), span(), span()) = R;
      R_tau_save.subcube(span(save_idx), span(), span()) = R_tau;
      xi_save.row(save_idx) = xi.t();
      R_epsilon_save.subcube(span(save_idx), span(), span()) = R_epsilon;
      R_tau_epsilon_save.subcube(span(save_idx), span(), span()) = R_tau_epsilon;
      xi_epsilon_save.row(save_idx) = xi_epsilon.t();
    }
  }

  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  if (sample_mu) {
    if (sample_mu_mh) {
      file_out << "Average acceptance rate for mu  = " << mean(mu_accept) <<
        " for chain " << n_chain << "\n";
    }
  }
  if (sample_eta_star) {
    if (sample_eta_star_mh) {
    file_out << "Average acceptance rate for eta_star  = " << mean(eta_star_accept) <<
      " for chain " << n_chain << "\n";
    }
  }
  if (sample_phi) {
    file_out << "Average acceptance rate for phi  = " << mean(phi_accept) <<
      " for chain " << n_chain << "\n";
  }
  // file_out << "Average acceptance rate for X  = " << mean(X_accept) <<
  // " for chain " << n_chain << "\n";
  if (sample_xi) {
    file_out << "Average acceptance rate for xi  = " << mean(xi_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_tau2) {
    file_out << "Average acceptance rate for tau2  = " << mean(tau2_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_xi_epsilon) {
    file_out << "Average acceptance rate for xi_epsilon  = " << mean(xi_epsilon_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_tau2_epsilon) {
    file_out << "Average acceptance rate for tau2_epsilon  = " << mean(tau2_epsilon_accept) <<
      " for chain " << n_chain << "\n";
  }
  // close output file
  file_out.close();

  return Rcpp::List::create(
    _["mu"] = mu_save,
    _["eta_star"] = eta_star_save,
    _["zeta"] = zeta_save,
    _["Omega"] = Omega_save,
    _["phi"] = phi_save,
    // _["sigma2"] = sigma2_save,
    _["tau2"] = tau2_save,
    _["tau2_epsilon"] = tau2_epsilon_save,
    // _["lambda_tau2"] = lambda_tau2_save,
    // _["s2_tau2"] = s2_tau2_save,
    _["R"] = R_save,
    _["R_tau"] = R_tau_save,
    _["xi"] = xi_save,
    _["R_epsilon"] = R_epsilon_save,
    _["R_tau_epsilon"] = R_tau_epsilon_save,
    _["xi_epsilon"] = xi_epsilon_save);

}
