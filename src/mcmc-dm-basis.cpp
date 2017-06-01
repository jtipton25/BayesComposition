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
///////////// Elliptical Slice Sampler for random effect eta_star /////////////
///////////////////////////////////////////////////////////////////////////////

// // [[Rcpp::export]]
// Rcpp::List ess (const arma::mat& eta_star_current,
//                 const arma::vec& prior_sample,
//                 const arma::mat& alpha_current,
//                 const arma::mat& mu_current,
//                 const arma::mat& zeta_current,
//                 const arma::mat& R_tau_current,
//                 const arma::mat& Z_current, const arma::mat& y,
//                 const int& N, const int& d, const int& j,
//                 const arma::vec& count,
//                 const std::string& file_name, const int& n_chain) {
//   // eta_star_current is the current value of the joint multivariate predictive process
//   // prior_sample is a sample from the prior joing multivariate predictive process
//   // R_tau is the current value of the Cholskey decomposition for  predictive process linear interpolator
//   // Z_current is the current predictive process linear
//
//   // calculate log likelihood of current value
//   double current_log_like = LL(alpha_current, y, N, d, count);
//   double hh = log(R::runif(0.0, 1.0)) + current_log_like;
//
//
//   // Setup a bracket and pick a first proposal
//   // Bracket whole ellipse with both edges at first proposed point
//   double phi_angle = R::runif(0.0, 1.0) * 2.0 * arma::datum::pi;
//   double phi_angle_min = phi_angle - 2.0 * arma::datum::pi;
//   double phi_angle_max = phi_angle;
//
//   arma::mat eta_star_ess = eta_star_current;
//   arma::mat eta_star_proposal = eta_star_current;
//   arma::mat zeta_ess = zeta_current;
//   arma::mat alpha_ess = alpha_current;
//   bool test = true;
//
//   // Slice sampling loop
//   while (test) {
//     // compute proposal for angle difference and check to see if it is on the slice
//     eta_star_proposal.col(j) = eta_star_current.col(j) * cos(phi_angle) +
//       prior_sample * sin(phi_angle);
//     arma::mat zeta_proposal = Z_current * eta_star_proposal * R_tau_current;
//     arma::mat alpha_proposal(N, d);
//     for (int i=0; i<N; i++) {
//       arma::rowvec tmp_row(d, arma::fill::ones);
//       tmp_row.subvec(1, d-1) = exp(mu_current.t() + zeta_proposal.row(i));
//       alpha_proposal.row(i) = tmp_row / sum(tmp_row);
//     }
//
//     // calculate log likelihood of proposed value
//     double proposal_log_like = LL(alpha_proposal, y, N, d, count);
//     // control to limit alpha from getting unreasonably large
//     if (alpha_proposal.max() > pow(10.0, 10.0) ) {
//       // Rprintf("Bug - alpha (eta_star) is to large for LL to be stable \n");
//       // // set up output messages
//       // std::ofstream file_out;
//       // file_out.open(file_name, std::ios_base::app);
//       // file_out << "Bug - alpha (eta_star) is to large for LL to be stable on chain " << n_chain << "\n";
//       // // close output file
//       // file_out.close();
//       if (phi_angle > 0.0) {
//         phi_angle_max = phi_angle;
//       } else if (phi_angle < 0.0) {
//         phi_angle_min = phi_angle;
//       } else {
//         Rprintf("Bug - ESS for eta_star shrunk to current position with large alpha \n");
//         // set up output messages
//         std::ofstream file_out;
//         file_out.open(file_name, std::ios_base::app);
//         file_out << "Bug - ESS for eta_star shrunk to current position with large alpha on chain " << n_chain << "\n";
//         // close output file
//         file_out.close();
//         // proposal failed and don't update the chain
//         // eta_star_ess = eta_star_proposal;
//         // zeta_ess = zeta_proposal;
//         // alpha_ess = alpha_proposal;
//         test = false;
//       }
//     } else {
//       if (proposal_log_like > hh) {
//         // proposal is on the slice
//         eta_star_ess = eta_star_proposal;
//         zeta_ess = zeta_proposal;
//         alpha_ess = alpha_proposal;
//         test = false;
//       } else if (phi_angle > 0.0) {
//         phi_angle_max = phi_angle;
//       } else if (phi_angle < 0.0) {
//         phi_angle_min = phi_angle;
//       } else {
//         Rprintf("Bug - ESS for eta_star shrunk to current position \n");
//         // set up output messages
//         std::ofstream file_out;
//         file_out.open(file_name, std::ios_base::app);
//         file_out << "Bug - ESS for eta_star shrunk to current position on chain " << n_chain << "\n";
//         // close output file
//         file_out.close();
//         // proposal failed and don't update the chain
//         // eta_star_ess = eta_star_proposal;
//         // zeta_ess = zeta_proposal;
//         // alpha_ess = alpha_proposal;
//         test = false;
//       }
//     }
//     // Propose new angle difference
//     phi_angle = R::runif(0.0, 1.0) * (phi_angle_max - phi_angle_min) + phi_angle_min;
//   }
//   return(Rcpp::List::create(
//       _["eta_star"] = eta_star_ess,
//       _["zeta"] = zeta_ess,
//       _["alpha"] = alpha_ess));
// }

///////////////////////////////////////////////////////////////////////////////
///////////// Elliptical Slice Sampler for unobserved covariate X /////////////
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess_X (const double& X_current, const double& X_prior,
                  const double& mu_X,
                  const arma::mat& beta_current,
                  const arma::rowvec& alpha_current,
                  const arma::rowvec& y_current,
                  const arma::rowvec& Xbs_current,
                  const arma::vec& knots,
                  const double& d, const int& degree, const int& df,
                  const arma::vec& rangeX, const double& count_double,
                  const std::string& file_name, const int& n_chain) {
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
    arma::rowvec alpha_proposal = exp(Xbs_proposal * beta_current);

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
        file_out << "Bug - ESS for X shrunk to current position with large alpha on chain " << n_chain << "\n";
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
        file_out << "Bug - ESS for X shrunk to current position on chain " << n_chain << "\n";
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
//////// MVN density using Cholesky decomposition of Covariance Sigma /////////
///////////////////////////////////////////////////////////////////////////////

// // [[Rcpp::export]]
// double dMVN (const arma::mat& y, const arma::vec& mu,
//              const arma::mat& Sigma_chol, const bool logd=true){
//   int n = y.n_cols;
//   int ydim = y.n_rows;
//   arma::vec out(n);
//   arma::mat rooti = trans(inv(trimatu(Sigma_chol)));
//   double rootisum = sum(log(rooti.diag()));
//   double constants = - (static_cast<double>(ydim) / 2.0) * log2pi;
//   for (int i=0; i<n; i++) {
//     arma::vec z = rooti * (y.col(i) - mu) ;
//     out(i) = constants - 0.5 * sum(z % z) + rootisum;
//   }
//   if(logd){
//     return(sum(out));
//   } else {
//     return(exp(sum(out)));
//   }
// }

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppDMBasis (const arma::mat& Y, const arma::vec& X,
                      const arma::mat& Y_pred, List params,
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
  double N = Y.n_rows;
  double d = Y.n_cols;

  // set up dimensions
  double N_pred = Y_pred.n_rows;

  // double B = Rf_choose(d, 2);
  double B = Rf_choose(d, 2);

  // count - sum of counts at each site
  arma::vec count(N);
  for (int i=0; i<N; i++) {
    count(i) = sum(Y.row(i));
  }

  // count - sum of counts at each site
  arma::vec count_pred(N_pred);
  for (int i=0; i<N_pred; i++) {
    count_pred(i) = sum(Y_pred.row(i));
  }

  // constant vectors
  arma::mat I_d(d, d, arma::fill::eye);
  arma::vec ones_d(d, arma::fill::ones);
  arma::vec ones_B(B, arma::fill::ones);
  arma::vec zero_df(df, arma::fill::zeros);

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

  // // default half cauchy scale for Covariance diagonal variance tau2
  // double s2_tau2 = 1.0;
  // if (params.containsElementNamed("s2_tau2")) {
  //   s2_tau2 = as<double>(params["s2_tau2"]);
  // }
  // // default half cauchy scale for Covariance diagonal variance tau2
  // double A_s2 = 1.0;
  // if (params.containsElementNamed("A_s2")) {
  //   A_s2 = as<double>(params["A_s2"]);
  // }
  // // default xi LKJ concentation parameter of 1
  // double eta = 1.0;
  // if (params.containsElementNamed("eta")) {
  //   eta = as<double>(params["eta"]);
  // }
  // default beta tuning parameter
  arma::vec lambda_beta_tune(d, arma::fill::ones);
  lambda_beta_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_beta_tune")) {
    lambda_beta_tune = as<vec>(params["lambda_beta_tune"]);
  }
  // // default tau2 tuning parameter
  // double lambda_tau2_tune = 0.25;
  // if (params.containsElementNamed("lambda_tau2_tune")) {
  //   lambda_tau2_tune = as<double>(params["lambda_tau2_tune"]);
  // }
  // // default xi tuning parameter
  // double lambda_xi_tune = 1.0 / pow(3.0, 0.8);
  // if (params.containsElementNamed("lambda_xi_tune")) {
  //   lambda_xi_tune = as<double>(params["lambda_xi_tune"]);
  // }
  // default X tuning parameter standard deviation of 0.25
  double X_tune_tmp = 0.5;
  if (params.containsElementNamed("X_tune")) {
    X_tune_tmp = as<double>(params["X_tune"]);
  }

  //
  // Turn on/off sampler for X
  //

  // default to centered missing covariate
  double mu_X = arma::mean(X);
  // default to scaled missing covariate
  double s2_X = arma::var(X);

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
  arma::mat Xbs_pred = bs_cpp(X_pred, df, knots, degree, true, rangeX);

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

  // //
  // // Variance parameter tau2 and hyperprior lambda_tau2
  // //
  //
  // arma::vec lambda_tau2(d);
  // arma::vec tau2(d);
  // for (int j=0; j<d; j++) {
  //   lambda_tau2(j) = R::rgamma(0.5, 1.0 / s2_tau2);
  //   tau2(j) = std::max(std::min(R::rgamma(0.5, 1.0 / lambda_tau2(j)), 5.0), 1.0);
  // }
  // arma::vec tau = sqrt(tau2);
  // if (params.containsElementNamed("tau2")) {
  //   tau2 = as<vec>(params["tau2"]);
  //   tau = sqrt(tau2);
  // }
  // bool sample_tau2 = true;
  // if (params.containsElementNamed("sample_tau2")) {
  //   sample_tau2 = as<bool>(params["sample_tau2"]);
  // }
  // // fix first element of Sigma_{1,1} at 1.0 for identifiability
  // if (Sigma_reference_category) {
  //   tau2(0) = 1.0;
  //   tau(0) = 1.0;
  // }

  // //
  // // Default LKJ hyperparameter xi
  // //
  //
  // arma::vec eta_vec(B);
  // int idx_eta = 0;
  // for (int j=0; j<d; j++) {
  //   for (int k=0; k<(d-j-1); k++) {
  //     eta_vec(idx_eta+k) = eta + d - 2.0 - j) / 2.0;
  //   }
  //   idx_eta += d-j-1;
  // }
  //
  // arma::vec xi(B);
  // for (int b=0; b<B; b++) {
  //   xi(b) = 2.0 * R::rbeta(eta_vec(b), eta_vec(b)) - 1.0;
  // }
  // arma::vec xi_tilde(B);
  // if (params.containsElementNamed("xi")) {
  //   xi = as<vec>(params["xi"]);
  // }
  // for (int b=0; b<B; b++) {
  //   xi_tilde(b) = 0.5 * (xi(b) + 1.0);
  // }
  // bool sample_xi = true;
  // if (params.containsElementNamed("sample_xi")) {
  //   sample_xi = as<bool>(params["sample_xi"]);
  // }
  // Rcpp::List R_out = makeRLKJ(xi, d, true, true);
  // double log_jacobian = as<double>(R_out["log_jacobian"]);
  // arma::mat R = as<mat>(R_out["R"]);
  // arma::mat R_tau = R * diagmat(tau);
  // arma::mat zeta = Z * eta_star * R_tau;
  arma::mat alpha = exp(Xbs * beta);
  arma::mat alpha_pred = exp(Xbs_pred * beta);

  // setup save variables
  int n_save = n_mcmc / n_thin;
  arma::cube alpha_save(n_save, N, d, arma::fill::zeros);
  arma::cube alpha_pred_save(n_save, N_pred, d, arma::fill::zeros);
  arma::cube beta_save(n_save, df, d, arma::fill::zeros);
  arma::mat X_save(n_save, N_pred, arma::fill::zeros);
  // arma::mat tau2_save(n_save, d, arma::fill::zeros);
  // arma::mat lambda_tau2_save(n_save, d, arma::fill::zeros);
  // arma::vec s2_tau2_save(n_save, arma::fill::zeros);
  // arma::cube R_save(n_save, d, d, arma::fill::zeros);
  // arma::mat xi_save(n_save, B, arma::fill::zeros);

  // initialize tuning

  // double s2_tau2_accept = 0.0;
  // double s2_tau2_accept_batch = 0.0;
  // double s2_tau2_tune = 1.0;
  // double tau2_accept = 0.0;
  // double tau2_accept_batch = 0.0;
  // arma::mat tau2_batch(50, d, arma::fill::zeros);
  // arma::mat Sigma_tau2_tune(d, d, arma::fill::eye);
  // arma::mat Sigma_tau2_tune_chol = chol(Sigma_tau2_tune);
  // if (Sigma_reference_category) {
  //   // reduce dimension if first element Sigma_{1, 1} is fixed at 1
  //   Sigma_tau2_tune.shed_col(d-2);
  //   Sigma_tau2_tune.shed_row(d-2);
  //   Sigma_tau2_tune_chol.shed_col(d-2);
  //   Sigma_tau2_tune_chol.shed_row(d-2);
  //   tau2_batch.shed_col(d-2);
  // }
  // double xi_accept = 0.0;
  // double xi_accept_batch = 0.0;
  // arma::mat xi_batch(50, B, arma::fill::zeros);
  // arma::mat Sigma_xi_tune(B, B, arma::fill::eye);
  // arma::mat Sigma_xi_tune_chol = chol(Sigma_xi_tune);
  arma::vec X_tune(N_pred, arma::fill::ones);
  X_tune *= X_tune_tmp;
  arma::vec X_accept(N_pred, arma::fill::zeros);
  arma::vec X_accept_batch(N_pred, arma::fill::zeros);
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

    //
    // Sample beta - block MH
    //

    if (sample_beta) {
      for (int j=0; j<d; j++) {
        arma::mat beta_star = beta;
        beta_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));
        arma::mat alpha_star = exp(Xbs * beta_star);
        // construct updated alpha for unobserved data
        arma::mat alpha_pred_star = exp(Xbs_pred * beta_star);
        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          dMVN(beta_star.col(j), mu_beta, Sigma_beta_chol);
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          dMVN(beta.col(j), mu_beta, Sigma_beta_chol);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha = alpha_star;
          alpha_pred = alpha_pred_star;
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

    // //
    // // sample tau2
    // //
    //
    // if (sample_tau2) {
    //   arma::vec log_tau2_star = log(tau2);
    //   if (Sigma_reference_category) {
    //     // first element is fixed at one
    //     log_tau2_star.subvec(1, d-2) =
    //       mvrnormArmaVecChol(log(tau2.subvec(1, d-2)),
    //                          lambda_tau2_tune * Sigma_tau2_tune_chol);
    //   } else {
    //     log_tau2_star =
    //       mvrnormArmaVecChol(log(tau2),
    //                          lambda_tau2_tune * Sigma_tau2_tune_chol);
    //   }
    //   arma::vec tau2_star = exp(log_tau2_star);
    //   if (all(tau2_star > 0.0)) {
    //     arma::vec tau_star = sqrt(tau2_star);
    //     arma::mat R_tau_star = R * diagmat(tau_star);
    //     arma::mat zeta_star = Z * eta_star * R_tau_star;
    //     arma::mat alpha_star(N, d, arma::fill::ones);
    //     for (int i=0; i<N; i++) {
    //       arma::rowvec tmp_row(d, arma::fill::ones);
    //       tmp_row.subvec(1, d-1) = exp(mu.t() + zeta_star.row(i));
    //       alpha_star.row(i) = tmp_row / sum(tmp_row);
    //     }
    //     double mh1 = LL(alpha_star, Y, N, d, count) +
    //       // jacobian of log-scale proposal
    //       sum(log_tau2_star);
    //     double mh2 = LL(alpha, Y, N, d, count) +
    //       // jacobian of log-scale proposal
    //       sum(log(tau2));
    //     // prior
    //     if (Sigma_reference_category) {
    //       for (int j=1; j<(d-1); j++) {
    //         mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
    //         mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
    //       }
    //     } else {
    //       for (int j=0; j<(d-1); j++) {
    //         mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
    //         mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
    //       }
    //     }
    //     double mh = exp(mh1-mh2);
    //     if (mh > R::runif(0.0, 1.0)) {
    //       tau2 = tau2_star;
    //       tau = tau_star;
    //       R_tau = R_tau_star;
    //       zeta = zeta_star;
    //       alpha = alpha_star;
    //       tau2_accept_batch += 1.0 / 50.0;
    //     }
    //   }
    //   // update tuning
    //   if (Sigma_reference_category) {
    //     tau2_batch.row(k % 50) = log(tau2.subvec(1, d-2)).t();
    //   } else {
    //     tau2_batch.row(k % 50) = log(tau2).t();
    //   }
    //   if ((k+1) % 50 == 0){
    //     updateTuningMV(k, tau2_accept_batch, lambda_tau2_tune, tau2_batch,
    //                    Sigma_tau2_tune, Sigma_tau2_tune_chol);
    //   }
    // }
    //
    // //
    // // sample lambda_tau2
    // //
    //
    // for (int j=0; j<(d-1); j++) {
    //   lambda_tau2(j) = R::rgamma(1.0, 1.0 / (s2_tau2 + tau2(j)));
    // }
    //
    // //
    // // sample s2_tau2
    // //
    //
    // if (pool_s2_tau2) {
    //   double s2_tau2_star = s2_tau2 + R::rnorm(0, s2_tau2_tune);
    //   if (s2_tau2_star > 0 && s2_tau2_star < A_s2) {
    //     double mh1 = 0.0;
    //     double mh2 = 0.0;
    //     for (int j=0; j<(d-1); j++) {
    //       mh1 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2_star, true);
    //       mh2 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2, true);
    //     }
    //     double mh = exp(mh1-mh2);
    //     if (mh > R::runif(0.0, 1.0)) {
    //       s2_tau2 = s2_tau2_star;
    //       s2_tau2_accept += 1.0 / n_mcmc;
    //       s2_tau2_accept_batch += 1.0 / 50.0;
    //     }
    //   }
    //   // update tuning
    //   if ((k+1) % 50 == 0){
    //     updateTuning(k, s2_tau2_accept_batch, s2_tau2_tune);
    //   }
    // }
    //
    // //
    // // sample xi - MH
    // //
    //
    // if (sample_xi) {
    //   arma::vec logit_xi_tilde_star = mvrnormArmaVecChol(logit(xi_tilde),
    //                                                      lambda_xi_tune * Sigma_xi_tune_chol);
    //   arma::vec xi_tilde_star = expit(logit_xi_tilde_star);
    //   arma::vec xi_star = 2.0 * xi_tilde_star - 1.0;
    //   if (all(xi_star > -1.0) && all(xi_star < 1.0)) {
    //     Rcpp::List R_out = makeRLKJ(xi_star, d-1, true, true);
    //     arma::mat R_star = as<mat>(R_out["R"]);
    //     arma::mat R_tau_star = R_star * diagmat(tau);
    //     arma::mat zeta_star = Z * eta_star * R_tau_star;
    //     arma::mat alpha_star(N, d, arma::fill::ones);
    //     for (int i=0; i<N; i++) {
    //       arma::rowvec tmp_row(d, arma::fill::ones);
    //       tmp_row.subvec(1, d-1) = exp(mu.t() + zeta_star.row(i));
    //       alpha_star.row(i) = tmp_row / sum(tmp_row);
    //     }
    //
    //     double log_jacobian_star = as<double>(R_out["log_jacobian"]);
    //     double mh1 = LL(alpha_star, Y, N, d, count) +
    //       // Jacobian adjustment
    //       sum(log(xi_tilde_star) + log(ones_B - xi_tilde_star));
    //     double mh2 = LL(alpha, Y, N, d, count) +
    //       // Jacobian adjustment
    //       sum(log(xi_tilde) + log(ones_B - xi_tilde));
    //     for (int b=0; b<B; b++) {
    //       mh1 += R::dbeta(0.5 * (xi_star(b) + 1.0), eta_vec(b), eta_vec(b), true);
    //       mh2 += R::dbeta(0.5 * (xi(b) + 1.0), eta_vec(b), eta_vec(b), true);
    //     }
    //     double mh = exp(mh1-mh2);
    //     if (mh > R::runif(0.0, 1.0)) {
    //       xi_tilde = xi_tilde_star;
    //       xi = xi_star;
    //       R = R_star;
    //       R_tau = R_tau_star;
    //       log_jacobian = log_jacobian_star;
    //       zeta = zeta_star;
    //       alpha = alpha_star;
    //       xi_accept_batch += 1.0 / 50.0;
    //     }
    //   }
    //   // update tuning
    //   xi_batch.row(k % 50) = logit(xi_tilde).t();
    //   if ((k+1) % 50 == 0){
    //     updateTuningMV(k, xi_accept_batch, lambda_xi_tune, xi_batch,
    //                    Sigma_xi_tune, Sigma_xi_tune_chol);
    //   }
    // }

    //
    // sample X - ESS
    //

    if (sample_X) {
      for (int i=0; i<N_pred; i++) {
        double X_prior = R::rnorm(0.0, s_X);
        Rcpp::List ess_out = ess_X(X_pred(i), X_prior, mu_X, beta, alpha_pred.row(i),
                                   Y_pred.row(i), Xbs_pred.row(i), knots, d,
                                   degree, df, rangeX, count_pred(i), file_name,
                                   n_chain);
        X_pred(i) = as<double>(ess_out["X"]);
        Xbs_pred.row(i) = as<rowvec>(ess_out["Xbs"]);
        alpha_pred.row(i) = as<rowvec>(ess_out["alpha"]);
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

        arma::mat alpha_star = exp(Xbs * beta_star);
        // construct updated alpha for unobserved data
        arma::mat alpha_pred_star = exp(Xbs_pred * beta_star);

        double mh1 = LL_DM(alpha_star, Y, N, d, count) +
          dMVN(beta_star.col(j), mu_beta, Sigma_beta_chol);
        double mh2 = LL_DM(alpha, Y, N, d, count) +
          dMVN(beta.col(j), mu_beta, Sigma_beta_chol);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha = alpha_star;
          alpha_pred = alpha_pred_star;
          beta_accept(j) += 1.0 / n_mcmc;
        }
      }
    }


    // //
    // // sample tau2
    // //
    //
    // if (sample_tau2) {
    //   arma::vec log_tau2_star = log(tau2);
    //   if (Sigma_reference_category) {
    //     // first element is fixed at one
    //     log_tau2_star.subvec(1, d-2) =
    //       mvrnormArmaVecChol(log(tau2.subvec(1, d-2)),
    //                          lambda_tau2_tune * Sigma_tau2_tune_chol);
    //   } else {
    //     log_tau2_star =
    //       mvrnormArmaVecChol(log(tau2),
    //                          lambda_tau2_tune * Sigma_tau2_tune_chol);
    //   }
    //   arma::vec tau2_star = exp(log_tau2_star);
    //   if (all(tau2_star > 0.0)) {
    //     arma::vec tau_star = sqrt(tau2_star);
    //     arma::mat R_tau_star = R * diagmat(tau_star);
    //     arma::mat zeta_star = Z * eta_star * R_tau_star;
    //     arma::mat alpha_star(N, d, arma::fill::ones);
    //     for (int i=0; i<N; i++) {
    //       arma::rowvec tmp_row(d, arma::fill::ones);
    //       tmp_row.subvec(1, d-1) = exp(mu.t() + zeta_star.row(i));
    //       alpha_star.row(i) = tmp_row / sum(tmp_row);
    //     }
    //     double mh1 = LL(alpha_star, Y, N, d, count) +
    //       // jacobian of log-scale proposal
    //       sum(log_tau2_star);
    //     double mh2 = LL(alpha, Y, N, d, count) +
    //       // jacobian of log-scale proposal
    //       sum(log(tau2));
    //     // prior
    //     if (Sigma_reference_category) {
    //       for (int j=1; j<(d-1); j++) {
    //         mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
    //         mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
    //       }
    //     } else {
    //       for (int j=0; j<(d-1); j++) {
    //         mh1 += R::dgamma(tau2_star(j), 0.5, 1.0 / lambda_tau2(j), true);
    //         mh2 += R::dgamma(tau2(j), 0.5, 1.0 / lambda_tau2(j), true);
    //       }
    //     }
    //     double mh = exp(mh1-mh2);
    //     if (mh > R::runif(0.0, 1.0)) {
    //       tau2 = tau2_star;
    //       tau = tau_star;
    //       R_tau = R_tau_star;
    //       zeta = zeta_star;
    //       alpha = alpha_star;
    //       tau2_accept += 1.0 / n_mcmc;
    //     }
    //   }
    // }
    //
    // //
    // // sample lambda_tau2
    // //
    //
    // for (int j=0; j<(d-1); j++) {
    //   lambda_tau2(j) = R::rgamma(1.0, 1.0 / (s2_tau2 + tau2(j)));
    // }
    //
    // //
    // // sample s2_tau2
    // //
    //
    // if (pool_s2_tau2) {
    //
    //   double s2_tau2_star = s2_tau2 + R::rnorm(0, s2_tau2_tune);
    //   if (s2_tau2_star > 0 && s2_tau2_star < A_s2) {
    //     double mh1 = 0.0;
    //     double mh2 = 0.0;
    //     for (int j=0; j<(d-1); j++) {
    //       mh1 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2_star, true);
    //       mh2 += R::dgamma(lambda_tau2(j), 0.5, 1.0 / s2_tau2, true);
    //     }
    //     double mh = exp(mh1-mh2);
    //     if (mh > R::runif(0.0, 1.0)) {
    //       s2_tau2 = s2_tau2_star;
    //       s2_tau2_accept += 1.0 / n_mcmc;
    //     }
    //   }
    // }
    //
    // //
    // // sample xi - MH
    // //
    //
    // if (sample_xi) {
    //   arma::vec logit_xi_tilde_star = mvrnormArmaVecChol(logit(xi_tilde),
    //                                                      lambda_xi_tune * Sigma_xi_tune_chol);
    //   arma::vec xi_tilde_star = expit(logit_xi_tilde_star);
    //   arma::vec xi_star = 2.0 * xi_tilde_star - 1.0;
    //   if (all(xi_star > -1.0) && all(xi_star < 1.0)) {
    //     Rcpp::List R_out = makeRLKJ(xi_star, d-1, true, true);
    //     arma::mat R_star = as<mat>(R_out["R"]);
    //     arma::mat R_tau_star = R_star * diagmat(tau);
    //     arma::mat zeta_star = Z * eta_star * R_tau_star;
    //     arma::mat alpha_star(N, d, arma::fill::ones);
    //     for (int i=0; i<N; i++) {
    //       arma::rowvec tmp_row(d, arma::fill::ones);
    //       tmp_row.subvec(1, d-1) = exp(mu.t() + zeta_star.row(i));
    //       alpha_star.row(i) = tmp_row / sum(tmp_row);
    //     }
    //
    //     double log_jacobian_star = as<double>(R_out["log_jacobian"]);
    //     double mh1 = LL(alpha_star, Y, N, d, count) +
    //       // Jacobian adjustment
    //       sum(log(xi_tilde_star) + log(ones_B - xi_tilde_star));
    //     double mh2 = LL(alpha, Y, N, d, count) +
    //       // Jacobian adjustment
    //       sum(log(xi_tilde) + log(ones_B - xi_tilde));
    //     for (int b=0; b<B; b++) {
    //       mh1 += R::dbeta(0.5 * (xi_star(b) + 1.0), eta_vec(b), eta_vec(b), true);
    //       mh2 += R::dbeta(0.5 * (xi(b) + 1.0), eta_vec(b), eta_vec(b), true);
    //     }
    //     double mh = exp(mh1-mh2);
    //     if (mh > R::runif(0.0, 1.0)) {
    //       xi_tilde = xi_tilde_star;
    //       xi = xi_star;
    //       R = R_star;
    //       R_tau = R_tau_star;
    //       log_jacobian = log_jacobian_star;
    //       zeta = zeta_star;
    //       alpha = alpha_star;
    //       xi_accept += 1.0 / n_mcmc;
    //     }
    //   }
    // }

    //
    // sample X - ESS
    //

    if (sample_X) {
      for (int i=0; i<N_pred; i++) {
        double X_prior = R::rnorm(0.0, s_X);
        Rcpp::List ess_out = ess_X(X_pred(i), X_prior, mu_X, beta, alpha_pred.row(i),
                                   Y_pred.row(i), Xbs_pred.row(i), knots, d,
                                   degree, df, rangeX, count_pred(i), file_name,
                                   n_chain);
        X_pred(i) = as<double>(ess_out["X"]);
        Xbs_pred.row(i) = as<rowvec>(ess_out["Xbs"]);
        alpha_pred.row(i) = as<rowvec>(ess_out["alpha"]);
      }
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      alpha_save.subcube(span(save_idx), span(), span()) = alpha;
      alpha_pred_save.subcube(span(save_idx), span(), span()) = alpha_pred;
      beta_save.subcube(span(save_idx), span(), span()) = beta;
      X_save.row(save_idx) = X_pred.t() + mu_X;
      // tau2_save.row(save_idx) = tau2.t();
      // lambda_tau2_save.row(save_idx) = lambda_tau2.t();
      // s2_tau2_save(save_idx) = s2_tau2;
      // R_save.subcube(span(save_idx), span(), span()) = R;
      // xi_save.row(save_idx) = xi.t();
    }
  }

  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  if (sample_beta) {
    file_out << "Average acceptance rate for beta  = " << mean(beta_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_X) {
    file_out << "Average acceptance rate for X  = " << mean(X_accept) <<
      " for chain " << n_chain << "\n";
  }
  // if (sample_xi) {
  //   file_out << "Average acceptance rate for xi  = " << mean(xi_accept) <<
  //     " for chain " << n_chain << "\n";
  // }
  // if (sample_tau2) {
  //   file_out << "Average acceptance rate for tau2  = " << mean(tau2_accept) <<
  //     " for chain " << n_chain << "\n";
  // }
  // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    _["alpha"] = alpha_save,
    _["alpha_pred"] = alpha_pred_save,
    _["beta"] = beta_save,
    _["X"] = X_save);
  // _["tau2"] = tau2_save,
  // _["lambda_tau2"] = lambda_tau2_save,
  // _["s2_tau2"] = s2_tau2_save,
  // _["R"] = R_save,
  // _["xi"] = xi_save);

}
