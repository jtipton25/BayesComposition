// #define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include "BayesComp.h"
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Dirichlet-Multinomial Multivariate Gaussian Process for spatio-temporal Inverse Inference

// Author: John Tipton
//
// Created 02.04.2019
// Last updated 02.04.2019

// ///////////////////////////////////////////////////////////////////////////////
// /////////////////////////// Functions for sampling ////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////
//
// ///////////////////////////////////////////////////////////////////////////////
// /////////////// construct a vector along a sequence of length n ///////////////
// ///////////////////////////////////////////////////////////////////////////////
//
// NumericVector seq_lenBS(const int& n) {
//   if (n < 2) {
//     stop("Inadmissible value");
//   }
//   NumericVector out(n);
//   out(0) = 0.0;
//   out(n-1) = 1.0;
//   if (n > 2) {
//     double n_double = double(n-1);
//     for (int i=1; i<(n-1); i++) {
//       double j = double(i);
//       out(i) = j / n_double;
//     }
//   }
//
//   return(out);
// }
//
// ///////////////////////////////////////////////////////////////////////////////
// ///////// construct a vector of percentiles for construction of knots /////////
// /////////  placed evenly along the percentiles ////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////
//
// NumericVector quantileBS(NumericVector x, NumericVector probs) {
//   Environment stats("package:stats");
//   Function quantile = stats["quantile"];
//   int npr = probs.size();
//   NumericVector ans(npr);
//   for(int i=0; i<npr; i++){
//     ans[i] = as<double>(quantile(x, probs[i]));
//   }
//   return(ans);
// }
//
// // [[Rcpp::export]]
// double basis_cpp (const double& x, const int& degree, const int& i,
//                   const arma::vec& knots) {
//   double B;
//   double alpha1;
//   double alpha2;
//   if (degree == 0) {
//     if ((x >= knots(i)) && (x < knots(i+1))) {
//       B = 1.0;
//     } else {
//       B = 0.0;
//     }
//   } else {
//     if((knots(degree+i) - knots(i)) == 0.0) {
//       alpha1 = 0.0;
//     } else {
//       alpha1 = (x - knots(i))/(knots(degree+i) - knots(i));
//     }
//     if((knots(i+degree+1) - knots(i+1)) == 0.0) {
//       alpha2 = 0.0;
//     } else {
//       alpha2 = (knots(i+degree+1) - x)/(knots(i+degree+1) - knots(i+1));
//     }
//     B = alpha1*basis_cpp(x, (degree-1), i, knots) +
//       alpha2*basis_cpp(x, (degree-1), (i+1), knots);
//   }
//   return(B);
// }
//
// // [[Rcpp::export]]
// arma::mat bs_cpp (const arma::vec& x, const int& df,
//                   const arma::vec& interior_knots, const int& degree,
//                   const bool& intercept,
//                   const arma::vec& Boundary_knots) {
//   arma::vec Boundary_knots_sorted = sort(Boundary_knots);
//   arma::vec knots_lower(degree+1);
//   knots_lower.fill(Boundary_knots_sorted(0));
//   arma::vec knots_upper(degree+1);
//   knots_upper.fill(Boundary_knots_sorted(1));
//   arma::vec knots_tmp = join_cols(knots_lower, interior_knots);
//   arma::vec knots = join_cols(knots_tmp, knots_upper);
//   int K = interior_knots.n_elem + degree + 1;
//   int nx = x.n_elem;
//   arma::mat B_mat(nx, K, arma::fill::zeros);
//   for (int i=0; i<nx; i++) {
//     for (int j=0; j<K; j++) {
//       B_mat(i, j) = basis_cpp(x(i), degree, j, knots);
//     }
//   }
//   if(any(x == Boundary_knots_sorted(1))){
//     arma::uvec idx = arma::find(x == Boundary_knots_sorted(1));
//     int n_idx = idx.n_elem;
//     arma::uvec K_uvec(n_idx);
//     K_uvec.fill(K-1);
//     arma::vec one_vec(n_idx, arma::fill::ones);
//     // B_mat.submat(idx, K_uvec) = one_vec;
//     for (int i=0; i<n_idx; i++) {
//       B_mat(idx(i), K_uvec(i)) = 1.0;
//     }
//   }
//   if(intercept) {
//     return(B_mat);
//   } else {
//     return(B_mat.cols(span(1, K-1)));
//   }
// }
//
//
// [[Rcpp::export]]
arma::rowvec pmax(const arma::rowvec& x,
                  double tol) {
  // x is the vector to take an elementwise maximum relative to the tolerance tol
  // tol is the lower bound to be applied to the vector.
  //
  // All elements of the output vector will be greater than tol
  int d = x.n_elem;
  arma::rowvec out = x;
  for (int j=0; j<d; j++) {
    if (x(j) <= tol) {
      out(j) = tol;
    }
  }
  return(out);
}


// [[Rcpp::export]]
double ddirichletmultinomial (const arma::rowvec& Y,
                              const arma::rowvec& alpha,
                              const bool& logd=true) {
  // alpha is an d-rowvector of transformed parameters
  // Y is a d-rowvector of species counts
  //
  // calculate the  (log) density of a dirichlet-multinomial variable
  // given parameter alpha.
  int d = alpha.n_elem;
  double count = sum(Y);
  double sumAlpha = sum(alpha);
  double mh = lgamma(sumAlpha) - lgamma(count + sumAlpha);
  for (int j=0; j<d; j++) {
    mh += lgamma(Y(j) + alpha(j)) - lgamma(alpha(j));
  }
  if (logd) {
    return(mh);
  } else {
    return(exp(mh));
  }
}
//
// // [[Rcpp::export]]
// arma::mat makeDistARMA(const arma::mat & coords1,
//                        const arma::mat & coords2) {
//   // coords1 is a matrix of coordinates.
//   // coords2 is a matrix of coordinates.
//   //
//   // calculate the pairwise distance between all locations
//   // in coords1 and coords2.
//   int nrows1 = coords1.n_rows;
//   int nrows2 = coords2.n_rows;
//   arma::mat D(nrows1, nrows2, fill::zeros);
//   int ncols1 = coords1.n_cols;
//   if(ncols1 == 1){
//     for(int i = 0; i < nrows1; i++){
//       for(int j = 0; j < nrows2; j++){
//         D(i, j) = sqrt(pow(coords1(i) - coords2(j), 2));
//       }
//     }
//   }
//   if(ncols1 == 2){
//     for(int i = 0; i < nrows1; i++){
//       for(int j = 0; j < nrows2; j++){
//         double tmp = 0;
//         for(int k = 0; k < 2; k++){
//           tmp += pow(coords1(i, k) - coords2(j, k), 2);
//         }
//         D(i, j) = sqrt(tmp);
//       }
//     }
//   }
//   return(D);
// }



// [[Rcpp::export]]
arma::mat make_alpha(const arma::mat& Xbs,
                     const arma::mat& beta,
                     const std::string& link) {
  // Xbs is a matrix of bspline expansions of climate
  // beta a matrix of coefficients
  // link is a character string indicating whether a log- or tobit- link is used
  //
  // this function calculates the link-transformation between the latent state
  // and the parameter in the Dirichlet-multinomial likelihood
  //
  int N = Xbs.n_rows;
  int d = beta.n_cols;
  arma::mat alpha(N, d);
  for (int i=0; i<N; i++) {
    if (link == "log") {
      alpha.row(i) = exp(Xbs.row(i) * beta);
    } else {
      alpha.row(i) = pmax(Xbs.row(i) * beta, pow(10.0, -8.0));
    }
  }
  return(alpha);
}


// [[Rcpp::export]]
arma::mat make_Xbs(const arma::mat& X_pred,
                   const arma::vec& X_mean,
                   const arma::vec& site_idx,
                   const arma::vec& year_idx,
                   const arma::vec& knots,
                   const arma::vec& rangeX,
                   const int& N,
                   const int& df,
                   const int& degree) {
  // X_pred is a matrix of reconstructed climate anomalies
  // X_mean a vector of climate fixed effects
  // site_idx is a vector of indicators of which site is observed
  // year_idx is a vector of indicators of which year is observed
  // knots is a vector of B-spline knot locations
  // rangeX is an extent vector from the modern calibration data
  // N is the number of observations
  // df is the B-spline degrees of freedom
  // degree is the B-spline polynomial degree
  //
  // this function calculates the B-spline basis expansion of climate
  //
  arma::mat Xbs(N, df);
  for (int i=0; i<N; i++) {
    arma::vec X_tmp(1);
    X_tmp = X_pred(site_idx(i), year_idx(i)) + X_mean(site_idx(i));
    Xbs.row(i) =  bs_cpp(X_tmp, df, knots, degree, false, rangeX);
  }
  return(Xbs);
}

// [[Rcpp::export]]
arma::mat make_Xbs_calibrate(const arma::mat& X,
                             const arma::vec& site_idx,
                             const arma::vec& year_idx,
                             const arma::vec& knots,
                             const arma::vec& rangeX,
                             const int& N,
                             const int& df,
                             const int& degree) {
  // X_pred is a matrix of reconstructed climate anomalies
  // X_mean a vector of climate fixed effects
  // site_idx is a vector of indicators of which site is observed
  // year_idx is a vector of indicators of which year is observed
  // knots is a vector of B-spline knot locations
  // rangeX is an extent vector from the modern calibration data
  // N is the number of observations
  // df is the B-spline degrees of freedom
  // degree is the B-spline polynomial degree
  //
  // this function calculates the B-spline basis expansion of climate
  //
  arma::mat Xbs(N, df);
  for (int i=0; i<N; i++) {
    Xbs.row(i) =  bs_cpp(X, df, knots, degree, false, rangeX);
  }
  return(Xbs);
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////// Elliptical Slice Samplers //////////////////////////
///////////////////////////////////////////////////////////////////////////////

Rcpp::List ess (const arma::mat& y,
                const arma::vec& X_current,
                const arma::vec& X_prior,
                const arma::vec& X_prior_mean,
                const arma::vec& X_mean,
                const arma::mat& Xbs,
                const arma::vec& knots,
                const int& degree,
                const int& df,
                const arma::vec& rangeX,
                const arma::mat& beta,
                const double& t,
                const arma::vec& site_idx,
                const arma::vec& year_idx,
                const std::string& link,
                const std::string& file_name,
                const int& n_chain) {
  // y is an N by d matrix of fossil pollen measurements
  // X is an N-vector of current climate value estimates
  // X_prior is a N-vector of proposed climate values
  // beta is a d by 2 matrix of species response curve coefficients

  // comment out this intercept when code is robust
  Rcpp::checkUserInterrupt();

  int N = y.n_rows;
  int d = y.n_cols;

  arma::mat alpha(N, d, arma::fill::zeros);
  for (int i=0; i<N; i++) {
    if (year_idx(i) == t) {
      if (link == "log") {
        alpha.row(i) = exp(Xbs.row(i) * beta);
      } else {
        alpha.row(i) = pmax(Xbs.row(i) * beta, 1e-8);
      }
    }
  }
  // calculate log likelihood of current value
  double current_log_like = 0.0;
  for (int i=0; i<N; i++) {
    if (year_idx(i) == t) {
      // only calculate log likelihood if site is observed
      // in the given time period
      current_log_like += ddirichletmultinomial(y.row(i), alpha.row(i), true);
    }
  }

  double hh = log(R::runif(0.0, 1.0)) + current_log_like;

  // Setup a bracket and pick a first proposal
  // Bracket whole ellipse with both edges at first proposed point
  double phi_angle = R::runif(0.0, 1.0) * 2.0 * arma::datum::pi;
  double phi_angle_min = phi_angle - 2.0 * arma::datum::pi;
  double phi_angle_max = phi_angle;

  // arma::mat alpha_ess = alpha;
  arma::mat X_ess = X_current;
  arma::vec X_center = X_current - X_prior_mean;
  arma::mat Xbs_ess = Xbs;
  arma::mat Xbs_proposal = Xbs;
  bool test = true;

  // Slice sampling loop
  while (test) {
    // compute proposal for angle difference and check to see if it is on the slice
    // adjust X so that it is mean 0 for the ellipse to be valid

    arma::vec X_proposal = X_center * cos(phi_angle) + X_prior * sin(phi_angle);
    // adjust X_proposal so that it has the proper mean
    X_proposal += X_prior_mean;
    arma::vec X_tilde = X_proposal + X_mean;
    // generate b-spline expansion of climate
    arma::mat alpha_proposal(N, d, arma::fill::zeros);

    for (int i=0; i<N; i++) {
      if (year_idx(i) == t) {
        arma::vec X_tmp(1);
        X_tmp = X_tilde(site_idx(i));
        Xbs_proposal.row(i) =  bs_cpp(X_tmp, df, knots, degree, false, rangeX);
        if (link == "log") {
          alpha_proposal.row(i) = exp(Xbs_proposal.row(i) * beta);
        } else {
          alpha_proposal.row(i) = pmax(Xbs_proposal.row(i) * beta, pow(10.0, -8.0));
        }
      }
    }

    // calculate log likelihood of proposed value
    double proposal_log_like = 0.0;
    for (int i=0; i<N; i++) {
      if (year_idx(i) == t) {
        // only calculate log likelihood if site is observed
        // in the given time period
        proposal_log_like += ddirichletmultinomial(y.row(i), alpha_proposal.row(i), true);
      }
    }
    // control to limit alpha from getting unreasonably large
    if (alpha_proposal.max() > pow(10.0, 10.0) ) {
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
        // alpha_ess = alpha_proposal;
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
      _["X"] = X_ess,
      _["Xbs"] = Xbs_ess));
  //,
  // _["alpha"] = alpha_ess));
}

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// MCMC Loop //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
List mcmcRcppSpatial (const arma::mat& Y_calibrate,
                      const arma::mat& Y,
                      const arma::vec& X,
                      List params,
                      const arma::mat& W,
                      const double mu_X, // mean of calibration climate
                      const double s2_X, // sd of calibration climate
                      const arma::mat& locs,
                      const arma::vec& site_idx,
                      const int& n_sites,
                      const int& tt,
                      const arma::vec& year_idx,
                      const Rcpp::List samples,
                      const Rcpp::List priors,
                      std::string link = "log",
                      int n_chain = 1,
                      std::string file_name = "fit") {

  // Load parameters
  int n_adapt = as<int>(params["n_adapt"]);
  int n_mcmc = as<int>(params["n_mcmc"]);
  int n_thin = as<int>(params["n_thin"]);

  // defaults b-spline order
  int degree = 3;
  if (priors.containsElementNamed("degree")) {
    degree = as<int>(priors["degree"]);
  }
  // default b-spline degrees of freedom
  int df = 6;
  if (priors.containsElementNamed("df")) {
    df = as<int>(priors["df"]);
  }

  // set up dimensions
  const int N_calibrate = Y_calibrate.n_rows;
  const int N = Y.n_rows;
  const int d = Y.n_cols;
  const int q = W.n_cols;


  // count - sum of counts at each site
  arma::vec count(N);
  for (int i=0; i<N; i++) {
    count(i) = sum(Y.row(i));
  }
  arma::vec count_calibrate(N);
  for (int i=0; i<N; i++) {
    count_calibrate(i) = sum(Y_calibrate.row(i));
  }


  // default to message output every 500 iterations
  int message = 500;

  // constant vectors
  arma::vec zero_df(df, arma::fill::zeros);
  arma::vec zero_q(q, arma::fill::zeros);
  // if (params.containsElementNamed("message")) {
  //   message = as<int>(params["message"]);
  // }

  // calculate total pollen counts per sample
  // arma::mat Ni(N, tt);
  // for (int i=0; i<N; i++) {
  //   for (int t=0; t<tt; t++) {
  //     Ni(i, t) = as_scalar(accu(y(span(i), span(), span(t))));
  //   }
  // }



  // initialize values



  // default half-cauchy prior for spatial range phi
  // might want to change to a uniform prior laters
  double s2_phi = 1000000.0;
  if (params.containsElementNamed("s2_phi")) {
    s2_phi = as<double>(params["s2_phi"]);
  }

  // initialize spatial range phi
  double phi = R::runif(0.0, 100.0);
  if (params.containsElementNamed("phi")) {
    phi = as<double>(params["phi"]);
  }
  bool sample_phi = true;
  if (params.containsElementNamed("sample_phi")) {
    sample_phi = as<bool>(params["sample_phi"]);
  }

  // default uniform (-1, 1) prior for rho
  // initialize rho
  double rho = R::runif(-1.0, 1.0);
  if (params.containsElementNamed("rho")) {
    rho = as<double>(params["rho"]);
  }
  double rho_tilde = (rho + 1.0) / 2.0;
  double rho2 = pow(rho, 2.0);

  bool sample_rho = true;
  if (params.containsElementNamed("sample_rho")) {
    sample_rho = as<bool>(params["sample_rho"]);
  }

  // default half-cauchy prior for spatial sill tau
  double s2_tau = 10.0;
  if (params.containsElementNamed("s2_tau")) {
    s2_tau = as<double>(params["s2_tau"]);
  }

  // initialize spatial sill tau
  double tau = R::runif(0.0, 5.0);
  if (params.containsElementNamed("tau")) {
    tau = as<double>(params["tau"]);
  }
  bool sample_tau = true;
  if (params.containsElementNamed("sample_tau")) {
    sample_tau = as<bool>(params["sample_tau"]);
  }


  // set up spatial autocorrelation structure
  arma::mat D = makeDistARMA(locs, locs);
  arma::mat Omega = exp( - D / phi);
  arma::mat R = chol(Omega);
  arma::mat Sigma_chol = tau * R;

  // initialize fixed effects coefficients gamma
  // default gamma tuning parameter
  double lambda_gamma_tune = 1.0;
  lambda_gamma_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_gamma_tune")) {
    lambda_gamma_tune = as<double>(params["lambda_gamma_tune"]);
  }

  // default normal prior for fixed regression coefficient means
  arma::vec mu_gamma(q, arma::fill::zeros);
  if (params.containsElementNamed("mu_gamma")) {
    mu_gamma = as<vec>(params["mu_gamma"]);
  }
  // default prior for fixed regression coefficient variance
  arma::mat Sigma_gamma(q, q, arma::fill::eye);
  Sigma_gamma *= 25.0;
  if (params.containsElementNamed("Sigma_gamma")) {
    Sigma_gamma = as<mat>(params["Sigma_gamma"]);
  }
  arma::mat Sigma_gamma_inv = inv_sympd(Sigma_gamma);
  arma::mat Sigma_gamma_chol = chol(Sigma_gamma);
  // initialize fixed effects coefficients
  arma::vec gamma = mvrnormArmaVec(mu_gamma, Sigma_gamma);
  if (params.containsElementNamed("gamma")) {
    gamma = as<vec>(params["gamma"]);
  }
  bool sample_gamma = true;
  if (params.containsElementNamed("sample_gamma")) {
    sample_gamma = as<bool>(params["sample_gamma"]);
  }

  // initialzed fixed effects mean
  arma::vec X_mean = W * gamma;
  arma::mat X_mean_mat(n_sites, tt, arma::fill::zeros);
  for (int t=0; t<tt; t++){
    X_mean_mat.col(t) = X_mean;
  }

  // initialize functional covariates beta
  // default beta tuning parameter
  arma::vec lambda_beta_tune(d, arma::fill::ones);
  lambda_beta_tune *= 1.0 / pow(3.0, 0.8);
  if (params.containsElementNamed("lambda_beta_tune")) {
    lambda_beta_tune = as<vec>(params["lambda_beta_tune"]);
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

  // Default for beta
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


  // proposal using generative AR1 process
  // X_pred.col(0) = rho * X + sqrt(1.0 - rho2) * Sigma_chol * randn(N);
  // for (int t=1; t<tt; t++) {
  //   X_pred.col(t) = rho * X_pred.col(t-1) +
  //     sqrt(1.0 - rho2) * Sigma_chol * randn(N);
  // }
  // proposal using prior full conditional distribution
  arma::mat X_pred(n_sites, tt, arma::fill::zeros);
  X_pred.col(0) = rho / (1.0 + rho2) * (X - X_mean + X_pred.col(1)) +
    1.0 / sqrt(1.0 + rho2) * Sigma_chol * randn(n_sites);
  for (int t=1; t<(tt-1); t++) {
    X_pred.col(t) = rho / (1.0 + rho2) * (X_pred.col(t-1) + X_pred.col(t+1)) +
      1.0 / sqrt(1.0 + rho2) * Sigma_chol * randn(n_sites);
  }
  X_pred.col(tt-1) = rho * (X_pred.col(tt-2)) +
    Sigma_chol * randn(n_sites);

  // set up basis expansion
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
  arma::mat Xbs = make_Xbs(X_pred, X_mean, site_idx, year_idx,
                           knots, rangeX, N, df, degree);
  arma::mat Xbs_calibrate = make_Xbs_calibrate(X, site_idx, year_idx,
                                               knots, rangeX, N, df, degree);
  arma::mat alpha = make_alpha(Xbs, beta, link);
  arma::mat alpha_calibrate = make_alpha(Xbs_calibrate, beta, link);

  // setup save variables
  int n_save = n_mcmc / n_thin;

  arma::cube beta_save(n_save, df, d, arma::fill::zeros);
  arma::mat gamma_save(n_save, q, arma::fill::zeros);
  arma::vec phi_save(n_save, arma::fill::zeros);
  arma::vec rho_save(n_save, arma::fill::zeros);
  arma::vec tau_save(n_save, arma::fill::zeros);
  arma::mat mu_beta_save(n_save, df, arma::fill::zeros);
  arma::cube Sigma_beta_save(n_save, df, df, arma::fill::zeros);
  arma::cube alpha_save(n_save, N, d, arma::fill::zeros);
  arma::cube alpha_calibrate_save(n_save, N, d, arma::fill::zeros);
  arma::cube X_save(n_save, n_sites, tt, arma::fill::zeros);



  // initialize tuning
  // tuning for functional coefficients beta
  arma::vec beta_accept(d, arma::fill::zeros);
  arma::vec beta_accept_batch(d, arma::fill::zeros);
  arma::cube beta_batch(50, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune(df, df, d, arma::fill::zeros);
  arma::cube Sigma_beta_tune_chol(df, df, d, arma::fill::zeros);
  for(int j=0; j<d; j++) {
    Sigma_beta_tune.slice(j).eye();
    Sigma_beta_tune_chol.slice(j) = chol(Sigma_beta_tune.slice(j));
  }
  // tuning for fixed effect coefficients gamma
  double gamma_accept = 0.0;
  double gamma_accept_batch = 0.0;
  arma::mat gamma_batch(50, q, arma::fill::zeros);
  arma::mat Sigma_gamma_tune(q, q, arma::fill::eye);
  arma::mat Sigma_gamma_tune_chol = chol(Sigma_gamma_tune);

  // tuning for spatial range
  double phi_accept = 0.0;
  double phi_accept_batch = 0.0;
  double phi_tune = 1.0;
  // tuning for spatial sill
  double tau_accept = 0.0;
  double tau_accept_batch = 0.0;
  double tau_tune = 1.0;
  // tuning for autocorrelation rho
  double rho_accept = 0.0;
  double rho_accept_batch = 0.0;
  double rho_tune = 1.0;

  // adaptation phase

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
    // sample beta using block MH
    //     (only for the calibration data - can try using full latent states later...)
    //

    if (sample_beta) {
      for (int j=0; j<d; j++) {
        arma::mat beta_star = beta;
        beta_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));
        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha_calibrate_star = make_alpha(Xbs_calibrate, beta_star, link);
        double mh1 = LL_DM(alpha_calibrate_star, Y_calibrate, N_calibrate, d, count_calibrate) +
          dMVN(beta_star.col(j), mu_beta, Sigma_beta_chol);
        double mh2 = LL_DM(alpha_calibrate, Y_calibrate, N_calibrate, d, count_calibrate) +
          dMVN(beta.col(j), mu_beta, Sigma_beta_chol);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha_calibrate = alpha_calibrate_star;
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
    // update the historical alpha using the calibration functional relationships
    alpha = make_alpha(Xbs, beta, link);

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
    // sample gamma using block MH
    //

    if (sample_gamma) {
      arma::vec gamma_star = gamma;
      gamma_star +=
        mvrnormArmaVecChol(zero_q, lambda_gamma_tune * Sigma_gamma_tune_chol);
      arma::vec X_mean_star = W * gamma_star;
      arma::mat X_mean_mat_star(n_sites, tt, arma::fill::zeros);
      for (int t=0; t<tt; t++){
        X_mean_mat_star.col(t) = X_mean_star;
      }
      arma::mat Xbs_star = make_Xbs(X_pred, X_mean_star, site_idx, year_idx,
                                    knots, rangeX, N, df, degree);
      arma::mat alpha_star = make_alpha(Xbs_star, beta, link);

      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        dMVN(X_pred.col(0), rho * (X - X_mean_star), Sigma_chol, true) +
        dMVN(gamma_star, mu_gamma, Sigma_gamma_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true) +
        dMVN(gamma, mu_gamma, Sigma_gamma_chol);
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0, 1.0)) {
        gamma = gamma_star;
        X_mean = X_mean_star;
        X_mean_mat = X_mean_mat_star;
        Xbs = Xbs_star;
        alpha = alpha_star;
        gamma_accept_batch += 1.0 / 50.0;
      }
      // update tuning
      gamma_batch.row(k % 50) = gamma.t();
      if ((k+1) % 50 == 0){
        updateTuningMV(k, gamma_accept_batch, lambda_gamma_tune,
                       gamma_batch, Sigma_gamma_tune,
                       Sigma_gamma_tune_chol);
      }
    }

    //
    // sample rho - MH
    //

    if (sample_rho) {
      double logit_rho_tilde_star = logit_double(rho_tilde);
      logit_rho_tilde_star += R::rnorm(0.0, rho_tune);
      double rho_tilde_star = expit_double(logit_rho_tilde_star);
      double rho_star = 2.0 * rho_tilde_star - 1.0;
      double rho2_star = pow(rho_star, 2.0);
      if (rho_star > -1.0 && rho_star < 1.0) {
        double mh1 = 0.0 + //uniform prior
          log(rho_tilde_star) + log(1.0 - rho_tilde_star);  // logit jacobian adjustment
        double mh2 = 0.0 +
          log(rho_tilde) + log(1.0 - rho_tilde);  // logit jacobian adjustment
        for (int t=0; t<tt; t++) {
          // proposal using generative AR1 process
          if (t == 0) {
            mh1 += dMVN(X_pred.col(0), rho_star * (X - X_mean), Sigma_chol, true);
            mh2 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true);
          } else {
            mh1 += dMVN(X_pred.col(t), rho_star * X_pred.col(t-1), Sigma_chol, true);
            mh2 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          rho_tilde = rho_tilde_star;
          rho = rho_star;
          rho2 = rho2_star;
          rho_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, rho_accept_batch, rho_tune);
      }
    }

    //
    // sample tau
    //

    if (sample_tau) {
      double log_tau_star = log(tau);
      log_tau_star += R::rnorm(0.0, tau_tune);
      double tau_star = exp(log_tau_star);
      if (tau_star > 0.0) {
        arma::mat Sigma_chol_star = tau_star * R;
        double mh1 = log_tau_star + // jacobian of log-scale proposal
          d_half_cauchy(tau_star, s2_tau, true);
        double mh2 = log(tau) + // jacobian of log-scale proposal
          d_half_cauchy(tau, s2_tau, true);
        for (int t=0; t<tt; t++) {
          // proposal using generative AR1 process
          if (t == 0) {
            mh1 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true);
          } else {
            mh1 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau = tau_star;
          Sigma_chol = Sigma_chol_star;
          tau_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, tau_accept_batch, tau_tune);
      }
    }

    //
    // sample phi
    //

    if (sample_phi) {
      double log_phi_star = log(phi);
      log_phi_star += R::rnorm(0.0, phi_tune);
      double phi_star = exp(log_phi_star);
      if (phi_star > 0.0) {
        arma::mat Omega_star = exp( - D / phi_star);
        arma::mat R_star = chol(Omega_star);
        arma::mat Sigma_chol_star = tau * R_star;
        double mh1 = log_phi_star + // jacobian of log-scale proposal
          d_half_cauchy(phi_star, s2_phi, true);
        double mh2 = log(phi) + // jacobian of log-scale proposal
          d_half_cauchy(phi, s2_phi, true);
        for (int t=0; t<tt; t++) {
          // proposal using generative AR1 process
          if (t == 0) {
            mh1 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true);
          } else {
            mh1 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          phi = phi_star;
          Omega = Omega_star;
          R = R_star;
          Sigma_chol = Sigma_chol_star;
          phi_accept_batch += 1.0 / 50.0;
        }
      }
      // update tuning
      if ((k+1) % 50 == 0){
        updateTuning(k, phi_accept_batch, phi_tune);
      }
    }


    //
    // sample X
    //

    for (int t=0; t<tt; t++) {
      // arma::vec X_pred_mean(n_sites);
      arma::vec X_prior(n_sites);
      arma::vec X_prior_mean(n_sites);
      // proposal using generative AR1 process
      if (t == 0) {
        X_prior_mean = rho / (1.0 + rho2) * (X - X_mean + X_pred.col(1));
        // X_prior =  1.0 / sqrt(1.0 + rho2) * Sigma_chol.t() * randn(n_sites);
        X_prior =  1.0 / sqrt(1.0 + rho2) * Sigma_chol * randn(n_sites);
      } else if (t == (tt-1)) {
        X_prior_mean = rho * X_pred.col(t-1);
        // X_prior = Sigma_chol.t() * randn(n_sites);
        X_prior = Sigma_chol * randn(n_sites);
      } else {
        X_prior_mean = rho / (1.0 + rho2) * (X_pred.col(t-1) + X_pred.col(t+1));
        // X_prior = 1.0 / sqrt(1.0 + rho2) * Sigma_chol.t() * randn(n_sites);
        X_prior = 1.0 / sqrt(1.0 + rho2) * Sigma_chol * randn(n_sites);
      }

      // Rcpp::List ess_out = ess(y, X_pred.col(t), X_prior, X_pred_mean, beta, t, site_idx,
      //                          year_idx, link, file_name, n_chain);
      Rcpp::List ess_out = ess(Y, X_pred.col(t), X_prior, X_prior_mean,
                               X_mean, Xbs, knots, degree, df,
                               rangeX, beta, t, site_idx, year_idx,
                               link, file_name, n_chain);
      // y is an N by d matrix of fossil pollen
      X_pred.col(t) = as<vec>(ess_out["X"]);
      Xbs = as<mat>(ess_out["Xbs"]);
      // alpha.slice(t) = as<mat>(ess_out["alpha"]);


      // end of adaptation loop
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
    // sample beta using block MH
    //     (only for the calibration data - can try using full latent states later...)
    //

    if (sample_beta) {
      for (int j=0; j<d; j++) {
        arma::mat beta_star = beta;
        beta_star.col(j) +=
          mvrnormArmaVecChol(zero_df,
                             lambda_beta_tune(j) * Sigma_beta_tune_chol.slice(j));
        // arma::mat alpha_star = exp(mu_mat + Xbs * beta_star);
        arma::mat alpha_calibrate_star = make_alpha(Xbs_calibrate, beta_star, link);
        double mh1 = LL_DM(alpha_calibrate_star, Y_calibrate, N_calibrate, d, count_calibrate) +
          dMVN(beta_star.col(j), mu_beta, Sigma_beta_chol);
        double mh2 = LL_DM(alpha_calibrate, Y_calibrate, N_calibrate, d, count_calibrate) +
          dMVN(beta.col(j), mu_beta, Sigma_beta_chol);
        double mh = exp(mh1 - mh2);
        if (mh > R::runif(0, 1.0)) {
          beta = beta_star;
          alpha_calibrate = alpha_calibrate_star;
          beta_accept += 1.0 / n_mcmc;
        }
      }
    }
    // update the historical alpha using the calibration functional relationships
    alpha = make_alpha(Xbs, beta, link);

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
    // sample gamma using block MH
    //

    if (sample_gamma) {
      arma::vec gamma_star = gamma;
      gamma_star +=
        mvrnormArmaVecChol(zero_q, lambda_gamma_tune * Sigma_gamma_tune_chol);
      arma::vec X_mean_star = W * gamma_star;
      arma::mat X_mean_mat_star(n_sites, tt, arma::fill::zeros);
      for (int t=0; t<tt; t++){
        X_mean_mat_star.col(t) = X_mean_star;
      }
      arma::mat Xbs_star = make_Xbs(X_pred, X_mean_star, site_idx, year_idx,
                                    knots, rangeX, N, df, degree);
      arma::mat alpha_star = make_alpha(Xbs_star, beta, link);

      double mh1 = LL_DM(alpha_star, Y, N, d, count) +
        dMVN(X_pred.col(0), rho * (X - X_mean_star), Sigma_chol, true) +
        dMVN(gamma_star, mu_gamma, Sigma_gamma_chol);
      double mh2 = LL_DM(alpha, Y, N, d, count) +
        dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true) +
        dMVN(gamma, mu_gamma, Sigma_gamma_chol);
      double mh = exp(mh1 - mh2);
      if (mh > R::runif(0, 1.0)) {
        gamma = gamma_star;
        X_mean = X_mean_star;
        X_mean_mat = X_mean_mat_star;
        Xbs = Xbs_star;
        alpha = alpha_star;
        gamma_accept += 1.0 / n_mcmc;
      }
    }


    //
    // sample rho - MH
    //

    if (sample_rho) {
      double logit_rho_tilde_star = logit_double(rho_tilde);
      logit_rho_tilde_star += R::rnorm(0.0, rho_tune);
      double rho_tilde_star = expit_double(logit_rho_tilde_star);
      double rho_star = 2.0 * rho_tilde_star - 1.0;
      double rho2_star = pow(rho_star, 2.0);
      if (rho_star > -1.0 && rho_star < 1.0) {
        double mh1 = 0.0 + //uniform prior
          log(rho_tilde_star) + log(1.0 - rho_tilde_star);  // logit jacobian adjustment
        double mh2 = 0.0 +
          log(rho_tilde) + log(1.0 - rho_tilde);  // logit jacobian adjustment
        for (int t=0; t<tt; t++) {
          // proposal using generative AR1 process
          if (t == 0) {
            mh1 += dMVN(X_pred.col(0), rho_star * (X - X_mean), Sigma_chol, true);
            mh2 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true);
          } else {
            mh1 += dMVN(X_pred.col(t), rho_star * X_pred.col(t-1), Sigma_chol, true);
            mh2 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          rho_tilde = rho_tilde_star;
          rho = rho_star;
          rho2 = rho2_star;
          rho_accept_batch += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample tau
    //

    if (sample_tau) {
      double log_tau_star = log(tau);
      log_tau_star += R::rnorm(0.0, tau_tune);
      double tau_star = exp(log_tau_star);
      if (tau_star > 0.0) {
        arma::mat Sigma_chol_star = tau_star * R;
        double mh1 = log_tau_star + // jacobian of log-scale proposal
          d_half_cauchy(tau_star, s2_tau, true);
        double mh2 = log(tau) + // jacobian of log-scale proposal
          d_half_cauchy(tau, s2_tau, true);
        for (int t=0; t<tt; t++) {
          // proposal using generative AR1 process
          if (t == 0) {
            mh1 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true);
          } else {
            mh1 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          tau = tau_star;
          Sigma_chol = Sigma_chol_star;
          tau_accept += 1.0 / n_mcmc;
        }
      }
    }

    //
    // sample phi
    //

    if (sample_phi) {
      double log_phi_star = log(phi);
      log_phi_star += R::rnorm(0.0, phi_tune);
      double phi_star = exp(log_phi_star);
      if (phi_star > 0.0) {
        arma::mat Omega_star = exp( - D / phi_star);
        arma::mat R_star = chol(Omega_star);
        arma::mat Sigma_chol_star = tau * R_star;
        double mh1 = log_phi_star + // jacobian of log-scale proposal
          d_half_cauchy(phi_star, s2_phi, true);
        double mh2 = log(phi) + // jacobian of log-scale proposal
          d_half_cauchy(phi, s2_phi, true);
        for (int t=0; t<tt; t++) {
          // proposal using generative AR1 process
          if (t == 0) {
            mh1 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(0), rho * (X - X_mean), Sigma_chol, true);
          } else {
            mh1 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol_star, true);
            mh2 += dMVN(X_pred.col(t), rho * X_pred.col(t-1), Sigma_chol, true);
          }
        }
        double mh = exp(mh1-mh2);
        if (mh > R::runif(0.0, 1.0)) {
          phi = phi_star;
          Omega = Omega_star;
          R = R_star;
          Sigma_chol = Sigma_chol_star;
          phi_accept += 1.0 / n_mcmc;
        }
      }
    }


    //
    // sample X
    //

    for (int t=0; t<tt; t++) {
      // arma::vec X_pred_mean(n_sites);
      arma::vec X_prior(n_sites);
      arma::vec X_prior_mean(n_sites);
      // proposal using generative AR1 process
      if (t == 0) {
        X_prior_mean = rho / (1.0 + rho2) * (X - X_mean + X_pred.col(1));
        // X_prior =  1.0 / sqrt(1.0 + rho2) * Sigma_chol.t() * randn(n_sites);
        X_prior =  1.0 / sqrt(1.0 + rho2) * Sigma_chol * randn(n_sites);
      } else if (t == (tt-1)) {
        X_prior_mean = rho * X_pred.col(t-1);
        // X_prior = Sigma_chol.t() * randn(n_sites);
        X_prior = Sigma_chol * randn(n_sites);
      } else {
        X_prior_mean = rho / (1.0 + rho2) * (X_pred.col(t-1) + X_pred.col(t+1));
        // X_prior = 1.0 / sqrt(1.0 + rho2) * Sigma_chol.t() * randn(n_sites);
        X_prior = 1.0 / sqrt(1.0 + rho2) * Sigma_chol * randn(n_sites);
      }

      // Rcpp::List ess_out = ess(y, X_pred.col(t), X_prior, X_pred_mean, beta, t, site_idx,
      //                          year_idx, link, file_name, n_chain);
      Rcpp::List ess_out = ess(Y, X_pred.col(t), X_prior, X_prior_mean,
                               X_mean, Xbs, knots, degree, df,
                               rangeX, beta, t, site_idx, year_idx,
                               link, file_name, n_chain);
      // y is an N by d matrix of fossil pollen
      X_pred.col(t) = as<vec>(ess_out["X"]);
      Xbs = as<mat>(ess_out["Xbs"]);
      // alpha.slice(t) = as<mat>(ess_out["alpha"]);
    }

    //
    // save variables
    //

    if ((k + 1) % n_thin == 0) {
      int save_idx = (k+1)/n_thin-1;
      X_save(span(save_idx), span(), span()) = X_pred + X_mean_mat;
      alpha_save.subcube(span(save_idx), span(), span()) = alpha;
      alpha_calibrate_save.subcube(span(save_idx), span(), span()) = alpha_calibrate;
      beta_save.subcube(span(save_idx), span(), span()) = beta;
      gamma_save.row(save_idx) = gamma.t();
      phi_save(save_idx) = phi;
      rho_save(save_idx) = rho;
      tau_save(save_idx) = tau;
      mu_beta_save.row(save_idx) = mu_beta.t();
      Sigma_beta_save.subcube(span(save_idx), span(), span()) = Sigma_beta;

    }

    // end MCMC loop
  }
  // print accpetance rates
  // set up output messages
  file_out.open(file_name, std::ios_base::app);
  if (sample_beta) {
    file_out << "Average acceptance rate for beta  = " << mean(beta_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_gamma) {
    file_out << "Average acceptance rate for gamma  = " << mean(gamma_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_phi) {
    file_out << "Average acceptance rate for phi  = " << mean(phi_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_rho) {
    file_out << "Average acceptance rate for rho  = " << mean(rho_accept) <<
      " for chain " << n_chain << "\n";
  }
  if (sample_tau) {
    file_out << "Average acceptance rate for tau  = " << mean(tau_accept) <<
      " for chain " << n_chain << "\n";
  }

  // close output file
  file_out.close();

  // output results

  return Rcpp::List::create(
    _["alpha"] = alpha_save,
    _["alpha_calibrate"] = alpha_calibrate_save,
    _["beta"] = beta_save,
    _["X"] = X_save,
    _["rho"] = rho_save,
    _["phi"] = phi_save,
    _["tau"] = tau_save,
    _["mu_beta"] = mu_beta_save,
    _["Sigma_beta"] = Sigma_beta_save);
  //,
  // _["alpha"] = alpha_save);
}
