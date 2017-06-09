#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

// //' Adaptively updates tuning parameters in a MCMC chain
// //'
// //' @param k The MCMC iteration
// //' @param accept_tmp A double or numeric vector that gives the acceptance rate over the last 50 iterations
// //' @param tune A double or numeric vector that is the current random walk tuning standard deviations
// //' @param x A numeric vector of the current state to be sampled
// //' @param mu A numeric vector that is the running mean of the MCMC samples
// //' @param Sig A numeric matrix that is the running covariance of the MCMC samples
// //' @export

// [[Rcpp::export]]
void updateTuning(const int k, double& accept_tmp, double& tune) {
  double delta = 1.0 / sqrt(k);
  if(accept_tmp > 0.44){
    tune = exp(log(tune) + delta);
  } else {
    tune = exp(log(tune) - delta);
  }
  accept_tmp = 0.0;
}

// [[Rcpp::export]]
void updateTuningVec(const int k, arma::vec& accept_tmp, arma::vec& tune) {
  double delta = 1.0 / sqrt(k);
  int n = tune.n_elem;
  for(int i = 0; i < n; i++){
    if(accept_tmp(i) > 0.44){
      tune(i) = exp(log(tune(i)) + delta);
    } else {
      tune(i) = exp(log(tune(i)) - delta);
    }
    accept_tmp(i) = 0.0;
  }
}

// [[Rcpp::export]]
void updateTuningMat(const int k, arma::mat& accept_tmp, arma::mat& tune) {
  double delta = 1.0 / sqrt(k);
  int n = tune.n_rows;
  int p = tune.n_cols;
  for(int i = 0; i < n; i++){
    for(int j = 0; j < p; j++){
      if(accept_tmp(i, j) > 0.44){
        tune(i, j) = exp(log(tune(i, j)) + delta);
      } else {
        tune(i, j) = exp(log(tune(i, j)) - delta);
      }
      accept_tmp(i, j) = 0.0;
    }
  }
}

// [[Rcpp::export]]
void updateTuningMV(const int& k, double& accept_rate, double& lambda,
                     arma::mat& batch_samples, arma::mat& Sigma_tune,
                     arma::mat Sigma_tune_chol) {
  static const double arr[] = {0.44, 0.35, 0.32, 0.25, 0.234};
  std::vector<double> acceptance_rates (arr, arr + sizeof(arr) / sizeof(arr[0]));
  double dimension = batch_samples.n_rows - 1.0;
  if (dimension >= 5) {
    dimension = 4;
  }
  double d = batch_samples.n_cols;
  double batch_size = batch_samples.n_rows;
  double optimal_accept = acceptance_rates[dimension];
  double times_adapted = floor(k / 50);
  //
  double gamma1 = 1.0 / (pow(times_adapted + 3.0, 0.8));
  double gamma2 = 10.0 * gamma1;
  double adapt_factor = exp(gamma2 * (accept_rate - optimal_accept));
  // update the MV scaling parameter
  lambda *= adapt_factor;
  // center the batch of MCMC samples
  for (int j=0; j<d; j++) {
    double mean_batch = mean(batch_samples.col(j));
    for (int i=0; i<batch_size; i++) {
      batch_samples(i, j) -= mean_batch;
    }
  }
  // 50 is an MCMC batch size, can make this function more general later...
  Sigma_tune += gamma1 *
    (batch_samples.t() * batch_samples / (50.0-1.0) - Sigma_tune);
  Sigma_tune_chol = chol(Sigma_tune);
  accept_rate = 0.0;
  batch_samples.zeros();
}

// [[Rcpp::export]]
void updateTuningMVMat(const int& k, arma::vec& accept_rate,
                       arma::vec& lambda, arma::cube& batch_samples,
                       arma::cube& Sigma_tune,
                       arma::cube Sigma_tune_chol) {
  static const double arr[] = {0.44, 0.35, 0.32, 0.25, 0.234};
  std::vector<double> acceptance_rates (arr, arr + sizeof(arr) / sizeof(arr[0]));
  double dimension = batch_samples.n_rows - 1.0;
  if (dimension >= 5) {
    dimension = 4;
  }
  double d = batch_samples.n_cols;
  double p = batch_samples.n_slices;
  double batch_size = batch_samples.n_rows;
  double optimal_accept = acceptance_rates[dimension];
  double times_adapted = floor(k / 50);
  //
  double gamma1 = 1.0 / (pow(times_adapted + 3.0, 0.8));
  double gamma2 = 10.0 * gamma1;
  arma::vec adapt_factor = exp(gamma2 * (accept_rate - optimal_accept));
  // update the MV scaling parameter
  lambda %= adapt_factor;
  // center the batch of MCMC samples
  for (int l=0; l<p; l++) {
    for (int j=0; j<d; j++) {
      arma::vec batch_vec = batch_samples.subcube(0, j, l, batch_size-1, j, l);
      double mean_batch = mean(batch_vec);
      for (int i=0; i<batch_size; i++) {
        batch_samples(i, j, l) -= mean_batch;
      }
    }
    Sigma_tune.slice(l) += gamma1 *
      (batch_samples.slice(l).t() * batch_samples.slice(l) / (50.0-1.0) -
      Sigma_tune.slice(l));
    Sigma_tune_chol.slice(l) = chol(Sigma_tune.slice(l));
  }
  // 50 is an MCMC batch size, can make this function more general later...
  accept_rate.zeros();
  batch_samples.zeros();
}
