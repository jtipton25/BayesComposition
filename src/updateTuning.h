#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

///////////////////////////////////////////////////////////////////////////////
///////////// Adaptively updates tuning parameters in a MCMC chain ////////////
///////////////////////////////////////////////////////////////////////////////

void updateTuning(const int k, double& accept_tmp, double& tune) ;

void updateTuningVec(const int k, arma::vec& accept_tmp, arma::vec& tune);

void updateTuningMat(const int k, arma::mat& accept_tmp, arma::mat& tune);

void updateTuningMV(const int& k, double& accept_rate, double& lambda,
                     arma::mat& batch_samples, arma::mat& Sigma_tune,
                     arma::mat Sigma_tune_chol);

void updateTuningMVMat(const int& k, arma::vec& accept_rate,
                       arma::vec& lambda, arma::cube& batch_samples,
                       arma::cube& Sigma_tune,
                       arma::cube Sigma_tune_chol);
