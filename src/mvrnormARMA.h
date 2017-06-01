#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

arma::mat mvrnormArma(const int & n, const arma::vec & mu, const arma::mat & Sigma);

arma::vec mvrnormArmaVec(const arma::vec & mu, const arma::mat & Sigma);
