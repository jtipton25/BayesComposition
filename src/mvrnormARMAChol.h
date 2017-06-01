#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

arma::mat mvrnormArmaChol(const int & n, const arma::vec & mu,
                      const arma::mat & Sigma_chol);

arma::vec mvrnormArmaVecChol(const arma::vec & mu, const arma::mat & Sigma_chol);
