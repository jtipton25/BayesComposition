#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::mat mvrnormArmaChol(const int & n, const arma::vec & mu,
                      const arma::mat & Sigma_chol) {
   int ncols = Sigma_chol.n_cols;
   arma::mat Z = arma::randn(n, ncols);
   return repmat(mu, 1, n).t() + Z * Sigma_chol;
}

// [[Rcpp::export]]
arma::vec mvrnormArmaVecChol(const arma::vec & mu, const arma::mat & Sigma_chol) {
   int ncols = Sigma_chol.n_cols;
   arma::vec z = arma::randn(ncols);
   return mu + Sigma_chol * z;
}
