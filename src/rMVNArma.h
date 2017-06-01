#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

arma::vec rMVNArma(arma::mat& A, arma::vec& b);

double rMVNArmaScalar(const double & a, const double & b);
