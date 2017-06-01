#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;


arma::mat makeDistARMA(const arma::mat & coords1, const arma::mat & coords2);
