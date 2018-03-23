#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

arma::mat rWishartArmaMat(const unsigned int& df, const arma::mat& S);

arma::mat rIWishartArmaMat(const unsigned int& df, const arma::mat& S);
