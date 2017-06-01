#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(cpp)]]

using namespace Rcpp;
using namespace arma;

///////////////////////////////////////////////////////////////////////////////
/// A function generating evenly spaced sequence of numbers between 0 and 1 ///
///////////////////////////////////////////////////////////////////////////////

arma::vec seq_lenC(const int& n);
