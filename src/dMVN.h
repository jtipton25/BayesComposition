#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

///////////////////////////////////////////////////////////////////////////////
//////// MVN density using Cholesky decomposition of Covariance Sigma /////////
///////////////////////////////////////////////////////////////////////////////

double dMVN (const mat& y, const vec& mu,
             const mat& Sigma_chol, const bool logd=true);
