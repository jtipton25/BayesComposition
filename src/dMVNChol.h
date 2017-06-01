// #define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

///////////////////////////////////////////////////////////////////////////////
/////////////// Gaussian density using Cholesky decomposition  ////////////////
////////////////////////////////////////////////////////////////////////////

double dMVNChol (const mat& y, const vec& mu,
                 const mat& Sigma_chol, const bool logd=true);
