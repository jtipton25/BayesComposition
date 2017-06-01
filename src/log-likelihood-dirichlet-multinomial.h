#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]


using namespace Rcpp;
using namespace arma;

///////////////////////////////////////////////////////////////////////////////
/// A function for calculating the log-likelihood of a Dirichlet-Multinomial //
// distribution over all observations /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

 double LL_DM (const arma::mat& alpha, const arma::mat& Y, const double& N,
              const double& d, const arma::vec& count);

 ///////////////////////////////////////////////////////////////////////////////
 /////////// Dirichlet Multinomial Log-Likelihood for a single row /////////////
 ///////////////////////////////////////////////////////////////////////////////

 double LL_DM_row (const arma::rowvec& alpha, const arma::rowvec& Y,
                   const double& d, const double& count);
