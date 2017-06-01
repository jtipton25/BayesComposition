#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

const double log2pi = std::log(2.0 * M_PI);

///////////////////////////////////////////////////////////////////////////////
/////////////// Gaussian density using Cholesky decomposition  ////////////////
///////////////////////////////////////////////////////////////////////////////

//[[Rcpp::export]]
double dMVNChol (const arma::vec& y, const arma::vec& mu,
                 const arma::mat& Sigma_chol, const bool logd=true){
  arma::mat rooti = trans(inv(trimatu(Sigma_chol)));
  double rootisum = sum(log(rooti.diag()));
  double constants = -(y.n_elem / 2.0) * log2pi;
  arma::vec z = rooti * (y - mu) ;
  double out = constants - 0.5 * sum(z % z) + rootisum;
  if(logd){
    return(out);
  } else {
    return(exp(out));
  }
}
