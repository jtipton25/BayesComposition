#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

const double log2pi = std::log(2.0 * M_PI);

///////////////////////////////////////////////////////////////////////////////
//////// MVN density using Cholesky decomposition of Covariance Sigma /////////
///////////////////////////////////////////////////////////////////////////////

// [[Rcpp::export]]
double dMVN (const arma::mat& y, const arma::vec& mu,
             const arma::mat& Sigma_chol, const bool logd=true){
  int n = y.n_cols;
  int ydim = y.n_rows;
  arma::vec out(n);
  arma::mat rooti = trans(inv(trimatu(Sigma_chol)));
  double rootisum = sum(log(rooti.diag()));
  double constants = - (static_cast<double>(ydim) / 2.0) * log2pi;
  for (int i=0; i<n; i++) {
    arma::vec z = rooti * (y.col(i) - mu) ;
    out(i) = constants - 0.5 * sum(z % z) + rootisum;
  }
  if(logd){
    return(sum(out));
  } else {
    return(exp(sum(out)));
  }
}
