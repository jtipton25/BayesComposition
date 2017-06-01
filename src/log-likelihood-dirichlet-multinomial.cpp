#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

//' A function for calculating the log-likelihood of a Dirichlet-Multinomial distribution over all observations
//'
//' @param \code{alpha} An \eqn{N \times d} \code{matrix} that represents the \eqn{N} realizations of the \eqn{d}-dimensional Dirichlet-multinomial parameters \eqn{alpha}
//' @param \code{Y} An \eqn{N \times d} \code{matrix} that represents the \eqn{N} observations of the \eqn{d}-dimensional Dirichlet-multinomial count data \eqn{Y}
//' @param \code{N} An \code{integer} of the number of observations
//' @param \code{d} An \code{integer} of the dimension of the process
//' @param \code{count} An \eqn{N}-dimensional \code{vector} of the number of draws for the multinomial process with one count per observtaion
//'
//' @return \code{LL_DM()} returns a double that is the evaluation of log-likelihood of the Dirichlet-multinomail distribution given the data and parameters
//' @export
//[[Rcpp::export]]
double LL_DM (const arma::mat& alpha, const arma::mat& Y, const double& N,
              const double& d, const arma::vec& count) {
  // alpha is an N by d matrix of transformed parameters
  // Y is a N by d dimensional vector of species counts
  double mh = 0.0;
  for (int i=0; i<N; i++) {
    arma::rowvec Y_row = Y.row(i);
    arma::rowvec alpha_row = alpha.row(i);
    double sumAlpha = sum(alpha_row);
    for (int j=0; j<d; j++) {
      mh += lgamma(Y_row(j) + alpha_row(j)) - lgamma(alpha_row(j));
    }
    mh += lgamma(sumAlpha) - lgamma(count(i) + sumAlpha);
  }
  return(mh);
}


///////////////////////////////////////////////////////////////////////////////
/////////// Dirichlet Multinomial Log-Likelihood for a single row /////////////
///////////////////////////////////////////////////////////////////////////////

//[[Rcpp::export]]
double LL_DM_row (const arma::rowvec& alpha, const arma::rowvec& Y,
                  const double& d, const double& count) {
  // alpha is an N by d matrix of transformed parameters
  // Y is a N by d dimensional vector of species counts
  double sumAlpha = sum(alpha);
  double mh = lgamma(sumAlpha) - lgamma(count + sumAlpha);
  for (int j=0; j<d; j++) {
    mh += lgamma(Y(j) + alpha(j)) - lgamma(alpha(j));
  }

  return(mh);
}
