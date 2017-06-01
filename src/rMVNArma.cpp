#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::vec rMVNArma(arma::mat& A, arma::vec& b){
  int ncols = A.n_cols;
  arma::mat A_chol(ncols, ncols);
  bool success = false;
  int counter = 0;
  while(success == false && counter < 100){
    success = arma::chol(A_chol, A);
    if(success == false){
      counter++;
      A += arma::mat(ncols, ncols, arma::fill::eye) * 1e-6;
    }
  }
	arma::vec devs = arma::randn(ncols);
	arma::vec temp = solve(trimatl(A_chol.t()), b);
	return arma::vec(solve(trimatu(A_chol), temp + devs));
}

//[[Rcpp::export]]
double rMVNArmaScalar(const double & a, const double & b) {
  double a_inv = 1.0 / a;
  double z = R::rnorm(0, 1);
  return(b * a_inv + z * sqrt(a_inv));
}
