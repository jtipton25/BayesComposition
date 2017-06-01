#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

//' Makes a conditional autoregressive (CAR) precision matrix
//'
//' @param theta The CAR parameter
//' @param t The dimension of the CAR matrix
//'
//' @export
//[[Rcpp::export]]
arma::mat makeQinv(const double& theta, const int& t){ // CAR precision matrix
  arma::mat out(t, t, fill::eye);
	arma::vec toep_vec(t, fill::zeros);
	toep_vec(1) = 1;
	arma::vec diag_vec(t, fill::zeros);
	for(int i = 1; i < (t - 1); i++){
		diag_vec(i) = pow(theta, 2);
	}
	out.diag() += diag_vec;
	return(out - theta * arma::toeplitz(toep_vec));
}
