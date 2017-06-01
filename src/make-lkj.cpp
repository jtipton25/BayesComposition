#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

//' A function for turning an \code{choose(d, 2)}-vector into a d by d matrix for use in constructing an upper-triangular Cholesky decomposition of a correlation matrix
//'
//' @param \code{xi} A \code{vector} of variables bounded between -1 and 1
//' @param \code{d} An \code{int} that gives the dimension of the correlation matrix
//'
//' @return \code{makeUpperLKJ()} returns an upper-triangular matrix
//' @export
// [[Rcpp::export]]
arma::mat makeUpperLKJ(const arma::vec& x, const int& d) {
  if (d < 2) {
    stop("Inadmissible value");
  }
  arma::mat out(d, d, arma::fill::zeros);
  int idx = 0;
  for(int j=0; j<d; j++) {
    for(int i=0; i<d; i++) {
      if (i < j) {
        out(i, j) = x(idx);
        idx++;
      }
    }
  }
  return(out);
}

//' A function for generating a d by d upper-triangular Cholesky decomposition of a correlation matrix using the helper function \code{makeUpperLKJ}
//'
//' @param \code{xi} A \code{vector} of variables bounded between -1 and 1
//' @param \code{d} An \code{int} that gives the dimension of the correlation matrix
//'
//' @return \code{makeRLKJ()} returns an upper-triangular Cholesky matrix \code{R} whose inner product \code{t(R)R} is a correlation matrix \code{Omega}. If \code{cholesky} = \code{TRUE}, \code{makeRLKJ} returns the correlation matrix \code{Omega}. If \code{jacobian} = \code{TRUE}, \code{makeRLKJ} calculates the jacobian of the transformation from a correlation matrix to a Cholesky factor. In general, the Jacobian is not needed.
//' @export
// [[Rcpp::export]]
Rcpp::List makeRLKJ(const arma::vec& xi, const int& d,
                    bool cholesky=false, bool jacobian=false) {
  if (d < 2) {
    stop("Inadmissible value");
  }
  arma::mat out(d, d, arma::fill::zeros);
  arma::mat z = makeUpperLKJ(xi, d);
  double log_j = 0.0;
  if (jacobian) {
    for (int i=0; i<(d-1); i++){
      for (int j=(i+1); j<d; j++){
        log_j += (d-i-2)*log(1.0 - pow(z(i, j), 2.0));  // d-i-1 in equation 11 LKJ
        // but we start counting at 0
        // instead of 1
      }
    }
    log_j *= 0.5;
  }
  arma::mat R(d, d, arma::fill::zeros);
  arma::rowvec W(d, arma::fill::zeros);
  for(int i=0; i<d; i++) {
    for(int j=0; j<d; j++) {
      if (i == j && i == 0) {
        R(0, 0) = 1.0;
      } else if (i < j && i == 0) {
        R(0, j) = z(0, j);
      } else if (i == j) {
        arma::vec z_sub = z.submat(span(0, i-1), span(j, j));
        arma::vec ones(i, arma::fill::ones);
        double z_prod = prod(sqrt((ones - pow(z_sub, 2.0))));
        R(i, j) =  z_prod;
      } else if (i < j) {
        arma::vec z_sub = z.submat(span(0, i-1), span(j, j));
        // Rcout << z_sub << ", i=" << i << ", j=" << j << """\n";
        arma::vec ones(i, arma::fill::ones);
        R(i, j) = z(i, j) * prod(sqrt((ones - pow(z_sub, 2.0))));
      }
    }
  }
  if(cholesky) {
    return(Rcpp::List::create(
        _["R"] = R,
        _["log_jacobian"] = log_j));
  } else {
    return(Rcpp::List::create(
        _["Omega"] = R.t() * R,
        _["log_jacobian"] = log_j));
  }
}


