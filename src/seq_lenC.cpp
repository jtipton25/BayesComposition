#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

//' A function for generating an evenly spaced sequence of numbers between 0 and 1
//'
//' @param \code{n} An \code{int}
//' @param \code{degree} An \code{int} that gives the degree of the B-spline polynomial
//' @param \code{i} An \code{int} giving the knot index at which to evaluate the algorithm
//' @param \code{knots} A \code{vector} that contains the knot locations
//'
//' @return \code{basis_cpp()} returns a double that is the evaluation of the \code{degree}^{th} polynomial B-spline algorithm at location \code{x} for the $i$^{th} knot index given \code{knots}
//' @export

// [[Rcpp::export]]
arma::vec seq_lenC(const int& n) {
  if (n < 2) {
    stop("Inadmissible value");
  }
  arma::vec out(n);
  out(0) = 0.0;
  out(n-1) = 1.0;
  if (n > 2) {
    double n_double = double(n-1);
    for (int i=1; i<(n-1); i++) {
      double j = double(i);
      out(i) = j / n_double;
    }
  }
  return(out);
}


