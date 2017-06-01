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
double d_half_cauchy(double& x, double& sigma,
                     bool logd=true) {
  if (logd) {
    return log(2.0/(M_PI*(1.0 + pow(x/sigma, 2.0)))/sigma);
  } else {
    return 2.0/(M_PI*(1.0 + pow(x/sigma, 2.0)))/sigma;
  }
}
