#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

//' A function for calculating the logit function
//'
//' @param \code{phi} A \code{vector} on the real line
//'
//' @return \code{logit()} returns the logit transform on \code{phi}
//' @export
// [[Rcpp::export]]
arma::vec logit(const arma::vec& phi) {
  double n = phi.n_elem;
  arma::vec out(n);
  for (int i=0; i<n; i++) {
    out(i) = log(phi(i) / (1.0 - phi(i)));
  }
  return(out);
}

//' A function for calculating the inverse logit function
//'
//' @param \code{phi} A \code{vector} on the interval (0, 1)
//'
//' @return \code{expit()} returns the inverse logit transform on \code{phi}
//' @export
// [[Rcpp::export]]
arma::vec expit(const arma::vec& phi) {
  double n = phi.n_elem;
  arma::vec out(n);
  for (int i=0; i<n; i++) {
    out(i) = exp(phi(i)) / (1.0 + exp(phi(i)));
  }
  return(out);
}


//' A function for calculating the logit function
//'
//' @param \code{phi} A \code{double} on the real line
//'
//' @return \code{logit_double()} returns the logit transform on \code{phi}
//' @export
// [[Rcpp::export]]
double logit_double(const double& phi) {
  return(log(phi / (1.0 - phi)));
}

//' A function for calculating the inverse logit function
//'
//' @param \code{phi} A \code{double} on the interval (0, 1)
//'
//' @return \code{expit_double()} returns the inverse logit transform on \code{phi}
//' @export
// [[Rcpp::export]]
double expit_double(double& phi) {
  return(exp(phi) / (1.0 + exp(phi)));
}
