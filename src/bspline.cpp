#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

///////////////////////////////////////////////////////////////////////////////
/////////////// construct a vector along a sequence of length n ///////////////
///////////////////////////////////////////////////////////////////////////////

NumericVector seq_lenBS(const int& n) {
  if (n < 2) {
    stop("Inadmissible value");
  }
  NumericVector out(n);
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

///////////////////////////////////////////////////////////////////////////////
///////// construct a vector of percentiles for construction of knots /////////
/////////  placed evenly along the percentiles ////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

NumericVector quantileBS(NumericVector x, NumericVector probs) {
  Environment stats("package:stats");
  Function quantile = stats["quantile"];
  int npr = probs.size();
  NumericVector ans(npr);
  for(int i=0; i<npr; i++){
    ans[i] = as<double>(quantile(x, probs[i]));
  }
  return(ans);
}

//' A function for implementing the Cox-de Boor algorithm for constructing B-splines
//'
//' @param \code{x} A \code{double} that is the point at which to evaluate the Cox-de Boor algorithm
//' @param \code{degree} An \code{int} that gives the degree of the B-spline polynomial
//' @param \code{i} An \code{int} giving the knot index at which to evaluate the algorithm
//' @param \code{knots} A \code{vector} that contains the knot locations
//'
//' @return \code{basis_cpp()} returns a double that is the evaluation of the \code{degree}^{th} polynomial B-spline algorithm at location \code{x} for the $i$^{th} knot index given \code{knots}
//' @export
// [[Rcpp::export]]
double basis_cpp (const double& x, const int& degree, const int& i,
                  const arma::vec& knots) {
  double B;
  double alpha1;
  double alpha2;
  if (degree == 0) {
    if ((x >= knots(i)) && (x < knots(i+1))) {
      B = 1.0;
    } else {
      B = 0.0;
    }
  } else {
    if((knots(degree+i) - knots(i)) == 0.0) {
      alpha1 = 0.0;
    } else {
      alpha1 = (x - knots(i))/(knots(degree+i) - knots(i));
    }
    if((knots(i+degree+1) - knots(i+1)) == 0.0) {
      alpha2 = 0.0;
    } else {
      alpha2 = (knots(i+degree+1) - x)/(knots(i+degree+1) - knots(i+1));
    }
    B = alpha1*basis_cpp(x, (degree-1), i, knots) +
      alpha2*basis_cpp(x, (degree-1), (i+1), knots);
  }
  return(B);
}

//' A function for implementing the Cox-de Boor algorithm for constructing B-splines
//'
//' @param \code{x} A \code{vector} that is the data at which to evaluate the Cox-de Boor algorithm
//' @param \code{degree} An \code{int} that gives the degree of the B-spline polynomial
//' @param \code{intercept} A \code{bool} of whether or not to include an intercept term in the basis expasion
//' @param \code{Boundary_knots} A \code{vector} that contains the boundary knot locations
//'
//' @return \code{bs_cpp()} returns a double that is the evaluation of the \code{degree}^{th} polynomial B-spline algorithm at location \code{x} for the $i$^{th} knot index given \code{knots}
//' @export
// [[Rcpp::export]]
arma::mat bs_cpp (const arma::vec& x, const int& df,
                  const arma::vec& interior_knots, const int& degree,
                  const bool& intercept,
                  const arma::vec& Boundary_knots) {
  arma::vec Boundary_knots_sorted = sort(Boundary_knots);
  arma::vec knots_lower(degree+1);
  knots_lower.fill(Boundary_knots_sorted(0));
  arma::vec knots_upper(degree+1);
  knots_upper.fill(Boundary_knots_sorted(1));
  arma::vec knots_tmp = join_cols(knots_lower, interior_knots);
  arma::vec knots = join_cols(knots_tmp, knots_upper);
  int K = interior_knots.n_elem + degree + 1;
  int nx = x.n_elem;
  arma::mat B_mat(nx, K, arma::fill::zeros);
  for (int i=0; i<nx; i++) {
    for (int j=0; j<K; j++) {
      B_mat(i, j) = basis_cpp(x(i), degree, j, knots);
    }
  }
  if(any(x == Boundary_knots_sorted(1))){
    arma::uvec idx = arma::find(x == Boundary_knots_sorted(1));
    int n_idx = idx.n_elem;
    arma::uvec K_uvec(n_idx);
    K_uvec.fill(K-1);
    arma::vec one_vec(n_idx, arma::fill::ones);
    B_mat.submat(idx, K_uvec) = one_vec;
  }
  if(intercept == FALSE) {
    return(B_mat.cols(span(1, K-1)));
  } else {
    return(B_mat);
  }
}

