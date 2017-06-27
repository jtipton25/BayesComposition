#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

//[[Rcpp::export]]
arma::vec makeCRPS(const arma::mat& estimate, const arma::vec& truth,
                   const int& n_samps){
  int N = truth.n_elem;
  arma::vec CRPS(N, fill::zeros);
  for(int j=0; j<N; j++){
    double accuracy = 0.0;
    double precision = 0.0;
    for(int k=0; k<n_samps; k++){
      Rcpp::checkUserInterrupt();
      accuracy += std::abs(estimate(k, j) - truth(j));
      for(int l=0; l<n_samps; l++){
        precision += std::abs(estimate(k, j) - estimate(l, j));
      }
    }
    CRPS(j) = 1.0 / n_samps * accuracy - 0.5 / pow(n_samps, 2) * precision;
  }
  return(CRPS);
}

