#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::interfaces(r, cpp)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
arma::mat makeDistARMA(const arma::mat & coords1, const arma::mat & coords2) {
  int nrows1 = coords1.n_rows;
  int nrows2 = coords2.n_rows;
  arma::mat D(nrows1, nrows2, fill::zeros);
  int ncols1 = coords1.n_cols;
  if(ncols1 == 1){
    for(int i = 0; i < nrows1; i++){
      for(int j = 0; j < nrows2; j++){
        D(i, j) = sqrt(pow(coords1(i) - coords2(j), 2));
      }
    }
  }
  if(ncols1 == 2){  
    for(int i = 0; i < nrows1; i++){
      for(int j = 0; j < nrows2; j++){
        double tmp = 0;
        for(int k = 0; k < 2; k++){
        tmp += pow(coords1(i, k) - coords2(j, k), 2);
        }
        D(i, j) = sqrt(tmp);
      }
    }
  }
  return(D);
}
