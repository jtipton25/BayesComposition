data {
  int<lower=1> N;
  int<lower=1> d;
  vector[N] X;
  matrix[N, d] y;
}
parameters {
  vector<lower=0>[d] a;
  vector<lower=0>[d] gamma;
  vector[d] mu;
  vector<lower=0>[d] sigma2;
}
transformed parameters {
  matrix<lower=0>[N, d] alpha;
  for (i in 1:N) {
    for (j in 1:d) {
      alpha[i, j] =  gamma[j] + a[j] * exp( - pow(mu[j] - X[i], 2.0) / sigma2[j]);
    }
  }

}
model {
  a ~ normal(0, 5);
  gamma ~ normal(0, 5);
  mu ~ normal(0, 5);
  sigma2 ~ cauchy(0, 5);
  for (i in 1:N) {
    real alpha_sum;
    vector[d] alpha_plus_y;
    vector[d] log_gamma_alpha_plus_y;
    vector[d] log_gamma_alpha;

    alpha_sum = sum(alpha[i, ]);
    alpha_plus_y = alpha[i, ]' + y[i, ]';
    for (j in 1:d) {
      log_gamma_alpha_plus_y[j] = lgamma(alpha_plus_y[j]);
      log_gamma_alpha[j] = lgamma(alpha[i, j]);
    }
    target += lgamma(alpha_sum) + sum(log_gamma_alpha_plus_y) -
              lgamma(alpha_sum + sum(y[i, ])) - sum(log_gamma_alpha);
  }
}

