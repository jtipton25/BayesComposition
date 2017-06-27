makeCRPSGauss <- function(mu, sdev, truth) {
  - sdev * (1 / sqrt(pi) - 2 * dnorm((truth- mu)/sdev) - 
            (truth- mu)/sdev * (2 * pnorm((truth- mu)/sdev) - 1))
}