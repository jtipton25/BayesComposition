# BayesComposition


The BayesComposition package provides Bayesian MCMC estimates for joint compositional count species response data. The package produces species response curves to covariates as well as generative inverse predictions of unobserved covariates given compositional count observations.


This repository contains code and data from the paper

To reproduce the results from:
    Nolan, C., Tipton, J., Booth, R.K., Hooten, M.B., and Jackson, S.T. (Submitted). Comparing and improving methods for reconstructing water table depth from testate amoebae.
    
1) install the BayesComposition `R` package using `devtools` as

```
library(devtools)
install_github("jtipton25/BayesComposition")
```

2) To generate the cross-validation results in the manuscript run the `Rmarkdown` document

```
Cross-validation.Rmd
```

3) To generate the reconstruction results in the manuscripte tun the RMarkdown file in RStudio
```
TA_ReconCompPaper.Rmd
```
