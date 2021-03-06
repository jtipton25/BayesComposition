#include <RcppArmadillo.h>
#include "bspline.h"
#include "colSums.h"
#include "d-half-cauchy.h"
#include "dMVN.h"
#include "dMVNChol.h"
#include "log-likelihood-dirichlet-multinomial.h"
#include "logDet.h"
#include "logit-expit.h"
#include "makeDistanceARMA.h"
#include "make-lkj.h"
#include "makeQinv.h"
#include "mvrnormARMA.h"
#include "mvrnormARMAChol.h"
#include "rMVNArma.h"
#include "rWishart.h"
#include "seq_lenC.h"
#include "updateTuning.h"

