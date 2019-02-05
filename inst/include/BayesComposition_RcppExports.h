// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#ifndef RCPP_BayesComposition_RCPPEXPORTS_H_GEN_
#define RCPP_BayesComposition_RCPPEXPORTS_H_GEN_

#include <RcppArmadillo.h>
#include <Rcpp.h>

namespace BayesComposition {

    using namespace Rcpp;

    namespace {
        void validateSignature(const char* sig) {
            Rcpp::Function require = Rcpp::Environment::base_env()["require"];
            require("BayesComposition", Rcpp::Named("quietly") = true);
            typedef int(*Ptr_validate)(const char*);
            static Ptr_validate p_validate = (Ptr_validate)
                R_GetCCallable("BayesComposition", "_BayesComposition_RcppExport_validate");
            if (!p_validate(sig)) {
                throw Rcpp::function_not_exported(
                    "C++ function with signature '" + std::string(sig) + "' not found in BayesComposition");
            }
        }
    }

    inline double basis_cpp(const double& x, const int& degree, const int& i, const arma::vec& knots) {
        typedef SEXP(*Ptr_basis_cpp)(SEXP,SEXP,SEXP,SEXP);
        static Ptr_basis_cpp p_basis_cpp = NULL;
        if (p_basis_cpp == NULL) {
            validateSignature("double(*basis_cpp)(const double&,const int&,const int&,const arma::vec&)");
            p_basis_cpp = (Ptr_basis_cpp)R_GetCCallable("BayesComposition", "_BayesComposition_basis_cpp");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_basis_cpp(Shield<SEXP>(Rcpp::wrap(x)), Shield<SEXP>(Rcpp::wrap(degree)), Shield<SEXP>(Rcpp::wrap(i)), Shield<SEXP>(Rcpp::wrap(knots)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline arma::mat bs_cpp(const arma::vec& x, const int& df, const arma::vec& interior_knots, const int& degree, const bool& intercept, const arma::vec& Boundary_knots) {
        typedef SEXP(*Ptr_bs_cpp)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_bs_cpp p_bs_cpp = NULL;
        if (p_bs_cpp == NULL) {
            validateSignature("arma::mat(*bs_cpp)(const arma::vec&,const int&,const arma::vec&,const int&,const bool&,const arma::vec&)");
            p_bs_cpp = (Ptr_bs_cpp)R_GetCCallable("BayesComposition", "_BayesComposition_bs_cpp");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_bs_cpp(Shield<SEXP>(Rcpp::wrap(x)), Shield<SEXP>(Rcpp::wrap(df)), Shield<SEXP>(Rcpp::wrap(interior_knots)), Shield<SEXP>(Rcpp::wrap(degree)), Shield<SEXP>(Rcpp::wrap(intercept)), Shield<SEXP>(Rcpp::wrap(Boundary_knots)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline arma::vec colSums(const arma::mat& X) {
        typedef SEXP(*Ptr_colSums)(SEXP);
        static Ptr_colSums p_colSums = NULL;
        if (p_colSums == NULL) {
            validateSignature("arma::vec(*colSums)(const arma::mat&)");
            p_colSums = (Ptr_colSums)R_GetCCallable("BayesComposition", "_BayesComposition_colSums");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_colSums(Shield<SEXP>(Rcpp::wrap(X)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline double d_half_cauchy(double& x, double& sigma, bool logd = true) {
        typedef SEXP(*Ptr_d_half_cauchy)(SEXP,SEXP,SEXP);
        static Ptr_d_half_cauchy p_d_half_cauchy = NULL;
        if (p_d_half_cauchy == NULL) {
            validateSignature("double(*d_half_cauchy)(double&,double&,bool)");
            p_d_half_cauchy = (Ptr_d_half_cauchy)R_GetCCallable("BayesComposition", "_BayesComposition_d_half_cauchy");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_d_half_cauchy(Shield<SEXP>(Rcpp::wrap(x)), Shield<SEXP>(Rcpp::wrap(sigma)), Shield<SEXP>(Rcpp::wrap(logd)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline double dMVN(const arma::mat& y, const arma::vec& mu, const arma::mat& Sigma_chol, const bool logd = true) {
        typedef SEXP(*Ptr_dMVN)(SEXP,SEXP,SEXP,SEXP);
        static Ptr_dMVN p_dMVN = NULL;
        if (p_dMVN == NULL) {
            validateSignature("double(*dMVN)(const arma::mat&,const arma::vec&,const arma::mat&,const bool)");
            p_dMVN = (Ptr_dMVN)R_GetCCallable("BayesComposition", "_BayesComposition_dMVN");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_dMVN(Shield<SEXP>(Rcpp::wrap(y)), Shield<SEXP>(Rcpp::wrap(mu)), Shield<SEXP>(Rcpp::wrap(Sigma_chol)), Shield<SEXP>(Rcpp::wrap(logd)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline double dMVNChol(const arma::vec& y, const arma::vec& mu, const arma::mat& Sigma_chol, const bool logd = true) {
        typedef SEXP(*Ptr_dMVNChol)(SEXP,SEXP,SEXP,SEXP);
        static Ptr_dMVNChol p_dMVNChol = NULL;
        if (p_dMVNChol == NULL) {
            validateSignature("double(*dMVNChol)(const arma::vec&,const arma::vec&,const arma::mat&,const bool)");
            p_dMVNChol = (Ptr_dMVNChol)R_GetCCallable("BayesComposition", "_BayesComposition_dMVNChol");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_dMVNChol(Shield<SEXP>(Rcpp::wrap(y)), Shield<SEXP>(Rcpp::wrap(mu)), Shield<SEXP>(Rcpp::wrap(Sigma_chol)), Shield<SEXP>(Rcpp::wrap(logd)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline double LL_DM(const arma::mat& alpha, const arma::mat& Y, const double& N, const double& d, const arma::vec& count) {
        typedef SEXP(*Ptr_LL_DM)(SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_LL_DM p_LL_DM = NULL;
        if (p_LL_DM == NULL) {
            validateSignature("double(*LL_DM)(const arma::mat&,const arma::mat&,const double&,const double&,const arma::vec&)");
            p_LL_DM = (Ptr_LL_DM)R_GetCCallable("BayesComposition", "_BayesComposition_LL_DM");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_LL_DM(Shield<SEXP>(Rcpp::wrap(alpha)), Shield<SEXP>(Rcpp::wrap(Y)), Shield<SEXP>(Rcpp::wrap(N)), Shield<SEXP>(Rcpp::wrap(d)), Shield<SEXP>(Rcpp::wrap(count)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline double LL_DM_row(const arma::rowvec& alpha, const arma::rowvec& Y, const double& d, const double& count) {
        typedef SEXP(*Ptr_LL_DM_row)(SEXP,SEXP,SEXP,SEXP);
        static Ptr_LL_DM_row p_LL_DM_row = NULL;
        if (p_LL_DM_row == NULL) {
            validateSignature("double(*LL_DM_row)(const arma::rowvec&,const arma::rowvec&,const double&,const double&)");
            p_LL_DM_row = (Ptr_LL_DM_row)R_GetCCallable("BayesComposition", "_BayesComposition_LL_DM_row");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_LL_DM_row(Shield<SEXP>(Rcpp::wrap(alpha)), Shield<SEXP>(Rcpp::wrap(Y)), Shield<SEXP>(Rcpp::wrap(d)), Shield<SEXP>(Rcpp::wrap(count)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline double logDet(const arma::mat Sig) {
        typedef SEXP(*Ptr_logDet)(SEXP);
        static Ptr_logDet p_logDet = NULL;
        if (p_logDet == NULL) {
            validateSignature("double(*logDet)(const arma::mat)");
            p_logDet = (Ptr_logDet)R_GetCCallable("BayesComposition", "_BayesComposition_logDet");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_logDet(Shield<SEXP>(Rcpp::wrap(Sig)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline double logDetChol(const arma::mat Sig_chol) {
        typedef SEXP(*Ptr_logDetChol)(SEXP);
        static Ptr_logDetChol p_logDetChol = NULL;
        if (p_logDetChol == NULL) {
            validateSignature("double(*logDetChol)(const arma::mat)");
            p_logDetChol = (Ptr_logDetChol)R_GetCCallable("BayesComposition", "_BayesComposition_logDetChol");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_logDetChol(Shield<SEXP>(Rcpp::wrap(Sig_chol)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline arma::vec logit(const arma::vec& phi) {
        typedef SEXP(*Ptr_logit)(SEXP);
        static Ptr_logit p_logit = NULL;
        if (p_logit == NULL) {
            validateSignature("arma::vec(*logit)(const arma::vec&)");
            p_logit = (Ptr_logit)R_GetCCallable("BayesComposition", "_BayesComposition_logit");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_logit(Shield<SEXP>(Rcpp::wrap(phi)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline arma::vec expit(const arma::vec& phi) {
        typedef SEXP(*Ptr_expit)(SEXP);
        static Ptr_expit p_expit = NULL;
        if (p_expit == NULL) {
            validateSignature("arma::vec(*expit)(const arma::vec&)");
            p_expit = (Ptr_expit)R_GetCCallable("BayesComposition", "_BayesComposition_expit");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_expit(Shield<SEXP>(Rcpp::wrap(phi)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline double logit_double(const double& phi) {
        typedef SEXP(*Ptr_logit_double)(SEXP);
        static Ptr_logit_double p_logit_double = NULL;
        if (p_logit_double == NULL) {
            validateSignature("double(*logit_double)(const double&)");
            p_logit_double = (Ptr_logit_double)R_GetCCallable("BayesComposition", "_BayesComposition_logit_double");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_logit_double(Shield<SEXP>(Rcpp::wrap(phi)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline double expit_double(double& phi) {
        typedef SEXP(*Ptr_expit_double)(SEXP);
        static Ptr_expit_double p_expit_double = NULL;
        if (p_expit_double == NULL) {
            validateSignature("double(*expit_double)(double&)");
            p_expit_double = (Ptr_expit_double)R_GetCCallable("BayesComposition", "_BayesComposition_expit_double");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_expit_double(Shield<SEXP>(Rcpp::wrap(phi)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline arma::mat makeUpperLKJ(const arma::vec& x, const int& d) {
        typedef SEXP(*Ptr_makeUpperLKJ)(SEXP,SEXP);
        static Ptr_makeUpperLKJ p_makeUpperLKJ = NULL;
        if (p_makeUpperLKJ == NULL) {
            validateSignature("arma::mat(*makeUpperLKJ)(const arma::vec&,const int&)");
            p_makeUpperLKJ = (Ptr_makeUpperLKJ)R_GetCCallable("BayesComposition", "_BayesComposition_makeUpperLKJ");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_makeUpperLKJ(Shield<SEXP>(Rcpp::wrap(x)), Shield<SEXP>(Rcpp::wrap(d)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline Rcpp::List makeRLKJ(const arma::vec& xi, const int& d, bool cholesky = false, bool jacobian = false) {
        typedef SEXP(*Ptr_makeRLKJ)(SEXP,SEXP,SEXP,SEXP);
        static Ptr_makeRLKJ p_makeRLKJ = NULL;
        if (p_makeRLKJ == NULL) {
            validateSignature("Rcpp::List(*makeRLKJ)(const arma::vec&,const int&,bool,bool)");
            p_makeRLKJ = (Ptr_makeRLKJ)R_GetCCallable("BayesComposition", "_BayesComposition_makeRLKJ");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_makeRLKJ(Shield<SEXP>(Rcpp::wrap(xi)), Shield<SEXP>(Rcpp::wrap(d)), Shield<SEXP>(Rcpp::wrap(cholesky)), Shield<SEXP>(Rcpp::wrap(jacobian)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<Rcpp::List >(rcpp_result_gen);
    }

    inline arma::vec makeCRPS(const arma::mat& estimate, const arma::vec& truth, const int& n_samps) {
        typedef SEXP(*Ptr_makeCRPS)(SEXP,SEXP,SEXP);
        static Ptr_makeCRPS p_makeCRPS = NULL;
        if (p_makeCRPS == NULL) {
            validateSignature("arma::vec(*makeCRPS)(const arma::mat&,const arma::vec&,const int&)");
            p_makeCRPS = (Ptr_makeCRPS)R_GetCCallable("BayesComposition", "_BayesComposition_makeCRPS");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_makeCRPS(Shield<SEXP>(Rcpp::wrap(estimate)), Shield<SEXP>(Rcpp::wrap(truth)), Shield<SEXP>(Rcpp::wrap(n_samps)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline arma::mat makeDistARMA(const arma::mat& coords1, const arma::mat& coords2) {
        typedef SEXP(*Ptr_makeDistARMA)(SEXP,SEXP);
        static Ptr_makeDistARMA p_makeDistARMA = NULL;
        if (p_makeDistARMA == NULL) {
            validateSignature("arma::mat(*makeDistARMA)(const arma::mat&,const arma::mat&)");
            p_makeDistARMA = (Ptr_makeDistARMA)R_GetCCallable("BayesComposition", "_BayesComposition_makeDistARMA");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_makeDistARMA(Shield<SEXP>(Rcpp::wrap(coords1)), Shield<SEXP>(Rcpp::wrap(coords2)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline arma::mat makeQinv(const double& theta, const int& t) {
        typedef SEXP(*Ptr_makeQinv)(SEXP,SEXP);
        static Ptr_makeQinv p_makeQinv = NULL;
        if (p_makeQinv == NULL) {
            validateSignature("arma::mat(*makeQinv)(const double&,const int&)");
            p_makeQinv = (Ptr_makeQinv)R_GetCCallable("BayesComposition", "_BayesComposition_makeQinv");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_makeQinv(Shield<SEXP>(Rcpp::wrap(theta)), Shield<SEXP>(Rcpp::wrap(t)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline arma::mat mvrnormArma(const int& n, const arma::vec& mu, const arma::mat& Sigma) {
        typedef SEXP(*Ptr_mvrnormArma)(SEXP,SEXP,SEXP);
        static Ptr_mvrnormArma p_mvrnormArma = NULL;
        if (p_mvrnormArma == NULL) {
            validateSignature("arma::mat(*mvrnormArma)(const int&,const arma::vec&,const arma::mat&)");
            p_mvrnormArma = (Ptr_mvrnormArma)R_GetCCallable("BayesComposition", "_BayesComposition_mvrnormArma");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_mvrnormArma(Shield<SEXP>(Rcpp::wrap(n)), Shield<SEXP>(Rcpp::wrap(mu)), Shield<SEXP>(Rcpp::wrap(Sigma)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline arma::vec mvrnormArmaVec(const arma::vec& mu, const arma::mat& Sigma) {
        typedef SEXP(*Ptr_mvrnormArmaVec)(SEXP,SEXP);
        static Ptr_mvrnormArmaVec p_mvrnormArmaVec = NULL;
        if (p_mvrnormArmaVec == NULL) {
            validateSignature("arma::vec(*mvrnormArmaVec)(const arma::vec&,const arma::mat&)");
            p_mvrnormArmaVec = (Ptr_mvrnormArmaVec)R_GetCCallable("BayesComposition", "_BayesComposition_mvrnormArmaVec");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_mvrnormArmaVec(Shield<SEXP>(Rcpp::wrap(mu)), Shield<SEXP>(Rcpp::wrap(Sigma)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline arma::mat mvrnormArmaChol(const int& n, const arma::vec& mu, const arma::mat& Sigma_chol) {
        typedef SEXP(*Ptr_mvrnormArmaChol)(SEXP,SEXP,SEXP);
        static Ptr_mvrnormArmaChol p_mvrnormArmaChol = NULL;
        if (p_mvrnormArmaChol == NULL) {
            validateSignature("arma::mat(*mvrnormArmaChol)(const int&,const arma::vec&,const arma::mat&)");
            p_mvrnormArmaChol = (Ptr_mvrnormArmaChol)R_GetCCallable("BayesComposition", "_BayesComposition_mvrnormArmaChol");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_mvrnormArmaChol(Shield<SEXP>(Rcpp::wrap(n)), Shield<SEXP>(Rcpp::wrap(mu)), Shield<SEXP>(Rcpp::wrap(Sigma_chol)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline arma::vec mvrnormArmaVecChol(const arma::vec& mu, const arma::mat& Sigma_chol) {
        typedef SEXP(*Ptr_mvrnormArmaVecChol)(SEXP,SEXP);
        static Ptr_mvrnormArmaVecChol p_mvrnormArmaVecChol = NULL;
        if (p_mvrnormArmaVecChol == NULL) {
            validateSignature("arma::vec(*mvrnormArmaVecChol)(const arma::vec&,const arma::mat&)");
            p_mvrnormArmaVecChol = (Ptr_mvrnormArmaVecChol)R_GetCCallable("BayesComposition", "_BayesComposition_mvrnormArmaVecChol");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_mvrnormArmaVecChol(Shield<SEXP>(Rcpp::wrap(mu)), Shield<SEXP>(Rcpp::wrap(Sigma_chol)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline arma::vec rMVNArma(arma::mat& A, arma::vec& b) {
        typedef SEXP(*Ptr_rMVNArma)(SEXP,SEXP);
        static Ptr_rMVNArma p_rMVNArma = NULL;
        if (p_rMVNArma == NULL) {
            validateSignature("arma::vec(*rMVNArma)(arma::mat&,arma::vec&)");
            p_rMVNArma = (Ptr_rMVNArma)R_GetCCallable("BayesComposition", "_BayesComposition_rMVNArma");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_rMVNArma(Shield<SEXP>(Rcpp::wrap(A)), Shield<SEXP>(Rcpp::wrap(b)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline double rMVNArmaScalar(const double& a, const double& b) {
        typedef SEXP(*Ptr_rMVNArmaScalar)(SEXP,SEXP);
        static Ptr_rMVNArmaScalar p_rMVNArmaScalar = NULL;
        if (p_rMVNArmaScalar == NULL) {
            validateSignature("double(*rMVNArmaScalar)(const double&,const double&)");
            p_rMVNArmaScalar = (Ptr_rMVNArmaScalar)R_GetCCallable("BayesComposition", "_BayesComposition_rMVNArmaScalar");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_rMVNArmaScalar(Shield<SEXP>(Rcpp::wrap(a)), Shield<SEXP>(Rcpp::wrap(b)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<double >(rcpp_result_gen);
    }

    inline arma::mat rWishartArmaMat(const unsigned int& df, const arma::mat& S) {
        typedef SEXP(*Ptr_rWishartArmaMat)(SEXP,SEXP);
        static Ptr_rWishartArmaMat p_rWishartArmaMat = NULL;
        if (p_rWishartArmaMat == NULL) {
            validateSignature("arma::mat(*rWishartArmaMat)(const unsigned int&,const arma::mat&)");
            p_rWishartArmaMat = (Ptr_rWishartArmaMat)R_GetCCallable("BayesComposition", "_BayesComposition_rWishartArmaMat");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_rWishartArmaMat(Shield<SEXP>(Rcpp::wrap(df)), Shield<SEXP>(Rcpp::wrap(S)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline arma::mat rIWishartArmaMat(const unsigned int& df, const arma::mat& S) {
        typedef SEXP(*Ptr_rIWishartArmaMat)(SEXP,SEXP);
        static Ptr_rIWishartArmaMat p_rIWishartArmaMat = NULL;
        if (p_rIWishartArmaMat == NULL) {
            validateSignature("arma::mat(*rIWishartArmaMat)(const unsigned int&,const arma::mat&)");
            p_rIWishartArmaMat = (Ptr_rIWishartArmaMat)R_GetCCallable("BayesComposition", "_BayesComposition_rIWishartArmaMat");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_rIWishartArmaMat(Shield<SEXP>(Rcpp::wrap(df)), Shield<SEXP>(Rcpp::wrap(S)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::mat >(rcpp_result_gen);
    }

    inline arma::vec seq_lenC(const int& n) {
        typedef SEXP(*Ptr_seq_lenC)(SEXP);
        static Ptr_seq_lenC p_seq_lenC = NULL;
        if (p_seq_lenC == NULL) {
            validateSignature("arma::vec(*seq_lenC)(const int&)");
            p_seq_lenC = (Ptr_seq_lenC)R_GetCCallable("BayesComposition", "_BayesComposition_seq_lenC");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_seq_lenC(Shield<SEXP>(Rcpp::wrap(n)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<arma::vec >(rcpp_result_gen);
    }

    inline void updateTuning(const int k, double& accept_tmp, double& tune) {
        typedef SEXP(*Ptr_updateTuning)(SEXP,SEXP,SEXP);
        static Ptr_updateTuning p_updateTuning = NULL;
        if (p_updateTuning == NULL) {
            validateSignature("void(*updateTuning)(const int,double&,double&)");
            p_updateTuning = (Ptr_updateTuning)R_GetCCallable("BayesComposition", "_BayesComposition_updateTuning");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_updateTuning(Shield<SEXP>(Rcpp::wrap(k)), Shield<SEXP>(Rcpp::wrap(accept_tmp)), Shield<SEXP>(Rcpp::wrap(tune)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
    }

    inline void updateTuningVec(const int k, arma::vec& accept_tmp, arma::vec& tune) {
        typedef SEXP(*Ptr_updateTuningVec)(SEXP,SEXP,SEXP);
        static Ptr_updateTuningVec p_updateTuningVec = NULL;
        if (p_updateTuningVec == NULL) {
            validateSignature("void(*updateTuningVec)(const int,arma::vec&,arma::vec&)");
            p_updateTuningVec = (Ptr_updateTuningVec)R_GetCCallable("BayesComposition", "_BayesComposition_updateTuningVec");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_updateTuningVec(Shield<SEXP>(Rcpp::wrap(k)), Shield<SEXP>(Rcpp::wrap(accept_tmp)), Shield<SEXP>(Rcpp::wrap(tune)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
    }

    inline void updateTuningMat(const int k, arma::mat& accept_tmp, arma::mat& tune) {
        typedef SEXP(*Ptr_updateTuningMat)(SEXP,SEXP,SEXP);
        static Ptr_updateTuningMat p_updateTuningMat = NULL;
        if (p_updateTuningMat == NULL) {
            validateSignature("void(*updateTuningMat)(const int,arma::mat&,arma::mat&)");
            p_updateTuningMat = (Ptr_updateTuningMat)R_GetCCallable("BayesComposition", "_BayesComposition_updateTuningMat");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_updateTuningMat(Shield<SEXP>(Rcpp::wrap(k)), Shield<SEXP>(Rcpp::wrap(accept_tmp)), Shield<SEXP>(Rcpp::wrap(tune)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
    }

    inline void updateTuningMV(const int& k, double& accept_rate, double& lambda, arma::mat& batch_samples, arma::mat& Sigma_tune, arma::mat Sigma_tune_chol) {
        typedef SEXP(*Ptr_updateTuningMV)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_updateTuningMV p_updateTuningMV = NULL;
        if (p_updateTuningMV == NULL) {
            validateSignature("void(*updateTuningMV)(const int&,double&,double&,arma::mat&,arma::mat&,arma::mat)");
            p_updateTuningMV = (Ptr_updateTuningMV)R_GetCCallable("BayesComposition", "_BayesComposition_updateTuningMV");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_updateTuningMV(Shield<SEXP>(Rcpp::wrap(k)), Shield<SEXP>(Rcpp::wrap(accept_rate)), Shield<SEXP>(Rcpp::wrap(lambda)), Shield<SEXP>(Rcpp::wrap(batch_samples)), Shield<SEXP>(Rcpp::wrap(Sigma_tune)), Shield<SEXP>(Rcpp::wrap(Sigma_tune_chol)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
    }

    inline void updateTuningMVMat(const int& k, arma::vec& accept_rate, arma::vec& lambda, arma::cube& batch_samples, arma::cube& Sigma_tune, arma::cube Sigma_tune_chol) {
        typedef SEXP(*Ptr_updateTuningMVMat)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_updateTuningMVMat p_updateTuningMVMat = NULL;
        if (p_updateTuningMVMat == NULL) {
            validateSignature("void(*updateTuningMVMat)(const int&,arma::vec&,arma::vec&,arma::cube&,arma::cube&,arma::cube)");
            p_updateTuningMVMat = (Ptr_updateTuningMVMat)R_GetCCallable("BayesComposition", "_BayesComposition_updateTuningMVMat");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_updateTuningMVMat(Shield<SEXP>(Rcpp::wrap(k)), Shield<SEXP>(Rcpp::wrap(accept_rate)), Shield<SEXP>(Rcpp::wrap(lambda)), Shield<SEXP>(Rcpp::wrap(batch_samples)), Shield<SEXP>(Rcpp::wrap(Sigma_tune)), Shield<SEXP>(Rcpp::wrap(Sigma_tune_chol)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
    }

}

#endif // RCPP_BayesComposition_RCPPEXPORTS_H_GEN_
