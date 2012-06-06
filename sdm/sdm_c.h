/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Portions included from libsvm:                                              *
 * Copyright (c) 2000-2012 Chih-Chung Chang and Chih-Jen Lin.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * Redistribution and use in source and binary forms, with or without          *
 * modification, are permitted provided that the following conditions are met: *
 *                                                                             *
 *     * Redistributions of source code must retain the above copyright        *
 *       notice, this list of conditions and the following disclaimer.         *
 *                                                                             *
 *     * Redistributions in binary form must reproduce the above copyright     *
 *       notice, this list of conditions and the following disclaimer in the   *
 *       documentation and/or other materials provided with the distribution.  *
 *                                                                             *
 *     * Neither the name of Carnegie Mellon University nor the names of the   *
 *       contributors may be used to endorse or promote products derived from  *
 *       this software without specific prior written permission.              *
 *                                                                             *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE   *
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE  *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE   *
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR         *
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF        *
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS    *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN     *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)     *
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE  *
 * POSSIBILITY OF SUCH DAMAGE.                                                 *
 ******************************************************************************/
#ifndef SDM_C_H_
#define SDM_C_H_

#include <stddef.h>

#include <flann/flann.h>

// not #include-ing svm.h, because of LINEAR constant clash with flann.h
// #include <svm.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _LIBSVM_H
// Don't want to redefine svm_parameter if we already know about it. This
// means if including both this header and svm.h, this one MUST come second.
// (This should only be possible in C++ code, because of the namespash clash.)


struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
const int PRECOMPUTED = 4; // only kernel type we use
#endif

const extern struct svm_parameter default_svm_params;


////////////////////////////////////////////////////////////////////////////////

// C equivalent of the DivParams class
typedef struct DivParamsC_s {
    int k;
    struct FLANNParameters flann_params;
    size_t num_threads;

    size_t show_progress;
    void (*print_progress)(size_t);
} DivParamsC;

// C structs so we can return an SDM object from these methods
// declared to have a single SDM<double> or SDM<float> member
struct SDMObjDouble_s;
struct SDMObjFloat_s;
typedef struct SDMObjDouble_s SDMObjDouble;
typedef struct SDMObjFloat_s  SDMObjFloat;

const char *SDMObjDouble_getName(SDMObjDouble *sdm);
const char *SDMObjFloat_getName (SDMObjFloat  *sdm);

// free SDM memory; doesn't free the data itself, though
void SDMObjDouble_freeModel(SDMObjDouble *sdm);
void SDMObjFloat_freeModel(SDMObjDouble *sdm);

// functions to train SDMs with
SDMObjDouble *train_sdm_double(
        const double **train_bags,
        size_t num_train,
        size_t dim,
        const size_t * rows,
        const int *labels,
        const char *div_func_spec,
        const char *kernel_spec,
        const DivParamsC *div_params,
        const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params,
        size_t tuning_folds,
        double *divs);
SDMObjFloat *train_sdm_float(
        const float **train_bags,
        size_t num_train,
        size_t dim,
        const size_t * rows,
        const int *labels,
        const char *div_func_spec,
        const char *kernel_spec,
        const DivParamsC *div_params,
        const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params,
        size_t tuning_folds,
        double *divs);

// prediction functions
// single item, label only
int sdm_predict_double(
        const SDMObjDouble *sdm,
        const double* test_bag, size_t rows);
int sdm_predict_float(
        const SDMObjFloat *sdm,
        const float* test_bag, size_t rows);

// single item, with decision values
// (allocating the storage for the values and changing vals to point to it)
int sdm_predict_vals_double(
        const SDMObjDouble *sdm,
        const double * test_bag, const size_t rows,
        double ** vals, size_t * num_vals);
int sdm_predict_vals_float(
        const SDMObjFloat *sdm,
        const float* test_bag, const size_t rows,
        double ** vals, size_t * num_vals);

// several items, labels only
void sdm_predict_many_double(
        const SDMObjDouble *sdm,
        const double ** test_bags, size_t num_test, const size_t * rows,
        int *labels);
void sdm_predict_many_float(
        const SDMObjFloat *sdm,
        const float ** test_bags, size_t num_test, const size_t * rows,
        int *labels);

// several items, with decision values
void sdm_predict_many_vals_double(
        const SDMObjDouble *sdm,
        const double ** test_bags, size_t num_test, const size_t * rows,
        int *labels,
        double *** vals, size_t * num_vals);
void sdm_predict_many_vals_float(
        const SDMObjFloat *sdm,
        const float ** test_bags, size_t num_test, const size_t * rows,
        int *labels,
        double *** vals, size_t * num_vals);


// functions to run cross-validation with
double crossvalidate_bags_double(
        const double ** bags,
        size_t num_bags,
        const size_t *rows,
        size_t dim,
        const int *labels,
        const char *div_func_spec,
        const char *kernel_spec,
        const DivParamsC *div_params,
        size_t folds,
        size_t num_cv_threads,
        short project_all_data,
        short shuffle_order,
        const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params,
        size_t tuning_folds);

double crossvalidate_bags_float(
        const float ** bags,
        size_t num_bags,
        const size_t *rows,
        size_t dim,
        const int *labels,
        const char *div_func_spec,
        const char *kernel_spec,
        const DivParamsC *div_params,
        size_t folds,
        size_t num_cv_threads,
        short project_all_data,
        short shuffle_order,
        const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params,
        size_t tuning_folds);

double crossvalidate_divs(
        const double * divs,
        size_t num_bags,
        const int *labels,
        const char *kernel_spec,
        size_t folds,
        size_t num_cv_threads,
        short project_all_data,
        short shuffle_order,
        const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params,
        size_t tuning_folds);


#ifdef __cplusplus
} // extern
#endif

#endif
