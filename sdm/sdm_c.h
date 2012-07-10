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

// TODO: better error handling from this api

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

struct svm_parameter {
    int svm_type;
    int kernel_type;
    int degree;   /* for poly */
    double gamma; /* for poly/rbf/sigmoid */
    double coef0; /* for poly/sigmoid */
    
    /* these are for training only */
    double cache_size; /* in MB */
    double eps;        /* stopping criteria */
    double C;          /* for C_SVC, EPSILON_SVR and NU_SVR */
    int nr_weight;     /* for C_SVC */
    int *weight_label; /* for C_SVC */
    double* weight;    /* for C_SVC */
    double nu;         /* for NU_SVC, ONE_CLASS, and NU_SVR */
    double p;          /* for EPSILON_SVR */
    int shrinking;     /* use the shrinking heuristics */
    int probability;   /* do probability estimates */
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };
const int PRECOMPUTED = 4; // only kernel type we use
#endif

const extern struct svm_parameter default_svm_params;

const extern double * default_c_vals;
const extern size_t num_default_c_vals;

////////////////////////////////////////////////////////////////////////////////

#ifndef __LOG_H__
// Want access to log-level constants. If including both this file and
// log.hpp, the latter must be included first.
enum TLogLevel {logERROR, logWARNING, logINFO,
    logDEBUG, logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4};
#endif

void sdm_set_log_level(enum TLogLevel level);
enum TLogLevel sdm_get_log_level();

////////////////////////////////////////////////////////////////////////////////

// C equivalent of the DivParams class
typedef struct DivParamsC_s {
    int k;
    struct FLANNParameters flann_params;
    size_t num_threads;

    size_t show_progress;
    void (*print_progress)(size_t);
} DivParamsC;

void print_progress_to_stderr(size_t num_left);

// function to compute divergences
// really belongs in an npdivs C wrapper, but it's here for now
// results should be caller-allocated array of length num_div_specs, 
//    whose elements are of length num_x * num_y
// if y_bags is NULL, it's like you passed x in again (but more efficient)
//    and num_y, y_rows are ignored
#define NPDIVS(intype) \
    void np_divs_##intype( \
        const intype ** x_bags, size_t num_x, const size_t * x_rows, \
        const intype ** y_bags, size_t num_y, const size_t * y_rows, \
        size_t dim, \
        const char ** div_specs, size_t num_div_specs, \
        double ** results, \
        const DivParamsC * div_params)
NPDIVS(float);
NPDIVS(double);
#undef NPDIVS

////////////////////////////////////////////////////////////////////////////////

// C structs so we can return an SDM object from these methods
// declared to have a single SDM<intype, labtype> member
typedef struct SDM_ClassifyD_s SDM_ClassifyD;
typedef struct SDM_ClassifyF_s SDM_ClassifyF;
typedef struct SDM_RegressD_s  SDM_RegressD;
typedef struct SDM_RegressF_s  SDM_RegressF;

const char *SDM_ClassifyD_getName(SDM_ClassifyD *sdm);
const char *SDM_ClassifyF_getName(SDM_ClassifyF *sdm);
const char *SDM_RegressD_getName (SDM_RegressD *sdm);
const char *SDM_RegressF_getName (SDM_RegressF *sdm);

// free SDM memory; doesn't free the data itself, though
void SDM_ClassifyD_freeModel(SDM_ClassifyD *sdm);
void SDM_ClassifyF_freeModel(SDM_ClassifyF *sdm);
void SDM_RegressD_freeModel (SDM_RegressD *sdm);
void SDM_RegressF_freeModel (SDM_RegressF *sdm);

// functions to train SDMs with
#define TRAIN(classname, intype, labtype) \
    classname * classname##_train(\
        const intype **train_bags,\
        size_t num_train,\
        size_t dim,\
        const size_t * rows,\
        const labtype * labels,\
        const char * div_func_spec,\
        const char * kernel_spec,\
        const DivParamsC *div_params,\
        const double * c_vals, size_t num_c_vals,\
        const struct svm_parameter * svm_params,\
        size_t tuning_folds,\
        double * divs)
TRAIN(SDM_ClassifyD, double, int);
TRAIN(SDM_ClassifyF, float,  int);
TRAIN(SDM_RegressD, double, double);
TRAIN(SDM_RegressF, float,  double);
#undef TRAIN

// transduction functions: train on train data and predict on test, but take
// unlabeled test data into account during training (for PSD projection)
#define TRANSDUCT(classname, intype, labtype) \
    void classname##_transduct( \
        const intype ** train_bags, \
        size_t num_train, \
        const size_t * train_rows, \
        const intype ** test_bags, \
        size_t num_test, \
        const size_t * test_rows, \
        size_t dim, \
        const labtype * train_labels, \
        const char * div_func_spec, \
        const char * kernel_spec, \
        const DivParamsC *div_params, \
        const double * c_vals, size_t num_c_vals, \
        const struct svm_parameter * svm_params, \
        size_t tuning_folds, \
        double * divs, \
        labtype * preds)
TRANSDUCT(SDM_ClassifyD, double, int);
TRANSDUCT(SDM_ClassifyF, float,  int);
TRANSDUCT(SDM_RegressD, double, double);
TRANSDUCT(SDM_RegressF, float,  double);
#undef TRANSDUCT



// prediction functions
// single item, label only
#define PRED(classname, intype, labtype) \
    labtype classname##_predict(\
        const classname * sdm,\
        const intype * test_bag,\
        size_t rows)
PRED(SDM_ClassifyD, double, int);
PRED(SDM_ClassifyF, float,  int);
PRED(SDM_RegressD, double, double);
PRED(SDM_RegressF, float,  double);
#undef PRED

// single item, with decision values
// (allocating the storage for the values and changing vals to point to it)
#define PRED_V(classname, intype, labtype) \
    labtype classname##_predict_vals(\
        const classname * sdm,\
        const intype * test_bag,\
        size_t rows,\
        double ** vals,\
        size_t * num_vals)
PRED_V(SDM_ClassifyD, double, int);
PRED_V(SDM_ClassifyF, float,  int);
PRED_V(SDM_RegressD, double, double);
PRED_V(SDM_RegressF, float,  double);
#undef PRED_V

// several items, labels only
#define PRED_M(classname, intype, labtype) \
    void classname##_predict_many(\
        const classname * sdm,\
        const intype ** test_bags,\
        size_t num_test,\
        const size_t * rows,\
        labtype * labels)
PRED_M(SDM_ClassifyD, double, int);
PRED_M(SDM_ClassifyF, float,  int);
PRED_M(SDM_RegressD, double, double);
PRED_M(SDM_RegressF, float,  double);
#undef PRED_M

// several items, with decision values
#define PRED_MV(classname, intype, labtype) \
    void classname##_predict_many_vals(\
        const classname * sdm,\
        const intype ** test_bags,\
        size_t num_test,\
        const size_t * rows,\
        labtype * labels,\
        double *** vals,\
        size_t * num_vals)
PRED_MV(SDM_ClassifyD, double, int);
PRED_MV(SDM_ClassifyF, float,  int);
PRED_MV(SDM_RegressD, double, double);
PRED_MV(SDM_RegressF, float,  double);
#undef PRED_MV

// cross-validate on bags
#define CV(name, intype, labtype) \
    double sdm_crossvalidate_##name##_##intype(\
        const intype ** bags,\
        size_t num_bags,\
        const size_t * rows,\
        size_t dim,\
        const labtype * labels,\
        const char * div_func_spec,\
        const char * kernel_spec,\
        const DivParamsC * div_params,\
        size_t folds,\
        size_t num_cv_threads,\
        short project_all_data,\
        short shuffle_order,\
        const double * c_vals,\
        size_t num_c_vals,\
        const struct svm_parameter * svm_params,\
        size_t tuning_folds)
CV(classify, double, int);
CV(classify, float,  int);
CV(regress,  double, double);
CV(regress,  float,  double);
#undef CV

// cross-validate on precomputed divs
#define CV_divs(name, labtype) \
    double sdm_crossvalidate_##name##_divs(\
        const double * divs,\
        size_t num_bags,\
        const labtype *labels,\
        const char *kernel_spec,\
        size_t folds,\
        size_t num_cv_threads,\
        short project_all_data,\
        short shuffle_order,\
        const double *c_vals, size_t num_c_vals,\
        const struct svm_parameter *svm_params,\
        size_t tuning_folds)
CV_divs(classify, int);
CV_divs(regress, double);
#undef CV_divs

#ifdef __cplusplus
} // extern
#endif

#endif
