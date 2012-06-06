/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Portions included from the FLANN library:                                   *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.  *
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.   *
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
 *     * Neither the name of Carnegie Mellon University, the University of     *
 *       British Columbia, nor the names of the contributors may be used to    &
 *       endorse or promote products derived from this software without        *
 *       specific prior written permission.                                    *
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
#include <svm.h>

// can't #include flann.h, because of LINEAR constant clash with svm.h
// #include <flann/flann.h>

#ifdef __cplusplus
extern "C" {
#endif

// necessary inclusions from flann/flann.h and flann/defines.h

enum flann_algorithm_t
{
    FLANN_INDEX_LINEAR = 0,
    FLANN_INDEX_KDTREE = 1,
    FLANN_INDEX_KMEANS = 2,
    FLANN_INDEX_COMPOSITE = 3,
    FLANN_INDEX_KDTREE_SINGLE = 4,
    FLANN_INDEX_HIERARCHICAL = 5,
    FLANN_INDEX_LSH = 6,
    FLANN_INDEX_KDTREE_CUDA = 7,
    FLANN_INDEX_SAVED = 254,
    FLANN_INDEX_AUTOTUNED = 255
};

enum flann_centers_init_t
{
    FLANN_CENTERS_RANDOM = 0,
    FLANN_CENTERS_GONZALES = 1,
    FLANN_CENTERS_KMEANSPP = 2
};

enum flann_log_level_t
{
    FLANN_LOG_NONE = 0,
    FLANN_LOG_FATAL = 1,
    FLANN_LOG_ERROR = 2,
    FLANN_LOG_WARN = 3,
    FLANN_LOG_INFO = 4,
    FLANN_LOG_DEBUG = 5
};

struct FLANNParameters
{
    enum flann_algorithm_t algorithm; /* the algorithm to use */

    /* search time parameters */
    int checks;                /* how many leafs (features) to check in one search */
    float cb_index;            /* cluster boundary index. Used when searching the kmeans tree */
    float eps;     /* eps parameter for eps-knn search */

    /*  kdtree index parameters */
    int trees;                 /* number of randomized trees to use (for kdtree) */
    int leaf_max_size;

    /* kmeans index parameters */
    int branching;             /* branching factor (for kmeans tree) */
    int iterations;            /* max iterations to perform in one kmeans cluetering (kmeans tree) */
    enum flann_centers_init_t centers_init;  /* algorithm used for picking the initial cluster centers for kmeans tree */

    /* autotuned index parameters */
    float target_precision;    /* precision desired (used for autotuning, -1 otherwise) */
    float build_weight;        /* build tree time weighting factor */
    float memory_weight;       /* index memory weigthing factor */
    float sample_fraction;     /* what fraction of the dataset to use for autotuning */

    /* LSH parameters */
    unsigned int table_number_; /** The number of hash tables to use */
    unsigned int key_size_;     /** The length of the key in the hash tables */
    unsigned int multi_probe_level_; /** Number of levels to use in multi-probe LSH, 0 for standard LSH */

    /* other parameters */
    enum flann_log_level_t log_level;    /* determines the verbosity of each flann function */
    long random_seed;            /* random seed to use */
};

struct FLANNParameters DEFAULT_FLANN_PARAMETERS = {
    FLANN_INDEX_KDTREE,
    32, 0.2f, 0.0f,
    4, 4,
    32, 11, FLANN_CENTERS_RANDOM,
    0.9f, 0.01f, 0, 0.1f,
    FLANN_LOG_NONE, 0
};

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
//struct SDMObjDouble_s;
//struct SDMObjFloat_s;
typedef struct SDMObjDouble_s SDMObjDouble;
typedef struct SDMObjFloat_s SDMObjFloat;

char *SDMObjDouble_getName(SDMObjDouble sdm);
char *SDMObjFloat_getName(SDMObjFloat sdm);

// functions to train SDMs with
SDMObjDouble train_sdm_double(
        const double **train_bags,
        size_t num_train,
        size_t dim,
        size_t * rows,
        const int *labels,
        const char *div_func_spec,
        const char *kernel_spec,
        const DivParamsC *div_params,
        const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params,
        size_t tuning_folds,
        double *divs);
SDMObjFloat train_sdm_float(
        const float **train_bags,
        size_t num_train,
        size_t dim,
        size_t * rows,
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
