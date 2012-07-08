/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
 * All rights reserved.                                                        *
 *                                                                             *
 * Portions included from the FLANN library:                                   *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.  *
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.   *
 *
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
 *       British Columbia, nor the names of the contributors may be used to    *
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
#include "sdm/sdm.hpp"
#include "sdm/kernels/from_str.hpp"
#include "sdm/log.hpp"
#include "sdm/sdm_c.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <flann/util/matrix.h>

#include <np-divs/div-funcs/from_str.hpp>
#include <np-divs/np_divs.hpp>

using namespace sdm;
using namespace npdivs;
using std::string;
using std::vector;
using flann::Matrix;

////////////////////////////////////////////////////////////////////////////////

void sdm_set_log_level(enum TLogLevel level) {
    try {
        FILELog::ReportingLevel() = level;
    } catch (std::exception &e) {
        FILE_LOG(logERROR) << e.what();
    }
}
enum TLogLevel sdm_get_log_level() {
    try {
        return FILELog::ReportingLevel();
    } catch (std::exception &e) {
        FILE_LOG(logERROR) << e.what();
        return logERROR;
    }
}

////////////////////////////////////////////////////////////////////////////////

void print_progress_to_stderr(size_t num_left) {
    npdivs::print_progress_cerr(num_left);
}

////////////////////////////////////////////////////////////////////////////////

struct SDM_ClassifyD_s { SDM<double, int> *sdm; };
struct SDM_ClassifyF_s { SDM<float,  int> *sdm; };
struct SDM_RegressD_s  { SDM<double, double> *sdm; };
struct SDM_RegressF_s  { SDM<float,  double> *sdm; };

#define GET_NAME(classname) \
const char * classname##_getName(classname *sdm) { \
    try { \
        return sdm->sdm->name().c_str(); \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        return NULL; \
    } \
}
GET_NAME(SDM_ClassifyD);
GET_NAME(SDM_ClassifyF);
GET_NAME(SDM_RegressD);
GET_NAME(SDM_RegressF);
#undef GET_NAME


#define FREE_MODEL(classname, intype, labtype) \
void classname##_freeModel(classname * sdm) { \
    try { \
        SDM<intype, labtype> * m = sdm->sdm; \
        m->destroyModelAndProb(); \
        m->destroyTrainBagMatrices(); \
        delete m; \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
    } \
}
FREE_MODEL(SDM_ClassifyD, double, int);
FREE_MODEL(SDM_ClassifyF, float,  int);
FREE_MODEL(SDM_RegressD,  double, double);
FREE_MODEL(SDM_RegressF,  float,  double);
#undef FREE_MODEL

////////////////////////////////////////////////////////////////////////////////

const struct svm_parameter default_svm_params = sdm::default_svm_params;

const double * default_c_vals = &sdm::default_c_vals[0];
const size_t num_default_c_vals = sdm::default_c_vals.size();

// copied from flann/flann.cpp
flann::IndexParams create_parameters(const FLANNParameters* p)
{
    flann::IndexParams params;

    params["algorithm"] = p->algorithm;

    params["checks"] = p->checks;
    params["cb_index"] = p->cb_index;
    params["eps"] = p->eps;

    if (p->algorithm == FLANN_INDEX_KDTREE) {
        params["trees"] = p->trees;
    }

    if (p->algorithm == FLANN_INDEX_KDTREE_SINGLE) {
        params["trees"] = p->trees;
        params["leaf_max_size"] = p->leaf_max_size;
    }
    
    if (p->algorithm == FLANN_INDEX_KDTREE_CUDA) {
        params["leaf_max_size"] = p->leaf_max_size;
    }

    if (p->algorithm == FLANN_INDEX_KMEANS) {
        params["branching"] = p->branching;
        params["iterations"] = p->iterations;
        params["centers_init"] = p->centers_init;
    }

    if (p->algorithm == FLANN_INDEX_AUTOTUNED) {
        params["target_precision"] = p->target_precision;
        params["build_weight"] = p->build_weight;
        params["memory_weight"] = p->memory_weight;
        params["sample_fraction"] = p->sample_fraction;
    }

    if (p->algorithm == FLANN_INDEX_HIERARCHICAL) {
        params["branching"] = p->branching;
        params["centers_init"] = p->centers_init;
        params["trees"] = p->trees;
        params["leaf_size"] = p->leaf_max_size;
    }

    if (p->algorithm == FLANN_INDEX_LSH) {
        params["table_number"] = p->table_number_;
        params["key_size"] = p->key_size_;
        params["multi_probe_level"] = p->multi_probe_level_;
    }

    params["log_level"] = p->log_level;
    params["random_seed"] = p->random_seed;

    return params;
}

DivParams make_div_params(const DivParamsC *params) {
    return DivParams(
            params->k,
            create_parameters(&(params->flann_params)),
            flann::SearchParams(params->flann_params.checks),
            params->num_threads,
            params->show_progress,
            params->print_progress);
}

template <typename T>
Matrix<T> *make_matrices(T ** data,
        size_t num, const size_t * rows, size_t dim) {

    Matrix<T> * array = new Matrix<T>[num];
    for (size_t i = 0; i < num; i++)
        array[i] = Matrix<T>(const_cast<T *>(data[i]), rows[i], dim);
    return array;
}

////////////////////////////////////////////////////////////////////////////////

#define NPDIVS(intype) \
    void np_divs_##intype( \
        const intype ** x_bags, size_t num_x, const size_t * x_rows, \
        const intype ** y_bags, size_t num_y, const size_t * y_rows, \
        size_t dim, \
        const char ** div_specs, size_t num_div_specs, \
        double ** results, \
        const DivParamsC * div_params) \
    { \
        typedef flann::Matrix<intype> Matrix; \
        typedef flann::Matrix<double> MatrixD; \
        \
        /* make arrays of flann::Matrix for x_bags, y_bags */ \
        Matrix * x_bags_ms = new Matrix[num_x]; \
        for (size_t i = 0; i < num_x; i++) { \
            x_bags_ms[i] = Matrix( \
                    const_cast<intype *>(x_bags[i]), x_rows[i], dim); \
        } \
        \
        Matrix * y_bags_ms; \
        if (y_bags == NULL) { \
            y_bags_ms = NULL; \
            num_y = num_x; \
        } else { \
            y_bags_ms = new Matrix[num_y]; \
            for (size_t i = 0; i < num_y; i++) { \
                y_bags_ms[i] = Matrix( \
                        const_cast<intype *>(y_bags[i]), y_rows[i], dim);\
            } \
        } \
        \
        /* make boost::ptr_vector<DivFunc> */ \
        boost::ptr_vector<DivFunc> div_funcs; \
        for (size_t i = 0; i < num_div_specs; i++) { \
            div_funcs.push_back(npdivs::div_func_from_str( \
                        std::string(div_specs[i]))); \
        } \
        \
        /* make flann::Matrix<double> with results as data */ \
        MatrixD * results_ms = new MatrixD[num_div_specs]; \
        for (size_t i = 0; i < num_div_specs; i++) { \
            results_ms[i] = MatrixD(results[i], num_x, num_y); \
        } \
        \
        try { \
            /* call function */ \
            npdivs::np_divs( \
                    x_bags_ms, num_x, \
                    y_bags_ms, num_y, \
                    div_funcs, \
                    results_ms, \
                    make_div_params(div_params)); \
            \
        } catch (std::exception &e) { \
            FILE_LOG(logERROR) << e.what(); \
            for (size_t i = 0; i < num_div_specs; i++) { \
                for (size_t j = 0; j < num_x * num_y; j++) { \
                    results[i][j] = -1; \
                } \
            } \
            \
            delete[] x_bags_ms; \
            delete[] y_bags_ms; \
            delete[] results_ms; \
        } \
        delete[] x_bags_ms; \
        delete[] y_bags_ms; \
        delete[] results_ms; \
    }
NPDIVS(float);
NPDIVS(double);
#undef NPDIVS

////////////////////////////////////////////////////////////////////////////////
// training functions

template <typename Scalar, typename label_type>
SDM<Scalar, label_type> *train_sdm_(
        const Scalar **train_bags,
        size_t num_train,
        size_t dim,
        const size_t * rows,
        const label_type *labels,
        const char *div_func_spec,
        const char *kernel_spec,
        const DivParamsC *div_params,
        const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params,
        size_t tuning_folds,
        double *divs)
{
    // NOTE: allocates Matrix objects for your data
    // caller needs to free them
    Matrix<Scalar> *train_bags_m = make_matrices(
            const_cast<Scalar **>(train_bags),
            num_train, rows, dim);

    npdivs::DivFunc *df = div_func_from_str(string(div_func_spec));
    KernelGroup *kernel = kernel_group_from_str(string(kernel_spec));

    SDM<Scalar, label_type> *sdm = train_sdm(
            train_bags_m, num_train,
            std::vector<label_type>(labels, labels + num_train),
            *df,
            *kernel,
            make_div_params(div_params),
            std::vector<double>(c_vals, c_vals + num_c_vals),
            *svm_params,
            tuning_folds,
            divs);

    delete kernel;
    delete df;

    return sdm;
}

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
        double * divs) \
{ \
    try { \
        SDM<intype, labtype> *sdm = train_sdm_( \
                train_bags, num_train, dim, rows, labels, \
                div_func_spec, kernel_spec, div_params, c_vals, num_c_vals, \
                svm_params, tuning_folds, divs); \
        classname *ret = (classname *) malloc(sizeof(classname *)); \
        ret->sdm = sdm; \
        return ret; \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        return NULL; \
    } \
}
TRAIN(SDM_ClassifyD, double, int);
TRAIN(SDM_ClassifyF, float,  int);
TRAIN(SDM_RegressD,  double, double);
TRAIN(SDM_RegressF,  float,  double);
#undef TRAIN

////////////////////////////////////////////////////////////////////////////////
// prediction functions

// single item, label only
#define PRED(classname, intype, labtype) \
    labtype classname##_predict( \
        const classname * sdm, \
        const intype * test_bag, \
        size_t rows) { \
    try { \
        labtype ret; \
        classname##_predict_many(sdm, &test_bag, 1, &rows, &ret); \
        return ret; \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        return -0xdead; \
    } \
}
PRED(SDM_ClassifyD, double, int);
PRED(SDM_ClassifyF, float,  int);
PRED(SDM_RegressD, double, double);
PRED(SDM_RegressF, float,  double);
#undef PRED


// single item, with decision values
// (allocating the storage for the values and changing vals to point to it)
#define PRED_V(classname, intype, labtype) \
    labtype classname##_predict_vals( \
        const classname * sdm, \
        const intype * test_bag,\
        size_t rows,\
        double ** vals,\
        size_t * num_vals) { \
    try { \
        labtype ret; \
        classname##_predict_many_vals( \
                sdm, &test_bag, 1, &rows, &ret, &vals, num_vals); \
        return ret; \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        vals[0] = NULL; \
        *num_vals = 0; \
        return -0xdead; \
    } \
}
PRED_V(SDM_ClassifyD, double, int);
PRED_V(SDM_ClassifyF, float,  int);
PRED_V(SDM_RegressD, double, double);
PRED_V(SDM_RegressF, float,  double);
#undef PRED_V


// several items, labels only
#define PRED_M(classname, intype, labtype) \
    void classname##_predict_many( \
        const classname * sdm, \
        const intype ** test_bags, \
        size_t num_test, \
        const size_t * rows, \
        labtype * labels) { \
    try { \
        Matrix<intype> *test_bags_m = make_matrices( \
                const_cast<intype **>(test_bags), num_test, rows, \
                sdm->sdm->getDim()); \
        \
        vector<labtype> labels_v = sdm->sdm->predict(test_bags_m, num_test); \
        \
        delete[] test_bags_m; \
        std::copy(labels_v.begin(), labels_v.end(), labels); \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        for (size_t i = 0; i < num_test; i++) \
            labels[i] = -0xdead; \
    } \
}
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
        size_t * num_vals) { \
    try { \
        Matrix<intype> *test_bags_m = make_matrices( \
                const_cast<intype **>(test_bags), num_test, rows, \
                sdm->sdm->getDim()); \
        \
        vector< vector<double> > vals_v; \
        \
        vector<labtype> labels_v = \
            sdm->sdm->predict(test_bags_m, num_test, vals_v); \
        \
        delete[] test_bags_m; \
        std::copy(labels_v.begin(), labels_v.end(), labels); \
        \
        vals[0] = (double **) std::malloc(num_test * sizeof(double *)); \
        size_t real_num_vals = vals_v[0].size(); \
        *num_vals = real_num_vals; \
        \
        for (size_t i = 0; i < num_test; i++) { \
            vals[0][i] = (double*) std::malloc(real_num_vals * sizeof(double));\
            std::copy(vals_v[i].begin(), vals_v[i].end(), vals[0][i]); \
        } \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        for (size_t i = 0; i < num_test; i++) \
            labels[i] = -0xdead; \
        vals[0] = NULL; \
        *num_vals = 0; \
    } \
}
PRED_MV(SDM_ClassifyD, double, int);
PRED_MV(SDM_ClassifyF, float,  int);
PRED_MV(SDM_RegressD, double, double);
PRED_MV(SDM_RegressF, float,  double);
#undef PRED_MV



////////////////////////////////////////////////////////////////////////////////
// cross-validation

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
        size_t tuning_folds) \
{ \
    try { \
        Matrix<intype> *bags_m = make_matrices( \
                const_cast<intype **>(bags), num_bags, rows, dim); \
        \
        npdivs::DivFunc *df = div_func_from_str(string(div_func_spec)); \
        KernelGroup *kernel = kernel_group_from_str(string(kernel_spec)); \
        \
        double acc = crossvalidate( \
                bags_m, num_bags, \
                vector<labtype>(labels, labels + num_bags), \
                *df, *kernel, \
                make_div_params(div_params), \
                folds, num_cv_threads, \
                (bool) project_all_data, (bool) shuffle_order, \
                vector<double>(c_vals, c_vals + num_c_vals), \
                *svm_params, \
                tuning_folds); \
        \
        delete kernel; \
        delete df; \
        delete[] bags_m; \
        \
        return acc; \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        return -1; \
    } \
}
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
        size_t tuning_folds) \
{ \
    try { \
        KernelGroup *kernel = kernel_group_from_str(string(kernel_spec)); \
        \
        double acc = crossvalidate<labtype>( \
                divs, num_bags, \
                vector<labtype>(labels, labels + num_bags), \
                *kernel, \
                folds, num_cv_threads, \
                (bool) project_all_data, (bool) shuffle_order, \
                vector<double>(c_vals, c_vals + num_c_vals), \
                *svm_params, \
                tuning_folds); \
        \
        delete kernel; \
        return acc; \
    } catch (std::exception &e) { \
        FILE_LOG(logERROR) << e.what(); \
        return -1; \
    } \
}
CV_divs(classify, int);
CV_divs(regress,  double);
#undef CV_divs
