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
#include "sdm/sdm_c.h"
#include "sdm/kernels/from_str.hpp"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <stdexcept>
#include <iostream>
#include <np-divs/matrix_io.hpp>
#include <boost/exception/diagnostic_information.hpp>

#include <flann/util/matrix.h>

#include <np-divs/div-funcs/from_str.hpp>

using namespace sdm;
using namespace npdivs;
using std::string;
using std::vector;
using flann::Matrix;

////////////////////////////////////////////////////////////////////////////////

struct SDMObjDouble_s { SDM<double> *sdm; };
struct SDMObjFloat_s  { SDM<float>  *sdm; };

const char *getName(SDMObjDouble sdm) { return sdm.sdm->name().c_str(); }
const char *getName(SDMObjFloat  sdm) { return sdm.sdm->name().c_str(); }

////////////////////////////////////////////////////////////////////////////////

const struct svm_parameter default_svm_params = {
    C_SVC, // svm_type
    PRECOMPUTED, // kernel_type
    0,    // degree - not used
    0,    // gamma - not used
    0,    // coef0 - not used
    1024, // cache_size, in MB
    1e-3, // eps
    1,    // C
    0,    // nr_weight
    NULL, // weight_label
    NULL, // weight
    0,    // nu - not used
    0,    // p - not used
    1,    // shrinking
    1     // probability
};


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
// training functions

template<typename T>
SDM<T> *train_sdm_(
        const T **train_bags,
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
        double *divs)
{
    Matrix<T> *train_bags_m = make_matrices(const_cast<T **>(train_bags),
            num_train, rows, dim);

    npdivs::DivFunc *df = div_func_from_str(string(div_func_spec));
    KernelGroup *kernel = kernel_group_from_str(string(kernel_spec));

    SDM<T> *sdm = train_sdm(
            train_bags_m, num_train,
            std::vector<int>(labels, labels + num_train),
            *df,
            *kernel,
            make_div_params(div_params),
            std::vector<double>(c_vals, c_vals + num_c_vals),
            *svm_params,
            tuning_folds,
            divs);

    delete kernel;
    delete df;
    delete[] train_bags_m;

    return sdm;
}

SDMObjDouble train_sdm_double(
        const double **train_bags, size_t num_train,
        size_t dim, const size_t * rows,
        const int *labels, const char *div_func_spec, const char *kernel_spec,
        const DivParamsC *div_params, const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params, size_t tuning_folds,
        double *divs)
{
    SDM<double> *sdm = train_sdm_(train_bags, num_train, dim, rows, labels,
            div_func_spec, kernel_spec, div_params, c_vals, num_c_vals,
            svm_params, tuning_folds, divs);
    SDMObjDouble ret;
    ret.sdm = sdm;
    return ret;
}

SDMObjFloat train_sdm_float(
        const float **train_bags, size_t num_train,
        size_t dim, const size_t * rows,
        const int *labels, const char *div_func_spec, const char *kernel_spec,
        const DivParamsC *div_params, const double *c_vals, size_t num_c_vals,
        const struct svm_parameter *svm_params, size_t tuning_folds,
        double *divs)
{
    SDM<float> *sdm = train_sdm_(train_bags, num_train, dim, rows, labels,
            div_func_spec, kernel_spec, div_params, c_vals, num_c_vals,
            svm_params, tuning_folds, divs);
    SDMObjFloat ret;
    ret.sdm = sdm;
    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// prediction functions

// single item, label only
int sdm_predict_double(
        const SDMObjDouble *sdm,
        const double * test_bag, size_t rows)
{
    int ret;
    sdm_predict_many_double(sdm, &test_bag, 1, &rows, &ret);
    return ret;
}

int sdm_predict_float(
        const SDMObjFloat *sdm,
        const float * test_bag, size_t rows)
{
    int ret;
    sdm_predict_many_float(sdm, &test_bag, 1, &rows, &ret);
    return ret;
}

// single item, with decision values
// (allocating the storage for the values and changing vals to point to it)
int sdm_predict_vals_double(
        const SDMObjDouble *sdm,
        const double * test_bag, size_t rows,
        double ** vals, size_t * num_vals)
{
    int ret;
    sdm_predict_many_vals_double(
            sdm, &test_bag, 1, &rows, &ret, &vals, num_vals);
    return ret;
}
int sdm_predict_vals_float(
        const SDMObjFloat *sdm,
        const float * test_bag, size_t rows,
        double ** vals, size_t * num_vals)
{
    int ret;
    sdm_predict_many_vals_float(
            sdm, &test_bag, 1, &rows, &ret, &vals, num_vals);
    return ret;
}


// several items, labels only
void sdm_predict_many_double(
        const SDMObjDouble *sdm,
        const double ** test_bags, size_t num_test, const size_t * rows,
        int *labels)
{
    Matrix<double> *test_bags_m = make_matrices(
            const_cast<double **>(test_bags), num_test, rows,
            sdm->sdm->getDim());

    vector<int> labels_v = sdm->sdm->predict(test_bags_m, num_test);

    delete[] test_bags_m;
    std::copy(labels_v.begin(), labels_v.end(), labels);
}


void sdm_predict_many_float(
        const SDMObjFloat *sdm,
        const float ** test_bags, size_t num_test, const size_t * rows,
        int *labels)
{
    Matrix<float> *test_bags_m = make_matrices(
            const_cast<float **>(test_bags), num_test, rows,
            sdm->sdm->getDim());

    vector<int> labels_v = sdm->sdm->predict(test_bags_m, num_test);

    delete[] test_bags_m;
    std::copy(labels_v.begin(), labels_v.end(), labels);
}

// several items, with decision values
void sdm_predict_many_vals_double(
        const SDMObjDouble *sdm,
        const double ** test_bags, size_t num_test, const size_t * rows,
        int *labels,
        double *** vals, size_t * num_vals)
{
    Matrix<double> *test_bags_m = make_matrices(
            const_cast<double **>(test_bags), num_test, rows,
            sdm->sdm->getDim());

    vector< vector<double> > vals_v;

    vector<int> labels_v = sdm->sdm->predict(test_bags_m, num_test, vals_v);

    delete[] test_bags_m;
    std::copy(labels_v.begin(), labels_v.end(), labels);

    vals[0] = (double **) std::malloc(num_test * sizeof(double *));
    size_t real_num_vals = vals_v[0].size();
    *num_vals = real_num_vals;

    for (size_t i = 0; i < num_test; i++) {
        vals[0][i] = (double *) std::malloc(real_num_vals * sizeof(double));
        std::copy(vals_v[i].begin(), vals_v[i].end(), vals[0][i]);
    }
}


void sdm_predict_many_vals_float(
        const SDMObjFloat *sdm,
        const float ** test_bags, size_t num_test, const size_t * rows,
        int *labels,
        double *** vals, size_t * num_vals)
{
    Matrix<float> *test_bags_m = make_matrices(
            const_cast<float **>(test_bags), num_test, rows,
            sdm->sdm->getDim());

    vector< vector<double> > vals_v;

    vector<int> labels_v = sdm->sdm->predict(test_bags_m, num_test, vals_v);

    delete[] test_bags_m;
    std::copy(labels_v.begin(), labels_v.end(), labels);

    vals[0] = (double **) std::malloc(num_test * sizeof(double *));
    size_t real_num_vals = vals_v[0].size();
    *num_vals = real_num_vals;

    for (size_t i = 0; i < num_test; i++) {
        vals[0][i] = (double *) std::malloc(real_num_vals * sizeof(double));
        std::copy(vals_v[i].begin(), vals_v[i].end(), vals[0][i]);
    }
}


////////////////////////////////////////////////////////////////////////////////
// cross-validation

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
        size_t tuning_folds)
{
    Matrix<double> *bags_m = make_matrices(
            const_cast<double **>(bags), num_bags, rows, dim);
    npdivs::matrix_array_to_csv(std::cout, bags_m, num_bags);
    
    npdivs::DivFunc *df = div_func_from_str(string(div_func_spec));
    KernelGroup *kernel = kernel_group_from_str(string(kernel_spec));

    FILELog::ReportingLevel() = logDEBUG2;
    double acc = -1;
    try {
    acc = crossvalidate(
            bags_m, num_bags,
            vector<int>(labels, labels + num_bags),
            *df, *kernel,
            make_div_params(div_params),
            folds, num_cv_threads,
            (bool) project_all_data, (bool) shuffle_order,
            vector<double>(c_vals, c_vals + num_c_vals),
            *svm_params,
            tuning_folds);
    } catch (std::exception &e) {
        printf("%s\n", boost::current_exception_diagnostic_information().c_str());
        printf("%s\n", e.what());
    }

    delete kernel;
    delete df;
    delete[] bags_m;

    return acc;
}

double crossvalidate_bags_float(
        const float **bags,
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
        size_t tuning_folds)
{
    Matrix<float> *bags_m = make_matrices(
            const_cast<float **>(bags), num_bags, rows, dim);
    
    npdivs::DivFunc *df = div_func_from_str(string(div_func_spec));
    KernelGroup *kernel = kernel_group_from_str(string(kernel_spec));

    double acc = crossvalidate(
            bags_m, num_bags,
            vector<int>(labels, labels + num_bags),
            *df, *kernel,
            make_div_params(div_params),
            folds, num_cv_threads,
            (bool) project_all_data, (bool) shuffle_order,
            vector<double>(c_vals, c_vals + num_c_vals),
            *svm_params,
            tuning_folds);

    delete kernel;
    delete df;
    delete[] bags_m;

    return acc;
}

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
        size_t tuning_folds)
{
    KernelGroup *kernel = kernel_group_from_str(string(kernel_spec));

    double acc = crossvalidate(
            divs, num_bags,
            vector<int>(labels, labels + num_bags),
            *kernel,
            folds, num_cv_threads,
            (bool) project_all_data, (bool) shuffle_order,
            vector<double>(c_vals, c_vals + num_c_vals),
            *svm_params,
            tuning_folds);

    delete kernel;
    return acc;
}
