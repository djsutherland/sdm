/*******************************************************************************
 * Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).             *
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
 *     * Neither the name of Carnegie Mellon University nor the                *
 *       names of the contributors may be used to endorse or promote products  *
 *       derived from this software without specific prior written permission. *
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
#ifndef SDM_HPP_
#define SDM_HPP_
#include "sdm/basics.hpp"
#include "sdm/kernels/kernel.hpp"
#include "sdm/kernel_projection.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <np-divs/matrix_arrays.hpp>
#include <np-divs/np_divs.hpp>
#include <flann/util/matrix.h>
#include <svm.h>

namespace sdm {

// TODO memory ownership with this is all screwy...make it clearer
template <typename Scalar>
class SDM {
    const svm_model &svm;
    const svm_problem &svm_prob; // needs to live at least as long as the model
    const NPDivs::DivFunc *div_func;
    const Kernel *kernel;
    const NPDivs::DivParams div_params;
    const size_t num_classes;

    const flann::Matrix<Scalar> *train_bags;
    const size_t num_train;

    public:
        SDM(const svm_model &svm, const svm_problem &svm_prob,
            const NPDivs::DivFunc &div_func, const Kernel &kernel,
            const NPDivs::DivParams &div_params,
            size_t num_classes,
            const flann::Matrix<Scalar> *train_bags, size_t num_train)
        :
            svm(svm), svm_prob(svm_prob),
            kernel(new_clone(kernel)), div_func(new_clone(div_func)),
            div_params(div_params), num_classes(num_classes),
            train_bags(train_bags), num_train(num_train)
        { }

        ~SDM() {
            delete div_func;
            delete kernel;
        }

        void destroyModelAndProb();

        int predict(const flann::Matrix<Scalar> &test_bag) const;
        int predict(const flann::Matrix<Scalar> &test_bag,
                std::vector<double> &vals) const;

        std::vector<int> predict(
                const flann::Matrix<Scalar> *test_bags, size_t num_test)
            const;
        std::vector<int> predict(
                const flann::Matrix<Scalar> *test_bags, size_t num_test,
                std::vector< std::vector<double> > &vals)
            const;


};

// set up default values for training

namespace detail {
    const double cvals[14] = { // 2^-9, 2^-6, ..., 2^30
        1./512., 1./64., 1./8., 1, 1 << 3, 1 << 6, 1 << 9, 1 << 12, 1 << 15,
        1 << 18, 1 << 21, 1 << 24, 1 << 27, 1 << 30
    };
}


const svm_parameter default_svm_params = {
    C_SVC, // svm_type
    PRECOMPUTED, // kernel_type
    0,    // degree - not used
    0,    // gamma - not used
    0,    // coef0 - not used
    100,  // cache_size, in MB
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
const std::vector<double> default_c_vals(detail::cvals, detail::cvals + 14);

// Function to train a new SDM. Note that the caller is responsible for deleting
// the svm and svm_prob attributes.
//
// TODO: support multi-class classification
// TODO: a mass-training method for more than one kernel
// TODO: option to project based on test data too
// TODO: more flexible tuning CV options>
template <typename Scalar>
SDM<Scalar> * train_sdm(
    const flann::Matrix<Scalar> *train_bags, size_t num_train,
    const std::vector<int> &labels,
    const NPDivs::DivFunc &div_func,
    const KernelGroup &kernel_group,
    const NPDivs::DivParams &div_params,
    const std::vector<double> &c_vals = default_c_vals,
    const svm_parameter &svm_params = default_svm_params,
    size_t tuning_folds = 3);

////////////////////////////////////////////////////////////////////////////////
// Helper functions

namespace detail {
    void store_kernel_matrix(svm_problem &prob, double *divs, bool alloc) {
        size_t n = prob.l;
        if (alloc) prob.x = new svm_node*[n];

        for (size_t i = 0; i < n; i++) {
            if (alloc) prob.x[i] = new svm_node[n+2];
            prob.x[i][0].value = i+1;
            for (size_t j = 0; j < n; j++) {
                prob.x[i][j+1].index = j+1;
                prob.x[i][j+1].value = divs[i + j*n];
            }
            prob.x[i][n+1].index = -1;
        }
    }

    void print_null(const char *s) {}
}

template <typename Scalar>
void SDM<Scalar>::destroyModelAndProb() {
    // FIXME: rework SDM memory model to avoid gross const_casts

    svm_free_model_content(const_cast<svm_model*> (&svm));
    delete const_cast<svm_model*>(&svm);

    for (size_t i = 0; i < svm_prob.l; i++)
        delete[] svm_prob.x[i];
    delete[] svm_prob.x;
    delete[] svm_prob.y;
    delete const_cast<svm_problem*>(&svm_prob);
}

////////////////////////////////////////////////////////////////////////////////
// Training

template <typename Scalar>
SDM<Scalar> * train_sdm(
        const flann::Matrix<Scalar> *train_bags, size_t num_train,
        const std::vector<int> &labels,
        const NPDivs::DivFunc &div_func,
        const KernelGroup &kernel_group,
        const NPDivs::DivParams &div_params,
        const std::vector<double> &c_vals,
        const svm_parameter &svm_params,
        size_t tuning_folds)
{   // TODO - logging

    // copy the svm params so we can change them
    svm_parameter svm_p = svm_params;
    svm_p.svm_type = C_SVC;
    svm_p.kernel_type = PRECOMPUTED;

    // make libSVM shut up  -  TODO real logging
    svm_set_print_string_function(&detail::print_null);

    // set up the basic svm_problem, except for the kernel values
    svm_problem *prob = new svm_problem;
    prob->l = num_train;
    prob->y = new double[num_train];
    std::copy(labels.begin(), labels.end(), prob->y);

    // first compute divergences
    flann::Matrix<double>* divs =
        NPDivs::alloc_matrix_array<double>(1, num_train, num_train);
    np_divs(train_bags, num_train, div_func, divs, div_params, false);


    // // cross-validate over possibilities for the svm/kernel parameters
    // for (std::vector<double>::const_iterator sig = sigma_mults.begin();
    //         sig != sigma_mults.end(); ++sig)
    // {
    //     double sigma = *sig * avg_div;

    // }

    const boost::ptr_vector<Kernel> &kernels =
        kernel_group.getTuningVector(divs->ptr(), num_train);
    const Kernel &kernel = kernels[0];

    // build up our best kernel matrix
    kernel.transformDivergences(divs->ptr(), num_train);
    project_to_symmetric_psd(divs->ptr(), num_train);

    // train an SVM on the whole thing
    // TODO after doing CV, won't need to alloc here
    detail::store_kernel_matrix(*prob, divs->ptr(), true);

    // temp values no longer needed
    NPDivs::free_matrix_array(divs, 1);

    const char* error = svm_check_parameter(prob, &svm_p);
    if (error != NULL) {
        std::cerr << "LibSVM parameter error: " << error << std::endl;
        throw std::domain_error(error);
    }
    svm_model *svm = svm_train(prob, &svm_p);
    return new SDM<Scalar>(*svm, *prob, div_func, kernel, div_params,
            svm_get_nr_class(svm), train_bags, num_train);
}

////////////////////////////////////////////////////////////////////////////////
// Prediction

template <typename Scalar>
int SDM<Scalar>::predict(const flann::Matrix<Scalar> &test_bag) const {
    std::vector< std::vector<double> > vals(1);
    return this->predict(&test_bag, 1, vals)[0];
}
template <typename Scalar>
int SDM<Scalar>::predict(const flann::Matrix<Scalar> &test_bag,
        std::vector<double> &val)
const {
    std::vector< std::vector<double> > vals(1);
    const std::vector<int> &pred_labels = this->predict(&test_bag, 1, vals);
    val = vals[0];
    return pred_labels[0];
}

template <typename Scalar>
std::vector<int> SDM<Scalar>::predict(
        const flann::Matrix<Scalar> *test_bags, size_t num_test)
const {
    std::vector< std::vector<double> > vals(num_test);
    return this->predict(test_bags, num_test, vals);
}

template <typename Scalar>
std::vector<int> SDM<Scalar>::predict(
        const flann::Matrix<Scalar> *test_bags, size_t num_test,
        std::vector< std::vector<double> > &vals)
const {
    // TODO: np_divs option to compute things both ways and/or save trees
    // TODO: only compute divergences from support vectors
    double fwd_data[num_train * num_test];
    flann::Matrix<double> forward(fwd_data, num_train, num_test);

    double bwd_data[num_test * num_train];
    flann::Matrix<double> backward(bwd_data, num_test, num_train);

    // compute divergences
    NPDivs::np_divs(train_bags, num_train, test_bags, num_test,
            *div_func, &forward, div_params);
    NPDivs::np_divs(test_bags, num_test, train_bags, num_train,
            *div_func, &backward, div_params);

    // pass through the kernel
    kernel->transformDivergences(forward.ptr(), num_train, num_test);
    kernel->transformDivergences(backward.ptr(), num_test, num_train);

    // we can't project here, so we just symmetrize
    // TODO - symmetrize divergence estimates or kernel estimates?
    for (size_t i = 0; i < num_test; i++)
        for (size_t j = 0; j < num_train; j++)
            backward[i][j] = (forward[j][i] + backward[i][j]) / 2.0;

    const flann::Matrix<double> &divs = backward;

    // figure out which prediction function we want to use
    double (*pred_fn)(const svm_model*, const svm_node*, double*) =
        svm_check_probability_model(&svm)
        ? &svm_predict_probability : &svm_predict_values;

    // we'll reuse this svm_node array for testing
    svm_node kernel_row[num_train+2];
    for (size_t i = 0; i <= num_train; i++)
        kernel_row[i].index = i;
    kernel_row[num_train+1].index = -1;

    // even though those fns return doubles, we'll round to an int because
    // we want integer class labels
    std::vector<int> pred_labels(num_test);

    // predict!
    for (size_t i = 0; i < num_test; i++) {
        // fill in our kernel evaluations
        kernel_row[0].value = -i - 1;
        for (size_t j = 0; j < num_train; j++)
            kernel_row[j+1].value = divs[i][j];

        // get space to store our decision/probability values
        vals[i].resize(num_classes);

        // ask the SVM for a prediction
        double res = pred_fn(&svm, kernel_row, &vals[i][0]);
        pred_labels[i] = std::floor(res + .5);
    }

    return pred_labels;
}



} // end namespace

#endif
