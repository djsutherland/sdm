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
#include "sdm/kernels.hpp"
#include "sdm/kernel_projection.hpp"

#include <algorithm>
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
    const Kernel &kernel;

    const flann::Matrix<Scalar> *training_bags;
    size_t num_training;

    public:
        SDM(const svm_model &svm, const svm_problem &svm_prob,
            const Kernel &kernel,
            const flann::Matrix<Scalar> *training_bags, size_t num_training)
        :
            svm(svm), svm_prob(svm_prob), kernel(kernel),
            training_bags(training_bags), num_training(num_training)
        { }

        std::vector<bool> predict(
                const flann::Matrix<Scalar> *test_dists, size_t num_test)
            const;

        std::vector<bool> predict(
                const flann::Matrix<Scalar> *test_dists, size_t num_test,
                std::vector<double> &values)
            const;


};

// TODO: support multi-class classification
// TODO: a mass-training method for more than one kernel
template <typename Scalar>
SDM<Scalar> train_sdm(
    const flann::Matrix<Scalar> *training_bags, size_t num_train,
    const std::vector<int> &labels,
    const Kernel &kernel,
    const NPDivs::DivParams &div_params,
    const svm_parameter &svm_params
    // TODO: cross-validation options
);

////////////////////////////////////////////////////////////////////////////////
// Template member function implementations

void store_kernel_matrix(svm_problem &prob, double *divs, bool alloc) {
    size_t n = prob.l;
    if (alloc) prob.x = new svm_node*[n];

    for (size_t i = 0; i < n; i++) {
        if (alloc) prob.x[i] = new svm_node[n];
        for (size_t j = 0; j < n; j++) {
            prob.x[i][j].index = j;
            prob.x[i][j].value = divs[i + j*n];
        }
    }
}

template <typename Scalar>
SDM<Scalar> train_sdm(
        const flann::Matrix<Scalar> *training_bags, size_t num_train,
        const std::vector<int> &labels,
        const Kernel &kernel,
        const NPDivs::DivParams &div_params,
        const svm_parameter &svm_params)
{   // TODO - logging

    // copy the svm params so we can change them
    svm_parameter svm_p = svm_params;
    svm_p.svm_type = C_SVC;
    svm_p.kernel_type = PRECOMPUTED;

    // set up the basic svm_problem, except for the kernel values
    svm_problem *prob = new svm_problem;
    prob->l = num_train;
    prob->y = new double[num_train];
    std::copy(labels.begin(), labels.end(), prob->y);
    // for (size_t i = 0; i < num_train; i++)
    //     prob->y[i] = labels[i];

    // first compute divergences
    flann::Matrix<double>* divs =
        NPDivs::alloc_matrix_array<double>(1, num_train, num_train);
    np_divs(training_bags, num_train, kernel.getDivFunc(), divs,
            div_params, false);

    // TODO cross-validate over possibilities for the svm/kernel parameters

    // build up our best kernel matrix
    kernel.transformDivergences(divs->ptr(), num_train);
    project_to_kernel(divs->ptr(), num_train);

    // train an SVM on the whole thing
    // TODO after doing CV, won't need to alloc here
    store_kernel_matrix(*prob, divs->ptr(), true);
    const char* error = svm_check_parameter(prob, &svm_p);
    if (error != NULL)
        throw std::domain_error(error);
    svm_model *svm = svm_train(prob, &svm_p);

    return SDM<Scalar>(*svm, *prob, kernel, training_bags, num_train);
}

} // end namespace

#endif
