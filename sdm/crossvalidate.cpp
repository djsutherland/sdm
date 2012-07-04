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
#include "sdm/crossvalidate.hpp"

namespace sdm {

template double crossvalidate<float, int>(
    const flann::Matrix<float> *bags, size_t num_bags,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds, size_t num_cv_threads,
    bool project_all_data, bool shuffle_order,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    const double* divs);

template double crossvalidate<double, int>(
    const flann::Matrix<double> *bags, size_t num_bags,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds, size_t num_cv_threads,
    bool project_all_data, bool shuffle_order,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    const double* divs);

template double crossvalidate<float, double>(
    const flann::Matrix<float> *bags, size_t num_bags,
    const std::vector<double> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds, size_t num_cv_threads,
    bool project_all_data, bool shuffle_order,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    const double* divs);

template double crossvalidate<double, double>(
    const flann::Matrix<double> *bags, size_t num_bags,
    const std::vector<double> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds, size_t num_cv_threads,
    bool project_all_data, bool shuffle_order,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    const double* divs);


template double crossvalidate<int>(
    const double *divs, size_t num_bags,
    const std::vector<int> &labels,
    const KernelGroup &kernel_group,
    size_t folds = 10,
    size_t num_cv_threads = 0,
    bool project_all_data = true,
    bool shuffle_order = true,
    const std::vector<double> &c_vals = default_c_vals,
    const svm_parameter &svm_params = default_svm_params,
    size_t tuning_folds = 3);

template double crossvalidate<double>(
    const double *divs, size_t num_bags,
    const std::vector<double> &labels,
    const KernelGroup &kernel_group,
    size_t folds = 10,
    size_t num_cv_threads = 0,
    bool project_all_data = true,
    bool shuffle_order = true,
    const std::vector<double> &c_vals = default_c_vals,
    const svm_parameter &svm_params = default_svm_params,
    size_t tuning_folds = 3);



namespace detail {

// random index < the passed one
ptrdiff_t rand_idx(ptrdiff_t i) {
    if (i < RAND_MAX) {
        return rand() % i;
    } else {
        int n = i / RAND_MAX + 1;
        size_t r = 0;
        for (int j = 0; j < n; j++)
            r += rand();
        return rand() % i;
    }
}

// split a full matrix into testing/training matrices
void copy_from_full_to_split(const double *source,
        double *train, double *test,
        size_t test_start, size_t num_train, size_t num_test)
{
    const size_t test_end = test_start + num_test;
    const size_t num_bags = num_train + num_test;

    double *dest;
    size_t dest_row;
    size_t full_row;

    for (size_t i = 0; i < num_bags; i++) {
        full_row = i * num_bags;

        if (i < test_start) { // into first part of training
            dest = train;
            dest_row = i * num_train;
        } else if (i < test_end) { // into testing
            dest = test;
            dest_row = (i - test_start) * num_train;
        } else { // into second part of training
            dest = train;
            dest_row = (i - num_test) * num_train;
        }

        // first part of training
        for (size_t j = 0; j < test_start; j++)
            dest[dest_row + j] = source[full_row + j];

        // second part of training
        for (size_t j = test_end; j < num_bags; j++)
            dest[dest_row + j - num_test] = source[full_row + j];
    }
}

} // detail
} // sdm
