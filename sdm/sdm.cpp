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
#include "sdm/basics.hpp"

#include "sdm/sdm.hpp"

#include <boost/format.hpp>

namespace sdm {

template <typename Scalar>
std::string SDM<Scalar>::name() const {
    return (boost::format("SDM(%s, %s, C=%g, %d training)")
            % div_func->name() % kernel->name()
            % svm.param.C % num_train).str();
}

// explicit instantiations

template class SDM<float>;
template class SDM<double>;

template SDM<float> * train_sdm(
    const flann::Matrix<float> *train_bags, size_t num_train,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params, const std::vector<double> &c_vals,
    const svm_parameter &svm_params, size_t tuning_folds);

template SDM<double> * train_sdm(
    const flann::Matrix<double> *train_bags, size_t num_train,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params, const std::vector<double> &c_vals,
    const svm_parameter &svm_params, size_t tuning_folds);


template double crossvalidate(
    const flann::Matrix<float> *bags, size_t num_bags,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params, size_t folds, bool project_all_data,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds);

template double crossvalidate(
    const flann::Matrix<double> *bags, size_t num_bags,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params, size_t folds, bool project_all_data,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds);

// helper function implementations
namespace detail {

    // copy kernel values into an svm_problem structure
    void store_kernel_matrix(svm_problem &prob, const double *divs, bool alloc)
    {
        size_t n = prob.l;
        if (alloc) prob.x = new svm_node*[n];

        for (size_t i = 0; i < n; i++) {
            if (alloc) prob.x[i] = new svm_node[n+2];
            prob.x[i][0].value = i+1;
            for (size_t j = 0; j < n; j++) {
                prob.x[i][j+1].index = j+1;
                prob.x[i][j+1].value = divs[i*n + j];
            }
            prob.x[i][n+1].index = -1;
        }
    }

    void print_null(const char *s) {}

    // see whether a kernel is just so horrendous we shouldn't bother
    bool terrible_kernel(double* km, size_t n, double const_thresh) {
        const double l = n*n;

        // is it all a constant?
        bool is_const = true;
        const double v = km[0];
        const_thresh = std::max(const_thresh, v*const_thresh);
        for (size_t i = 1; i < l; i++) {
            if (std::abs(km[i] - v) > const_thresh) {
                is_const = false;
                break;
            }
        }
        if (is_const) { // TODO: real logging
            //fprintf(stderr, "Skipping tuning over constant kernel matrix\n");
            return true;
        }

        // TODO: other tests?

        return false;
    }


    // cross-validation over possible svm / kernel parameters
    // TODO: optionally parallelize tuning
    std::pair<size_t, size_t> tune_params(
            const double* divs, size_t num_bags,
            const std::vector<int> &labels,
            const boost::ptr_vector<Kernel> &kernels,
            const std::vector<double> &c_vals,
            svm_parameter &svm_params,
            size_t folds)
    {
        typedef std::pair<size_t, size_t> config;
        size_t num_kernels = kernels.size();

        if (num_kernels == 0) {
            throw std::domain_error("no kernels in the kernel group");
        } else if (num_kernels == 1 && c_vals.size() == 1) {
            return config(0, 0);
        }

        // make our svm_problem that we'll be reusing throughout
        svm_problem *prob = new svm_problem;
        prob->l = num_bags;
        prob->y = new double[num_bags];
        for (size_t i = 0; i < num_bags; i++)
            prob->y[i] = (double) labels[i];

        // store un-transformed divs in the problem just so it's all allocated
        store_kernel_matrix(*prob, divs, true);

        // keep track of the best kernel/C combos
        // keep all with same acc, to avoid bias towards ones we see earlier
        std::vector<config> best_configs;
        size_t best_correct = 0;

        // make a copy of divergences so we can mangle it
        double *km = new double[num_bags * num_bags];

        // used to store labels into during CV
        double *cv_labels = new double[num_bags];

        for (size_t k = 0; k < num_kernels; k++) {
            // turn into a kernel matrix
            std::copy(divs, divs + num_bags * num_bags, km);
            kernels[k].transformDivergences(km, num_bags);
            project_to_symmetric_psd(km, num_bags);

            // is it a constant matrix or something else awful?
            if (num_kernels != 1 && terrible_kernel(km, num_bags))
                continue;

            // store in the svm_problem
            store_kernel_matrix(*prob, km, false);

            for (size_t ci = 0; ci < c_vals.size(); ci++) {
                // do SVM cross-validation with these params
                svm_params.C = c_vals[ci];
                svm_cross_validation(prob, &svm_params, folds, cv_labels);

                size_t num_correct = 0;
                for (size_t i = 0; i < num_bags; i++)
                    if (cv_labels[i] == labels[i])
                        num_correct++;

                if (num_correct >= best_correct) {
                    if (num_correct > best_correct) {
                        best_configs.clear();
                        best_correct = num_correct;
                    }
                    best_configs.push_back(std::make_pair(k, ci));
                }
            }
        }

        delete[] km;
        delete[] cv_labels;

        for (size_t i = 0; i < num_bags; i++)
            delete[] prob->x[i];
        delete[] prob->x;
        delete[] prob->y;
        delete prob;

        return pick_rand(best_configs);
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
}

} // end namespace
