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

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <boost/format.hpp>
#include <boost/thread.hpp>

namespace sdm {

// explicit instantiations

template class SDM<float>;
template class SDM<double>;


template SDM<float> * train_sdm<float>(
    const flann::Matrix<float> *train_bags, size_t num_train,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params, const std::vector<double> &c_vals,
    const svm_parameter &svm_params, size_t tuning_folds, double* divs);

template SDM<double> * train_sdm<double>(
    const flann::Matrix<double> *train_bags, size_t num_train,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params, const std::vector<double> &c_vals,
    const svm_parameter &svm_params, size_t tuning_folds, double* divs);


template double crossvalidate<float>(
    const flann::Matrix<float> *bags, size_t num_bags,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds, size_t num_cv_threads, bool project_all_data,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    double* divs);

template double crossvalidate<double>(
    const flann::Matrix<double> *bags, size_t num_bags,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds, size_t num_cv_threads, bool project_all_data,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    double* divs);



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
        if (is_const) {
            FILE_LOG(logDEBUG2) << "near-constant kernel";
            return true;
        }

        // TODO: other tests?

        return false;
    }


    // cross-validation over possible svm / kernel parameters
    // this is the single-threaded version, called by the multithreaded below
    std::vector< std::pair<size_t, size_t> > tune_params_single(
            const double* divs, size_t num_bags,
            const std::vector<int> &labels,
            const Kernel * const * kernels, size_t num_kernels,
            const std::vector<double> &c_vals,
            svm_parameter svm_params,
            size_t folds,
            size_t *final_correct)
    {
        typedef std::pair<size_t, size_t> config;

        // keep track of the best kernel/C combos
        // keep all with same acc, to avoid bias towards ones we see earlier
        std::vector<config> best_configs;
        size_t best_correct = 0;

        if (num_kernels == 0) {
            *final_correct = best_correct;
            return best_configs;
        }

        // make our svm_problem that we'll be reusing throughout
        svm_problem *prob = new svm_problem;
        prob->l = num_bags;
        prob->y = new double[num_bags];
        for (size_t i = 0; i < num_bags; i++)
            prob->y[i] = (double) labels[i];

        // store un-transformed divs in the problem just so it's all allocated
        store_kernel_matrix(*prob, divs, true);

        // make a copy of divergences so we can mangle it
        double *km = new double[num_bags * num_bags];

        // used to store labels into during CV
        double *cv_labels = new double[num_bags];

        for (size_t k = 0; k < num_kernels; k++) {
            // turn into a kernel matrix
            std::copy(divs, divs + num_bags * num_bags, km);
            kernels[k]->transformDivergences(km, num_bags);
            project_to_symmetric_psd(km, num_bags);

            // is it a constant matrix or something else awful?
            if (num_kernels != 1 && terrible_kernel(km, num_bags)) {
                FILE_LOG(logDEBUG1) << "tuning: skipping terrible kernel "
                    << kernels[k]->name();
                continue;
            }

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

                FILE_LOG(logDEBUG2) << "tuning: " <<
                    num_correct << "/" << num_bags <<
                    " by " << kernels[k]->name() << ", C = " << c_vals[ci];


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

        *final_correct = best_correct;
        return best_configs;
    }

    // functor object to do multithreading of tune_params
    struct tune_params_worker : boost::noncopyable {
    protected:
        typedef std::pair<size_t, size_t> config;

        const double* divs;
        const size_t num_bags;
        const std::vector<int> &labels;
        const Kernel * const * kernels;
        const size_t num_kernels;
        const std::vector<double> &c_vals;
        const svm_parameter &svm_params;
        const size_t folds;
        std::vector<config> *results;
        size_t *num_correct;

    public:
        tune_params_worker(
            const double* divs,
            const size_t num_bags,
            const std::vector<int> &labels,
            const Kernel * const * kernels,
            const size_t num_kernels,
            const std::vector<double> &c_vals,
            const svm_parameter &svm_params,
            const size_t folds,
            std::vector<config> *results,
            size_t *num_correct)
        :
            divs(divs), num_bags(num_bags), labels(labels),
            kernels(kernels), num_kernels(num_kernels), c_vals(c_vals),
            svm_params(svm_params), folds(folds),
            results(results), num_correct(num_correct)
        {}

        void operator()() {
            *results = tune_params_single(divs, num_bags, labels,
                    kernels, num_kernels, c_vals,
                    svm_params, folds, num_correct);
        }
    };

    // cross-validation over possible svm / kernel parameters
    //
    // This is the multithreaded version, which combines results from the
    // single-threaded version above. We split up a certain number of kernels
    // to each thread. Each thread does all the C values for a given kernel,
    // to avoid having to re-project matrices. It might make sense to split
    // that up in certain cases in the future.
    std::pair<size_t, size_t> tune_params(
            const double* divs, size_t num_bags,
            const std::vector<int> &labels,
            const boost::ptr_vector<Kernel> &kernels,
            const std::vector<double> &c_vals,
            const svm_parameter &svm_params,
            size_t folds,
            size_t num_threads)
    {
        typedef std::pair<size_t, size_t> config;

        size_t num_kernels = kernels.size();

        if (num_kernels == 0) {
            throw std::domain_error("no kernels in the kernel group");
        } else if (num_kernels == 1 && c_vals.size() == 1) {
            // only one option, we already know what's best
            return config(0, 0);
        }


        // want to be able to take sub-lists of kernels.
        // this is like c_array(), but constness is slightly different and
        // it works in old, old boosts.
        const Kernel * const * kern_array =
            reinterpret_cast<const Kernel* const*>(&kernels.begin().base()[0]);

        // how many threads are we using?
        num_threads = npdivs::get_num_threads(num_threads);
        if (num_threads > num_kernels)
            num_threads = num_kernels;

        if (num_threads == 1) {
            // don't actually make a new thread if it's just 1-threaded
            size_t num_correct;
            return pick_rand(tune_params_single(divs, num_bags, labels,
                        kern_array, num_kernels, c_vals, svm_params, folds,
                        &num_correct));
        }

        // grunt work to set up multithreading
        boost::ptr_vector<tune_params_worker> workers;
        boost::thread_group worker_threads;

        std::vector< std::vector<config> > results(num_threads);
        std::vector<size_t> nums_correct(num_threads, 0);

        size_t kerns_per_thread = (size_t)
                std::ceil(double(num_kernels) / num_threads);
        size_t kern_start = 0;

        // give each thread a few kernels and get their most-accurate configs
        for (size_t i = 0; i < num_threads; i++) {
            size_t n_kerns =
                std::min(kern_start+kerns_per_thread, num_kernels)
                - kern_start;

            workers.push_back(new tune_params_worker(
                        divs, num_bags, labels,
                        kern_array + kern_start, n_kerns,
                        c_vals, svm_params, folds,
                        &results[i], &nums_correct[i]
            ));

            worker_threads.create_thread(boost::ref(workers[i]));

            kern_start += kerns_per_thread;
        }
        worker_threads.join_all();

        // get all the best configs into one vector
        size_t best_correct = *std::max_element(
                nums_correct.begin(), nums_correct.end());
        std::vector<config> best_configs;

        for (size_t i = 0; i < num_threads; i++)
            if (nums_correct[i] == best_correct)
                best_configs.insert(best_configs.end(),
                        results[i].begin(), results[i].end());

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

    std::string SVMtoString(const svm_model &model) {
        std::stringstream ss (std::stringstream::in | std::stringstream::out);

        int n = model.nr_class;
        int l = model.l;
        int n_choose_two = (n * (n-1)) / 2;

        ss << "nr_class: " << n << "; # SV: " << l << "\n";
        ss << "rho:";
        for (int i = 0; i < n_choose_two; i++)
            ss << " " << model.rho[i];
        ss << "\n";

        if (model.label != NULL) {
            ss << "labels:";
            for (int i = 0; i < n; i++)
                ss << " " << model.label[i];
            ss << "\n";
        }

        if (model.probA != NULL) {
            ss << "probA:";
            for (int i = 0; i < n_choose_two; i++)
                ss << " " << model.probA[i];
            ss << "\n";
        }

        if (model.probB != NULL) {
            ss << "probB:";
            for (int i = 0; i < n_choose_two; i++)
                ss << " " << model.probB[i];
            ss << "\n";
        }

        if (model.nSV) {
            ss << "nSVs:";
            for (int i = 0; i < n; i++)
                ss << " " << model.nSV[i];
            ss << "\n";
        }

        ss << "sv_coef:\n";
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < l; j++) {
                ss << " " << model.sv_coef[i][j];
            }
            ss << "\n";
        }
        ss << "\n";

        ss << "not printing SVs out of laziness\n";

        return ss.str();
    }

} // end namespace detail

} // end namespace sdm
