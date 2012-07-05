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
#ifndef SDM_TUNE_PARAMS_HPP_
#define SDM_TUNE_PARAMS_HPP_
#include "sdm/basics.hpp"
#include "sdm/kernels/kernel.hpp"
#include "sdm/kernel_projection.hpp"
#include "sdm/log.hpp"

#include <limits>

#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

#include <np-divs/np_divs.hpp>

#include <svm.h>

namespace sdm {
namespace detail {

// the main thing: cross-validation over possible svm / kernel parameters
template <typename label_type>
std::pair<size_t, size_t> tune_params(
        const double* divs, size_t num_bags,
        const std::vector<label_type> &labels,
        const boost::ptr_vector<Kernel> &kernels,
        const std::vector<double> &c_vals,
        const svm_parameter &svm_params,
        size_t folds,
        size_t num_threads);

// pick random element from a vector
template <typename T>
T pick_rand(std::vector<T> vec) {
    size_t n = vec.size();
    if (n == 0) {
        BOOST_THROW_EXCEPTION(std::length_error(
                    "picking from empty vector"));
    } else if (n == 1) {
        return vec[0];
    } else {
        // use c++'s silly random number generator; good enough for this
        // note that srand() should've been called before this
        return vec[std::rand() % n];
    }
}

// copy kernel values into an svm_problem structure
void store_kernel_matrix(svm_problem &prob, const double *divs, bool alloc);

// see whether a kernel is just so horrendous we shouldn't bother
bool terrible_kernel(double* km, size_t n, double const_thresh=1e-4);


////////////////////////////////////////////////////////////////////////////////
// Template implementations

// cross-validation over possible svm / kernel parameters
// this is the single-threaded version, called by the multithreaded below
//
// Score means:
//    * number of correct classifications for integral label_type
//    * negative mean squared error for double label_type
template <typename label_type>
std::vector< std::pair<size_t, size_t> > tune_params_single(
        const double* divs, size_t num_bags,
        const std::vector<label_type> &labels,
        const Kernel * const * kernels, size_t num_kernels,
        const std::vector<double> &c_vals,
        svm_parameter svm_params,
        size_t folds,
        double *final_score)
{
    typedef std::pair<size_t, size_t> config;
    
    // this only works for int or double label_types
    BOOST_STATIC_ASSERT((
        boost::is_same<label_type, int>::value ||
        boost::is_same<label_type, double>::value
    ));

    // keep track of the best kernel/C combos
    // keep all with same acc, to avoid bias towards ones we see earlier
    std::vector<config> best_configs;
    double best_score = -std::numeric_limits<double>::infinity();

    if (num_kernels == 0) {
        *final_score = best_score;
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
        if (terrible_kernel(km, num_bags)) {
            FILE_LOG(logDEBUG1) << "tuning: skipping terrible kernel "
                << kernels[k]->name();
            continue;
        }

        // store in the svm_problem
        store_kernel_matrix(*prob, km, false);

        // TODO: add epsilon-SVR's epsilon (svm_params.p) to the grid search
        for (size_t ci = 0; ci < c_vals.size(); ci++) {
            // do SVM cross-validation with these params
            svm_params.C = c_vals[ci];
            svm_cross_validation(prob, &svm_params, folds, cv_labels);

            double score = 0;

            if (boost::is_same<label_type, int>::value) {
                // integer type; get classification accuracy
                for (size_t i = 0; i < num_bags; i++)
                    if (cv_labels[i] == labels[i])
                        score++;

                FILE_LOG(logDEBUG2) << "tuning: " <<
                    score << "/" << num_bags
                     << " by " << kernels[k]->name() << ", C = " << c_vals[ci];


            } else {
                // double type; get *negative* squared err (so max is best)
                for (size_t i = 0; i < num_bags; i++) {
                    double diff = cv_labels[i] - labels[i];
                    score -= diff * diff;
                }
                    
                FILE_LOG(logDEBUG2) << "tuning: " << score
                     << " by " << kernels[k]->name() << ", C = " << c_vals[ci];
            }

            if (score >= best_score) {
                if (score > best_score) {
                    best_configs.clear();
                    best_score = score;
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

    *final_score = best_score;
    return best_configs;
}


// functor object to do multithreading of tune_params
template <typename label_type>
struct tune_params_worker : boost::noncopyable {
protected:
    typedef std::pair<size_t, size_t> config;

    const double* divs;
    const size_t num_bags;
    const std::vector<label_type> &labels;
    const Kernel * const * kernels;
    const size_t num_kernels;
    const std::vector<double> &c_vals;
    const svm_parameter &svm_params;
    const size_t folds;
    std::vector<config> *results;
    double *score;
    boost::exception_ptr &error;

public:
    tune_params_worker(
        const double* divs,
        const size_t num_bags,
        const std::vector<label_type> &labels,
        const Kernel * const * kernels,
        const size_t num_kernels,
        const std::vector<double> &c_vals,
        const svm_parameter &svm_params,
        const size_t folds,
        std::vector<config> *results,
        double *score,
        boost::exception_ptr &error)
    :
        divs(divs), num_bags(num_bags), labels(labels),
        kernels(kernels), num_kernels(num_kernels), c_vals(c_vals),
        svm_params(svm_params), folds(folds),
        results(results), score(score), error(error)
    {}

    void operator()() {
        try {
            *results = tune_params_single(divs, num_bags, labels,
                    kernels, num_kernels, c_vals,
                    svm_params, folds, score);
        } catch (...) {
            error = boost::current_exception();
        }
    }
};

// cross-validation over possible svm / kernel parameters
//
// This is the multithreaded version, which combines results from the
// single-threaded version above. We split up a certain number of kernels
// to each thread. Each thread does all the C values for a given kernel,
// to avoid having to re-project matrices. It might make sense to split
// that up in certain cases in the future.
template <typename label_type>
std::pair<size_t, size_t> tune_params(
        const double* divs, size_t num_bags,
        const std::vector<label_type> &labels,
        const boost::ptr_vector<Kernel> &kernels,
        const std::vector<double> &c_vals,
        const svm_parameter &svm_params,
        size_t folds,
        size_t num_threads)
{
    typedef std::pair<size_t, size_t> config;

    size_t num_kernels = kernels.size();

    if (num_kernels == 0) {
        BOOST_THROW_EXCEPTION(std::domain_error(
                    "no kernels in the kernel group"));
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
        double score;
        return pick_rand(tune_params_single(divs, num_bags, labels,
                    kern_array, num_kernels, c_vals, svm_params, folds,
                    &score));
    }

    // grunt work to set up multithreading
    boost::ptr_vector< tune_params_worker<label_type> > workers;
    std::vector<boost::exception_ptr> errors(num_threads);
    boost::thread_group worker_threads;

    std::vector< std::vector<config> > results(num_threads);
    std::vector<double> scores(num_threads, 0);

    size_t kerns_per_thread = (size_t)
            std::ceil(double(num_kernels) / num_threads);
    size_t kern_start = 0;

    // give each thread a few kernels and get their most-accurate configs
    for (size_t i = 0; i < num_threads; i++) {
        size_t n_kerns =
            std::min(kern_start+kerns_per_thread, num_kernels)
            - kern_start;

        workers.push_back(new tune_params_worker<label_type>(
                    divs, num_bags, labels,
                    kern_array + kern_start, n_kerns,
                    c_vals, svm_params, folds,
                    &results[i], &scores[i],
                    errors[i]
        ));

        worker_threads.create_thread(boost::ref(workers[i]));

        kern_start += kerns_per_thread;
    }
    worker_threads.join_all();
    for (size_t i = 0; i < num_threads; i++)
        if (errors[i])
            boost::rethrow_exception(errors[i]);

    // get all the best configs into one vector
    double best_score = *std::max_element(
            scores.begin(), scores.end());
    std::vector<config> best_configs;

    if (best_score == -std::numeric_limits<double>::infinity()) {
        FILE_LOG(logERROR) << "all kernels were terrible";
        BOOST_THROW_EXCEPTION(std::domain_error("all kernels were terrible"));
    }

    kern_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        if (scores[i] == best_score) {
            for (size_t j = 0; j < results[i].size(); j++) {
                config cfg = results[i][j];
                best_configs.push_back(
                        config(cfg.first + kern_start, cfg.second));
            }
        }
        kern_start += kerns_per_thread;
    }

    return pick_rand(best_configs);
}

} // detail
} // sdm

#endif
