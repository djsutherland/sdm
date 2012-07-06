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
#ifndef SDM_CROSSVALIDATE_HPP_
#define SDM_CROSSVALIDATE_HPP_
#include "sdm/basics.hpp"
#include "sdm/kernels/kernel.hpp"
#include "sdm/kernel_projection.hpp"
#include "sdm/log.hpp"
#include "sdm/defaults.hpp"
#include "sdm/sdm_model.hpp"

#include <cmath>

#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

#include <np-divs/np_divs.hpp>

#include <svm.h>

namespace sdm {

// Function to do cross-validation on the passed distributions.
//
// Specifying 0 folds means leave-one-out; specifying 1 fold or more
// folds than there are bags will cause a std::domain_error.
//
// The project_all_data parameter specifies whether to project the entire
// kernel matrix to PSD, or only the training data for a given fold.
//
// The divs parameter, if passed, should be a pointer to a row-major matrix of
// precomputed divergences for the given bags, as given by
// npdivs::np_divs(...)[i].ptr(). If NULL, will be computed.
template <typename Scalar, typename label_type>
double crossvalidate(
    const flann::Matrix<Scalar> *bags, size_t num_bags,
    const std::vector<label_type> &labels,
    const npdivs::DivFunc &div_func,
    const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds = 10,
    size_t num_cv_threads = 0,
    bool project_all_data = true,
    bool shuffle_order = true,
    const std::vector<double> &c_vals = sdm::default_c_vals,
    const svm_parameter &svm_params = sdm::default_svm_params,
    size_t tuning_folds = 3,
    const double *divs = NULL);

// Cross-validate directly on precomputed divergences.
template <typename label_type>
double crossvalidate(
    const double *divs, size_t num_bags,
    const std::vector<label_type> &labels,
    const KernelGroup &kernel_group,
    size_t folds = 10,
    size_t num_cv_threads = 0,
    bool project_all_data = true,
    bool shuffle_order = true,
    const std::vector<double> &c_vals = default_c_vals,
    const svm_parameter &svm_params = default_svm_params,
    size_t tuning_folds = 3);


////////////////////////////////////////////////////////////////////////////////
// Helper functions

namespace detail {

// random index < the passed one
ptrdiff_t rand_idx(ptrdiff_t i);

// split a full matrix into testing/training matrices
void copy_from_full_to_split(const double *source,
        double *train, double *test,
        size_t test_start, size_t num_train, size_t num_test);

// worker functor class for doing CV
template <typename label_type>
class cv_worker : boost::noncopyable {
    protected:
        typedef std::pair<size_t, size_t> size_pair;

        const double *divs;
        const size_t num_bags;
        const std::vector<label_type> &labels;
        const boost::ptr_vector<Kernel> *kernels;
        const svm_parameter &svm_params;
        const std::vector<double> c_vals;
        const size_t tuning_folds;
        const bool project_all_data;
        const npdivs::DivParams &div_params;

        boost::mutex &jobs_mutex;
        std::queue<size_pair> &jobs;
        boost::mutex &results_mutex;
        double &score;

        boost::exception_ptr &error;

    public:
        cv_worker(
                const double *divs,
                const size_t num_bags,
                const std::vector<label_type> &labels,
                const boost::ptr_vector<Kernel> *kernels,
                const svm_parameter &svm_params,
                const std::vector<double> c_vals,
                const size_t tuning_folds,
                const bool project_all_data,
                const npdivs::DivParams &div_params,
                boost::mutex &jobs_mutex,
                std::queue<size_pair> &jobs,
                boost::mutex &results_mutex,
                double &score,
                boost::exception_ptr &error)
            :
                divs(divs), num_bags(num_bags), labels(labels),
                kernels(kernels), svm_params(svm_params),
                c_vals(c_vals), tuning_folds(tuning_folds),
                project_all_data(project_all_data), div_params(div_params),
                jobs_mutex(jobs_mutex), jobs(jobs),
                results_mutex(results_mutex), score(score),
                error(error)
    {}

        void operator()() {
            size_pair job;
            double this_score = 0;

            try {
                while (true) {
                    { // get a job
                        boost::mutex::scoped_lock the_lock(jobs_mutex);
                        if (jobs.size() == 0)
                            break;
                        job = jobs.front();
                        jobs.pop();
                    }

                    this_score += do_job(job.first, job.second);
                }

                { // write the result
                    boost::mutex::scoped_lock the_lock(results_mutex);
                    score += this_score;
                }
            } catch (...) {
                error = boost::current_exception();
                return;
            }
        }

        double do_job(size_t test_start, size_t test_end) const {
            // testing is in [test_start, test_end)
            size_t num_test = test_end - test_start;
            size_t num_train = num_bags - num_test;

            // set up training labels
            std::vector<label_type> train_labels(num_train);
            std::copy(labels.begin(), labels.begin() + test_start,
                    train_labels.begin());
            std::copy(labels.begin() + test_end, labels.end(),
                    train_labels.begin() + test_start);

            // tune the kernel parameters
            double *train_km = new double[num_train * num_train];
            double *test_km = new double[num_test * num_train];
            copy_from_full_to_split(divs, train_km, test_km,
                    test_start, num_train, num_test);

            svm_parameter svm_p = svm_params;

            const std::pair<size_t, size_t> &best_config = tune_params(
                    train_km, num_train, train_labels, *kernels, c_vals,
                    svm_p, tuning_folds, div_params.num_threads);

            const Kernel &kernel = (*kernels)[best_config.first];
            svm_p.C = c_vals[best_config.second];

            // project either the whole matrix or just the training

            if (project_all_data) {
                double *full_km = new double[num_bags * num_bags];
                std::copy(divs, divs + num_bags * num_bags, full_km);

                kernel.transformDivergences(full_km, num_bags);
                project_to_symmetric_psd(full_km, num_bags);

                copy_from_full_to_split(full_km, train_km, test_km,
                        test_start, num_train, num_test);

                delete[] full_km;

            } else {
                copy_from_full_to_split(
                    divs, train_km, test_km, test_start, num_train, num_test);

                kernel.transformDivergences(train_km, num_train);
                project_to_symmetric_psd(train_km, num_train);

                kernel.transformDivergences(test_km, num_test, num_train);
            }

            // set up svm parameters
            svm_problem prob;
            prob.l = num_train;
            prob.y = new double[num_train];
            for (size_t i = 0; i < num_train; i++)
                prob.y[i] = double(train_labels[i]);
            detail::store_kernel_matrix(prob, train_km, true);

            // check svm params
            const char* error = svm_check_parameter(&prob, &svm_p);
            if (error != NULL) {
                FILE_LOG(logERROR) << "LibSVM parameter error: " << error;
                BOOST_THROW_EXCEPTION(std::domain_error(error));
            }

            // train!
            svm_model *svm = svm_train(&prob, &svm_p);

            npdivs::DivL2 fake_df; // XXX figure out a better way around this
            SDM<float, label_type> sdm(*svm, prob, fake_df, kernel, div_params,
                    svm_get_nr_class(svm), NULL, num_train);

            // predict!
            std::vector<label_type> preds;
            std::vector< std::vector<double> > vals;
            sdm.predict_from_kerns(test_km, num_test, preds, vals);

            double score = 0;
            if (boost::is_same<label_type, int>::value) {
                // how many did we get right?
                for (size_t i = 0; i < num_test; i++)
                    if (preds[i] == labels[test_start + i])
                        score++;

                FILE_LOG(logINFO) << "CV " <<
                    test_start << " - " << test_end-1 << ": "
                    << score << "/" << num_test
                    << " correct by " << kernel.name() << ", C=" << svm_p.C;

            } else {
                // get sum squared error
                for (size_t i = 0; i < num_test; i++) {
                    double diff = preds[i] - labels[test_start + i];
                    score += diff * diff;
                }

                FILE_LOG(logINFO) << "CV " <<
                    test_start << " - " << test_end << ": "
                    << std::sqrt(score / num_test)
                    << " RMSE by " << kernel.name() << ", C=" << svm_p.C;
            }


            // clean up
            delete[] train_km;
            delete[] test_km;
            for (size_t i = 0; i < num_train; i++)
                delete[] prob.x[i];
            delete[] prob.x;
            delete[] prob.y;
            svm_free_model_content(svm);
            delete svm;

            return score;
        }
};


} // detail


////////////////////////////////////////////////////////////////////////////////
// Template implementations


// cross-validation directly from bags
template <typename Scalar, typename label_type>
double crossvalidate(
    const flann::Matrix<Scalar> *bags, size_t num_bags,
    const std::vector<label_type> &labels,
    const npdivs::DivFunc &div_func,
    const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds,
    size_t num_cv_threads,
    bool project_all_data,
    bool shuffle_order,
    const std::vector<double> &c_vals,
    const svm_parameter &svm_params,
    size_t tuning_folds,
    const double *divs)
{
    typedef flann::Matrix<Scalar> Matrix;

    if (folds == 0) {
        folds = num_bags;
    } else if (folds == 1) {
        BOOST_THROW_EXCEPTION(std::domain_error((boost::format(
                    "Can't cross-validate with %d folds...") % folds).str()));
    } else if (folds > num_bags) {
        BOOST_THROW_EXCEPTION(std::domain_error((boost::format(
           "Can't use %d folds with only %d bags.") % folds % num_bags).str()));
    }

    // calculate the full matrix of divergences if necessary
    bool free_divs = false;

    if (divs == NULL) {
        flann::Matrix<double>* divs_mat =
            npdivs::alloc_matrix_array<double>(1, num_bags, num_bags);
        np_divs(bags, num_bags, div_func, divs_mat, div_params, false);

        divs = divs_mat[0].ptr();
        free_divs = true;
        delete[] divs_mat; // doesn't delete content, just the Matrix array
    }

    double acc = crossvalidate(
            divs, num_bags, labels, kernel_group, folds,
            num_cv_threads, project_all_data, shuffle_order,
            c_vals, svm_params, tuning_folds);

    if (free_divs)
        delete[] divs;

    return acc;
}


// cross-validate with precomputed divergences
template <typename label_type>
double crossvalidate(
    const double *divs, size_t num_bags,
    const std::vector<label_type> &labels,
    const KernelGroup &kernel_group,
    size_t folds,
    size_t num_cv_threads,
    bool project_all_data,
    bool shuffle_order,
    const std::vector<double> &c_vals,
    const svm_parameter &svm_params,
    size_t tuning_folds)
{
    // this only works for int or double label_types
    BOOST_STATIC_ASSERT((
        boost::is_same<label_type, int>::value ||
        boost::is_same<label_type, double>::value
    ));

    // check that folds is appropriate
    if (folds == 0) {
        folds = num_bags;
    } else if (folds == 1) {
        BOOST_THROW_EXCEPTION(std::domain_error((boost::format(
                    "Can't cross-validate with %d folds...") % folds).str()));
    } else if (folds > num_bags) {
        BOOST_THROW_EXCEPTION(std::domain_error((boost::format(
           "Can't use %d folds with only %d bags.") % folds % num_bags).str()));
    }

    // copy the svm params so we can change them
    svm_parameter svm_p = svm_params;
    svm_p.kernel_type = PRECOMPUTED;
    // TODO: better handling of svm_type
    if (boost::is_same<label_type, int>::value)
        svm_p.svm_type = C_SVC;
    else
        svm_p.svm_type = EPSILON_SVR;

    // make libSVM log properly
    svm_set_print_string_function(&log_string<logDEBUG4>);


    // ask for the list of kernels to choose from
    const boost::ptr_vector<Kernel> *kernels =
        kernel_group.getTuningVector(divs, num_bags);

    // figure out the numbers of threads to use
    num_cv_threads = npdivs::get_num_threads(num_cv_threads);
    size_t real_cv_threads = std::min(num_cv_threads, folds);

    // don't want to blow up total # of threads when the workers are tuning
    // XXX reorganize so don't need to pass mostly-fake div params
    npdivs::DivParams pred_div_params(3,
        flann::KDTreeSingleIndexParams(), flann::SearchParams(-1),
        std::max(size_t(1), size_t(double(num_cv_threads) / real_cv_threads)));

    // need this for parameter tuning and shuffling
    std::srand(unsigned(std::time(NULL)));

    // reorder the divs/labels, so folds are randomized
    double* shuff_divs = new double[num_bags*num_bags];
    std::vector<label_type> shuff_labels;

    if (shuffle_order) {
        shuff_labels.resize(num_bags);

        size_t* perm = new size_t[num_bags];
        for (size_t i = 0; i < num_bags; i++)
            perm[i] = i;
        std::random_shuffle(perm, perm+num_bags, detail::rand_idx);

        for (size_t i = 0; i < num_bags; i++) {
            size_t p_i = perm[i];
            shuff_labels[p_i] = labels[i];
            for (size_t j = 0; j < num_bags; j++)
                shuff_divs[p_i * num_bags + perm[j]] = divs[i*num_bags + j];
        }

        delete[] perm;

    } else {
        std::copy(divs, divs + num_bags*num_bags, shuff_divs);
        shuff_labels = labels;
    }

    // symmetrize divergence estimates
    symmetrize(shuff_divs, num_bags);

    // do out each fold
    size_t num_test = (size_t) std::ceil(double(num_bags) / folds);

    std::queue< std::pair<size_t, size_t> > jobs;
    double score = 0;
    boost::mutex jobs_mutex, results_mutex;

    if (real_cv_threads <= 1) {
        // don't actually launch separate threads, 'tis a waste

        boost::exception_ptr error;
        detail::cv_worker<label_type> worker(shuff_divs, num_bags, shuff_labels,
                kernels, svm_p, c_vals, tuning_folds,
                project_all_data, pred_div_params, jobs_mutex, jobs,
                results_mutex, score, error);

        for (size_t fold = 0; fold < folds; fold++) {
            size_t test_start = fold * num_test;
            size_t test_end = std::min(test_start + num_test, num_bags);
            if (test_start < test_end)
                score += worker.do_job(test_start, test_end);
        }

    } else {
        // queue up the jobs
        for (size_t fold = 0; fold < folds; fold++) {
            size_t test_start = fold * num_test;
            size_t test_end = std::min(test_start + num_test, num_bags);
            if (test_start < test_end)
                jobs.push(std::make_pair(test_start, test_end));
        }

        // keep worker objects in this ptr_vector to avoid copying but also
        // have them be destroyed when appropriate
        boost::ptr_vector< detail::cv_worker<label_type> > workers;
        std::vector<boost::exception_ptr> errors(real_cv_threads);

        boost::thread_group worker_threads;

        for (size_t i = 0; i < real_cv_threads; i++) {
            workers.push_back(new detail::cv_worker<label_type>(
                shuff_divs, num_bags, shuff_labels, kernels,
                svm_p, c_vals, tuning_folds, project_all_data, pred_div_params,
                jobs_mutex, jobs, results_mutex, score, errors[i]
            ));
            worker_threads.create_thread(boost::ref(workers[i]));
        }

        worker_threads.join_all();
        for (size_t i = 0; i < real_cv_threads; i++)
            if (errors[i])
                boost::rethrow_exception(errors[i]);
    }

    // clean up
    delete kernels;
    delete[] shuff_divs;

    // return accuracy / RMSE
    if (boost::is_same<label_type, int>::value) {
        return score / num_bags;
    } else {
        return std::sqrt(score / num_bags);
    }
}


} // sdm
#endif
