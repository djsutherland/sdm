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

#include <np-divs/div-funcs/div_l2.hpp>

#include <boost/exception_ptr.hpp>
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
    size_t folds, size_t num_cv_threads,
    bool project_all_data, bool shuffle_order,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    const double* divs);

template double crossvalidate<double>(
    const flann::Matrix<double> *bags, size_t num_bags,
    const std::vector<int> &labels,
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    size_t folds, size_t num_cv_threads,
    bool project_all_data, bool shuffle_order,
    const std::vector<double> &c_vals, const svm_parameter &svm_params,
    size_t tuning_folds,
    const double* divs);


// helper function implementations
namespace detail {

    // random index < the passed one; needed by crossvalidate
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


        // any absolutely enormous values?
        for (size_t i = 0; i < l; i++) {
            if (std::abs(km[i]) > 1e70) {
                FILE_LOG(logWARNING) << "enormous kernel value: " << km[i];
                return true;
            }
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
            if (terrible_kernel(km, num_bags)) {
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
        boost::exception_ptr &error;

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
            size_t *num_correct,
            boost::exception_ptr &error)
        :
            divs(divs), num_bags(num_bags), labels(labels),
            kernels(kernels), num_kernels(num_kernels), c_vals(c_vals),
            svm_params(svm_params), folds(folds),
            results(results), num_correct(num_correct), error(error)
        {}

        void operator()() {
            try {
                *results = tune_params_single(divs, num_bags, labels,
                        kernels, num_kernels, c_vals,
                        svm_params, folds, num_correct);
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
            size_t num_correct;
            return pick_rand(tune_params_single(divs, num_bags, labels,
                        kern_array, num_kernels, c_vals, svm_params, folds,
                        &num_correct));
        }

        // grunt work to set up multithreading
        boost::ptr_vector<tune_params_worker> workers;
        std::vector<boost::exception_ptr> errors(num_threads);
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
                        &results[i], &nums_correct[i],
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
        size_t best_correct = *std::max_element(
                nums_correct.begin(), nums_correct.end());
        std::vector<config> best_configs;

        if (best_correct == 0) {
            FILE_LOG(logERROR) << "all kernels were terrible";
            BOOST_THROW_EXCEPTION(std::domain_error("all kernels were terrible"));
        }

        kern_start = 0;
        for (size_t i = 0; i < num_threads; i++) {
            if (nums_correct[i] == best_correct) {
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

    // worker functor class for doing CV
    class cv_worker : boost::noncopyable {
        protected:
            typedef std::pair<size_t, size_t> size_pair;

            const double *divs;
            const size_t num_bags;
            const std::vector<int> &labels;
            const boost::ptr_vector<Kernel> *kernels;
            const svm_parameter &svm_params;
            const std::vector<double> c_vals;
            const size_t tuning_folds;
            const bool project_all_data;
            const npdivs::DivParams &div_params;

            boost::mutex &jobs_mutex;
            std::queue<size_pair> &jobs;
            boost::mutex &results_mutex;
            size_t &total_correct;

            boost::exception_ptr &error;

        public:
            cv_worker(
                    const double *divs,
                    const size_t num_bags,
                    const std::vector<int> &labels,
                    const boost::ptr_vector<Kernel> *kernels,
                    const svm_parameter &svm_params,
                    const std::vector<double> c_vals,
                    const size_t tuning_folds,
                    const bool project_all_data,
                    const npdivs::DivParams &div_params,
                    boost::mutex &jobs_mutex,
                    std::queue<size_pair> &jobs,
                    boost::mutex &results_mutex,
                    size_t &total_correct,
                    boost::exception_ptr &error)
                :
                    divs(divs), num_bags(num_bags), labels(labels),
                    kernels(kernels), svm_params(svm_params),
                    c_vals(c_vals), tuning_folds(tuning_folds),
                    project_all_data(project_all_data), div_params(div_params),
                    jobs_mutex(jobs_mutex), jobs(jobs),
                    results_mutex(results_mutex), total_correct(total_correct),
                    error(error)
        {}

            void operator()() {
                size_pair job;
                size_t num_correct = 0;

                try {
                    while (true) {
                        { // get a job
                            boost::mutex::scoped_lock the_lock(jobs_mutex);
                            if (jobs.size() == 0)
                                break;
                            job = jobs.front();
                            jobs.pop();
                        }

                        num_correct += do_job(job.first, job.second);
                    }

                    { // write the result
                        boost::mutex::scoped_lock the_lock(results_mutex);
                        total_correct += num_correct;
                    }
                } catch (...) {
                    error = boost::current_exception();
                    return;
                }
            }

            size_t do_job(size_t test_start, size_t test_end) const {
                // testing is in [test_start, test_end)
                size_t num_test = test_end - test_start;
                size_t num_train = num_bags - num_test;

                // set up training labels
                std::vector<int> train_labels(num_train);
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
                const char* error = svm_check_parameter(&prob, &svm_params);
                if (error != NULL) {
                    FILE_LOG(logERROR) << "LibSVM parameter error: " << error;
                    BOOST_THROW_EXCEPTION(std::domain_error(error));
                }

                // train!
                svm_model *svm = svm_train(&prob, &svm_p);

                npdivs::DivL2 fake_df; // XXX figure out a better way around this
                SDM<float> sdm(*svm, prob, fake_df, kernel, div_params,
                        svm_get_nr_class(svm), NULL, num_train);

                // predict!
                std::vector<int> preds;
                std::vector< std::vector<double> > vals;
                sdm.predict_from_kerns(test_km, num_test, preds, vals);

                // how many did we get right?
                size_t num_correct = 0;
                for (size_t i = 0; i < num_test; i++)
                    if (preds[i] == labels[test_start + i])
                        num_correct++;

                FILE_LOG(logINFO) << "CV " << test_start << " - " << test_end
                    << ": " << num_correct << "/" << num_test << " correct by "
                    << kernel.name() << ", C=" << svm_p.C;

                // clean up
                delete[] train_km;
                delete[] test_km;
                for (size_t i = 0; i < num_train; i++)
                    delete[] prob.x[i];
                delete[] prob.x;
                delete[] prob.y;
                svm_free_model_content(svm);
                delete svm;

                return num_correct;
            }
    };




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


double crossvalidate(
    const double *divs, size_t num_bags,
    const std::vector<int> &labels,
    const KernelGroup &kernel_group,
    size_t folds,
    size_t num_cv_threads,
    bool project_all_data,
    bool shuffle_order,
    const std::vector<double> &c_vals,
    const svm_parameter &svm_params,
    size_t tuning_folds)
{
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
    svm_p.svm_type = C_SVC;
    svm_p.kernel_type = PRECOMPUTED;

    // make libSVM log properly
    svm_set_print_string_function(&detail::log<logDEBUG4>);


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
    std::vector<int> shuff_labels;

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
    size_t num_correct = 0;
    boost::mutex jobs_mutex, results_mutex;

    if (real_cv_threads <= 1) {
        // don't actually launch separate threads, 'tis a waste

        boost::exception_ptr error;
        detail::cv_worker worker(shuff_divs, num_bags, shuff_labels,
                kernels, svm_p, c_vals, tuning_folds,
                project_all_data, pred_div_params, jobs_mutex, jobs,
                results_mutex, num_correct, error);

        for (size_t fold = 0; fold < folds; fold++) {
            size_t test_start = fold * num_test;
            size_t test_end = std::min(test_start + num_test, num_bags);
            if (test_start < test_end)
                num_correct += worker.do_job(test_start, test_end);
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
        boost::ptr_vector<detail::cv_worker> workers;
        std::vector<boost::exception_ptr> errors(real_cv_threads);

        boost::thread_group worker_threads;

        for (size_t i = 0; i < real_cv_threads; i++) {
            workers.push_back(new detail::cv_worker(
                shuff_divs, num_bags, shuff_labels, kernels,
                svm_p, c_vals, tuning_folds, project_all_data, pred_div_params,
                jobs_mutex, jobs, results_mutex, num_correct, errors[i]
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

    // return accuracy
    return double(num_correct) / num_bags;
}


} // end namespace sdm
