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
#ifndef SDM_MODEL_HPP_
#define SDM_MODEL_HPP_
#include "sdm/basics.hpp"
#include "sdm/defaults.hpp"
#include "sdm/kernels/kernel.hpp"
#include "sdm/kernel_projection.hpp"
#include "sdm/log.hpp"
#include "sdm/tune_params.hpp"
#include "sdm/utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <ios>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/exception_ptr.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/thread.hpp>
#include <boost/utility.hpp>

#include <np-divs/matrix_arrays.hpp>
#include <np-divs/np_divs.hpp>
#include <flann/util/matrix.h>
#include <svm.h>


namespace sdm {

// TODO: optionally save and reuse flann indices and/or rhos, nus
//       (requires adding these options to npdivs...)

// TODO: probability models seem to just suck (at least on one example)
//       - is this a bug or something wrong with the model?
// TODO: memory ownership with this is all screwy...make it clearer
template <typename Scalar, typename label_type>
class SDM {
    const svm_model &svm;
    const svm_problem &svm_prob; // needs to live at least as long as the model
    const npdivs::DivFunc *div_func;
    const Kernel *kernel;
    const npdivs::DivParams div_params;
    const size_t num_classes;

    const flann::Matrix<Scalar> *train_bags;
    const size_t num_train;
    const size_t dim;

    public:
        SDM(const svm_model &svm, const svm_problem &svm_prob,
            const npdivs::DivFunc &div_func, const Kernel &kernel,
            const npdivs::DivParams &div_params,
            size_t num_classes,
            const flann::Matrix<Scalar> *train_bags, size_t num_train,
            size_t dim=0)
        :
            svm(svm), svm_prob(svm_prob),
            div_func(new_clone(div_func)), kernel(new_clone(kernel)),
            div_params(div_params), num_classes(num_classes),
            train_bags(train_bags), num_train(num_train),
            dim(dim > 0 ? dim :
                (num_train > 0 && train_bags != NULL ? train_bags[0].cols : 0))
        { }

        ~SDM() {
            delete div_func;
            delete kernel;
        }

        void destroyModelAndProb();
        void destroyTrainBags();
        void destroyTrainBagMatrices();

        // getters and so on
        const Kernel *getKernel() const { return kernel; }
        const npdivs::DivFunc *getDivFunc() const { return div_func; }
        const svm_model *getSVM() const { return &svm; }
        const npdivs::DivParams *getDivParams() const { return &div_params; }

        bool isRegression() const;
        bool doesProbability() const {
            return svm_check_probability_model(&svm);
        }

        size_t getNumTrain() const { return num_train; }
        size_t getDim() const { return dim; }
        size_t getNumClasses() const { return num_classes; }
        const flann::Matrix<Scalar> * getTrainBags() { return train_bags; }

        std::string name() const {
            return (boost::format(
                "SDM(%d classes, dim %d, %s, kernel %s, C %g, %d training%s)")
                    % num_classes % dim
                    % div_func->name() % kernel->name()
                    % svm.param.C
                    % num_train
                    % (doesProbability() ? ", with prob" : "")
                ).str();
        }

        // Prediction functions, for either one or many test bags.
        // TODO: do we want dec values for regression? does it mean anything?
        //       or is there a good equivalent?
        //
        // The overloads which take a vector reference will put either decision
        // values or probabilities into it, depending on doesProbability().
        // Any existing contents will be discarded.
        //
        // For decision values:
        //    There are num_classes * (num_classes - 1) / 2 values, for all of
        //    the pairwise class comparisons.  The order is
        //        label[0]-vs-label[1], ..., label[0]-vs-label[n-1],
        //        label[1]-vs-label[2], ..., label[n-2] vs label[n-1],
        //    where the label ordering is as in svm_get_labels(getSVM(), ...).
        //
        // For probabilities:
        //    There is one estimate per class, in the same order as in
        //    getLabels().

        label_type predict(const flann::Matrix<Scalar> &test_bag) const;
        label_type predict(const flann::Matrix<Scalar> &test_bag,
                std::vector<double> &vals) const;

        std::vector<label_type> predict(
                const flann::Matrix<Scalar> *test_bags, size_t num_test)
            const;
        std::vector<label_type> predict(
                const flann::Matrix<Scalar> *test_bags, size_t num_test,
                std::vector< std::vector<double> > &vals)
            const;

        // Prediction if you already have kernel values; used in CV.
        // km should be a row-major flat matrix with num_test rows and
        // num_train columns.
        void predict_from_kerns(double* km, size_t num_test,
                std::vector<label_type> &preds,
                std::vector< std::vector<double> > &vals)
            const;
};

// Function to train a new SDM. Note that the caller is responsible for
// deleting the svm and svm_prob attributes, as well as train_bags.
// The destroyModelAndProb() and destroyTrainBags() functions might be
// helpful for this.
//
// TODO: a mass-training method for more than one kernel
// TODO: option to do the projection on test data as well
// TODO: more flexible tuning CV options...tune on a subset of the data?
template <typename Scalar, typename label_type>
SDM<Scalar, label_type> * train_sdm(
    const flann::Matrix<Scalar> *train_bags, size_t num_train,
    const std::vector<label_type> &labels,
    const npdivs::DivFunc &div_func,
    const KernelGroup &kernel_group,
    const npdivs::DivParams &div_params,
    const std::vector<double> &c_vals = default_c_vals,
    const svm_parameter &svm_params = default_svm_params,
    size_t tuning_folds = 3,
    double *divs = NULL);



////////////////////////////////////////////////////////////////////////////////
// Helper functions

namespace detail {

    // string representation of an SVM model
    std::string SVMtoString(const svm_model &model);

    // silly helper XXX - change to boost::is_same
    template <typename T> struct SVMType { static const int val; };
    template <> struct SVMType<int> { static const int val = C_SVC; };
    template <> struct SVMType<double> { static const int val = EPSILON_SVR; };
}

template <typename Scalar, typename label_type>
void SDM<Scalar, label_type>::destroyModelAndProb() {
    // FIXME: rework SDM memory model to avoid gross const_casts
    // TODO: rename
    // TODO: just have SDM copy everything it needs?

    svm_free_model_content(const_cast<svm_model*> (&svm));
    delete const_cast<svm_model*>(&svm);

    for (size_t i = 0; i < svm_prob.l; i++)
        delete[] svm_prob.x[i];
    delete[] svm_prob.x;
    delete[] svm_prob.y;
    delete const_cast<svm_problem*>(&svm_prob);
}

template <typename Scalar, typename label_type>
void SDM<Scalar, label_type>::destroyTrainBags() {
    npdivs::free_matrix_array(
            const_cast<flann::Matrix<Scalar> *>(train_bags),
            num_train);
}

template <typename Scalar, typename label_type>
void SDM<Scalar, label_type>::destroyTrainBagMatrices() {
    delete[] train_bags;
}

////////////////////////////////////////////////////////////////////////////////
// Training

template <typename Scalar, typename label_type>
SDM<Scalar, label_type> * train_sdm(
        const flann::Matrix<Scalar> *train_bags, size_t num_train,
        const std::vector<label_type> &labels,
        const npdivs::DivFunc &div_func,
        const KernelGroup &kernel_group,
        const npdivs::DivParams &div_params,
        const std::vector<double> &c_vals,
        const svm_parameter &svm_params,
        size_t tuning_folds,
        double* divs)
{
    if (c_vals.size() == 0) {
        BOOST_THROW_EXCEPTION(std::length_error("c_vals is empty"));
    } else if (labels.size() != num_train) {
        BOOST_THROW_EXCEPTION(std::length_error(
                    "labels.size() disagrees with num_train"));
    }

    // copy the svm params so we can change them
    svm_parameter svm_p = svm_params;
    svm_p.svm_type = detail::SVMType<label_type>::val;
    svm_p.kernel_type = PRECOMPUTED;

    // make libSVM log properly
    svm_set_print_string_function(&log_string<logDEBUG4>);

    // first compute divergences, if necessary
    bool free_divs = false;
    if (divs == NULL) {
        FILE_LOG(logDEBUG) << "train: computing divergences";

        flann::Matrix<double>* divs_f =
            npdivs::alloc_matrix_array<double>(1, num_train, num_train);
        np_divs(train_bags, num_train, div_func, divs_f, div_params, false);

        divs = divs_f[0].ptr();
        free_divs = true;
        delete[] divs_f; // just deletes the Matrix objects, not the data
    }

    // symmetrize divergence estimates
    symmetrize(divs, num_train);

    // ask the kernel group for the kernels we'll pick from for tuning
    const boost::ptr_vector<Kernel>* kernels =
        kernel_group.getTuningVector(divs, num_train);

    // tune_params will need this
    std::srand(unsigned(std::time(NULL)));

    // tuning: cross-validate over possible svm/kernel parameters
    FILE_LOG(logDEBUG) << "train: tuning parameters";
    const std::pair<size_t, size_t> &best_config =
        detail::tune_params<label_type>(divs, num_train, labels, *kernels,
                c_vals, svm_p, tuning_folds, div_params.num_threads);

    // copy the kernel object so we can free the rest
    const Kernel *kernel = new_clone((*kernels)[best_config.first]);
    svm_p.C = c_vals[best_config.second];
    delete kernels; // FIXME: potential leaks in here if something crashes

    FILE_LOG(logINFO) << "train: using " << kernel->name() <<
        "; C = " << c_vals[best_config.second];

    // compute the final kernel matrix
    kernel->transformDivergences(divs, num_train);
    project_to_symmetric_psd(divs, num_train);

    FILE_LOG(logDEBUG3) << "train: final kernel matrix is:\n" <<
        detail::matrixToString(divs, num_train, num_train);

    // set up the svm_problem
    svm_problem *prob = new svm_problem;
    prob->l = num_train;
    prob->y = new double[num_train];
    for (size_t i = 0; i < num_train; i++)
        prob->y[i] = labels[i];
    detail::store_kernel_matrix(*prob, divs, true);

    // don't need the kernel values anymore
    if (free_divs)
        delete[] divs;

    // check svm parameters
    const char* error = svm_check_parameter(prob, &svm_p);
    if (error != NULL) {
        FILE_LOG(logERROR) << "LibSVM parameter error: " << error;
        BOOST_THROW_EXCEPTION(std::domain_error(error));
    }

    // train away!
    FILE_LOG(logDEBUG) << "train: training SVM";
    svm_model *svm = svm_train(prob, &svm_p);
    FILE_LOG(logDEBUG3) << "train: final SVM:\n" << detail::SVMtoString(*svm);

    SDM<Scalar, label_type>* sdm = new SDM<Scalar, label_type>(
            *svm, *prob, div_func, *kernel, div_params, svm_get_nr_class(svm),
            train_bags, num_train);

    delete kernel; // copied in the constructor
    return sdm;
}

////////////////////////////////////////////////////////////////////////////////
// Prediction

// one distribution
template <typename Scalar, typename label_type>
label_type SDM<Scalar, label_type>::predict(
        const flann::Matrix<Scalar> &test_bag)
const {
    std::vector< std::vector<double> > vals(1);
    return this->predict(&test_bag, 1, vals)[0];
}

// one distribution, with decision values
template <typename Scalar, typename label_type>
label_type SDM<Scalar, label_type>::predict(
        const flann::Matrix<Scalar> &test_bag,
        std::vector<double> &val)
const {
    std::vector< std::vector<double> > vals(1);
    const std::vector<label_type> &pred_labels =
        this->predict(&test_bag, 1, vals);
    val = vals[0];
    return pred_labels[0];
}

// many distributions
template <typename Scalar, typename label_type>
std::vector<label_type> SDM<Scalar, label_type>::predict(
        const flann::Matrix<Scalar> *test_bags, size_t num_test)
const {
    std::vector< std::vector<double> > vals(num_test);
    return this->predict(test_bags, num_test, vals);
}

// many distributions, with decision values
template <typename Scalar, typename label_type>
std::vector<label_type> SDM<Scalar, label_type>::predict(
        const flann::Matrix<Scalar> *test_bags, size_t num_test,
        std::vector< std::vector<double> > &vals)
const {
    // TODO: np_divs option to compute things both ways and/or save trees
    // TODO: only compute divergences from support vectors

    if (train_bags == NULL) {
        BOOST_THROW_EXCEPTION(std::domain_error("this SDM doesn't have "
                    "its training bags saved; can only predict_from_kerns"));
    }

    double *div_data = new double[num_test * num_train];
    flann::Matrix<double> divs(div_data, num_test, num_train);

    double *div_data_oth = new double[num_train * num_test];
    flann::Matrix<double> divs_oth(div_data_oth, num_train, num_test);

    // compute divergences
    npdivs::np_divs(test_bags, num_test, train_bags, num_train,
            *div_func, &divs, div_params);
    npdivs::np_divs(train_bags, num_train, test_bags, num_test,
            *div_func, &divs_oth, div_params);

    // symmetrize divergence estimates
    for (size_t i = 0; i < num_test; i++)
        for (size_t j = 0; j < num_train; j++)
            divs[i][j] = (divs_oth[j][i] + divs[i][j]) / 2.0;

    delete[] div_data_oth;

    // pass through the kernel
    kernel->transformDivergences(div_data, num_test, num_train);

    // can't project because it's not square...

    std::vector<label_type> preds;
    predict_from_kerns(div_data, num_test, preds, vals);
    delete[] div_data;
    return preds;
}

// many distributions, with decision values, from kernels
template <typename Scalar, typename label_type>
void SDM<Scalar, label_type>::predict_from_kerns(
        double* km, size_t num_test,
        std::vector<label_type> &preds,
        std::vector< std::vector<double> > &vals)
const {
    // figure out which prediction function we want to use
    bool do_prob = doesProbability();

    double (*pred_fn)(const svm_model*, const svm_node*, double*) =
        do_prob ? &svm_predict_probability : &svm_predict_values;

    size_t num_vals = do_prob
        ? num_classes : (num_classes * (num_classes - 1)) / 2;

    // we'll reuse this svm_node array for testing
    svm_node *kernel_row = new svm_node[num_train+2];
    for (size_t i = 0; i <= num_train; i++)
        kernel_row[i].index = i;
    kernel_row[num_train+1].index = -1;

    // predict!
    preds.resize(num_test);
    vals.resize(num_test);

    for (size_t i = 0; i < num_test; i++) {
        // fill in our kernel evaluations
        kernel_row[0].value = -i - 1;
        for (size_t j = 0; j < num_train; j++)
            kernel_row[j+1].value = km[i * num_train + j];

        // get space to store our decision/probability values
        vals[i].resize(num_vals);

        // ask the SVM for a prediction
        double res = pred_fn(&svm, kernel_row, &vals[i][0]);
        preds[i] = (label_type) res;
    }

    delete[] kernel_row;
}

} // end namespace

#endif
