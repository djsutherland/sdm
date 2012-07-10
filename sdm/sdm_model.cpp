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

template class SDM<float, int>;
template class SDM<double, int>;
template class SDM<float, double>;
template class SDM<double, double>;


#define TRAIN_INST(intype, labtype) \
template SDM<intype, labtype> * train_sdm<intype, labtype>( \
    const flann::Matrix<intype> *train_bags, size_t num_train, \
    const std::vector<labtype> &labels, \
    const npdivs::DivFunc &div_func, const KernelGroup &kernel_group, \
    const npdivs::DivParams &div_params, const std::vector<double> &c_vals, \
    const svm_parameter &svm_params, size_t tuning_folds, double* divs, \
    bool allow_transduction);
TRAIN_INST(float,  int);
TRAIN_INST(double, int);
TRAIN_INST(float,  double);
TRAIN_INST(double, double);
#undef TRAIN_INST

#define TRANS_INST(Scalar, label_type) \
template std::vector<label_type> transduct_sdm<Scalar, label_type>( \
    const flann::Matrix<Scalar> *train_bags, size_t num_train, \
    const std::vector<label_type> &train_labels, \
    const flann::Matrix<Scalar> *test_bags, size_t num_test, \
    const npdivs::DivFunc &div_func, \
    const KernelGroup &kernel_group, \
    const npdivs::DivParams &div_params, \
    const std::vector<double> &c_vals, \
    const svm_parameter &svm_params, \
    size_t tuning_folds, \
    double *divs);
TRANS_INST(float,  int);
TRANS_INST(double, int);
TRANS_INST(float,  double);
TRANS_INST(double, double);
#undef TRANS_INST

#define TRANS2_INST(Scalar, label_type) \
template std::vector<label_type> transduct_sdm<Scalar, label_type>( \
    const flann::Matrix<Scalar> *train_test_bags, \
    size_t num_train, size_t num_test, \
    const std::vector<label_type> &train_labels, \
    const npdivs::DivFunc &div_func, \
    const KernelGroup &kernel_group, \
    const npdivs::DivParams &div_params, \
    const std::vector<double> &c_vals, \
    const svm_parameter &svm_params, \
    size_t tuning_folds, \
    double *divs);
TRANS2_INST(float,  int);
TRANS2_INST(double, int);
TRANS2_INST(float,  double);
TRANS2_INST(double, double);
#undef TRANS2_INST


// helper function implementations
namespace detail {
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
