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

#include <vector>

#include <np-divs/div_params.hpp>
#include <flann/util/matrix.h>
#include <svm.h>

namespace sdm {

template <typename Scalar>
class SDM {
    svm_model &svm;
    Kernel &kernel;

    const flann::Matrix<Scalar> *training_bags;
    size_t num_training;

    public:
        SDM(svm_model &svm, Kernel &kernel,
            const flann::Matrix<Scalar> *training_bags, size_t num_training)
        :
            svm(svm), kernel(kernel),
            training_bags(training_bags), num_training(num_training)
        { }

        std::vector<bool> predict(
                const flann::Matrix<Scalar> *test_dists, size_t num_test);

        std::vector<bool> predict(
                const flann::Matrix<Scalar> *test_dists, size_t num_test,
                std::vector<double> &values);


        // TODO: a mass-training method for more than one Kernel
        static SDM train(
                const flann::Matrix<Scalar> *training_bags, size_t num_train,
                const std::vector<bool> &labels,
                const Kernel &kernel,
                const NPDivs::DivParams &div_params,
                const svm_parameter &svm_params
                // cross-validation options
        );

};

}

#endif
