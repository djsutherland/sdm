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
#ifndef SDM_DEFAULTS_HPP_
#define SDM_DEFAULTS_HPP_
#include "sdm/basics.hpp"

#include <svm.h>

namespace sdm {

namespace detail {
    const double cvals[10] = { // 2^-9, 2^-6, ..., 2^18
        1./512., 1./64., 1./8., 1, 1<<3, 1<<6, 1<<9, 1<<12, 1<<15, 1<<18
    };
}

const std::vector<double> default_c_vals(detail::cvals, detail::cvals + 10);

const svm_parameter default_svm_params = {
    C_SVC, // svm_type
    PRECOMPUTED, // kernel_type
    0,    // degree - not used
    0,    // gamma - not used
    0,    // coef0 - not used
    1024, // cache_size, in MB
    1e-3, // eps - stopping condition tolerance
    1,    // C - tuned in CV for both classification and regression
    0,    // nr_weight
    NULL, // weight_label
    NULL, // weight
    0,    // nu - not used
    0.1,  // p - epsilon for regression, not currently tuned
    1,    // use shrinking heuristics
    0     // do probability estimates
};

}
#endif
