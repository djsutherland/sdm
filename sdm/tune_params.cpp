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
#include "sdm/tune_params.hpp"

namespace sdm {
namespace detail {

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






} // detail
} // sdm
