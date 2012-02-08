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
#include "sdm/kernels/gaussian.hpp"

#include <cmath>
#include <string>
#include <boost/format.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "sdm/utils.hpp"


namespace sdm {

std::string GaussianKernel::name() const {
    return (boost::format("Gaussian(%g)") % sigma).str();
}

double GaussianKernel::transformDivergence(double div) const {
    div /= sigma;
    return std::exp(-.5 * div * div);
}

GaussianKernel* GaussianKernel::do_clone() const {
    return new GaussianKernel(sigma);
}


////////////////////////////////////////////////////////////////////////////////

typedef std::vector<double>::const_iterator double_iter;

const boost::ptr_vector<Kernel> GaussianKernelGroup::getTuningVector(
        double* divs, size_t n) const
{
    double scale;
    if (scale_sigma) {
        // find median of the nonzero divergences, for scaling sigma
        std::vector<double> pos_divs;
        pos_divs.reserve(n * (n - 1)); // diag is usually 0
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                double d = divs[i*n + j];
                if (d > 0)
                    pos_divs.push_back(d);
            }
        }
        scale = median(pos_divs);
    } else {
        scale = 1.0;
    }

    boost::ptr_vector<Kernel> kerns;
    kerns.reserve(sigmas.size());
    for (double_iter i = sigmas.begin(); i != sigmas.end(); ++i) {
        kerns.push_back(new GaussianKernel(*i * scale));
    }
    return kerns;
}

GaussianKernelGroup* GaussianKernelGroup::do_clone() const {
    return new GaussianKernelGroup();
}

} // end namespace
