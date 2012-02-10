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
#include "sdm/kernels/polynomial.hpp"

#include <cmath>
#include <string>

#include <boost/format.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

namespace sdm {

double PolynomialKernel::transformDivergence(double div) const {
    return std::pow(div + coef0, (int) degree);
}

std::string PolynomialKernel::name() const {
    return (boost::format("Polynomial(%d, %g)") % degree % coef0).str();
}
size_t PolynomialKernel::getDegree() const { return degree; }
double PolynomialKernel::getCoef0() const { return coef0; }

PolynomialKernel* PolynomialKernel::do_clone() const {
    return new PolynomialKernel(degree, coef0);
}

////////////////////////////////////////////////////////////////////////////////

const boost::ptr_vector<Kernel> PolynomialKernelGroup::getTuningVector(
        double* divs, size_t n)
const {
    boost::ptr_vector<Kernel> kerns;
    for (size_t d = 0; d < degrees.size(); d++) {
        for (size_t c = 0; c < coef0s.size(); c++) {
            kerns.push_back(new PolynomialKernel(degrees[d], coef0s[c]));
        }
    }
    return kerns;
}

PolynomialKernelGroup* PolynomialKernelGroup::do_clone() const {
    return new PolynomialKernelGroup(degrees, coef0s);
}

} // end namespace
