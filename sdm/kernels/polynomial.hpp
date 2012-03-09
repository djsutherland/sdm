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
#ifndef SDM_KERNELS_POLYNOMIAL_HPP_
#define SDM_KERNELS_POLYNOMIAL_HPP_
#include "sdm/basics.hpp"
#include "sdm/kernels/kernel.hpp"

#include <vector>

#include <boost/ptr_container/ptr_vector.hpp>

namespace sdm {

class PolynomialKernel : public Kernel {
    typedef Kernel super;

protected:
    size_t degree;
    double coef0;

public:
    PolynomialKernel(size_t degree, double coef0 = 1)
        : super(), degree(degree), coef0(coef0)
    {}

    virtual std::string name() const;

    virtual double transformDivergence(double div) const;

    size_t getDegree() const;
    double getCoef0() const;

private:
    virtual PolynomialKernel* do_clone() const;
};

////////////////////////////////////////////////////////////////////////////////

namespace detail {
    const size_t degs[7] = { 1, 2, 3, 4, 5, 7, 9 };
    const double coef0s[2] = { 1, 0 };
}
const std::vector<size_t> default_degrees(detail::degs, detail::degs + 7);
const std::vector<double> default_coef0s(detail::coef0s, detail::coef0s + 2);


class PolynomialKernelGroup : public KernelGroup {
protected:
    const std::vector<size_t> degrees;
    const std::vector<double> coef0s;

public:
    typedef PolynomialKernel KernelType;

    PolynomialKernelGroup(
            const std::vector<size_t> &degrees = default_degrees,
            const std::vector<double> &coef0s = default_coef0s)
        : degrees(degrees), coef0s(coef0s)
    {} 

    virtual const boost::ptr_vector<Kernel>* getTuningVector(
            const double* divs, size_t n) const;

private:
    virtual PolynomialKernelGroup* do_clone() const;
};

} // end namespace
#endif
