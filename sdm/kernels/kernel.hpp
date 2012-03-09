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
#ifndef KERNELS_HPP_
#define KERNELS_HPP_
#include "sdm/basics.hpp"

#include <string>

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/utility.hpp>

#include <np-divs/div-funcs/div_func.hpp>

namespace sdm {

class Kernel : boost::noncopyable {
public:
    Kernel() {}
    virtual ~Kernel() {}

    virtual std::string name() const = 0;

    virtual double transformDivergence(double div) const = 0;
    virtual void transformDivergences(double* divs, size_t n) const;
    virtual void transformDivergences(double* divs, size_t m, size_t n) const;

    Kernel* clone() const;

private:
    virtual Kernel* do_clone() const = 0;
};

class KernelGroup : boost::noncopyable {
public:
    typedef Kernel KernelType;

    KernelGroup() {}
    virtual ~KernelGroup() {}

    virtual const boost::ptr_vector<Kernel>* getTuningVector(
            const double* divs, size_t n) const = 0;

    KernelGroup* clone() const;

private:
    virtual KernelGroup* do_clone() const = 0;
};


inline Kernel* new_clone(const Kernel &kernel) { return kernel.clone(); }
inline KernelGroup* new_clone(const KernelGroup &grp) { return grp.clone(); }

}
#endif
