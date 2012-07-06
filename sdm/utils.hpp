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
#ifndef SDM_UTILS_HPP_
#define SDM_UTILS_HPP_
#include "sdm/basics.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <boost/throw_exception.hpp>

#include <flann/util/matrix.h>

namespace sdm {
namespace detail {

/* Finds the median of a vector with numeric type, reordering the vector
 * in doing so. Throws std::domain_error if the vector is empty.
 */
template <typename T>
double median(std::vector<T> &vec) {
    size_t n = vec.size();
    size_t mid = n / 2;

    if (n == 0) {
        BOOST_THROW_EXCEPTION(std::length_error("median of an empty list"));
    }

    std::nth_element(vec.begin(), vec.begin() + mid, vec.end());

    if (n % 2 != 0) {
        return vec[mid];
    } else {
        T next = *std::max_element(vec.begin(), vec.begin() + mid);
        return (vec[mid] + next) / 2.0;
    }
}

// return a string representation of a matrix
template <typename T>
std::string matrixToString(const T* mat, size_t rows, size_t cols) {
    std::stringstream ss (std::stringstream::in | std::stringstream::out);

    ss << std::setprecision(8);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++)
            ss << "\t" << mat[i*cols + j];
        ss << "\n";
    }
    return ss.str();
}

// string representation of a flann::Matrix
template <typename T>
std::string matrixToString(const flann::Matrix<T> &mat) {
    return matrixToString(mat.ptr(), mat.rows, mat.cols);
}

} // detail
} // sdm
#endif
