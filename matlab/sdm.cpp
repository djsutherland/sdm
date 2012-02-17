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


/* A MATLAB interface to the C++ SDM class.
 *
 * Don't use this directly; instead, use the SDM.m wrapper class. That class
 * maintains a pointer to an instance of the C++ class, which is passed into
 * this MEX function, which then calls the appropriate C++ methods.
 */

#include <cmath>
#include <string>
#include <vector>

#include <mex.h>

#include <svm.h>

#include <flann/flann.hpp>

#include <np-divs/matrix_arrays.hpp>
#include <np-divs/div-funcs/from_str.hpp>

#include "sdm/sdm.hpp"
#include "sdm/kernels/gaussian.hpp"
#include "sdm/kernels/linear.hpp"
#include "sdm/kernels/polynomial.hpp"

typedef flann::Matrix<float> MatrixF;
typedef flann::Matrix<double> MatrixD;

using std::string;
using std::vector;

using sdm::SDM;
typedef SDM<float> SDMF;

////////////////////////////////////////////////////////////////////////////////
// Helper functions for passing class pointers between MATLAB/MEX
//
// Loosely based on code by Oliver Woodford:
//   http://www.mathworks.com/matlabcentral/newsreader/view_thread/278243

// Define types
#ifdef _MSC_VER
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

#define CLASS_HANDLE_SIGNATURE 0xa5a50f0f
template<class base>
class class_handle {
    public:
        class_handle(base* pointer) : pointer(pointer) {
            signature = CLASS_HANDLE_SIGNATURE;
        }

        ~class_handle() {
            signature = 0;
            delete pointer;
        }

        bool isValid() { return signature == CLASS_HANDLE_SIGNATURE; }

        base* getPointer() { return pointer; }

    private:
        base* pointer;
        uint32_t signature;
};

template<class base>
inline mxArray *convertPtr2Mat(class_handle<base> *ptr) {
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *) mxGetData(out)) = reinterpret_cast<uint64_t>(ptr);
    return out;
}

template<class base>
inline class_handle<base> *convertMat2Ptr(const mxArray *in) {
    if (mxGetNumberOfElements(in) != 1
            || mxGetClassID(in) != mxUINT64_CLASS
            || mxIsComplex(in))
        mexErrMsgTxt("Input must be a real uint64 scalar.");

    class_handle<base> *ptr =
        reinterpret_cast<class_handle<base> *>(*((uint64_t *) mxGetData(in)));

    if (!ptr->isValid())
        mexErrMsgTxt("Handle not valid.");
    return ptr;
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions to convert between MATLAB matrices and flann::Matrix

// Copy a MATLAB array of type T into a flann::Matrix<float>.
template <typename T>
void copyIntoFlann(const mxArray *bag, mwSize rows, mwSize cols,
                   MatrixF &target)
{
    const T* bag_data = (T*) mxGetData(bag);

    // copy from column-major source to row-major dest, also cast contents
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            target[i][j] = (float) bag_data[j*rows + i];
}

// Copy a MATLAB cell array of distribution samples (with consistent number
// of columns) into a newly-allocated array of flann::Matrix<float>s.
MatrixF *get_matrix_array(const mxArray *bags, mwSize n) {
    MatrixF *flann_bags = (MatrixF *) mxCalloc(n, sizeof(MatrixF));

    mwSize rows;
    const mwSize cols = mxGetN(mxGetCell(bags, 0));

    const mxArray *bag;

    for (mwSize i = 0; i < n; i++) {
        bag = mxGetCell(bags, i);

        // check dimensions
        if (mxGetNumberOfDimensions(bag) != 2)
            mexErrMsgTxt("bag has too many dimensions");
        rows = mxGetM(bag);
        if (mxGetN(bag) != cols)
            mexErrMsgTxt("inconsistent number of columns in bags");

        // allocate the result matrix
        flann_bags[i] = MatrixF(
                (float*) mxCalloc(rows * cols, sizeof(float)),
                rows, cols);

        if (mxIsDouble(bag))
            copyIntoFlann<double>(bag, rows, cols, flann_bags[i]);
        else if (mxIsSingle(bag))
            copyIntoFlann<float>(bag, rows, cols, flann_bags[i]);
        else
            mexErrMsgTxt("unsupported bag type");
    }

    return flann_bags;
}

////////////////////////////////////////////////////////////////////////////////
// Helpers to convert from MATLAB to C++ types

string get_string(const mxArray *thing, const char* err_msg) {
    if (mxIsChar(thing) != 1 || mxGetM(thing) != 1)
        mexErrMsgTxt(err_msg);

    char* c_str = mxArrayToString(thing);
    string str(c_str);
    mxFree(c_str);
    return str;
}

double get_double(const mxArray *thing, const char* err_msg) {
    if (mxIsNumeric(thing) != 1 || mxGetNumberOfElements(thing) != 1)
        mexErrMsgTxt(err_msg);
    return mxGetScalar(thing);
}

int get_int(const mxArray *thing, const char* err_msg) {
    double d = get_double(thing, err_msg);
    int i = (int) (d+.1);
    if (std::abs(d - i) > 1e-10)
        mexErrMsgTxt(err_msg);
    return i;
}

size_t get_size_t(const mxArray *thing, const char *err_msg) {
    double d = get_double(thing, err_msg);
    if (d < 0) mexErrMsgTxt(err_msg);
    size_t s = (size_t) (d + .1);
    if (std::abs(d - s) > 1e-10)
        mexErrMsgTxt(err_msg);
    return s;
}

bool get_bool(const mxArray *thing, const char* err_msg) {
    if (mxGetNumberOfElements(thing) == 1) {
        if (mxIsLogical(thing) == 1)
            return mxGetScalar(thing) != 0;

        if (mxIsNumeric(thing) == 1) {
            double d = mxGetScalar(thing);
            if (d == 0)
                return false;
            else if (d == 1)
                return true;
        }
    }
    mexErrMsgTxt(err_msg);
    return false; // to make compilers happy, but this'll never happen
}


template <typename T, typename K>
vector<K> _convert(const void* data_void, mwSize n) {
    vector<K> vec;
    vec.reserve(n);

    T* data = (T*) data_void;
    for (mwSize i = 0; i < n; i++) {
        vec.push_back((K) data[i]);
    }
    return vec;
}

template <typename K>
vector<K> get_vector(const mxArray *thing, const char* err_msg) {
    // this ignores overflow / unsignedness, because that'd be more work
    if (!mxIsNumeric(thing) && !mxIsLogical(thing))
        mexErrMsgTxt(err_msg);

    mwSize n = mxGetNumberOfElements(thing);
    const void* data = mxGetData(thing);

    // switch on the input type, since we can only get a void*
    // ...thanks for being the worst, mex
    switch (mxGetClassID(thing)) {
        case mxINT8_CLASS:   return _convert<char,               K>(data, n);
        case mxUINT8_CLASS:  return _convert<unsigned char,      K>(data, n);
        case mxINT16_CLASS:  return _convert<short,              K>(data, n);
        case mxUINT16_CLASS: return _convert<unsigned short,     K>(data, n);
        case mxINT32_CLASS:  return _convert<int,                K>(data, n);
        case mxUINT32_CLASS: return _convert<unsigned int,       K>(data, n);
        case mxINT64_CLASS:  return _convert<long long,          K>(data, n);
        case mxUINT64_CLASS: return _convert<unsigned long long, K>(data, n);
        case mxSINGLE_CLASS: return _convert<float,              K>(data, n);
        case mxDOUBLE_CLASS: return _convert<double,             K>(data, n);
        default:
            mexErrMsgTxt(err_msg);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Constructor / destructor

template <typename Scalar>
void destroy(SDM<Scalar> *model) {
    model->destroyModelAndProb();
    delete model;
}

////////////////////////////////////////////////////////////////////////////////
// Training

struct TrainingOptions {
    string div_func;
    string kernel;
    int k;
    size_t tuning_folds;
    bool probability;
    size_t num_threads;
    string index_type;

    TrainingOptions() :
        div_func("l2"), kernel("gaussian"), k(3), tuning_folds(3),
        probability(false), num_threads(0), index_type("kdtree")
    {}
};

template <typename Scalar>
SDM<Scalar> * train(
        const mxArray* bags_m, const mxArray* labels_m, const mxArray* opts_m)
{
    // first argument: training bags
    mwSize num_train = mxGetNumberOfElements(bags_m);
    MatrixF *bags = get_matrix_array(bags_m, num_train); // FIXME: this'll leak

    // second argument: labels
    const vector<int> labels = get_vector<int>(labels_m,
            "second argument must be an array of integers");
    if (labels.size() > num_train)
        mexErrMsgTxt("got more labels than bags");
    else if (labels.size() < num_train)
        mexErrMsgTxt("got more bags than labels");

    // third argument: options
    if (!mxIsStruct(opts_m) && mxGetNumberOfElements(opts_m) == 1)
        mexErrMsgTxt("train 3rd argument must be a single struct");
    mwSize nfields = mxGetNumberOfFields(opts_m);

    TrainingOptions opts;
    for (mwSize i = 0; i < nfields; i++) {
        const char* name_ = mxGetFieldNameByNumber(opts_m, i);
        std::string name(name_);
        mxArray* val = mxGetFieldByNumber(opts_m, 0, i);

        if (name == "div_func") {
            opts.div_func = get_string(val, "div_func must be a string");

        } else if (name == "kernel") {
            opts.kernel = get_string(val, "kernel must be a string");

        } else if (name == "k") {
            opts.k = get_size_t(val, "k must be a positive integer");
            if (opts.k < 1)
                mexErrMsgTxt("k must be a positive integer");

        } else if (name == "tuning_folds") {
            opts.tuning_folds = get_size_t(val,
                    "tuning_folds must be a positive integer");
            if (opts.tuning_folds < 2)
                mexErrMsgTxt("tuning_folds must be at least 2");

        } else if (name == "probability") {
            opts.probability = get_bool(val, "probability must be boolean");

        } else if (name == "num_threads") {
            opts.num_threads = get_size_t(val,
                    "num_threads must be a nonnegative integer");

        } else if (name == "index") {
            opts.index_type = get_string(val, "index must be a string");

        } else {
            mexErrMsgTxt(("unknown training option: " + name).c_str());
        }
    }

    // build up the actual parameters we need based on opts
    npdivs::DivFunc* div_func = npdivs::div_func_from_str(opts.div_func);

    sdm::KernelGroup* kernel_group;
    if (opts.kernel == "gaussian") {
        kernel_group = new sdm::GaussianKernelGroup;
    } else if (opts.kernel == "linear") {
        kernel_group = new sdm::LinearKernelGroup;
    } else if (opts.kernel == "polynomial") {
        kernel_group = new sdm::PolynomialKernelGroup;
    } else {
        mexErrMsgTxt(("unkown kernel type: " + opts.kernel).c_str());
    }

    flann::IndexParams * index_params;
    if (opts.index_type == "linear" || opts.index_type == "brute") {
        index_params = new flann::LinearIndexParams;
    } else if (opts.index_type == "kdtree" || opts.index_type == "kd") {
        index_params = new flann::KDTreeSingleIndexParams;
    } else {
        mexErrMsgTxt(("unknown index type: " + opts.index_type).c_str());
    }

    flann::SearchParams search_params(-1);

    npdivs::DivParams div_params(
            opts.k, index_params, &search_params, opts.num_threads);

    svm_parameter svm_params = sdm::default_svm_params;
    svm_params.probability = (int) opts.probability;


    // train away!
    SDMF *model = sdm::train_sdm<float>(bags, num_train, labels, *div_func,
            *kernel_group, div_params, sdm::default_c_vals,
            svm_params, opts.tuning_folds);

    // cleanup
    delete kernel_group;
    delete div_func;
    delete index_params;

    return model;
}

////////////////////////////////////////////////////////////////////////////////
// Cross-validation


////////////////////////////////////////////////////////////////////////////////
// Prediction

////////////////////////////////////////////////////////////////////////////////
// Dispatch function

void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    // first arg is a string saying what the desired operation is
    string op = get_string(prhs[0], "first input must be a string");

    if (op == "create") {
        // TODO: constructor?
    } else if (op == "delete") {
        if (nrhs != 2) mexErrMsgTxt("delete needs exactly one argument");
        if (nlhs > 0) mexErrMsgTxt("delete doesn't return anything");

        destroy(convertMat2Ptr<SDMF>(prhs[1])->getPointer());

    } else if (op == "name") {
        if (nrhs != 2) mexErrMsgTxt("name takes exactly one argument");
        if (nlhs != 1) mexErrMsgTxt("name returns exactly 1 output");

        SDMF *model = convertMat2Ptr<SDMF>(prhs[1])->getPointer();
        plhs[0] = mxCreateString(model->name().c_str());

    } else if (op == "train") {
        if (nrhs != 4) mexErrMsgTxt("train needs exactly three arguments");
        if (nlhs != 1) mexErrMsgTxt("train returns exactly 1 output");

        SDMF *model = train<float>(prhs[1], prhs[2], prhs[3]);

        class_handle<SDMF> * ptr = new class_handle<SDMF>(model);
        plhs[0] = convertPtr2Mat<SDMF>(ptr);

    } else if (op == "crossvalidate") {
        // TODO: cross-validation

    } else if (op == "predict") {
        // TODO: prediction

    } else {
        mexErrMsgTxt(("Unknown operation '" + op + "'.").c_str());
    }
}
