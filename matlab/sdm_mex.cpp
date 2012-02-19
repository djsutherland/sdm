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

using npdivs::DivParams;

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
        case mxINT8_CLASS:   return _convert<int8_T,   K>(data, n);
        case mxUINT8_CLASS:  return _convert<uint8_T,  K>(data, n);
        case mxINT16_CLASS:  return _convert<int16_T,  K>(data, n);
        case mxUINT16_CLASS: return _convert<uint16_T, K>(data, n);
        case mxINT32_CLASS:  return _convert<int32_T,  K>(data, n);
        case mxUINT32_CLASS: return _convert<uint32_T, K>(data, n);
        case mxINT64_CLASS:  return _convert<int64_T,  K>(data, n);
        case mxUINT64_CLASS: return _convert<uint64_T, K>(data, n);
        case mxSINGLE_CLASS: return _convert<float,    K>(data, n);
        case mxDOUBLE_CLASS: return _convert<double,   K>(data, n);
        default:
            mexErrMsgTxt(err_msg);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions to convert from MATLAB matrices to flann::Matrix

// Copy a MATLAB array of type T into a flann::Matrix<K>.
template <typename T, typename K>
void copyIntoFlann(const mxArray *bag, mwSize rows, mwSize cols,
                   flann::Matrix<K> &target)
{
    const T* bag_data = (T*) mxGetData(bag);

    // copy from column-major source to row-major dest, also cast contents
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            target[i][j] = (K) bag_data[j*rows + i];
}

template <typename K>
flann::Matrix<K> get_matrix(const mxArray *mat, K* data) {
    mwSize rows = mxGetM(mat);
    mwSize cols = mxGetM(mat);
    flann::Matrix<K> fla(data, rows, cols);

    switch (mxGetClassID(mat)) {
        case mxINT8_CLASS:   copyIntoFlann<int8_T,   K>(mat, rows, cols, fla);
            break;
        case mxUINT8_CLASS:  copyIntoFlann<uint8_T,  K>(mat, rows, cols, fla);
            break;
        case mxINT16_CLASS:  copyIntoFlann<int16_T,  K>(mat, rows, cols, fla);
            break;
        case mxUINT16_CLASS: copyIntoFlann<uint16_T, K>(mat, rows, cols, fla);
            break;
        case mxINT32_CLASS:  copyIntoFlann<int32_T,  K>(mat, rows, cols, fla);
            break;
        case mxUINT32_CLASS: copyIntoFlann<uint32_T, K>(mat, rows, cols, fla);
            break;
        case mxINT64_CLASS:  copyIntoFlann<int64_T,  K>(mat, rows, cols, fla);
            break;
        case mxUINT64_CLASS: copyIntoFlann<uint64_T, K>(mat, rows, cols, fla);
            break;
        case mxSINGLE_CLASS: copyIntoFlann<float,    K>(mat, rows, cols, fla);
            break;
        case mxDOUBLE_CLASS: copyIntoFlann<double,   K>(mat, rows, cols, fla);
            break;
        default:
            mexErrMsgTxt("unsupported bag type");
    }
    return fla;
}

// Copy a MATLAB cell array of distribution samples (with consistent number
// of columns) into a newly-allocated array of flann::Matrix<K>s.
template <typename K>
flann::Matrix<K> *get_matrix_array(const mxArray *bags, mwSize n,
        bool mat_alloc=true)
{
    typedef flann::Matrix<K> Matrix;

    if (!mxIsCell(bags))
        mexErrMsgTxt("get_matrix_array: non-cell argument");

    Matrix *flann_bags;
    if (mat_alloc)
        flann_bags = (Matrix *) mxCalloc(n, sizeof(Matrix));
    else
        flann_bags = new Matrix[n];

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
        K* data;
        if (mat_alloc)
            data = (K*) mxCalloc(rows * cols, sizeof(K));
        else
            data = new K[rows * cols];
        flann_bags[i] = Matrix(data, rows, cols);

        switch (mxGetClassID(bag)) {
            case mxINT8_CLASS:
                copyIntoFlann<int8_T,   K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxUINT8_CLASS:
                copyIntoFlann<uint8_T,  K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxINT16_CLASS:
                copyIntoFlann<int16_T,  K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxUINT16_CLASS:
                copyIntoFlann<uint16_T, K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxINT32_CLASS:
                copyIntoFlann<int32_T,  K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxUINT32_CLASS:
                copyIntoFlann<uint32_T, K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxINT64_CLASS:
                copyIntoFlann<int64_T,  K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxUINT64_CLASS:
                copyIntoFlann<uint64_T, K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxSINGLE_CLASS:
                copyIntoFlann<float,    K>(bag, rows, cols, flann_bags[i]);
                break;

            case mxDOUBLE_CLASS:
                copyIntoFlann<double,   K>(bag, rows, cols, flann_bags[i]);
                break;

            default:
                mexErrMsgTxt("unsupported bag type");
        }
    }

    return flann_bags;
}

void free_matalloced_matrix_array(MatrixF *bags, mwSize n) {
    for (mwSize i = 0; i < n; i++)
        mxFree(bags[i].ptr());
    mxFree(bags);
}


////////////////////////////////////////////////////////////////////////////////
// Helpers to convert from C++ to MATLAB types

template <typename T> struct matlab_classid {
    mxClassID mxID;
    matlab_classid() : mxID(mxUNKNOWN_CLASS) {}
};
template<> matlab_classid<int8_T>  ::matlab_classid() : mxID(mxINT8_CLASS) {}
template<> matlab_classid<uint8_T> ::matlab_classid() : mxID(mxUINT8_CLASS) {}
template<> matlab_classid<int16_T> ::matlab_classid() : mxID(mxINT16_CLASS) {}
template<> matlab_classid<uint16_T>::matlab_classid() : mxID(mxUINT16_CLASS) {}
template<> matlab_classid<int32_T> ::matlab_classid() : mxID(mxINT32_CLASS) {}
template<> matlab_classid<uint32_T>::matlab_classid() : mxID(mxUINT32_CLASS) {}
template<> matlab_classid<int64_T> ::matlab_classid() : mxID(mxINT64_CLASS) {}
template<> matlab_classid<uint64_T>::matlab_classid() : mxID(mxUINT64_CLASS) {}
template<> matlab_classid<float>   ::matlab_classid() : mxID(mxSINGLE_CLASS) {}
template<> matlab_classid<double>  ::matlab_classid() : mxID(mxDOUBLE_CLASS) {}

// make a MATLAB row vector from a vector<T>
template <typename T>
mxArray *make_vector(const vector<T> vec) {
    mwSize n = vec.size();

    mxArray* mat =
        mxCreateNumericMatrix(1, n, matlab_classid<T>().mxID, mxREAL);
    T* data = (T*) mxGetData(mat);

    for (size_t i = 0; i < n; i++)
        data[i] = vec[i];
    return mat;
}

// make a MATLAB matrix from a vector<vector<T>>
// assumes the inner vectors are of equal length
template <typename T>
mxArray *make_matrix(const vector< vector<T> > vec_matrix) {
    mwSize m = vec_matrix.size();
    mwSize n = m > 0 ? vec_matrix[0].size() : 0;

    mxClassID id = matlab_classid<T>().mxID;
    mxArray* mat = mxCreateNumericMatrix(m, n, id, mxREAL);
    T* data = (T*) mxGetData(mat);

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++)
            data[i + j*m] = vec_matrix[i][j];
    return mat;
}

// make a MATLAB matrix from a flann::Matrix<T>
template <typename T>
mxArray *make_matrix(const flann::Matrix<T> bag) {
    mxClassID id = matlab_classid<T>().mxID;
    mxArray* mat = mxCreateNumericMatrix(bag.rows, bag.cols, id, mxREAL);
    T* data = (T*) mxGetData(mat);

    for (size_t i = 0; i < bag.rows; i++)
        for (size_t j = 0; j < bag.cols; j++)
            data[i + j*bag.rows] = bag[i][j];
    return mat;
}

// make a MATLAB cell vector of matrices
template <typename T>
mxArray *make_matrix_cells(const flann::Matrix<T> *bags, size_t n) {
    mxArray *cells = mxCreateCellMatrix(1, n);

    for (size_t i = 0; i < n; i++)
        mxSetCell(cells, i, make_matrix(bags[i]));

    return cells;
}


////////////////////////////////////////////////////////////////////////////////
// Constructor / destructor

template <typename Scalar>
void destroy(SDM<Scalar> *model) {
    model->destroyModelAndProb();
    model->destroyTrainBags();
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

    void parseOpt(string name, mxArray* val) {
        if (name == "div_func") {
            div_func = get_string(val, "div_func must be a string");

        } else if (name == "kernel") {
            kernel = get_string(val, "kernel must be a string");

        } else if (name == "k") {
            k = get_size_t(val, "k must be a positive integer");
            if (k < 1)
                mexErrMsgTxt("k must be a positive integer");

        } else if (name == "tuning_folds") {
            tuning_folds = get_size_t(val,
                            "tuning_folds must be a positive integer");
            if (tuning_folds < 2)
                mexErrMsgTxt("tuning_folds must be at least 2");

        } else if (name == "probability") {
            probability = get_bool(val, "probability must be boolean");

        } else if (name == "num_threads") {
            num_threads = get_size_t(val,
                    "num_threads must be a nonnegative integer");

        } else if (name == "index") {
            index_type = get_string(val, "index must be a string");

        } else {
            mexErrMsgTxt(("unknown training option: " + name).c_str());
        }
    }

    npdivs::DivFunc* getDivFunc() const {
        return npdivs::div_func_from_str(div_func);
    }

    sdm::KernelGroup* getKernelGroup() const {
        if (kernel == "gaussian") {
            return new sdm::GaussianKernelGroup;
        } else if (kernel == "linear") {
            return new sdm::LinearKernelGroup;
        } else if (kernel == "polynomial") {
            return new sdm::PolynomialKernelGroup;
        } else {
            mexErrMsgTxt(("unkown kernel type: " + kernel).c_str());
        }
    }

    // even though this looks like object slicing, it's not, i promise
    flann::IndexParams getIndexParams() const {
        if (index_type == "linear" || index_type == "brute") {
            flann::LinearIndexParams ps;
            return ps;
        } else if (index_type == "kdtree" || index_type == "kd") {
            flann::KDTreeSingleIndexParams ps;
            return ps;
        } else {
            mexErrMsgTxt(("unknown index type: " + index_type).c_str());
        }
    }

    svm_parameter getSVMParams() const {
        svm_parameter svm_params = sdm::default_svm_params;
        svm_params.probability = (int) probability;
        return svm_params;
    }

    DivParams getDivParams() const {
        flann::SearchParams search_params(-1);
        return DivParams(k, getIndexParams(), search_params, num_threads);
    }

    vector<double> getCvals() const {
        return sdm::default_c_vals;
    }
};

template <typename Scalar>
SDM<Scalar> * train(
        const mxArray* bags_m, const mxArray* labels_m, const mxArray* opts_m)
{
    // first argument: training bags
    mwSize num_train = mxGetNumberOfElements(bags_m);
    MatrixF *bags = get_matrix_array<float>(bags_m, num_train, false);
    // XXX these bags need to live as long as the SDM does, so alloc w/ new

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
        opts.parseOpt(
                string(mxGetFieldNameByNumber(opts_m, i)),
                mxGetFieldByNumber(opts_m, 0, i));
    }

    npdivs::DivFunc* div_func = opts.getDivFunc();
    sdm::KernelGroup* kernel_group = opts.getKernelGroup();

    SDMF *model;

    try {
        // train away!
        model = sdm::train_sdm<Scalar>(
                bags, num_train, labels,
                *div_func, *kernel_group,
                opts.getDivParams(),
                opts.getCvals(),
                opts.getSVMParams(),
                opts.tuning_folds);

    } catch (...) { delete kernel_group; delete div_func; throw; }
    delete kernel_group;
    delete div_func;

    return model;
}

////////////////////////////////////////////////////////////////////////////////
// Cross-validation

struct CrossValidationOptions : public TrainingOptions {
    typedef TrainingOptions super;

    size_t folds;
    size_t cv_threads;
    bool project_all;

    CrossValidationOptions() :
        super(), folds(10), cv_threads(0), project_all(true)
    {}

    void parseOpt(string name, mxArray* val) {
        if (name == "folds") {
            folds = get_size_t(val, "folds must be a positive integer");
            if (folds < 2)
                mexErrMsgTxt("folds must be at least 2");

        } else if (name == "cv_threads") {
            cv_threads = get_size_t(val,
                    "cv_threads must be a nonnegative integer");
        } else if (name == "project_all") {
            project_all = get_bool(val, "project_all must be a boolean");
        } else {
            super::parseOpt(name, val);
        }
    }
};

template <typename Scalar>
double crossvalidate(
        const mxArray* bags_m, const mxArray* labels_m, const mxArray* opts_m,
        const mxArray* divs_m)
{
    // first argument: bags
    mwSize num = mxGetNumberOfElements(bags_m);
    MatrixF *bags = get_matrix_array<float>(bags_m, num, true);
    // XXX these bags can (and should) die when we exit the function

    // second argument: labels
    const vector<int> labels = get_vector<int>(labels_m,
            "second argument must be an array of integers");
    if (labels.size() > num)
        mexErrMsgTxt("got more labels than bags");
    else if (labels.size() < num)
        mexErrMsgTxt("got more bags than labels");

    // third argument: options
    if (!mxIsStruct(opts_m) && mxGetNumberOfElements(opts_m) == 1)
        mexErrMsgTxt("crossvalidate options must be a single struct");
    mwSize nfields = mxGetNumberOfFields(opts_m);

    CrossValidationOptions opts;
    for (mwSize i = 0; i < nfields; i++) {
        opts.parseOpt(
                string(mxGetFieldNameByNumber(opts_m, i)),
                mxGetFieldByNumber(opts_m, 0, i));
    }

    npdivs::DivFunc* div_func = opts.getDivFunc();
    sdm::KernelGroup* kernel_group = opts.getKernelGroup();

    if (opts.folds > num)
        opts.folds = num;

    // fourth argument: precomputed divergences
    double *divs;
    if (divs_m == NULL) {
        divs = NULL;
    } else {
        if (mxGetNumberOfDimensions(divs_m) != 2 ||
                mxGetM(divs_m) != num || mxGetN(divs_m) != num) {
            mexWarnMsgTxt("precomputed divergences not n x n; ignoring them");
            divs = NULL;
        } else {
            divs = (double*) mxCalloc(num * num, sizeof(double));
            MatrixD divs_f = get_matrix(divs_m, divs);
        }
    }

    double acc;
    try {
        // train away!
        acc = sdm::crossvalidate<Scalar>(
                bags, num, labels,
                *div_func, *kernel_group,
                opts.getDivParams(),
                opts.folds,
                opts.project_all,
                opts.cv_threads,
                opts.getCvals(),
                opts.getSVMParams(),
                opts.tuning_folds,
                divs);

    } catch (...) {
        delete kernel_group; delete div_func;
        free_matalloced_matrix_array(bags, num);
        throw;
    }
    delete kernel_group;
    delete div_func;
    free_matalloced_matrix_array(bags, num);
    if (divs != NULL)
        mxFree(divs);

    return acc;
}


////////////////////////////////////////////////////////////////////////////////
// Dispatch function

void dispatch(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    // first arg is a string saying what the desired operation is
    string op = get_string(prhs[0], "first input must be a string");

    if (op == "predict") {
        if (nrhs != 3) mexErrMsgTxt("predict needs exactly 2 arguments");
        if (nlhs < 1 || nlhs > 2) mexErrMsgTxt("predict returns 1-2 values");

        SDMF *model = convertMat2Ptr<SDMF>(prhs[1])->getPointer();

        mwSize n = mxGetNumberOfElements(prhs[2]);
        MatrixF *test_bags = get_matrix_array<float>(prhs[2], n);
        // these are allocated by matlab, so will die on mex exit

        if (nlhs == 2) {
            vector< vector<double> > vals;
            const vector<int> labels = model->predict(test_bags, n, vals);
            plhs[0] = make_vector(labels);
            plhs[1] = make_matrix(vals);
        } else {
            const vector<int> labels = model->predict(test_bags, n);
            plhs[0] = make_vector(labels);
        }

        free_matalloced_matrix_array(test_bags, n);

    } else if (op == "train") {
        if (nrhs != 4) mexErrMsgTxt("train needs exactly three arguments");
        if (nlhs != 1) mexErrMsgTxt("train returns exactly 1 output");

        SDMF *model = train<float>(prhs[1], prhs[2], prhs[3]);

        class_handle<SDMF> * ptr = new class_handle<SDMF>(model);
        plhs[0] = convertPtr2Mat<SDMF>(ptr);

    } else if (op == "delete") {
        if (nrhs != 2) mexErrMsgTxt("delete needs exactly one argument");
        if (nlhs > 0) mexErrMsgTxt("delete doesn't return anything");

        destroy(convertMat2Ptr<SDMF>(prhs[1])->getPointer());

    } else if (op == "train_bags") {
        if (nrhs != 2) mexErrMsgTxt("train_bags takes exactly one argument");
        if (nlhs != 1) mexErrMsgTxt("train_bags returns exactly 1 output");

        SDMF *model = convertMat2Ptr<SDMF>(prhs[1])->getPointer();
        plhs[0] = make_matrix_cells(model->getTrainBags(),
                                    model->getNumTrain());

    } else if (op == "info") {
        if (nrhs != 2) mexErrMsgTxt("info takes exactly one argument");
        if (nlhs != 4) mexErrMsgTxt("info returns exactly 4 outputs");

        SDMF *model = convertMat2Ptr<SDMF>(prhs[1])->getPointer();
        plhs[0] = mxCreateString(model->name().c_str());
        plhs[1] = mxCreateString(model->getKernel()->name().c_str());
        plhs[2] = mxCreateString(model->getDivFunc()->name().c_str());
        plhs[3] = mxCreateDoubleScalar(model->getSVM()->param.C);

    } else if (op == "crossvalidate") {
        if (nrhs < 4 || nrhs > 5)
            mexErrMsgTxt("crossvalidate takes 3-4 arguments");
        if (nlhs != 1) mexErrMsgTxt("crossvalidate returns exactly 1 output");

        const mxArray* divs = (nrhs >= 5 && !mxIsEmpty(prhs[4])) ?
                              prhs[4] : NULL;
        double acc = crossvalidate<float>(prhs[1], prhs[2], prhs[3], divs);
        plhs[0] = mxCreateDoubleScalar(acc);

    }  else {
        mexErrMsgTxt(("Unknown operation '" + op + "'.").c_str());
    }
}

void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    try {
        dispatch(nlhs, plhs, nrhs, prhs);
    } catch (std::exception &e) {
        mexErrMsgTxt((boost::format("exception: %s") % e.what()).str().c_str());
    } catch (...) {
        mexErrMsgTxt("unknown error in sdm_mex");
    }
}
