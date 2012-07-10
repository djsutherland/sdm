################################################################################
# Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).              #
# All rights reserved.                                                         #
#                                                                              #
# Redistribution and use in source and binary forms, with or without           #
# modification, are permitted provided that the following conditions are met:  #
#                                                                              #
#     * Redistributions of source code must retain the above copyright         #
#       notice, this list of conditions and the following disclaimer.          #
#                                                                              #
#     * Redistributions in binary form must reproduce the above copyright      #
#       notice, this list of conditions and the following disclaimer in the    #
#       documentation and/or other materials provided with the distribution.   #
#                                                                              #
#     * Neither the name of Carnegie Mellon University nor the names of the    #
#       contributors may be used to endorse or promote products derived from   #
#       this software without specific prior written permission.               #
#                                                                              #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"  #
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE    #
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE   #
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE    #
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR          #
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS     #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN      #
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)      #
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                                  #
################################################################################

from __future__ import absolute_import

from ctypes import (POINTER, pointer, c_int, c_float, c_double, c_char_p,
                    cdll, sizeof, byref)
from ctypes.util import find_library
import numbers

import numpy as np

from . import sdm_ctypes as lib
from .sdm_ctypes import c_size_t
from .six import b

libc = cdll[find_library('c')]

_dtypes = [('i', c_int), ('i', c_size_t), ('f', c_float), ('f', c_double)]
_np_to_c_types = {}
_c_to_np_types = {}
for pref, c_type in _dtypes:
    np_type = np.dtype('<%s%d' % (pref, sizeof(c_type)))
    _np_to_c_types[np_type] = c_type
    _c_to_np_types[c_type] = np_type

c_double_p = POINTER(c_double)
c_size_t_p = POINTER(c_size_t)

# TODO: check for error conditions in responses

################################################################################
### Parameter checking

_intypes = frozenset(map(np.dtype, (np.float, np.double)))
_labtypes = frozenset(map(np.dtype, (np.int, np.double)))

_nothing = object()

def _check_bags(bags, dim=None, _intypes=_intypes):
    bags = [np.ascontiguousarray(bag) for bag in bags]
    if len(bags) <= 1:
        raise ValueError("not enough bags to cross-validate")

    if len(bags[0].shape) != 2:
        raise ValueError("bags must be 2d arrays with consistent 2nd dim")
    if dim is None:
        dim = bags[0].shape[1]

    dtype = bags[0].dtype
    if dtype not in _intypes:
        raise TypeError("%r not valid datatype for bags" % dtype)

    for bag in bags:
        if len(bag.shape) != 2 or bag.shape[1] != dim:
            raise ValueError("bags must be 2d arrays with consistent 2nd dim")
        if bag.dtype != dtype:
            raise TypeError("bags must have consistent datatype")

    intype = _np_to_c_types[dtype]
    intype_p = POINTER(intype)

    bag_ptrs = (intype_p * len(bags))(
            *(bag.ctypes.data_as(intype_p) for bag in bags))

    bag_rows = np.ascontiguousarray(
            [bag.shape[0] for bag in bags], dtype=_c_to_np_types[c_size_t])

    return bags, intype, bag_ptrs, bag_rows, dim

def _check_divs(divs, num_bags=None):
    # copy divs into row-major contiguous order
    # copy, so the caller's matrix isn't confusingly messed with
    divs = np.array(divs, dtype=_c_to_np_types[c_double], copy=True)
    if len(divs.shape) != 2:
        raise ValueError("divs must be 2-dimensional")
    a, b = divs.shape
    if a != b:
        raise ValueError("divs must be square")
    if num_bags is not None and a != num_bags:
        raise ValueError("divs must be num_bags x num_bags")
    return divs

def _check_labels(labels, num_bags):
    if isinstance(labels[0], str):
        raise NotImplemented("str label mapping not done yet") # TODO
    elif isinstance(labels[0], numbers.Integral):
        ctype = c_int
    elif isinstance(labels[0], numbers.Real):
        ctype = c_double
    else:
        raise TypeError("unknown label type %r" % labels[0].__class__)

    dtype = _c_to_np_types[ctype]

    labels = np.squeeze(np.ascontiguousarray(labels, dtype=dtype))
    if labels.shape != (num_bags,):
        raise ValueError("must be as many labels as bags")

    return labels, ctype


def _check_c_vals(c_vals):
    if c_vals is not None:
        c_vals = np.squeeze(np.ascontiguousarray(c_vals, dtype=np.double))
        if len(c_vals.shape) != 1 or c_vals.size < 1:
            raise ValueError("c_vals must be a non-empty 1d array of values")
        return c_vals, c_vals.ctypes.data_as(c_double_p), c_vals.size
    else:
        return None, lib.default_c_vals, lib.num_default_c_vals


def _make_svm_params(labtype, regression_eps=_nothing,
                     svm_cache_size=_nothing, svm_eps=_nothing,
                     svm_shrinking=_nothing, probability=_nothing, **junk):
    svm_params = lib.SVMParams()
    if labtype == c_double:
        svm_params.svm_type = lib.SVMType.EPSILON_SVR

    if regression_eps is not _nothing:
        svm_params.p = regression_eps
    if svm_cache_size is not _nothing:
        svm_params.cache_size = svm_cache_size
    if svm_eps is not _nothing:
        svm_params.eps = svm_eps
    if svm_shrinking is not _nothing:
        svm_params.shrinking = svm_shrinking
    if probability is not _nothing:
        svm_params.probability = probability

    return svm_params

def _make_flann_div_params(k=_nothing, num_threads=_nothing,
        show_progress=_nothing, print_progress=_nothing,
        flann_params={}, **junk):

    flann_p = lib.FLANNParameters()
    flann_p.update(**flann_params)

    div_params = lib.DivParams()
    div_params.flann_params = flann_p
    if k is not _nothing:
        div_params.k = k
    if num_threads is not _nothing:
        div_params.num_threads = num_threads
    if show_progress is not _nothing:
        div_params.show_progress = show_progress
    if print_progress is not _nothing:
        div_params.print_progress = lib.print_progress_type(
                print_progress if print_progress is not None else 0)

    return flann_p, div_params

################################################################################

def get_divs(x_bags, y_bags=None, div_funcs=['renyi:.9'], **kwargs):
    # check x bags
    x_bags, intype, x_bag_ptrs, x_bag_rows, dim = _check_bags(x_bags)
    num_x = len(x_bags)
    x_bag_rows_p = x_bag_rows.ctypes.data_as(c_size_t_p)

    # check y bags and make pointers
    if y_bags is None:
        num_y = num_x
        y_bag_ptrs = None
        y_bag_rows_p = None
    else:
        y_bags, _, y_bag_ptrs, y_bag_rows, _ = _check_bags(
                y_bags, dim=dim, _intypes=(x_bags[0].dtype,))
        num_y = len(y_bags)

        y_bag_rows_p = y_bag_rows.ctypes.data_as(c_size_t_p)

    # make pointer to div_funcs
    div_func_ptrs = (c_char_p * len(div_funcs))(*(b(df) for df in div_funcs))

    # make div params
    flann_p, div_params = _make_flann_div_params(**kwargs)

    # allocate results
    results = np.empty((len(div_funcs), num_x, num_y),
                       dtype=_c_to_np_types[c_double], order='C')
    results.fill(np.nan)
    results_p = (c_double_p * len(div_funcs))(*(
        r_df.ctypes.data_as(c_double_p) for r_df in results))

    # call the function
    lib.get_divs[intype](
            x_bag_ptrs, num_x, x_bag_rows_p,
            y_bag_ptrs, num_y, y_bag_rows_p,
            dim,
            div_func_ptrs, len(div_funcs),
            results_p,
            div_params
    )

    return results

################################################################################

# TODO: stuff involving SDM models

class SDM():
    def __init__(self, handle, intype, labtype, dim):
        self._handle = handle
        self._intype = intype
        self._labtype = labtype
        self._dim = dim

        self._name = lib.get_name[intype, labtype](handle)

    #def __del__(self):
    #    self.free()

    # XXX FIXME XXX causes segfaults...? (but doesn't from C interface)
    def free(self):
        lib.free_model[self.intype, self.labtype](self._handle)
        self._handle = None

    intype = property(lambda self: self._intype)
    labtype = property(lambda self: self._labtype)

    name = property(lambda self: self._name)
    dim = property(lambda self: self._dim)

    in_dtype = property(lambda self: _c_to_np_types[self.intype])
    lab_dtype = property(lambda self: _c_to_np_types[self.labtype])

    def __repr__(self):
        return '<%s>' % self.name.decode('ascii')

    def predict(self, bags, get_dec_values=False):
        intype = self.intype
        labtype = self.labtype

        elem_ndim = len(np.asanyarray(bags[0]).shape)
        if not 1 <= elem_ndim <= 2:
            raise ValueError("bags must have 2 or 3 dimensions")

        if elem_ndim == 1:
            bag = np.ascontiguousarray(bags, dtype=self.in_dtype)
            if bag.shape[1] != self.dim:
                raise ValueError("test bag must be of dimension %d" % self.dim)

            bag_ptr = bag.ctypes.data_as(POINTER(self.intype))

            if get_dec_values:
                vals_p = c_double_p()
                num_vals = c_size_t()

                pred_label = lib.predict_vals[intype, labtype](
                        self._handle, bag_ptr, bag.shape[0],
                        byref(vals_p), byref(num_vals))

                vals = np.empty(num_vals.value)
                for i in range(num_vals.value):
                    vals[i] = vals_p[i]
                libc.free(vals_p)

                return pred_label, vals

            else:
                pred_label = lib.predict[intype, labtype](
                        self._handle, bag_ptr, bag.shape[0])
                return pred_label

        elif elem_ndim == 2:
            bags, _, bag_ptrs, bag_rows, _ = \
                    _check_bags(bags, self.dim, (self.in_dtype,))
            bag_rows_p = bag_rows.ctypes.data_as(c_size_t_p)

            preds = np.empty(len(bags), dtype=self.lab_dtype)
            preds_p = preds.ctypes.data_as(POINTER(labtype))

            if get_dec_values:
                vals_p = (POINTER(c_double_p))()
                num_vals = c_size_t()

                lib.predict_many_vals[intype, labtype](
                        self._handle,
                        bag_ptrs, len(bags), bag_rows_p,
                        preds_p,
                        byref(vals_p), byref(num_vals))

                vals = np.empty((len(bags), num_vals.value))
                for i in range(len(bags)):
                    bag_vals_p = vals_p[i]
                    for j in range(num_vals.value):
                        vals[i, j] = bag_vals_p[j]
                    libc.free(bag_vals_p)
                libc.free(vals_p)

                return preds, vals

            else:
                lib.predict_many[intype, labtype](
                        self._handle,
                        bag_ptrs, len(bags), bag_rows_p,
                        preds_p)

                return preds



def train(bags, labels,
          div_func='renyi:.9', kernel='gaussian',
          tuning_folds=3, c_vals=None, divs=None, **kwargs):

    bags, intype, bag_ptrs, bag_rows, dim = _check_bags(bags)
    labels, labtype = _check_labels(labels, len(bags))
    if divs is not None:
        divs = _check_divs(divs, len(bags))

    flann_p, div_params = _make_flann_div_params(**kwargs)
    c_vals, c_vals_p, num_c_vals = _check_c_vals(c_vals)
    svm_params = _make_svm_params(labtype=labtype, **kwargs)

    handle = lib.train[intype, labtype](
            bag_ptrs, len(bags), dim,
            bag_rows.ctypes.data_as(c_size_t_p),
            labels.ctypes.data_as(POINTER(labtype)),
            div_func.encode('ascii'),
            kernel.encode('ascii'),
            div_params,
            c_vals_p, num_c_vals,
            svm_params,
            tuning_folds,
            divs.ctypes.data_as(c_double_p) if divs is not None else None)

    return SDM(handle, intype, labtype, dim)

################################################################################

def transduct(train_bags, train_labels, test_bags,
          div_func='renyi:.9', kernel='gaussian',
          tuning_folds=3, c_vals=None, divs=None, **kwargs):

    train_bags, intype, train_bag_ptrs, train_bag_rows, dim = \
            _check_bags(train_bags)
    train_bag_rows_p = train_bag_rows.ctypes.data_as(c_size_t_p)

    test_bags, _, test_bag_ptrs, test_bag_rows, _ = \
            _check_bags(test_bags, dim=dim, _intypes=(intype,))
    test_bag_rows_p = test_bag_rows.ctypes.data_as(c_size_t_p)

    train_labels, labtype = _check_labels(train_labels, len(train_bags))
    train_labels_p = train_labels.ctypes.data_as(POINTER(labtype))

    if divs is not None:
        divs = _check_divs(divs, len(train_bags) + len(test_bags))

    flann_p, div_params = _make_flann_div_params(**kwargs)
    c_vals, c_vals_p, num_c_vals = _check_c_vals(c_vals)
    svm_params = _make_svm_params(labtype=labtype, **kwargs)

    preds = np.empty(len(test_bags), dtype=_c_to_np_types[labtype])
    preds_p = preds.ctypes.data_as(POINTER(labtype))

    lib.transduct[intype, labtype](
            train_bag_ptrs, len(train_bags), train_bag_rows_p,
            test_bag_ptrs, len(test_bags), test_bag_rows_p,
            dim, train_labels_p,
            div_func.encode('ascii'),
            kernel.encode('ascii'),
            div_params,
            c_vals_p, num_c_vals,
            svm_params,
            tuning_folds,
            divs.ctypes.data_as(c_double_p) if divs is not None else None,
            preds_p)

    return preds


################################################################################

def crossvalidate(bags, labels, folds=10, project_all=True, shuffle=True,
        div_func="renyi:.9", kernel="gaussian", tuning_folds=3,
        cv_threads=0, c_vals=None, **kwargs):
    '''
    Cross-validates an SDM's ability to classify/regress bags into labels.

        * flann_params: a dict of args for a FLANNParameters struct
    '''

    bags, intype, bag_ptrs, bag_rows, dim = _check_bags(bags)
    labels, labtype = _check_labels(labels, len(bags))

    flann_p, div_params = _make_flann_div_params(**kwargs)
    c_vals, c_vals_p, num_c_vals = _check_c_vals(c_vals)
    svm_params = _make_svm_params(labtype=labtype, **kwargs)

    score = lib.crossvalidate[intype, labtype](
            bag_ptrs, len(bags),
            bag_rows.ctypes.data_as(c_size_t_p), dim,
            labels.ctypes.data_as(POINTER(labtype)),
            div_func.encode('ascii'),
            kernel.encode('ascii'),
            div_params,
            folds, cv_threads, int(project_all), int(shuffle),
            c_vals_p, num_c_vals,
            svm_params,
            tuning_folds)
    return score

def crossvalidate_divs(divs, labels, folds=10, project_all=True, shuffle=True,
        kernel='gaussian', tuning_folds=3, cv_threads=0, c_vals=None, **kwargs):
    '''
    Cross-validates an SDM's ability to classify/regress samples with
    precomputed divergences divs into labels.
    '''

    divs = _check_divs(divs)
    labels, labtype = _check_labels(labels, divs.shape[0])

    flann_p, div_params = _make_flann_div_params(**kwargs)
    c_vals, c_vals_p, num_c_vals = _check_c_vals(c_vals)
    svm_params = _make_svm_params(labtype=labtype, **kwargs)

    score = lib.crossvalidate_divs[labtype](
            divs.ctypes.data_as(c_double_p),
            divs.shape[0],
            labels.ctypes.data_as(POINTER(labtype)),
            kernel.encode('ascii'),
            folds, cv_threads, int(project_all), int(shuffle),
            c_vals_p, num_c_vals,
            svm_params,
            tuning_folds)
    return score
