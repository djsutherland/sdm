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

from ctypes import POINTER, c_int, c_float, c_double, sizeof
import numbers

import numpy as np

from . import sdm_ctypes as lib
from .sdm_ctypes import c_size_t


_dtypes = [('i', c_int), ('i', c_size_t), ('f', c_float), ('f', c_double)]
_np_to_c_types = {}
_c_to_np_types = {}
for pref, c_type in _dtypes:
    np_type = np.dtype('<%s%d' % (pref, sizeof(c_type)))
    _np_to_c_types[np_type] = c_type
    _c_to_np_types[c_type] = np_type


################################################################################
### Parameter checking

_intypes = frozenset(map(np.dtype, (np.float, np.double)))
_labtypes = frozenset(map(np.dtype, (np.int, np.double)))

def _check_bags(bags):
    bags = [np.ascontiguousarray(bag) for bag in bags]
    if len(bags) <= 1:
        raise ValueError("not enough bags to cross-validate")

    if len(bags[0].shape) != 2:
        raise ValueError("bags must be 2d arrays with consistent 2nd dim")
    dim = bags[0].shape[1]

    dtype = bags[0].dtype
    if dtype not in _intypes:
        raise TypeError("%r not valid datatype for bags" % dtype)

    for bag in bags:
        if len(bag.shape) != 2 or bag.shape[1] != dim:
            raise ValueError("bags must be 2d arrays with consistent 2nd dim")
        if bag.dtype != dtype:
            raise TypeError("bags must have consistent datatype")

    return bags


def _check_labels(labels, num_bags):
    if isinstance(labels[0], str):
        raise NotImplemented("str label mapping not done yet") # TODO

    elif isinstance(labels[0], numbers.Integral):
        dtype = _c_to_np_types[c_int]

    elif isinstance(labels[0], numbers.Real):
        dtype = _c_to_np_types[c_double]

    else:
        raise TypeError("unknown label type %r" % labels[0].__class__)

    labels = np.squeeze(np.ascontiguousarray(labels, dtype=dtype))
    if labels.shape != (num_bags,):
        raise ValueError("must be as many labels as bags")

    return labels


def _check_c_vals(c_vals):
    if c_vals is not None:
        c_vals = np.squeeze(np.ascontiguousarray(c_vals, dtype=np.double))
        if len(c_vals.shape) != 1 or c_vals.size < 1:
            raise ValueError("c_vals must be a non-empty 1d array of values")
        return c_vals, c_vals.ctypes.data_as(POINTER(c_double)), c_vals.size
    else:
        return None, lib.default_c_vals, lib.num_default_c_vals


def _make_svm_params(labtype, p=None, cache_size=None, eps=None,
                     shrinking=None, probability=None):
    svm_params = lib.SVMParams()
    if labtype == np.double:
        svm_params.svm_type = lib.SVMType.EPSILON_SVR

    if p is not None:
        svm_params.p = p
    if cache_size is not None:
        svm_params.cache_size = cache_size
    if eps is not None:
        svm_params.eps = eps
    if shrinking is not None:
        svm_params.shrinking = shrinking
    if probability is not None:
        svm_params.probability = probability

    return svm_params

################################################################################


def crossvalidate(bags, labels, folds=10, project_all=True, shuffle=True,
        k=None, tuning_folds=3,
        div_func="renyi:.9", kernel="gaussian",
        cv_threads=0, num_threads=None,
        flann_params={},
        svm_regression_eps=None, svm_cache_size=None, svm_eps=None,
        svm_shrinking=None, probability=False, 
        c_vals=None, show_progress=0, print_progress=None):
    '''
    Cross-validates an SDM's ability to classify/regress bags into labels.

        * flann_params: a dict of args for a FLANNParameters struct
    '''

    # check params
    bags = _check_bags(bags)
    labels = _check_labels(labels, len(bags))

    # get ctypes types for input, output
    intype = _np_to_c_types[bags[0].dtype]
    intype_p = POINTER(intype)

    labtype = _np_to_c_types[labels.dtype]
    
    # make needed bag data
    bag_ptrs = (intype_p * len(bags))(
            *[bag.ctypes.data_as(intype_p) for bag in bags])
    bag_rows = np.ascontiguousarray(
            [bag.shape[0] for bag in bags], dtype=_c_to_np_types[c_size_t])

    # make div params
    flann_p = lib.FLANNParameters()
    flann_p.update(**flann_params)

    div_params = lib.DivParams()
    div_params.flann_params = flann_p
    if k is not None:
        div_params.k = k
    if num_threads is not None:
        div_params.num_threads = num_threads
    if show_progress is not None:
        div_params.show_progress = show_progress
    if print_progress is not None:
        div_params.print_progress = lib.print_progress_type(print_progress)

    # make c_vals array
    c_vals, c_vals_p, num_c_vals = _check_c_vals(c_vals)

    # make svm params
    svm_params = _make_svm_params(labtype=labels.dtype,
            p=svm_regression_eps,
            cache_size=svm_cache_size,
            eps=svm_eps,
            shrinking=svm_shrinking,
            probability=probability)

    # call the function!
    score = lib.crossvalidate[intype, labtype](
            bag_ptrs,
            len(bags),
            bag_rows.ctypes.data_as(POINTER(c_size_t)),
            bags[0].shape[1],
            labels.ctypes.data_as(POINTER(labtype)),
            div_func.encode('ascii'),
            kernel.encode('ascii'),
            div_params,
            folds, cv_threads, int(project_all), int(shuffle),
            c_vals_p, num_c_vals,
            svm_params,
            tuning_folds)
    return score
