################################################################################
# Copyright (c) 2012, Dougal J. Sutherland (dsutherl@cs.cmu.edu).              #
# All rights reserved.                                                         #
#                                                                              #
# Portions included from FLANN:                                                #
# Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.   #
# Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.    #
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
#     * Neither the name of Carnegie Mellon University, the University of      #
#       British Columbia, nor the names of the contributors may be used to     #
#       endorse or promote products derived from this software without         #
#       specific prior written permission.                                     #
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

# TODO: use ndpointer to verify things are contiguous, aligned, right dims...

from __future__ import absolute_import

from ctypes import (cdll, Structure, POINTER, CFUNCTYPE, pointer,
        c_short, c_int, c_uint, c_long, c_ulong, c_float, c_double, c_char_p)
from ctypes.util import find_library

from itertools import product

from .six import with_metaclass, itervalues, iteritems

from numpy import issubclass_

c_size_t = c_ulong
_LIB = cdll[find_library('sdm')]

################################################################################

# Enumeration class
# based on http://code.activestate.com/recipes/576415-ctype-enumeration-class/

class EnumMeta(type(c_uint)):
    def __new__(cls, name, bases, classdict):
        # figure out which members of the class are enum items
        _members = {}
        _rev_members = {}
        for k, v in iteritems(classdict):
            if not k.startswith('_'):
                try:
                    c_uint(v)
                except TypeError:
                    pass
                else:
                    if not k == k.upper():
                        raise ValueError("Enum values must be all-uppercase")

                    classdict[k] = _members[k] = v
                    _rev_members[v] = k

        # construct the class
        classdict['_members'] = _members
        classdict['_rev_members'] = _rev_members
        the_type = type(c_uint).__new__(cls, name, bases, classdict)

        # now that the class is finalized, switch members to be of the class
        for k, v in iteritems(_members):
            as_class = the_type(v)
            the_type._members[k] = as_class
            setattr(the_type, k, as_class)

        return the_type

    def __contains__(self, value):
        return value in itervalues(self._members)

    def __repr__(self):
        return "<Enumeration %s>" % self.__name__


class Enum(with_metaclass(EnumMeta, c_uint)):
    def __init__(self, value):
        if isinstance(value, self.__class__):
            value = value.value
        else:
            try:
                value = self._members[value.upper()].value
            except (AttributeError, KeyError):
                if value not in self._rev_members:
                    raise ValueError("invalid %s value %r" %
                            (self.__class__.__name__, value))

        super(Enum, self).__init__(value)

    @property
    def name(self):
        try:
            return self._rev_members[self.value]
        except KeyError:
            raise ValueError("Bad %r value %r" % (self.__class__, self.value))

    @classmethod
    def from_param(cls, param):
        if hasattr(param, 'upper'):
            s = param.upper()
            try:
                return getattr(cls, s)
            except AttributeError:
                raise ValueError("Bad %s value %r" % (cls.__name__, param))
        return param

    def __repr__(self):
        return "<member %s=%d of %r>" % (self.name, self.value, self.__class__)


# XXX: here because otherwise __init__ doesn't seem to get called
def returns_enum(enum_subclass):
    def inner(value):
        return enum_subclass(value)
    return inner

################################################################################
### Custom structure class, with defaults and nicer enumeration support

# extremely loosely based on code from pyflann.flann_ctypes

_identity = lambda x: x
class CustomStructure(Structure):
    _defaults_ = {}
    __enums = {}

    def __init__(self):
        Structure.__init__(self)
        self.__enums = dict((f, t) for f, t in self._fields_
                            if issubclass_(t, Enum))

        for field, val in iteritems(self._defaults_):
            setattr(self, field, val)

    def __setattr__(self, k, v):
        class_wrapper = self.__enums.get(k, _identity)
        super(CustomStructure, self).__setattr__(k, class_wrapper(v))

    def update(self, **vals):
        for k, v in iteritems(vals):
            setattr(self, k, v)


################################################################################
### FLANN parameters

class FLANNAlgorithm(Enum):
    LINEAR = 0
    KDTREE = 1
    KMEANS = 2
    COMPOSITE = 3
    KDTREE_SIMPLE = 4
    SAVED = 254
    AUTOTUNED = 255

class FLANNCentersInit(Enum):
    RANDOM = 0
    GONZALES = 1
    KMEANSPP = 2

class FLANNLogLevel(Enum):
    NONE = 0
    FATAL = 1
    ERROR = 2
    WARNING = 3
    INFO = 4


class FLANNParameters(CustomStructure):
    _fields_ = [
        ('algorithm', FLANNAlgorithm),
        ('checks', c_int),
        ('cb_index', c_float),
        ('eps', c_float),
        ('trees', c_int),
        ('leaf_max_size', c_int),
        ('branching', c_int),
        ('iterations', c_int),
        ('centers_init', FLANNCentersInit),
        ('target_precision', c_float),
        ('build_weight', c_float),
        ('memory_weight', c_float),
        ('sample_fraction', c_float),
        ('table_number_', c_uint),
        ('key_size_', c_uint),
        ('multi_probe_level_', c_uint),
        ('log_level', FLANNLogLevel),
        ('random_seed', c_long),
    ]
    _defaults_ = {
        'algorithm' : FLANNAlgorithm.KDTREE,
        'checks' : 32,
        'eps' : 0.0,
        'cb_index' : 0.5,
        'trees' : 1,
        'leaf_max_size' : 4,
        'branching' : 32,
        'iterations' : 5,
        'centers_init' : FLANNCentersInit.RANDOM,
        'target_precision' : 0.9,
        'build_weight' : 0.01,
        'memory_weight' : 0.0,
        'sample_fraction' : 0.1,
        'table_number_': 12,
        'key_size_': 20,
        'multi_probe_level_': 2,
        'log_level' : FLANNLogLevel.WARNING,
        'random_seed' : -1
    }


################################################################################
### LibSVM parameters

class SVMType(Enum):
    C_SVC = 0
    NU_SVC = 1
    ONE_CLASS = 2
    EPSILON_SVR = 3
    NU_SVR = 4

class SVMKernelType(Enum):
    LINEAR = 0
    POLY = 1
    RBF = 2
    SIGMOID = 3
    PRECOMPUTED = 4


class SVMParams(CustomStructure):
    _fields_ = [
        ('svm_type', SVMType),
        ('kernel_type', SVMKernelType),
        ('degree', c_int),
        ('gamma', c_double),
        ('coef0', c_double),

        ('cache_size', c_double), # in MB
        ('eps', c_double),
        ('C', c_double),

        ('nr_weight', c_int),
        ('weight_label', POINTER(c_int)),
        ('weight', POINTER(c_double)),

        ('nu', c_double),
        ('p', c_double),
        ('shrinking', c_int),
        ('probability', c_int),
    ]

    _defaults_ = {
        'svm_type': SVMType.C_SVC,
        'kernel_type': SVMKernelType.PRECOMPUTED,
        'degree': 0, 'gamma': 0, 'coef0': 0,

        'cache_size': 1024,
        'eps': 1e-3,
        'C': 1,

        'nr_weight': 0,
        'weight_label': None,
        'weight': None,

        'nu': 0,
        'p': 0.1,
        'shrinking': 1,
        'probability': 0,
    }


################################################################################
### Logging stuff

class LogLevel(Enum):
    ERROR, WARNING, INFO, DEBUG, DEBUG1, DEBUG2, DEBUG3, DEBUG4 = range(8)

set_log_level = _LIB.sdm_set_log_level
set_log_level.restype = None
set_log_level.argtypes = [LogLevel]

get_log_level = _LIB.sdm_get_log_level
get_log_level.restype = returns_enum(LogLevel)
get_log_level.argtypes = []

set_log_level('warning') # defaults to debug2 for some reason

################################################################################
### Div parameters

print_progress_to_stderr = _LIB.print_progress_to_stderr
print_progress_to_stderr.restype = None
print_progress_to_stderr.argtypes = [c_size_t]

print_progress_type = CFUNCTYPE(None, c_size_t)

class DivParams(CustomStructure):
    _fields_ = [
        ('k', c_int),
        ('flann_params', FLANNParameters),
        ('num_threads', c_size_t),
        ('show_progress', c_size_t),
        ('print_progress', POINTER(print_progress_type)),
    ]

    _defaults_ = {
        'k': 3,
        'num_threads': 0,
        'show_progress': 0,
        'print_progress': pointer(print_progress_type(print_progress_to_stderr))
    }
    # FIXME: print_progress causes bus error :(

################################################################################
### np_divs wrapper

get_divs = {}
for name, intype in [('double', c_double), ('float', c_float)]:
    get_divs[intype] = fn = _LIB['np_divs_' + name]
    fn.restype = None
    fn.argtypes = [
        POINTER(POINTER(intype)), c_size_t, POINTER(c_size_t),
        POINTER(POINTER(intype)), c_size_t, POINTER(c_size_t),
        c_size_t,
        POINTER(c_char_p), c_size_t,
        POINTER(POINTER(c_double)),
        POINTER(DivParams),
    ]

################################################################################
### C values

default_c_vals = POINTER(c_double).in_dll(_LIB, 'default_c_vals')
num_default_c_vals = c_size_t.in_dll(_LIB, 'num_default_c_vals')

################################################################################
### SDM model stuff
class SDM_ClassifyD(Structure): pass
class SDM_ClassifyF(Structure): pass
class SDM_RegressD(Structure): pass
class SDM_RegressF(Structure): pass

_sdm_classes = [
    (SDM_ClassifyD, c_double, c_int),
    (SDM_ClassifyF, c_float,  c_int),
    (SDM_RegressD,  c_double, c_double),
    (SDM_RegressF,  c_float,  c_double),
]

# name-getting
get_name = {}
for cls, intype, labtype in _sdm_classes:
    get_name[intype, labtype] = fn = _LIB[cls.__name__ + '_getName']
    fn.restype = c_char_p
    fn.argtypes = [POINTER(cls)]

# freeing
free_model = {}
for cls, intype, labtype in _sdm_classes:
    free_model[intype, labtype] = fn = _LIB[cls.__name__ + '_freeModel']
    fn.restype = c_char_p
    fn.argtypes = [POINTER(cls)]

# training
train = {}
for cls, intype, labtype in _sdm_classes:
    train[intype, labtype] = fn = _LIB[cls.__name__ + '_train']
    fn.restype = POINTER(cls)
    fn.argtypes = [
            POINTER(POINTER(intype)), c_size_t, c_size_t, POINTER(c_size_t),
            POINTER(labtype),
            c_char_p,
            c_char_p,
            POINTER(DivParams),
            POINTER(c_double), c_size_t,
            POINTER(SVMParams),
            c_size_t,
            POINTER(c_double),
    ]

# predict a single item, label only
predict = {}
for cls, intype, labtype in _sdm_classes:
    predict[intype, labtype] = fn = _LIB[cls.__name__ + '_predict']
    fn.restype = labtype
    fn.argtypes = [
            POINTER(cls),
            POINTER(intype),
            c_size_t,
    ]

# predict a single item, with decision values
predict_vals = {}
for cls, intype, labtype in _sdm_classes:
    predict_vals[intype, labtype] = fn = _LIB[cls.__name__ + '_predict_vals']
    fn.restype = labtype
    fn.argtypes = [
            POINTER(cls),
            POINTER(intype),
            c_size_t,
            POINTER(POINTER(intype)),
            POINTER(c_size_t),
    ]

# predict several items, labels only
predict_many = {}
for cls, intype, labtype in _sdm_classes:
    predict_many[intype, labtype] = fn = _LIB[cls.__name__ + '_predict_many']
    fn.restype = None
    fn.argtypes = [
            POINTER(cls),
            POINTER(POINTER(intype)),
            c_size_t,
            POINTER(c_size_t),
            POINTER(labtype),
    ]

# predict several items, with decision values
predict_many_vals = {}
for cls, intype, labtype in _sdm_classes:
    predict_many_vals[intype, labtype] = fn = \
            _LIB[cls.__name__ + '_predict_many_vals']
    fn.restype = None
    fn.argtypes = [
            POINTER(cls),
            POINTER(POINTER(intype)),
            c_size_t,
            POINTER(c_size_t),
            POINTER(labtype),
            POINTER(POINTER(POINTER(c_double))),
            POINTER(c_size_t),
    ]


################################################################################
### Cross-validation

# cross-validate on bags
_cv_kinds = [
    ('classify', 'double', c_double, c_int),
    ('classify', 'float',  c_float,  c_int),
    ('regress',  'double', c_double, c_double),
    ('regress',  'float',  c_float,  c_double),
]

crossvalidate = {}
for name, in_name, intype, labtype in _cv_kinds:
    crossvalidate[intype, labtype] = fn = \
            _LIB['sdm_crossvalidate_%s_%s' % (name, in_name)]
    fn.restype = c_double
    fn.argtypes = [
            POINTER(POINTER(intype)),
            c_size_t,
            POINTER(c_size_t),
            c_size_t,
            POINTER(labtype),
            c_char_p,
            c_char_p,
            POINTER(DivParams),
            c_size_t,
            c_size_t,
            c_short,
            c_short,
            POINTER(c_double), c_size_t,
            POINTER(SVMParams),
            c_size_t,
    ]


# CV on precomputed divs
crossvalidate_divs = {}
for name, labtype in \
        set((name, labtype) for name, in_name, intype, labtype in _cv_kinds):

    crossvalidate_divs[labtype] = fn = _LIB['sdm_crossvalidate_%s_divs' % name]
    fn.restype = c_double
    fn.argtypes = [
            POINTER(c_double),
            c_size_t,
            POINTER(labtype),
            c_char_p,
            c_size_t,
            c_size_t,
            c_short,
            c_short,
            POINTER(c_double), c_size_t,
            POINTER(SVMParams),
            c_size_t,
    ]
