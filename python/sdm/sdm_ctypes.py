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

# TODO: python 2/3 compatability
# things to change:
#     - metaclass syntax for Enum
#     - .values(), .items()

from ctypes import (cdll, Structure, POINTER, CFUNCTYPE,
        c_short, c_int, c_uint, c_long, c_ulong, c_float, c_double, c_char_p)
from ctypes.util import find_library

from itertools import product

c_size_t = c_ulong
_LIB = cdll[find_library('sdm')]

################################################################################

# based on http://code.activestate.com/recipes/576415-ctype-enumeration-class/

class EnumMeta(type(c_uint)):
    def __new__(cls, name, bases, classdict):
        _members = {}
        for k, v in classdict.items():
            if not k.startswith('_'):
                try:
                    v = c_uint(v)
                except TypeError:
                    pass
                else:
                    if not k == k.upper():
                        raise ValueError("Enum values must be all-uppercase")
                    classdict[k] = v
                    _members[k] = v
        classdict['_members'] = _members
        return type(c_uint).__new__(cls, name, bases, classdict)

    def __contains__(self, value):
        return value in self._members.values()

    def __repr__(self):
        return "<Enumeration %s>" % self.__name__

class Enum(c_uint, metaclass=EnumMeta):
    def __init__(self, value):
        try:
            value = value.upper()
            self.name, value = value, self._members[value].value
        except (AttributeError, TypeError, KeyError):
            try:
                self.name = next(k for k, v in self._members.items()
                                 if value == v.value)
            except StopIteration:
                raise ValueError("Bad %r value %r" % (self.__class__, value))
        super().__init__(value)

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
### LibSVM stuff

class SVMParams(Structure):
    _fields_ = [
        ('svm_type', c_int),
        ('kernel_type', c_int),
        ('degree', c_int),
        ('gamma', c_int),
        ('coef0', c_int),

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


class SVMTypes(Enum):
    C_SVC = 0
    NU_SVC = 1
    ONE_CLASS = 2
    EPSILON_SVR = 3
    NU_SVR = 4

class SVMKernelTypes(Enum):
    LINEAR = 0
    POLY = 1
    RBF = 2
    SIGMOID = 3
    PRECOMPUTED = 4

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

################################################################################
### Parameter structures

class FLANNParameters(Structure):
    _fields_ = [
        ('algorithm', c_int),
        ('checks', c_int),
        ('cb_index', c_float),
        ('eps', c_float),
        ('trees', c_int),
        ('leaf_max_size', c_int),
        ('branching', c_int),
        ('iterations', c_int),
        ('centers_init', c_int),
        ('target_precision', c_float),
        ('build_weight', c_float),
        ('memory_weight', c_float),
        ('sample_fraction', c_float),
        ('table_number_', c_uint),
        ('key_size_', c_uint),
        ('multi_probe_level_', c_uint),
        ('log_level', c_int),
        ('random_seed', c_long),
    ]

class DivParams(Structure):
    _fields_ = [
        ('k', c_int),
        ('flann_params', FLANNParameters),
        ('num_threads', c_size_t),
        ('show_progress', c_size_t),
        ('print_progress', CFUNCTYPE(None, c_size_t)),
    ]

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
            c_char_p, c_char_p,
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
