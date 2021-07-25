##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: myasserts.py
#
# Description: Miscellaneous python/numpy assertions (mostly vestigial)
#
##############################################################################

import numpy as np
def assert_np_array_float(x):
    assert isinstance(x,np.ndarray)
    assert x.dtype in [np.dtype('float32'),np.dtype('float64')]

def assert_np_array_complex(x):
    assert isinstance(x,np.ndarray)
    #allow float or complex
    assert x.dtype in [np.dtype('complex64'),np.dtype('complex128'),np.dtype('float32'),np.dtype('float64')]

#python 'int' is arbitrary size
def assert_is_int(x):
    assert isinstance(x,(int,np.integer))
