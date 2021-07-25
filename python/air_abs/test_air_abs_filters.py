##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: test_air_abs_filters.py
#
# Description: This is a testbench for different air absorption filters
#
##############################################################################

import numpy as np
import numba as nb
from numpy import array as npa
from numpy import exp, sqrt, log, pi
import matplotlib.pyplot as plt
from air_abs.get_air_absorption import get_air_absorption
from air_abs.visco_filter import apply_visco_filter
from air_abs.modal_filter import apply_modal_filter
from air_abs.ola_filter import apply_ola_filter
from numpy.random import random_sample
from common.myfuncs import iround, iceil

Tc = 20
rh = 60

SR = 48e3

#single channel test
t_end = 1
Nt0 = iround(t_end*SR)
tx = np.arange(Nt0)/SR
td = 0.02
tau = t_end/(6*log(10))
x = exp(-(tx-td)/tau)*(2*random_sample(Nt0)-1)
x[:int(td*SR)]=0

#multi channel test
#x = np.zeros((2,iround(1.2*SR)))
#nstart = iround(0.01*SR)
#nskip = iround(0.005*SR)
#x[0,nstart::nskip] = 1.0
#x[1,nstart::nskip] = -1.0
#tx = np.arange(x.shape[-1])/SR

y1 = apply_visco_filter(x, SR, Tc, rh)
y2 = apply_modal_filter(x, SR, Tc, rh)
y3 = apply_ola_filter(x, SR, Tc, rh)

ty1 = np.arange(0,y1.shape[-1])/SR
ty2 = np.arange(0,y2.shape[-1])/SR
ty3 = np.arange(0,y3.shape[-1])/SR

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(tx,x.T,linestyle='-',color='b',label='orig') 
ax.plot(ty1,y1.T,linestyle='-',color='g',label='stokes') 
ax.plot(ty2,y2.T,linestyle='-',color='r',label='modal') 
ax.plot(ty3,y3.T,linestyle='-',color='y',label='OLA') 
ax.margins(0, 0.1)
ax.grid(which='both', axis='both')
ax.legend()
plt.show()


#multi channel test
SR = 48e3

