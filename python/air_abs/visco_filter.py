##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: visco_filter.py
#
# Description: This is an implementation of an air absorption filter based on
# approximate Green's function Stoke's equation (viscothermal wave equation)
#
# See paper for details:
# Hamilton, B. "Air absorption filtering method based on approximate Green's
# function for Stokes' equation", to be presented at the DAFx2021 e-conference.
#
##############################################################################

import numpy as np
import numba as nb
from numpy import array as npa
from numpy import exp, sqrt, log, pi
from common.myfuncs import iround, iceil
from air_abs.get_air_absorption import get_air_absorption
from tqdm import tqdm

#function to apply filter, main input being x, np.ndarray (Nchannels,Nsamples)
#enter temperature (Tc) and relative humidity (rh)
#NdB should be above 60dB, that is for truncation of Gaussian kernel
def apply_visco_filter(x,Fs,Tc,rh,NdB=120,t_start=None):
    rd = get_air_absorption(1,Tc,rh)
    c = rd['c']
    g = rd['gamma_p']

    Ts = 1/Fs
    if t_start is None:
       t_start = Ts**2/(2*pi*g) 
       print(f'{t_start=}')

    x = np.atleast_2d(x)
    Nt0 = x.shape[-1]

    n_last = Nt0-1
    dt_end = Fs*sqrt(0.1*log(10)*NdB*n_last*Ts*g)
    Nt = Nt0+iceil(dt_end)

    y = np.zeros((x.shape[0],Nt))
    n_start = iceil(t_start*Fs)
    assert(n_start>0)

    y[:,:n_start] =  x[:,:n_start]
    Tsg2 = 2*Ts*g
    Tsg2pi = 2*Ts*g*pi
    gTs = g*Ts
    dt_fac = 0.1*log(10)*NdB*gTs
    pbar = tqdm(total=Nt,desc=f'visco filter',ascii=True)
    for n in range(n_start,Nt0):
        dt = sqrt(dt_fac*n)/Ts
        dt_int = iceil(dt)
        nv = np.arange(n-dt_int,n+dt_int+1)
        assert n>=dt_int
        y[:,nv] += (Ts/sqrt(n*Tsg2pi))*x[:,n][:,None] * exp(-((n-nv)*Ts)**2/(n*Tsg2))[None,:]
        pbar.update(1)
    pbar.close()

    return np.squeeze(y) #squeeze to 1d in case
