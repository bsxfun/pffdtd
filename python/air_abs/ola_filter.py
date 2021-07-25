##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: ola_filter.py
#
# Description: This is an implementation of overlap-add (STFT/iSTFT) air 
# absorption filtering.  Tuned for 75% overlap and 1024-sample Hann window at
# 48kHz. 
#
# Used in paper:
# Hamilton, B. "Adding air attenuation to simulated room impulse responses: A
# modal approach", to be presented at I3DA 2021 conference in Bologna Italy.
#
##############################################################################

import numpy as np
import numba as nb
from numpy import array as npa
from numpy import exp, sqrt, log2, pi, ceil, cos
from scipy.fft import rfft,irfft 
from air_abs.get_air_absorption import get_air_absorption
from common.myfuncs import iround, iceil
from tqdm import tqdm

#apply the filter, x is (Nchannels,Nsamples) array
#Tc is temperature degrees Celsius
#rh is relative humidity
def apply_ola_filter(x,Fs,Tc,rh,Nw=1024):
    Ts = 1/Fs

    x = np.atleast_2d(x)
    Nt0 = x.shape[-1]

    OLF = 0.75
    Ha = iround(Nw*(1-OLF))
    Nfft = np.int_(2**ceil(log2(Nw)))
    NF = iceil((Nt0+Nw)/Ha)
    Np = (NF-1)*Ha-Nt0
    assert Np >= Nw-Ha
    assert Np < Nw
    Nfft_h = np.int_(Nfft//2+1)

    xp = np.zeros((x.shape[0],Nw+Nt0+Np))
    xp[:,Nw:Nw+Nt0] = x
    y = np.zeros((x.shape[0],Nt0+Np))
    del x

    wa = 0.5*(1-cos(2*pi*np.arange(Nw)/Nw)) #hann window
    ws = wa/(3/8*Nw/Ha) #scaled for COLA 

    fv = np.arange(Nfft_h)/Nfft*Fs
    rd = get_air_absorption(fv,Tc,rh)
    c = rd['c']
    absNp = rd['absfull_Np']

    for i in range(xp.shape[0]):
        pbar = tqdm(total=NF,desc=f'OLA filter channel {i}',ascii=True)
        yp = np.zeros((xp.shape[-1],))
        for m in range(NF):
            na0 = m*Ha
            assert na0+Nw<=Nw+Nt0+Np
            dist = c*Ts*(na0-Nw/2)
            xf = xp[i,na0:na0+Nw]
            if dist<0: #dont apply gain (negative times - pre-padding)
                yp[na0:na0+Nw] += ws*xf
            else:
                Yf = rfft(wa*xf,Nfft)*exp(-absNp*dist)
                yf = irfft(Yf,Nfft)[:Nw]
                yp[na0:na0+Nw] += ws*yf

            pbar.update(1)
        y[i] = yp[Nw:]
        pbar.close()
    return np.squeeze(y) #squeeze to 1d in case

