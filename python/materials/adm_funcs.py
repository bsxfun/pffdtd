##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: adm_funcs.py
#
# Description: Miscellaneous functions for dealing with wall admittances (materials)
#
# DEF is normalised RLC triplet (i.e., for specific admittance)
##############################################################################

import numpy as np
from numpy import array as npa
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from numpy import sqrt,pi,log10,log
import scipy.optimize as scpo

#normal-incidence absorption to reflection coefficient
def convert_nabs_to_R(nabs):
    nabs = np.float64(nabs)
    assert nabs>=0
    assert nabs<=1
    return np.sqrt(1.0-nabs)

#specific admittance to reflection coefficient
def convert_Yn_to_R(Yn):
    assert np.all(Yn>0.0)
    R = (1.0-Yn)/(1.0+Yn)
    return R

#reflection coefficient to specific admittance
def convert_R_to_Yn(R):
    assert np.all(R<1.0)
    Yn = (1.0-R)/(1.0+R)
    return Yn

#reflection coefficient to specific impedance
def convert_R_to_Zn(R):
    assert np.all(R<1.0)
    Zn = 1.0/convert_R_to_Yn(R)
    return Zn

#sabine absorption to specific admittance, Paris formula inversion with Newton method
def convert_Sabs_to_Yn(Sabs,max_iter=100):
    def _print(fstring):
        print(f'--MATS: {fstring}')
    if Sabs>0.9512:
        _print('warning, Sabs>0.9512 -- not possible for locally-reactive model')
        Sabs = 0.9512
    fg = lambda g: 8.0*g*(1+g/(1+g)-2*g*np.log((g+1)/g))
    fgd = lambda g: -8.0*(-4*g**2 - 6*g +4*(1+g)**2*g*np.log((g+1)/g)-1)/(1+g)**2
    if Sabs==0:
        g=0
    else: #newton solve
        x_old = Sabs/8.0 #starting guess
        x_new = 0
        err = np.inf
        niter=0
        while (niter<max_iter) and (err > 1e-6):
            x_new = x_old - (fg(x_old)-Sabs)/fgd(x_old)
            niter += 1
            err = np.abs(1-x_new/x_old)
            #_print(f'Iteration {niter}: {x_new=}, {err=}')
            x_old = x_new;
        g = x_new
    return g

#write HDF5 mat file from a specific impedance (frequency-independent)
def write_freq_ind_mat_from_Zn(Zn,filename):
    def _print(fstring):
        print(f'--MATS: {fstring}')

    assert ~np.isnan(Zn)
    assert ~np.isinf(Zn) #rigid should be specified in scene (no material)
    assert Zn>=0
    filename = Path(filename)
    h5f = h5py.File(filename,'w')
    DEF = npa([0,Zn,0])
    assert np.all(np.sum(DEF>0,axis=-1)) #at least one non-zero
    _print(f'{DEF=}')
    h5f.create_dataset(f'DEF', data=np.atleast_2d(DEF))
    h5f.close()

#write HDF5 mat file from a specific impedance (frequency-independent)
def write_freq_ind_mat_from_Yn(Yn,filename):
    def _print(fstring):
        print(f'--MATS: {fstring}')

    assert ~np.isnan(Yn)
    assert ~np.isinf(Yn) 
    assert Yn>0 #rigid should be specified in scene (no material)
    write_freq_ind_mat_from_Zn(1/Yn,filename)

#write HDF5 mat file from frequency-independent triplet (D=F=0)
def write_freq_dep_mat(DEF,filename):
    def _print(fstring):
        print(f'--MATS: {fstring}')

    _print(f'{DEF=}')
    DEF = np.atleast_2d(DEF)
    assert ~np.any(np.isinf(DEF)) #rigid should be specified in scene (no material)
    assert ~np.any(np.isnan(DEF)) 
    assert np.all(DEF>=0) #all non-neg
    assert np.all(np.sum(DEF>0,axis=-1)) #at least one non-zero
    assert DEF.shape[1]==3
    filename = Path(filename)
    h5f = h5py.File(filename,'w')
    _print(f'{DEF=}')
    h5f.create_dataset(f'DEF', data=DEF)
    h5f.close()

#write HDF5 mat file from frequency-independent triplet (D=F=0)
def read_mat_DEF(filename):
    h5f = h5py.File(Path(filename),'r')
    DEF = h5f['DEF'][()] 
    h5f.close()
    return DEF

#plot some admittance based on RLC triplets (not specific admittance)
def plot_RLC_admittance(RLC,Y0):
    plot_DEF_admittance(RLC*Y0)

#plot some admittance based on DEF triplets (specific admittance)
def plot_DEF_admittance(fv,DEF,model_Rf=None,show=True):
    from numpy import pi

    DEF = np.atleast_2d(DEF)
    assert DEF.shape[1] == 3
    D,E,F = DEF.T
    jw = 1j*fv*2*pi

    Rf,Yn,Zn_br,Rf_br = compute_Rf_from_DEF(jw,D,E,F)

    if model_Rf is not None:
        model_Yn = convert_R_to_Yn(model_Rf)

    fig = plt.figure()
    #reflection coefficient
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fv,np.abs(Rf),linestyle='-')
    ax.plot(fv,np.abs(Rf_br),linestyle=':')
    if model_Rf is not None:
        ax.plot(fv,np.abs(model_Rf),linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|R|$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(fv,np.angle(Rf),linestyle='-')
    ax.plot(fv,np.angle(Rf_br),linestyle=':')
    if model_Rf is not None:
        ax.plot(fv,np.angle(model_Rf),linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\angle R$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    fig = plt.figure()
    #specific (normalised) admittance (abs/ang)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fv,np.abs(Yn),linestyle='-')
    ax.plot(fv,np.abs(1.0/Zn_br),linestyle=':')
    if model_Rf is not None:
        ax.plot(fv,np.abs(model_Yn),linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$|Y|$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(fv,np.angle(Yn),linestyle='-')
    ax.plot(fv,np.angle(1.0/Zn_br),linestyle=':')
    if model_Rf is not None:
        ax.plot(fv,np.angle(model_Yn),linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\angle Y$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    fig = plt.figure()
    #specific (normalised) admittance (real/imag)
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(fv,np.real(Yn),linestyle='-')
    ax.plot(fv,np.real(1.0/Zn_br),linestyle=':')
    if model_Rf is not None:
        ax.plot(fv,np.real(model_Yn),linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\Re(Y)$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(fv,np.imag(Yn),linestyle='-')
    ax.plot(fv,np.imag(1.0/Zn_br),linestyle=':')
    if model_Rf is not None:
        ax.plot(fv,np.imag(model_Yn),linestyle='--')
    ax.set_xscale('log')
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel(r'$\Im(Y)$')
    ax.margins(0, 0.1)
    ax.grid(which='both', axis='both')

    if show:
        plt.show()

#plot compute reflection coefficient from DEF triplets (specific admittance)
def compute_Rf_from_DEF(jw,D,E,F):
    Zn_br = jw[:,None]*D[None,:] + E + F[None,:]/jw[:,None]
    Yn = np.sum(1.0/Zn_br,axis=-1)
    Rf = (1.0-Yn)/(1.0+Yn)
    Rf_br = (Zn_br-1.0)/(Zn_br+1.0)
    return Rf,Yn,Zn_br,Rf_br

#Ynm is max abs normalised admittance
#dw is half-power bandwidth in radians/s
#w0 is resonant frequency in radians/s
def _to_DEF(Ynm,dw,w0):
    D = 1.0/Ynm/dw
    E = 1.0/Ynm
    F = w0**2/Ynm/dw
    return D,E,F

def _from_DEF(D,E,F):
    Ynm = 1.0/E
    dw = E/D
    w0 = np.sqrt(F/D)
    return Ynm,dw,w0

#fit Yn from 11 Sabine octave-band coefficients (16Hz to 16KHz)
#this is a simple fitting routine that is a starting point for octave-band input data
def fit_to_Sabs_oct_11(Sabs,filename,plot=False):
    assert Sabs.size == 11
    Noct = Sabs.size
    fv = np.logspace(log10(10),log10(20e3),1000) #frequency vector to fit over
    jw = 1j*fv*2*pi
    fcv = 1000*(2.0**np.arange(-6,5))
    ymv = np.zeros(Noct)
    dwv = np.zeros(Noct)
    w0v = np.zeros(Noct)
    Y_target = np.zeros(fv.shape)
    for j in range(Noct):
        fc = fcv[j]
        sa = Sabs[j]
        Ynm = convert_Sabs_to_Yn(sa)
        if j==0:
            i1=0
        else:
            i1 = np.flatnonzero(fv>=fc/sqrt(2))[0]
        if j==Noct-1:
            i2=fv.size
        else:
            i2 = np.flatnonzero(fv>=fc*sqrt(2))[0]

        Y_target[i1:i2] = Ynm 
        w0 = 2*pi*fc
        dw = w0/sqrt(2) #using resonance with half-octave bandwidth
        ymv[j] = Ynm
        dwv[j] = dw
        w0v[j] = w0

    R_target = (1.0-Y_target)/(1.0+Y_target)
    def cost_function_3(x0):
        if np.any(x0<0):
            return np.finfo(np.float64).max #something big to disallow neg values
        else:
            x0 = x0.reshape((-1,3))
            ym = x0[:,0]
            dw = x0[:,1]
            w0 = x0[:,2]

            D,E,F = _to_DEF(ym,dw,w0)

            Rf_opt,Yn_opt,_,_ = compute_Rf_from_DEF(jw,D,E,F)
            assert Yn_opt.ndim==1
            assert Rf_opt.ndim==1
            abs_opt = 1-np.abs(Rf_opt)**2
            abs_target = 1-np.abs(R_target)**2
            cost = np.sum(np.abs(abs_opt-abs_target)) #simply fit to absorption coefficients
            #other possible cost functions
            #cost = np.sum(np.abs(np.abs(Y_target) - np.abs(Yn_opt)))
            #cost = np.sum(np.abs(1-np.abs(Y_target)/np.abs(Yn_opt))) 
            #cost = np.sum(np.abs(Y_target - Yn_opt)) 
            #cost = np.sum(np.abs(R_target - Rf_opt)) 
            return cost

    #limit optimisation to just Yabs (keep bandwidths fix and w0 fixed)
    cost_function = lambda x0: cost_function_3(np.c_[x0,dwv,w0v].flat[:])

    #run optimisation
    x0 = ymv
    initial_cost = cost_function(x0)
    res = scpo.minimize(cost_function,x0,method='Nelder-Mead') #'Powell' works well too
    final_cost = cost_function(res.x)
    assert (final_cost <= initial_cost)
    ymv_opt = res.x
    dwv_opt = dwv
    w0v_opt = w0v

    D,E,F = _to_DEF(ymv_opt,dwv_opt,w0v_opt)
    DEF = np.c_[D,E,F]
    
    #now save
    h5f = h5py.File(filename,'w')
    assert np.all(np.sum(DEF>0,axis=-1)) #at least one non-zero
    print(f'{DEF=}')
    h5f.create_dataset(f'DEF', data=np.atleast_2d(DEF))
    h5f.close()

    if plot:
        plot_DEF_admittance(fv,np.c_[D,E,F],model_Rf = R_target)
