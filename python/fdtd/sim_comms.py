##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: sim_comms.py
#
# Description: Class for source/receiver positions and input signals
#
##############################################################################

import numpy as np
from numpy import array as npa
from voxelizer.cart_grid import CartGrid
from pathlib import Path
from common.timerdict import TimerDict
from common.myfuncs import iceil
import h5py
from scipy.signal import lfilter,bilinear_zpk
from numpy import pi,cos,sin

class SimComms: 
    def __init__(self,save_folder):
        #will read h,xv,yv,zv from h5 data 
        save_folder = Path(save_folder)
        assert save_folder.exists()
        assert save_folder.is_dir()
        h5f = h5py.File(save_folder / Path('sim_consts.h5'),'r')
        self.h       = h5f['h'][()]
        self.Ts      = h5f['Ts'][()]
        self.l2      = h5f['l2'][()]
        self.fcc_flag     = h5f['fcc_flag'][()]
        h5f.close()

        h5f = h5py.File(save_folder / Path('cart_grid.h5'),'r')
        self.xv      = h5f['xv'][()]
        self.yv      = h5f['yv'][()] 
        self.zv      = h5f['zv'][()] 
        h5f.close()

        self.fcc = (self.fcc_flag>0)

        if self.fcc:
            assert (self.xv.size%2)==0
            assert (self.yv.size%2)==0
            assert (self.zv.size%2)==0

        self.save_folder = save_folder
        self._diff = False

    def print(self,fstring):
        print(f'--COMMS: {fstring}')

    def prepare_source_pts(self,Sxyz):
        in_alpha,in_ixyz = self.get_linear_interp_weights(Sxyz)
        self.in_alpha = in_alpha
        self.in_ixyz = in_ixyz

    #a few signals to choose from
    def prepare_source_signals(self,duration,sig_type='impulse'):
        in_alpha = self.in_alpha
        Ts = self.Ts

        Nt = np.int_(np.ceil(duration/Ts))
        in_sigs = np.zeros((in_alpha.size,Nt),dtype=np.float64)
        in_sig = np.zeros((Nt,))

        if sig_type=='impulse': #for RIRs
            in_sig[0] = 1.0
        elif sig_type=='hann10': #for viz
            N = 10
            n = np.arange(N)
            in_sig[:N] = 0.5*(1.0-cos(2*pi*n/N))
        elif sig_type=='hann20': #for viz
            N = 20
            n = np.arange(N)
            in_sig[:N] = 0.5*(1.0-cos(2*pi*n/N))
        elif sig_type=='dhann30': #symmetric, for viz
            N = 30
            n = np.arange(N)
            in_sig[:N] = cos(pi*n/N)*sin(pi*n/N);
        elif sig_type=='hann5ms': #for consistency checking
            N = iceil(5e-3/Ts)
            n = np.arange(N)
            in_sig[:N] = 0.5*(1.0-cos(2*pi*n/N))

        in_sigs = in_alpha[:,None]*in_sig[None,:]

        self.in_sigs = in_sigs
        self._scale_source_signals()

    def _scale_source_signals(self): #scaling for FDTD sim
        l2 = self.l2
        h = self.h
        in_sigs = self.in_sigs
        if self.fcc:
            in_sigs *= 0.5*l2/h #c²Ts²/cell-vol
        else:
            in_sigs *= l2/h #c²Ts²/cell-vol

        self.in_sigs = in_sigs

    def diff_source(self): 
        #differentiate with bilinear transform, to undo after sim
        #required for single precision (safeguard against DC instability)
        in_sigs = self.in_sigs
        Ts = self.Ts
        if self._diff:
            return #do nothing if already applied

        b = 2/Ts*npa([1.0,-1.0]) #don't need Ts scaling but simpler to keep for receiver post-processing
        a = npa([1.0,1.0])
        in_sigs = lfilter(b,a,in_sigs,axis=-1);

        self._diff = True
        self.in_sigs = in_sigs

    def prepare_receiver_pts(self,Rxyz):
        Rxyz = np.atleast_2d(Rxyz)
        #many receivers, can have duplicates
        out_alpha = np.zeros((Rxyz.shape[0],8),np.float64) 
        out_ixyz = np.zeros((Rxyz.shape[0],8),dtype=np.int64) 
        for rr in range(Rxyz.shape[0]):
            out_alpha[rr],out_ixyz[rr] = self.get_linear_interp_weights(Rxyz[rr])

        self.out_alpha = out_alpha
        self.out_ixyz = out_ixyz

    def save(self,save_folder=None,compress=None):
        if save_folder is None:
            save_folder = self.save_folder
            assert save_folder.exists()
            assert save_folder.is_dir()
        else:
            save_folder = Path(save_folder)
            print(f'{save_folder=}')
            if not save_folder.exists():
                save_folder.mkdir(parents=True)
            else:
                assert save_folder.is_dir()

        in_ixyz = self.in_ixyz
        in_alpha = self.in_alpha #don't need this anymore but update just in case
        out_ixyz = self.out_ixyz.flat[:]
        out_alpha = self.out_alpha
        #out_alpha = self.out_alpha.flat[:]
        out_reorder = np.arange(out_ixyz.size) #no sorting here
        in_sigs = self.in_sigs

        if compress is not None:
            kw = {'compression': "gzip", 'compression_opts': compress}
        else:
            kw = {}
        h5f = h5py.File(save_folder / Path('comms_out.h5'),'w')
        h5f.create_dataset('in_ixyz', data=in_ixyz, **kw)
        h5f.create_dataset('out_ixyz', data=out_ixyz, **kw)
        h5f.create_dataset('out_alpha', data=out_alpha, **kw)
        h5f.create_dataset('out_reorder', data=out_reorder, **kw)
        h5f.create_dataset('in_sigs', data=in_sigs, **kw)
        h5f.create_dataset('Ns', data=np.int64(in_ixyz.size))
        h5f.create_dataset('Nr', data=np.int64(out_ixyz.size))
        h5f.create_dataset('Nt', data=np.int64(in_sigs.shape[-1]))
        h5f.create_dataset('diff', data=np.int8(self._diff))
        h5f.close()

        #reattach updated values
        self.out_ixyz = out_ixyz
        self.out_alpha = out_alpha
        self.in_alpha = in_alpha
        self.in_ixyz = in_ixyz
        self.in_sigs = in_sigs

    def get_linear_interp_weights(self,pos_xyz):
        h = self.h
        xv = self.xv
        yv = self.yv
        zv = self.zv
        ix_iy_iz = np.empty(pos_xyz.shape,dtype=np.int64)
        Nx = xv.size
        Ny = yv.size
        Nz = zv.size

        xyzv_list = [xv,yv,zv]

        alpha_xyz = np.zeros((3,))
        for j in [0,1,2]:
            ix_iy_iz[j] = np.flatnonzero(xyzv_list[j]>=pos_xyz[j])[0] #return first item
            alpha_xyz[j] = (xyzv_list[j][ix_iy_iz[j]] - pos_xyz[j])/h

        #look back
        ix_iy_iz8_off = npa([[0,0,0],\
                             [-1,0,0],\
                             [0,-1,0],\
                             [0,0,-1],\
                             [-1,-1,0],\
                             [-1,0,-1],\
                             [0,-1,-1],\
                             [-1,-1,-1]])

        if self.fcc: #adapt to subgrid, with two times grid spacing 
            #note: this much simpler than interpolating within tetra/octa tiling (holes of FCC grid) using barycentric coords (PITA)
            ix_iy_iz8_off *= 2
            if np.mod(np.sum(ix_iy_iz),2) == 1:
                aa = np.argmin(alpha_xyz) #not unique
                ix_iy_iz[aa] += 1
            for j in [0,1,2]:
                alpha_xyz[j] = (xyzv_list[j][ix_iy_iz[j]] - pos_xyz[j])/(2*h)

        alpha8 = np.ones((8,))
        xyz8 = np.zeros((8,3))
        for i in range(8):
            for j in range(3):
                xyz8[i,j] = xyzv_list[j][ix_iy_iz[j]+ix_iy_iz8_off[i,j]]
                if ix_iy_iz8_off[i,j]==0:
                    alpha8[i] *= (1-alpha_xyz[j])
                else:
                    alpha8[i] *= alpha_xyz[j]

        assert np.allclose(np.sum(alpha8),1)
        assert np.allclose(np.sum(alpha8*xyz8.T,-1),pos_xyz)

        ix_iy_iz8 = ix_iy_iz + ix_iy_iz8_off
        ixyz8 = ix_iy_iz8 @ npa([Nz*Ny,Nz,1])

        if self.fcc:
            assert np.all(np.mod(np.sum(ix_iy_iz8,axis=-1),2)==0)

        return alpha8,ixyz8

    def check_for_clashes(self,bn_ixyz):
        #scheme implementation designed to only have source/receiver in 'regular' air nodes
        def _check_for_clashes(_ixyz,bn_ixyz):
            ixyz = np.unique(_ixyz) #can have duplicates in receivers
            #could speed this up with a numba routine (don't need to know clashes, just say yes or no)
            assert np.union1d(ixyz.flat[:],bn_ixyz).size == ixyz.size + bn_ixyz.size
            self.print(f'intersection with boundaries: passed')

        timer = TimerDict()
        timer.tic('check in_xyz')
        self.print(f'boundary intersection check with in_ixyz..')
        _check_for_clashes(self.in_ixyz,bn_ixyz)
        self.print(timer.ftoc('check in_xyz'))
        timer.tic('check in_xyz')
        self.print(f'boundary intersection check with out_ixyz..')
        _check_for_clashes(self.out_ixyz,bn_ixyz)
        self.print(timer.ftoc('check in_xyz'))

