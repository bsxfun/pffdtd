##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: sim_consts.py
#
# Description: Class to keep simulation constants mostly in one place, writes to HDF5
#
##############################################################################

import numpy as np
from numpy import array as npa
import h5py
from pathlib import Path
class SimConsts:
    def __init__(self,Tc,rh,h=None,SR=None,fmax=None,PPW=None,fcc=False):
        #Tc is temperature, rh is relative humidity <- this gives c (speed of sound)
        assert Tc >= -20
        assert Tc <= 50
        assert rh <= 100
        assert rh >= 10
        c = 343.2*np.sqrt(Tc/20)

        assert (h is not None) or (SR is not None) or (fmax is not None and PPW is not None)
        if fcc:
            l2 = 1.0
            l = np.sqrt(l2)
            assert l<=1.0 #of course true
        else:
            l2 = 1/3
            l = np.sqrt(l2)
            assert l<=np.sqrt(1/3) #check with round-off errors

        #back off to remove nyquist mode
        l *= 0.999 
        l2 = l*l

        if h is not None:
            Ts = h/c*l
            SR = 1/Ts
        elif SR is not None:
            Ts = 1/SR
            h = c*Ts/l
        elif fmax is not None and PPW is not None:
            h = c/(fmax*PPW) #PPW is points per wavelength (on Cartesian grid)
            Ts = h/c*l
            SR = 1/Ts
        else:
            raise

        self.print(f'{c=}')
        self.print(f'{Ts=}')
        self.print(f'{SR=}')
        self.print(f'{h=}')
        self.print(f'{l=}')
        self.print(f'{l2=}')

        self.h = h
        self.c = c
        self.Ts = Ts
        self.SR = SR
        self.l = l
        self.l2 = l2
        self.fcc = fcc

        self.Tc = Tc
        self.rh = rh

    def print(self,fstring):
        print(f'--CONSTS: {fstring}')

    #save to HDF5 file
    def save(self,save_folder):
        c = self.c
        h = self.h
        Ts = self.Ts
        l = self.l
        l2 = self.l2
        SR = self.SR
        fcc = self.fcc
        Tc = self.Tc
        rh = self.rh

        save_folder = Path(save_folder)
        self.print(f'{save_folder=}')
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
        else:
            assert save_folder.is_dir()

        h5f = h5py.File(save_folder / Path('sim_consts.h5'),'w')
        h5f.create_dataset('c', data=np.float64(c))
        h5f.create_dataset('h', data=np.float64(h))
        h5f.create_dataset('Ts', data=np.float64(Ts))
        h5f.create_dataset('SR', data=np.float64(SR))
        h5f.create_dataset('l', data=np.float64(l))
        h5f.create_dataset('l2', data=np.float64(l2))
        h5f.create_dataset('fcc_flag', data=np.int8(fcc))
        h5f.create_dataset('Tc', data=np.float64(Tc))
        h5f.create_dataset('rh', data=np.float64(rh))

        h5f.close()
