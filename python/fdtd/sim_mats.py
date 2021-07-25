##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: sim_mats.py
#
# Description: Class to pack up materials from individual HDF5 files
# in ordering coresponding to room_geo materials
#
##############################################################################

import numpy as np
from numpy import array as npa
from voxelizer.cart_grid import CartGrid
from pathlib import Path
from common.timerdict import TimerDict
import h5py

class SimMats: 
    def __init__(self,save_folder):
        save_folder = Path(save_folder)
        assert save_folder.exists()
        assert save_folder.is_dir()

        self.save_folder = save_folder

    def print(self,fstring):
        print(f'--MATS: {fstring}')

    def package(self,mat_files_dict,mat_list,read_folder): 
        mat_list=mat_list[:] #make copy of input
        if '_RIGID' in mat_list:
            mat_list.remove('_RIGID')
        mat_list.sort()
        mat_list2 = list(mat_files_dict.keys())
        mat_list2.sort()
        assert mat_list==mat_list2 #mat dict coming in has to match list from room_geo

        save_folder = self.save_folder
        read_folder = Path(read_folder)
        DEF_list = []
        for mat in mat_list:
            h5f = h5py.File(Path(read_folder / Path(mat_files_dict[mat])),'r')
            DEF_list.append( h5f['DEF'][()] )
            h5f.close()

        Nmat = len(DEF_list)
        Mb = np.zeros((Nmat,),dtype=np.int8) #number of circuit branches 
        h5f = h5py.File(Path(save_folder / Path('sim_mats.h5')) ,'w')
        h5f.create_dataset('Nmat', data=np.int8(Nmat))
        for i in range(Nmat):
            mat=mat_list[i]
            DEF = DEF_list[i]
            assert DEF.ndim==2
            assert DEF.shape[1]==3

            print(f'{mat=} {DEF=}')
            h5f.create_dataset(f'mat_{i:02d}_DEF', data=DEF)
            Mb[i] = DEF.shape[0]

        h5f.create_dataset('Mb', data=Mb)
        h5f.close()
