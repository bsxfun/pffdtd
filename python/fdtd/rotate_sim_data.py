##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: rotate_sim_data.py
#
# Description: For multi-GPU execution:
#       -best to permute dimensions for descending order (last dim continguous)
#       -indices all need to be sorted (and corresponding data reordered)
#       -fold FCC subgrid onto itself here (fills half Cartesian grid)
#
##############################################################################

import numpy as np
from numpy import array as npa
import numba as nb
from pathlib import Path
import time
import h5py
from common.myfuncs import ind2sub3d
from common.timerdict import TimerDict
import shutil

#NB: we keep cart_grid.h5 untouched and that has original Nx,Ny,Nz if needed

def rotate_sim_data(data_dir,tr=None,compress=False):
    def _print(fstring):
        print(f'--ROTATE_DATA: {fstring}')
    timer = TimerDict()
    data_dir = Path(data_dir)

    timer.tic('read')
    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r')
    Nx      = h5f['Nx'][()] 
    Ny      = h5f['Ny'][()]
    Nz      = h5f['Nz'][()]
    h5f.close()
    if tr is None:
        tr = np.argsort(npa([Nx,Ny,Nz]))[::-1] #descending (Nx is non-contiguous -- want Ny*Nz min)
    else:
        assert np.all(np.sort(tr)==npa([0,1,2]))
    _print(f'{tr=}')
    if np.all(tr==npa([0,1,2])):
        _print('no rotate')
        _print(timer.ftoc('read'))
        return #no op

    #read
    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r')
    xv      = h5f['xv'][()] 
    yv      = h5f['yv'][()] 
    zv      = h5f['zv'][()] 
    adj_bn  = h5f['adj_bn'][...]
    bn_ixyz = h5f['bn_ixyz'][...]
    h5f.close()

    NN = adj_bn.shape[1] 
    if NN==6:
        iVV = npa([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    else:
        iVV = npa([[+1,+1,0],[-1,-1,0],[0,+1,+1],[0,-1,-1],[+1,0,+1],[-1,0,-1], \
                   [+1,-1,0],[-1,+1,0],[0,+1,-1],[0,-1,+1],[+1,0,-1],[-1,0,+1]])

    h5f = h5py.File(data_dir / Path('comms_out.h5'),'r')
    in_ixyz      = h5f['in_ixyz'][...]
    out_ixyz     = h5f['out_ixyz'][...]
    Nr      = h5f['Nr'][()]
    Ns      = h5f['Ns'][()]
    Nt      = h5f['Nt'][()]
    h5f.close()
    _print(timer.ftoc('read'))

    assert in_ixyz.shape[0]==Ns
    assert out_ixyz.shape[0]==Nr

    timer.tic('transpose')
    #swap and reorder
    def _swap3(a,b,c,tr):
        abcl = [a,b,c]
        return [abcl[i] for i in tr] #swap with order

    Nxt,Nyt,Nzt = _swap3(Nx,Ny,Nz,tr) 
    bn_ixyzt = npa(_swap3(*ind2sub3d(bn_ixyz,Nx,Ny,Nz),tr)).T @ npa([Nzt*Nyt,Nzt,1])
    in_ixyzt = npa(_swap3(*ind2sub3d(in_ixyz,Nx,Ny,Nz),tr)).T @ npa([Nzt*Nyt,Nzt,1])
    out_ixyzt = npa(_swap3(*ind2sub3d(out_ixyz,Nx,Ny,Nz),tr)).T @ npa([Nzt*Nyt,Nzt,1])
    xvt,yvt,zvt = _swap3(xv,yv,zv,tr) 
    assert xvt.size == Nxt
    assert yvt.size == Nyt
    assert zvt.size == Nzt
    _print(timer.ftoc('transpose'))

    timer.tic('reorder adj')
    #reorder adj_bn columns
    jj = np.zeros((NN,),dtype=np.int_)
    jj = npa([np.flatnonzero(np.all(ivv[tr]==iVV,axis=-1))[0] for ivv in iVV])
    _print(f'{jj=}')
    ia = np.argsort(jj)
    adj_bnt = adj_bn[:,ia]
    timer.toc('reorder adj')

    timer.tic('write')
    if compress:
        kw = {'compression': "gzip", 'compression_opts': 9}
    else:
        kw = {}
    #overwrite
    h5f = h5py.File(data_dir / Path('comms_out.h5'),'r+')
    h5f['in_ixyz'][...] = in_ixyzt
    h5f['out_ixyz'][...] = out_ixyzt
    h5f.close()

    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r+')
    h5f['bn_ixyz'][...] = bn_ixyzt
    h5f['adj_bn'][...] = adj_bnt
    h5f['Nx'][()] = Nxt
    h5f['Ny'][()] = Nyt
    h5f['Nz'][()] = Nzt
    #these take different sizes, have to clobber
    del h5f['xv']
    h5f.create_dataset('xv', data=xvt, **kw)
    del h5f['yv']
    h5f.create_dataset('yv', data=yvt, **kw)
    del h5f['zv']
    h5f.create_dataset('zv', data=zvt, **kw)
    h5f.close()
    _print(timer.ftoc('write'))

def sort_sim_data(data_dir):
    def _print(fstring):
        print(f'--SORT_DATA: {fstring}')
    timer = TimerDict()
    data_dir = Path(data_dir)

    timer.tic('read')
    #read
    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r')
    adj_bn  = h5f['adj_bn'][...]
    bn_ixyz = h5f['bn_ixyz'][...]
    mat_bn  = h5f['mat_bn'][...]
    saf_bn  = h5f['saf_bn'][...]
    h5f.close()

    h5f = h5py.File(data_dir / Path('comms_out.h5'),'r')
    in_ixyz      = h5f['in_ixyz'][...]
    out_ixyz     = h5f['out_ixyz'][...]
    out_alpha    = h5f['out_alpha'][...]
    in_sigs      = h5f['in_sigs'][...]
    h5f.close()
    _print(timer.ftoc('read'))

    timer.tic('reorder')
    #sort rows
    ii = np.argsort(bn_ixyz)
    bn_ixyz = bn_ixyz[ii]
    adj_bn = adj_bn[ii] 
    mat_bn = mat_bn[ii]
    saf_bn = saf_bn[ii]

    ii = np.argsort(in_ixyz)
    in_ixyz = in_ixyz[ii]
    in_sigs = in_sigs[ii]

    ii = np.argsort(out_ixyz)
    out_ixyz = out_ixyz[ii]
    #out_alpha = out_alpha[ii] #will apply to reordered signals
    out_reorder = np.argsort(ii)
    _print(timer.ftoc('reorder'))

    timer.tic('write')
    #overwrite
    h5f = h5py.File(data_dir / Path('comms_out.h5'),'r+')
    h5f['in_ixyz'][...] = in_ixyz
    h5f['in_sigs'][...] = in_sigs
    h5f['out_ixyz'][...] = out_ixyz
    h5f['out_alpha'][...] = out_alpha
    h5f['out_reorder'][...] = out_reorder
    h5f.close()

    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r+')
    h5f['bn_ixyz'][...] = bn_ixyz
    h5f['adj_bn'][...] = adj_bn
    h5f['mat_bn'][...] = mat_bn
    h5f['saf_bn'][...] = saf_bn
    h5f.close()
    _print(timer.ftoc('write'))

def fold_fcc_sim_data(data_dir):
    def _print(fstring):
        print(f'--FOLD_FCC_DATA: {fstring}')

    timer = TimerDict()
    data_dir = Path(data_dir)
    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r')
    Nx      = h5f['Nx'][()] 
    Ny      = h5f['Ny'][()]
    Nz      = h5f['Nz'][()]
    h5f.close()
    assert (Ny%2)==0

    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r')
    adj_bn  = h5f['adj_bn'][...]
    bn_ixyz = h5f['bn_ixyz'][...]
    h5f.close()

    h5f = h5py.File(data_dir / Path('comms_out.h5'),'r')
    in_ixyz      = h5f['in_ixyz'][...]
    out_ixyz     = h5f['out_ixyz'][...]
    h5f.close()

    h5f = h5py.File(data_dir / Path('sim_consts.h5'),'r')
    fcc_flag     = h5f['fcc_flag'][...]
    h5f.close()

    assert fcc_flag==1

    Nyh = np.int_(Ny/2)+1

    bix,biy,biz = ind2sub3d(bn_ixyz,Nx,Ny,Nz)
    ii = (biy>=Ny/2)

    #bn_ixyz
    bn_ixyz[ii] = np.c_[bix[ii],Ny-biy[ii]-1,biz[ii]] @ npa([Nz*Nyh,Nz,1])
    bn_ixyz[~ii] = np.c_[bix[~ii],biy[~ii],biz[~ii]] @ npa([Nz*Nyh,Nz,1])

    adj_bn[ii,0],adj_bn[ii,6] = adj_bn[ii,6],adj_bn[ii,0]
    adj_bn[ii,1],adj_bn[ii,7] = adj_bn[ii,7],adj_bn[ii,1]
    adj_bn[ii,2],adj_bn[ii,9] = adj_bn[ii,9],adj_bn[ii,2]
    adj_bn[ii,3],adj_bn[ii,8] = adj_bn[ii,8],adj_bn[ii,3]

    #in_ixyz
    bix,biy,biz = ind2sub3d(in_ixyz,Nx,Ny,Nz)
    ii = (biy>=Ny/2)
    in_ixyz[ii] = np.c_[bix[ii],Ny-biy[ii]-1,biz[ii]] @ npa([Nz*Nyh,Nz,1])
    in_ixyz[~ii] = np.c_[bix[~ii],biy[~ii],biz[~ii]] @ npa([Nz*Nyh,Nz,1])

    #out_ixyz
    bix,biy,biz = ind2sub3d(out_ixyz,Nx,Ny,Nz)
    ii = (biy>=Ny/2)
    out_ixyz[ii] = np.c_[bix[ii],Ny-biy[ii]-1,biz[ii]] @ npa([Nz*Nyh,Nz,1])
    out_ixyz[~ii] = np.c_[bix[~ii],biy[~ii],biz[~ii]] @ npa([Nz*Nyh,Nz,1])

    timer.tic('write')
    #write
    h5f = h5py.File(data_dir / Path('comms_out.h5'),'r+')
    h5f['in_ixyz'][...] = in_ixyz
    h5f['out_ixyz'][...] = out_ixyz
    h5f.close()

    h5f = h5py.File(data_dir / Path('vox_out.h5'),'r+')
    h5f['bn_ixyz'][...] = bn_ixyz
    h5f['adj_bn'][...] = adj_bn
    h5f['Ny'][()] = Nyh
    h5f.close()

    h5f = h5py.File(data_dir / Path('sim_consts.h5'),'r+')
    h5f['fcc_flag'][()] = 2
    h5f.close()
    _print(timer.ftoc('write'))

def copy_sim_data(src_data_dir,dst_data_dir):
    def _print(fstring):
        print(f'--COPY DATA: {fstring}')
    src_data_dir = Path(src_data_dir)
    _print(f'{src_data_dir=}')
    assert src_data_dir.is_dir()

    dst_data_dir = Path(dst_data_dir)
    _print(f'{dst_data_dir=}')
    if not dst_data_dir.exists():
        dst_data_dir.mkdir(parents=True)
    else:
        assert dst_data_dir.is_dir()
    for file in src_data_dir.glob('*.h5'):
        _print(f'copying {file}')
        shutil.copy(file, dst_data_dir)

#def main():
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir', type=str,help='run directory')
    #parser.set_defaults(data_dir=None)
#
    #args = parser.parse_args()
    #rotate_sim_data(args.data_dir)
#
#if __name__ == '__main__':
    #main()
