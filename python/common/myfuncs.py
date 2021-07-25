##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: myfuncs.py
#
# Description: Miscellaneous python/numpy functions.  Not all used or useful.
#
##############################################################################

import numpy as np
from numpy import array as npa
import numpy.linalg as npl
from numpy import cos,sin,pi
from typing import Any,Optional
from common.myasserts import assert_np_array_float, assert_np_array_complex
from pathlib import Path
import struct
import os
import shutil
import hashlib
import multiprocessing as mp
import scipy.io.wavfile

EPS = np.finfo(np.float).eps

def rotmatrix_ax_ang(Rax:Any, Rang:float):
    assert isinstance(Rax,np.ndarray)
    assert Rax.shape == (3,)
    assert type(Rang) is float

    Rax = Rax/npl.norm(Rax) #to be sure

    theta = Rang/180.0*pi #in rad
    #see https://en.wikipedia.org/wiki/Rotation_matrix
    R = np.array([[cos(theta) + Rax[0]*Rax[0]*(1-cos(theta)), Rax[0]*Rax[1]*(1-cos(theta)) - Rax[2]*sin(theta), Rax[0]*Rax[2]*(1-cos(theta)) + Rax[1]*sin(theta)],\
                  [Rax[1]*Rax[0]*(1-cos(theta)) + Rax[2]*sin(theta), cos(theta) + Rax[1]*Rax[1]*(1-cos(theta)), Rax[1]*Rax[2]*(1-cos(theta)) - Rax[0]*sin(theta)],\
                  [Rax[2]*Rax[0]*(1-cos(theta)) - Rax[1]*sin(theta), Rax[2]*Rax[1]*(1-cos(theta)) + Rax[0]*sin(theta), cos(theta) + Rax[2]*Rax[2]*(1-cos(theta))]])
    assert npl.norm(npl.inv(R)-R.T) < 1e-8
    return R

def rotate_xyz_deg(thx_d,thy_d,thz_d):
    #R applies Rz then Ry then Rx (opposite to wikipedia)
    #rotations about x,y,z axes, right hand rule

    thx = np.deg2rad(thx_d)
    thy = np.deg2rad(thy_d)
    thz = np.deg2rad(thz_d)

    Rx = npa([[1,0,0],\
              [0,np.cos(thx),-np.sin(thx)],\
              [0,np.sin(thx),np.cos(thx)]])

    Ry = npa([[np.cos(thy),0,np.sin(thy)],\
             [0,1,0],\
             [-np.sin(thy),0,np.cos(thy)]])

    Rz = npa([[np.cos(thz),-np.sin(thz),0],\
             [np.sin(thz),np.cos(thz),0],\
             [0,0,1]])
   
    R = Rx @ Ry @ Rz 
   
    return R,Rx,Ry,Rz


def rotate_az_el_deg(az_d,el_d):
    #R applies Rel then Raz
    #Rel is rotation about negative y-axis, right hand rule
    #Raz is rotation z-axis, right hand rule
    #this uses matlab convention
    #note, this isn't the most general rotation
    _,_,Ry,Rz = rotate_xyz_deg(0,-el_d,az_d)
    Rel = Ry
    Raz = Rz
    R = Raz @ Rel

    return R,Raz,Rel 

#only used in box
def mydefault(a:Any, default:Any):
    if a is None:
        a = default
    return a

#seconds to day/hour/min/s
def s2dhms(time:int):
    #give time in seconds
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return (day, hour, minutes, seconds)

#misc numpy functions
def clamp(x:Any, xmin:Any, xmax:Any):
    assert isinstance(x,(np.ndarray,int,np.integer,float))
    assert isinstance(xmin,(np.ndarray,int,np.integer,float))
    assert isinstance(xmax,(np.ndarray,int,np.integer,float))
    return np.where(x<xmin,xmin,np.where(x>xmax,xmax,x)) 

def vclamp(x, xmin, xmax):
    return np.where(x<xmin,xmin,np.where(x>xmax,xmax,x)) #only this works for vectorized xmin,xmax

def dot2(v):
    return dotv(v,v)

def dotv(v1,v2):
    return np.sum(v1*v2,axis=-1)

def vecnorm(v1):
    return np.sqrt(dot2(v1))

def vecnorm2(v1):
    return np.sum(v1**2,axis=-1)

def normalise(v1,eps=EPS):
    return (v1.T/(vecnorm(v1)+eps)).T

def ceilint(x:float):
    assert isinstance(x,float)
    return np.int_(np.ceil(x))

def roundint(x:float):
    assert isinstance(x,float)
    return np.int_(np.round(x))

def floorint(x:float):
    assert isinstance(x,float)
    return np.int_(np.floor(x))

def maxabs(x:Any):
    assert_np_array_complex(x)
    if x.ndim==2:
        return np.amax(np.abs(x),axis=-1)[:,None]
    else:
        return np.amax(np.abs(x))

def to_col_2d(x:Any):
    assert x.ndim<3
    x = x.reshape(x.shape[-1],-1)
    assert x.shape[0]>x.shape[1]
    return x

def to_row_2d(x:Any):
    assert x.ndim<3
    x = x.reshape(-1,x.shape[-1])
    assert x.shape[1]>x.shape[0]
    return x

def ind2sub3d(ii,Nx,Ny,Nz):
    iz = ii % Nz
    iy = (ii - iz)//Nz % Ny
    ix = ((ii - iz)//Nz-iy)//Ny
    return ix,iy,iz

def rel_diff(x0,x1): #relative difference at machine epsilon level
    return (x0-x1)/(2.0**np.floor(np.log2(x0)))

def get_default_nprocs():
    return max(1,int(0.8*mp.cpu_count()))

#taken from SO, for argparse bool problem
def str2bool(v:str):
    assert isinstance(v, str)
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def clear_dat_folder(dat_folder_str=None):
    #clear dat folder
    dat_path = Path(dat_folder_str)
    if not dat_path.exists():
        dat_path.mkdir(parents=True)
    else:
        assert dat_path.is_dir()

    assert dat_path.exists()
    assert dat_path.is_dir()
    for f in dat_path.glob('*'):
        try:
            f.unlink()
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def clear_folder(folder_str):
    print(f'clearing {folder_str}...')
    os.system('rm -rf ' + str(Path(folder_str)) + '/*')
    print(f'{folder_str} cleared')

def double_to_hex(d):
    return hex(struct.unpack('<Q', struct.pack('<d', d))[0])

def hex_to_double(s):
    s_list = list(s)
    assert s_list[0]=='0'
    assert s_list[1]=='x'
    return struct.unpack('!d', bytes.fromhex(s.split('0x')[1]))[0]

def clear_console(lines):
    _, columns = os.popen('stty size', 'r').read().split()
    for n in range(lines):
        print(' ' * int(columns))
    for n in range(lines):
        print("\033[F",end='')

def read_txt_values(filepath_str):
    assert Path(filepath_str).exists()
    rdict={}
    print(f'reading {filepath_str}...')
    with open(Path(filepath_str)) as tfile:
        lines = tfile.readlines()
        keyvals = [line.rstrip('\n').split(' ') for line in lines]
        for keyval in keyvals:
            assert len(keyval)==2
            key = keyval[0] 
            val = keyval[1] #still a string
            rdict[key] = val
            print(f'key = {key}, val = {val}')
    return rdict
    
def gen_md5hash(*args):
    hash_list = []
    out_hash = ''
    for arg in args:
        if isinstance(arg,np.ndarray):
            tmp_hash = hashlib.md5(arg.tobytes()).hexdigest()
        elif isinstance(arg,str):
            tmp_hash = hashlib.md5(arg.encode()).hexdigest()
        else:
            tmp_hash = hashlib.md5(str(arg).encode()).hexdigest()
        out_hash = hashlib.md5((tmp_hash + out_hash).encode()).hexdigest()

    return out_hash

def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False

def iceil(x):
    return np.int_(np.ceil(x))

def iround(x):
    return np.int_(np.round(x))

def wavread(fname):
    SR,data = scipy.io.wavfile.read(fname) #reads in (Nsamples,Nchannels)
    if data.dtype == np.int16:
        data = data/32768.0
        SR = np.float_(SR)
    return SR,np.float_(data.T)

def wavwrite(fname,SR,data):
    data = np.atleast_2d(data) #expects (Nchannels,Nsamples), this will also assert that
    scipy.io.wavfile.write(fname,int(SR),np.float32(data.T)) #reads in (Nsamples,Nchannels)
    print(f'wrote {fname} at SR={SR/1000:.2f} kHz')
