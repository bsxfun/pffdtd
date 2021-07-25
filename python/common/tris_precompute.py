##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: tris_precompute.py
#
# Description: Use numpy structured arrays to precompute a bunch of stuff for arrays of triangles
# Used in triangle intersection routines 
#
##############################################################################

import numpy as np
from numpy import array as npa
from common.myfuncs import normalise,dot2,vecnorm

#N.B. careful with slicing tris_pre[idx]['var'] (bad) vs tris_pre['var'][idx] (good)
def tris_precompute(pts=None,tris=None):
    assert tris is not None
    assert pts is not None

    Ntris = tris.shape[0]

    #points of triangle
    a = pts[tris[:,0],:]
    b = pts[tris[:,1],:]
    c = pts[tris[:,2],:]

    assert a.ndim == 2

    #edge vectors
    ab = b - a
    bc = c - b 
    ca = a - c

    #centroid
    cent = (a + b + c)/3.0

    #normal (scaled by area)
    nor = (np.cross(ab,-ca) + np.cross(bc,-ab) + np.cross(ca,-bc))/3.0

    #area
    area = 0.5*vecnorm(nor)

    #outward edge vectors
    eab_unor = normalise(np.cross(ab,nor))
    ebc_unor = normalise(np.cross(bc,nor))
    eca_unor = normalise(np.cross(ca,nor))

    #unit normal
    unor = normalise(nor)

    #lengths
    l2ab = dot2(ab)
    l2bc = dot2(bc)
    l2ca = dot2(ca)

    #bbox
    bmin = np.min(np.stack([a,b,c],axis=2),axis=2)
    bmax = np.max(np.stack([a,b,c],axis=2),axis=2)
    assert np.all(a>=bmin)
    assert np.all(b>=bmin)
    assert np.all(c>=bmin)
    assert np.all(a<=bmax)
    assert np.all(b<=bmax)
    assert np.all(c<=bmax)
    assert np.all(bmin<=bmax)

    custom_dtype = [\
        #('a',np.float64,(3,)),\
        #('b',np.float64,(3,)),\
        #('c',np.float64,(3,)),\
        ('v',np.float64,(3,3)),\
        ('ab',np.float64,(3,)),\
        ('bc',np.float64,(3,)),\
        ('ca',np.float64,(3,)),\
        ('nor',np.float64,(3,)),\
        ('unor',np.float64,(3,)),\
        ('eab_unor',np.float64,(3,)),\
        ('ebc_unor',np.float64,(3,)),\
        ('eca_unor',np.float64,(3,)),\
        ('cent',np.float64,(3,)),\
        ('bmin',np.float64,(3,)),\
        ('bmax',np.float64,(3,)),\
        ('l2ab',np.float64),\
        ('l2bc',np.float64),\
        ('l2ca',np.float64),\
        ('area',np.float64),\
    ]

    tris_pre = np.zeros(Ntris,dtype=custom_dtype)

    tris_pre['v'] = np.concatenate((a[:,None,:],b[:,None,:],c[:,None,:]),axis=1)
    #can read with
    #a = tris_pre[ti]['v'][0]
    #b = tris_pre[ti]['v'][1]
    #c = tris_pre[ti]['v'][2]
    #or 
    #a = tris_pre['v'][:,0,:]
    #b = tris_pre['v'][:,1,:]
    #c = tris_pre['v'][:,2,:]

    tris_pre['ab'] = ab
    tris_pre['bc'] = bc
    tris_pre['ca'] = ca
    tris_pre['nor'] = nor
    tris_pre['unor'] = unor
    tris_pre['eab_unor'] = eab_unor
    tris_pre['ebc_unor'] = ebc_unor
    tris_pre['eca_unor'] = eca_unor
    tris_pre['cent'] = cent
    tris_pre['bmin'] = bmin
    tris_pre['bmax'] = bmax
    tris_pre['l2ab'] = l2ab
    tris_pre['l2bc'] = l2bc
    tris_pre['l2ca'] = l2ca
    tris_pre['area'] = area

    return tris_pre

