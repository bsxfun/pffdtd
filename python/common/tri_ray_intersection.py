##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: tri_ray_intersection.py
#
# Description: Triangle-ray intersection routines.  
# One single ray / triangle, and one vectorised for one-ray-many-tri or one-tri-many-ray
# some tests (__main__ entry)
# 
# returns boolean for hit. distance is Inf when no hit
#
# N.B. these intersection routines are customised for their purposes in PFFDTD
# you may need to tailor for other uses (e..g, may fail for degenerate
# triangles, pruned first here)
#
##############################################################################

import numpy as np
from numpy import array as npa
from common.myfuncs import dotv,normalise,vecnorm
from common.myasserts import assert_np_array_float
from common.tris_precompute import tris_precompute

#d_eps is a distance eps, cp is for coplanarity (non-dimensional)
#pretty standard test: check if coplanar, then check point-on-plane inside edge functions
def tri_ray_intersection(ray_o,ray_d,tri_pre,d_eps=1e-6,cp_eps=1e-6):
    #returns hit independent of triangle orientation wrt ray

    assert_np_array_float(ray_o)
    assert_np_array_float(ray_d)

    assert ray_o.ndim == 1
    assert ray_d.ndim == 1
    assert ray_o.size == 3
    assert ray_d.size == 3
    d_eps=np.abs(d_eps)
    cp_eps=np.abs(cp_eps)

    ray_un = normalise(ray_d) #normalise in case

    tri_unor = tri_pre['unor']
    tri_cent = tri_pre['cent']
    tri_c = tri_pre['v'][2]
    tri_b = tri_pre['v'][1]
    tri_a = tri_pre['v'][0]

    tinf = np.inf

    #check if coplanar
    beta = np.dot(ray_un,tri_unor)
    #print(f'{beta=}')
    if np.abs(beta)<cp_eps:
        return False,tinf

    #get distance to plane
    t = np.dot(tri_unor,(tri_cent-ray_o))/beta
    #check if triangle behind origin (not implemented but could allow neg distances with False hit)
    if t<0:
        return False,tinf
    #get point on plane
    pop = ray_o + t*ray_un

    #check inside edge vectors
    if np.dot(pop-0.5*(tri_a+tri_b),tri_pre['eab_unor'])>d_eps:
        return False,tinf
    if np.dot(pop-0.5*(tri_b+tri_c),tri_pre['ebc_unor'])>d_eps:
        return False,tinf
    if np.dot(pop-0.5*(tri_c+tri_a),tri_pre['eca_unor'])>d_eps:
        return False,tinf

    return True,t

#one ray / many tris, but also works with one tri / many rays
def tri_ray_intersection_vec(ray_o,ray_d,tris_pre,d_eps=1e-6,cp_eps=1e-6):
    assert_np_array_float(ray_o)
    assert_np_array_float(ray_d)

    assert ray_o.shape[-1] == 3
    assert ray_d.shape[-1] == 3

    d_eps = np.abs(d_eps)
    cp_eps = np.abs(cp_eps)

    ray_un = normalise(ray_d) #normalise in case

    tris_unor = tris_pre['unor']
    tris_cent = tris_pre['cent']
    tris_c = tris_pre['v'][:,2,:]
    tris_b = tris_pre['v'][:,1,:]
    tris_a = tris_pre['v'][:,0,:]

    #check if coplanar
    beta = dotv(ray_un,tris_unor)

    fail = np.abs(beta)<cp_eps 
    beta[fail] = -np.finfo(np.float64).eps #fake value to avoid divide by zero

    t_ret = np.full(beta.size,np.inf)

    #get distance to plane
    t = dotv(tris_unor,(tris_cent-ray_o))/beta
    fail |= t<0

    #get point on plane
    pop = ray_o + ray_un*t[:,None]

    #check inside edge vectors with distance epsilon
    fail |= dotv(pop-0.5*(tris_a+tris_b),tris_pre['eab_unor'])>d_eps
    fail |= dotv(pop-0.5*(tris_b+tris_c),tris_pre['ebc_unor'])>d_eps
    fail |= dotv(pop-0.5*(tris_c+tris_a),tris_pre['eca_unor'])>d_eps

    t_ret[~fail]=t[~fail]

    return ~fail,t_ret

def main():
    #some randomized tests
    import numpy.random as npr
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodraw', action='store_true',help='draw')
    parser.add_argument('--trials', type=int,help='Nvox roughly')
    parser.set_defaults(nodraw=False)
    parser.set_defaults(trials=1)
    args = parser.parse_args()
    print(args)
    draw = not args.nodraw

    if draw:
        from mayavi import mlab
        from tvtk.api import tvtk #only for z-up
        assert args.trials<4

    for tt in range(args.trials):
        ####################
        #one ray many tris
        ####################
        Ntris = 5
        #generate tris
        vv = npr.randn(Ntris,3,3)
        pts = vv.reshape((-1,3))
        tris = np.arange(Ntris*3).reshape(-1,3)

        bmin = np.amin(pts,axis=0)
        bmax = np.amax(pts,axis=0)
        scale = vecnorm(bmax-bmin)
        
        tris_pre = tris_precompute(pts=pts,tris=tris)

        #ray direction and origin (make coming from outside and pointing in)
        ro = normalise(npr.randn(3))*scale
        rd = normalise(np.mean(pts,axis=0)-ro)

        swap = npr.randint(0,2)
        if swap==1: #origin inside triangle cluster and pointing outwards
            ro,rd = rd,normalise(ro)

        hit = np.full(Ntris,False)
        dist = np.full(Ntris,np.nan)
        for ti in range(0,Ntris):
            hit[ti],dist[ti] = tri_ray_intersection(ro,rd,tris_pre[ti])
            
        hit2,dist2 = tri_ray_intersection_vec(ro,rd,tris_pre)
        print(f'{dist=}')

        assert np.all(hit==hit2),'mismatch!'
        assert np.all(dist==dist2),'mismatch!' #maybe need np.allclose

        #print(scale)
        if draw:
            fig = mlab.figure()
            mlab.quiver3d(*ro, *rd,color=(0,1,0))
            if swap:
                mlab.plot3d(*np.c_[ro,ro+rd*0.5*scale],color=(0,1,0))
            else:
                mlab.plot3d(*np.c_[ro,ro+rd*2*scale],color=(0,1,0))
            mlab.points3d(*ro,mode='cube',scale_factor=scale/100.0,scale_mode='vector')
            for ti in range(0,Ntris):
                trip = tris_pre[ti]
                mlab.plot3d(*np.c_[trip['cent'],trip['cent']+trip['unor']*np.sqrt(trip['area'])],color=(0,0,0))
                if hit[ti]:
                    color = (1,0,0)
                    mlab.points3d(*(ro+rd*dist[ti]),mode='sphere',scale_factor=scale/50.0,scale_mode='vector',color=(0,0,1))
                else:
                    color = (1,1,1)
                mlab.triangular_mesh(*(pts.T),tris[ti][None,:],opacity=1,color=color)

            mlab.orientation_axes()
            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        ####################
        #one tri many rays
        ####################
        Nrays = 5
        #generate tris
        pts = npr.randn(3,3)
        tri = np.arange(3)

        bmin = np.amin(pts,axis=0)
        bmax = np.amax(pts,axis=0)
        scale = vecnorm(bmax-bmin)

        
        tri_pre = tris_precompute(pts=pts,tris=npa([tri]))

        ro = normalise(npr.randn(Nrays,3))*scale
        rd = normalise(npr.random((Nrays,3))*(bmax-bmin)+bmin - ro)

        swap = npr.randint(0,2)
        if swap==1:
            ro,rd = rd,normalise(ro)

        hit = np.full(Nrays,False)
        dist = np.full(Nrays,np.nan)
        for ri in range(0,Nrays):
            hit[ri],dist[ri] = tri_ray_intersection(ro[ri],rd[ri],tri_pre[0])

        print(f'{dist=}')
            
        hit2,dist2 = tri_ray_intersection_vec(ro,rd,tri_pre)

        assert np.all(hit==hit2),'mismatch!'
        assert np.all(dist==dist2),'mismatch!' #maybe need np.allclose

        if draw:
            fig = mlab.figure()
            mlab.triangular_mesh(*(pts.T),tri[None,:],opacity=1,color=(1,1,1))
            #mlab.points3d(*(pts.T),mode='sphere',resolution=3,opacity=1,color=(1,1,1),scale_factor=scale/50.0)
            for ri in range(0,Nrays):
                if hit[ri]:
                    color = (1,0,0)
                    mlab.points3d(*(ro[ri]+rd[ri]*dist[ri]),mode='sphere',scale_factor=scale/25.0,scale_mode='vector',color=(0,0,1))
                else:
                    color = (1,1,1)
                mlab.points3d(*ro[ri],mode='cube',scale_factor=scale/20.0,scale_mode='vector')
                mlab.quiver3d(*ro[ri], *rd[ri],color=color)
                if swap:
                    mlab.plot3d(*np.c_[ro[ri],ro[ri]+rd[ri]*scale],color=color)
                else:
                    mlab.plot3d(*np.c_[ro[ri],ro[ri]+rd[ri]*4*scale],color=color)

            mlab.orientation_axes()
            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
    if draw:
        mlab.show()

if __name__ == '__main__':
    main()
