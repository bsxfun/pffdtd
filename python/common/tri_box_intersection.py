##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: tri_box_intersection.py
#
# Description: Triangle-box intersection routines.  
# One single box / triangle, and one vectorised for one-ray-many-box or one-tri-many-box
# some tests (__main__ entry)
# 
# returns boolean for hit
#
# Following Schwarz-Seidel method (2010)
#
##############################################################################

import numpy as np
from numpy import array as npa
from common.myfuncs import dotv
from common.tris_precompute import tris_precompute

#p is lower corner of box
#dp is distance to upper corner of box
#v is triangle pts
def tri_box_intersection(bbmin,bbmax,tri_pre,debug=False):
    n = tri_pre['nor']
    tbmin = tri_pre['bmin']
    tbmax = tri_pre['bmax']
    v = tri_pre['v'] 

    p = bbmin
    dp = bbmax - bbmin
    assert np.all(dp>0)
    if debug:
        print('bbmin = ' + str(bbmin))
        print('tbmax = ' + str(tbmax))
        print('tbmin = ' + str(tbmin))
        print('bbmax = ' + str(bbmax))

    if np.any(np.logical_or(tbmin > bbmax, bbmin > tbmax)):
        if debug:
            print('bbox fail')
        return False

    #plane through box check
    c = np.where(n>0,dp,npa([0.,0.,0.]))
    #vc = v[0,:]
    vc = tri_pre['cent']
    d1 = np.dot(n,c-vc)
    d2 = np.dot(n,(dp-c)-vc)
    if ((np.dot(n,p) + d1)*(np.dot(n,p) + d2) > 0):
        if debug:
            print('tri-plane fail')
        return False

    #2d overlap checks
    for q in [0,1,2]:
        xq = q%3
        yq = (q+1)%3
        zq = (q+2)%3
        for i in [0,1,2]:
            ei = v[(i+1)%3,:] - v[i,:]
            vixy = 0.5*(v[(i+1)%3,[xq,yq]] + v[i,[xq,yq]])
            neixy = npa([-ei[yq],ei[xq]])
            if n[zq]<0:
                neixy *= -1 
            deixy = -np.dot(neixy,vixy) + np.amax([0,dp[xq]*neixy[0]]) + np.amax([0,dp[yq]*neixy[1]])
            if ((np.dot(neixy,p[[xq,yq]]) + deixy) < 0):
                if debug:
                    print('2d q=%d fail' % q)
                return False
    if debug:
        print('intersects!')
    return True

#many triangles one box, or one-triangle many boxes (vectorised)
def tri_box_intersection_vec(bbmin,bbmax,tris_pre):
    nor = tris_pre['nor']
    tbmin = tris_pre['bmin']
    tbmax = tris_pre['bmax']
    v = tris_pre['v'] 

    p = bbmin
    dp = bbmax - bbmin
    assert np.all(dp>0)

    fail1 = np.any((tbmin > bbmax) | (bbmin > tbmax),axis=-1)

    #plane through box check
    c = np.where(nor>0,dp,np.full(dp.shape,0.))
    vc = tris_pre['cent']
    d1 = dotv(nor,c-vc)
    d2 = dotv(nor,(dp-c)-vc)
    fail2 = ((dotv(nor,p) + d1)*(dotv(nor,p) + d2) > 0)

    fail3 = np.full(fail2.shape,False)

    zer0 = np.zeros((v.shape[0],))
    #2d overlap checks
    for q in [0,1,2]:
        xq = q%3
        yq = (q+1)%3
        zq = (q+2)%3
        for i in [0,1,2]:
            ei = v[:,(i+1)%3,:] - v[:,i,:]
            vixy = 0.5*(v[:,(i+1)%3,[xq,yq]] + v[:,i,[xq,yq]])
            neixy = np.c_[-ei[:,yq],ei[:,xq]]
            neixy[nor[:,zq]<0] *= -1 
            dpx = (dp.T[xq]).T*neixy[:,0]
            dpy = (dp.T[yq]).T*neixy[:,1]
            deixy =  -dotv(neixy,vixy) \
                    + np.where(dpx>0,dpx,np.full(dpx.shape,0.)) \
                    + np.where(dpy>0,dpy,np.full(dpy.shape,0.))
            fail3 |= ((dotv(neixy,(p.T[[xq,yq]]).T) + deixy) < 0)

    return ~(fail1 | fail2 | fail3)

def main():
    import numpy.random as npr
    from common.box import Box 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodraw', action='store_true',help='don''t draw')
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
        Ntris = 4
        #points for tris
        vv = npr.randn(Ntris,3,3)
        pts = vv.reshape((-1,3))
        tris = np.arange(Ntris*3).reshape(-1,3)
        tris_pre = tris_precompute(pts=pts,tris=tris)

        
        b = npr.randn(3,3) #for a random box
        bmin = np.amin(b,axis=0)
        bmax = np.amax(b,axis=0)

        box = Box(*(bmax-bmin),shift=bmin,centered=False)
        hit = np.full(Ntris,False)
        for ti in range(0,Ntris):
            hit[ti] = tri_box_intersection(bmin,bmax,tris_pre[ti],debug=True)
        hit2 = tri_box_intersection_vec(bmin,bmax,tris_pre)

        print(str(vv[ti]))
        assert np.all(hit==hit2),'mismatch!'

        if draw:
            fig = mlab.figure()
            box._draw(color = (0,0,1),r=0.02)
            for ti in range(0,Ntris):
                tbmin = tris_pre[ti]['bmin']
                tbmax = tris_pre[ti]['bmax']
                tbox = Box(*(tbmax-tbmin),shift=tbmin,centered=False)

                if hit[ti]:
                    color = (1,0,0)
                else:
                    color = (1,1,1)
                mlab.triangular_mesh(*(pts.T),tris[ti][None,:],opacity=1,color=color)
                tbox._draw(color=color)

            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
            mlab.orientation_axes()
    print('all good')
    if draw:
        mlab.show()

if __name__ == '__main__':
    main()
