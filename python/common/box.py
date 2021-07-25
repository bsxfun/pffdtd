##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: box.py
#
# Description: This is a class for a box.  Used in a few places (vox_grid, tri_box).
#
##############################################################################

import numpy as np
from numpy import array as npa
import numpy.random as npr
import numpy.linalg as npl
from common.myfuncs import mydefault,rotmatrix_ax_ang

class Box:
    def __init__(self,Lx=None,Ly=None,Lz=None,Rax=None,Rang=None,shift=None,centered=True):
        #defaults
        Lx = mydefault(Lx,1.0)
        Ly = mydefault(Ly,1.0)
        Lz = mydefault(Lz,1.0)
        Rax = mydefault(Rax,npa([1.,1.,1.]))
        Rang = mydefault(Rang,0.)
        shift = mydefault(shift,npa([0.,0.,0.]))

        self.centered = centered

        self.init(Lx,Ly,Lz,Rax,Rang,shift)

    def init(self,Lx,Ly,Lz,Rax,Rang,shift):

        verts = npa([[0.,0.,0.],[0.,0.,Lz],[0.,Ly,0.],[0.,Ly,Lz],[Lx,0.,0.],[Lx,0.,Lz],[Lx,Ly,0.],[Lx,Ly,Lz]]) 
        if self.centered:
            verts -= 0.5*npa([Lx,Ly,Lz])

        A = npa([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

        if self.centered:
            b = 0.5*npa([Lx,Ly,Lz,Lx,Ly,Lz])
        else:
            b = npa([0,0,0,Lx,Ly,Lz])

        #rotate verts
        R = rotmatrix_ax_ang(Rax,Rang)
        verts = verts.dot(R.T) + shift

        #get AABB min/max
        bmin = np.amin(verts,0)
        bmax = np.amax(verts,0)

        #update A,b
        A = A.dot(R.T) 
        b += shift.dot(A.T)

        #set member variables
        self.A = A
        self.b = b
        self.verts = verts
        self.bmin = bmin
        self.bmax = bmax

        self.edges = npa([[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[4,5],[4,6],[3,7],[5,7],[6,7]])
        self.tris = npa([[0,1,3],[0,3,2],[1,7,3],[1,5,7],[0,2,6],[0,6,4],[4,7,5],[4,6,7],[2,3,7],[2,7,6],[0,5,1],[0,4,5]])
        self.quads = npa([[0,1,3,2],[0,4,5,1],[4,6,7,5],[1,5,7,3],[2,3,7,6],[0,2,6,4]])

    def randomise(self):
        Lx = 10*npr.random()
        Ly = 10*npr.random()
        Lz = 10*npr.random()
        Rax = npr.rand(3)
        Rang = (-1.0 + 2.0*npr.random())*90
        shift = npr.randn(3)*2 - 1
        self.init(Lx,Ly,Lz,Rax,Rang,shift)

    #just draw (no show)
    def _draw(self,backend='mayavi',r=None,color=(0,1,0)):
        box = self
        
        if backend == 'mayavi':
            from mayavi import mlab
            for edge in box.edges:
                mlab.plot3d(box.verts[edge,0],box.verts[edge,1],box.verts[edge,2],color=color,tube_radius=r)
            mlab.draw()
            print('drawing box..')
        else:
            raise

    #draw and show
    def draw(self,backend='mayavi',r=None,color=(0,1,0)):
        box = self
        if backend == 'mayavi':
            from mayavi import mlab
            from tvtk.api import tvtk 
            fig = mlab.gcf()
            box._draw()
            box._draw_faces()
            fake_verts = np.c_[box.bmin,box.bmax] #stacks as column vectors 2x3
            mlab.plot3d(fake_verts[0],fake_verts[1],fake_verts[2],transparent=True,opacity=0)
            mlab.axes(xlabel='x', ylabel='y', zlabel='z', color=(0., 0., 0.))
            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()
            mlab.show()
        else:
            raise

    def show(self,backend='mayavi'):
        if backend == 'mayavi':
            from mayavi import mlab
            mlab.show()
        elif backend == 'mplot3d':
            import matplotlib.pyplot as plt
            plt.show()

    #draw faces without show()
    def _draw_faces(self,backend='mayavi'):
        box = self
        
        if backend == 'mayavi':
            from mayavi import mlab
            x = box.verts[:,0]; y = box.verts[:,1]; z = box.verts[:,2] 
            mlab.triangular_mesh(x,y,z,box.tris,color=(0,0,0),opacity=0.6)
            mlab.draw()
            print('drawing box faces..')
        else:
            raise

def main():
    #no shift, no rotation
    Lx = npr.random()
    Ly = npr.random()
    Lz = npr.random()
    print('Lx = %.2f, Ly = %.2f, Lz = %.2f' % (Lx,Ly,Lz)) 

    box = Box(Lx,Ly,Lz)
    print('box 0 ... bmin = %.2f,%.2f,%.2f' % (box.bmin[0],box.bmin[1],box.bmin[2]))
    print('box 0 ... bmax =  %.2f,%.2f,%.2f' % (box.bmax[0],box.bmax[1],box.bmax[2]))
    assert np.all(np.mean(box.verts,0).dot(box.A.T) < box.b)
    assert np.all((box.verts*(1-1e-4)).dot(box.A.T) < box.b)
    assert np.any((box.verts*(1+1e-4)).dot(box.A.T) > box.b)

    #rotated
    Lx = npr.random()
    Ly = npr.random()
    Lz = npr.random()
    print('Lx = %.2f, Ly = %.2f, Lz = %.2f' % (Lx,Ly,Lz)) 
    Rax = npr.rand(3)
    Rang = (-1.0 + 2.0*npr.random())*90
    print('Rax = %.2f,%.2f,%.2f, Rang = %.2f degrees' % (Rax[0],Rax[1],Rax[2],Rang))
    print()

    box = Box(Lx,Ly,Lz,Rax,Rang)
    print('Lx = %.2f, Ly = %.2f, Lz = %.2f' % (Lx,Ly,Lz)) 
    print('Rax = %.2f,%.2f,%.2f, Rang = %.2f degrees' % (Rax[0],Rax[1],Rax[2],Rang))
    print('box 1 ... bmin = %.2f,%.2f,%.2f' % (box.bmin[0],box.bmin[1],box.bmin[2]))
    print('box 1 ... bmax =  %.2f,%.2f,%.2f' % (box.bmax[0],box.bmax[1],box.bmax[2]))
    assert np.all(np.mean(box.verts,0).dot(box.A.T) < box.b)
    assert np.all((box.verts*(1-1e-4)).dot(box.A.T) < box.b)
    assert np.any((box.verts*(1+1e-4)).dot(box.A.T) > box.b)
    print()

    #shifted and rotated
    Lx = npr.random()
    Ly = npr.random()
    Lz = npr.random()
    print('Lx = %.2f, Ly = %.2f, Lz = %.2f' % (Lx,Ly,Lz)) 
    Rax = npr.rand(3)
    Rang = (-1.0 + 2.0*npr.random())*90
    print('Rax = %.2f,%.2f,%.2f, Rang = %.2f degrees' % (Rax[0],Rax[1],Rax[2],Rang))
    shift = npr.rand(3)*100
    print('shift = %.2f,%.2f,%.2f' % (shift[0],shift[1],shift[2]))

    box = Box(Lx,Ly,Lz,Rax,Rang,shift)
    print('box 2 ... bmin = %.2f,%.2f,%.2f' % (box.bmin[0],box.bmin[1],box.bmin[2]))
    print('box 2 ... bmax =  %.2f,%.2f,%.2f' % (box.bmax[0],box.bmax[1],box.bmax[2]))
    assert np.all(np.mean(box.verts,0).dot(box.A.T) < box.b)
    assert np.all((((box.verts-shift)*(1-1e-4))+shift).dot(box.A.T) < box.b)
    assert np.any((((box.verts-shift)*(1+1e-4))+shift).dot(box.A.T) > box.b)
    box.draw()

if __name__ == '__main__':
    main()
