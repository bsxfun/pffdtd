##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: sim_fdtd.py
#
# Description: This is the main FDTD engine, which is re-implemented in C/CUDA
#
# Notes:
#   - This does FVTD-inspired FDTD simulation with frequenency-dependent boundaries
#   - Lossless air (use air absorption filter after)
#   - sided materials (keep one side rigid to save memory and compute time
#   - uses surface area corrections
#   - 13-point FCC (CCP here) or 7-point cartesian schemes
#   - This implementation is straightforward with few optimisations (optimisations in C/CUDA)
#   - Optional numerical energy calculation (energy balance to machine precision)
#   - Plots simulations (mayavi is best, matplotlib is fallback)
#
##############################################################################

import numpy as np
from numpy import array as npa
import numba as nb
from pathlib import Path
from common.timerdict import TimerDict
from tqdm import tqdm
import time
import h5py
import json as json
from common.myfuncs import ind2sub3d,rel_diff,get_default_nprocs

MMb = 12 #max allowed number of branches

class SimEngine:
    def __init__(self,data_dir,energy_on=False,nthreads=None):
        self.data_dir = Path(data_dir)
        self.energy_on = energy_on #will calculate energy
        if nthreads is None:
            nthreads = get_default_nprocs()
        self.print(f'numba set for {nthreads=}')
        nb.set_num_threads(nthreads)

    def print(self,fstring):
        print(f'--ENGINE: {fstring}')

    def load_h5_data(self):
        self.print('loading data..')
        data_dir = self.data_dir
        #here:
        #bn: bn (full)
        #bnr: bn-rigid
        #bnl: bn-lossy (fd updates)
        #bnl ∩ bnr = ø , bnl ∪ bnr = bn 

        h5f = h5py.File(data_dir / Path('vox_out.h5'),'r')
        self.adj_bn   = h5f['adj_bn'][...] #full
        self.bn_ixyz  = h5f['bn_ixyz'][...] #full
        self.Nx       = h5f['Nx'][()] 
        self.Ny       = h5f['Ny'][()]
        self.Nz       = h5f['Nz'][()]
        self.xv       = h5f['xv'][()] #for plotting
        self.yv       = h5f['yv'][()] #for plotting
        self.zv       = h5f['zv'][()] #for plotting
        mat_bn        = h5f['mat_bn'][...]
        saf_bn        = h5f['saf_bn'][...]
        h5f.close()

        ii = (mat_bn>-1)
        self.saf_bnl = saf_bn[ii]
        self.mat_bnl = mat_bn[ii]
        self.bnl_ixyz = self.bn_ixyz[ii]

        h5f = h5py.File(data_dir / Path('comms_out.h5'),'r')
        self.in_ixyz      = h5f['in_ixyz'][...]
        self.out_ixyz     = h5f['out_ixyz'][...]
        self.out_alpha    = h5f['out_alpha'][...]
        self.out_reorder  = h5f['out_reorder'][...]
        self.in_sigs      = h5f['in_sigs'][...]
        self.Ns           = h5f['Ns'][()]
        self.Nr           = h5f['Nr'][()]
        self.Nt           = h5f['Nt'][()]
        h5f.close()

        h5f = h5py.File(data_dir / Path('sim_consts.h5'),'r')
        self.c       = h5f['c'][()]
        self.h       = h5f['h'][()]
        self.Ts      = h5f['Ts'][()]
        self.l       = h5f['l'][()]
        self.l2      = h5f['l2'][()]
        self.fcc_flag = h5f['fcc_flag'][()]
        h5f.close()

        self.fcc = self.fcc_flag>0
        if self.fcc:
            assert self.fcc_flag==1
            
        self.print(f'Nx={self.Nx} Ny={self.Ny} Nz={self.Nz}')
        self.print(f'h={self.h} Ts={self.Ts} c={self.c}, SR={1/self.Ts}')
        self.print(f'l={self.l} l2={self.l2} fcc={self.fcc}')
        self.print(f'Nr={self.Nr} Ns={self.Ns} Nt={self.Nt}')

        if self.fcc:
            assert (self.Nx%2)==0
            assert (self.Ny%2)==0
            assert (self.Nz%2)==0
            self.print(f'On FCC subgrid')
            assert (self.adj_bn.shape[1] == 12)
        if self.fcc:
            self.ssaf_bnl = self.saf_bnl*0.5/np.sqrt(2.0) #rescaled by S*h/V
            assert (self.l<=1.0)
            assert (self.l2<=1.0)
        else:
            self.ssaf_bnl = self.saf_bnl
            assert (self.l<=np.sqrt(1/3))
            assert (self.l2<=1/3)

        
        h5f = h5py.File(Path(data_dir / Path('sim_mats.h5')),'r')
        Nmat = h5f['Nmat'][()]
        DEF = np.zeros((Nmat,MMb,3))
        Mb = h5f['Mb'][...]
        for i in range(Nmat):
            dataset = h5f[f'mat_{i:02d}_DEF'][...]
            assert Mb[i] == dataset.shape[0]
            assert Mb[i] <= MMb
            assert dataset.shape[1] == 3
            DEF[i,:Mb[i]] = dataset
            self.print(f'mat {i}, Mb={Mb[i]}, DEF={DEF[i,:Mb[i]]}')
        h5f.close()

        self.DEF = DEF
        self.Nm = Nmat
        self.Mb = Mb
        self._load_abc()

    def _load_abc(self):
        #calculate ABC nodes (exterior of grid)
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        Nba = 2*(Nx*Ny+Nx*Nz+Ny*Nz) - 12*(Nx+Ny+Nz) + 56
        if self.fcc:
            Nba = Nba//2
        Q_bna = np.full((Nba,),0,dtype=np.int8)
        bna_ixyz = np.full((Nba,),0,dtype=np.int64)
        #get indices
        nb_get_abc_ib(bna_ixyz,Q_bna,Nx,Ny,Nz,self.fcc)
        #assert np.union1d(self.bn_ixyz,bna_ixyz).size == self.bn_ixyz.size + bna_ixyz.size
        self.Q_bna = Q_bna
        self.bna_ixyz = bna_ixyz
        self.Nba = Nba
        if self.energy_on:
            self.V_bna = (2.0**-Q_bna)

    def allocate_mem(self):
        self.print('allocating mem..')
        Nt = self.Nt
        Nr = self.Nr
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        Nm = self.Nm

        u0 = np.zeros((Nx,Ny,Nz),dtype=np.float64)
        u1 = np.zeros((Nx,Ny,Nz),dtype=np.float64)
        Lu1 = np.zeros((Nx,Ny,Nz),dtype=np.float64) #laplacian applied to u1

        u_out = np.zeros((Nr,Nt),dtype=np.float64)

        Nbl = self.bnl_ixyz.size #reduced (non-rigid only)
        u2b = np.zeros((Nbl,),dtype=np.float64)
        u2ba = np.zeros((self.Nba,),dtype=np.float64) 

        vh0 = np.zeros((Nbl,MMb),dtype=np.float64)
        vh1 = np.zeros((Nbl,MMb),dtype=np.float64)
        gh1 = np.zeros((Nbl,MMb),dtype=np.float64)

        if self.energy_on:
            self.H_tot = np.zeros((Nt,),dtype=np.float64)
            self.E_lost = np.zeros((Nt+1,),dtype=np.float64)
            self.E_in = np.zeros((Nt+1,),dtype=np.float64)
            self.u2in = np.zeros((self.Ns,),dtype=np.float64) 

        self.u_out = u_out
        self.u0 = u0
        self.u1 = u1
        self.Lu1 = Lu1
        self.u2b = u2b
        self.vh0 = vh0
        self.vh1 = vh1
        self.gh1 = gh1
        self.u2ba = u2ba

    def setup_mask(self):
        self.print('setting up bn mask..')
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        bn_ixyz = self.bn_ixyz

        bn_mask = np.full((Nx,Ny,Nz),False)
        bn_mask.flat[bn_ixyz] = True

        self.bn_mask = bn_mask

    def set_coeffs(self):
        l = self.l
        l2 = self.l2
        Ts = self.Ts
        mat_bnl = self.mat_bnl
        DEF = self.DEF
        Mb = self.Mb
        Nm = self.Nm

        if self.fcc:
            assert(l2<=1.0)
            a1 = (2.0-l2*3.0) #0.25*12
            a2 = 0.25*l2
        else:
            assert(l2<=1/3)
            a1 = (2.0-l2*6.0)
            a2 = l2

        av = np.array([a1,a2],dtype=np.float64)
        #'b' here means premultiplied by b, +1 material for rigid (and extra coeffs for energy)
        mat_coeffs_struct = np.zeros((Nm+1,),dtype = [('b',np.float64,(MMb,)),\
                                                      ('bd',np.float64,(MMb,)),\
                                                      ('bDh',np.float64,(MMb,)),\
                                                      ('bFh',np.float64,(MMb,)),\
                                                      ('beta',np.float64),\
                                                      ('D',np.float64,(MMb,)),\
                                                      ('E',np.float64,(MMb,)),\
                                                      ('F',np.float64,(MMb,))])
        assert np.all(mat_bnl<Nm) 

        for k in range(Nm):
            M = Mb[k]
            D,E,F = DEF[k][:M].T
            Dh = D/Ts
            Eh = E
            Fh = F*Ts

            b = 1.0/(2.0*Dh+Eh+0.5*Fh)
            d = (2.0*Dh-Eh-0.5*Fh)
            assert ~np.any(np.isnan(np.r_[b,d]))
            assert ~np.any(np.isinf(np.r_[b,d]))

            mat_coeffs_struct[k]['b'][:M]    = b
            mat_coeffs_struct[k]['bd'][:M]   = b*d
            mat_coeffs_struct[k]['bDh'][:M]  = b*Dh
            mat_coeffs_struct[k]['bFh'][:M]  = b*Fh
            mat_coeffs_struct[k]['beta'] = np.sum(b)
            mat_coeffs_struct[k]['D'][:M]  = D
            mat_coeffs_struct[k]['E'][:M]  = E
            mat_coeffs_struct[k]['F'][:M]  = F

            print(f"{k=} {mat_coeffs_struct[k]['b']=}")
            print(f"{k=} {mat_coeffs_struct[k]['bd']=}")
            print(f"{k=} {mat_coeffs_struct[k]['bDh']=}")
            print(f"{k=} {mat_coeffs_struct[k]['bFh']=}")
            print(f"{k=} {mat_coeffs_struct[k]['beta']=}")

        #because of initialising to zero, shouldn't have any nan or inf
        assert ~np.any(np.isnan(mat_coeffs_struct['b']))
        assert ~np.any(np.isinf(mat_coeffs_struct['b']))
        assert ~np.any(np.isnan(mat_coeffs_struct['bd']))
        assert ~np.any(np.isinf(mat_coeffs_struct['bd']))
        assert ~np.any(np.isnan(mat_coeffs_struct['beta']))
        assert ~np.any(np.isinf(mat_coeffs_struct['beta']))
        assert np.all(mat_coeffs_struct['beta']>=0)

        if self.energy_on:
            #copy for continguous memory access
            self.D_bnl = np.copy(mat_coeffs_struct[mat_bnl]['D']) 
            self.E_bnl = np.copy(mat_coeffs_struct[mat_bnl]['E']) #this picks up fake rigid (zeros)
            self.F_bnl = np.copy(mat_coeffs_struct[mat_bnl]['F'])

        self.av = av
        self.l2 = l2
        self.mat_coeffs_struct = mat_coeffs_struct

    def checks(self):
        saf_bnl = self.saf_bnl
        if self.fcc:
            assert np.all(saf_bnl<=12) #unscaled
        else:
            assert np.all(saf_bnl<=6)

    def run_all(self,nsteps=1):
        self.print('running..')
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        Nt = self.Nt
        Npts = Nx*Ny*Nz
        timer = TimerDict()

        pbar = {}
        pbar['vox']= tqdm(total=Nt*Npts,desc=f'FDTD run',unit='vox',unit_scale=True,ascii=True,leave=False,position=0,dynamic_ncols=True)
        pbar['samples'] = tqdm(total=Nt,desc=f'FDTD run',unit='samples',unit_scale=True,ascii=True,leave=False,position=1,ncols=0)

        timer.tic('run')
        for n in range(0,Nt,nsteps):
            nrun = min(nsteps,Nt-n)

            self.run_steps(n,nrun)

            pbar['vox'].update(Npts*nrun)
            pbar['samples'].update(nrun)

        t_elapsed = timer.toc('run',print_elapsed=False)
        pbar['vox'].close()
        pbar['samples'].close()

        self.print(f'Run-time loop: {t_elapsed:.6f}, {Nt*Npts/1e6/t_elapsed:.2f} MVox/s')
    
    def run_plot(self,nsteps=1,draw_backend='mayavi',json_model=None):
        self.print('running..')
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        Nt = self.Nt
        Npts = Nx*Ny*Nz
        bn_mask = self.bn_mask
        in_ixyz = self.in_ixyz
        ix,iy,iz = ind2sub3d(in_ixyz,Nx,Ny,Nz)
        iz_in = np.int_(np.median(iz))
        ix_in = np.int_(np.median(ix))
        iy_in = np.int_(np.median(iy))
        xv = self.xv
        yv = self.yv
        zv = self.zv
        bnm_xy = bn_mask[:,:,iz_in]
        bnm_xz = bn_mask[:,iy_in,:]
        bnm_yz = bn_mask[ix_in,:,:]

        uxy = self.gather_slice(iz=iz_in)
        uxz = self.gather_slice(iy=iy_in)
        uyz = self.gather_slice(ix=ix_in)
        xy_x, xy_y = np.meshgrid(xv, yv, indexing='xy')
        xz_x, xz_z = np.meshgrid(xv, zv, indexing='xy')
        yz_y, yz_z = np.meshgrid(yv, zv, indexing='xy')
        if draw_backend=='matplotlib':
            import matplotlib.pyplot as plt
            #initialise subplots
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            ax.set_title('xy-plane')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            extent=[xv[0],xv[-1],yv[0],yv[-1],]
            hh_xy = ax.imshow(uxy.T,extent=extent,origin='lower',aspect='equal')
            ax.imshow(~bnm_xy.T,extent=extent,origin='lower',aspect='equal',alpha=np.float_(bnm_xy).T)
            fig.colorbar(hh_xy)

            ms = 4
            co = (0,0,0)
            #marker boundary points
            plt.plot(xy_x.flat[bnm_xy.T.flat[:]],xy_y.flat[bnm_xy.T.flat[:]],marker='.',markersize=ms,linestyle='none',color=co)

            ax = fig.add_subplot(1, 3, 2)
            ax.set_title('xz-plane')
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            extent=[xv[0],xv[-1],zv[0],zv[-1],]
            hh_xz = ax.imshow(uxz.T,extent=extent,origin='lower',aspect='equal')
            fig.colorbar(hh_xz)

            plt.plot(xz_x.flat[bnm_xz.T.flat[:]],xz_z.flat[bnm_xz.T.flat[:]],marker='.',markersize=ms,linestyle='none',color=co)

            ax = fig.add_subplot(1, 3, 3)
            ax.set_title('yz-plane')
            ax.set_xlabel('y')
            ax.set_ylabel('z')
            extent=[yv[0],yv[-1],zv[0],zv[-1],]
            hh_yz = ax.imshow(uyz.T,extent=extent,origin='lower',aspect='equal')
            fig.colorbar(hh_yz)

            plt.plot(yz_y.flat[bnm_yz.T.flat[:]],yz_z.flat[bnm_yz.T.flat[:]],marker='.',markersize=ms,linestyle='none',color=co)
            plt.draw()

        elif draw_backend=='mayavi':
            from mayavi import mlab
            from tvtk.api import tvtk 

            fig = mlab.gcf()
            if json_model is not None:
                with open(json_model) as json_file:
                    json_data = json.load(json_file)
                mats_dict = json_data['mats_hash']
                mat_str = list(mats_dict.keys())

                #for section cuts
                edges_xy_0 = npa([[],[]]).T
                edges_xy_1 = npa([[],[]]).T
                edges_xz_0 = npa([[],[]]).T
                edges_xz_1 = npa([[],[]]).T
                edges_yz_0 = npa([[],[]]).T
                edges_yz_1 = npa([[],[]]).T
                for mat in mat_str:
                    pts = npa(mats_dict[mat]['pts'],dtype=np.float64) #no rotation
                    tris = npa(mats_dict[mat]['tris'],dtype=np.int64)
                    for j in range(3):
                        ii = np.nonzero((pts[tris[:,j%3],2]>zv[iz_in]) ^ (pts[tris[:,(j+1)%3],2]>zv[iz_in]))[0]
                        edges_xy_0 = np.r_[edges_xy_0,pts[tris[ii,j%3]][:,[0,1]]]
                        edges_xy_1 = np.r_[edges_xy_1,pts[tris[ii,(j+1)%3]][:,[0,1]]]
                        del ii

                        ii = np.nonzero((pts[tris[:,j%3],1]>yv[iy_in]) ^ (pts[tris[:,(j+1)%3],1]>yv[iy_in]))[0]
                        edges_xz_0 = np.r_[edges_xz_0,pts[tris[ii,j%3]][:,[0,2]]]
                        edges_xz_1 = np.r_[edges_xz_1,pts[tris[ii,(j+1)%3]][:,[0,2]]]
                        del ii

                        ii = np.nonzero((pts[tris[:,j%3],0]>xv[ix_in]) ^ (pts[tris[:,(j+1)%3],0]>xv[ix_in]))[0]
                        edges_yz_0 = np.r_[edges_yz_0,pts[tris[ii,j%3]][:,[1,2]]]
                        edges_yz_1 = np.r_[edges_yz_1,pts[tris[ii,(j+1)%3]][:,[1,2]]]
                        del ii

                    u = edges_xy_1[:,0]-edges_xy_0[:,0]; 
                    v = edges_xy_1[:,1]-edges_xy_0[:,1]; 
                    w=np.zeros(u.shape)
                    mlab.quiver3d(edges_xy_0[:,0],edges_xy_0[:,1],np.full(u.shape,zv[iz_in]),u,v,w,mode='2ddash',scale_factor=1.,color=(0,0,0))

                    u = edges_xz_1[:,0]-edges_xz_0[:,0]; 
                    v=np.zeros(u.shape)
                    w = edges_xz_1[:,1]-edges_xz_0[:,1]; 
                    mlab.quiver3d(edges_xz_0[:,0],np.full(u.shape,yv[iy_in]),edges_xz_0[:,1],u,v,w,mode='2ddash',scale_factor=1.,color=(0,0,0))

                    v = edges_yz_1[:,0]-edges_yz_0[:,0]; 
                    w = edges_yz_1[:,1]-edges_yz_0[:,1]; 
                    u=np.zeros(w.shape)
                    mlab.quiver3d(np.full(u.shape,xv[ix_in]),edges_yz_0[:,0],edges_yz_0[:,1],u,v,w,mode='2ddash',scale_factor=1.,color=(0,0,0))

                    mlab.triangular_mesh(*(pts.T),tris,color=(1., 1., 1.),opacity=0.2)
                    #mlab.triangular_mesh(*(pts.T),tris,color=(0., 0., 0.),representation='wireframe',opacity=0.2)


            #https://matplotlib.org/stable/tutorials/colors/colormaps.html

            #cmap = 'Greys' #(sequential)
            #ldraw = lambda u : 20*np.log10(np.abs(u)+np.spacing(1))

            cmap = 'seismic' #best to diff input so symmetric (diverging colormap)
            ldraw = lambda u : u

            hh_xy = mlab.mesh(xy_x,xy_y,np.full(xy_x.shape,zv[iz_in]),scalars=ldraw(uxy.T),colormap=cmap)
            hh_xz = mlab.mesh(xz_x,np.full(xz_x.shape,yv[iy_in]),xz_z,scalars=ldraw(uxz.T),colormap=cmap)
            hh_yz = mlab.mesh(np.full(yz_y.shape,xv[ix_in]),yz_y,yz_z,scalars=ldraw(uyz.T),colormap=cmap)

            #need three colorbars, but we only see one
            mcb_xy = mlab.colorbar(object=hh_xy,orientation='vertical') #not sure if global scaling
            mcb_xz = mlab.colorbar(object=hh_xz,orientation='vertical') #not sure if global scaling
            mcb_yz = mlab.colorbar(object=hh_yz,orientation='vertical') #not sure if global scaling

            mlab.orientation_axes()

            fake_verts = np.c_[npa([xv[0],yv[0],zv[0]]),npa([xv[-1],yv[-1],zv[-1]])]
            mlab.plot3d(*fake_verts,transparent=True,opacity=0)
            mlab.axes(xlabel='x', ylabel='y', zlabel='z', color=(0., 0., 0.))

            #fix z-up
            fig.scene.interactor.interactor_style = tvtk.InteractorStyleTerrain()

        pbar = tqdm(total=Nt,desc=f'FDTD run',unit='samples',unit_scale=True,ascii=True,leave=False,position=1,ncols=0)
        for n in range(0,Nt,nsteps):
            nrun = min(nsteps,Nt-n)
            self.run_steps(n,nrun)
            pbar.update(nrun)

            uxy = self.gather_slice(iz=iz_in)
            uxz = self.gather_slice(iy=iy_in)
            uyz = self.gather_slice(ix=ix_in)

            if draw_backend=='matplotlib':
                cmax = np.max(np.abs(uxy.flat[:]))
                hh_xy.set_data(uxy.T)
                hh_xy.set_clim(vmin=-cmax*1.1,vmax=cmax*1.1)

                cmax = np.max(np.abs(uxz.flat[:]))
                hh_xz.set_data(uxz.T)
                hh_xz.set_clim(vmin=-cmax*1.1,vmax=cmax*1.1)

                cmax = np.max(np.abs(uyz.flat[:]))
                hh_yz.set_data(uyz.T)
                hh_yz.set_clim(vmin=-cmax*1.1,vmax=cmax*1.1)

                plt.draw()
                plt.pause(1e-10)
                if len(plt.get_fignums())==0:
                    break
            elif draw_backend=='mayavi':
                cmax_xy = np.max(np.abs(uxy.flat[:]))
                cmax_xz = np.max(np.abs(uxz.flat[:]))
                cmax_yz = np.max(np.abs(uyz.flat[:]))
                cmax =  np.max(npa([cmax_xy,cmax_yz,cmax_xz]))

                hh_xy.mlab_source.scalars = ldraw(uxy.T)
                hh_xy.mlab_source.update()

                hh_xz.mlab_source.scalars = ldraw(uxz.T)
                hh_xz.mlab_source.update()

                hh_yz.mlab_source.scalars = ldraw(uyz.T)
                hh_yz.mlab_source.update()

                #mcb_xy.data_range = (cmax-40, cmax)
                #mcb_yz.data_range = (cmax-40, cmax)
                #mcb_xz.data_range = (cmax-40, cmax)
                mcb_xy.data_range = (-cmax*1.1, cmax*1.1) 
                mcb_yz.data_range = (-cmax*1.1, cmax*1.1)
                mcb_xz.data_range = (-cmax*1.1, cmax*1.1)

                #update
                mlab.draw()
                time.sleep(1e-10)
                mlab.process_ui_events()

        if draw_backend=='matplotlib':
            plt.close()
        elif draw_backend=='mayavi':
            mlab.close()

        pbar.close()

    def run_steps(self,nstart,nsteps):
        u0 = self.u0
        u1 = self.u1
        Lu1 = self.Lu1
        in_sigs = self.in_sigs
        u_out = self.u_out

        bn_mask = self.bn_mask
        bn_ixyz = self.bn_ixyz
        adj_bn = self.adj_bn

        bnl_ixyz = self.bnl_ixyz
        ssaf_bnl = self.ssaf_bnl

        in_ixyz = self.in_ixyz
        out_ixyz = self.out_ixyz
        energy_on = self.energy_on
        av = self.av
        l = self.l
        l2 = self.l2
        u2b = self.u2b
        mat_coeffs_struct = self.mat_coeffs_struct
        vh0 = self.vh0
        vh1 = self.vh1
        gh1 = self.gh1
        mat_bnl = self.mat_bnl

        u2ba = self.u2ba
        bna_ixyz = self.bna_ixyz
        Q_bna = self.Q_bna

        if energy_on:
            u2 = self.u0
            Lu2 = self.Lu1
            H_tot = self.H_tot
            E_lost = self.E_lost
            E_in = self.E_in
            h = self.h
            c = self.c
            Ts = self.Ts
            D_bnl = self.D_bnl
            E_bnl = self.E_bnl
            F_bnl = self.F_bnl
            V_bna = self.V_bna
            u2in = self.u2in

        if self.fcc:
            nb_stencil_air = nb_stencil_air_fcc
            nb_stencil_bn = nb_stencil_bn_fcc
            V_fac = 2.0 #cell-vol/h^3
        else:
            nb_stencil_air = nb_stencil_air_cart
            nb_stencil_bn = nb_stencil_bn_cart
            V_fac = 1.0 #cell-vol /h^3

        #run N steps (one at a time, in blocks, or full sim -- for port)
        for n in range(nstart,nstart+nsteps):
            
            if energy_on:
                u2 = self.u0
                Lu2 = self.Lu1
                u2in[:] = u0.flat[in_ixyz]

                #NB: this is an 'energy-like' quantity, but not necessarily in Joules (off by ρ for u as velocity potential)
                H_tot[n] = V_fac*0.5*h*nb_energy_int(u1,u2,Lu2,l2) #H_tot[n] = V_fac*0.5*h*np.sum((((u1-u2)**2)/l2 - u1*Lu2)[1:Nx-1,1:Ny-1,1:Nz-1])
                H_tot[n] -=  V_fac*0.5*h*np.sum((1.0-V_bna)*(((u1.flat[bna_ixyz]-u2.flat[bna_ixyz])**2)/l2 - u1.flat[bna_ixyz]*Lu2.flat[bna_ixyz]))
                #H_tot[n] -=  V_fac*0.5*h*nb_energy_int_corr(V_bna,u1,u2,Lu2,l2,bna_ixyz) #problem with numba fn signature

                H_tot[n] += V_fac*0.5*c/l2*nb_energy_stored(ssaf_bnl,vh1,D_bnl,gh1,F_bnl,Ts) #H_tot[n] += V_fac*0.5*c/l2*np.sum(ssaf_bnl*((vh1**2)*D_bnl + ((Ts*gh1)**2)*F_bnl).T)

            nb_save_bn(u0,u2ba,bna_ixyz)
            nb_flip_halos(u1)

            nb_stencil_air(Lu1,u1,bn_mask)
            nb_stencil_bn(Lu1,u1,bn_ixyz,adj_bn)
            nb_save_bn(u0,u2b,bnl_ixyz)
            nb_leapfrog_update(u0,u1,Lu1,l2)
            nb_update_bnl_fd(u0,u2b,l,bnl_ixyz,ssaf_bnl,vh0,vh1,gh1,mat_bnl,mat_coeffs_struct)

            nb_update_abc(u0,u2ba,l,bna_ixyz,Q_bna)

            #inout
            u0.flat[in_ixyz] += in_sigs[:,n]
            u_out[:,n] = u1.flat[out_ixyz.flat[:]]

            if energy_on:
                E_lost[n+1] = E_lost[n] + V_fac*0.25*h/l*nb_energy_loss(ssaf_bnl,vh0,vh1,E_bnl) #E_lost[n+1] = E_lost[n] + V_fac*0.25*h/l*np.sum(ssaf_bnl*(((vh0+vh1)**2)*E_bnl).T)
                
                E_lost[n+1] += 0.5*V_fac*h/l*np.sum((V_bna*Q_bna)*(u0.flat[bna_ixyz]-u2ba)**2)  #E_lost[n+1] += 0.5*V_fac*h/l*nb_energy_loss_abc(V_bna,Q_bna,u0,u2ba,bna_ixyz) #problem with numba fn signature
                #H_tot[n] += E_lost[n]

                E_in[n+1] = E_in[n] + (V_fac*h/l2)*0.5*np.sum((u0.flat[in_ixyz]-u2in)*in_sigs[:,n]) #have to undo (l2/h/V_fac) scaling applied to in_sigs

            u0,u1 = u1,u0
            vh0,vh1 = vh1,vh0

        #need to rebind these
        self.u0 = u0
        self.u1 = u1
        self.Lu1 = Lu1 #reuse for energy

        self.vh0 = vh0
        self.vh1 = vh1

        self.gh1 = gh1

        if energy_on:
            self.H_tot = H_tot
            self.E_lost = E_lost
            self.E_in = E_in

    def gather_slice(self,ix=None,iy=None,iz=None):
        u1 = self.u1
        if ix is not None:
            uslice = u1[ix,:,:] 
            #fill in checkerboard effect if FCC (subgrid)
            if self.fcc:
                nb_fcc_fill_plot_holes(uslice,ix)

        elif iy is not None:
            uslice = u1[:,iy,:] 
            if self.fcc:
                nb_fcc_fill_plot_holes(uslice,iy)

        elif iz is not None:
            uslice = u1[:,:,iz] 
            if self.fcc:
                nb_fcc_fill_plot_holes(uslice,iz)
            
        return uslice

    def print_last_samples(self,Np):
        self.print('GRID OUTPUTS')
        u_out = self.u_out
        out_reorder = self.out_reorder
        Nt = self.Nt
        Nr = self.Nr
        for i in range(0,Nr):
            self.print(f'out {i}')
            for n in range(Nt-Np,Nt):
                self.print(f'sample {n}: {u_out[out_reorder[i],n]:.16e}')

    def print_last_energy(self,Np):
        self.print('ENERGY')
        H_tot = self.H_tot
        E_lost = self.E_lost
        E_in = self.E_in
        Nt = self.Nt
        for n in range(Nt-Np,Nt):
            self.print(f'normalised energy balance:{rel_diff(H_tot[n]+E_lost[n],E_in[n]):.16e}')

        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1)
        #for i in range(-20,20,1):
            #ax.axhline(i*np.spacing(1), linestyle='-', color=(0.9,0.9,0.9))
        #ax.plot(rel_diff(H_tot+E_lost[:-1],E_in[:-1]),linestyle='None',marker='.')
        #ax.grid(which='both', axis='both')
        #plt.show()

    def save_outputs(self):
        data_dir = self.data_dir
        u_out = self.u_out
        Ts = self.Ts
        out_reorder = self.out_reorder
        #just raw outputs, recombine elsewhere
        h5f = h5py.File(data_dir / Path('sim_outs.h5'),'w')
        h5f.create_dataset(f'u_out', data=u_out[out_reorder,:])
        h5f.close()
        self.print('saved outputs in {data_dir}')

@nb.jit(nopython=True,parallel=True)
def nb_stencil_air_cart(Lu1,u1,bn_mask):
    Nx,Ny,Nz = u1.shape
    for ix in nb.prange(1,Nx-1):
        for iy in range(1,Ny-1):
            for iz in range(1,Nz-1):
                if not bn_mask[ix,iy,iz]:
                    Lu1[ix,iy,iz] = -6.0*u1[ix,iy,iz] \
                                       + u1[ix+1,iy,iz] \
                                       + u1[ix-1,iy,iz] \
                                       + u1[ix,iy+1,iz] \
                                       + u1[ix,iy-1,iz] \
                                       + u1[ix,iy,iz+1] \
                                       + u1[ix,iy,iz-1] 

@nb.jit(nopython=True,parallel=True)
def nb_stencil_air_fcc(Lu1,u1,bn_mask):
    Nx,Ny,Nz = u1.shape
    for ix in nb.prange(1,Nx-1):
        for iy in range(1,Ny-1):
            for iz in range(1,Nz-1):
                if (np.mod(ix+iy+iz,2)==0) and (not bn_mask[ix,iy,iz]):
                    Lu1[ix,iy,iz] = 0.25*(-12.0*u1[ix,iy,iz] \
                                              + u1[ix+1,iy+1,iz] \
                                              + u1[ix-1,iy-1,iz] \
                                              + u1[ix,iy+1,iz+1] \
                                              + u1[ix,iy-1,iz-1] \
                                              + u1[ix+1,iy,iz+1] \
                                              + u1[ix-1,iy,iz-1] \
                                              + u1[ix+1,iy-1,iz] \
                                              + u1[ix-1,iy+1,iz] \
                                              + u1[ix,iy+1,iz-1] \
                                              + u1[ix,iy-1,iz+1] \
                                              + u1[ix+1,iy,iz-1] \
                                              + u1[ix-1,iy,iz+1])

@nb.jit(nopython=True,parallel=True)
def nb_stencil_bn_fcc(Lu1,u1,bn_ixyz,adj_bn):
    Nx,Ny,Nz = u1.shape 
    Nb = bn_ixyz.size
    for i in nb.prange(Nb):
        K = np.sum(adj_bn[i,:])
        ib = bn_ixyz[i] 
        Lu1.flat[ib] = 0.25*(-K*u1.flat[ib]  \
                + adj_bn[i,0] * u1.flat[ib+Ny*Nz+Nz ] \
                + adj_bn[i,1] * u1.flat[ib-Ny*Nz-Nz ] \
                + adj_bn[i,2] * u1.flat[ib+Nz+1] \
                + adj_bn[i,3] * u1.flat[ib-Nz-1] \
                + adj_bn[i,4] * u1.flat[ib+Ny*Nz+1] \
                + adj_bn[i,5] * u1.flat[ib-Ny*Nz-1] \
                + adj_bn[i,6] * u1.flat[ib+Ny*Nz-Nz ] \
                + adj_bn[i,7] * u1.flat[ib-Ny*Nz+Nz ] \
                + adj_bn[i,8] * u1.flat[ib+Nz-1] \
                + adj_bn[i,9] * u1.flat[ib-Nz+1] \
                + adj_bn[i,10]* u1.flat[ib+Ny*Nz-1] \
                + adj_bn[i,11]* u1.flat[ib-Ny*Nz+1])


@nb.jit(nopython=True,parallel=True)
def nb_stencil_bn_cart(Lu1,u1,bn_ixyz,adj_bn):
    Nx,Ny,Nz = u1.shape
    Nb = bn_ixyz.size
    for i in nb.prange(Nb):
        K = np.sum(adj_bn[i,:])
        ib = bn_ixyz[i] 
        Lu1.flat[ib] =  -K*u1.flat[ib]\
             + adj_bn[i,0]*u1.flat[ib+Ny*Nz]\
             + adj_bn[i,1]*u1.flat[ib-Ny*Nz]\
             + adj_bn[i,2]*u1.flat[ib+Nz]\
             + adj_bn[i,3]*u1.flat[ib-Nz]\
             + adj_bn[i,4]*u1.flat[ib+1]\
             + adj_bn[i,5]*u1.flat[ib-1]

@nb.jit(nopython=True,parallel=True)
def nb_flip_halos(u1):
    Nx,Ny,Nz = u1.shape
    for ix in nb.prange(Nx):
        for iy in range(Ny):
            u1[ix,iy,0] = u1[ix,iy,2] 
            u1[ix,iy,Nz-1] = u1[ix,iy,Nz-3] 

    for ix in nb.prange(Nx):
        for iz in range(Nz):
            u1[ix,0,iz] = u1[ix,2,iz] 
            u1[ix,Ny-1,iz] = u1[ix,Ny-3,iz] 

    for iy in nb.prange(Ny):
        for iz in range(Nz):
            u1[0,iy,iz] = u1[2,iy,iz] 
            u1[Nx-1,iy,iz] = u1[Nx-3,iy,iz] 


@nb.jit(nopython=True,parallel=True)
def nb_save_bn(u0,u2b,bn_ixyz):
    #using for bnl and bna
    Nb = bn_ixyz.size
    for i in nb.prange(Nb):
        ib = bn_ixyz[i] 
        u2b.flat[i] = u0.flat[ib] #save before overwrite

@nb.jit(nopython=True,parallel=True)
def nb_leapfrog_update(u0,u1,Lu1,l2):
    Nx,Ny,Nz = u0.shape 
    for ix in nb.prange(1,Nx-1):
        for iy in range(1,Ny-1):
            for iz in range(1,Nz-1):
                u0[ix,iy,iz] = 2.0*u1[ix,iy,iz] - u0[ix,iy,iz] + l2*Lu1[ix,iy,iz] 

@nb.jit(nopython=True,parallel=True)
def nb_update_abc(u0,u2ba,l,bna_ixyz,Q_bna):
    Nba = bna_ixyz.size
    for i in nb.prange(Nba):
        lQ = l*Q_bna[i]
        ib = bna_ixyz[i] 
        u0.flat[ib] = (u0.flat[ib] + lQ*u2ba[i])/(1.0 + lQ)

@nb.jit(nopython=True,parallel=True)
def nb_update_bnl_fd(u0,u2b,l,bnl_ixyz,ssaf_bnl,vh0,vh1,gh1,mat_bnl,mat_coeffs_struct):
    Nbl = bnl_ixyz.size
    for i in nb.prange(Nbl):
        k = mat_bnl[i] 
        if k==-1: #shouldn't happen, but leaving as reminder
            continue 
        b   = mat_coeffs_struct[k]['b']
        bd  = mat_coeffs_struct[k]['bd']
        bDh = mat_coeffs_struct[k]['bDh']
        bFh = mat_coeffs_struct[k]['bFh']
        beta = mat_coeffs_struct[k]['beta']

        lo2Kbg = 0.5*l*ssaf_bnl[i]*beta #has fcc scaling

        ib = bnl_ixyz[i]
        ##add branches
        u0.flat[ib] -= l*ssaf_bnl[i]*np.sum(2.0*bDh*vh1[i,:]-bFh*gh1[i,:])
        u0.flat[ib] = (u0.flat[ib] + lo2Kbg*u2b[i])/(1.0 + lo2Kbg)

        ##update temp variables (for loop implicit)
        vh0[i,:] = b*(u0.flat[ib]-u2b[i]) + bd*vh1[i,:] - 2.0*bFh*gh1[i,:]
        gh1[i,:] += 0.5*vh0[i,:] + 0.5*vh1[i,:]


@nb.jit(nopython=True,parallel=True)
def nb_energy_int(u1,u2,Lu2,l2):
    Nx,Ny,Nz = u1.shape 
    return np.sum((((u1-u2)**2)/l2 - u1*Lu2)[1:Nx-1,1:Ny-1,1:Nz-1])
    #psum = 0.0
    #for i in nb.prange(u1.size):
        #psum += ((u1.flat[i]-u2.flat[i])**2)/l2 - u1.flat[i]*Lu2.flat[i]
        #return psum


@nb.jit(nopython=True,parallel=True)
def nb_energy_stored(ssaf_bnl,vh1,D_bnl,gh1,F_bnl,Ts):
    return np.sum(ssaf_bnl*((vh1**2)*D_bnl + ((Ts*gh1)**2)*F_bnl).T)

@nb.jit(nopython=True,parallel=True)
def nb_energy_loss(ssaf_bnl,vh0,vh1,E_bnl):
    return np.sum(ssaf_bnl*(((vh0+vh1)**2)*E_bnl).T)

#@nb.jit(nopython=True,parallel=True)
#def nb_energy_loss_abc(V_bna,Q_bna,u0,u2ba,bna_ixyz):
    #return np.sum((V_bna*Q_bna)*(u0.flat[bna_ixyz]-u2ba)**2)

#@nb.jit(nopython=True,parallel=True)
#def nb_energy_int_corr(V_bna,u1,u2,Lu2,l2,bna_ixyz):
    #return np.sum((1.0-V_bna)*(((u1.flat[bna_ixyz]-u2.flat[bna_ixyz])**2)/l2 - u1.flat[bna_ixyz]*Lu2.flat[bna_ixyz]))

@nb.jit(nopython=True,parallel=False)
def nb_get_abc_ib(bna_ixyz,Q_bna,Nx,Ny,Nz,fcc):
    ii = 0
    #just doing naive full pass
    for ix in range(1,Nx-1):
        for iy in range(1,Ny-1):
            for iz in range(1,Nz-1):
                if fcc and (ix+iy+iz)%2==1:
                    continue
                Q = 0
                if ((ix==1) or (ix==Nx-2)): 
                    Q+=1
                if ((iy==1) or (iy==Ny-2)):
                    Q+=1
                if ((iz==1) or (iz==Nz-2)):
                    Q+=1
                if Q>0:
                    bna_ixyz[ii] = ix*Ny*Nz + iy*Nz + iz
                    Q_bna[ii] = Q
                    ii += 1
    assert ii==bna_ixyz.size


@nb.jit(nopython=True,parallel=True)
def nb_fcc_fill_plot_holes(uslice,i3):
    N1,N2 = uslice.shape
    for i1 in nb.prange(1,N1-1):
        for i2 in range(1,N2-1):
            if (i1+i2+i3)%2==1:
                uslice[i1,i2] = 0.25*(uslice[i1+1,i2] + uslice[i1-1,i2] + uslice[i1,i2+1] + uslice[i1,i2-1])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,help='run directory')
    parser.add_argument('--json_model', type=str,help='json to plot section cuts')
    parser.add_argument('--plot', action='store_true',help='plot 2d slice')
    parser.add_argument('--draw_backend',type=str,help='matplotlib or mayavi')
    parser.add_argument('--energy', action='store_true',help='do energy calc')
    parser.add_argument('--nsteps', type=int,help='run in batches of steps (less frequent progress)')
    parser.add_argument('--nthreads', type=int,help='number of threads for parallel execution')
    parser.add_argument('--abc', action='store_true',help='apply ABCs')
    parser.set_defaults(draw_backend='matplotlib')
    parser.set_defaults(plot=False)
    parser.set_defaults(energy=False)
    parser.set_defaults(json_model=None)
    parser.set_defaults(abc=False)
    parser.set_defaults(data_dir=None)
    parser.set_defaults(nthreads=get_default_nprocs())
    parser.set_defaults(nsteps=1)

    args = parser.parse_args()

    if args.json_model is not None:
        assert args.draw_backend=='mayavi'

    eng = SimEngine(args.data_dir,energy_on=args.energy,nthreads=args.nthreads)
    eng.load_h5_data()
    eng.setup_mask()
    eng.allocate_mem()
    eng.set_coeffs()
    eng.checks()
    if args.plot:
        eng.run_plot(draw_backend=args.draw_backend,json_model=args.json_model)
    else:
        eng.run_all(args.nsteps)
    eng.save_outputs()
    eng.print_last_samples(5)

    if args.energy:
        eng.print_last_energy(5)

if __name__ == '__main__':
    main()
