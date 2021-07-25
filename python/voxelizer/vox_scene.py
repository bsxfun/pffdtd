##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: vox_scene.py
#
# Description: a scene voxelizer for FDTD  
#
#  Sets up primary input for FDTD (adjacencies, materials, boundary nodes)
#  CPU-based and uses multiprocessing.  
#  Tested up to ~10^10 points.  Meant for single-node use (no MPI)
#
# Notes:
#  - Performance will depend on geometry, grid spacing, voxel size and # processes
#  - Simple *heuristic* default auto-tuning (Nvox_est) provided (manual choice usually better)
#  - This temporarily uses disk space for multiprocessing, will warn if not enough space
#  - In single precision or with FCC, Disk usage relative to simulation size goes up
#
# About voxelisation:
#  - despite the use of term 'voxelizer' this is not necessarily a solid or surface voxelizer
#  - this computes a lower-level adjacency graph for points on a grid (a grid of voxels)
#  - this allows for non-watertight scenes
#  - expects sides of materials properly oriented, boundary conditions depend on this (also FDTD performance)
#  - has surface-area corrections to mitigate staircasing errors
#  - points too near boundary (within some eps) are set not adjacent to neighbours
#  - this is meant for interpretation of FDTD grid as FVTD mesh of voxels/cells
#  - this exports data just for boundary nodes (anything with non-adjacency to a neighbour)
#
##############################################################################

import numpy as np
from numpy import array as npa
from common.room_geo import RoomGeo
from common.timerdict import TimerDict
from common.tri_ray_intersection import tri_ray_intersection_vec
from common.tri_box_intersection import tri_box_intersection_vec
from voxelizer.cart_grid import CartGrid
from voxelizer.vox_grid import VoxGrid
from common.myfuncs import clear_dat_folder
from common.myfuncs import yes_or_no,ind2sub3d
from common.myfuncs import dotv
from common.myfuncs import get_default_nprocs
from pathlib import Path
import numba as nb
import sys
import h5py

import psutil

import multiprocessing as mp
from multiprocessing import shared_memory
from tqdm import tqdm
from memory_profiler import profile as memory_profile

F_EPS = np.finfo(np.float64).eps
R_EPS = 1e-6 #relative eps (to grid spacing) for near hits
DAT_FOLDER = 'mmap_dat' #change as needed

class VoxScene:
    def __init__(self,room_geo=None,cart_grid=None,vox_grid=None,fcc=False):
        self.room_geo = room_geo
        self.vox_grid = vox_grid
        self.cart_grid = cart_grid
        h = cart_grid.h #grid spacing

        self.NN = 6 #number of nearest neighbours
        self.hf = h #scaled h (for FCC)
        self.face_area = h*h
        self.VV = npa([[1.,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        self.uvv = self.VV
        if fcc:
            self.NN = 12
            self.face_area /= np.sqrt(2.0)
            self.hf *= np.sqrt(2.0) #actually grid spacing on FCC subgrid
            self.VVc = self.VV
            self.VV = npa([[+1.,+1,0],[-1,-1,0],[0,+1,+1],[0,-1,-1],[+1,0,+1],[-1,0,-1], \
                           [+1,-1,0],[-1,+1,0],[0,+1,-1],[0,-1,+1],[+1,0,-1],[-1,0,+1]])
            self.uvv = self.VV/np.sqrt(2.0) #normalised
            self.print(f'Using FCC subgrid')

        self.vvh = h * self.VV
        self.fcc = fcc
        self.nprocs = get_default_nprocs()

        self.timer = TimerDict()

    def print(self,fstring):
        print(f'--VOX_SCENE: {fstring}')

    #@memory_profile
    def calc_adj(self,Nprocs=None):
        if Nprocs is None:
            Nprocs = self.nprocs
        self.print(f'using {Nprocs} processes')

        cg = self.cart_grid
        vg = self.vox_grid
        rg = self.room_geo

        h = cg.h
        hf = self.hf
        VV = self.VV
        vvh = self.vvh #vectors scaled by h (with length gf)
        uvv = self.uvv #normalised
        ivv = np.int_(VV) #integer grid steps
        face_area = self.face_area
        Nx,Ny,Nz = cg.Nxyz
        Ngridpoints = cg.Npts
        xv = cg.xv
        yv = cg.yv
        zv = cg.zv
        xmin,ymin,zmin = cg.xyzmin
        Nh = vg.Nh
        Nvox = vg.Nvox

        #only process non-empty voxels from vox_grid
        Nvox_nonempty = len(vg.nonempty_idx)
        self.print(f'Non-empty voxels: {Nvox_nonempty}, {Nvox_nonempty/Nvox*100.0:.2f}%')

        min_vox_shape = (Nh,Nh,Nh) #for memory calculation

        #set up shared memory
        Nb_proc_shm = shared_memory.SharedMemory(create=True,size=Nprocs*np.dtype(np.int64).itemsize)
        Nb_proc = np.frombuffer(Nb_proc_shm.buf,dtype=np.int64)
        Nb_proc[:] = 0 

        NN = self.NN

        #will need this much for check_adj (less needed for vox data)
        disk_space_needed = Nx*Ny*Nz*(1+self.fcc)
        self.print(f'{disk_space_needed/2**30=:.3f} GiB')
        disk_space_available = psutil.disk_usage('.').free
        self.print(f'{disk_space_available/2**30=:.3f} GiB')
        #proceed without asking unless more than 50% of free space
        if disk_space_needed > 0.5*disk_space_available:
            self.print("WARNING: -- disk space usage high")
            if not yes_or_no('continue?'):
                raise Exception('cancelled')

        clear_dat_folder(DAT_FOLDER)

        #this is main function called by mp
        def process_voxel(idx,proc_idx):
            vox_idx = vg.nonempty_idx[idx]
            vox = vg.voxels[vox_idx]

            ##voxel start indices (absolute) and including halos
            ix_start,iy_start,iz_start = vox.ixyz_start
            #these are widths of voxel, but number points is plus one
            Nhx,Nhy,Nhz = vox.Nhxyz

            vox_shape = (Nhx,Nhy,Nhz) #in points

            #local indices for vox
            ix_vox,iy_vox,iz_vox = np.mgrid[0:Nhx,0:Nhy,0:Nhz]

            vox_ndist = np.full(vox_shape,np.inf,dtype=np.float64) #distance to nearest hit
            vox_bp = np.full(vox_shape,False,dtype=np.bool8) #boundary point?
            vox_adj = np.full((*vox_shape,NN),True,dtype=np.bool8) #adjacency to neighbours
            vox_nb = np.full(vox_shape,False,dtype=np.bool8) #near a boundary (nothing to do with numba)
            vox_tidx = np.full(vox_shape,-1,dtype=np.int32) #tri index for nearest hit

            #to store distances to tris
            hit_dist = np.full(vox_shape,np.inf)

            xyz_vox = np.c_[xv[ix_start+ix_vox.flat[:]],\
                            yv[iy_start+iy_vox.flat[:]],\
                            zv[iz_start+iz_vox.flat[:]]]
            in_mask = np.full(vox_shape,False)
            in_mask[1:-1,1:-1,1:-1] = True,

            if self.fcc:
                fcc_mask = (np.mod(ix_start+ix_vox+iy_start+iy_vox+iz_start+iz_vox,2)==0)
            else:
                fcc_mask = np.full(vox_shape,True)

            #loop through triangles in voxel
            for tri_pre,tri_ind in zip(vox.tris_pre,vox.tri_idxs):
                cent = tri_pre['cent']
                unor = tri_pre['unor']
                tbmin = tri_pre['bmin']
                tbmax = tri_pre['bmax']
                #first mask by bounding box
                bb_mask = (np.all(xyz_vox >= tbmin - hf*(1+R_EPS),axis=-1) \
                         & np.all(xyz_vox <= tbmax + hf*(1+R_EPS),axis=-1)).reshape(vox_shape)

                #bb_mask &= in_mask.flat[:] #inside halo

                bb_mask &= fcc_mask
                if ~np.any(bb_mask):
                    continue

                #then mask by distance to plane
                dtp = np.full(vox_shape,np.inf,dtype=np.float64)
                dtp.flat[bb_mask.flat[:]] = dotv(unor,cent-xyz_vox[bb_mask.flat[:]]) 
                dist_mask1 = (np.abs(dtp)<= hf*(1+R_EPS))
                if ~np.any(dist_mask1):
                    continue

                ray_mask = dist_mask1
                tnb_mask = np.full(vox_shape,False) #this is reset at triangle, accumulates across directions
                for k in range(0,NN):
                    ray_o = xyz_vox[ray_mask.flat[:]]-vvh[k]
                    rd = uvv[k]

                    ray_d = rd*np.ones(ray_o.shape)
                
                    #returns np.inf or dist>0 if hit
                    hit_dist = np.full(vox_shape,np.inf)
                    _,hit_dist.flat[ray_mask.flat[:]] = tri_ray_intersection_vec(ray_o,ray_d,npa([tri_pre]),d_eps=1.0e-3*h)

                    assert np.all(hit_dist>=0.0)
                    hit_dist -= hf #shift, doesn't affect np.inf entries
                    hit_dist[hit_dist<-R_EPS*hf] = np.inf #overwrite hits behind point 

                    tnb_mask |= (np.abs(hit_dist)<=R_EPS*hf)
                    hit_dist[tnb_mask] = np.abs(hit_dist[tnb_mask]) #so ndist is positive
                    vox_nb |= tnb_mask

                    if ~np.any(hit_dist<=hf):
                        continue
                    hit_dist[hit_dist>(1+R_EPS)*hf] = np.inf #zero those out so they don't interfere

                    ii0 = np.flatnonzero(hit_dist<=(1+R_EPS)*hf) #linear indices

                    #mark non-adjencies (later use to detect boundary nodes)
                    vox_adj.reshape(-1,NN)[ii0,k] = False
                    vox_bp.flat[ii0] = True

                    #indices where new nearest hit 
                    nh_mask = np.full(vox_shape,False)
                    nh_mask.flat[ii0] =  (hit_dist.flat[ii0] < vox_ndist.flat[ii0])
                    vox_ndist[nh_mask] = hit_dist[nh_mask] #update to abs for neg dist
                    vox_tidx[nh_mask] = tri_ind

                #NB can have fictitious boundary faces at edges
                #leg on correct side but not overtop triangle... (since intersection not registered)
                #..same problem with jordan's theorem tests
 
                #finally zero out nb points 
                vox_adj.reshape(-1,NN)[vox_nb.flat[:],:]=False

                assert np.all(~vox_adj.reshape(-1,NN)[tnb_mask.flat[:],:])

            vox_adj = vox_adj.reshape((-1,NN))
            assert np.all(~vox_adj[vox_nb.flat[:],:])

            vox_adj[~in_mask.flat[:],:] = True
            vox_bp[~in_mask] = False
            vox_tidx[~in_mask] = -1 #just so doesn't go into surface-area correction calc

            #now extract boundary points (redundant)
            qq = np.flatnonzero(np.any(~vox_adj,axis=-1))
            qq2 = np.flatnonzero(vox_bp.flat[:])
            #print(f'{qq.size=}')

            assert qq.size == qq2.size
            #assert np.intersect1d(qq,qq2).size == qq2.size
            assert np.all(qq==qq2)
            #tally boundary points inside vox without halo (add on to tally for process)
            Nb_proc[proc_idx] += np.sum(vox_bp.flat[:])

            ndist_bn_vox = vox_ndist.flat[qq]
            tidx_bn_vox = vox_tidx.flat[qq]
            assert np.all(tidx_bn_vox>=-1) #all marked

            adj_bn_vox = vox_adj[qq,:]
            bn_ixyz_loc_vox = qq

            #store vox info on disk (variable size, can't use shared mem), no compression for speed
            h5f_vox = h5py.File(Path(DAT_FOLDER) / Path(f'vox_data_{vox.idx}.h5'),'w')
            h5f_vox.create_dataset('adj_bn', data=adj_bn_vox)
            h5f_vox.create_dataset('tidx_bn', data=tidx_bn_vox)
            h5f_vox.create_dataset('ndist_bn', data=ndist_bn_vox)
            h5f_vox.create_dataset('bn_ixyz_loc', data=bn_ixyz_loc_vox)
            h5f_vox.close()

        def process_voxels(idx_list,proc_idx):
            #using one progress bar because tqdm has problems with multiple, and cleaner 
            pbar = tqdm(total=len(idx_list),desc=f'process {proc_idx:02d} voxeliser processing',ascii=True,leave=False,position=0)
            for idx in idx_list:
                process_voxel(idx,proc_idx)
                pbar.update(1)
            pbar.close()

        self.timer.tic('calc_adj total')
        self.timer.tic('ray-tri checks')

        if Nprocs==1: #no need to use mp
            process_voxels(range(Nvox_nonempty),0)

        else: #multiproc with vox grid
            procs = []
            idx_lists = [[] for i in range(Nprocs)]
            #only random shuffle for balancing
            vox_order = np.random.permutation(Nvox_nonempty)
            for qq in range(Nvox_nonempty):
                cc = np.argmin([len(l) for l in idx_lists])
                idx_lists[cc].append(vox_order[qq])

            for proc_idx in range(Nprocs):
                idx_list = idx_lists[proc_idx]
                proc = mp.Process(target=process_voxels, args=(idx_list,proc_idx))
                procs.append(proc)

            for proc_idx in range(Nprocs):
                procs[proc_idx].start()

            for one_proc in procs:
                one_proc.join()

        self.print(self.timer.ftoc('ray-tri checks'))

        self.timer.tic('consolidate')

        #total number of boundary points, to allocate unified arrays
        Nbt = np.sum(Nb_proc)
        self.print(f'{Nbt=}')

        #clean up shared memory
        Nb_proc_shm.close()
        Nb_proc_shm.unlink()
        #self.print(f'unlink')

        #unified arrays
        bn_ixyz = np.full((Nbt,),-1,dtype=np.int64)
        adj_bn = np.full((Nbt,NN),True,dtype=bool)
        tidx_bn = np.full((Nbt,),-1,dtype=np.int32)
        ndist_bn = np.full((Nbt,),np.inf,dtype=np.float64)

        #consolidate now with single process 
        pbar = tqdm(total=Nvox_nonempty,desc=f'process 0: consolidate',ascii=True,leave=False,position=0)
        bb = 0 #boundary point counter
        for idx in range(Nvox_nonempty):
            vox_idx = vg.nonempty_idx[idx]
            vox = vg.voxels[vox_idx]
            Nhx,Nhy,Nhz = vox.Nhxyz 
            vox_shape = (Nhx,Nhy,Nhz) #in points
            ix_start,iy_start,iz_start = vox.ixyz_start

            if len(vox.tri_idxs)==0:
                continue

            #extract boundary points only
            h5f_vox = h5py.File(Path(DAT_FOLDER) / Path(f'vox_data_{vox.idx}.h5'),'r')
            adj_bn_vox = h5f_vox['adj_bn'][...]
            tidx_bn_vox = h5f_vox['tidx_bn'][...]
            ndist_bn_vox = h5f_vox['ndist_bn'][...]
            bn_ixyz_loc_vox = h5f_vox['bn_ixyz_loc'][...]
            h5f_vox.close()

            qq = bn_ixyz_loc_vox

            bn_ix,bn_iy,bn_iz = ind2sub3d(qq,*vox_shape)

            bn_ixyz_vox = (bn_iz+iz_start) \
                        + (bn_iy+iy_start)*Nz \
                        + (bn_ix+ix_start)*Ny*Nz

            assert np.all(bn_ixyz_vox<Ngridpoints)
            assert np.all(bn_ixyz_vox>=0)

            Nb_vox = bn_ixyz_vox.size
            if Nb_vox>0:
                assert bb+Nb_vox<=Nbt
                adj_bn[bb:bb+Nb_vox] = adj_bn_vox
                tidx_bn[bb:bb+Nb_vox] = tidx_bn_vox
                ndist_bn[bb:bb+Nb_vox] = ndist_bn_vox
                bn_ixyz[bb:bb+Nb_vox] = bn_ixyz_vox
            bb += Nb_vox
            pbar.update(1)
        assert bb==Nbt
        self.print(self.timer.ftoc('consolidate'))
        pbar.close()

        #merge 
        self.timer.tic('merge')

        assert np.unique(bn_ixyz).size == bn_ixyz.size

        self.print(self.timer.ftoc('merge'))

        clear_dat_folder(DAT_FOLDER)

        #surface area corrections (effective cell surface seen by walls)
        self.print('materials (+sides)...')
        self.timer.tic('sides')
        bn_ix,bn_iy,bn_iz = ind2sub3d(bn_ixyz,Nx,Ny,Nz) 
        xyz_bn = np.c_[xv[bn_ix],yv[bn_iy],zv[bn_iz]]
        xyz_bn = np.c_[xv[bn_ix],yv[bn_iy],zv[bn_iz]]
        dv = dotv(xyz_bn-rg.tris_pre['cent'][tidx_bn],rg.tris_pre['unor'][tidx_bn]) 

        mat_bn = rg.mat_ind[tidx_bn] #default choice
        #unmark wrong sides of one-sided triangles
        mat_bn[(dv>0) & (rg.mat_side[tidx_bn]==1)] = -1
        mat_bn[(dv<0) & (rg.mat_side[tidx_bn]==2)] = -1
        #anything 'near boundary' mark as rigid
        mat_bn[np.all(~adj_bn,axis=-1)] = -1

        self.print(f'Npts = {cg.Npts}, Nbl = {np.sum(mat_bn>-1)}')

        if np.any(mat_bn[rg.mat_side[tidx_bn]==0]):
            assert rg.mat_str[-1] == '_RIGID'
            assert len(rg.mat_str)==rg.Nmat+1
            assert np.all(mat_bn[rg.mat_side[tidx_bn]==0] == -1)
        self.print(self.timer.ftoc('sides'))

        self.print('surface area corrections...')
        self.timer.tic('surface area corrections')

        saf_bn_0 = np.sum(~adj_bn,axis=-1) #this will be number of faces by default
        saf_bn = np.zeros(bn_ixyz.size,dtype=np.float64) #this will be a number between 0 and NN
        for j in range(0,NN,2):
            saf = np.abs(dotv(uvv[j],rg.tris_pre['unor'][tidx_bn]))
            saf_bn += (~adj_bn[:,j] + ~adj_bn[:,j+1])*saf

        mat_approx_sa = np.zeros((rg.Nmat+1,),dtype=np.float64)
        mat_approx_sa_0 = np.zeros((rg.Nmat+1,),dtype=np.float64)

        #could parallel reduce (maybe with numba)
        np.add.at(mat_approx_sa,mat_bn,face_area*saf_bn) #-1, rigid goes to end
        np.add.at(mat_approx_sa_0,mat_bn,face_area*saf_bn_0) #-1, rigid goes to end

        self.print(self.timer.ftoc('surface area corrections'))
        #N.B: rg.mat_area takes into account two-sided
        for i in range(rg.Nmat):
            self.print(f'mat: {rg.mat_str[i]}, original: {(mat_approx_sa_0[i]/rg.mat_area[i]-1)*100.:.3f}% over, corrected: {(mat_approx_sa[i]/rg.mat_area[i]-1)*100:.3f}% over')


        #attach to class
        self.bn_ixyz = bn_ixyz 
        self.adj_bn  = adj_bn
        self.mat_bn  = mat_bn
        self.saf_bn  = saf_bn

        self.print(self.timer.ftoc('calc_adj total'))

    def save(self,save_folder,compress=None):
        #save to HDF5 data file 
        save_folder = Path(save_folder)
        self.print(f'{save_folder=}')
        if not save_folder.exists():
            save_folder.mkdir(parents=True)
        else:
            assert save_folder.is_dir()

        bn_ixyz = self.bn_ixyz #boundary node (bn) linear indices
        adj_bn = self.adj_bn #adjacencies for bn
        mat_bn = self.mat_bn #material indices for bn
        saf_bn = self.saf_bn #boundary-surface area factors (corrections) for bn
        xv = self.cart_grid.xv
        yv = self.cart_grid.yv
        zv = self.cart_grid.zv
        h = self.cart_grid.h
        Nx,Ny,Nz = self.cart_grid.Nxyz

        memory_saved = 0
        memory_saved += (bn_ixyz.size * bn_ixyz.itemsize)
        memory_saved += (adj_bn.size * adj_bn.itemsize)
        memory_saved += (mat_bn.size * mat_bn.itemsize)
        memory_saved += (saf_bn.size * saf_bn.itemsize)
        memory_saved += (xv.size * xv.itemsize)
        memory_saved += (yv.size * yv.itemsize)
        memory_saved += (zv.size * zv.itemsize)
        self.print(f'memory saved: {memory_saved/2**20:.3f} MiB')

        self.print(f'saving with compression: {compress} ...')
        if compress is not None:
            kw = {'compression': "gzip", 'compression_opts': compress}
        else:
            kw = {}
        h5f = h5py.File(save_folder / Path('vox_out.h5'),'w')
        h5f.create_dataset('bn_ixyz', data=bn_ixyz, **kw)
        h5f.create_dataset('adj_bn', data=adj_bn, **kw)
        h5f.create_dataset('mat_bn', data=mat_bn, **kw)
        h5f.create_dataset('saf_bn', data=saf_bn, **kw)
        h5f.create_dataset('xv', data=xv, **kw) #also in cart_grid, but this one can get transformed
        h5f.create_dataset('yv', data=yv, **kw)
        h5f.create_dataset('zv', data=zv, **kw)
        h5f.create_dataset('h', data=np.float64(h)) #giving types just to be clear
        h5f.create_dataset('Nx', data=np.int64(Nx))
        h5f.create_dataset('Ny', data=np.int64(Ny))
        h5f.create_dataset('Nz', data=np.int64(Nz))
        h5f.create_dataset('Nb', data=np.int64(bn_ixyz.size))
        h5f.close()

        #uncomment if importing data to Matlab (Matlab reads HDF5 bool data as strings)
        #h5f = h5py.File(save_folder / Path('adj_bn.h5'),'w')
        #h5f.create_dataset('adj_bn', data=adj_bn.astype(np.int8), **kw)
        #h5f.close()

    def check_adj_full(self):
        #check full adjacency map (pre-req for stability)
        #uses disk space (but bit compressed)
        cg = self.cart_grid
        Nx,Ny,Nz = cg.Nxyz
        bn_ixyz = self.bn_ixyz
        adj_bn  = self.adj_bn
        mat_bn  = self.mat_bn
        Nb = bn_ixyz.size

        self.print('checking adj...')
        self.timer.tic('check_full')

        self.print('mmap...')
        #numba has no problem with memmap'd arrays
        if self.fcc:
            #FCC uses int16
            adj_full = np.memmap(Path(DAT_FOLDER) / Path('adj_check.dat'), dtype='uint16', mode='w+', shape=(Nx,Ny,Nz))
            self.print('filling full adj map (16-bit compressed)...')
            adj_full[:] = ~np.uint16(0)
            nb_fill_adj_fcc(bn_ixyz,adj_bn,adj_full)
            self.print('check...')
            nb_check_adj_full_fcc(adj_full,Nx,Ny,Nz)
        else:
            adj_full = np.memmap(Path(DAT_FOLDER) / Path('adj_check.dat'), dtype='uint8', mode='w+', shape=(Nx,Ny,Nz))
            self.print('filling full adj map (8-bit compressed)...')
            adj_full[:] = ~np.uint8(0)
            nb_fill_adj(bn_ixyz,adj_bn,adj_full)
            self.print('check...')
            nb_check_adj_full(adj_full,Nx,Ny,Nz)

        del adj_full #release mmap
        clear_dat_folder(DAT_FOLDER)
        self.print(self.timer.ftoc('check_full'))

    def draw(self,backend='mayavi'):
        #better to use use polyscope for large grids
        cg = self.cart_grid
        rg = self.room_geo
        bn_ixyz = self.bn_ixyz
        adj_bn  = self.adj_bn
        mat_bn  = self.mat_bn
        h  = cg.h
        xv = cg.xv
        yv = cg.yv
        zv = cg.zv
        Nx,Ny,Nz = cg.Nxyz
        Nmat = rg.Nmat
        colors = rg.colors
        
        bn_ix,bn_iy,bn_iz = ind2sub3d(bn_ixyz,Nx,Ny,Nz)

        if backend == 'mayavi':
            from mayavi import mlab
        elif backend == 'polyscope':
            import polyscope as ps

        self.print('drawing...')
        for i in range(-1,Nmat): #zero-indexing, -1 is rigid
            if i==-1:
                sf = h/4
                color = (1,1,1)
                mat = 'rigid'
            else:
                sf = h/2
                color = tuple([c/255.0 for c in colors[i]])
                mat = rg.mat_str[i]
            self.print(f'drawing mat #{i}: {mat}')

            qq = np.flatnonzero((mat_bn==i))
            if backend == 'mayavi':
                if i==-1:
                    mlab.points3d(xv[bn_ix[qq]],yv[bn_iy[qq]],zv[bn_iz[qq]],color=color,\
                            mode='cube',scale_factor=sf)
                else:
                    mlab.points3d(xv[bn_ix[qq]],yv[bn_iy[qq]],zv[bn_iz[qq]],color=color,\
                            mode='sphere',resolution=8,scale_factor=sf)

            elif backend == 'polyscope':
                ps_cloud_in = ps.register_point_cloud(mat,\
                        np.c_[xv[bn_ix[qq]],yv[bn_iy[qq]],zv[bn_iz[qq]]], color=color)
                ps_cloud_in.set_radius(sf/2, relative=False)


        vvh = self.vvh
        for j in range(0,vvh.shape[0],2): #to draw legs only once
            qq = np.flatnonzero(~adj_bn[:,j])
            aa = vvh[j,:] 
            x = xv[bn_ix[qq]]
            y = yv[bn_iy[qq]]
            z = zv[bn_iz[qq]]
            if backend == 'mayavi':
                uvw = np.tile(aa,(qq.size,1))
                u = uvw[:,0]; v=uvw[:,1]; w=uvw[:,2]
                mlab.quiver3d(x,y,z,u,v,w,color=(0,1,0),mode='2ddash',scale_factor=1.)
            elif backend == 'polyscope':
                #could condense to one edge network, but keeping simple
                nodes = np.r_[np.c_[x,y,z],np.c_[x,y,z]+aa]
                edges = np.c_[np.arange(qq.size),np.arange(qq.size)+qq.size]
                ps_net = ps.register_curve_network(f'notADJ legs - {j}', nodes, edges, color=(0,1,0))
                ps_net.set_radius(h/40, relative=False)

        if backend == 'mayavi':
            mlab.draw()

        self.print('drawn')

#numba funcs
#these are in fact faster in serial as written and current numba version
#possible to improve with option to choose scheduling types (maybe a feature in future numba versions)
@nb.jit(nopython=True,parallel=False) 
def nb_fill_adj(bn_ixyz,adj_bn,adj_full):
    for i in nb.prange(bn_ixyz.size):
        bitmask = np.uint8(0)
        for jj in np.arange(6):
            bitmask |= (adj_bn[i,jj] << jj)
        adj_full.flat[bn_ixyz[i]] = bitmask
        #print(f'{bitmask=}, {adj_bn[i]=}')

@nb.jit(nopython=True,parallel=False) 
def nb_fill_adj_fcc(bn_ixyz,adj_bn,adj_full):
    for i in nb.prange(bn_ixyz.size):
        bitmask = np.uint16(0)
        for jj in np.arange(12):
            bitmask |= (adj_bn[i,jj] << jj)
        adj_full.flat[bn_ixyz[i]] = bitmask
        #print(f'{bitmask=}, {adj_bn[i]=}')

@nb.jit(nopython=True,parallel=False)
def nb_check_adj_full(adj,Nx,Ny,Nz):
    assert adj.shape == (Nx,Ny,Nz)
    for ix in nb.prange(1,Nx-1):
        for iy in np.arange(1,Ny-1):
            for iz in np.arange(1,Nz-1):
                assert ~(((adj[ix,iy,iz] >> 0) & 1) ^ ((adj[ix+1,iy,iz] >> 1) & 1))
                assert ~(((adj[ix,iy,iz] >> 1) & 1) ^ ((adj[ix-1,iy,iz] >> 0) & 1))
                assert ~(((adj[ix,iy,iz] >> 2) & 1) ^ ((adj[ix,iy+1,iz] >> 3) & 1))
                assert ~(((adj[ix,iy,iz] >> 3) & 1) ^ ((adj[ix,iy-1,iz] >> 2) & 1))
                assert ~(((adj[ix,iy,iz] >> 4) & 1) ^ ((adj[ix,iy,iz+1] >> 5) & 1))
                assert ~(((adj[ix,iy,iz] >> 5) & 1) ^ ((adj[ix,iy,iz-1] >> 4) & 1))

@nb.jit(nopython=True,parallel=False)
def nb_check_adj_full_fcc(adj,Nx,Ny,Nz): 
    assert adj.shape == (Nx,Ny,Nz)
    for ix in nb.prange(1,Nx-1):
        for iy in np.arange(1,Ny-1):
            #for iz in np.arange(1,Nz-1):
                #if np.mod(ix+iy+iz,2)==1:
                    #continue
            for iz in np.arange(2-(ix+iy)%2,Nz-1):
                assert ~(((adj[ix,iy,iz] >> 0) & 1)  ^ ((adj[ix+1,iy+1,iz] >> 1) & 1))
                assert ~(((adj[ix,iy,iz] >> 1) & 1)  ^ ((adj[ix-1,iy-1,iz] >> 0) & 1))
                assert ~(((adj[ix,iy,iz] >> 2) & 1)  ^ ((adj[ix,iy+1,iz+1] >> 3) & 1))
                assert ~(((adj[ix,iy,iz] >> 3) & 1)  ^ ((adj[ix,iy+1,iz+1] >> 2) & 1))
                assert ~(((adj[ix,iy,iz] >> 4) & 1)  ^ ((adj[ix+1,iy,iz+1] >> 5) & 1))
                assert ~(((adj[ix,iy,iz] >> 5) & 1)  ^ ((adj[ix-1,iy,iz-1] >> 4) & 1))
                assert ~(((adj[ix,iy,iz] >> 6) & 1)  ^ ((adj[ix+1,iy-1,iz] >> 7) & 1))
                assert ~(((adj[ix,iy,iz] >> 7) & 1)  ^ ((adj[ix-1,iy+1,iz] >> 6) & 1))
                assert ~(((adj[ix,iy,iz] >> 8) & 1)  ^ ((adj[ix,iy+1,iz-1] >> 9) & 1))
                assert ~(((adj[ix,iy,iz] >> 9) & 1)  ^ ((adj[ix,iy-1,iz+1] >> 8) & 1))
                assert ~(((adj[ix,iy,iz] >> 10) & 1) ^ ((adj[ix+1,iy,iz-1] >> 11) & 1))
                assert ~(((adj[ix,iy,iz] >> 11) & 1) ^ ((adj[ix-1,iy,iz+1] >> 10) & 1))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str,help='json file to import')
    parser.add_argument('--Nvox_est', type=int,help='Nvox roughly')
    parser.add_argument('--draw', action='store_true',help='draw')
    parser.add_argument('--Nh', type=int,help='Nh')
    parser.add_argument('--h', type=float,help='h')
    parser.add_argument('--fcc', action='store_true',help='fcc grid')
    parser.add_argument('--offset', type=float,help='offset')
    parser.add_argument('--Nprocs', type=int,help='number of processes')
    #parser.add_argument('--draw_backend', type=str,help='mayavi or polyscope')
    parser.add_argument('--area_eps', type=float,help='for pruning degenerate triangels')
    parser.add_argument('--check_full', action='store_true',help='check whole adj')
    parser.add_argument('--save_folder', type=str,help='where to save')
    parser.add_argument('--az_el', nargs=2, type=float, help='two angles in deg')
    parser.add_argument('--polyscope', action='store_true',help='use polyscope backend')
    parser.set_defaults(draw=False)
    parser.set_defaults(fcc=False)
    #parser.set_defaults(draw_backend='mayavi')
    parser.set_defaults(polyscope=False)
    parser.set_defaults(Nvox_est=None)
    parser.set_defaults(Nprocs=get_default_nprocs())
    parser.set_defaults(offset=3.0)
    parser.set_defaults(area_eps=1.0e-10)
    parser.set_defaults(az_el=[0.,0.])
    parser.set_defaults(h=None)
    parser.set_defaults(Nh=None)
    parser.set_defaults(json=None)
    parser.set_defaults(check_full=False)
    parser.set_defaults(save_folder=None)
    args = parser.parse_args()
    print(args)
    assert args.Nprocs>0
    #assert args.Nvox_est is not None or args.Nh is not None 
    assert args.h is not None
    assert args.json is not None
    assert args.offset>2.0

    if args.polyscope:
        draw_backend='polyscope'
    else:
        draw_backend='mayavi'

    room_geo = RoomGeo(args.json,az_el=args.az_el,area_eps=args.area_eps)
    room_geo.print_stats()

    cart_grid = CartGrid(args.h,args.offset,room_geo.bmin,room_geo.bmax)
    cart_grid.print_stats()

    vox_grid = VoxGrid(room_geo,cart_grid,args.Nvox_est,args.Nh)
    vox_grid.fill(Nprocs=args.Nprocs)
    vox_grid.print_stats()

    vox_scene = VoxScene(room_geo,cart_grid,vox_grid,fcc=args.fcc)
    vox_scene.calc_adj(Nprocs=args.Nprocs)

    if args.check_full:
        vox_scene.check_adj_full()

    if args.save_folder:
        vox_scene.save(args.save_folder) 

    if args.draw:
        room_geo.draw(wireframe=False,backend=draw_backend)
        vox_scene.draw(backend=draw_backend)
        room_geo.show(backend=draw_backend)

if __name__ == '__main__':
    main()
