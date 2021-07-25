// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: gpu_engine.h
//
// Description: GPU-based implementation of FDTD engine (using CUDA).  
//
///////////////////////////////////////////////////////////////////////////////

#ifndef _GPU_ENGINE_H
#define _GPU_ENGINE_H
#ifndef _STDIO_H
#include <stdio.h>
#endif
#ifndef _STDLIB_H
#include <stdlib.h> //for malloc
#endif
#ifndef _STDINT_H
#include <stdint.h> 
#endif
#ifndef _ASSERT_H
#include <assert.h> //for assert
#endif
#ifndef _STDBOOL_H
#include <stdbool.h> //for bool
#endif
#ifndef _MATH_H
#include <math.h> //NAN
#endif
#ifndef _HELPER_FUNCS_H
#include <helper_funcs.h> 
#endif
#ifndef _FDTD_COMMON_H
#include <fdtd_common.h>
#endif
#ifndef _OMP_H
#include <omp.h>
#endif

#define CU_DIV_CEIL(x,y) ((DIV_CEIL(x,y)==0)? (1) : (DIV_CEIL(x,y))) //want 0 to map to 1, otherwise kernel errors

#if !USING_CUDA
#error
#endif

//thread-block dims for 3d kernels
#define cuBx 32
#define cuBy 2
#define cuBz 2

//thread-block dims for 2d kernels (fcc fold, ABCs)
#define cuBx2 16
#define cuBy2 8

//thread-block dims for 1d kernels (bn, ABC loss) 
#define cuBrw 128
#define cuBb 128

//constant memory (all per device)
__constant__ Real c1;
__constant__ Real c2;
__constant__ Real cl;
__constant__ Real csl2;
__constant__ Real clo2;
__constant__ int64_t cuNx; 
__constant__ int64_t cuNy; 
__constant__ int64_t cuNz; 
__constant__ int64_t cuNb; 
__constant__ int64_t cuNbl;
__constant__ int64_t cuNba; 
__constant__ int64_t cuNxNy; 
__constant__ int8_t cuMb[MNm]; //to store Mb per mat (MNm has to be hash-defined)

uint64_t print_gpu_details(int i);
void check_sorted(const struct SimData *sd);
void split_data(const struct SimData *sd, struct gpuHostData *ghds, int ngpus);
double run_sim(const struct SimData *sd);
//CUDA kernels 
__global__ void KernelAirCart(Real * __restrict__ u0, const Real * __restrict__ u1, const uint8_t * __restrict__ bn_mask);
__global__ void KernelAirFCC(Real * __restrict__ u0, const Real * __restrict__ u1, const uint8_t * __restrict__ bn_mask);
__global__ void KernelFoldFCC(Real * __restrict__ u1);
__global__ void KernelBoundaryRigidCart(Real * __restrict__ u0, const Real * __restrict__ u1,  
                                            const uint16_t * __restrict__ adj_bn,
                                            const int64_t * __restrict__ bn_ixyz, 
                                            const int8_t * __restrict__ K_bn);
__global__ void KernelBoundaryRigidFCC(Real * __restrict__ u0, const Real * __restrict__ u1,  
                                            const uint16_t * __restrict__ adj_bn,
                                            const int64_t * __restrict__ bn_ixyz, 
                                            const int8_t * __restrict__ K_bn);
__global__ void KernelBoundaryABC(Real * __restrict__ u0, 
                                  const Real * __restrict__ u2ba,
                                  const int8_t * __restrict__ Q_bna,
                                  const int64_t *  __restrict__ bna_ixyz);
__global__ void KernelBoundaryFD(Real * __restrict__ u0b, const Real *u2b,
                                  Real * __restrict__ vh1, Real * __restrict__ gh1,
                                  const Real * ssaf_bnl, const int8_t * mat_bnl,
                                  const Real * __restrict__ mat_beta, const struct MatQuad * __restrict__ mat_quads);
__global__ void AddIn(Real *u0, Real sample);
__global__ void CopyToGridKernel(Real *u, const Real *buffer,  const int64_t *locs, int64_t N);
__global__ void CopyFromGridKernel(Real *buffer, const Real *u,  const int64_t *locs, int64_t N);
__global__ void FlipHaloXY_Zbeg(Real * __restrict__ u1);
__global__ void FlipHaloXY_Zend(Real * __restrict__ u1);
__global__ void FlipHaloXZ_Ybeg(Real * __restrict__ u1);
__global__ void FlipHaloXZ_Yend(Real * __restrict__ u1);
__global__ void FlipHaloYZ_Xbeg(Real * __restrict__ u1);
__global__ void FlipHaloYZ_Xend(Real * __restrict__ u1);

//this is data on host, sometimes copied and recomputed for copy to GPU devices (indices), sometimes just aliased pointers (scalar arrays)
struct gpuHostData { //arrays on host (for copy), mirrors gpu local data 
   double *in_sigs; //aliased
   Real *u_out_buf; //aliased
   double *u_out; //aliased 
   Real *ssaf_bnl; //aliased
   int64_t *in_ixyz; //recomputed
   int64_t *out_ixyz; //recomputed
   int64_t *bn_ixyz; //recomputed
   int64_t *bnl_ixyz; //recomputed
   int64_t *bna_ixyz; //recomputed
   int8_t *Q_bna; //aliased
   uint16_t *adj_bn; //aliased
   int8_t   *mat_bnl; //aliased
   uint8_t  *bn_mask; //recomputed
   int8_t  *K_bn; //aliased
   int64_t Ns;
   int64_t Nr;
   int64_t Npts;
   int64_t Nx;
   int64_t Nxh;
   int64_t Nb;
   int64_t Nbl;
   int64_t Nba;
   int64_t Nbm; //bytes for bn_mask
};

//these are arrays pointing to GPU device memory, or CUDA stuff (dim3, events)
struct gpuData { //for or on gpu (arrays all on GPU)
   int64_t *bn_ixyz;
   int64_t *bnl_ixyz;
   int64_t *bna_ixyz;
   int8_t *Q_bna;
   int64_t *out_ixyz;
   uint16_t *adj_bn;
   Real *ssaf_bnl; 
   uint8_t  *bn_mask;
   int8_t   *mat_bnl;
   int8_t  *K_bn;
   Real  *mat_beta;
   struct MatQuad *mat_quads;
   Real *u0;
   Real *u1;
   Real *u0b; 
   Real *u1b; 
   Real *u2b; 
   Real *u2ba; 
   Real *vh1; 
   Real *gh1; 
   Real *u_out_buf;
   dim3 block_dim_air;
   dim3 grid_dim_air;
   dim3 block_dim_fold;
   dim3 grid_dim_fold;
   dim3 block_dim_readout;
   dim3 grid_dim_readout; 
   dim3 block_dim_bn;
   dim3 block_dim_halo_xy;
   dim3 block_dim_halo_yz;
   dim3 block_dim_halo_xz;
   dim3 grid_dim_bn; 
   dim3 grid_dim_bnl; 
   dim3 grid_dim_bna; 
   dim3 grid_dim_halo_xy; 
   dim3 grid_dim_halo_yz; 
   dim3 grid_dim_halo_xz; 
   cudaStream_t cuStream_air;
   cudaStream_t cuStream_bn;
   cudaEvent_t cuEv_air_start;
   cudaEvent_t cuEv_air_end;
   cudaEvent_t cuEv_bn_roundtrip_start;
   cudaEvent_t cuEv_bn_roundtrip_end;
   cudaEvent_t cuEv_readout_end;
   int64_t totalmembytes;
};


//standard error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//print some device details
uint64_t print_gpu_details(int i) {
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, i);
   printf("\nDevice Number: %d [%s]\n", i, prop.name);
   printf("  Compute: %d.%d\n",prop.major,prop.minor);
   printf("  Peak Memory Bandwidth: %.3f GB/s\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
   printf("  Total global memory: [ %.3f GB | %.3f GiB | %lu MiB ]\n", (double)prop.totalGlobalMem/(1e9), (double)prop.totalGlobalMem/1073741824ULL, prop.totalGlobalMem>>20);
   printf("  Registers per block: %d\n", prop.regsPerBlock);
   printf("  Concurrent Kernels: %d\n", prop.concurrentKernels);
   printf("  Async Engine: %d\n", prop.asyncEngineCount);
   printf("\n");
   return prop.totalGlobalMem;
}

//NB. 'x' is contiguous dim in CUDA domain

//vanilla scheme, unrolled, intrinsics to control rounding errors
__global__ void KernelAirCart(Real * __restrict__ u0, const Real * __restrict__ u1,  
                                            const uint8_t * __restrict__ bn_mask)
{
   int64_t cx = blockIdx.x*cuBx + threadIdx.x + 1;
   int64_t cy = blockIdx.y*cuBy + threadIdx.y + 1;
   int64_t cz = blockIdx.z*cuBz + threadIdx.z + 1;
   if ((cx<cuNx-1) && (cy<cuNy-1) && (cz<cuNz-1)) {
      int64_t ii = cz*cuNxNy + cy*cuNx + cx;
      //divide-conquer add for better accuracy
      Real tmp1,tmp2;
      tmp1 = ADD_O(u1[ii + cuNxNy],u1[ii - cuNxNy]);
      tmp2 = ADD_O(u1[ii + cuNx],u1[ii - cuNx]);
      tmp1 = ADD_O(tmp1,tmp2);
      tmp2 = ADD_O(u1[ii + 1],u1[ii - 1]);
      tmp1 = ADD_O(tmp1,tmp2); 
      tmp1 = FMA_D(c1,u1[ii],FMA_D(c2,tmp1,-u0[ii]));

      //write final value back to global memory
      if ( !(GET_BIT(bn_mask[ii>>3],ii%8)) ) { 
         u0[ii] = tmp1;
      }
   }
}

//air update for FCC, on folded grid (improvement to 2013 DAFx paper) 
__global__ void KernelAirFCC(Real * __restrict__ u0, const Real * __restrict__ u1,  
                                        const uint8_t * __restrict__ bn_mask)
{
   // get ix,iy,iz from thread and block Id's
   int64_t cx = blockIdx.x*cuBx + threadIdx.x + 1;
   int64_t cy = blockIdx.y*cuBy + threadIdx.y + 1;
   int64_t cz = blockIdx.z*cuBz + threadIdx.z + 1;
   if ((cx<cuNx-1) && (cy<cuNy-1) && (cz<cuNz-1)) {
      //x is contiguous
      int64_t ii = cz*cuNxNy + cy*cuNx + cx;
      Real tmp1,tmp2,tmp3,tmp4;
      //divide-conquer add as much as possible
      tmp1 = ADD_O(u1[ii + cuNxNy + cuNx],u1[ii - cuNxNy - cuNx]);
      tmp2 = ADD_O(u1[ii + cuNx + 1],u1[ii - cuNx - 1]);
      tmp1 = ADD_O(tmp1,tmp2);
      tmp3 = ADD_O(u1[ii + cuNxNy + 1],u1[ii - cuNxNy - 1]);
      tmp4 = ADD_O(u1[ii + cuNxNy - cuNx],u1[ii - cuNxNy + cuNx]);
      tmp3 = ADD_O(tmp3,tmp4);
      tmp2 = ADD_O(u1[ii + cuNx - 1],u1[ii - cuNx + 1]);
      tmp1 = ADD_O(tmp1,tmp2);
      tmp4 = ADD_O(u1[ii + cuNxNy - 1],u1[ii - cuNxNy + 1]);
      tmp3 = ADD_O(tmp3,tmp4);
      tmp1 = ADD_O(tmp1,tmp3);
      tmp1 = FMA_D(c1,u1[ii],FMA_D(c2,tmp1,-u0[ii]));
      //write final value back to global memory
      if ( !(GET_BIT(bn_mask[ii>>3],ii%8)) ) { 
         u0[ii] = tmp1;
      }
   }
}

//this folds in half of FCC subgrid so everything is nicely homogenous (no braching for stencil)
__global__ void KernelFoldFCC(Real * __restrict__ u1)
{
   int64_t cx = blockIdx.x*cuBx2 + threadIdx.x; 
   int64_t cz = blockIdx.y*cuBy2 + threadIdx.y;
   //fold is along middle dimension
   if ((cx<cuNx) && (cz<cuNz)) {
      u1[cz*cuNxNy + (cuNy-1)*cuNx + cx] = u1[cz*cuNxNy + (cuNy-2)*cuNx + cx];
   }
}

//rigid boundaries, cartesian, using adj info
__global__ void KernelBoundaryRigidCart(Real * __restrict__ u0, const Real * __restrict__ u1,  
                                            const uint16_t * __restrict__ adj_bn,
                                            const int64_t * __restrict__ bn_ixyz, 
                                            const int8_t * __restrict__ K_bn)
{
   int64_t nb = blockIdx.x*cuBb + threadIdx.x;
   if (nb<cuNb) {
      int64_t ii = bn_ixyz[nb];
      uint16_t adj = adj_bn[nb];
      Real K = K_bn[nb];

      Real _2 = 2.0;
      Real b1 = (_2-csl2*K);
      Real b2 = c2;

      Real tmp1,tmp2;
      tmp1 = ADD_O((Real)GET_BIT(adj,0)*u1[ii + cuNxNy],(Real)GET_BIT(adj,1)*u1[ii - cuNxNy]);
      tmp2 = ADD_O((Real)GET_BIT(adj,2)*u1[ii + cuNx],  (Real)GET_BIT(adj,3)*u1[ii - cuNx]);
      tmp1 = ADD_O(tmp1,tmp2);
      tmp2 = ADD_O((Real)GET_BIT(adj,4)*u1[ii + 1],     (Real)GET_BIT(adj,5)*u1[ii - 1]);
      tmp1 = ADD_O(tmp1,tmp2); 
      tmp1 = FMA_D(b1,u1[ii],FMA_D(b2,tmp1,-u0[ii]));

      //u0[ii] = partial; //write back to global memory
      u0[ii] = tmp1; //write back to global memory
   }
}

//rigid boundaries, FCC, using adj info
__global__ void KernelBoundaryRigidFCC(Real * __restrict__ u0, const Real * __restrict__ u1,  
                                            const uint16_t * __restrict__ adj_bn,
                                            const int64_t * __restrict__ bn_ixyz, 
                                            const int8_t * __restrict__ K_bn)
{
   int64_t nb = blockIdx.x*cuBb + threadIdx.x;
   if (nb<cuNb) {
      int64_t ii = bn_ixyz[nb];
      uint16_t adj = adj_bn[nb];
      Real K = K_bn[nb];

      Real _2 = 2.0;
      Real b1 = (_2-csl2*K);
      Real b2 = c2;

      Real tmp1,tmp2,tmp3,tmp4;
      tmp1 = ADD_O((Real)GET_BIT(adj,0)*u1[ii + cuNxNy + cuNx],(Real)GET_BIT(adj,1)*u1[ii - cuNxNy - cuNx]);
      tmp2 = ADD_O((Real)GET_BIT(adj,2)*u1[ii + cuNx + 1],     (Real)GET_BIT(adj,3)*u1[ii - cuNx - 1]);
      tmp1 = ADD_O(tmp1,tmp2);
      tmp3 = ADD_O((Real)GET_BIT(adj,4)*u1[ii + cuNxNy + 1],   (Real)GET_BIT(adj,5)*u1[ii - cuNxNy - 1]);
      tmp4 = ADD_O((Real)GET_BIT(adj,6)*u1[ii + cuNxNy - cuNx],(Real)GET_BIT(adj,7)*u1[ii - cuNxNy + cuNx]);
      tmp3 = ADD_O(tmp3,tmp4);
      tmp2 = ADD_O((Real)GET_BIT(adj,8)*u1[ii + cuNx - 1],     (Real)GET_BIT(adj,9)*u1[ii - cuNx + 1]);
      tmp1 = ADD_O(tmp1,tmp2);
      tmp4 = ADD_O((Real)GET_BIT(adj,10)*u1[ii + cuNxNy - 1],  (Real)GET_BIT(adj,11)*u1[ii - cuNxNy + 1]);
      tmp3 = ADD_O(tmp3,tmp4);
      tmp1 = ADD_O(tmp1,tmp3);
      tmp1 = FMA_D(b1,u1[ii],FMA_D(b2,tmp1,-u0[ii]));

      u0[ii] = tmp1; //write back to global memory
   }
}

//ABC loss at boundaries of simulation grid
__global__ void KernelBoundaryABC(Real * __restrict__ u0, 
                                  const Real * __restrict__ u2ba,
                                  const int8_t * __restrict__ Q_bna,
                                  const int64_t *  __restrict__ bna_ixyz)
{
   int64_t nb = blockIdx.x*cuBb + threadIdx.x;
   if (nb<cuNba) {
      Real _1 = 1.0;
      Real lQ = cl*Q_bna[nb];
      int64_t ib = bna_ixyz[nb];
      Real partial = u0[ib];
      partial = ( partial + lQ*u2ba[nb] )/(_1 + lQ );
      u0[ib] = partial;
   }
}

//Part of freq-dep boundary update 
__global__ void KernelBoundaryFD(Real * __restrict__ u0b, const Real *u2b,
                                  Real * __restrict__ vh1, Real * __restrict__ gh1,
                                  const Real * ssaf_bnl, const int8_t * mat_bnl,
                                  const Real * __restrict__ mat_beta, const struct MatQuad * __restrict__ mat_quads)
{  
   int64_t nb = blockIdx.x*cuBb + threadIdx.x;
   if (nb<cuNbl) {
      Real _1 = 1.0;
      Real _2 = 2.0;
      int32_t k = mat_bnl[nb];
      Real ssaf = ssaf_bnl[nb];
      Real lo2Kbg = clo2*ssaf*mat_beta[k];
      Real fac = _2*clo2*ssaf / (_1 + lo2Kbg);

      Real u0bint = u0b[nb];
      Real u2bint = u2b[nb];

      u0bint = (u0bint + lo2Kbg*u2bint) / (_1 + lo2Kbg);

      Real vh1int[MMb]; //size has to be constant at compile time
      Real gh1int[MMb];
      for (int8_t m=0; m<cuMb[k]; m++) { //faster on average than MMb
         int64_t nbm = m*cuNbl + nb;
         int32_t mbk = k*MMb+m;
         const struct MatQuad *tm;
         tm = &(mat_quads[mbk]);
         vh1int[m] = vh1[nbm];
         gh1int[m] = gh1[nbm];
         u0bint -= fac*( _2*(tm->bDh)*vh1int[m] - (tm->bFh)*gh1int[m] );
      }

      Real du = u0bint-u2bint;

      for (int8_t m=0; m<cuMb[k]; m++) { //faster on average than MMb
         int64_t nbm = m*cuNbl + nb;
         int32_t mbk = k*MMb+m;
         const struct MatQuad *tm; 
         tm = &(mat_quads[mbk]);
         Real vh0m = (tm->b)*du + (tm->bd)*vh1int[m] - _2*(tm->bFh)*gh1int[m];
         gh1[nbm] = gh1int[m] + (vh0m + vh1int[m])/_2;
         vh1[nbm] = vh0m;
      }
      u0b[nb] = u0bint;
   }
}

//add source input (one at a time for simplicity)
__global__ void AddIn(Real *u0, Real sample)
{
   u0[0] += sample;
}

//dst-src copy from buffer to grid
__global__ void CopyToGridKernel(Real *u, const Real *buffer,  const int64_t *locs, int64_t N)
{
   int64_t i = blockIdx.x*cuBrw + threadIdx.x;
   if (i<N) u[locs[i]] = buffer[i];
}

//dst-src copy to buffer from  grid (not needed, but to make more explicit)
__global__ void CopyFromGridKernel(Real *buffer, const Real *u,  const int64_t *locs, int64_t N)
{
   int64_t i = blockIdx.x*cuBrw + threadIdx.x;
   if (i<N) buffer[i] = u[locs[i]];
}

//flip halos for ABCs
__global__ void FlipHaloXY_Zbeg(Real * __restrict__ u1)
{
   int64_t cx = blockIdx.x*cuBx2 + threadIdx.x;
   int64_t cy = blockIdx.y*cuBy2 + threadIdx.y;
   if ((cx<cuNx) && (cy<cuNy)) {
      int64_t ii;
      ii = 0*cuNxNy + cy*cuNx + cx;
      u1[ii] = u1[ii+2*cuNxNy];
   }
}
__global__ void FlipHaloXY_Zend(Real * __restrict__ u1)
{
   int64_t cx = blockIdx.x*cuBx2 + threadIdx.x;
   int64_t cy = blockIdx.y*cuBy2 + threadIdx.y;
   if ((cx<cuNx) && (cy<cuNy)) {
      int64_t ii;
      ii = (cuNz-1)*cuNxNy + cy*cuNx + cx;
      u1[ii] = u1[ii-2*cuNxNy];
   }
}
__global__ void FlipHaloXZ_Ybeg(Real * __restrict__ u1)
{
   int64_t cx = blockIdx.x*cuBx2 + threadIdx.x;
   int64_t cz = blockIdx.y*cuBy2 + threadIdx.y;
   if ((cx<cuNx) && (cz<cuNz)) {
      int64_t ii;
      ii = cz*cuNxNy + 0*cuNx + cx;
      u1[ii] = u1[ii+2*cuNx];
   }
}
__global__ void FlipHaloXZ_Yend(Real * __restrict__ u1)
{
   int64_t cx = blockIdx.x*cuBx2 + threadIdx.x;
   int64_t cz = blockIdx.y*cuBy2 + threadIdx.y;
   if ((cx<cuNx) && (cz<cuNz)) {
      int64_t ii;
      ii = cz*cuNxNy + (cuNy-1)*cuNx + cx;
      u1[ii] = u1[ii-2*cuNx];
   }
}
__global__ void FlipHaloYZ_Xbeg(Real * __restrict__ u1)
{
   int64_t cy = blockIdx.x*cuBx2 + threadIdx.x;
   int64_t cz = blockIdx.y*cuBy2 + threadIdx.y;
   if ((cy<cuNy) && (cz<cuNz)) {
      int64_t ii;
      ii = cz*cuNxNy + cy*cuNx + 0;
      u1[ii] = u1[ii+2];
   }
}
__global__ void FlipHaloYZ_Xend(Real * __restrict__ u1)
{
   int64_t cy = blockIdx.x*cuBx2 + threadIdx.x;
   int64_t cz = blockIdx.y*cuBy2 + threadIdx.y;
   if ((cy<cuNy) && (cz<cuNz)) {
      int64_t ii;
      ii = cz*cuNxNy + cy*cuNx + (cuNx-1);
      u1[ii] = u1[ii-2];
   }
}

//input indices need to be sorted for multi-device allocation
void check_sorted(const struct SimData *sd) {
   int64_t *bn_ixyz = sd->bn_ixyz;
   int64_t *bnl_ixyz = sd->bnl_ixyz;
   int64_t *bna_ixyz = sd->bna_ixyz;
   int64_t *in_ixyz = sd->in_ixyz;
   int64_t *out_ixyz = sd->out_ixyz;
   int64_t Nb = sd->Nb;
   int64_t Nbl = sd->Nbl;
   int64_t Nba = sd->Nba;
   int64_t Ns = sd->Ns;
   int64_t Nr = sd->Nr;
   for (int64_t i=1; i<Nb; i++) assert (bn_ixyz[i] > bn_ixyz[i-1]); //check save_gpu_folder
   for (int64_t i=1; i<Nbl; i++) assert (bnl_ixyz[i] > bnl_ixyz[i-1]);
   for (int64_t i=1; i<Nba; i++) assert (bna_ixyz[i] > bna_ixyz[i-1]);
   for (int64_t i=1; i<Ns; i++) assert (in_ixyz[i] > in_ixyz[i-1]);
   for (int64_t i=1; i<Nr; i++) assert (out_ixyz[i] >= out_ixyz[i-1]); //possible to have duplicates
}

//counts for splitting data across GPUs 
void split_data(const struct SimData *sd, struct gpuHostData *ghds, int ngpus) {
   int64_t Nx = sd->Nx;
   int64_t Ny = sd->Ny;
   int64_t Nz = sd->Nz;
   struct gpuHostData *ghd;
   //initialise
   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      ghd->Nx = 0;
      ghd->Nb = 0;
      ghd->Nbl = 0;
      ghd->Nba = 0;
      ghd->Ns = 0;
      ghd->Nr = 0;
   }

   //split Nx layers (Nz contiguous)
   int64_t Nxm = Nx/ngpus;
   int64_t Nxl = Nx % ngpus;

   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      ghd->Nx = Nxm;
   }
   for (int gid=0; gid<Nxl; gid++)  {
      ghd = &ghds[gid];
      ghd->Nx += 1;
   }
   int64_t Nx_check = 0;
   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      printf("gid=%d, Nx[%d]=%ld, Nx=%ld\n",gid,gid,ghd->Nx,Nx);
      Nx_check += ghd->Nx;
   }
   assert(Nx_check==Nx);

   //now count Nr,Ns,Nb for each device
   int64_t Nxcc[ngpus];
   Nxcc[0] = ghds[0].Nx;
   printf("Nxcc[%d]=%ld\n",0,Nxcc[0]);
   for (int gid=1; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      Nxcc[gid] = ghd->Nx + Nxcc[gid-1];
      printf("Nxcc[%d]=%ld\n",gid,Nxcc[gid]);
   }

   //bn_ixyz - Nb
   int64_t *bn_ixyz = sd->bn_ixyz;
   int64_t Nb = sd->Nb;
   {
      int gid=0;
      for (int64_t i=0; i<Nb; i++) { 
         while (bn_ixyz[i] >= Nxcc[gid]*Ny*Nz) {
            gid++;
         }
         (ghds[gid].Nb)++;
      }
   }
   int64_t Nb_check = 0;
   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      printf("gid=%d, Nb[%d]=%ld, Nb=%ld\n",gid,gid,ghd->Nb,Nb);
      Nb_check += ghd->Nb;
   }
   assert(Nb_check==Nb);

   //bnl_ixyz - Nbl
   int64_t *bnl_ixyz = sd->bnl_ixyz;
   int64_t Nbl = sd->Nbl;
   {
      int gid=0;
      for (int64_t i=0; i<Nbl; i++) { 
         while (bnl_ixyz[i] >= Nxcc[gid]*Ny*Nz) {
            gid++;
         }
         (ghds[gid].Nbl)++;
      }
   }
   int64_t Nbl_check = 0;
   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      printf("gid=%d, Nbl[%d]=%ld, Nbl=%ld\n",gid,gid,ghd->Nbl,Nbl);
      Nbl_check += ghd->Nbl;
   }
   assert(Nbl_check==Nbl);

   //bna_ixyz - Nba
   int64_t *bna_ixyz = sd->bna_ixyz;
   int64_t Nba = sd->Nba;
   {
      int gid=0;
      for (int64_t i=0; i<Nba; i++) { 
         while (bna_ixyz[i] >= Nxcc[gid]*Ny*Nz) {
            gid++;
         }
         (ghds[gid].Nba)++;
      }
   }
   int64_t Nba_check = 0;
   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      printf("gid=%d, Nba[%d]=%ld, Nbl=%ld\n",gid,gid,ghd->Nba,Nba);
      Nba_check += ghd->Nba;
   }
   assert(Nba_check==Nba);


   //in_ixyz - Ns
   int64_t *in_ixyz = sd->in_ixyz;
   int64_t Ns = sd->Ns;
   {
      int gid=0;
      for (int64_t i=0; i<Ns; i++) { 
         while (in_ixyz[i] >= Nxcc[gid]*Ny*Nz) {
            gid++;
         }
         (ghds[gid].Ns)++;
      }
   }
   int64_t Ns_check = 0;
   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      printf("gid=%d, Ns[%d]=%ld, Ns=%ld\n",gid,gid,ghd->Ns,Ns);
      Ns_check += ghd->Ns;
   }
   assert(Ns_check==Ns);

   //out_ixyz - Nr
   int64_t *out_ixyz = sd->out_ixyz;
   int64_t Nr = sd->Nr;
   {
      int gid=0;
      for (int64_t i=0; i<Nr; i++) { 
         while (out_ixyz[i] >= Nxcc[gid]*Ny*Nz) {
            gid++;
         }
         (ghds[gid].Nr)++;
      }
   }
   int64_t Nr_check = 0;
   for (int gid=0; gid<ngpus; gid++) {
      ghd = &ghds[gid];
      printf("gid=%d, Nr[%d]=%ld, Nr=%ld\n",gid,gid,ghd->Nr,Nr);
      Nr_check += ghd->Nr;
   }
   assert(Nr_check==Nr);
}

//run the sim!
double run_sim(const struct SimData *sd) 
{
   //if you want to test synchronous, env variable for that
   const char* s = getenv("CUDA_LAUNCH_BLOCKING");
   if (s != NULL) {
      if (s[0]=='1') {
         printf("******************SYNCHRONOUS (DEBUG ONLY!!!)*********************\n");
         printf("...continue?\n");
         getchar();
      }
   }

   assert((sd->fcc_flag != 1)); //uses either cartesian or FCC folded grid

   int ngpus,max_ngpus;
   cudaGetDeviceCount(&max_ngpus); //control outside with CUDA_VISIBLE_DEVICES
   ngpus = max_ngpus; 
   assert(ngpus < (sd->Nx));
   struct gpuData *gds;
   mymalloc((void **)&gds, ngpus*sizeof(gpuData)); 
   struct gpuHostData *ghds;
   mymalloc((void **)&ghds, ngpus*sizeof(gpuHostData)); //one bit per 

   if (ngpus>1) check_sorted(sd); //needs to be sorted for multi-GPU

   //get local counts for Nx,Nb,Nr,Ns
   split_data(sd, ghds, ngpus);

   for (int gid=0; gid < ngpus; gid++) {
      gds[gid].totalmembytes = print_gpu_details(gid); 
   }

   Real lo2 = sd->lo2;
   Real a1 = sd->a1;
   Real a2 = sd->a2;
   Real l = sd->l;
   Real sl2 = sd->sl2;

   //timing stuff
   double time_elapsed=0.0;
   double time_elapsed_bn=0.0;
   double time_elapsed_sample;
   double time_elapsed_sample_bn;
   double time_elapsed_air=0.0; //feed into print/process
   double time_elapsed_sample_air; //feed into print/process
   float millis_since_start;
   float millis_since_sample_start;

   printf("a1 = %.16g\n",a1);
   printf("a2 = %.16g\n",a2);

   //start moving data to GPUs
   for (int gid=0; gid < ngpus; gid++) {
      struct gpuHostData *ghd = &(ghds[gid]);
      printf("GPU %d -- ",gid);
      printf("Nx=%ld Ns=%ld Nr=%ld Nb=%ld Nbl=%ld Nba=%ld\n",ghd->Nx,ghd->Ns,ghd->Nr,ghd->Nb,ghd->Nbl,ghd->Nba);
   }

   int64_t Ns_read=0;
   int64_t Nr_read=0;
   int64_t Nb_read=0;
   int64_t Nbl_read=0;
   int64_t Nba_read=0;
   int64_t Nx_read=0;
   int64_t Nx_pos=0;
   //uint64_t Nx_pos2=0;

   Real *u_out_buf; 
   gpuErrchk( cudaMallocHost(&u_out_buf, (size_t)(sd->Nr*sizeof(Real))) );
   memset(u_out_buf, 0, (size_t)(sd->Nr*sizeof(Real))); //set floats to zero

   int64_t Nzy = (sd->Nz)*(sd->Ny); //area-slice 

   //here we recalculate indices to move to devices
   for (int gid=0; gid < ngpus; gid++) {
      gpuErrchk( cudaSetDevice(gid) );

      struct gpuData *gd = &(gds[gid]);
      struct gpuHostData *ghd = &(ghds[gid]);
      printf("---------\n");
      printf("GPU %d\n",gid);
      printf("---------\n");

      printf("Nx to read = %ld\n",ghd->Nx);
      printf("Nb to read = %ld\n",ghd->Nb);
      printf("Nbl to read = %ld\n",ghd->Nbl);
      printf("Nba to read = %ld\n",ghd->Nba);
      printf("Ns to read = %ld\n",ghd->Ns);
      printf("Nr to read = %ld\n",ghd->Nr);

      //Nxh (effective Nx with extra halos)
      ghd->Nxh = ghd->Nx;
      if (gid>0) (ghd->Nxh)++; //add bottom halo
      if (gid<ngpus-1) (ghd->Nxh)++; //add top halo
      //calculate Npts for this device
      ghd->Npts = Nzy*(ghd->Nxh);
      //boundary mask
      ghd->Nbm = CU_DIV_CEIL(ghd->Npts,8);

      printf("Nx=%ld Ns=%ld Nr=%ld Nb=%ld, Npts=%ld\n",ghd->Nx,ghd->Ns,ghd->Nr,ghd->Nb,ghd->Npts);

      //aliased pointers (to memory already allocated)
      ghd->in_sigs   = sd->in_sigs + Ns_read*sd->Nt;
      ghd->ssaf_bnl   = sd->ssaf_bnl + Nbl_read;
      ghd->adj_bn    = sd->adj_bn + Nb_read;
      ghd->mat_bnl   = sd->mat_bnl + Nbl_read;
      ghd->K_bn      = sd->K_bn + Nb_read;
      ghd->Q_bna     = sd->Q_bna + Nba_read;
      ghd->u_out     = sd->u_out + Nr_read*sd->Nt;
      ghd->u_out_buf = u_out_buf + Nr_read;

      //recalculate indices, these are associated host versions to copy over to devices
      mymalloc((void **)&(ghd->bn_ixyz), ghd->Nb*sizeof(int64_t));
      mymalloc((void **)&(ghd->bnl_ixyz), ghd->Nbl*sizeof(int64_t));
      mymalloc((void **)&(ghd->bna_ixyz), ghd->Nba*sizeof(int64_t));
      mymalloc((void **)&(ghd->bn_mask), ghd->Nbm*sizeof(uint8_t));
      mymalloc((void **)&(ghd->in_ixyz), ghd->Ns*sizeof(int64_t));
      mymalloc((void **)&(ghd->out_ixyz), ghd->Nr*sizeof(int64_t));

      int64_t offset = Nzy*Nx_pos;
      for (int64_t nb=0; nb<(ghd->Nb); nb++) {
         int64_t ii = sd->bn_ixyz[nb+Nb_read]; //global index
         int64_t jj = ii - offset; //local index
         assert(jj>=0);
         assert(jj < ghd->Npts);
         ghd->bn_ixyz[nb] = jj;
         SET_BIT_VAL(ghd->bn_mask[jj>>3],jj%8,GET_BIT(sd->bn_mask[ii>>3],ii%8)); //set bit
      }
      for (int64_t nb=0; nb<(ghd->Nbl); nb++) {
         int64_t ii = sd->bnl_ixyz[nb+Nbl_read]; //global index
         int64_t jj = ii - offset; //local index
         assert(jj>=0);
         assert(jj < ghd->Npts);
         ghd->bnl_ixyz[nb] = jj;
      }

      for (int64_t nb=0; nb<(ghd->Nba); nb++) {
         int64_t ii = sd->bna_ixyz[nb+Nba_read]; //global index
         int64_t jj = ii - offset; //local index
         assert(jj>=0);
         assert(jj < ghd->Npts);
         ghd->bna_ixyz[nb] = jj;
      }


      for (int64_t ns=0; ns<(ghd->Ns); ns++) {
         int64_t ii = sd->in_ixyz[ns+Ns_read];
         int64_t jj = ii - offset;
         assert(jj>=0);
         assert(jj < ghd->Npts);
         ghd->in_ixyz[ns] = jj;
      }
      for (int64_t nr=0; nr<(ghd->Nr); nr++) {
         int64_t ii = sd->out_ixyz[nr+Nr_read];
         int64_t jj = ii - offset;
         assert(jj>=0);
         assert(jj < ghd->Npts);
         ghd->out_ixyz[nr] = jj;
      }

      gpuErrchk( cudaMalloc(&(gd->u0), (size_t)((ghd->Npts)*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->u0, 0, (size_t)((ghd->Npts)*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->u1), (size_t)((ghd->Npts)*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->u1, 0, (size_t)((ghd->Npts)*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->K_bn), (size_t)(ghd->Nb*sizeof(int8_t))) );    
      gpuErrchk( cudaMemcpy(gd->K_bn, ghd->K_bn, ghd->Nb*sizeof(int8_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->ssaf_bnl), (size_t)(ghd->Nbl*sizeof(Real))) );    
      gpuErrchk( cudaMemcpy(gd->ssaf_bnl, ghd->ssaf_bnl, ghd->Nbl*sizeof(Real), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->u0b), (size_t)(ghd->Nbl*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->u0b, 0, (size_t)(ghd->Nbl*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->u1b), (size_t)(ghd->Nbl*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->u1b, 0, (size_t)(ghd->Nbl*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->u2b), (size_t)(ghd->Nbl*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->u2b, 0, (size_t)(ghd->Nbl*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->u2ba), (size_t)(ghd->Nba*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->u2ba, 0, (size_t)(ghd->Nba*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->vh1), (size_t)(ghd->Nbl*MMb*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->vh1, 0, (size_t)(ghd->Nbl*MMb*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->gh1), (size_t)(ghd->Nbl*MMb*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->gh1, 0, (size_t)(ghd->Nbl*MMb*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->u_out_buf), (size_t)(ghd->Nr*sizeof(Real))) );
      gpuErrchk( cudaMemset(gd->u_out_buf, 0, (size_t)(ghd->Nr*sizeof(Real))) );

      gpuErrchk( cudaMalloc(&(gd->bn_ixyz), (size_t)(ghd->Nb*sizeof(int64_t))) );
      gpuErrchk( cudaMemcpy(gd->bn_ixyz, ghd->bn_ixyz, (size_t)ghd->Nb*sizeof(int64_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->bnl_ixyz), (size_t)(ghd->Nbl*sizeof(int64_t))) );
      gpuErrchk( cudaMemcpy(gd->bnl_ixyz, ghd->bnl_ixyz, (size_t)ghd->Nbl*sizeof(int64_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->bna_ixyz), (size_t)(ghd->Nba*sizeof(int64_t))) );
      gpuErrchk( cudaMemcpy(gd->bna_ixyz, ghd->bna_ixyz, (size_t)ghd->Nba*sizeof(int64_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->Q_bna), (size_t)(ghd->Nba*sizeof(int8_t))) );    
      gpuErrchk( cudaMemcpy(gd->Q_bna, ghd->Q_bna, ghd->Nba*sizeof(int8_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->out_ixyz), (size_t)(ghd->Nr*sizeof(int64_t))) );
      gpuErrchk( cudaMemcpy(gd->out_ixyz, ghd->out_ixyz, (size_t)ghd->Nr*sizeof(int64_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->adj_bn), (size_t)(ghd->Nb*sizeof(uint16_t))) );    
      gpuErrchk( cudaMemcpy(gd->adj_bn, ghd->adj_bn, (size_t)ghd->Nb*sizeof(uint16_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->mat_bnl), (size_t)(ghd->Nbl*sizeof(int8_t))) );    
      gpuErrchk( cudaMemcpy(gd->mat_bnl, ghd->mat_bnl, (size_t)ghd->Nbl*sizeof(int8_t), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->mat_beta), (size_t)sd->Nm*sizeof(Real)) );    
      gpuErrchk( cudaMemcpy(gd->mat_beta, sd->mat_beta, (size_t)sd->Nm*sizeof(Real), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->mat_quads), (size_t)sd->Nm*MMb*sizeof(struct MatQuad)) );    
      gpuErrchk( cudaMemcpy(gd->mat_quads, sd->mat_quads, (size_t)sd->Nm*MMb*sizeof(struct MatQuad), cudaMemcpyHostToDevice) );

      gpuErrchk( cudaMalloc(&(gd->bn_mask), (size_t)(ghd->Nbm*sizeof(uint8_t))) );    
      gpuErrchk( cudaMemcpy(gd->bn_mask, ghd->bn_mask, (size_t)ghd->Nbm*sizeof(uint8_t), cudaMemcpyHostToDevice) );

      Ns_read += ghd->Ns;
      Nr_read += ghd->Nr;
      Nb_read += ghd->Nb;
      Nbl_read += ghd->Nbl;
      Nba_read += ghd->Nba;
      Nx_read += ghd->Nx; 
      Nx_pos = Nx_read-1; //back up one at subsequent stage

      printf("Nx_read = %ld\n",Nx_read);
      printf("Nb_read = %ld\n",Nb_read);
      printf("Nbl_read = %ld\n",Nbl_read);
      printf("Ns_read = %ld\n",Ns_read);
      printf("Nr_read = %ld\n",Nr_read);

      printf("Global memory allocation done\n");
      printf("\n");

      //swapping x and z here (CUDA has first dim contiguous)
      gpuErrchk( cudaMemcpyToSymbol(cuNx,&(sd->Nz),sizeof(int64_t)) );
      gpuErrchk( cudaMemcpyToSymbol(cuNy,&(sd->Ny),sizeof(int64_t)) );
      gpuErrchk( cudaMemcpyToSymbol(cuNz,&(ghd->Nxh),sizeof(int64_t)) );
      gpuErrchk( cudaMemcpyToSymbol(cuNb,&(ghd->Nb),sizeof(int64_t)) );
      gpuErrchk( cudaMemcpyToSymbol(cuNbl,&(ghd->Nbl),sizeof(int64_t)) );
      gpuErrchk( cudaMemcpyToSymbol(cuNba,&(ghd->Nba),sizeof(int64_t)) );
      gpuErrchk( cudaMemcpyToSymbol(cuMb,sd->Mb,sd->Nm*sizeof(int8_t)) );
      gpuErrchk( cudaMemcpyToSymbol(cuNxNy,&Nzy,sizeof(int64_t)) ); //same for all devices

      gpuErrchk( cudaMemcpyToSymbol(c1,&a1,sizeof(Real)) );
      gpuErrchk( cudaMemcpyToSymbol(c2,&a2,sizeof(Real)) );
      gpuErrchk( cudaMemcpyToSymbol(cl,&l,sizeof(Real)) );
      gpuErrchk( cudaMemcpyToSymbol(csl2,&sl2,sizeof(Real)) );
      gpuErrchk( cudaMemcpyToSymbol(clo2,&lo2,sizeof(Real)) );

      printf("Constant memory loaded\n");
      printf("\n");

      //threads grids and blocks (swap x and z)
      int64_t cuGx = CU_DIV_CEIL(sd->Nz-2,cuBx);
      int64_t cuGy = CU_DIV_CEIL(sd->Ny-2,cuBy);
      int64_t cuGz = CU_DIV_CEIL(ghd->Nxh-2,cuBz); 
      int64_t cuGr = CU_DIV_CEIL(ghd->Nr,cuBrw); 
      int64_t cuGb = CU_DIV_CEIL(ghd->Nb,cuBb);
      int64_t cuGbl = CU_DIV_CEIL(ghd->Nbl,cuBb);
      int64_t cuGba = CU_DIV_CEIL(ghd->Nba,cuBb);

      int64_t cuGx2 = CU_DIV_CEIL(sd->Nz,cuBx2); //full face
      int64_t cuGz2 = CU_DIV_CEIL(ghd->Nxh,cuBy2);  //full face

      assert(cuGx >= 1);
      assert(cuGy >= 1);
      assert(cuGz >= 1);
      assert(cuGr >= 1);
      assert(cuGb >= 1);
      assert(cuGbl >= 1);
      assert(cuGba >= 1);

      gd->block_dim_air     = dim3(cuBx, cuBy, cuBz);
      gd->block_dim_readout = dim3(cuBrw, 1, 1);
      gd->block_dim_bn      = dim3(cuBb, 1, 1);

      gd->grid_dim_air      = dim3(cuGx, cuGy, cuGz);
      gd->grid_dim_readout  = dim3(cuGr, 1, 1); 
      gd->grid_dim_bn       = dim3(cuGb, 1, 1); 
      gd->grid_dim_bnl       = dim3(cuGbl, 1, 1); 
      gd->grid_dim_bna       = dim3(cuGba, 1, 1); 

      gd->block_dim_halo_xy = dim3(cuBx2, cuBy2, 1);
      gd->block_dim_halo_yz = dim3(cuBx2, cuBy2, 1);
      gd->block_dim_halo_xz = dim3(cuBx2, cuBy2, 1);
      gd->grid_dim_halo_xy  = dim3(CU_DIV_CEIL(sd->Nz,cuBx2), CU_DIV_CEIL(sd->Ny,cuBy2), 1);
      gd->grid_dim_halo_yz  = dim3(CU_DIV_CEIL(sd->Ny,cuBx2), CU_DIV_CEIL(ghd->Nxh,cuBy2), 1);
      gd->grid_dim_halo_xz  = dim3(CU_DIV_CEIL(sd->Nz,cuBx2), CU_DIV_CEIL(ghd->Nxh,cuBy2), 1);

      gd->block_dim_fold     = dim3(cuBx2,cuBy2,1);
      gd->grid_dim_fold      = dim3(cuGx2,cuGz2,1);

      //create streams
      gpuErrchk( cudaStreamCreate(&(gd->cuStream_air)) );
      gpuErrchk( cudaStreamCreate(&(gd->cuStream_bn)) ); //no priority

      //cuda events
      gpuErrchk( cudaEventCreate(&(gd->cuEv_air_start)) );
      gpuErrchk( cudaEventCreate(&(gd->cuEv_air_end)) );
      gpuErrchk( cudaEventCreate(&(gd->cuEv_bn_roundtrip_start)) );
      gpuErrchk( cudaEventCreate(&(gd->cuEv_bn_roundtrip_end)) );
      gpuErrchk( cudaEventCreate(&(gd->cuEv_readout_end)) );
   }
   assert(Nb_read == sd->Nb);
   assert(Nbl_read == sd->Nbl);
   assert(Nba_read == sd->Nba);
   assert(Nr_read == sd->Nr);
   assert(Ns_read == sd->Ns);
   assert(Nx_read == sd->Nx);

   //these will be on first GPU only
   cudaEvent_t cuEv_main_start;
   cudaEvent_t cuEv_main_end;
   cudaEvent_t cuEv_main_sample_start;
   cudaEvent_t cuEv_main_sample_end;
   gpuErrchk( cudaSetDevice(0) );
   gpuErrchk( cudaEventCreate(&cuEv_main_start) );
   gpuErrchk( cudaEventCreate(&cuEv_main_end) );
   gpuErrchk( cudaEventCreate(&cuEv_main_sample_start) );
   gpuErrchk( cudaEventCreate(&cuEv_main_sample_end) );

   for (int64_t n=0; n<sd->Nt; n++) { //loop over time-steps
      for (int gid=0; gid < ngpus; gid++) { //loop over GPUs (one thread launches all kernels)
         gpuErrchk( cudaSetDevice(gid) );
         struct gpuData *gd = &(gds[gid]); //get struct of device pointers
         struct gpuHostData *ghd = &(ghds[gid]);//get struct of host points (corresponding to device)

         //start first timer
         if (gid==0) {
            if (n==0) gpuErrchk( cudaEventRecord(cuEv_main_start,0) ); //not sure if to put on stream, check slides again
            gpuErrchk( cudaEventRecord(cuEv_main_sample_start,0) );
         }
         //boundary updates (using intermediate buffer)
         gpuErrchk( cudaEventRecord(gd->cuEv_bn_roundtrip_start,gd->cuStream_bn) );

         //boundary updates
         if (sd->fcc_flag==0) {
            KernelBoundaryRigidCart<<<gd->grid_dim_bn,gd->block_dim_bn,0,gd->cuStream_bn>>>(gd->u0,gd->u1,gd->adj_bn,gd->bn_ixyz,gd->K_bn);
         }
         else {
            KernelFoldFCC<<<gd->grid_dim_fold,gd->block_dim_fold,0,gd->cuStream_bn>>>(gd->u1);
            KernelBoundaryRigidFCC<<<gd->grid_dim_bn,gd->block_dim_bn,0,gd->cuStream_bn>>>(gd->u0,gd->u1,gd->adj_bn,gd->bn_ixyz,gd->K_bn);
         }
         //using buffer to then update FD boundaries
         CopyFromGridKernel<<<gd->grid_dim_bnl,gd->block_dim_bn,0,gd->cuStream_bn>>>(gd->u0b, gd->u0, gd->bnl_ixyz, ghd->Nbl);
         //possible this could be moved to host
         KernelBoundaryFD<<<gd->grid_dim_bnl,gd->block_dim_bn,0,gd->cuStream_bn>>>(gd->u0b,gd->u2b,gd->vh1,gd->gh1,gd->ssaf_bnl,gd->mat_bnl,gd->mat_beta,gd->mat_quads);
         //copy to back to grid
         CopyToGridKernel<<<gd->grid_dim_bnl,gd->block_dim_bn,0,gd->cuStream_bn>>>(gd->u0, gd->u0b, gd->bnl_ixyz, ghd->Nbl);
         gpuErrchk( cudaEventRecord(gd->cuEv_bn_roundtrip_end,gd->cuStream_bn) );

         //air updates (including source
         gpuErrchk( cudaStreamWaitEvent(gd->cuStream_air,gd->cuEv_bn_roundtrip_end,0) ); //might as well wait
         //run air kernel (with mask)
         gpuErrchk( cudaEventRecord(gd->cuEv_air_start,gd->cuStream_air) );

         //for absorbing boundaries at boundaries of grid
         CopyFromGridKernel<<<gd->grid_dim_bna,gd->block_dim_bn,0,gd->cuStream_air>>>(gd->u2ba, gd->u0, gd->bna_ixyz, ghd->Nba);
         if (gid==0) {
            FlipHaloXY_Zbeg<<<gd->grid_dim_halo_xy,gd->block_dim_halo_xy,0,gd->cuStream_air>>>(gd->u1);
         }
         if (gid==ngpus-1) {
            FlipHaloXY_Zend<<<gd->grid_dim_halo_xy,gd->block_dim_halo_xy,0,gd->cuStream_air>>>(gd->u1);
         }
         FlipHaloXZ_Ybeg<<<gd->grid_dim_halo_xz,gd->block_dim_halo_xz,0,gd->cuStream_air>>>(gd->u1);
         if (sd->fcc_flag==0) {
            FlipHaloXZ_Yend<<<gd->grid_dim_halo_xz,gd->block_dim_halo_xz,0,gd->cuStream_air>>>(gd->u1);
         }
         FlipHaloYZ_Xbeg<<<gd->grid_dim_halo_yz,gd->block_dim_halo_yz,0,gd->cuStream_air>>>(gd->u1);
         FlipHaloYZ_Xend<<<gd->grid_dim_halo_yz,gd->block_dim_halo_yz,0,gd->cuStream_air>>>(gd->u1);

         //injecting source first, negating sample to add it in first (NB source on different stream than bn)
         for (int64_t ns=0; ns<ghd->Ns; ns++) {
            AddIn<<<1,1,0,gd->cuStream_air>>>(gd->u0 + ghd->in_ixyz[ns],(Real)(-(ghd->in_sigs[ns*sd->Nt+n]))); 
         }
         //now air updates (not conflicting with bn updates because of bn_mask)
         if (sd->fcc_flag==0) {
            KernelAirCart<<<gd->grid_dim_air,gd->block_dim_air,0,gd->cuStream_air>>>(gd->u0,gd->u1,gd->bn_mask);
         }
         else {
            KernelAirFCC<<<gd->grid_dim_air,gd->block_dim_air,0,gd->cuStream_air>>>(gd->u0,gd->u1,gd->bn_mask);
         }
         //boundary ABC loss
         KernelBoundaryABC<<<gd->grid_dim_bna,gd->block_dim_bn,0,gd->cuStream_air>>>(gd->u0,gd->u2ba,gd->Q_bna,gd->bna_ixyz);
         gpuErrchk( cudaEventRecord(gd->cuEv_air_end,gd->cuStream_air) ); //for timing

         //readouts
         CopyFromGridKernel<<<gd->grid_dim_readout,gd->block_dim_readout,0,gd->cuStream_bn>>>(gd->u_out_buf, gd->u1, gd->out_ixyz, ghd->Nr);
         //then async memory copy of outputs (not really async because on same stream as CopyFromGridKernel)
         gpuErrchk( cudaMemcpyAsync(ghd->u_out_buf, gd->u_out_buf, ghd->Nr*sizeof(Real), cudaMemcpyDeviceToHost, gd->cuStream_bn) );
         gpuErrchk( cudaEventRecord(gd->cuEv_readout_end,gd->cuStream_bn) );
      }

      //readouts
      for (int gid=0; gid < ngpus; gid++) {
         gpuErrchk( cudaSetDevice(gid) );
         struct gpuData *gd = &(gds[gid]);
         struct gpuHostData *ghd = &(ghds[gid]);
         gpuErrchk( cudaEventSynchronize(gd->cuEv_readout_end) ); 
         //copy grid points off output buffer
         for (int64_t nr=0; nr<ghd->Nr; nr++) {
            ghd->u_out[nr*sd->Nt + n] = (double)(ghd->u_out_buf[nr]);
         }
      }
      //synchronise streams
      for (int gid=0; gid < ngpus; gid++) {
         gpuErrchk( cudaSetDevice(gid) );
         struct gpuData *gd = &(gds[gid]); //don't really need to set gpu device to sync
         gpuErrchk( cudaStreamSynchronize(gd->cuStream_air) ); //interior complete
         gpuErrchk( cudaStreamSynchronize(gd->cuStream_bn) ); //transfer complete

      }
      //dst then src, stream with src gives best performance (CUDA thing)

      //now asynchronous halo swaps, even/odd pairs concurrent
      //these are not async to rest of scheme, just async to other swaps

      //copy forward (even)
      for (int gid=0; gid < ngpus-1; gid+=2) {
         gpuErrchk( cudaSetDevice(gid) );
         gpuErrchk( cudaMemcpyPeerAsync(gds[gid+1].u0, gid+1, 
                                        gds[gid].u0 + Nzy*(ghds[gid].Nxh-2), gid, 
                                        (size_t)(Nzy*sizeof(Real)), 
                                        gds[gid].cuStream_bn) );
      }
      //copy back (odd)
      for (int gid=1; gid < ngpus; gid+=2) {
         gpuErrchk( cudaSetDevice(gid) );
         gpuErrchk( cudaMemcpyPeerAsync(gds[gid-1].u0 + Nzy*(ghds[gid-1].Nxh-1), gid-1, 
                                        gds[gid].u0 + Nzy, gid, 
                                        (size_t)(Nzy*sizeof(Real)), 
                                        gds[gid].cuStream_bn) );
      }
      //copy forward (odd)
      for (int gid=1; gid < ngpus-1; gid+=2) {
         gpuErrchk( cudaSetDevice(gid) );
         gpuErrchk( cudaMemcpyPeerAsync(gds[gid+1].u0, gid+1, 
                                        gds[gid].u0 + Nzy*(ghds[gid].Nxh-2), gid, 
                                        (size_t)(Nzy*sizeof(Real)), 
                                        gds[gid].cuStream_bn) );
      }
      //copy back (even) -- skip zero
      for (int gid=2; gid < ngpus; gid+=2) {
         gpuErrchk( cudaSetDevice(gid) );
         gpuErrchk( cudaMemcpyPeerAsync(gds[gid-1].u0 + Nzy*(ghds[gid-1].Nxh-1), gid-1, 
                                        gds[gid].u0 + Nzy, gid, 
                                        (size_t)(Nzy*sizeof(Real)), 
                                        gds[gid].cuStream_bn) );
      }

      for (int gid=0; gid < ngpus; gid++) {
         gpuErrchk( cudaSetDevice(gid) ); 
         struct gpuData *gd = &(gds[gid]);
         gpuErrchk( cudaStreamSynchronize(gd->cuStream_bn) ); //transfer complete
      }
      for (int gid=0; gid < ngpus; gid++) {
         struct gpuData *gd = &(gds[gid]);
         //update pointers
         Real *tmp_ptr; 
         tmp_ptr = gd->u1;
         gd->u1 = gd->u0;
         gd->u0 = tmp_ptr;

         //will use extra vector for this (simpler than extra copy kernel)
         tmp_ptr = gd->u2b;
         gd->u2b = gd->u1b;
         gd->u1b = gd->u0b;
         gd->u0b = tmp_ptr;

         if (gid==0) {
            gpuErrchk( cudaSetDevice(gid) );
            gpuErrchk( cudaEventRecord(cuEv_main_sample_end,0) );
         }
      }

      {
         //timing only on gpu0
         gpuErrchk( cudaSetDevice(0) );
         struct gpuData *gd = &(gds[0]);
         gpuErrchk( cudaEventSynchronize(cuEv_main_sample_end) ); //not sure this is correct
         gpuErrchk( cudaEventElapsedTime(&millis_since_start, cuEv_main_start, cuEv_main_sample_end) );
         gpuErrchk( cudaEventElapsedTime(&millis_since_sample_start, cuEv_main_sample_start, cuEv_main_sample_end) );

         time_elapsed = millis_since_start/1000;
         time_elapsed_sample = millis_since_sample_start/1000;

         float millis_air, millis_bn;
         gpuErrchk( cudaEventElapsedTime(&millis_air, gd->cuEv_air_start, gd->cuEv_air_end) );
         time_elapsed_sample_air = 0.001*millis_air;
         time_elapsed_air += time_elapsed_sample_air;

         //not full picutre, only first gpu
         gpuErrchk( cudaEventElapsedTime(&millis_bn, gd->cuEv_bn_roundtrip_start, gd->cuEv_bn_roundtrip_end) );

         time_elapsed_sample_bn = millis_bn/1000.0;
         time_elapsed_bn += time_elapsed_sample_bn;

         print_progress(n, sd->Nt, sd->Npts, sd->Nb, time_elapsed, time_elapsed_sample, time_elapsed_air, time_elapsed_sample_air, time_elapsed_bn, time_elapsed_sample_bn, ngpus);
      }
   }
   printf("\n");

   for (int gid=0; gid < ngpus; gid++) {
      gpuErrchk( cudaSetDevice(gid) );
      gpuErrchk( cudaPeekAtLastError() );
      gpuErrchk( cudaDeviceSynchronize() );
   }
   {
      //timing (on device 0)
      gpuErrchk( cudaSetDevice(0) );
      gpuErrchk( cudaEventRecord(cuEv_main_end) );
      gpuErrchk( cudaEventSynchronize(cuEv_main_end) );

      gpuErrchk( cudaEventElapsedTime(&millis_since_start, cuEv_main_start, cuEv_main_end) );
      time_elapsed = millis_since_start/1000;
   }

   /*------------------------
    * FREE WILLY 
   ------------------------*/
   gpuErrchk( cudaSetDevice(0) );
   gpuErrchk( cudaEventDestroy(cuEv_main_start) );
   gpuErrchk( cudaEventDestroy(cuEv_main_end) );
   gpuErrchk( cudaEventDestroy(cuEv_main_sample_start) );
   gpuErrchk( cudaEventDestroy(cuEv_main_sample_end) );
   for (int gid=0; gid < ngpus; gid++) {
      gpuErrchk( cudaSetDevice(gid) );
      struct gpuData *gd = &(gds[gid]);
      struct gpuHostData *ghd = &(ghds[gid]);
      //cleanup streams
      gpuErrchk( cudaStreamDestroy(gd->cuStream_air) );
      gpuErrchk( cudaStreamDestroy(gd->cuStream_bn) );

      //cleanup events
      gpuErrchk( cudaEventDestroy(gd->cuEv_air_start) );
      gpuErrchk( cudaEventDestroy(gd->cuEv_air_end) );
      gpuErrchk( cudaEventDestroy(gd->cuEv_bn_roundtrip_start) );
      gpuErrchk( cudaEventDestroy(gd->cuEv_bn_roundtrip_end) );
      gpuErrchk( cudaEventDestroy(gd->cuEv_readout_end) );

      //free memory
      gpuErrchk( cudaFree(gd->u0) );
      gpuErrchk( cudaFree(gd->u1) );
      gpuErrchk( cudaFree(gd->out_ixyz) );
      gpuErrchk( cudaFree(gd->bn_ixyz) );
      gpuErrchk( cudaFree(gd->bnl_ixyz) );
      gpuErrchk( cudaFree(gd->bna_ixyz) );
      gpuErrchk( cudaFree(gd->Q_bna) );
      gpuErrchk( cudaFree(gd->adj_bn) );
      gpuErrchk( cudaFree(gd->mat_bnl) );
      gpuErrchk( cudaFree(gd->K_bn) );
      gpuErrchk( cudaFree(gd->ssaf_bnl) );
      gpuErrchk( cudaFree(gd->mat_beta) );
      gpuErrchk( cudaFree(gd->mat_quads) );
      gpuErrchk( cudaFree(gd->bn_mask) );
      gpuErrchk( cudaFree(gd->u0b) );
      gpuErrchk( cudaFree(gd->u1b) );
      gpuErrchk( cudaFree(gd->u2b) );
      gpuErrchk( cudaFree(gd->u2ba) );
      gpuErrchk( cudaFree(gd->vh1) );
      gpuErrchk( cudaFree(gd->gh1) );
      gpuErrchk( cudaFree(gd->u_out_buf) );
      free(ghd->bn_mask);
      free(ghd->bn_ixyz);
      free(ghd->bnl_ixyz);
      free(ghd->bna_ixyz);
      free(ghd->in_ixyz);
      free(ghd->out_ixyz);
   }
   gpuErrchk( cudaFreeHost(u_out_buf) );
   free(gds);
   free(ghds);

   //reset after frees (for some reason it conflicts with cudaFreeHost)
   for (int gid=0; gid < ngpus; gid++) {
      gpuErrchk( cudaSetDevice(gid) );
      gpuErrchk( cudaDeviceReset() );
   }

   printf("Boundary loop: %.6fs, %.2f Mvox/s\n",time_elapsed_bn,sd->Nb*sd->Nt/1e6/time_elapsed_bn);
   printf("Air update: %.6fs, %.2f Mvox/s\n",time_elapsed_air,sd->Npts*sd->Nt/1e6/time_elapsed_air);
   printf("Combined (total): %.6fs, %.2f Mvox/s\n",time_elapsed,sd->Npts*sd->Nt/1e6/time_elapsed);
   return time_elapsed;
}
#endif

