// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: cpu_engine.h
//
// Description: CPU-based implementation of FDTD engine, with OpenMP
//
///////////////////////////////////////////////////////////////////////////////

#ifndef _CPU_ENGINE_H
#define _CPU_ENGINE_H
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
#ifndef _OMP_H
#include <omp.h>
#endif
#ifndef _HELPER_FUNCS_H
#include <helper_funcs.h> 
#endif
#ifndef _FDTD_COMMON_H
#include <fdtd_common.h>
#endif
#ifndef _FDTD_DATA_H
#include <fdtd_data.h>
#endif

double run_sim(struct SimData *sd);
double process_bnl_pts_fd(Real *u0b, const Real *u2b, const Real *ssaf_bnl, const int8_t *mat_bnl, int64_t Nbl, int8_t *Mb, Real lo2, Real *vh1, Real *gh1, const struct MatQuad *mat_quads, const Real *mat_beta);

double run_sim(struct SimData *sd) 
{
   //keep local ints, scalars
   int64_t Ns   = sd->Ns;
   int64_t Nr   = sd->Nr;
   int64_t Nt   = sd->Nt;
   int64_t Npts = sd->Npts;
   int64_t Nx   = sd->Nx;
   int64_t Ny   = sd->Ny;
   int64_t Nz   = sd->Nz;
   int64_t Nb   = sd->Nb;
   int64_t Nbl  = sd->Nbl;
   int64_t Nba  = sd->Nba;
   int8_t *Mb  = sd->Mb;

   //keep local copies of pointers (style choice)
   int64_t *bn_ixyz   = sd->bn_ixyz;
   int64_t *bnl_ixyz  = sd->bnl_ixyz;
   int64_t *bna_ixyz  = sd->bna_ixyz;
   int64_t *in_ixyz   = sd->in_ixyz;
   int64_t *out_ixyz  = sd->out_ixyz;
   uint16_t *adj_bn   = sd->adj_bn;
   uint8_t *bn_mask   = sd->bn_mask;
   int8_t *mat_bnl    = sd->mat_bnl;
   int8_t *Q_bna      = sd->Q_bna;
   double *in_sigs    = sd->in_sigs;
   double *u_out      = sd->u_out;
   int8_t fcc_flag    = sd->fcc_flag;
   Real *ssaf_bnl   = sd->ssaf_bnl;
   Real *mat_beta   = sd->mat_beta;
   struct MatQuad *mat_quads = sd->mat_quads;

   //these are states grids
   Real *u0,*u1;
   //local copies for boundary
   Real *u0b,*u1b,*u2b,*u2ba;
   //for FD-boundaries
   Real *vh1,*gh1;

   //allocate memory
   mymalloc((void **)&u0, (size_t)Npts*sizeof(Real));
   mymalloc((void **)&u1, (size_t)Npts*sizeof(Real));
   mymalloc((void **)&u0b, (size_t)Nbl*sizeof(Real));
   mymalloc((void **)&u1b, (size_t)Nbl*sizeof(Real));
   mymalloc((void **)&u2b, (size_t)Nbl*sizeof(Real));
   mymalloc((void **)&u2ba, (size_t)Nba*sizeof(Real));

   //size hash-defined, but not using all of it necesssarily
   mymalloc((void **)&vh1, (size_t)Nbl*MMb*sizeof(Real));
   mymalloc((void **)&gh1, (size_t)Nbl*MMb*sizeof(Real));

   //sim coefficients
   Real lo2 = sd->lo2;
   Real sl2 = sd->sl2;
   Real l = sd->l;
   Real a1 = sd->a1;
   Real a2 = sd->a2;

   //can control outside with OMP_NUM_THREADS env variable
   int num_workers = omp_get_max_threads();

   printf("ENGINE: fcc_flag=%d\n",fcc_flag);
   printf("%s", (fcc_flag>0) ? "fcc=true\n" : "fcc=false\n");

   //for timing
   double time_elapsed;
   double time_elapsed_air=0.0;
   double time_elapsed_bn=0.0;
   double time_elapsed_sample;
   double time_elapsed_sample_air=0.0;
   double time_elapsed_sample_bn=0.0;
   double start_time = omp_get_wtime();
   double sample_start_time;
   double current_time;
   int64_t NzNy = Nz*Ny;
   for (int64_t n=0; n<Nt; n++) {
      sample_start_time = omp_get_wtime();

      //copy last state ABCs
      #pragma omp parallel for
      for (int64_t nb=0; nb<Nba; nb++) {
         u2ba[nb] = u0[bna_ixyz[nb]]; 
      }
      if (fcc_flag==2) { //copy y-z face for FCC folded grid
         #pragma omp parallel for
         for (int64_t ix=0; ix<Nx; ix++) { 
            for (int64_t iz=0; iz<Nz; iz++) { 
               u1[ix*NzNy + (Ny-1)*Nz + iz] = u1[ix*NzNy + (Ny-2)*Nz + iz];
            }
         }
      }

      //halo flips for ABCs
      #pragma omp parallel for
      for (int64_t ix=0; ix<Nx; ix++) {
         for (int64_t iy=0; iy<Ny; iy++) {
            u1[ix*NzNy+iy*Nz+0]    = u1[ix*NzNy+iy*Nz+2];
            u1[ix*NzNy+iy*Nz+Nz-1] = u1[ix*NzNy+iy*Nz+Nz-3];
         }
      }
      #pragma omp parallel for
      for (int64_t ix=0; ix<Nx; ix++) {
         for (int64_t iz=0; iz<Nz; iz++) { 
            u1[ix*NzNy+0*Nz+iz] = u1[ix*NzNy+2*Nz+iz];
         }
      }
      if (fcc_flag!=2) { //only this y-face if not FCC folded grid
         #pragma omp parallel for
         for (int64_t ix=0; ix<Nx; ix++) {
            for (int64_t iz=0; iz<Nz; iz++) { 
               u1[ix*NzNy+(Ny-1)*Nz+iz] = u1[ix*NzNy+(Ny-3)*Nz+iz];
            }
         }
      }
      #pragma omp parallel for
      for (int64_t iy=0; iy<Ny; iy++) {
         for (int64_t iz=0; iz<Nz; iz++) { 
            u1[0*NzNy+iy*Nz+iz]      = u1[2*NzNy+iy*Nz+iz];
            u1[(Nx-1)*NzNy+iy*Nz+iz] = u1[(Nx-3)*NzNy+iy*Nz+iz];
         }
      }

      //air update for schemes
      if (fcc_flag==0) { //cartesian scheme
         #pragma omp parallel for
         for (int64_t ix=1; ix<Nx-1; ix++) {
            for (int64_t iy=1; iy<Ny-1; iy++) {
               for (int64_t iz=1; iz<Nz-1; iz++) { //contiguous
                  int64_t ii = ix*NzNy + iy*Nz + iz;
                  if ( !(GET_BIT(bn_mask[ii>>3],ii%8)) ) { 
                     Real partial = a1*u1[ii] - u0[ii];
                     partial += a2*u1[ii + NzNy];
                     partial += a2*u1[ii - NzNy];
                     partial += a2*u1[ii + Nz];
                     partial += a2*u1[ii - Nz];
                     partial += a2*u1[ii + 1];
                     partial += a2*u1[ii - 1];
                     u0[ii] = partial;
                  }
               }
            }
         }
      }
      else if (fcc_flag>0) { 
         #pragma omp parallel for
         for (int64_t ix=1; ix<Nx-1; ix++) {
            for (int64_t iy=1; iy<Ny-1; iy++) {
               //while loop iterates iterates over both types of FCC grids
               int64_t iz = (fcc_flag==1)? 2-(ix+iy)%2 : 1;
               while (iz<Nz-1) { 
                  int64_t ii = ix*NzNy + iy*Nz + iz;
                  if ( !(GET_BIT(bn_mask[ii>>3],ii%8)) ) {
                     Real partial = a1*u1[ii] - u0[ii];
                     partial += a2*u1[ii + NzNy + Nz];
                     partial += a2*u1[ii - NzNy - Nz];
                     partial += a2*u1[ii + Nz + 1];
                     partial += a2*u1[ii - Nz - 1];
                     partial += a2*u1[ii + NzNy + 1];
                     partial += a2*u1[ii - NzNy - 1];
                     partial += a2*u1[ii + NzNy - Nz];
                     partial += a2*u1[ii - NzNy + Nz];
                     partial += a2*u1[ii + Nz - 1];
                     partial += a2*u1[ii - Nz + 1];
                     partial += a2*u1[ii + NzNy - 1];
                     partial += a2*u1[ii - NzNy + 1];
                     u0[ii] = partial;
                  }
                  iz += ((fcc_flag==1)? 2 : 1);
               }
            }
         }
      }
      //ABC loss (2nd-order accurate first-order Engquist-Majda)
      for (int64_t nb=0; nb<Nba; nb++) {
         Real lQ = l*Q_bna[nb];
         int64_t ib = bna_ixyz[nb];
         u0[ib] = (u0[ib] + lQ*u2ba[nb])/(1.0 + lQ);
      }

      //rigid boundary nodes, using adj data 
      time_elapsed_sample_air = omp_get_wtime() - sample_start_time;
      time_elapsed_air += time_elapsed_sample_air;
      if (fcc_flag==0) {
         #pragma omp parallel for
         for (int64_t nb=0; nb<Nb; nb++) {
            int64_t ii = bn_ixyz[nb];
            uint8_t Kint;
            uint16_t v = adj_bn[nb];
            for (Kint = 0; v; Kint++) v &= v - 1; // clear the least significant bit set

            Real _2 = 2.0;
            Real K = Kint;
            Real b2 = a2;
            Real b1 = (_2-sl2*K);

            Real partial = b1*u1[ii] - u0[ii];
            uint16_t adj = adj_bn[nb];
            partial += b2*(Real)GET_BIT(adj,0)*u1[ii + NzNy];
            partial += b2*(Real)GET_BIT(adj,1)*u1[ii - NzNy];
            partial += b2*(Real)GET_BIT(adj,2)*u1[ii + Nz];
            partial += b2*(Real)GET_BIT(adj,3)*u1[ii - Nz];
            partial += b2*(Real)GET_BIT(adj,4)*u1[ii + 1];
            partial += b2*(Real)GET_BIT(adj,5)*u1[ii - 1];
            u0[ii] = partial;
         }
      }
      else if (fcc_flag>0) {
         #pragma omp parallel for
         for (int64_t nb=0; nb<Nb; nb++) {
            int64_t ii = bn_ixyz[nb];
            uint8_t Kint;
            uint16_t v = adj_bn[nb];
            for (Kint = 0; v; Kint++) v &= v - 1; // clear the least significant bit set

            Real _2 = 2.0;
            Real K = Kint;
            Real b2 = a2;
            Real b1 = (_2-sl2*K);

            Real partial = b1*u1[ii] - u0[ii];
            uint16_t adj = adj_bn[nb];
            partial += b2*(Real)GET_BIT(adj,0) *u1[ii + NzNy + Nz];
            partial += b2*(Real)GET_BIT(adj,1) *u1[ii - NzNy - Nz];
            partial += b2*(Real)GET_BIT(adj,2) *u1[ii + Nz + 1];
            partial += b2*(Real)GET_BIT(adj,3) *u1[ii - Nz - 1];
            partial += b2*(Real)GET_BIT(adj,4) *u1[ii + NzNy + 1];
            partial += b2*(Real)GET_BIT(adj,5) *u1[ii - NzNy - 1];
            partial += b2*(Real)GET_BIT(adj,6) *u1[ii + NzNy - Nz];
            partial += b2*(Real)GET_BIT(adj,7) *u1[ii - NzNy + Nz];
            partial += b2*(Real)GET_BIT(adj,8) *u1[ii + Nz - 1];
            partial += b2*(Real)GET_BIT(adj,9) *u1[ii - Nz + 1];
            partial += b2*(Real)GET_BIT(adj,10)*u1[ii + NzNy - 1];
            partial += b2*(Real)GET_BIT(adj,11)*u1[ii - NzNy + 1];
            u0[ii] = partial;
         }
      }

      //read bn points (not strictly necessary, just mirrors CUDA implementation)
      #pragma omp parallel for
      for (int64_t nb=0; nb<Nbl; nb++) {
         u0b[nb] = u0[bnl_ixyz[nb]]; 
      }
      //process FD boundary nodes
      time_elapsed_sample_bn = process_bnl_pts_fd(u0b, u2b, ssaf_bnl, mat_bnl, Nbl, Mb, lo2, vh1, gh1, mat_quads, mat_beta);
      time_elapsed_bn += time_elapsed_sample_bn;
      //write back
      #pragma omp parallel for
      for (int64_t nb=0; nb<Nbl; nb++) {
         u0[bnl_ixyz[nb]] = u0b[nb];
      }

      //read output at current sample
      for (int64_t nr=0; nr<Nr; nr++) {
         int64_t ii = out_ixyz[nr];
         u_out[nr*Nt + n] = (double)u1[ii];
      }

      //add current sample to next (as per update)
      for (int64_t ns=0; ns<Ns; ns++) {
         int64_t ii = in_ixyz[ns];
         u0[ii] +=  (Real)in_sigs[ns*Nt + n];
      }

      //swap pointers
      Real *tmp_ptr; 
      tmp_ptr = u1;
      u1 = u0;
      u0 = tmp_ptr;

      //using extra state here for simplicity
      tmp_ptr = u2b;
      u2b = u1b;
      u1b = u0b;
      u0b = tmp_ptr;

      current_time = omp_get_wtime();
      time_elapsed = current_time-start_time;
      time_elapsed_sample = current_time-sample_start_time;
      //print progress (can be removed or changed)
      print_progress(n, Nt, Npts, Nb, time_elapsed, time_elapsed_sample, time_elapsed_air, time_elapsed_sample_air, time_elapsed_bn, time_elapsed_sample_bn, num_workers);

   }
   printf("\n");

   //timing
   double end_time = omp_get_wtime();
   time_elapsed = end_time - start_time;

   /*------------------------
    * FREE WILLY 
   ------------------------*/
   free(u0);
   free(u1);
   free(u0b);
   free(u1b);
   free(u2b);
   free(u2ba);
   free(vh1);
   free(gh1);

   /*------------------------
    * RETURN
   ------------------------*/
   printf("Air update: %.6fs, %.2f Mvox/s\n",time_elapsed_air,Npts*Nt/1e6/time_elapsed_air);
   printf("Boundary loop: %.6fs, %.2f Mvox/s\n",time_elapsed_bn,Nb*Nt/1e6/time_elapsed_bn);
   printf("Combined (total): %.6fs, %.2f Mvox/s\n",time_elapsed,Npts*Nt/1e6/time_elapsed);

   return time_elapsed;
}

//function that does freq-dep RLC boundaries.  See 2016 ISMRA paper and accompanying webpage (slightly improved here)
double process_bnl_pts_fd(Real *u0b, const Real *u2b, const Real *ssaf_bnl, const int8_t *mat_bnl, int64_t Nbl, int8_t *Mb, Real lo2,
                         Real *vh1, Real *gh1, const struct MatQuad *mat_quads, const Real *mat_beta) {
   double tstart = omp_get_wtime();
   #pragma omp parallel for schedule(static) 
   for (int64_t nb=0; nb<Nbl; nb++) {
      Real _1 = 1.0;
      Real _2 = 2.0;
      int32_t k = mat_bnl[nb];

      Real lo2Kbg = lo2*ssaf_bnl[nb]*mat_beta[k];
      Real fac = _2*lo2*ssaf_bnl[nb] / (_1 + lo2Kbg);

      Real u0bint = u0b[nb];
      Real u2bint = u2b[nb];

      u0bint = (u0bint + lo2Kbg*u2bint) / (_1 + lo2Kbg);

      Real vh1nb[MMb];
      for (int8_t m=0; m<Mb[k]; m++) {
         int64_t nbm = nb*MMb+m;
         int32_t mbk = k*MMb+m;
         const struct MatQuad *tm;
         tm = &(mat_quads[mbk]);
         vh1nb[m] = vh1[nbm];
         u0bint -= fac*( _2*(tm->bDh)*vh1nb[m] - (tm->bFh)*gh1[nbm] );
      }

      Real du = u0bint-u2bint;

      for (int8_t m=0; m<Mb[k]; m++) {
         int64_t nbm = nb*MMb+m;
         int32_t mbk = k*MMb+m;
         const struct MatQuad *tm; 
         tm = &(mat_quads[mbk]);
         Real vh0nbm = (tm->b)*du + (tm->bd)*vh1nb[m] - _2*(tm->bFh)*gh1[nbm];
         gh1[nbm] += (vh0nbm + vh1nb[m])/_2;
         vh1[nbm] = vh0nbm;
      }

      u0b[nb] = u0bint;
   }
   return omp_get_wtime()-tstart;
}
#endif

