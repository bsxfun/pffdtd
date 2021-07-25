// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: fdtd_common.h
//
// Description: Header-only misc function definitions related to FDTD simulation
//
///////////////////////////////////////////////////////////////////////////////

#ifndef _FDTD_COMMON_H
#define _FDTD_COMMON_H
#ifndef _STDINT_H
#include <stdint.h> 
#endif
#ifndef _STRING_H
#include <string.h> 
#endif
#ifndef _MATH_H
#include <math.h> 
#endif
#ifndef _FLOAT_H
#include <float.h> 
#endif
#ifndef _TIME_H
#include <time.h> //date and time
#endif
#ifndef _OMP_H
#include <omp.h>
#endif
#ifndef _IOCTL_H
#include <sys/ioctl.h> //terminal width
#endif
#ifndef _HELPER_FUNCS_H
#include <helper_funcs.h> 
#endif

//flag passed in at compilation (see Makefile)
#if PRECISION==2 //double
   typedef double Real;

   #define REAL_MAX_EXP DBL_MAX_EXP
   #define REAL_MIN_EXP DBL_MIN_EXP

   //using CUDA math intrinsics
   #define FMA_O __fma_rn 
   #define FMA_D __fma_rn
   #define ADD_O __dadd_rn
   #define ADD_D __dadd_rn
   #define EPS 0.0

#elif PRECISION==1 //float with safeguards
   typedef float Real;

   #define REAL_MAX_EXP FLT_MAX_EXP
   #define REAL_MIN_EXP FLT_MIN_EXP

   //using CUDA math intrinsics
   #define FMA_O __fmaf_rz //off-diag
   #define FMA_D __fmaf_rn //diag
   #define ADD_O __fadd_rz
   #define ADD_D __fadd_rn
   #define EPS 1.19209289e-07 //helps with stability in single
#else
   #error "PRECISION = 1 (single) or 2 (double)"
#endif

//declarations, defs below
void ind2sub3d(int64_t idx, int64_t Nx, int64_t Ny, int64_t Nz, int64_t *ix, int64_t *iy, int64_t *iz);
void check_inside_grid(int64_t *idx, int64_t N, int64_t Nx, int64_t Ny, int64_t Nz);
void print_progress(uint32_t n, uint32_t Nt, uint64_t Npts, uint64_t Nb, 
                     double time_elapsed, double time_elapsed_sample, 
                     double time_elapsed_air, double time_elapsed_sample_air, 
                     double time_elapsed_bn, double time_elapsed_sample_bn, int num_workers);


//linear indices to sub-indices in 3d, Nz continguous
void ind2sub3d(int64_t idx, int64_t Nx, int64_t Ny, int64_t Nz, int64_t *ix, int64_t *iy, int64_t *iz) {
   *iz = idx % Nz;
   *iy = (idx - (*iz))/Nz%Ny;
   *ix = ((idx - (*iz))/Nz-(*iy))/Ny;
   assert(*ix>0);
   assert(*iy>0);
   assert(*iz>0);
   assert(*ix<Nx-1);
   assert(*iy<Ny-1);
   assert(*iz<Nz-1);
}

//double check some index inside grid
void check_inside_grid(int64_t *idx, int64_t N, int64_t Nx, int64_t Ny, int64_t Nz) {
   for (int64_t i=0; i<N; i++) {
      int64_t iz,iy,ix;
      ind2sub3d(idx[i], Nx, Ny, Nz, &ix, &iy, &iz);
   }
}

//hacky print progress (like tqdm).. 
//N.B. this conflicts with tmux scrolling (stdout needs to flush)
//and not great for piping output to log (better to disable or change for those cases)
void print_progress(uint32_t n, uint32_t Nt, uint64_t Npts, uint64_t Nb, 
                     double time_elapsed, double time_elapsed_sample, 
                     double time_elapsed_air, double time_elapsed_sample_air, 
                     double time_elapsed_bn, double time_elapsed_sample_bn, int num_workers) {
   //progress bar (doesn't impact performance unless simulation is really tiny)
   struct winsize w;
   ioctl(0, TIOCGWINSZ, &w);
   int ncols = w.ws_col;
   int ncolsl = 80;
   //int ncolsl = 120;
   //int ncolsp = w.ws_col-ncolsl;

   double pcnt = (100.0*n)/Nt;
   int nlines = 6;
   if (n>0) {
      //back up
      for (int nl=0; nl<nlines; nl++) {
         printf("\033[1A");
      }
      //clear lines
      for (int nl=0; nl<nlines; nl++) {
         for (int cc=0; cc < ncols; cc++) {
            printf(" ");
         }
         printf("\n");
      }
      //back up
      for (int nl=0; nl<nlines; nl++) {
         printf("\033[1A");
      }
   }
   printf("\n");
   //progress bar
   printf("Running [%.1f%%]", pcnt);
   if (ncols>=ncolsl) {
      for (int cc=0; cc < (0.01*pcnt*ncolsl); cc++) {
         printf("=");
      }
      printf(">");
      for (int cc=(0.01*pcnt*ncolsl); cc<ncolsl; cc++) {
         printf(".");
      }
   }
   double est_total = time_elapsed*Nt/n;

   int sec, h_e, m_e, s_e, h_t, m_t, s_t;
   sec = (int)time_elapsed;
   h_e = (sec/3600); 
   m_e = (sec -(3600*h_e))/60;
   s_e = (sec -(3600*h_e)-(m_e*60));

   sec = (int)est_total;
   h_t = (sec/3600); 
   m_t = (sec -(3600*h_t))/60;
   s_t = (sec -(3600*h_t)-(m_t*60));

   printf("[");
   printf("%02d:%02d:%02d<%02d:%02d:%02d]",h_e,m_e,s_e,h_t,m_t,s_t);
   printf("\n");
   printf("T: %06.1f", 1e-6*Npts*n/time_elapsed); //"total" Mvox/s (averaged up to current time)
   printf(" - ");
   printf("I: %06.1f", 1e-6*Npts/time_elapsed_sample); //instantaneous Mvox/s (per time-step)
   printf(" | ");
   printf("TPW: %06.1f", 1e-6*Npts*n/time_elapsed/num_workers); //total per worker
   printf(" - ");
   printf("IPW: %06.1f", 1e-6*Npts/time_elapsed_sample/num_workers); //inst per worker
   printf("\n");

   printf("TA: %06.1f", 1e-6*Npts*n/time_elapsed_air); //total for air bit
   printf(" - ");
   printf("IA: %06.1f", 1e-6*Npts/time_elapsed_sample_air); //inst for air bit

   printf("\n");
   printf("TB: %06.1f", 1e-6*Nb*n/time_elapsed_bn); //total for bn
   printf(" - ");
   printf("IB: %06.1f", 1e-6*Nb/time_elapsed_sample_bn); //inst for bn

   printf("\n");

   printf("T: %02.1f%%", 100.0*time_elapsed_air/time_elapsed); //% for air (total)
   printf(" - ");
   printf("I: %02.1f%%", 100.0*time_elapsed_sample_air/time_elapsed_sample); //% for air (inst)
   printf("\n");
   fflush(stdout);
}
#endif
