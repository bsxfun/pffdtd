// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: fdtd_data.h
//
// Description: Header-only function definitions for handling loading of 
// simulation data from HDF5 files, preparing for simulation, and writing outputs
//
///////////////////////////////////////////////////////////////////////////////

#ifndef _FDTD_DATA_H
#define _FDTD_DATA_H
#ifndef _STDINT_H
#include <stdint.h> 
#endif
#ifndef _HELPER_FUNCS_H
#include <helper_funcs.h> 
#endif
#ifndef _FDTD_COMMON_H
#include <fdtd_common.h> 
#endif
#ifndef _HDF5_H
#include "hdf5.h"
#endif

//maximum number of RLC branches in freq-dep (FD) boundaries (needed at compile-time for CUDA kernels)
#define MMb 12 //change as necssary
//maximum number of materials allows (needed at compile-time for CUDA)
#define MNm 64 //change as necssary 

//main sim data, on host 
struct SimData { 
   int64_t *bn_ixyz; //boundary node indices
   int64_t *bnl_ixyz; //lossy boundary node indices
   int64_t *bna_ixyz; //absorbing boundary node indices
   int8_t *Q_bna; //integer for ABCs (wall 1,edge 2,corner 3)
   int64_t *in_ixyz; //input points
   int64_t *out_ixyz; //output points
   int64_t *out_reorder; //ordering for outputs point for final print/save
   uint16_t *adj_bn; //nearest-neighbour adjancencies for all boundary nodes
   Real  *ssaf_bnl; //surface area corrections (with extra volume scaling)
   uint8_t *bn_mask; //bit mask for bounday nodes
   int8_t *mat_bnl; //material indices for lossy boundary nodes
   int8_t *K_bn; //number of adjacent neighbours, boundary nodesa
   double *in_sigs; //input signals
   double *u_out; //for output signals
   int64_t Ns; //number of input grid points
   int64_t Nr; //number of output grid points
   int64_t Nt; //number of samples simulation
   int64_t Npts; //number of Cartesian grid points
   int64_t Nx; //x-dim (non-continguous)
   int64_t Ny; //y-dim
   int64_t Nz; //z-dim (continguous)
   int64_t Nb; //number of boundary nodes
   int64_t Nbl; //number of lossy boundary nodes
   int64_t Nba; //number of ABC nodes
   double l; //Courant number (CFL)
   double l2; // CFL number squared
   int8_t fcc_flag; //boolean for FCC
   int8_t NN; //integer, neareast neighbours
   int8_t Nm; //number of materials used 
   int8_t *Mb; //number of branches per material
   struct MatQuad *mat_quads; //RLC coefficients (essentially)
   Real *mat_beta; //part of FD-boundaries one per material
   double infac; //rescaling of input (for numerical reason)
   Real sl2; //scaled l2 (for single precision)
   Real lo2; //0.5*l
   Real a2; //update stencil coefficient
   Real a1; //update stencil coefficient
};

//see python code and 2016 ISMRA paper
struct MatQuad {
   Real b; //b 
   Real bd; //b*d
   Real bDh; //b*D-hat
   Real bFh; //b*F-hat
};

//some declarations (comments at definitions)
void read_h5_constant(hid_t file, char *dset_str, void *data_container, TYPE t);
void read_h5_dataset(hid_t file, char *dset_str, int ndims, hsize_t *dims, void **data_array, TYPE t);
void load_sim_data(struct SimData *sd);
void free_sim_data(struct SimData *sd);
void read_h5_dataset(hid_t file, char *dset_str, int ndims, hsize_t *dims, void **data_array, TYPE t);
void read_h5_constant(hid_t file, char *dset_str, void *data_container, TYPE t);
void print_last_samples(struct SimData *sd);
void scale_input(struct SimData *sd);
void rescale_output(struct SimData *sd);
void write_outputs(struct SimData *sd);

//load the sim data from Python-written HDF5 files
void load_sim_data(struct SimData *sd) {
   //local values, to read in and attach to struct at end
   int64_t Nx,Ny,Nz;
   int64_t Nb,Nbl,Nba;
   int64_t Npts;
   int64_t Ns,Nr,Nt;
   int64_t *bn_ixyz;
   int64_t *bnl_ixyz;
   int64_t *bna_ixyz;
   int8_t *Q_bna;
   int64_t *in_ixyz,*out_ixyz,*out_reorder;
   bool *adj_bn_bool;
   int8_t *K_bn;
   uint16_t *adj_bn; //large enough for FCC 
   uint8_t *bn_mask;
   int8_t *mat_bn,*mat_bnl;
   double *saf_bn;
   Real *ssaf_bn,*ssaf_bnl;
   double *in_sigs;
   double *u_out;
   double l;
   double l2;
   int8_t fcc_flag;
   int8_t NN;
   int8_t *Mb;
   int8_t Nm;
   struct MatQuad *mat_quads;
   Real *mat_beta; //one per material

   double Ts;
   bool diff;

   hid_t file; //HDF5 type
   hsize_t dims[2]; //HDF5 type 
   int expected_ndims;
   char dset_str[80];
   char filename[80];

   ////////////////////////////////////////////////////////////////////////
   // 
   // Read sim_consts HDF5 dataset
   // 
   ////////////////////////////////////////////////////////////////////////
   strcpy(filename, "sim_consts.h5");
   if (!check_file_exists(filename)) assert(true==false);

   file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

   //////////////////
   // constants
   //////////////////
   strcpy(dset_str, "l");
   read_h5_constant(file, dset_str,(void *)&l,FLOAT64);
   printf("l=%.16g\n",l);

   strcpy(dset_str, "l2");
   read_h5_constant(file, dset_str,(void *)&l2,FLOAT64);
   printf("l2=%.16g\n",l2);

   strcpy(dset_str, "Ts");
   read_h5_constant(file, dset_str,(void *)&Ts,FLOAT64);
   printf("Ts=%.16g\n",Ts);

   strcpy(dset_str, "fcc_flag");
   read_h5_constant(file, dset_str,(void *)&fcc_flag,INT8);
   printf("fcc_flag=%d\n",fcc_flag);
   assert((fcc_flag>=0) && (fcc_flag<=2));
   //printf("%s", fcc>0 ? "fcc=true\n" : "fcc=false\n");

   if (H5Fclose(file) != 0) {
      printf("error closing file %s",filename);
      assert(true==false);
   }
   else printf("closed file %s\n",filename);

   if (fcc_flag>0) { //FCC (1 is CPU-based, 2 is CPU or GPU)
      assert(l2<=1.0);
      assert(l<=1.0);
      NN = 12;
   }
   else { //simple Cartesian
      assert(l2<=1.0/3.0);
      assert(l<=sqrt(1.0/3.0));
      NN = 6;
   }

   //calculate some update coefficients
   double lfac = (fcc_flag>0)? 0.25 : 1.0; //laplacian factor
   double dsl2 = (1.0+EPS)*lfac*l2; //scale for stability (EPS in fdtd_common.h)
   double da1 = (2.0-dsl2*NN); //scaling for stability in single
   double da2 = lfac*l2;
   //Real is defined in fdtd_common.h (float or double)
   Real a1 = da1;
   Real a2 = da2;
   Real sl2 = dsl2;
   Real lo2 = 0.5*l;

   printf("a2 (double): %.16g\n",da2);
   printf("a2 (Real): %.16g\n",a2);
   printf("a1 (double): %.16g\n",da1);
   printf("a1 (Real): %.16g\n",a1);
   printf("sl2 (double): %.16g\n",dsl2);
   printf("sl2 (Real): %.16g\n",sl2);

   printf("l2=%.16g\n",l2);
   printf("NN=%d\n",NN);

   ////////////////////////////////////////////////////////////////////////
   // 
   // Read vox HDF5 dataset
   // 
   ////////////////////////////////////////////////////////////////////////
   strcpy(filename, "vox_out.h5");
   if (!check_file_exists(filename)) assert(true==false);

   file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

   //////////////////
   // integers
   //////////////////
   strcpy(dset_str, "Nx");
   read_h5_constant(file, dset_str,(void *)&Nx,INT64);
   printf("Nx=%ld\n",Nx);

   strcpy(dset_str, "Ny");
   read_h5_constant(file, dset_str,(void *)&Ny,INT64);
   printf("Ny=%ld\n",Ny);

   strcpy(dset_str, "Nz");
   read_h5_constant(file, dset_str,(void *)&Nz,INT64);
   printf("Nz=%ld\n",Nz);

   Npts = Nx*Ny*Nz;
   printf("Npts=%ld\n",Npts);

   strcpy(dset_str, "Nb");
   read_h5_constant(file, dset_str,(void *)&Nb,INT64);
   printf("Nb=%ld\n",Nb);

   //////////////////
   // bn_ixyz dataset
   //////////////////
   strcpy(dset_str, "bn_ixyz");expected_ndims=1;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&bn_ixyz,INT64);
   assert((int64_t)dims[0]==Nb);

   //printf("bn_ixyz=");
   //for (int64_t i=0; i<Nb; i++) {
      //printf("%ld, ",bn_ixyz[i]);
   //}
   //printf("\n");

   //////////////////
   // adj_bn dataset
   //////////////////
   strcpy(dset_str, "adj_bn");expected_ndims=2;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&adj_bn_bool,BOOL);
   assert((int64_t)dims[0]==Nb);
   assert(dims[1]==(hsize_t)NN);
   
   //printf("adj_bn=");
   //for (int64_t i=0; i<Nb; i++) {
      //printf("[");
      //for (int8_t j=0; j<NN; j++) {
         //printf("%d,",adj_bn_bool[i*NN+j]);
      //}
      //printf("] -- ");
   //}
   //printf("\n");

   //////////////////
   // mat_bn dataset
   //////////////////
   strcpy(dset_str, "mat_bn");expected_ndims=1;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&mat_bn,INT8);
   assert((int64_t)dims[0]==Nb);

   //////////////////
   // saf_bn dataset
   //////////////////
   strcpy(dset_str, "saf_bn");expected_ndims=1;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&saf_bn,FLOAT64);
   assert((int64_t)dims[0]==Nb);

   mymalloc((void **)&ssaf_bn, Nb*sizeof(Real));
   for (int64_t i=0; i<Nb; i++) {
      if (fcc_flag>0) 
         ssaf_bn[i] = (Real)(0.5/sqrt(2.0))*saf_bn[i]; //rescale for S*h/V and cast
      else
         ssaf_bn[i] = (Real)saf_bn[i]; //just cast
   }
   free(saf_bn);

   //printf("saf=");
   //for (int64_t i=0; i<Nb; i++) {
      //printf("%.16g, ",saf_bn[i]);
   //}
   //printf("\n");

   if (H5Fclose(file) != 0) {
      printf("error closing file %s",filename);
      assert(true==false);
   }
   else printf("closed file %s\n",filename);

   ////////////////////////////////////////////////////////////////////////
   // 
   // Read comms HDF5 dataset
   // 
   ////////////////////////////////////////////////////////////////////////
   strcpy(filename, "comms_out.h5");
   if (!check_file_exists(filename)) assert(true==false);

   file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

   //////////////////
   // integers
   //////////////////
   strcpy(dset_str, "Nt");
   read_h5_constant(file, dset_str,(void *)&Nt,INT64);
   printf("Nt=%ld\n",Nt);

   strcpy(dset_str, "Ns");
   read_h5_constant(file, dset_str,(void *)&Ns,INT64);
   printf("Ns=%ld\n",Ns);

   strcpy(dset_str, "Nr");
   read_h5_constant(file, dset_str,(void *)&Nr,INT64);
   printf("Nr=%ld\n",Nr);

   strcpy(dset_str, "Nr");
   read_h5_constant(file, dset_str,(void *)&Nr,INT64);
   printf("Nr=%ld\n",Nr);

   strcpy(dset_str, "diff");
   read_h5_constant(file, dset_str,(void *)&diff,BOOL);
   printf("diff=%d\n",diff);

   //////////////////
   // in_ixyz dataset
   //////////////////
   strcpy(dset_str, "in_ixyz");expected_ndims=1;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&in_ixyz,INT64);
   assert((int64_t)dims[0]==Ns);

   //printf("in_ixyz=");
   //for (int64_t i=0; i<Ns; i++) {
      //printf("%ld, ",in_ixyz[i]);
   //}
   //printf("\n");

   //////////////////
   // out_ixyz dataset
   //////////////////
   strcpy(dset_str, "out_ixyz");expected_ndims=1;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&out_ixyz,INT64);
   assert((int64_t)dims[0]==Nr);

   strcpy(dset_str, "out_reorder");expected_ndims=1;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&out_reorder,INT64);
   assert((int64_t)dims[0]==Nr);

   //printf("out_ixyz=");
   //for (int64_t i=0; i<Nr; i++) {
      //printf("%ld, ",out_ixyz[i]);
   //}
   //printf("\n");
   
   //////////////////
   // in_sigs dataset
   //////////////////
   strcpy(dset_str, "in_sigs");expected_ndims=2;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&in_sigs,FLOAT64);
   assert((int64_t)dims[0]==Ns);
   assert((int64_t)dims[1]==Nt);

   //printf("in_sigs=");
   //for (int64_t i=0; i<Ns; i++) {
      //printf("in %ld: ",i);
      //for (int64_t j=0; j<Nt; j++) {
         //printf("%.16g, ",in_sigs[i*Nt + j]);
      //}
      //printf("\n");
   //}
   //printf("\n");

   if (H5Fclose(file) != 0) {
      printf("error closing file %s",filename);
      assert(true==false);
   }
   else printf("closed file %s\n",filename);

   //not recommended to run single without differentiating input
   if (sizeof(Real)==4) assert(diff);

   ////////////////////////////////////////////////////////////////////////
   // 
   // Read materials HDF5 dataset
   // 
   ////////////////////////////////////////////////////////////////////////
   strcpy(filename, "sim_mats.h5");
   if (!check_file_exists(filename)) assert(true==false);

   file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

   //////////////////
   // integers
   //////////////////
   
   strcpy(dset_str, "Nmat");
   read_h5_constant(file, dset_str,(void *)&Nm,INT8);
   printf("Nm=%d\n",Nm);

   assert(Nm<=MNm);

   strcpy(dset_str, "Mb");expected_ndims=1;
   read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&Mb,INT8);

   for (int8_t i=0; i<Nm; i++) {
      printf("Mb[%d]=%d\n",i,Mb[i]);
   }

   //////////////////
   // DEF (RLC) datasets
   //////////////////
   mymalloc((void **)&mat_quads, Nm*MMb*sizeof(struct MatQuad)); //initalises to zero
   mymalloc((void **)&mat_beta, Nm*sizeof(Real)); 
   for (int8_t i=0; i<Nm; i++) {
      double *DEF; //for one material
      sprintf(dset_str,"mat_%02d_DEF",i);expected_ndims=2;
      read_h5_dataset(file, dset_str, expected_ndims, dims,(void **)&DEF,FLOAT64);
      assert((int8_t)dims[0] == Mb[i]);
      assert((int8_t)dims[1]==3);
      assert(Mb[i]<=MMb);

      for (int8_t j=0; j<Mb[i]; j++) {
         double D = DEF[j*3+0];
         double E = DEF[j*3+1];
         double F = DEF[j*3+2];
         printf("DEF[%d,%d]=[%.16g, %.16g, %.16g] \n",i,j,D,E,F);

         //see 2016 ISMRA paper
         double Dh = D/Ts;
         double Eh = E;
         double Fh = F*Ts;

         double b = 1.0/(2.0*Dh+Eh+0.5*Fh);
         double bd = b*(2.0*Dh-Eh-0.5*Fh);
         double bDh = b*Dh;
         double bFh = b*Fh;
         assert(!isinf(b));assert(!isnan(b));
         assert(!isinf(bd));assert(!isnan(bd));

         int32_t mij = (int32_t)MMb*i+j;
         mat_quads[mij].b   = (Real)b;
         mat_quads[mij].bd  = (Real)bd;
         mat_quads[mij].bDh = (Real)bDh;
         mat_quads[mij].bFh = (Real)bFh;
         mat_beta[i] += (Real)b;
      }
      free(DEF);
   }
   /*
   for (int8_t i=0; i<Nm; i++) {
      printf("b[%d]=",i);
      for (int8_t j=0; j<Mb[i]; j++) {
         int32_t mij = (int32_t)MMb*i+j;
         printf(" %.16g, ",mat_quads[mij].b);
      }
      printf("\n");
      printf("bd[%d]=",i);
      for (int8_t j=0; j<Mb[i]; j++) {
         int32_t mij = (int32_t)MMb*i+j;
         printf(" %.16g, ",mat_quads[mij].bd);
      }
      printf("\n");
      printf("bDh[%d]=",i);
      for (int8_t j=0; j<Mb[i]; j++) {
         int32_t mij = (int32_t)MMb*i+j;
         printf(" %.16g, ",mat_quads[mij].bDh);
      }
      printf("\n");
      printf("bFh[%d]=",i);
      for (int8_t j=0; j<Mb[i]; j++) {
         int32_t mij = (int32_t)MMb*i+j;
         printf(" %.16g, ",mat_quads[mij].bFh);
      }
      printf("\n");
      printf("mat_beta[%d]= %.16g \n",i,mat_beta[i]);
   }

   for (int8_t i=0; i<Nm; i++) {
      printf("Mb[%d]=%d\n",i,Mb[i]);
   }
   */

   if (H5Fclose(file) != 0) {
      printf("error closing file %s",filename);
      assert(true==false);
   }
   else printf("closed file %s\n",filename);

   ////////////////////////////////////////////////////////////////////////
   // 
   // Checks and repacking
   // 
   ////////////////////////////////////////////////////////////////////////
   
   //////////////////
   // check bn_ixyz 
   //////////////////
   check_inside_grid(bn_ixyz, Nb, Nx, Ny, Nz);
   printf("bn_ixyz checked\n");

   //////////////////
   // check adj_bn_bool and mat_bn 
   //////////////////
   for (int64_t i=0; i<Nb; i++) {
      bool at_least_one_not_adj = false;
      bool all_not_adj = true;
      for (int8_t j=0; j<NN; j++) { 
         bool adj = adj_bn_bool[i*NN+j];
         at_least_one_not_adj |= !adj;
         all_not_adj &= !adj;
      }
      assert(at_least_one_not_adj);
      if (all_not_adj) assert(mat_bn[i]==-1);
   }
   printf("checked adj_bn against mat_bn.\n");

   //////////////////
   // bit-pack and check adj_bn
   //////////////////
   mymalloc((void **)&adj_bn, Nb*sizeof(uint16_t));
   //#pragma omp parallel for
   for (int64_t i=0; i<Nb; i++) {
      for (int8_t j=0; j<NN; j++) { 
         SET_BIT_VAL(adj_bn[i],j,adj_bn_bool[i*NN+j]); 
      }
   }
   printf("adj_bn filled\n");

   //#pragma omp parallel for
   for (int64_t i=0; i<Nb; i++) {
      for (int8_t j=0; j<NN; j++) { //avoids race conditions
         assert(GET_BIT(adj_bn[i],j) == adj_bn_bool[i*NN+j]);
      }
   }
   printf("adj_bn double checked\n");
   free(adj_bn_bool);

   //////////////////
   // calculate K_bn from adj_bn
   //////////////////
   mymalloc((void **)&K_bn, Nb*sizeof(int8_t));
   //#pragma omp parallel for
   for (int64_t nb=0; nb<Nb; nb++) {
      K_bn[nb] = 0;
      for (uint8_t nn=0; nn<NN; nn++) {
         K_bn[nb] += GET_BIT(adj_bn[nb],nn);
      }
   }
   printf("K_bn calculated\n");

   //////////////////
   // bit-pack and check bn_mask
   //////////////////
   //make compressed bit-mask 
   int64_t Nbm = (Npts-1)/8+1;
   mymalloc((void **)&bn_mask, Nbm); //one bit per 
   for (int64_t i=0; i<Nb; i++) {
      int64_t ii = bn_ixyz[i];
      SET_BIT(bn_mask[ii>>3],ii%8);
   }

   // create bn_mask_raw to double check
   bool *bn_mask_raw;
   mymalloc((void **)&bn_mask_raw, Npts*sizeof(bool));
   //#pragma omp parallel for
   for (int64_t i=0; i<Nb; i++) {
      int64_t ii = bn_ixyz[i];
      assert(ii<Npts);
      bn_mask_raw[ii] = true;
   }
   printf("bn_mask_raw filled\n");
   //#pragma omp parallel for
   for (int64_t j=0; j<Nbm; j++) {
      for (int64_t q=0; q<8; q++) { //avoid race conditions
         int64_t i = j*8+q;
         if (i<Npts) assert(GET_BIT(bn_mask[i>>3],i%8) == bn_mask_raw[i]);
      }
   }
   printf("bn_mask double checked\n");
   free(bn_mask_raw);

   //count Nbl 
   Nbl = 0;
   for (int64_t i=0; i<Nb; i++) {
      Nbl += mat_bn[i]>=0;
   }
   printf("Nbl = %ld\n",Nbl);
   mymalloc((void **)&mat_bnl, Nbl*sizeof(int8_t));
   mymalloc((void **)&bnl_ixyz, Nbl*sizeof(int64_t));
   mymalloc((void **)&ssaf_bnl, Nbl*sizeof(Real));
   {
      int64_t j=0;
      for (int64_t i=0; i<Nb; i++) {
         if (mat_bn[i]>=0) {
            mat_bnl[j]=mat_bn[i];
            ssaf_bnl[j]=ssaf_bn[i];
            bnl_ixyz[j]=bn_ixyz[i];
            j++;
         }
      }
      assert(j==Nbl);
   }
   free(mat_bn);
   free(ssaf_bn);

   printf("separated non-rigid bn\n");

   //ABC ndoes
   int64_t Nyf;
   Nyf = (fcc_flag==2)? 2*(Ny-1) : Ny; //full Ny dim, taking into account FCC fold
   Nba = 2*(Nx*Nyf+Nx*Nz+Nyf*Nz) - 12*(Nx+Nyf+Nz) + 56;
   if (fcc_flag>0) Nba /= 2;

   mymalloc((void **)&bna_ixyz, Nba*sizeof(int64_t));
   mymalloc((void **)&Q_bna, Nba*sizeof(int8_t));
   {
      int64_t ii = 0;
      for (int64_t ix=1; ix<Nx-1; ix++) {
         for (int64_t iy=1; iy<Nyf-1; iy++) {
            for (int64_t iz=1; iz<Nz-1; iz++) {

               if ((fcc_flag>0) && (ix+iy+iz)%2==1) continue;

               int8_t Q = 0;
               Q += ((ix==1) || (ix==Nx-2));
               Q += ((iy==1) || (iy==Nyf-2));
               Q += ((iz==1) || (iz==Nz-2));
               if (Q>0) {
                  if ((fcc_flag==2) && (iy>=Nyf/2)) {
                     bna_ixyz[ii] = ix*Nz*Ny + (Nyf-iy-1)*Nz + iz; //index on folded grid
                  }
                  else {
                     bna_ixyz[ii] = ix*Nz*Ny + iy*Nz + iz;
                  }
                  Q_bna[ii] = Q;
                  ii += 1;
               }
            }
         }
      }
      assert(ii==Nba);
      printf("ABC nodes\n");
      if (fcc_flag==2) { //need to sort bna_ixyz 
         int64_t *bna_sort_keys;
         mymalloc((void **)&bna_sort_keys, Nba*sizeof(int64_t));
         qsort_keys(bna_ixyz,bna_sort_keys,Nba);

         //now sort corresponding Q_bna array
         int8_t *Q_bna_sorted;
         int8_t *Q_bna_unsorted;
         mymalloc((void **)&Q_bna_sorted, Nba*sizeof(int8_t));
         //swap pointers
         Q_bna_unsorted = Q_bna;
         Q_bna = Q_bna_sorted;

         for (int64_t cc=0; cc<Nba; cc++) {
            Q_bna_sorted[cc] = Q_bna_unsorted[bna_sort_keys[cc]];
         }
         free(bna_sort_keys);
         free(Q_bna_unsorted);
         printf("sorted ABC nodes for FCC/GPU\n");
      }
   }

   //for outputs
   mymalloc((void **)&u_out, Nr*Nt*sizeof(double));
   /*------------------------
    * ATTACH 
   ------------------------*/
   sd->Ns   = Ns;
   sd->Nr   = Nr;
   sd->Nt   = Nt;
   sd->Npts = Npts;
   sd->Nx   = Nx;
   sd->Ny   = Ny;
   sd->Nz   = Nz;
   sd->Nb   = Nb;
   sd->Nbl  = Nbl;
   sd->Nba  = Nba;
   sd->l    = l;
   sd->l2   = l2;
   sd->fcc_flag  = fcc_flag;
   sd->Nm   = Nm;
   sd->NN   = NN;
   sd->a2   = a2;
   sd->a1   = a1;
   sd->sl2  = sl2;
   sd->lo2  = lo2;
   sd->Mb   = Mb;
   sd->bn_ixyz   = bn_ixyz;
   sd->bnl_ixyz  = bnl_ixyz;
   sd->bna_ixyz  = bna_ixyz;
   sd->Q_bna     = Q_bna;
   sd->adj_bn    = adj_bn;
   sd->ssaf_bnl  = ssaf_bnl;
   sd->bn_mask   = bn_mask;
   sd->mat_bnl   = mat_bnl;
   sd->K_bn      = K_bn;
   sd->out_ixyz  = out_ixyz;
   sd->out_reorder  = out_reorder;
   sd->in_ixyz   = in_ixyz;
   sd->in_sigs   = in_sigs;
   sd->u_out     = u_out;
   sd->mat_beta  = mat_beta;
   sd->mat_quads  = mat_quads;
}

//free everything
void free_sim_data(struct SimData *sd) {
   /*------------------------
    * FREE WILLY 
   ------------------------*/
   free(sd->bn_ixyz);
   free(sd->bnl_ixyz);
   free(sd->bna_ixyz);
   free(sd->Q_bna);
   free(sd->adj_bn);
   free(sd->mat_bnl);
   free(sd->bn_mask);
   free(sd->ssaf_bnl);
   free(sd->K_bn);
   free(sd->in_ixyz);
   free(sd->out_ixyz);
   free(sd->out_reorder);
   free(sd->in_sigs);
   free(sd->u_out);
   free(sd->Mb);
   free(sd->mat_beta);
   free(sd->mat_quads);
   printf("sim data freed\n");
}

//read HDF5 files
void read_h5_dataset(hid_t file, char *dset_str, int ndims, hsize_t *dims, void **data_array, TYPE t) {
   hid_t dset,dspace;
   uint64_t N=0;
   //herr_t status;

   dset = H5Dopen(file, dset_str, H5P_DEFAULT);
   dspace = H5Dget_space(dset);
   assert(H5Sget_simple_extent_ndims(dspace) == ndims);
   H5Sget_simple_extent_dims(dspace, dims, NULL);
   if (ndims==1) {
      //printf("size dim 0 = %llu\n",dims[0]);
      N = dims[0];
   }
   else if (ndims==2) {
      //printf("size dim 0 = %llu\n",dims[0]);
      //printf("size dim 1 = %llu\n",dims[1]);
      N = dims[0]*dims[1];
   }
   else {
      assert(true==false);
   }
   switch(t) {
      case FLOAT64:
         *data_array = (double*)malloc(N*sizeof(double));
         break;
      case FLOAT32:
         *data_array = (double*)malloc(N*sizeof(float));
         break;
      case INT64:
         *data_array = (int64_t*)malloc(N*sizeof(int64_t));
         break;
      case INT8:
         *data_array = (int8_t*)malloc(N*sizeof(int8_t));
         break;
      case BOOL:
         *data_array = (bool*)malloc(N*sizeof(bool));
         break;
      default:
         assert(true==false);
   }
   if (*data_array == NULL) {
      printf("Memory allocation failed");
      assert(true==false); //to break
   }
   herr_t status;
   switch(t) {
      case FLOAT64:
         status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data_array);
         break;
      case FLOAT32:
         status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data_array);
         break;
      case INT64:
         status = H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data_array);
         break;
      case INT8:
         status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data_array);
         break;
      case BOOL: //bool read in as INT8
         status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, *data_array);
         break;
      default:
         assert(true==false);
   }

   if (status != 0) {
      printf("error reading dataset: %s\n",dset_str);
      assert(true==false);
   }
   if (H5Dclose(dset) != 0) {
      printf("error closing dataset: %s\n",dset_str);
      assert(true==false);
   }
   else {
      printf("read and closed dataset: %s\n",dset_str);
   }
}

//read scalars from HDF5 datasets
void read_h5_constant(hid_t file, char *dset_str, void *data_container, TYPE t) {
   hid_t dset,dspace;

   dset = H5Dopen(file, dset_str, H5P_DEFAULT);
   dspace = H5Dget_space(dset);
   assert(H5Sget_simple_extent_ndims(dspace) == 0);
   herr_t status;
   switch(t) {
      case FLOAT64:
         status = H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_container);
         break;
      case INT64:
         status = H5Dread(dset, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_container);
         break;
      case BOOL:
         status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_container);
         break;
      case INT8:
         status = H5Dread(dset, H5T_NATIVE_INT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_container);
         break;
      default:
         assert(true==false);
   }

   if (status != 0) {
      printf("error reading dataset: %s\n",dset_str);
      assert(true==false);
   }
   if (H5Dclose(dset) != 0) {
      printf("error closing dataset: %s\n",dset_str);
      assert(true==false);
   }
   else {
      printf("read constant: %s\n",dset_str);
   }
}

//print last samples of simulation (for correctness checking..)
void print_last_samples(struct SimData *sd) {
   int64_t Nt = sd->Nt;
   int64_t Nr = sd->Nr;
   double *u_out = sd->u_out;
   int64_t *out_reorder = sd->out_reorder;
   //print last samples
   printf("RAW OUTPUTS\n");
   for (int64_t nr=0; nr<Nr; nr++) {
      printf("receiver %ld\n",nr);
      for (int64_t n=Nt-5; n<Nt; n++) {
         printf("sample %ld: %.16e\n",n,u_out[out_reorder[nr]*Nt + n]);
      }
   }
}

//scale input to be in middle of floating-point range 
void scale_input(struct SimData *sd) {
   double *in_sigs = sd->in_sigs;
   int64_t Nt = sd->Nt;
   int64_t Ns = sd->Ns;

   //normalise input signals (and save gain)
   double max_in = 0.0;
   for (int64_t n=0; n<Nt; n++) {
      for (int64_t ns=0; ns<Ns; ns++) {
          max_in = MAX(max_in,fabs(in_sigs[(int64_t)ns*Nt+n]));
      }
   }
   double aexp = 0.5; //normalise to middle power of two
   int32_t pow2 = (int32_t)round(aexp*REAL_MAX_EXP+(1-aexp)*REAL_MIN_EXP);
   //int32_t pow2 = 0; //normalise to one
   double norm1 = pow(2.0,pow2);
   double inv_infac = norm1/max_in;
   double infac = 1.0/inv_infac;

   printf("max_in = %.16e, pow2 = %d, norm1 = %.16e, inv_infac = %.16e, infac = %.16e\n",max_in,pow2,norm1,inv_infac,infac);

   //normalise input data
   for (int64_t ns=0; ns<Ns; ns++) {
      for (int64_t n=0; n<Nt; n++) {
         in_sigs[ns*Nt + n] *= inv_infac;
      }
   }

   sd->infac = infac;
   sd->in_sigs = in_sigs;
}

//undo that scaling
void rescale_output(struct SimData *sd) {
   double *u_out = sd->u_out;
   int64_t Nt = sd->Nt;
   int64_t Nr = sd->Nr;
   double infac = sd->infac;

   for (int64_t nr=0; nr<Nr; nr++) {
      for (int64_t n=0; n<Nt; n++) {
         u_out[nr*Nt + n] *= infac;
      }
   }

   sd->u_out   = u_out;
}

//save outputs in HDF5 (to processed subsequently with Python script)
void write_outputs(struct SimData *sd) {
   //write outputs in correct order
   int64_t *out_reorder = sd->out_reorder;
   int64_t Nt = sd->Nt;
   int64_t Nr = sd->Nr;
   char dset_str[80];
   char filename[80];
   hsize_t dims[2];
   herr_t status;
   hid_t file, space, dset;

   double *u_out;
   mymalloc((void **)&u_out, Nt*Nr*sizeof(double));
   for (int64_t nr=0; nr<Nr; nr++) {
      for (int64_t n=0; n<Nt; n++) {
         u_out[nr*Nt + n] = sd->u_out[out_reorder[nr]*Nt + n];
      }
   }

   dims[0] = Nr;
   dims[1] = Nt;
   strcpy(filename, "sim_outs.h5");
   strcpy(dset_str, "u_out");
   file = H5Fcreate (filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   space = H5Screate_simple (2, dims, NULL);
   dset = H5Dcreate (file, dset_str, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   status = H5Dwrite (dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, u_out);
   if (status != 0) {
      printf("error writing dataset\n");
      assert(true==false);
   }

   status = H5Dclose (dset);
   if (status != 0) {
      printf("error closing dataset\n");
      assert(true==false);
   }

   status = H5Sclose (space);
   if (status != 0) {
      printf("error closing dataset space\n");
      assert(true==false);
   }

   status = H5Fclose (file);
   if (status != 0) {
      printf("error closing file\n");
      assert(true==false);
   }
   printf("wrote output dataset\n");
   //free this temp array
   free(u_out);
}

#endif
