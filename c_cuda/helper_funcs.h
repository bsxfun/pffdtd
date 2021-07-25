// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: helper_funcs.h
//
// Description: Header-only misc function definitions not specific to FDTD simulation
//
///////////////////////////////////////////////////////////////////////////////

#ifndef _HELPER_FUNCS_H
#define _HELPER_FUNCS_H
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
#ifndef _STAT_H
#include <sys/stat.h> //for stat
#endif
#ifndef _STRING_H
#include <string.h> //for memset
#endif

//some useful macros
#ifndef MIN
   #define MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
   #define MAX(a,b) (((a)>(b))?(a):(b))
#endif
#ifndef CLAMP
   #define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif
#ifndef SWAP
   #define SWAP(a, b) (((a) ^= (b)), ((b) ^= (a)), ((a) ^= (b)))
#endif
#ifndef DIV_CEIL
   #define DIV_CEIL(x,y) (((x) + (y) - 1) / (y)) //this works for x≥0 and y>0
#endif
#define GET_BIT(var,pos) (((var)>>(pos)) & 1)
#define SET_BIT(var,pos) ((var) |= (1ULL<<(pos)))
#define CLEAR_BIT(var,pos) ((var) &= ~(1ULL<<(pos)))
#define SET_BIT_VAL(var,pos,val) ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

typedef enum {
   FLOAT64,
   FLOAT32,
   INT64,
   INT8,
   BOOL,
} TYPE;


bool check_file_exists(char *filename);
void mymalloc(void **arr, uint64_t Nbytes);
int cmpfunc_int64 (const void * a, const void * b);
int cmpfunc_int64_keys (const void * a, const void * b);
void qsort_keys(int64_t *val_arr, int64_t *key_arr, int64_t N);

//self-explanatory
bool check_file_exists(char *filename) {
   struct stat st;
   bool file_exists = stat(filename, &st)==0;
   if (!file_exists) printf("%s doesn't exist!\n",filename);
   return file_exists;
}

//malloc check malloc, and initialise to zero
//hard stop program if failed
void mymalloc(void **arr, uint64_t Nbytes) {
   *arr = malloc(Nbytes);
   if (*arr == NULL) {
      printf("Memory allocation failed");
      assert(true==false); //to break
   }
   //initialise to zero
   memset(*arr,0,(size_t)Nbytes);
}

//for sorting int64 arrays and returning keys
struct sort_int64_struct {
   int64_t val;
   int64_t idx;
};
//comparator (for FCC ABC nodes)
int cmpfunc_int64 (const void * a, const void * b) {
   if ( *(const int64_t*)a < *(const int64_t*)b ) return -1;
   if ( *(const int64_t*)a > *(const int64_t*)b ) return  1;
   return 0;
}
//comparator with indice keys (for FCC ABC nodes)
int cmpfunc_int64_keys (const void * a, const void * b) {
   if ( (*(const struct sort_int64_struct*)a).val < (*(const struct sort_int64_struct*)b).val) return -1;
   if ( (*(const struct sort_int64_struct*)a).val > (*(const struct sort_int64_struct*)b).val) return  1;
   return 0;
}

//sort and return indices
void qsort_keys(int64_t *val_arr, int64_t *key_arr, int64_t N) {
   struct sort_int64_struct *struct_arr;
   struct_arr = (struct sort_int64_struct*) malloc(N*sizeof(struct sort_int64_struct));
   if (struct_arr == NULL) {
      printf("Memory allocation failed");
      assert(true==false); //to break
   }
   for (int64_t i=0; i<N; i++) {
      struct_arr[i].val = val_arr[i];
      struct_arr[i].idx = i;
   }
   qsort(struct_arr,N,sizeof(struct sort_int64_struct),cmpfunc_int64_keys);
   for (int64_t i=0; i<N; i++) {
      val_arr[i] = struct_arr[i].val;
      key_arr[i] = struct_arr[i].idx;
   }
   free(struct_arr);
}
#endif
