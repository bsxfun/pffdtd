// vim: tabstop=3: ai
///////////////////////////////////////////////////////////////////////////////
// This file is a part of PFFDTD.
//
// PFFTD is released under the MIT License.
// For details see the LICENSE file.
//
// Copyright 2021 Brian Hamilton.
//
// File name: fdtd_main.c
//
// Description: Main entry point of the CPU/GPU PFFDTD engines.
//
///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdint.h> //uint types
#include <assert.h> //for assert
#include <time.h> //date and time

#include <helper_funcs.h>
#include <fdtd_common.h>
#include <fdtd_data.h>

#ifndef USING_CUDA //this is defined in Makefile
#error "forgot to define USING_CUDA"
#endif

#if USING_CUDA
   #include <gpu_engine.h>
#else
   #include <cpu_engine.h>
#endif

int main() {
   //print date & time to start
   time_t t0;
   time(&t0);
   printf("--Date and time: %s", ctime(&t0)); //prints new line

   // double-check on 64-bit system
   assert(sizeof(size_t)==sizeof(int64_t));

   struct SimData sd;

   //load, scale, run, rescale, free..
   load_sim_data(&sd);
   scale_input(&sd);
   run_sim(&sd);
   rescale_output(&sd);
   write_outputs(&sd);
   print_last_samples(&sd);
   free_sim_data(&sd);

   //print date & time to end
   time(&t0);
   printf("--Date and time: %s", ctime(&t0));
   return 0;
}
