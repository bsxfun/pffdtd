##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: Makefile
#
# Description: Simple Makefile.  Run 'make clean && make all'
# 					This requires make HDF5, OpenMP, and Nvidia CUDA toolkit (and gcc and make)
#
##############################################################################

CC = gcc
NVCC = nvcc
IDIR = .

#set the HDF5 env variables for your system

#on Ubuntu (install libhdf5-dev with aptitude)
HDF5_INC = /usr/include/hdf5/serial/
HDF5_LIB = /usr/lib/x86_64-linux-gnu/hdf5/serial/ 

#on Arch (install hdf5 with pacman)
#HDF5_INC = /usr/include/
#HDF5_LIB = /usr/lib/

#on Centos (install hdf5-devel with yum)
#HDF5_INC = /usr/include/
#HDF5_LIB = /usr/lib64/

#if you download and extract shared libraries here, use something like this
#HDF5_INC ?= ./hdf5/include/
#HDF5_LIB ?= ./hdf5/lib/ #will also need to add this to LD_LIBRARY_PATH when running .x


CFLAGS = -I$(IDIR) -Wall -Wextra -Wundef -Wpedantic -std=c99
CFLAGS += -I$(HDF5_INC)
CFLAGS += -O3
LDFLAGS = -fopenmp -lgomp -lm
LDFLAGS += -L$(HDF5_LIB) -lhdf5

NVCCFLAGS = -I$(IDIR) -O3 -arch=sm_35
NVCCFLAGS += -x cu #compile .c or .h as cuda
NVCCFLAGS += -I$(HDF5_INC)
#NVCCFLAGS += -ptx
NVCCFLAGS += -Xcompiler -Wall -Xcompiler -Wextra #not using -Wpedantic -Wundef with nvcc
CULDFLAGS = -Xcompiler -fopenmp -lgomp -lm
CULDFLAGS += -L$(HDF5_LIB) -lhdf5

SRC= fdtd_main.c

cpu: cpu_double cpu_single 

gpu: gpu_double gpu_single 

all: cpu gpu

#build cpu/gpu double/single precision binaries
cpu_double:
	$(CC) $(CFLAGS) $(SRC) -o fdtd_main_cpu_double.x -DPRECISION=2 -DUSING_CUDA=false $(LDFLAGS)

cpu_single:
	$(CC) $(CFLAGS) $(SRC) -o fdtd_main_cpu_single.x -DPRECISION=1 -DUSING_CUDA=false $(LDFLAGS)

gpu_double:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o fdtd_main_gpu_double.x -DPRECISION=2 -DUSING_CUDA=true $(CULDFLAGS)

gpu_single:
	$(NVCC) $(NVCCFLAGS) $(SRC) -o fdtd_main_gpu_single.x -DPRECISION=1 -DUSING_CUDA=true $(CULDFLAGS)

.PHONY: clean

clean:
	rm -f *.x
