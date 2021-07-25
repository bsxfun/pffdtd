# PFFDTD (pretty fast FDTD)

![PFFDTD Screenshot](https://github.com/bsxfun/pffdtd/raw/main/screenshot.png)

PFFDTD is an implementation of finite-difference time-domain (FDTD) simulation for 3D room acoustics, which includes an accompanying set of tools for simulation setup and processing of input/output signals.  This software is intended for research use with powerful workstations or single-node remote servers with one or more Nvidia GPUs (using CUDA). PFFDTD was designed be "pretty fast" when run on GPUs – at least for FDTD simulations (the name is mostly intended as a pun).

## Features
- Multi-GPU-accelerated execution (with CUDA-enabled Nvidia GPUs)
- Multi-threaded CPU execution
- Python-based voxelization with CPU multiprocessing
- Energy conservation to machine precision (numerical stability)
- 7-point Cartesian and 13-point face-centered cubic (FCC) schemes
- Frequency-dependent impedance boundaries
- Works with non-watertight models
- Stability safeguards for single-precision operation
- A novel staircase boundary-surface-area correction scheme
- 3D visualization

## Installation
PFFDTD is designed to run on a Linux system (e.g. Ubuntu/Centos/Arch).

### Installation (Python)
PFFDTD requires at least Python 3.9 to run, with additional required packages in pip_requirements.txt (for pip) or
conda_pffdtd.yml (for conda).  Conda (or miniconda) is recommended to get started with a PFFDTD-specific conda environment (see .yml file).

### Installation (C/CUDA)
To compile, run 'make all' in the c_cuda folder.  

You will need the CUDA toolkit, which you can install from [this link](https://developer.nvidia.com/cuda-downloads).

You will also need HDF5 runtime and development files.  In Ubuntu it suffices to install the libhdf5-dev package.  For Centos install hdf5-devel.  You can also link to the shared libraries [which you can download here](https://www.hdfgroup.org/downloads/hdf5/).  Check the Makefile for environment variables to set.

## Examples
There are two models provided with the code, which you can start to run from the four test scripts in this root folder.

## Starting a project from scratch
To start a project (a model) from scratch, follow the provided examples, but essentially you will need to:

1. Build a Sketchup model, set source/receiver locations in CSV files, and export to a JSON file.
2. Fit absorption/impedance data to the passive boundary impedance model used [BHBS16].  Only a simple routine is provided to fit to 11 octave-band absorption coefficients (16Hz to 16kHz).
3. Fill out a setup script to configure the model and simulation.
3. Run your setup script, which in turns runs a 'voxelization' routine and sets up input/output grid positions and signals.
4. Simulate your model from the exported .h5 data files with either the CPU-based Python code (which includes energy + visualization), or the C/CPU-based engine, or the C/CUDA/GPU-based engine.
5. Post-process the output files from output .h5 files (if not just a run for visualization purposes).

The setup phase (signals + voxelization) can be carried out on a different machine to the simulation itself.  This is useful if you're using GPUs on a remote server.  There are GZIP compression options in the HDF5 exporting if
network uploading is a bottleneck (you can also repack HDF5 files with 'h5repack').  Note, however, the voxelization phase is compute-intensive.  It is best to have a many-core CPU server or workstation for this, or least a
powerful laptop.

### Sketchup
After building a Sketchup model of your room/scene, you can export it using the provided Sketchup plugin (.rbz file under ruby_SU folder), which exports the model and source/receiver positions (defined in separate CSV files – see examples) to a JSON file.  Walls should be labelled with Sketchup Materials (which you can rename as necessary), and you should pay attention to the orientation of faces.  Unlabelled materials are taken to be rigid.  It is important to only label the 'active' side of a surface in order to save on computation time and memory in the FDTD scheme (non-rigid boundary nodes require extra state for internal ODEs).  It is possible to have two-sided materials if needed, but both sides must be the same material.  The model does not need to be closed (watertight) but it is good practice to have that.  The exported model is expected to have at least four triangles.  It only exports visible entities, and only Face entities (not Groups or Components – explode them first).  Layers (Tags) are not taken into account.  

The rest of the code works from the JSON export, so it's possible to export to this JSON format from other CAD software (in theory, but you would need to develop plugins for that).  There is a
Three.JS based viewer in a [separate repository](https://github.com/bsxfun/pffdtd_js_model_viewer) to double check the export (and orientation of faces).

### Python
Follow the test script examples to set up your simulation.  You need to choose input-signal types, link up materials to impedance data files (.h5 HDF5 format files), and choose your grid resolution.  It helps to know some basics of Python/Numpy and FDTD here.  When you run the script you will get a bunch of info related to the scheme, and it will give you an estimate of memory requirements for the simulation, and save some necessary data in .h5 files.

### Simulation engine
Once you have your .h5 files ready in some folder, you can run the Python FDTD engine (pointing to the folder), or run the compiled C binaries from that folder.  You will of course need Nvidia GPUs to run the CUDA version of the FDTD engine.  The different engine implementations produce identical results (to within machine accuracy).  Generally, the Python version is for visualization purposes and correctness checking (comparing outputs or calculating system energy), whereas the CUDA code is meant for larger simulations.  Single-precision GPU execution will generally be the fastest (and cheapest) to run.

### Post-processing
When the simulation has completed there will be a 'sim_outs.h5' file which has the raw signal read from grid locations.  You can process these signals to get final output files, which do need some
cleanup to remove, e.g., high frequencies with too much error (numerical dispersion) or to resample to a standard audio rate like 48kHz.  There are also filters to apply air absorption to the output
responses [Ham21a,Ham21b].  PFFDTD is only designed to output monoaural RIRs, but you can build arrays of outputs and feed those into frequency-domain microphone-array processing tools [(e.g.,)](http://research.spa.aalto.fi/projects/sparta_vsts/) for spatial audio, or encode Ambisonics directly in the time-domain [BPH19] (a similar approach can be used for directive sources [BAH19]).  

## Enhancements
This code largely implements algorithms that have already been published (see key reference list below) but it does feature some enhancements that have not appeared in articles that are worth
mentioning below.  

### Operation in single precision
Instabilities can occur with FDTD simulations in finite-precision if you wait long enough, and that length will depend on the precision chosen and the particular simulation.  You should not experience instabilities in double precision (indeed, the code conserves energy to machine precision).  Single precision is generally faster and uses less memory, but operating in single precision can be give rise to instabilities (due to rounding errors) (see, e.g., [HBW14]).  

If you choose to use single precision, this code has a few safeguards to mitigate/delay long-time instabilities.  First, the eigenvalues of the underlying finite-difference Laplacian operator are perturbed to prevent DC-mode instabilities.  In view of a matrix operator, off-diagonal elements are rounded towards zero (RTZ), while diagonal elements are shifted (on the order of machine epsilon) such that the finite-difference Laplacian operator remains negative semi-definite (a condition for stability).  These DC-mode safeguards are only active in the single-precision CUDA version (they aren't needed in double precision).  Secondly, the input signal to the scheme (e.g., an impulse) must be differentiated (using 'diff_source' option for sim_setup.py) to remove any DC input.  At post-processing, the output will be integrated with a combined integrator/high-pass Butterworth filter [HPB20], which integrates while suppressing any leftover problematic DC modes.

### Staircasing in FDTD
Staircase effects in regular-grid FDTD schemes is a known issue [BHBS16].  In essence, surfaces areas of boundaries are incorrectly estimated, leading to an overestimation of decay times.  A novel correction scheme is provided in this code, based on the idea of an effective surface area.  This concept is simple – it amounts to a weighting factor on the boundary-surface based on the inner product between the target boundary surface and a voxel boundary surface.  Surface area errors before and after correction are tabulated in the voxelizer, and for refined grids the corrections brings errors down from as high as 50% to generally below 1% (going to zero in the limit of small cells).  This helps provide more consistent estimates of decay times, and makes the simulation more robust to changes in grid resolution or scene rotations, while keeping the boundary updates efficient to implement and carry out on GPUs.

### FCC scheme
For efficiency the 13-point FCC scheme is recommended over the 7-point Cartesian scheme, as it is typically requires ~5x less memory than the Cartesian scheme for a 1%-2% levels of dispersion error [HW13,Ham16].  However, the FCC scheme is tricky to implement due to its setting on a non-Cartesian grid.  One solution to this has been compress the FCC grid like an accordion so it fits on a Cartesian grid, but there's a better solution implemented in this code.  Namely, the FCC subgrid is folded onto itself across one dimension such that the stencil operation is uniform throughout (the old solution has some branching involved).  Only the C and C/CUDA versions have this.  The Python engine version uses a straightforward, yet redundant, Cartesian grid (aka using the CCP scheme). 

## Performance benchmarks
See (TODO:) for some performance benchmark results using single-node Nvidia GPUs servers, with GPU architectures ranging from Kepler to Ampere.  This software has been tested with up to 30b nodes on the FCC grid (~250GB).  For even larger, multi-node (MPI-based) FDTD simulations, see [ParallelFDTD](https://github.com/AaltoRSE/ParallelFDTD) and [SCM18].

## License
This software is released under the MIT license.  See the LICENSE file for details.  The Sketchup models found in the data/models folder are released under different licenses; see the README files in their corresponding folders.

## Credits
The development of this code was not funded by any body or institution, but some credit can be given to grants ERC-StG-2011-279068-NESS and ERC-PoC-2016-WRAM, which funded many of the underlying (published) simulation algorithms developed within the Acoustics & Audio Group at the University of Edinburgh (see, e.g., list of references below).  Some of the performance benchmarks of this code were carried out on GPUs (K80 / GTX1080Ti) paid for by those grants.  Also, the Sketchup models provided were initially created by Heather Lai [LH20] and Nathaniel Fletcher [HWFB16] .

## Citing this work
If PFFDTD contributes to an academic publication, please cite it as:

      @misc{hamilton2021pffdtd,
        title = {PFFDTD Software},
        author = {Brian Hamilton},
        note = {https://github.com/bsxfun/pffdtd},
        year = {2021}
      }

## Third-party credits
PFFDTD relies on a number of open-source projects to work properly:

- [Python](https://github.com/python)
- [numpy](https://github.com/numpy/numpy)
- [numba](https://github.com/numba/numba)
- [HDF5](https://www.hdfgroup.org/solutions/hdf5)
- [scipy](https://github.com/scipy/scipy)
- [mayavi](https://github.com/enthought/mayavi)
- [polyscope](https://github.com/nmwsharp/polyscope)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [resampy](https://github.com/bmcfee/resampy)

The above list is non-exhaustive.  Use of the third-party software, libraries or code referred to above may be governed by separate licenses.

## Some background references
[HW13] B. Hamilton and C. J. Webb. Room acoustics modelling using GPU-accelerated finite difference and
finite volume methods on a face-centered cubic grid. In Proc. Digital Audio Effects (DAFx), pages
336–343, Maynooth, Ireland, September 2013.

[HBW14] B. Hamilton, S. Bilbao, and C. J. Webb. Revisiting implicit finite difference schemes for 3-D room
acoustics simulations on GPU. In Proc. Digital Audio Effects (DAFx), pages 41–48, Erlangen, Germany,
September 2014.

[BHBS16] S. Bilbao, B. Hamilton, J. Botts, and L. Savioja. Finite volume time domain room acoustics simulation
under general impedance boundary conditions. IEEE/ACM Trans. Audio, Speech, Lang. Process.,
24(1):161–173, 2016.

[Ham16] B. Hamilton. Finite Difference and Finite Volume Methods for Wave-based Modelling of Room
Acoustics. Ph.D. thesis, University of Edinburgh, 2016.

[HWFB16] B. Hamilton, C. J. Webb, N. D. Fletcher, and S. Bilbao. Finite difference room acoustics simulation with
general impedance boundaries and viscothermal losses in air: Parallel implementation on multiple
GPUs. In Proc. Int. Symp. Music & Room Acoust., La Plata, Argentina, September 2016.

[SCM18] J. Saarelma, J. Califa, and R. Mehra. Challenges of distributed real-time finite-difference time-domain room acoustic simulation for auralization. In AES Int. Conf. Spatial Reproduction, July 2018.

[BPH19] S. Bilbao, A. Politis, and B. Hamilton. Local time-domain spherical harmonic spatial encoding for
wave-based acoustic simulation. IEEE Signal Process. Lett., 26(4):617–621, 2019.

[BAH19] S. Bilbao, J. Ahrens, and B. Hamilton. Incorporating source directivity in wave-based virtual acoustics:
Time-domain models and fitting to measured data. J. Acoust. Soc. Am., 146(4):2692–2703, 2019.

[LH20] H. Lai and B. Hamilton. Computer modeling of barrel-vaulted sanctuary exhibiting flutter echo with
comparison to measurements. Acoustics, 2(1):87–109, 2020.

[HPB20] I. Henderson, A. Politis, and S. Bilbao. Filter design for real-time ambisonics encoding during
wave-based acoustic simulations. In Proc. e-Forum Acusticum, Lyon, France, December 2020.

[Ham21a] B. Hamilton. Adding air attenuation to simulated room impulse responses: A modal approach. In Proc.
Int. Conf. Immersive & 3D Audio, Bologna, Italy, September 2021.

[Ham21b] B. Hamilton. Air absorption filtering method based on approximate Green’s function for Stokes’
equation. In Proc. Digital Audio Effects (DAFx), Vienna, Austria, September 2021.

