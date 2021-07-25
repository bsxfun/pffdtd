##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: test_script_MV_fcc_viz.py
#
# Description: this shows a simple setup with FCC scheme, for a larger single-precision GPU run (<12GB VRAM)
#
##############################################################################
from sim_setup import sim_setup

sim_setup(
    model_json_file='../data/models/Musikverein_ConcertHall/model_export.json',
    draw_backend='mayavi',
    mat_folder='../data/materials',
    source_num=3,
    insig_type='impulse',
    diff_source=True,
    mat_files_dict={
                    'Floor': 'mv_floor.h5',
                    'Chairs' : 'mv_chairs.h5',
                    'Plasterboard': 'mv_plasterboard.h5',
                    'Window': 'mv_window.h5',
                    'Wood': 'mv_wood.h5',
                    }, #see build_mats.py to set these material impedances from absorption data
    duration=3.0,
    Tc=20,
    rh=50,
    fcc_flag=True,
    PPW=7.7, #for 1% phase velocity error at fmax
    fmax=2500.0,
    save_folder='../data/sim_data/mv_fcc/gpu',
    save_folder_gpu='../data/sim_data/mv_fcc/gpu',
    compress=3, #apply level-3 GZIP compression to larger h5 files
)

#then from '../data/sim_data/mv_fcc/gpu' folder, run (relative path for default folder structure):
#   ../../../../c_cuda/fdtd_main_gpu_single.x

#then post-process with something like:
#   python -m fdtd.process_outputs --data_dir='../data/sim_data/mv_fcc/gpu/' --fcut_lowpass 2500.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --air_abs_filter='stokes' --save_wav --plot
