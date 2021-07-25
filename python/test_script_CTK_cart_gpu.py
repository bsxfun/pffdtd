##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: test_script_CTK_cart_gpu.py
#
# Description: this shows a simple setup with Cartesian scheme, for a single-precision GPU run (<2GB VRAM)
#
##############################################################################

from sim_setup import sim_setup

sim_setup(
    model_json_file='../data/models/CTK_Church/model_export.json',
    mat_folder='../data/materials',
    source_num=1,
    insig_type='impulse', #for RIR
    diff_source=True, #for single precision
    mat_files_dict={
                    'AcousticPanel': 'ctk_acoustic_panel.h5',
                    'Altar' : 'ctk_altar.h5',
                    'Carpet': 'ctk_carpet.h5',
                    'Ceiling': 'ctk_ceiling.h5',
                    'Glass': 'ctk_window.h5',
                    'PlushChair': 'ctk_chair.h5',
                    'Tile': 'ctk_tile.h5',
                    'Walls': 'ctk_walls.h5',
                    }, #see build_mats.py to set these material impedances from absorption data
    duration=3.0, #duration in seconds
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=10.5, #for 1% phase velocity error at fmax
    fmax=1400.0,
    save_folder='../data/sim_data/ctk_cart/gpu', 
    save_folder_gpu='../data/sim_data/ctk_cart/gpu',
    compress=0,
)
#then from '../data/sim_data/ctk_cart/gpu' folder, run (relative path for default folder structure):
#   ../../../../c_cuda/fdtd_main_gpu_single.x

#then post-process with something like:
# python -m fdtd.process_outputs --data_dir='../data/sim_data/ctk_cart/gpu/' --fcut_lowpass 1400.0 --N_order_lowpass=8 --symmetric --fcut_lowcut 10.0 --N_order_lowcut=4 --save_wav --plot
