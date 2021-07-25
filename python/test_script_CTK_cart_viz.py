##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: test_script_CTK_cart_viz.py
#
# Description: this shows a simple setup with Cartesian scheme, for visualization purposes
#
##############################################################################
from sim_setup import sim_setup

sim_setup(
    model_json_file='../data/models/CTK_Church/model_export.json',
    mat_folder='../data/materials',
    source_num=1,
    insig_type='dhann30', #for viz
    diff_source=False,
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
    duration=0.1, #duration in seconds
    Tc=20,
    rh=50,
    fcc_flag=False,
    PPW=7.5, #for 2% phase velocity error at fmax
    fmax=500.0,
    save_folder='../data/sim_data/ctk_cart/viz', #can run python from here
    compress=0,
    draw_vox=True,
    draw_backend='mayavi', #will draw 'voxelization' (spheres are active boundary nodes, cubes rigid boundary nodes)
)

#then run with python and 3D visualization:
#   python3 -m fdtd.sim_fdtd --data_dir='../data/sim_data/ctk_cart/viz' --plot --draw_backend='mayavi' --json_model='../data/models/CTK_Church/model_export.json'
