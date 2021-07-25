##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: build_mats.py
#
# Description: Examples of setting/saving material impedances data
#
##############################################################################

import numpy as np
from pathlib import Path
from numpy import array as npa
from materials.adm_funcs import *

plot=False #set to True to see fits, but have to close plots to proceed to next material
write_folder = Path('../data/materials')
#some examples to save admittance/impedance data 
#these are Sabine coefficients, 16Hz to 16kHz centre frequencies
mv_chairs       = npa([0.22  , 0.22  , 0.22 , 0.22 , 0.26 , 0.3  , 0.33 , 0.34 , 0.34 , 0.34 , 0.34])
mv_floor        = npa([0.14  , 0.14  , 0.14 , 0.14 , 0.1  , 0.06 , 0.08 , 0.1  , 0.1  , 0.1  , 0.1])
mv_plasterboard = npa([ 0.15 ,  0.15 , 0.15 , 0.15 , 0.1  , 0.06 , 0.04 , 0.04 , 0.05 , 0.05 , 0.05])
mv_window       = npa([0.35  , 0.35  , 0.35 , 0.35 , 0.25 , 0.18 , 0.12 , 0.07 , 0.04 , 0.04 , 0.04])
mv_wood         = npa([0.25  , 0.25  , 0.25 , 0.25 , 0.15 , 0.1  , 0.09 , 0.08 , 0.07 , 0.07 , 0.07])
fit_to_Sabs_oct_11(mv_chairs       , filename=Path(write_folder / 'mv_chairs.h5')       , plot=plot)
fit_to_Sabs_oct_11(mv_floor        , filename=Path(write_folder / 'mv_floor.h5')        , plot=plot)
fit_to_Sabs_oct_11(mv_plasterboard , filename=Path(write_folder / 'mv_plasterboard.h5') , plot=plot)
fit_to_Sabs_oct_11(mv_window       , filename=Path(write_folder / 'mv_window.h5')       , plot=plot)
fit_to_Sabs_oct_11(mv_wood         , filename=Path(write_folder / 'mv_wood.h5')         , plot=plot)

#these are Sabine coefficients, 16Hz to 16kHz centre frequencies
ctk_acoustic_panel  = npa([0.2  ,  0.2   , 0.42  , 0.89  , 1     , 1     , 1     , 1     , 1     , 1     , 1])
ctk_altar           = npa([0.25 ,  0.25  , 0.25  , 0.25  , 0.15  , 0.1   , 0.09  , 0.08  , 0.07  , 0.07  , 0.07])
ctk_audience        = npa([0.1  ,  0.1   , 0.1   , 0.1   , 0.07  , 0.08  , 0.1   , 0.1   , 0.11  , 0.11  , 0.11])
ctk_carpet          = npa([0.08 ,  0.08  , 0.08  , 0.08  , 0.24  , 0.57  , 0.69  , 0.71  , 0.73  , 0.73  , 0.73])
ctk_ceiling         = npa([0.19 ,  0.19  , 0.19  , 0.19  , 0.06  , 0.05  , 0.08  , 0.07  , 0.05  , 0.05  , 0.05])
ctk_chair           = npa([0.44 ,  0.44  , 0.44  , 0.44  , 0.56  , 0.67  , 0.74  , 0.83  , 0.87  , 0.87  , 0.87])
ctk_tile            = npa([0.015,  0.015 , 0.015 , 0.015 , 0.015 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005 , 0.005])
ctk_walls           = npa([0.19 ,  0.19  , 0.19  , 0.19  , 0.06  , 0.05  , 0.08  , 0.07  , 0.05  , 0.05  , 0.05])
ctk_window          = npa([0.35 ,  0.35  , 0.35  , 0.35  , 0.25  , 0.18  , 0.12  , 0.07  , 0.04  , 0.04  , 0.04])

fit_to_Sabs_oct_11(ctk_acoustic_panel , filename=Path(write_folder / 'ctk_acoustic_panel.h5') , plot=plot)
fit_to_Sabs_oct_11(ctk_altar          , filename=Path(write_folder / 'ctk_altar.h5')          , plot=plot)
fit_to_Sabs_oct_11(ctk_audience       , filename=Path(write_folder / 'ctk_audience.h5')       , plot=plot)
fit_to_Sabs_oct_11(ctk_carpet         , filename=Path(write_folder / 'ctk_carpet.h5')         , plot=plot)
fit_to_Sabs_oct_11(ctk_ceiling        , filename=Path(write_folder / 'ctk_ceiling.h5')        , plot=plot)
fit_to_Sabs_oct_11(ctk_chair          , filename=Path(write_folder / 'ctk_chair.h5')          , plot=plot)
fit_to_Sabs_oct_11(ctk_tile           , filename=Path(write_folder / 'ctk_tile.h5')           , plot=plot)
fit_to_Sabs_oct_11(ctk_walls          , filename=Path(write_folder / 'ctk_walls.h5')          , plot=plot)
fit_to_Sabs_oct_11(ctk_window         , filename=Path(write_folder / 'ctk_window.h5')         , plot=plot)

#freq-independent impedance from reflection coefficients
write_freq_ind_mat_from_Yn(convert_R_to_Yn(0.90),filename=Path(write_folder / 'R90_mat.h5'))
write_freq_ind_mat_from_Yn(convert_R_to_Yn(0.5),filename=Path(write_folder / 'R50.h5'))

#freq-independent impedance from Sabine abs coefficient
write_freq_ind_mat_from_Yn(convert_Sabs_to_Yn(0.5),filename=Path(write_folder / 'a50.h5'))

#input DEF values directly
write_freq_dep_mat(npa([[0,1.0,0],[2,3,4]]),filename=Path(write_folder / 'ex_mat.h5'))

#to read and plot a material file
#plot_DEF_admittance(np.logspace(np.log10(10),np.log10(20e3),4000),read_mat_DEF(write_folder/'chairs.h5'))
