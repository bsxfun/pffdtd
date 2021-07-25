##############################################################################
# This file is a part of PFFDTD.
#
# PFFTD is released under the MIT License.
# For details see the LICENSE file.
#
# Copyright 2021 Brian Hamilton.
#
# File name: timer.py
#
# Description: This is a timer.  Was set up specifically for FDTD simulations,
# but mostly superseded by timerdict.py
#
##############################################################################

import time

class Timer:
    def __init__(self,max_tic:int=2):
        assert isinstance(max_tic,int)
        tic = time.time()
        #generic tic/toc (e.g., one for updates, one for drawing frames)
        self.t_tic = [0.0]*max_tic
        self.t_toc = [0.0]*max_tic
        for i in range(0,max_tic):
            self.t_tic[i] = tic
            self.t_toc[i] = tic

        #calc/draw
        self.t_tic_draw = tic
        self.t_toc_draw = tic
        self.t_tic_calc = tic
        self.t_toc_calc = tic

        self.t_slf = tic
        self.t_elapsed = 0.
        self.t_elapsed_calc = 0.
        self.t_elapsed_draw = 0.

    def tic(self,i:int=0):
        assert isinstance(i,int)
        self.t_tic[i] = time.time()

    def toc(self,i:int=0,print_elapsed:bool=True,prestr:str='')->float:
        assert isinstance(i,int)
        assert isinstance(print_elapsed,bool)
        assert isinstance(prestr,str)
        self.t_toc[i] = time.time()
        delta =  self.t_toc[i] - self.t_tic[i]
        if print_elapsed:
            print('%selapsed = %.4f s' % (prestr,delta))
        return delta


    def tic_draw(self):
        self.t_tic_draw = time.time()

    def toc_draw(self):
        self.t_toc_draw = time.time()
        delta = self.t_toc_draw - self.t_tic_draw
        self.t_elapsed_draw += delta
        self.t_elapsed += delta

    def tic_calc(self):
        self.t_tic_calc = time.time()

    def toc_calc(self):
        self.t_toc_calc = time.time()
        delta = self.t_toc_calc - self.t_tic_calc
        self.t_elapsed_calc += delta
        self.t_elapsed += delta
