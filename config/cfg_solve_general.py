#! /usr/bin/env python
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [80.1]
#T_list = [53., 99.]
#T_list = [41.,77.]

delta_T = 30.
nev = 10
D3sq_filter = 10e8
dth_filter = 10e8
eq_split = 1.0
eq_var = 'p'
real_var = 'ur'
filemodel = 'model.p'
fileA = 'A'
fileB = 'B'
savefile = 'data.p'
use_initial_guess = False
oscillate = False
plot_robinson = False
plot_B_obs = False
plot_vel = True
zeros_wanted = [0,1,2,3,4]
min_Q = 0.3
target_Q = 5.
tol = 1e-8


dCyr_list = [68.44627]
data_dir = [
'../data/k20_l200_m0_nu8e-01_139km_constantG100_absDipoleBrB31_Bn0.3e-3/'
#'../data/k20_l200_m0_nu8e-01_139km_constantG100_absDipoleBrB50_Bn0.5e-3/'
#'../data/k20_l200_m0_nu8e-01_139km_constantG100_absDipoleBrB50_Bn0.1e-3/'
#'../data/k40_l200_m0_nu8e-01_139km_linearG200_constantBrB62_2/'
#'../data/k20_l200_m0_nu8e-01_139km_constantG70_constantBrB62_2/'
#'../data/k20_l200_m0_nu8e-01_139km_constantG90_constantBrB62_2/'

]



