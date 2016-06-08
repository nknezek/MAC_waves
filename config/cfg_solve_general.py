#! /usr/bin/env python3
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [80.1]
delta_T = 30.

nev = 10
d2_filter = 10e8
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
    '../data/k20_l200_m0_nu8e-01_139km_constantN100_absDipoleBrB31_bdiv/',
    # '../data/k20_l200_m0_nu8e-01_139km_constantN100_absDipoleBrB31_py3matrix7/',
    # '../data/k20_l200_m0_nu8e-01_139km_constantN100_absDipoleBrB31_py3matrix5/',
    # '../data/k20_l200_m0_nu8e-01_139km_constantN100_absDipoleBrB31_py3matrix4/',
    # '../data/k20_l200_m0_nu8e-01_139km_constantN100_absDipoleBrB31_py3matrix3/',
    # '../data/k20_l200_m0_nu8e-01_139km_constantN100_absDipoleBrB31_py3matrix2/',
    # '../data/k20_l200_m0_nu8e-01_139km_constantN100_absDipoleBrB31_py3matrix/',
    # '../data/k20_l200_m0_nu8e-01_139km_constantG100_absDipoleBrB31_python3/',
# '../data/k20_l200_m0_nu8e-01_139km_constantG100_absDipoleBrB31_8d7f6ab/'
]



