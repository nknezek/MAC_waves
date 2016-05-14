#! /usr/bin/env python
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [65.5]
delta_T = 15.

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
zeros_wanted = [0,1,2,3,4,5,6,7,8,9,10]
min_Q = 0.3
target_Q = 5.
tol = 1e-8


dCyr_list = [65.]
data_dir = [
    # '../data/k20_l100_m6_nu8e-01_50km_constantN400_constantBrB70_noghost/',
    # '../data/k20_l100_m0_nu8e-01_150km_constantN100_constantBrB60_noghost/',
    # '../data/k20_l100_m0_nu8e-01_140km_constantN100_constantBrB60_noghost/',
    # '../data/k20_l100_m0_nu8e-01_140km_constantN200_constantBrB60_noghost/',
    '../data/k20_l100_m0_nu1e-02_80km_constantN200_constantBrB60_noghost/'
]




