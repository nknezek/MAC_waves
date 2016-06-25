#! /usr/bin/env python3
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [120.]
delta_T = 120.

nev = 3
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
zeros_wanted = range(100)
min_Q = 0.
target_Q = 1.
tol = 1e-3


dCyr_list = [120.]
data_dir = [
    '../data/k20_l200_m6_nu1e+02_135km_constantN400_constantBrB62_EMRT_rlor_noCC/',
    # '../data/k20_l200_m6_nu1e+02_135km_constantN400_constantBrB62_EMRTrlor/',
    # '../data/k20_l200_m6_nu1e+02_135km_constantN400_constantBrB62_EMRthick/',
    # '../data/k20_l200_m6_nu1e-02_135km_constantN400_constantBrB62_EMRthick/',
    # '../data/k20_l200_m6_nu1e-02_135km_constantN200_constantBrB62_EMRthick/',
    # '../data/k20_l200_m6_nu1e-02_135km_constantN80_constantBrB62_EMRthick/',
]



