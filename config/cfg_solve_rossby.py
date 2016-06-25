#! /usr/bin/env python3
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [1.5/365.25]
delta_T = 10.

nev = 20
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
zeros_wanted = list(range(100))
min_Q = 0.0
target_Q = 300.
tol = 1e-8


dCyr_list = [1.0]
data_dir = [
    '../data/k20_l200_m2_nu8e-01_100km_constantN100_constantBrB0_rossby/',
]



