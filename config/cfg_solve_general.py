#! /usr/bin/env python
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [69.,]

nev = 6
D3sq_filter = 10e8
dth_filter = 10e8
eq_split = 0.7
eq_var = 'p'
real_var = 'ur'
filemodel = 'model.p'
fileA = 'A'
fileB = 'B'
savefile = None
use_initial_guess = False
oscillate = False
plot_robinson = False
plot_B_obs = False
plot_vel = True
zeros_wanted = [0,1,2,3,4]
min_Q = 0.3
target_Q = 1.
tol = 1e-8

dCyr_list = [71.0]
data_dir = [
'../data/k40_l200_m0_nu1e-05_140km_linearG200_constantBrB60/'
#'../data/k20_l200_m0_nu1e-05_140km_linearG200_constantBrB60/'
]



