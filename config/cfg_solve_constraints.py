#! /usr/bin/env python
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """
Nyr = 10
yr_min = 11.
yr_max = 25.
absyrs = 10.**np.linspace(np.log10(yr_min), np.log10(yr_max), num=Nyr)
# Period of waves to targets
#T_list = [1.5, 2., 2.5, 3., 4., 4.5, 5., 6., 7.5, 10., 15., 20., 30.]
#T_list = [4.1, 5., 6., 7., 8., 9., 10.]
#T_list = [12., 15., 20., 30., 40.]
T_list = [10.4]
#T_list = [12.5, 22.4, 82.9]
#T_list = [1.61, 2.11, 2.61, 3.11, 4.011, 5.011, 6.011, 7.511, 9.51]
#T_list = [90., 60., 30.]

nev = 2
D3sq_filter = 10e8
dth_filter = 10e8
eq_split = 1.0
eq_var = 'p'
real_var = 'ur'

data_dir = ['../data/k20_l180_m6_nu1e+02_60km_constantG200_constant_BrB140_noCC_asymp_allasym_sm4C/']


filemodel = 'model.p'
fileA = 'Ac'
fileB = 'Bc'
savefile = 'data'
use_initial_guess = False
oscillate = True
plot_robinson = False