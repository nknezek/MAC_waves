#! /usr/bin/env python
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

# Period and Decay to search for
T_list = [8.,]
target_Q = 1.
delta_T = 6.

# Solution mechanics
nev = 20
tol = 1e-8
use_initial_guess = False

# Filtering Solutions
D3sq_filter = 10e8
dth_filter = 10e8
eq_split = 0.6
eq_var = 'p'
zeros_wanted = [0,1,2,3,4]
min_Q = 0.3
oscillate = True

# Saving Solutions
savefile = None

# Plotting Solutions
real_var = 'ur'
plot_robinson = False
plot_B_obs = True
plot_vel = True

# Model Information and Data Directory
filemodel = 'model.p'
fileA = 'A'
fileB = 'B'
dCyr_list = [9.0]
data_dir = [
#==============================================================================
###  m = 2
#==============================================================================
## 0.6 mT, 8 Om
           # '../data/k20_l120_m2_nu1e-02_35km_constantG800_constantBrB60/',
           # '../data/k20_l120_m2_nu1e-02_40km_constantG800_constantBrB60/',
## 0.65 mT, 8 Om
           '../data/k20_l120_m2_nu1e-02_35km_constantG800_constantBrB65/',
           '../data/k20_l120_m2_nu1e-02_40km_constantG800_constantBrB65/',
## 0.7 mT, 8 Om
           '../data/k20_l120_m2_nu1e-02_35km_constantG800_constantBrB70/',
           '../data/k20_l120_m2_nu1e-02_40km_constantG800_constantBrB70/',
## 0.8 mT, 8 Om
           # '../data/k20_l120_m2_nu1e-02_35km_constantG800_constantBrB80/',
           # '../data/k20_l120_m2_nu1e-02_40km_constantG800_constantBrB80/',

#==============================================================================
###  m = 3
#==============================================================================

## 0.6 mT, 8 Om
           # '../data/k20_l120_m3_nu1e-02_35km_constantG800_constantBrB60/',
           # '../data/k20_l120_m3_nu1e-02_40km_constantG800_constantBrB60/',

## 0.65 mT, 8 Om
           '../data/k20_l120_m3_nu1e-02_35km_constantG800_constantBrB65/',
           '../data/k20_l120_m3_nu1e-02_40km_constantG800_constantBrB65/',

## 0.7 mT, 8 Om
           '../data/k20_l120_m3_nu1e-02_35km_constantG800_constantBrB70/',
           '../data/k20_l120_m3_nu1e-02_40km_constantG800_constantBrB70/',

## 0.8 mT, 8 Om
           # '../data/k20_l120_m3_nu1e-02_30km_constantG800_constantBrB80/',
           # '../data/k20_l120_m3_nu1e-02_35km_constantG800_constantBrB80/',

#==============================================================================
###  m = 6
#==============================================================================


## B 0.6mT, 6 Om
           # '../data/k20_l120_m6_nu1e-02_40km_constantG600_constantBrB60/',

## 0.65mT, 6 Om
           '../data/k20_l120_m6_nu1e-02_35km_constantG600_constantBrB65/',
           '../data/k20_l120_m6_nu1e-02_40km_constantG600_constantBrB65/',

# B 0.70mT, 6 Om
           '../data/k20_l120_m6_nu1e-02_35km_constantG600_constantBrB70/',
           '../data/k20_l120_m6_nu1e-02_40km_constantG600_constantBrB70/',

# B 0.8 mT, 6 Om
#            '../data/k20_l120_m6_nu1e-02_35km_constantG600_constantBrB80/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG600_constantBrB80/',
]



