#! /usr/bin/env python
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [8.,]
dCyr_list = [9.0]

nev = 40
D3sq_filter = 10e8
dth_filter = 10e8
eq_split = 0.6
eq_var = 'p'
real_var = 'ur'
filemodel = 'model.p'
fileA = 'A'
fileB = 'B'
savefile = None
use_initial_guess = False
oscillate = True
plot_robinson = False
plot_B_obs = True
plot_vel = True
zeros_wanted = [0,1,2,3,4]
min_Q = 0.3

data_dir = [

#==============================================================================
###  m = 2
#==============================================================================
## 0.15 mT, 1.5 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG150_constant_BrB15_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.15 mT, 2 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG200_constant_BrB15_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.15 mT, 2.5 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG250_constant_BrB15_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.2 mT, 1.5 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG150_constant_BrB20_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.2 mT, 2 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG200_constant_BrB20_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.2 mT, 2.5 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG250_constant_BrB20_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 0.6 mT, 1 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG100_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG100_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG100_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG100_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG100_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 2 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 3 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 4 Om
#            '../data/k20_l120_m2_nu1e-02_35km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 6 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 8 Om
#            '../data/k20_l120_m2_nu1e-02_30km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_35km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',


### 0.65 mT, 4 Om
#            '../data/k20_l120_m2_nu1e-02_30km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_35km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65 mT, 8 Om
#            '../data/k20_l120_m2_nu1e-02_30km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_35km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',


### 0.7 mT, 1 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG100_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG100_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG100_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG100_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG100_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.7 mT, 2 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.7 mT, 3 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.7 mT, 4 Om
#            '../data/k20_l120_m2_nu1e-02_30km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_35km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.7 mT, 6 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.7 mT, 8 Om
#            '../data/k20_l120_m2_nu1e-02_30km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_35km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 0.8 mT, 1 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.8 mT, 2 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.8 mT, 3 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.8 mT, 4 Om
#            '../data/k20_l120_m2_nu1e-02_35km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.8 mT, 6 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.8 mT, 8 Om
#            '../data/k20_l120_m2_nu1e-02_30km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_35km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',


### 1.0 mT, 1 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 1.0 mT, 2 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 1.0 mT, 3 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 1.0 mT, 4 Om
#            '../data/k20_l120_m2_nu1e-02_35km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_100km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_120km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_140km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 1.0 mT, 6 Om
#            '../data/k20_l120_m2_nu1e-02_40km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

## 1.0 mT, 70km
#            '../data/k20_l120_m2_nu1e-02_70km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG800_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',


### 1.2 mT
#            '../data/k20_l120_m2_nu1e-02_35km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_40km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_50km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_60km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_70km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_80km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_100km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_120km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m2_nu1e-02_140km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',


#==============================================================================
###  m = 3
#==============================================================================

### 0.6 mT, 2 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##            '../data/k20_l120_m3_nu1e-02_100km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##            '../data/k20_l120_m3_nu1e-02_120km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##            '../data/k20_l120_m3_nu1e-02_140km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 3 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

## 0.6 mT, 4 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 6 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.6 mT, 8 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 0.65 mT, 2 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65 mT, 3 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65 mT, 4 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65 mT, 6 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65 mT, 8 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG800_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#

### 0.70 mT, 2 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.70 mT, 3 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

## 0.7 mT, 4 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.70 mT, 6 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.7 mT, 8 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG800_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#
## 0.7 mT, 10 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG1000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG1000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG1000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##            '../data/k20_l120_m3_nu1e-02_45km_constantG1000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
### 0.7 mT, 20 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG2000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG2000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG2000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##            '../data/k20_l120_m3_nu1e-02_45km_constantG2000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
### 0.7 mT, 40 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG4000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG4000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG4000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##            '../data/k20_l120_m3_nu1e-02_45km_constantG4000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
### 0.7 mT, 80 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG8000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG8000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG8000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG8000_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 0.80 mT, 2 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.80 mT, 3 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.8 mT, 4 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_70km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_80km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.80 mT, 6 Om
#            '../data/k20_l120_m3_nu1e-02_35km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.8 mT, 8 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG800_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

###  1.0 mT, 2 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##  1.0 mT, 3 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG300_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##  1.0 mT, 4 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_70km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_80km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
##  1.0 mT, 6 Om
#            '../data/k20_l120_m3_nu1e-02_30km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_35km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_45km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG600_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

# H-variation 1.2 mT
#            '../data/k20_l120_m3_nu1e-02_35km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_40km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_50km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_60km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_70km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m3_nu1e-02_80km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

#==============================================================================
###  m = 6
#==============================================================================
### 0.15 mT, 1.5 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG150_constant_BrB15_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.15 mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB15_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.15 mT, 2.5 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG250_constant_BrB15_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
### 0.2 mT, 1.5 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG150_constant_BrB20_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.2 mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB20_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.2 mT, 2.5 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG250_constant_BrB20_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',


### 0.6 mT, High-resolution
#            '../data/k40_l240_m6_nu1e-02_30km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',


## B 0.6mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_100km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_120km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_140km_constantG200_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## B 0.6mT, 2.5 Om
#            '../data/k20_l120_m6_nu1e-02_35km_constantG250_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG250_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG250_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## B 0.6mT, 3 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG300_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

## B 0.6 mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG400_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

## B 0.6mT, 6 Om
#            '../data/k20_l120_m6_nu1e-02_25km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_30km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG600_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

## B 0.6mT, 8 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_25km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG800_constant_BrB60_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 0.65mT, 2 Om HD
#            '../data/k40_l200_m6_nu1e-02_35km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG200_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65mT, 2.5 Om
#            '../data/k20_l120_m6_nu1e-02_35km_constantG250_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG250_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG250_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65mT, 3 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG300_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_25km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_30km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_45km_constantG400_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 0.65mT, 6 Om
#            '../data/k20_l120_m6_nu1e-02_35km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG600_constant_BrB65_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 0.7mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG200_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## B 0.7mT, 3 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG300_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# B 0.7mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_25km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_30km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_45km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG400_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# B 0.70mT, 6 Om
#            '../data/k20_l120_m6_nu1e-02_35km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG600_constant_BrB70_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 0.8 mT, 1 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_47km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG100_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# B 0.8mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_47km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG200_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## B 0.8mT, 3 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG300_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

# 0.8 mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_100km_constantG400_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# B 0.8 mT, 6 Om
#            '../data/k20_l120_m6_nu1e-02_35km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG600_constant_BrB80_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 1.0 mT buoy 1G
#            '../data/k20_l120_m6_nu1e-02_30km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_35km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_47km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG100_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

## 1.0mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_30km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_47km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG200_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

# 1.0 mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_160km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_140km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_120km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_100km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_45km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_40km_constantG400_constant_BrB100_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

### 1.2 mT, 1 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG100_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_47km_constantG100_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG100_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG100_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 1.2 mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_40km_constantG200_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_47km_constantG200_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_50km_constantG200_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG200_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# 1.2 mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_50km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_60km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG400_constant_BrB120_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

# 1.3 mT, 1 Om
#            '../data/k20_l120_m6_nu1e-02_80km_constantG100_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# 1.3 mT, 1.5 Om
#            '../data/k20_l120_m6_nu1e-02_80km_constantG150_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# 1.3 mT, 2 Om
#            '../data/k20_l120_m6_nu1e-02_80km_constantG200_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# 1.3 mT, 3 Om
#            '../data/k20_l120_m6_nu1e-02_80km_constantG300_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
## 1.3mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_60km_constantG400_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_70km_constantG400_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_80km_constantG400_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_100km_constantG400_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_120km_constantG400_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
#            '../data/k20_l120_m6_nu1e-02_140km_constantG400_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',
# 1.3 mT, 5 Om
#            '../data/k20_l120_m6_nu1e-02_80km_constantG500_constant_BrB130_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

# 1.4mT, 4 Om
#            '../data/k20_l120_m6_nu1e-02_80km_constantG400_constant_BrB140_noCC_asymp_nuth0e+00_etath0e+00_Bobs/',

]



