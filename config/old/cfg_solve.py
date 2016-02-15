#! /usr/bin/env python
from datetime import datetime
""" Configuration File for SLEPc Run solving MAC model """

T = -80. # Period of waves to target in (yrs)
nev = 10
D3sq_filter = 6e5
dth_filter = 6e5
eq_split = 0.7
eq_var = 'p'
real_var = 'ur'
data_dir = '../data/k20_l180_m5_nu1e-02_80km_constantG200_constant_BrB50_noCC_nuth0e+00_etath0e+00_3/'
filemodel = 'model.p'
fileA = data_dir + 'A.dat'
fileB = data_dir + 'M.dat'

out_dir = '../output/SLEPc_run_{0}_{1}/'.format(datetime.today().date(),
                                                 datetime.today().time())
savefile = 'data.p'
