#! /usr/bin/env python
from datetime import datetime
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """
Nyr = 10
yr_min = 11.
yr_max = 25.
absyrs = 10.**np.linspace(np.log10(yr_min), np.log10(yr_max), num=Nyr)
# Period of waves to targets
#T = np.dstack((absyrs,-absyrs)).flatten() # code to target both propogation directions
#T = [0.17048168448]
#T = [1., 3., 5., 7., 10., 15., 20.,30.,40.,60.]
#T = [2.1, 5.1, 7.1]
#T = [10.1, 12.01]
#T = [15., 20., 30.1]
T = [3.5, 7.9, 10.4]
#dCyr_list = [0.12, 0.25, 0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.] # Period of wave for magnetic boundary condition (years)
dCyr_list = [0.1]
#dCyr_list = [2., 4., 6., 8., 10.]

nev = 10
D3sq_filter = 10e6
dth_filter = 10e6
eq_split = 1.0
eq_var = 'p'
real_var = 'ur'
data_dir = '../data/k20_l180_m5_nu1e-02_80km_constantG200_constant_BrB50_noCC_nuth0e+00_etath0e+00_3/'
filemodel = 'model.p'
fileA = data_dir + 'A'
fileB = data_dir + 'M'

out_dir = '../output/SLEPc_run_{0}_{1}/'.format(datetime.today().date(),
                                                 datetime.today().time())
savefile = 'data.p'

use_initial_guess = True
oscillate = False