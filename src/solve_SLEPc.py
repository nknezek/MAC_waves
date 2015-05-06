#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:44:50 2015

@author: nknezek
"""

from datetime import datetime
import macloglib as mlog
import macplotlib as mplt
import sys
import slepc4py
import cPickle as pkl
from optparse import OptionParser
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
opts = PETSc.Options()

Target = 5.5e-5j
nev = 10
data_dir = '../data/k20_l201_nu1.00e+00_dC=3.80e+04/'
out_dir = '../output/SLEPc_run_{0}_{1}/'.format(datetime.today().date(),
                                                 datetime.today().time())    

# Allow usage of command line arguments 
parser = OptionParser()
parser.add_option('-t','--Target', type='complex', dest='Target',
                  help="set Target in model", default=Target)
parser.add_option('-n','--nev', type='int', dest='nev',
                  help="set nev in model", default=nev)
parser.add_option('--data_dir', type='string', dest='data_dir',
                  help="set data_dir in model", default=data_dir)
parser.add_option('--out_dir', type='string', dest='out_dir',
                  help="set data_dir in model", default=out_dir)
(options, args) = parser.parse_args()
Target = options.Target
nev = options.nev
data_dir = options.data_dir
out_dir = options.out_dir

logger = mlog.setup_custom_logger(dir_name=out_dir, filename='SLEPc.log')

#==============================================================================
#%%  Load Matrices and model from files
#==============================================================================
filemodel = 'model.p'
fileA = data_dir + 'A.dat'
fileB = data_dir + 'M.dat'

viewer = PETSc.Viewer().createBinary(fileA, 'r')
A = PETSc.Mat().load(viewer)
viewer = PETSc.Viewer().createBinary(fileB, 'r')
B = PETSc.Mat().load(viewer)
model = pkl.load(open(data_dir+filemodel,'rb'))
logger.info('matrices and model loaded into memory from ' + data_dir)

#==============================================================================
#%% Set up SLEPc Solver
#==============================================================================


EPS = SLEPc.EPS().create()
EPS.setDimensions(nev, PETSc.DECIDE)
EPS.setOperators(A, B)
EPS.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
EPS.setTarget(Target)
EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE)
EPS.setFromOptions()
ST = EPS.getST()
ST.setType(SLEPc.ST.Type.SINVERT)
logger.info('solver set up, Target = {0:.2e}, nev = {1}'.format(Target, nev))

#==============================================================================
#%% Solve Problem
#==============================================================================
EPS.solve()
logger.info('problem solved')

#==============================================================================
#%% Save Computed Solutions
#==============================================================================
savefile = 'data.p'
conv = EPS.getConverged()
logger.info('{0} eigenvalues converged'.format(conv))

vals = []
vecs = []
for ind in range(conv):
    vs, ws = PETSc.Mat.getVecs(A)
    v = EPS.getEigenpair(ind, ws)
    vals.append(v)
    vecs.append(ws.getArray())
pkl.dump({'vals': vals, 'vecs': vecs},open(out_dir + savefile, 'wb'))
logger.info('vals and vecs saved to ' + out_dir + savefile)

#==============================================================================
#%% Plot Found Vectors
#==============================================================================
for ind in range(conv):
    title = 'SLEPc Run Nk={0}, Nl={1}, ind={2}, val={3:.2e}'.format(model.Nk,
                                                                    model.Nl,
                                                                    ind,
                                                                    vals[ind])
    mplt.plot_pcolor_rth(model, vals[ind], vecs[ind], dir_name=out_dir,
                         title=title)
    logger.info('plotted ind={0}'.format(ind))

logger.info('run complete')

