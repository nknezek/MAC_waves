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
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
opts = PETSc.Options()


dir_name = '../output/SLEPc_run_{0}_{1}/'.format(datetime.today().date(),
                                                 datetime.today().time())

logger = mlog.setup_custom_logger(dir_name=dir_name, filename='SLEPc.log')

#==============================================================================
#%%  Load Matrices from files
#==============================================================================
fileA = 'A_k20_l201.dat'
fileB = 'M_k20_l201.dat'
viewer = PETSc.Viewer().createBinary(fileA, 'r')
A = PETSc.Mat().load(viewer)
viewer = PETSc.Viewer().createBinary(fileB, 'r')
B = PETSc.Mat().load(viewer)
logger.info('matrices loaded into memory from ' + fileA + ' and ' + fileB)

#==============================================================================
#%% Set up SLEPc Solver
#==============================================================================
Target = 4.4e-5j
nev = 10
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
vs, ws = PETSc.Mat.getVecs(A)
vals = []
vecs = []
for ind in range(conv):
    vals.append(EPS.getEigenpair(ind, ws))
    vecs.append(ws.getArray())
pkl.dump({'vals': vals, 'vecs': vecs},open(dir_name + savefile, 'wb'))
logger.info('vals and vecs saved to '+dir_name + savefile)
logger.info('run complete')
