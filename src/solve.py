#! /usr/bin/env python

import macloglib as mlog
import macplotlib as mplt
import mac_analysis as mana

import cPickle as pkl
import numpy as np
import itertools as it
from datetime import datetime
import sys
import importlib

import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
opts = PETSc.Options()

# Import configuration file
sys.path.append('../config')
config_file = sys.argv[1].rstrip('\.py')
cfg = importlib.import_module(config_file)
print "used config file {0}".format(config_file)

# function to find nearest skin-depth value for a wave period
def find_closest_CC(Target, dCyr_list):
    dist = [abs(x-abs(Target)) for x in dCyr_list]
    return dCyr_list[dist.index(min(dist))]


# Import constant parameters from config file
nev = cfg.nev
D3sq_filter = cfg.D3sq_filter
dth_filter = cfg.dth_filter
eq_split = cfg.eq_split
eq_var = cfg.eq_var
real_var = cfg.real_var
filemodel = cfg.filemodel
fileA = cfg.fileA
fileB = cfg.fileB
T_list = cfg.T_list
dCyr_list = cfg.dCyr_list
savefile = cfg.savefile
plot_robinson = cfg.plot_robinson
plot_B_obs = cfg.plot_B_obs
plot_vel = cfg.plot_vel
target_Q = cfg.target_Q
tol = cfg.tol


# Iterate over parameters that can vary
iter_param_names = ['data_dir', 'use_initial_guess', 'oscillate']
iter_params = {}
for name in iter_param_names:
    value = eval('cfg.'+name)
    if type(value) is not list:
        value = [value]
    iter_params[name] = value
varNames = sorted(iter_params)
combinations = [ dict(zip(varNames, prod)) for prod in it.product(*(iter_params[varName] for varName in varNames))]

# Store main output directory
out_dir_base = '../output/{0}_{1}/'.format(datetime.today().strftime("%Y-%m-%d_%H-%M-%S"), config_file[9:])

for cnum, c in enumerate(combinations):
    data_dir = c['data_dir']
    use_initial_guess = c['use_initial_guess']
    oscillate = c['oscillate']

    # Get information on layer thickness, buoyancy frequency, and magnetic field from data directory
    data_dir_info = data_dir.strip('/').split('_')
    H = data_dir_info[4]
    N = data_dir_info[5]
    B = data_dir_info[6]

    # Set up directory to store solution data
    out_dir = out_dir_base + '{0}/{1}/{2}/'.format(N, B, H)
    logger = mlog.setup_custom_logger(dir_name=out_dir, filename='run.log')
    logger.info('used config file {0}'.format(config_file))
    logger.info('Main output directory set to {0}'.format(out_dir))

    for tnum,T in enumerate(T_list):
        logger.info('\n\nRun {0}, Target {1} yrs\n'.format(tnum, T))

        # Convert Time in years to model frequency
        t_star = (23.9345*3600)/(2*np.pi)
        Target_j = 2*np.pi/(T*365.25*24*3600/t_star)*1j
        Target = Target_j + Target_j*1j/(2*target_Q)

        # Find which CC matrix to use
        dCyr_use = find_closest_CC(T, dCyr_list)
        logger.info('{0} dCyr used'.format(dCyr_use))

        # Set out_dir
        out_dir_T = out_dir+'{0:.2f}/'.format(T)
        mlog.ensure_dir(out_dir_T)
        logger.info('out_dir set to {0}'.format(out_dir_T))

        # %% Load Matrices and model from files
        #==============================================================================
        try:
            viewer = PETSc.Viewer().createBinary(data_dir+fileA+str(dCyr_use)+'.dat', 'r')
            A = PETSc.Mat().load(viewer)
            viewer = PETSc.Viewer().createBinary(data_dir+fileB+'.dat', 'r')
            B = PETSc.Mat().load(viewer)
            model = pkl.load(open(data_dir+filemodel,'rb'))
            logger.info('A'+str(dCyr_use)+' matrix used')
            logger.info('matrices and model loaded into memory from ' + data_dir)
        except:
            logger.info( "Could not load matrices from file: ", sys.exc_info()[0])

        # %% Make initial vector guess
        #==============================================================================
        try:
            if use_initial_guess:
                v_eq_a = ((np.ones((model.Nk, model.Nl))*(np.cos(model.th[1:-1])*np.sin(model.th[1:-1])**50)).T*np.sin(np.linspace(0,np.pi,model.Nk)))*1e-5
                v_eq_s = abs(v_eq_a)
                bc0 = np.ones((2,model.Nl))*1e-16
                v0 = np.ones((model.Nk, model.Nl))*1e-16
                variables = [v_eq_s, v_eq_a, v_eq_s*1j, v_eq_a, v_eq_a*1j, v_eq_s, v_eq_s*1j, v_eq_s*1j]
                boundaries = [bc0, bc0, bc0, bc0, bc0, bc0, bc0]
                start_vec = model.create_vector(variables, boundaries)
                mplt.plot_pcolormesh_rth(model, '-1', start_vec, dir_name=out_dir, title='initial guess', physical_units=True)
                logger.info('created initial guess')
        except:
            logger.info('problem creating initial vector guess: ', sys.exc_info()[0])

        # %% Set up SLEPc Solver
        #==============================================================================
        try:
            EPS = SLEPc.EPS().create()
            EPS.setDimensions(nev, PETSc.DECIDE)
            EPS.setOperators(A, B)
            EPS.setProblemType(SLEPc.EPS.ProblemType.PGNHEP)
            EPS.setTarget(Target)
            if use_initial_guess:
                V = PETSc.Vec().createSeq(model.SizeM)
                V.setValues(range(model.SizeM), start_vec)
                V.assemble()
                EPS.setInitialSpace(V)
            EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE)
            EPS.setTolerances(tol)
            EPS.setFromOptions()
            ST = EPS.getST()
            ST.setType(SLEPc.ST.Type.SINVERT)
            logger.info('solver set up, Period = {0:.2e}, nev = {1}'.format(T, nev))
            logger.info('eigenvalue target = {0:.1e}'.format(Target))
        except:
            print "Could not set up SLEPc solver: ", sys.exc_info()[0]

        # %% Solve Problem
        #==============================================================================
        try:
            EPS.solve()
            logger.info('problem solved')
        except:
            logger.info("Could not solve problem: ", sys.exc_info()[0])
        try:
            # Save Computed Solutions
            conv = EPS.getConverged()
            logger.info('{0} eigenvalues converged'.format(conv))
            vals = []
            vecs = []
            for ind in range(conv):
                vs, ws = PETSc.Mat.getVecs(A)
                v = EPS.getEigenpair(ind, ws)
                vals.append(v)
                vecs.append(ws.getArray())
            Period_max = (2*np.pi/min([x.imag for x in vals]))*model.t_star/(24.*3600.*365.25)
            Period_min = (2*np.pi/max([x.imag for x in vals]))*model.t_star/(24.*3600.*365.25)
            logger.info('min Period = {0:.1f}yrs, max Period = {1:.1f}yrs'.format(Period_min, Period_max))
        except:
            logger.info("Could not get converged eigenvalues: ", sys.exc_info()[0])

        #%% Filter Solutions
        #==============================================================================
        try:
            logger.info('Filtering Eigenvalues:')

            fvals = vals
            fvecs = vecs
            # Filter by Power near Equator
            fvals, fvecs = mana.filter_by_equator_power(model, vals, vecs, equator_fraction=eq_split, var=eq_var)
            logger.info('\t{0} filtered eigenvalues to plot with power at equator, split={1:.1f}'.format(len(fvals), eq_split))

            # Filter by dth
            fvals, fvecs = mana.filter_by_D3sq(model, fvals, fvecs, dth_filter)
            logger.info('\t{0} filtered eigenvalues to plot with max dth < {1:.1e}'.format(len(fvals), dth_filter))

            # Filter by Second Derivative
            fvals, fvecs = mana.filter_by_D3sq(model, fvals, fvecs, D3sq_filter)
            logger.info('\t{0} filtered eigenvalues to plot with max D3sq < {1:.1e}'.format(len(fvals), D3sq_filter))

            # %% Filter by number of zeros in theta-direction
            fvals, fvecs = mana.filter_by_theta_zeros(model, fvals, fvecs, cfg.zeros_wanted)
            logger.info('\t{0} eigenvectors found with requested number of zeros'.format(len(fvecs)))

            # %% Filter by Quality Factor
            fvals, fvecs = mana.filter_by_Q(model, fvals, fvecs, cfg.min_Q)
            logger.info('\t{0} eigenvectors found with Q > {1:.2f}'.format(len(fvecs), cfg.min_Q))

            #%% Filter by Period
            fvals, fvecs = mana.filter_by_period(model, fvals, fvecs, T-cfg.delta_T, T+cfg.delta_T)
            logger.info('\t{0} eigenvectors found with {1:.2f} < T < {2:.2f}'.format(len(fvecs), T-cfg.delta_T, T+cfg.delta_T))

            #%% Convert Eigenvectors to ur real
            fvecs_tmp = []
            for vec in fvecs:
                fvecs_tmp.append(mana.shift_vec_real(model, vec, var=real_var))
            fvecs = fvecs_tmp
            logger.info('\t{0} eigenvectors shifted so that {1} is real to plot'.format(len(fvecs), real_var))
        except:
            logger.info("Problem Filtering Eigenvalues: ", sys.exc_info()[0])

        # %% Save Filtered Eigenvectors
        #==============================================================================
        try:
            if savefile:
                pkl.dump({'vals': vals, 'vecs': vecs, 'model':model},open(out_dir_T + savefile, 'wb'))
                logger.info('vals and vecs saved to ' + out_dir_T + savefile)
        except:
            logger.info("Problem Saving Filtered Eigenvalues: ", sys.exc_info()[0])

        # %% Plot Filtered Eigenvectors
        #==============================================================================
        try:
            logger.info('Plotting:')
            for ind, (val, vec) in enumerate(zip(fvals,fvecs)):
                Period = (2*np.pi/val.imag)*model.t_star/(24.*3600.*365.25)
                Decay = (2*np.pi/val.real)*model.t_star/(24.*3600.*365.25)
                Q = abs(val.imag/(2*val.real))
                if abs(Period) < 1.0:
                    title = ('T{1:.2f}_Q{2:.2f}_{0:.0f}'.format(ind, Period*365.25, Q))
                else:
                    title = ('T{1:.2f}yrs_Q{2:.2f}_{0:.0f}'.format(ind, Period, Q))
                if plot_vel:
                    mplt.plot_pcolormesh_rth(model, val, vec, dir_name=out_dir_T, title=title, physical_units=True, oscillate_values=oscillate)
                if plot_robinson:
                    mplt.plot_robinson(model, vec, model.m, oscillate=oscillate, dir_name=out_dir_T, title=str(ind)+'_T{0:.2f}yrs_'.format(Period)+'Divergence')
                if plot_B_obs:
                    mplt.plot_B_obs(model, vec, model.m, oscillate=oscillate, dir_name=out_dir_T, title='T{0:.2f}yrs_'.format(Period)+'B-perturbation'+str(ind))
                logger.info('\t plotted ind={0}, T={1:.2f}yrs'.format(ind, Period))
            logger.info('run complete')
        except:
            logger.info("Problem Plotting Eigenvalues: ", sys.exc_info()[0])
