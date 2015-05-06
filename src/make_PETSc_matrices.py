#! /usr/bin/env python

import numpy as np
from numpy import sin
from numpy import cos
from datetime import datetime
import sys
from optparse import OptionParser
import macmodel_condcore as mac
import macloglib as mlog


# mode to simulate
m = 0
m_min = m
m_max = m
m_values = range(m_min, m_max+1)

# Size of grid
Nk = 20
Nl = 201

# Define Physical Constants
R = 3480e3  # Outer core radius in m
h = 80e3  # layer thickness in m
Omega = 2*np.pi/(24.0*3600.0)  # rotation rate in rad/s
rho = 1.e4   # density in kg/m^3
nu = 1.  # momentum diffusivity in m^2/s
eta = 2.  # magnetic diffusivity in m^2/s
mu_0 = 4.*np.pi*10.**-7  # vacuum permeability in (kg*m/(A^2s^2))
g = 10.  # Gravity in m/s^2
T1 = 72.0*365.25*24*3600  # Period of first mode in seconds
l1 = 2*np.pi/(T1)  # non-dimensional frequency of first mode in rad/s
delta_C = np.sqrt(2*eta/(l1))  # Core magnetic skin depth in m

#==============================================================================
# Allow usage of command line arguments 
#==============================================================================
parser = OptionParser()
parser.add_option('-k','--Nk', type='int', dest='Nk', help="set Nk in model",
                    default=Nk)
parser.add_option('-l','--Nl', type='int', dest='Nl', help="set Nl in model",
                    default=Nl)
parser.add_option('-n','--nu', type='float', dest='nu',
                    help="set nu in model", default=nu)
parser.add_option('-d','--delta_C', type='float', dest='delta_C',
                    help="set delta_C in model", default=delta_C)
(options, args) = parser.parse_args()
Nk = options.Nk
Nl = options.Nl
nu = options.nu
delta_C = options.delta_C

#===============================================================================
#%% Set up model
#===============================================================================

model_parameters = {'Nk': Nk, 'Nl': Nl, 'm_values': m_values}
physical_constants = {'R': R, 'Omega': Omega, 'rho': rho, 'h': h, 'nu': nu,
                      'eta': eta, 'mu_0': mu_0, 'delta_C': delta_C, 'g': g,
                      'T1': T1, 'l1': l1}
model_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p', 'r_disp')
boundary_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p')
model = mac.Model(model_variables, boundary_variables, model_parameters,
                  physical_constants)

#===============================================================================
#%% Set up background fields
#===============================================================================

Bd = (0.446e-2)*(4*np.pi*rho)**0.5*(mu_0/(4*np.pi))**0.5  # Dipole Field constant in Teslas (Bd = Br*cos(theta))
model.set_dipole_Br(Bd)

Uphi = np.ones((Nk+2, Nl+2))*1e-16
model.set_Uphi(Uphi)

omega_g0 = 2*Omega  # Buoyancy frequency in rad/s
drho_dr_0 = -omega_g0**2*rho/g  # Density gradient corresponding to omega_g0
drho_dr = np.ones((Nk+2, Nl+2))*drho_dr_0  # density gradient for boussinesq approximation in kg/m^4 (constant gradient in this model)
model.set_buoyancy(drho_dr)
model.omega_g = omega_g0

#==============================================================================
#%% Save Model
#==============================================================================
dir_name = ('../data/k'+str(Nk) + '_l' + str(Nl) +
            '_nu{0:.2e}_dC={1:.2e}/'.format(nu, delta_C))
mlog.ensure_dir(dir_name)
filemodel = 'model.p'
model.save_model(dir_name + filemodel)

#===============================================================================
#%% Create Matrices
#===============================================================================
model.make_A(m)
model.make_M(m)

#===============================================================================
#%% Save PETSc Matrices
#===============================================================================
fileA = 'A.dat'
fileM = 'M.dat'

model.save_mat_PETSc(dir_name+fileA, model.A.toPETSc())
model.save_mat_PETSc(dir_name+fileM, model.M.toPETSc())

