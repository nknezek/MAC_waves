#! /usr/bin/env python
import numpy as np
#import macmodel_condcore2 as mac
import macmodel_noCC as mac
import macloglib as mlog

# mode to simulate (longitudinal)
m = 6

# Size of grid
Nk = 20 # Radial cells
Nl = 180 # Latitudinal cells

# Define Physical Constants
R = 3480e3  # Outer core radius in (m)
h = 80e3  # layer thickness in (m)
Omega = 2*np.pi/(23.9345*3600.0)  # rotation rate in (rad/s)
rho = 1.e4   # density in (kg/m^3)
nu = 1e2  # momentum diffusivity in (m^2/s)
eta = 0.8  # magnetic diffusivity in (m^2/s)
mu_0 = 4.*np.pi*10.**-7  # vacuum permeability in (kg*m/(A^2s^2))
g = 10.  # Gravity in m/s^2
#dCyr = [0.12, 0.25, 0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.] # Period of wave for magnetic boundary condition (years)
dCyr = [0.1]
# background magnetic field in (Tesla)
# chocies: dipole, dipole_Br, abs_dipole, abs_dipole_Br, constant_Br, set, Br_sinfunc
B_type = 'constant_Br'
B_mag = 0.6e-3
Bd = None
Br = B_mag
Bth = None
const = None
Bmax = 1.1e-3
Bmin = 0.4e-3
sin_exp = 2.5

# background velocity field in (m/s)
Uphi = np.ones((Nk+2, Nl+2))*0.0

# Buoyancy Frequency
# choices: constant, linear
buoy_type = 'constant'
buoy_ratio = 1.0
#omega_g0 = buoy_ratio*Omega  # Maximum buoyancy frequency in (rad/s)
#drho_dr_0 = -omega_g0**2*rho/g  # Maximum density gradient
# linear distribution from core fluid to mantle
#drho_dr = (np.ones((Nk+2, Nl+2)).T*np.linspace(0, drho_dr_0, Nk+2)).T
# constant distribution
#drho_dr = np.ones((Nk+2, Nl+2))*drho_dr_0

# model parameters
model_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p', 'r_disp')
boundary_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p')

# Directory name to save model
dir_name = ('../data/k'+str(Nk) + '_l' + str(Nl) +
            '_m{1:.0f}_nu{2:.0e}_{3:.0f}km_{7}G{4:.0f}_{6}B{5:.0f}_noCC/'.format(dCyr, m, nu, h/1e3, buoy_ratio*100., B_mag*1e5, B_type, buoy_type))
filemodel = 'model.p' # name of model in directory
fileA = 'A' # name of A matrix data
fileM = 'M' # name of M matrix data

