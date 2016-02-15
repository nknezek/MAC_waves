#! /usr/bin/env python
import numpy as np
import macmodel_braginsky as mac
import macloglib as mlog

# mode to simulate (longitudinal)
m = [0]

# Size of grid
Nk = 30 # Radial cells
Nl = 601 # Latitudinal cells
h = [80e3]  # layer thickness in (m)

# Define Physical Constants
R = 3480e3  # Outer core radius in (m)
Omega = 2*np.pi/(23.9345*3600.0)  # rotation rate in (rad/s)
rho = 1.e4   # density in (kg/m^3)
nu = 1.e-6  # momentum diffusivity in (m^2/s)
eta = 2.  # magnetic diffusivity in (m^2/s)
nu_th = 0.0
eta_th = 0.0
mu_0 = 4.*np.pi*10.**-7  # vacuum permeability in (kg*m/(A^2s^2))
g = 10.  # Gravity in m/s^2
dCyr = 71.

# background magnetic field in (Tesla)
# chocies: dipole, dipole_Br, abs_dipole, abs_dipole_Br, constant_Br, set, Br_sinfunc
B_type = 'dipole_Br'
B_mag = [0.25e-3]
Bd = B_mag
Br = B_mag
Bth = None
const = 0.0
Bmax = 1.e-3
Bmin = 1.e-3
sin_exp = 2.5

# background velocity field in (m/s)
Uphi = np.ones((Nk+2, Nl+2))*0.0

# Buoyancy Frequency
# choices: constant, linear
buoy_type = 'constant'
buoy_ratio =  [2]

# model parameters
model_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p', 'r_disp')
boundary_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p')
dir_suf = '_dipoleBr_n1e-6_ep1e-5_rlorentz'
ep = 1e-5

