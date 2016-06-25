#! /usr/bin/env python
import numpy as np
import macmodel_rossbyNoR as mac
import macloglib as mlog

# mode to simulate (longitudinal)
m = 2

# Size of grid
Nk = 2 # Radial cells
Nl = 200 # Latitudinal cells

# Define Physical Constants
R = 3480e3  # Outer core radius in (m)
h = 100e3  # layer thickness in (m)
Omega = 2*np.pi/(23.9345*3600.0)  # rotation rate in (rad/s)
rho = 1.e4   # density in (kg/m^3)
nu = 0.8  # momentum diffusivity in (m^2/s)
eta = 0.8  # magnetic diffusivity in (m^2/s)
mu_0 = 4.*np.pi*10.**-7  # vacuum permeability in (kg*m/(A^2s^2))
g = 10.  # Gravity in m/s^2
dCyr = 1.0 # Period of wave for magnetic boundary condition (years)

# background magnetic field in (Tesla)
# chocies: dipole, dipoleBr, absDipole, absDipoleBr, constantBr, set, sinfuncBr
B_type = 'constantBr'
B_mag = 0.0
Bd = 0.0
Br = B_mag
Bth = 0.0
Bph = 0.0
const = 0.0
Bmax = 0.0
Bmin = 0.0
sin_exp = 0.0
Bnoise = 0.0

# background velocity field in (m/s)
Uphi = 0.0

# Buoyancy Frequency
# choices: constant, linear
buoy_type = 'constant'
buoy_ratio = 0.0

# model parameters
model_variables = ('uth', 'uph', 'p')

# Directory suffix name to save model
dir_suf = '_rossbyNoR'
ep = 1e-3

