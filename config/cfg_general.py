#! /usr/bin/env python
import numpy as np
#import macmodel_condcore2 as mac
#import macmodel_noCC_sympBC as mac
import macmodel_rlorentz as mac
import macloglib as mlog

# mode to simulate (longitudinal)
m = [3]

# Size of grid
Nk = 20 # Radial cells
Nl = 120 # Latitudinal cells

# Define Physical Constants
R = 3480e3  # Outer core radius in (m)
h = [30e3, 35e3, 40e3, 45e3]  # layer thickness in (m)
Omega = 2*np.pi/(23.9345*3600.0)  # rotation rate in (rad/s)
rho = 1.e4   # density in (kg/m^3)
nu = 1e-2  # momentum diffusivity in (m^2/s)
nu_th = 0.0
eta = 0.8  # magnetic diffusivity in (m^2/s)
eta_th = 0.0
mu_0 = 4.*np.pi*10.**-7  # vacuum permeability in (kg*m/(A^2s^2))
g = 10.  # Gravity in m/s^2
dCyr = 9.

# background magnetic field in (Tesla)
# chocies: dipole, dipole_Br, abs_dipole, abs_dipole_Br, constant_Br, set, Br_sinfunc
B_type = 'constant_Br'
B_mag = [0.7e-3]
Bd = None
Br = B_mag
Bth = None
const = None
Bmax = 1.e-3
Bmin = 1.e-3
sin_exp = 2.5

# background velocity field in (m/s)
Uphi = np.ones((Nk+2, Nl+2))*0.0

# Buoyancy Frequency
# choices: constant, linear
buoy_type = 'constant'
buoy_ratio =  [80]
#omega_g0 = buoy_ratio*Omega  # Maximum buoyancy frequency in (rad/s)
#drho_dr_0 = -omega_g0**2*rho/g  # Maximum density gradient
# linear distribution from core fluid to mantle
#drho_dr = (np.ones((Nk+2, Nl+2)).T*np.linspace(0, drho_dr_0, Nk+2)).T
# constant distribution
#drho_dr = np.ones((Nk+2, Nl+2))*drho_dr_0

# model parameters
model_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p', 'r_disp')
boundary_variables = ('ur', 'uth', 'uph', 'br', 'bth', 'bph', 'p')
dir_suf = '_Bobs'


