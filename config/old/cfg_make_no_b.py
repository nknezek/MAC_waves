#! /usr/bin/env python
import numpy as np
#import macmodel_condcore2 as mac
import macmodel_no_b as mac
import macloglib as mlog

# mode to simulate (longitudinal)
m = 2

# Size of grid
Nk = 30 # Radial cells
Nl = 90 # Latitudinal cells

# Define Physical Constants
R = 3480e3  # Outer core radius in (m)
h = 140e3  # layer thickness in (m)
Omega = 2*np.pi/(23.9345*3600.0)  # rotation rate in (rad/s)
rho = 1.e4   # density in (kg/m^3)
nu = 0.8  # momentum diffusivity in (m^2/s)
eta = 0.8  # magnetic diffusivity in (m^2/s)
mu_0 = 4.*np.pi*10.**-7  # vacuum permeability in (kg*m/(A^2s^2))
g = 10.  # Gravity in m/s^2
dCyr = 0.5 # Period of wave for magnetic boundary condition (years)
delta_C = np.sqrt(2*eta/(2*np.pi/(dCyr*365.25*24*3600)))  # Core magnetic skin depth in (m)

# background magnetic field in (Tesla)
B_type = 'set'
Br = np.ones((Nk+2, Nl+2))*0.0
Bth = np.ones((Nk+2, Nl+2))*0.0
#B_type = 'dipole'
#Bd = 0.6e-3

# background velocity field in (m/s)
Uphi = np.ones((Nk+2, Nl+2))*0.0

# Buoyancy Frequency
omega_g0 = 1.02**0.5*Omega  # Maximum buoyancy frequency in (rad/s)
drho_dr_0 = -omega_g0**2*rho/g  # Maximum density gradient
# linear distribution from core fluid to mantle
drho_dr = (np.ones((Nk+2, Nl+2)).T*np.linspace(0, drho_dr_0, Nk+2)).T
# constant distribution
#drho_dr = np.ones((Nk+2, Nl+2))*drho_dr_0

# model parameters
model_variables = ('ur', 'uth', 'uph', 'p', 'r_disp')
boundary_variables = ('ur', 'uth', 'uph', 'p')

# Directory name to save model
dir_name = ('../data/k'+str(Nk) + '_l' + str(Nl) +
            '_dC{0:.0e}_m{1:.0f}_nu{2:.0e}_{3:.0f}km_noB_linG/'.format(dCyr, m, nu, h/1e3))
filemodel = 'model.p' # name of model in directory
fileA = 'A.dat' # name of A matrix data
fileM = 'M.dat' # name of M matrix data

