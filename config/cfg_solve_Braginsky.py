#! /usr/bin/env python
import numpy as np
""" Configuration File for SLEPc Run solving MAC model """

T_list = [65.]
dCyr_list = [71.0]

nev = 3
D3sq_filter = 10e8
dth_filter = 10e8
eq_split = 0.8
eq_var = 'p'
real_var = 'ur'
filemodel = 'model.p'
fileA = 'A'
fileB = 'B'
savefile = None
use_initial_guess = False
oscillate = False
plot_robinson = False
plot_B_obs = False
plot_vel = True
zeros_wanted = [0,1,2,3,4]
min_Q = 0.3
target_Q = 0.9

data_dir = [
'../data/k30_l601_braginksy_MAC_dipoleBr_n1e-6_ep1e-5_rlorentz/'
#'../data/k20_l401_braginksy_MAC_dipoleBr_n1e-6_ep1e-5_rlorentz/'
#'../data/k20_l400_braginksy_MAC_dipoleBr_n1e-6_ep1e-5_rlorentz/'
#'../data/k20_l301_braginksy_MAC_dipoleBr_n1e-6_ep1e-5_rlorentz/'
#'../data/k20_l201_braginksy_MAC_dipoleBr_n1e-6_ep1e-5_rlorentz/'
#'../data/k20_l201_braginksy_MAC_dipoleBr_n4e2_ep1e-5_rlorentz/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n4e2_ep1e-5_rlorentz/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e2_ep1e-5_rlorentz/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n2e1_ep1e-5_rlorentz/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-4_ep1e-5_rlorentz_testall/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4_ep1e-5_rlorentz_fixed/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4_ep1e-3_rlorentz_fixed/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4_ep1e-12_rlorentz_fixed/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4_ep1e-8_rlorentz_fixed/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-4_ep1e-8_rlorentz_fixed/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-4_ep1d-8_rlorentz/'
#'../data/k20_l80_braginksy_MAC_dipoleB_n1e-6_ep1e-8_bdiv/'
#'../data/k20_l80_braginksy_MAC_dipoleB_n1e-1_ep1e-13_bdiv/'
#'../data/k20_l80_braginksy_MAC_dipoleBr_n1e-1_ep1e-13_bdiv/'
#'../data/k20_l80_braginksy_MAC_dipoleBr_n1e-4_ep1e-13_bdiv/'
#'../data/k20_l80_braginksy_MAC_dipoleBr_n1e-4_ep1e-12_bdiv/'
#'../data/k20_l80_braginksy_MAC_dipoleBr_n1e-4_ep1e-10_bdiv/'
#'../data/k20_l80_braginksy_MAC_dipoleBr_n1e-4_ep1e-14_bdiv/'
#'../data/k20_l120_braginksy_MAC_dipoleBr_n1e-4_ep1e-18_reduced/'
#'../data/k20_l120_braginksy_MAC_dipoleBr_n1e-4_ep1e-4_reduced/'
#'../data/k20_l300_braginksy_MAC_dipoleBr_n1e-4_ep1e-12_reduced/'
#'../data/k20_l300_braginksy_MAC_dipoleBr_n1e-4_ep1e-8_reduced/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-4_ep1e-8_reduced/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-5_ep1e-8_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-5_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-10_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleB_n1e-10_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-10_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e2_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n2e0_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-8_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-6_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4_ep1e-12_bdiv/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4_ep1e-12/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4_ep1e-14/'
#'../data/k20_l200_braginksy_MAC_dipoleBr_n1e-4/'
#'../data/k20_l200_braginksy_MAC_absdipoleBr_n1e-4/'
#'../data/k20_l180_braginksy_MAC_dipoleBr_n1e-6/'
#'../data/k20_l100_braginksy_MAC_dipoleBr_n1e-6/'
#'../data/k20_l100_braginksy_MAC_constantBr_n1e-4/'
#'../data/k20_l100_braginksy_MAC_absdipoleBr/'
#'../data/k20_l100_braginksy_MAC_test/'
#'../data/k40_l200_braginksy_MAC_Bdipole/'
#'../data/k20_l100_braginksy_MAC_dipoleBr/'
#'../data/k20_l100_braginksy_MAC/'
#'../data/k20_l100_braginksy_MAC_dipoleBr_n1e-4/'
#'../data/k20_l100_braginksy_MAC_dipoleBr_n1e2/'
]



