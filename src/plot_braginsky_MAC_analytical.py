#! /usr/bin/env python
"""
Created on Wed Feb 10 11:25:21 2016

@author: nknezek
"""
import numpy as np
from numpy import cos, sin, pi
from matplotlib import pyplot as plt
from scipy.special import lpmv
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
#%%
H = 80e3 # Layer thickness [m]
Tyrs = 65 # Wave period [s]
BdG = 5 # Magnetic Dipole constant [Gauss]
eta = 2.  # magnetic diffusivity in (m^2/s)
n = 2 # wave radial structure

# Define Physical Constants
R = 3480e3  # Outer core radius in (m)
Omega = 2*np.pi/(23.9345*3600.0)  # rotation rate in (rad/s)
rho = 1.e4   # density in (kg/m^3)
nu = 1e-2  # momentum diffusivity in (m^2/s)
nu_th = 0.0
eta_th = 0.0
mu_0 = 4.*np.pi*10.**-7  # vacuum permeability in (kg*m/(A^2s^2))
g = 10.  # Gravity in m/s^2

Bd = BdG* 10**-4  # magnetic dipole field [Tesla]
T = Tyrs*365.25*24*3600 # wave periods [s]
omega =  2*np.pi/T # Wave frequency [rad/s]

#%%

kH = pi/H
ep_thph = kH**2*Bd**2 / (2*Omega*omega)

v_ph0 = 1
v_th0 = v_ph0*ep_thph
v_r0 = v_th0*n*(n+1)/(kH*R)
b_ph0 = v_ph0*kH*Bd/omega
b_th0 = b_ph0*ep_thph
b_r0 = b_th0/(kH*R)

def Pmn(m,n,z):
    return np.array([lpmv(m,n,x) for x in z])
#%%
th = np.linspace(0,pi, 180)
thpl = 180.*th/pi
r = np.linspace(0,1,100)
rpl = r*H/1e3
m = 1

#%%
v_ph = v_ph0*(1+cos(pi*r))*np.ones((180,100))*((Pmn(m,n,cos(th))/cos(th))*np.ones((100,180))).T
v_th = -1j*v_th0*((cos(pi*r))*np.ones((180,100)))*(Pmn(m,n,cos(th))*np.ones((100,180))).T*1e3
v_r = -1j*v_r0*(sin(pi*r)*np.ones((180,100)))*(Pmn(m,n,cos(th))*np.ones((100,180))).T*1e6

#%%
## Plot Figures
from mpl_toolkits.axes_grid1 import AxesGrid
cmaps = [plt.get_cmap('bwr'), plt.get_cmap('PuOr')]
fig = plt.figure(1, (6.,6.))
grid1 = AxesGrid(fig, 131,
                    nrows_ncols=(1, 2),
                    axes_pad=0.10,
                    share_all=True,
                    label_mode="L",
                    cbar_location="bottom",
                    cbar_mode="single",
                    cbar_pad=0.4,
                    cbar_size="1.5%",
                    direction="column",
                    aspect = False
                    )
pc1 = grid1[0].pcolor(rpl,thpl,v_ph[:,::-1].real, cmap='bwr', vmin=-np.abs(v_ph).max(), vmax=np.abs(v_ph).max())
grid1.cbar_axes[0].colorbar(pc1)
grid1[0].set_xticks([10, 40, 70])
grid1[0].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12)
grid1[0].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=12)
grid1[0].set_xlabel('depth (km)', fontsize=14)
grid1[0].set_ylabel('colatitude (degrees)', fontsize=14)
grid1[0].xaxis.set_label_coords(1.1, -0.042)
grid1.cbar_axes[0].set_xlabel('m/s', fontsize=14)
grid1.cbar_axes[0].get_xaxis().labelpad=-2
grid1.cbar_axes[0].tick_params(labelsize=12)

grid2 = AxesGrid(fig, 132,
                    nrows_ncols=(1, 2),
                    axes_pad=0.10,
                    share_all=True,
                    label_mode="L",
                    cbar_location="bottom",
                    cbar_mode="single",
                    cbar_pad=0.4,
                    cbar_size="1.5%",
                    direction="column",
                    aspect = False
                    )
pc2 = grid2[0].pcolor(rpl,thpl,v_th[:,::-1].imag, cmap='bwr', vmin=-np.abs(v_th).max(), vmax=np.abs(v_th).max())
grid2.cbar_axes[0].colorbar(pc2)
grid2[0].set_xticks([10, 40, 70])
grid2[0].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12 )
grid2[0].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=12)
grid2[0].set_xlabel('depth (km)', fontsize=14)
grid2[0].xaxis.set_label_coords(1.1, -0.042)
grid2.cbar_axes[0].set_xlabel('mm/s', fontsize=14)
grid2.cbar_axes[0].get_xaxis().labelpad=-2
grid2.cbar_axes[0].tick_params(labelsize=12)

grid3 = AxesGrid(fig, 133,  # similar to subplot(132)
                    nrows_ncols=(1, 2),
                    axes_pad=0.10,
                    share_all=True,
                    label_mode="L",
                    cbar_location="bottom",
                    cbar_mode="single",
                    cbar_pad=0.4,
                    cbar_size="1.5%",
                    direction="column",
                    aspect = False
                    )
pc3 = grid3[0].pcolor(rpl,thpl,v_r[:,::-1].imag, cmap='bwr', vmin=-np.abs(v_r).max(), vmax=np.abs(v_r).max())
grid3.cbar_axes[0].colorbar(pc3)
grid3[0].set_xticks([10, 40, 70])
grid3[0].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12)
grid3[0].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=12)
grid3[0].set_xlabel('depth (km)', fontsize=14)
grid3[0].xaxis.set_label_coords(1.1, -0.042)
grid3.cbar_axes[0].set_xlabel('$\mu$m/s', fontsize=14)
grid3.cbar_axes[0].tick_params(labelsize=12)
grid3.cbar_axes[0].get_xaxis().labelpad=-2

plt.tight_layout()

#plt.savefig('FVFBraginskyMACcompare.pdf')


