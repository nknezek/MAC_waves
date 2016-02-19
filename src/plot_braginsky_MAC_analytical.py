#! /usr/bin/env python
"""
Created on Wed Feb 10 11:25:21 2016

@author: nknezek
"""
import numpy as np
from numpy import cos, sin, pi
from matplotlib import pyplot as plt
from scipy.special import lpmv, lpn
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#%% Create Analytical Solution

def Pmn(m,n,z):
    return np.array([lpmv(m,n,x) for x in z])
def Pn(n,z):
    return np.array([lpn(n,x)[0][-1] for x in z])
H = 80e3 # Layer thickness [m]
Tyrs = 71 # Wave period [s]
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

Bd = BdG*1e-4*np.sqrt(4*np.pi/mu_0)/np.sqrt(4*np.pi*rho)  # magnetic dipole field [m/s]
T = Tyrs*365.25*24*3600 # wave periods [s]
omega =  2*np.pi/T # Wave frequency [rad/s]

kH = pi/H
ep_thph = kH**2*Bd**2 / (2*Omega*omega)

v_ph0 = 1.
v_th0 = v_ph0*ep_thph
v_r0 = v_th0*n*(n+1)/(kH*R)
b_ph0 = v_ph0*kH*Bd/omega
b_th0 = b_ph0*ep_thph
b_r0 = b_th0/(kH*R)



th = np.linspace(0,pi, 180)
thpl = 180.*th/pi
r = np.linspace(0,1,100)
rpl = r*H/1e3
m = 1

v_ph = v_ph0*(1+cos(pi*r))*np.ones((180,100))*((Pmn(m,n,cos(th))/cos(th))*np.ones((100,180))).T
v_th = -1j*v_th0*((cos(pi*r))*np.ones((180,100)))*(Pmn(m,n,cos(th))*np.ones((100,180))).T
v_r = -1j*v_r0*(sin(pi*r)*np.ones((180,100)))*(Pn(n,cos(th))*np.ones((100,180))).T

#%% Import Numerical Solution
import cPickle as pkl
filename = '../output/braginsky/dip_nu1e-5_data.p'
data = pkl.load(open(filename,'rb'))
#%%
import mac_analysis as mana
model = data['model']
vals = data['vals']
vecs = data['vecs']

ind = 0
val = vals[ind]
vec = vecs[ind]
vec = mana.shift_vec_real(model,vec)

Nk = model.Nk
Nl = model.Nl
thpln = model.th[1:-1]*180./np.pi
rpln = (model.r[1:-1]*model.r_star - model.r_star)/1e3 +80.
ur = model.get_variable(vec,'ur', returnBC=False)
uth = model.get_variable(vec,'uth', returnBC=False)
uph = model.get_variable(vec,'uph', returnBC=False)

Cph = np.max(np.abs(v_ph))/np.max(np.abs(uph))
Cth = np.max(np.abs(v_th))/np.max(np.abs(uth))
Cr = np.max(np.abs(v_r))/np.max(np.abs(ur))
C = np.mean([Cr, Cth, Cph])
ur = ur*C
uth = uth*C
uph = uph*C
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
grid1[0].text(85, 180,'a) $\phi$ velocity', horizontalalignment='center', 
              verticalalignment='bottom', fontsize=14)
pc1 = grid1[0].pcolor(rpl,thpl,-v_ph[:,::-1].real, cmap='bwr', vmin=-np.abs(v_ph).max(), vmax=np.abs(v_ph).max())
grid1.cbar_axes[0].colorbar(pc1)
grid1[0].set_xticks([10, 40, 70])
grid1[0].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12)
grid1[0].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=12)
grid1[0].set_xlabel('depth (km)', fontsize=14)
grid1[0].set_ylabel('colatitude (degrees)', fontsize=14)
grid1[0].xaxis.set_label_coords(1.1, -0.042)
grid1.cbar_axes[0].set_xlabel('km/yr', fontsize=14)
grid1.cbar_axes[0].get_xaxis().labelpad=-2
grid1.cbar_axes[0].tick_params(labelsize=12)

grid1[1].pcolor(rpln,thpln,uph.T.imag, cmap='bwr', vmin=-np.abs(v_ph).max(), vmax=np.abs(v_ph).max())
grid1[1].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12)

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
grid2[0].text(85, 180,r'b) $\theta$ velocity', horizontalalignment='center', 
              verticalalignment='bottom', fontsize=14)
Cthpl = 1e3
v_thpl = v_th*Cthpl
uthpl=uth*Cthpl
thbd = np.abs(v_thpl).max()             
pc2 = grid2[0].pcolor(rpl,thpl,-v_thpl[:,::-1].imag, cmap='bwr', vmin=-thbd, vmax=thbd)
grid2.cbar_axes[0].colorbar(pc2)
grid2[0].set_xticks([10, 40, 70])
grid2[0].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12 )
grid2[0].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=12)
grid2[0].set_xlabel('depth (km)', fontsize=14)
grid2[0].xaxis.set_label_coords(1.1, -0.042)
grid2.cbar_axes[0].set_xlabel('m/yr', fontsize=14)
grid2.cbar_axes[0].get_xaxis().labelpad=-2
grid2.cbar_axes[0].tick_params(labelsize=12)

grid2[1].pcolor(rpln,thpln,uthpl.T.real, cmap='bwr', vmin=-thbd, vmax=thbd)
grid2[1].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12)

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
grid3[0].text(85, 180,'c) $r$ velocity', horizontalalignment='center', 
              verticalalignment='bottom', fontsize=14)
Crpl = 1e3
v_rpl = v_r*Crpl
urpl=ur*Crpl
rbd = np.abs(v_rpl).max()                       
pc3 = grid3[0].pcolor(rpl,thpl,-v_rpl[:,::-1].imag, cmap='bwr', vmin=-rbd, vmax=rbd)
grid3.cbar_axes[0].colorbar(pc3)
grid3[0].set_xticks([10, 40, 70])
grid3[0].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12)
grid3[0].set_yticklabels([180, 160, 140, 120, 100, 80, 60, 40, 20, 0], fontsize=12)
grid3[0].set_xlabel('depth (km)', fontsize=14)
grid3[0].xaxis.set_label_coords(1.1, -0.042)
grid3.cbar_axes[0].set_xlabel('m/yr', fontsize=14)
grid3.cbar_axes[0].tick_params(labelsize=12)
grid3.cbar_axes[0].get_xaxis().labelpad=-2

grid3[1].pcolor(rpln,thpln,-urpl.T.real, cmap='bwr', vmin=-rbd, vmax=rbd)
grid3[1].set_xticklabels([str(x) for x in [70, 40, 10]], fontsize=12)

plt.tight_layout()
#plt.gcf().subplots_adjust(bottom=0.15)

plt.savefig('FVFBraginskyMACcompare.pdf', bbox_inches='tight', pad_inches=0.13)


