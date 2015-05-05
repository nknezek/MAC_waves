import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
import matplotlib as mpl

import MAC_functions_conducting_core as mf
import MAC_plotting as mplt
import MAC_logging as mlog

from numpy import sin
from numpy import cos
import sys
from datetime import datetime
import cPickle as pkl
import os


######################################################################
##########	Define Parameters	######################################
######################################################################

######################################################################
#### Edit These: ####

# mode to simulate
m = 0

# Size of grid
Nk = 20
Nl = 300
Nm = 1

# Set up variables to use
model_variables = ('ur','uth','uph','br','bth','bph','p','r_disp')
boundary_variables = ('ur','uth','uph','br','bth','bph','p')

# Physical Constants
R = 3480e3 # Outer core radius in m
Omega = 2*np.pi/(24.0*3600.0) # rotation rate in rad/s
rho = 1.e4	# density in kg/m^3
h = 80e3 # layer thickness in m
nu = 1. # momentum diffusivity in m^2/s
eta = 2. # magnetic diffusivity in m^2/s
mu_0 = 4.*np.pi*10.**-7 # vacuum permeability in (kg*m/(A^2s^2))
g = 10. # Gravity in m/s^2
drho_dr = -1e-1 # density gradient for boussinesq approximation in kg/m^4 (constant gradient in this model)
omega_g = 2*Omega # Buoyancy frequency in rad/s
T1 = 65.0*365.25*24*3600 # Period of first mode in seconds
Bd = (0.446e-2)*(4*np.pi*rho)**0.5*(mu_0/(4*np.pi))**0.5  # Dipole Field constant in Teslas (Bd = Br*cos(theta))
delta_C = np.sqrt(2*nu/(2*np.pi/T1))  # Core magnetic skin depth for first mode

### Parameters to set for automated run####
# sigmas = 10.**np.linspace(-4,-7,1)*1j
num_eigs =20
max_iter = 2000
which = 'LI'
dir_name = './output/automated_run_{0}_{1}/'.format(datetime.today().date(),datetime.today().time())
tol = 1e-5
####
#### End Editable parameters ####
######################################################################

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
		os.makedirs(d)
ensure_dir(dir_name)


######## Create log file and direct all output to log file ####
logger = mlog.setup_custom_logger('autorun',dir_name+'autorun.log')


# Calculate Non-Dimensionalized Parameters
t_star = 1/Omega  # seconds
r_star = R	# meters
P_star = rho* r_star**2/t_star**2
B_star = (eta*mu_0*rho/t_star)**0.5
u_star = r_star/t_star
E = nu*t_star/r_star**2
Prm = nu/eta
G = (omega_g*t_star)**2
l1 = 2*np.pi/(T1/t_star)

######################## Important: LOOK HERE! #################
sigmas = [l1*1j*0.1, l1*1j*0.5, l1*1.0j]
################################################################

physical_constants = {'R':R,'Omega':Omega,'rho':rho,'h':h,'nu':nu,'eta':eta,'mu_0':mu_0,'omega_g':omega_g,'Bd':Bd,
					  't_star':t_star,'r_star':r_star,'u_star':u_star,'P_star':P_star,'B_star':B_star,
					  'E':E,'Prm':Prm,'G':G,'l1':l1,'delta_C':delta_C}

# Create model parameter vectors
SizeM = len(model_variables)*Nk*Nl+2*len(boundary_variables)*Nl
rmin = (R-h)/r_star
rmax = R/r_star
dr = (rmax-rmin)/(Nk)
r = np.linspace(rmin-dr/2.,rmax+dr/2.,num=Nk+2.) # r value at center of each cell
rm = np.linspace(rmin-dr,rmax,num=Nk+2.) # r value at plus border (top) of cell
rp = np.linspace(rmin,rmax+dr,num=Nk+2.) # r value at minus border (bottom) of cell
dth = np.pi/(Nl)
th = np.linspace(-dth/2.,np.pi+dth/2.,num=Nl+2.) # theta value at center of cell
thm = np.linspace(-dth,np.pi,num=Nl+2.) # theta value at plus border (top) of cell
thp = np.linspace(0,np.pi+dth,num=Nl+2.) # theta value at minus border (bottom) of cell
m_min = m
m_max = m
m_values = range(m_min,m_max+1)

### Set Background Fields
B0 = np.ones((Nk+2,Nl+2))*cos(th)*Bd/B_star
U0 = np.ones((Nk+2,Nl+2))*1e-16

model_parameters = {'Nk':Nk,'Nl':Nl,'Nm':Nm,'SizeM':SizeM,
					'dr':dr,'r':r,'rm':rm,'rp':rp,
					'dth':dth,'th':th,'thm':thm,'thp':thp,
					'm_values':m_values,
					'B0':B0, 'U0':U0
				   }

######## Start Log File ############
sigmaList = ''
for sigma in sigmas:
	sigmaList = sigmaList+ '\n'+str(sigma)

logger.info('\nAutomated Run begin' +
'\nwhich eigenvalues to search for = {0}'.format(which) +
'\nNumber of eigenvalues to return per sigma = {0}'.format(num_eigs) +
'\nsigmas requested:'+sigmaList)

######################################################################
##########	Create Model   ###########################################
######################################################################
logger.info('creating model')
model = mf.ModelBraginskyConductingCore(model_variables,boundary_variables,physical_constants,model_parameters)

A_matrices = {}
M_matrices = {}
model.make_A(m)
logger.info('created A, now making M')
model.make_M(m)

mlog.log_model(logger,model)

######################################################################
# Braginsky Solution for starting vector
######################################################################

from scipy.special import clpmn, lpmv
mu = np.linspace(-1,1,Nl)
x = np.linspace(1,0,Nk)
omega = 2*np.pi/(65*365.25*24.*3600.) # Oscillation period for T=1
k_H = np.pi/h

n=2 # n=2 for T1, n=4 for T2

# Constants
upha = 1.
utha = upha*k_H**2*Bd**2/(2*Omega*omega)
ura = utha*n*(n+1)/(k_H*R)
bpha = upha*k_H*Bd/omega
btha = bpha*k_H**2*Bd**2/(2*Omega*omega)
bra = btha/(k_H*R)

# Solution Vectors
uph = (upha*(np.ones((len(x),len(mu))).T*(1+cos(np.pi*x))).T*(lpmv(1,n,mu)/mu)).T
uph[np.isnan(uph)] = 0
uth = (utha*(np.ones((len(x),len(mu))).T*(-1j*cos(np.pi*x))).T*lpmv(1,n,mu)).T
ur =  (ura*(np.ones((len(x),len(mu))).T*(-1j*sin(np.pi*x))).T*lpmv(0,n,mu)).T
bph = (bpha*(np.ones((len(x),len(mu))).T*(1j*sin(np.pi*x))).T*lpmv(1,n,mu)).T
bth = (btha*(np.ones((len(x),len(mu))).T*(sin(np.pi*x))).T*lpmv(1,n,mu)*mu).T
br = (bra*(np.ones((len(x),len(mu))).T*(-cos(np.pi*x))).T*np.append(np.diff(lpmv(1,n,mu)*mu),0)).T
p = (np.zeros_like(br)).T
r_disp = (np.zeros_like(br)).T
bound = np.zeros((2,Nl))

# Create starting vector
variables = [ur,uth,uph,br,bth,bph,p,r_disp]
boundaries = [bound,bound,bound,bound,bound,bound,bound]
v0 = model.create_vector(variables,boundaries)

mplt.plot_pcolor_rth(model,'v0',v0,dir_name,'Braginsky T1 Mode Starting Vector')
logger.info('calculated braginsky solution for starting vector')

######################################################################
### Search Parameter Space of Eigenvalues, save values and plot
######################################################################


found_vecs = {}
found_vals = {}
for snum,sigma in enumerate(sigmas):
	try:
		found_vals[sigma],eigenvectors_tmp = LA.eigs(model.A, k=num_eigs, M=model.M, v0=v0,
											 sigma=sigma, return_eigenvectors=True, which=which, maxiter=max_iter, tol=tol)
	except:
		logger.info('simga {2} of {3}: no convergence for sigma={0}, {1} iterations\n'.format(sigma,max_iter,snum+1,len(sigmas)))
	else:
		eigenvectors_list_temp = []
		for ind in range(num_eigs):
			eigenvectors_list_temp.append(eigenvectors_tmp[:,ind])
		found_vecs[sigma] = np.array(eigenvectors_list_temp)
		logger.info('\nsigma {2} of {3}: {0} eigenvalues found for sigma = {1}'.format(num_eigs,sigma,snum+1,len(sigmas)))
		for ind in range(num_eigs):
			logger.info('sigma {0}: {1:.4e}'.format(ind,found_vals[sigma][ind]))
			if not np.isclose(found_vals[sigma][ind].imag, 0.0, rtol=1e-10, atol=1e-10):
				mplt.plot_pcolor_rth(model,found_vals[sigma][ind],found_vecs[sigma][ind],dir_name,'wanted_'+str(sigma)+'val_'+str(found_vals[sigma][ind]))
				logger.info('plotted {0:.4e}'.format(found_vals[sigma][ind]))

pkl.dump({'found_vals':found_vals,'found_vecs':found_vecs},open(dir_name+'data.p', 'wb'))
logger.info('run complete')
