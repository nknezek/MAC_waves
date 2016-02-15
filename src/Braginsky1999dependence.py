#! /usr/bin/env python
"""
Created on Wed Dec  2 17:05:47 2015

@author: nknezek
"""
import numpy as np
from matplotlib import pyplot as plt

#%%

def omega_beta(beta, k_Phi, k_H, N):
    return beta*k_Phi*(N/2)**2 / k_H**2
def omega_B(k_Tau, Br, N):
    return k_Tau*Br*N/2
def gamma_MR(Tau_eta, omega_B, omega_beta):
    return omega_B**2/(omega_beta**2 * Tau_eta)
def omega_MR(omega_beta, omega_B, gamma_MR):
    return -omega_beta - omega_B**2/omega_beta - 1j*gamma_MR

#%% Buoyancy Variation
H = 100e3 # m
H_var = [x*1e4 for x in range(2,14,2)]
#H_var = [80e3]
plt.figure(figsize=[15,5])
for H in H_var:
    k_H = np.pi/H # 1/m
    k_Phi = 2e-6 # 1/m
    k_Tau = 4.4e-6 # 1/m
    Br = 0.45e-2 # m/s (~0.5 mT)
    N = 4 # Om
    beta = 3e-11 # 1/(m s)
    eta = 2. # m^2/s
    Tau_eta = 1/(eta*k_H**2)

    N = np.linspace(0.5,6.,100)
    f_N = omega_MR(omega_beta(beta,k_Phi,k_H,N),
                   omega_B(k_Tau,Br,N),
                    gamma_MR(Tau_eta,omega_B(k_Tau, Br, N),omega_beta(beta, k_Phi, k_H, N)))
    wBwbeta = omega_B(k_Tau,Br,N)/omega_beta(beta, k_Phi, k_H, N)
    plt.subplot(131)
    plt.plot(N, -2*np.pi/(f_N.real*3600*24*365.25))
    plt.ylim([0,20])
    plt.subplot(132)
    plt.plot(N,wBwbeta)
    plt.ylim([0,4])
    plt.subplot(133)
    plt.plot(N,f_N.real/(2*f_N.imag))
    plt.ylim([0,10])

plt.subplot(131)
plt.legend([x/1e3 for x in H_var], loc='best')
plt.title('Braginsky Buoyancy Variation B=0.5mT')
plt.xlabel('Buoyancy Frequency/Omega')
plt.ylabel('Wave Period (yr)')
plt.grid(axis='both')
plt.subplot(132)
plt.legend([x/1e3 for x in H_var], loc='best')
plt.title('Small Parameter w_B/w_beta B=0.5mT')
plt.xlabel('Buoyancy Frequency/Omega')
plt.ylabel('')
plt.grid(axis='both')
plt.subplot(133)
plt.legend([x/1e3 for x in H_var], loc='best')
plt.title('Quality Factor B=0.5mT')
plt.xlabel('Buoyancy Frequency/Omega')
plt.ylabel('')
plt.grid(axis='both')


#%% Magnetic Field Variation
H = 100e3 # m
plt.figure(figsize=[15,5])

H_var = [x*10e3 for x in range(2,14,2)]
for H in H_var:
    k_H = np.pi/H # 1/m
    k_Phi = 2e-6 # 1/m
    k_Tau = 4.4e-6 # 1/m
    Br = 0.45e-2 # m/s (~0.5 mT)
    N = 2. # Om
    beta = 3e-11 # 1/(m s)
    eta = 2. # m^2/s
    Tau_eta = 1/(eta*k_H**2)

    Br = np.linspace(0.1e-2,1.0e-2,100)
    f_B = omega_MR(omega_beta(beta,k_Phi,k_H,N),
                   omega_B(k_Tau,Br,N),
                    gamma_MR(Tau_eta,omega_B(k_Tau, Br, N),omega_beta(beta, k_Phi, k_H, N)))
    wBwbeta = omega_B(k_Tau,Br,N)/omega_beta(beta, k_Phi, k_H, N)
    plt.subplot(131)
    plt.plot(Br*0.5/0.45e-2, -2*np.pi/(f_B.real*3600*24*365.25))
    plt.subplot(132)
    plt.plot(Br*0.5/0.45e-2,wBwbeta)
    plt.ylim([0,4])
    plt.subplot(133)
    plt.plot(Br*0.5/0.45e-2,f_N.real/(2*f_N.imag))
    plt.ylim([0,100])
plt.subplot(131)
plt.legend([x/1e3 for x in H_var], loc='best')
plt.title('Braginsky Buoyancy Variation N = %0.0f Omega'%N)
plt.xlabel('Br (mT)')
plt.ylabel('Wave Period (yr)')
plt.grid(axis='both')
plt.subplot(132)
plt.legend([x/1e3 for x in H_var], loc='best')
plt.title('Small Parameter w_B/w_beta N=2')
plt.xlabel('Br (mT)')
plt.ylabel('')
plt.grid(axis='both')
plt.subplot(133)
plt.legend([x/1e3 for x in H_var], loc='best')
plt.title('Quality Factor N=2')
plt.xlabel('Br (mT)')
plt.ylabel('')
plt.grid(axis='both')