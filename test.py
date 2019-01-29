#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:11:50 2019

@author: sergio
"""

path = '/home/sergio/Documents/SpiPGen/'
import os
os.chdir(path)

import sys
sys.path.append(path)

import pandas as pd
import numpy as np
import spipgen_plot 

rho = np.load("profile.npy")
times = np.load("exec_times.npy")

data = pd.read_csv("1layer_n100.csv", header=0)
Rs = data.R[0]
dr = data.R[0] - data.R[1]
spipgen_plot.plotrho(rho[-1], Rs)

import seagen

#####
N = 10**5
densities = rho[-1, (rho[-1, :, 0] > 0), 0]
radii = np.arange(0, densities.shape[0]*dr, dr)

particles = seagen.GenSphere(N, radii[1:], densities[1:])

####
zp = np.zeros((particles.m.shape[0]))
radius = np.zeros((particles.m.shape[0]))
count = 0

for i in range(particles.m.shape[0]):

    radius[i] = np.sqrt(particles.x[i]**2 + particles.y[i]**2 + particles.z[i]**2)
    
    if np.abs(particles.z[i]) < 0.5*dr:
        zp[i] = particles.z[i]
        count = count +1
    else:
        
        rc = np.sqrt(particles.x[i]**2 + particles.y[i]**2) 
        k = int(radii[radius[i] > radii][-1]/dr)
        
        density_sph = (densities[k + 1] - densities[k])/dr
        density_sph = density_sph*(radius[i] - radii[k]) + densities[k]
        
        k = int(rc/dr)
        z_p = np.sum(rho[-1, k, :] > density_sph) - 1
        kk = int(z_p)
        z_p = z_p*dr + dr*(rho[-1, k, kk] - density_sph)/(rho[-1, k, kk] - rho[-1, k, kk + 1])
        if particles.z[i] < 0:
            zp[i] = -z_p
        else:
            zp[i] = z_p


i = 60000
print(zp[i], particles.z[i])

import matplotlib.pyplot as plt

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
plt.scatter(particles.x, zp, alpha=0.1, s = 1)
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(particles.x, particles.y, zp, alpha = 0.2, s = 1)
plt.show()
#######
#######
#######
#######
Re = (np.sum(rho[-1, :, 0] > 0) - 1)*dr
Rp = (np.sum(rho[-1, 0, :] > 0) - 1)*dr
N_tot = 10**5
rho_ptcle = 3*N_tot/4/np.pi/Re/Re/Rp

x = np.zeros((2*N_tot))
y = np.zeros((2*N_tot))
z = np.zeros((2*N_tot))
m = np.zeros((2*N_tot))
k = 0

for i in range(rho.shape[2]):
    
    densities = rho[-1, (rho[-1, :, i] > 0), i]
    if densities.shape[0] <= 1:
        break
    radii = np.arange(0, densities.shape[0]*dr, dr)
    N = int(4*np.pi*radii[-1]**3*rho_ptcle/3)
    
    particles = seagen.GenSphere(N, radii[1:], densities[1:])
    
    for j in range(particles.m.shape[0]):
        if np.abs(particles.z[j]) < 0.5*dr:
            if i == 0:
                x[k] = particles.x[j]
                y[k] = particles.y[j]
                z[k] = particles.z[j]
                m[k] = particles.m[j]
                k += 1
            else:
                x[k] = particles.x[j]
                y[k] = particles.y[j]
                z[k] = particles.z[j] + i*dr
                m[k] = particles.m[j]
                k += 1
                x[k] = particles.x[j]
                y[k] = particles.y[j]
                z[k] = particles.z[j] - i*dr
                m[k] = particles.m[j]
                k += 1
    
# v = w x r (units of R_earth/hour)    
vx = np.zeros((2*N_tot))
vy = np.zeros((2*N_tot))
vz = np.zeros((2*N_tot))

Tw = 4 # in hours
# v = w x r (units of R_earth/hour)  
wz = 2*np.pi/Tw 
for i in range(x.shape[0]):
    vx[i] = -y[i]*wz
    vy[i] = x[i]*wz
    


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
plt.scatter(x, z, alpha=0.2, s = 1.5)
plt.scatter(1, 1, s = 0)
plt.scatter(-1, -1, s = 0)
plt.show()
