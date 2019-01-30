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
import spipgen_func 
import matplotlib.pyplot as plt
from useful import *
import seagen

rho = np.load("profile.npy")
times = np.load("exec_times.npy")

data = pd.read_csv("1layer_n100.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]
dr = data.R[0] - data.R[1]
spipgen_plot.plotrho(rho[-1], Rs)


#####
rho_grid = rho[-1,:,:]
#######
#######
#######
#######
Re = (np.sum(rho_grid[:, 0] > 0) - 1)*dr
Rp = (np.sum(rho_grid[0, :] > 0) - 1)*dr
N_tot = 10**5
rho_ptcle = 3*N_tot/4/np.pi/Re/Re/Rp

x = np.zeros((2*N_tot))
y = np.zeros((2*N_tot))
z = np.zeros((2*N_tot))
m = np.zeros((2*N_tot))
k = 0

for i in range(rho_grid.shape[1]):
    
    densities = rho_grid[(rho_grid[:, i] > 0), i]
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
    
m = m[m > 0]
x = x[0:m.shape[0]]
y = y[0:m.shape[0]]
z = z[0:m.shape[0]]
vx = vx[0:m.shape[0]]*R_earth/hour_to_s
vy = vy[0:m.shape[0]]*R_earth/hour_to_s
vz = vz[0:m.shape[0]]*R_earth/hour_to_s

rho = np.zeros((m.shape[0]))
u = np.zeros((m.shape[0]))
for k in range(m.shape[0]):
    rc = np.sqrt(x[k]*x[k] + y[k]*y[k])
    i = int(rc/dr)
    j = int(np.abs(z[k])/dr)
    rho[k] = rho_grid[i,j]
    #rho[k] = rho[k] + (rc - i*dr)/dr*(rho_grid[i + 1, j] - rho_grid[i, j])
    #rho[k] = rho[k] + (z[k] - j*dr)/dr*(rho_grid[i, j + 1] - rho_grid[i, j])
    u[k] = spipgen_func.ucold(rho[k], spipgen_func.granite, 10000) + spipgen_func.granite[11]*Ts


## Smoothing lengths, crudely estimated from the densities
num_ngb = 48    # Desired number of neighbours
w_edge  = 2     # r/h at which the kernel goes to zero
A1_h    = np.cbrt(num_ngb * m / (4/3*np.pi * rho)) / w_edge

A1_P = np.ones((m.shape[0],))
A1_id = np.arange(m.shape[0])
A1_mat_id = np.ones((m.shape[0],))*Di_mat_id['Til_granite']

swift_to_SI = Conversions(M_earth, R_earth, 1)

filename = 'init_test.hdf5'
with h5py.File(filename, 'w') as f:
    save_picle_data(f, np.array([x, y, z]).T, np.array([vx, vy, vz]).T,
                    m, A1_h, rho, A1_P, u, A1_id, A1_mat_id,
                    4*Rs, swift_to_SI)
    
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
plt.scatter(x, z, alpha=0.2, s = 2.5)
plt.scatter(1, 1, s = 0)
plt.scatter(-1, -1, s = 0)
plt.show()

plt.hist(m, bins = int(m.shape[0]/1000))
plt.show()



