#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:11:50 2019

@author: sergio
"""
import os
path = '/home/sergio/Documents/SpiPGen/'
os.chdir(path)
import sys
sys.path.append(path)
import pandas as pd
import numpy as np
import spipgen_plot 
import matplotlib.pyplot as plt
from swift_io import *
import seagen
import spipgen
from scipy.interpolate import RectBivariateSpline

# Load spining profile
rho = np.load("profile.npy")
times = np.load("exec_times.npy")

data = pd.read_csv("1layer_n100.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]
dr = data.R[0] - data.R[1]
spipgen_plot.plotrho(rho[-1], Rs)

rho_grid = rho[-1,:,:]
#######
#######
# cake idea
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
    u[k] = spipgen.ucold(rho[k], spipgen.granite, 10000) + spipgen.granite[11]*Ts


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

##################
##################
##################
# model densities
data = pd.read_csv("1layer_n100.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]
dr = data.R[0] - data.R[1]

rho = np.load("profile.npy")
rho_grid = rho[-1,:,:]
Re = (np.sum(rho_grid[:, 0] > 0) - 1)*dr

r = np.arange(0, rho_grid.shape[0]*dr, dr)
z = np.arange(0, rho_grid.shape[1]*dr, dr)
rho_model = RectBivariateSpline(r, z, rho_grid, kx = 1, ky = 1)
"""
# plot the model density
rM = np.linspace(0, r[-1], 1000)
zM = np.linspace(0, z[-1], 1000)
X, Y = np.meshgrid(rM, zM)
Z = rho_model.ev(X, Y)

plt.contour(X, Y, Z, levels = np.arange(0, 8000, 500))
plt.xlim(1,1.2)
plt.ylim(0,0.3)
plt.show()
"""
# Get equatorial profile
radii = np.arange(0, Re, dr/100)
densities = rho_model.ev(radii,0)

# Generate seagen sphere
N = 10**5
particles = seagen.GenSphere(N, radii[1:], densities[1:])
"""
def _eq_density(x, y, z, radii, densities):
    rho = np.zeros(x.shape[0])
    r = np.sqrt(x*x + y*y + z*z)
    dr = radii[1] - radii[0]
    for i in range(x.shape[0]):
        k = int(r[i]/dr)
        rho[i] = densities[k] + (densities[k + 1] - densities[k])*(r[i] - k*dr)/dr 
    return rho
"""

def _bisection(f,a,b,N = 10):
    if f(a)*f(b) >= 0:
        # print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            # print("Found exact solution.")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2

#rho_seagen = _eq_density(particles.x, particles.y, particles.z, radii, densities)
rho_seagen = particles.A1_rho

zP = np.zeros(particles.m.shape[0])

for i in range(particles.m.shape[0]):
    rc = np.sqrt(particles.x[i]**2 + particles.y[i]**2)
    rho_spherical = rho_seagen[i]
    z = particles.z[i]
    f = lambda z: rho_model.ev(rc, np.abs(z)) - rho_spherical
    if z < 0:
        zP[i] = _bisection(f, z, 0.)
        #if np.isnan(zP[i]):
        #    zP[i] = _bisection(f, 100*z, z)
    else:
        zP[i] = _bisection(f, 0., z)
        #if np.isnan(zP[i]):
        #    zP[i] = _bisection(f, z, 100*z)
 
"""
plt.scatter(particles.z, zP, s = 1)
plt.plot(particles.z, particles.z, c = 'r', linewidth = 0.1)
plt.xlabel(r"$z$ $[R_{earth}]$")
plt.ylabel(r"$z'$ $[R_{earth}]$")
plt.show()
"""
mask = np.isfinite(zP)
slope = np.linalg.lstsq(particles.z[mask].reshape((-1,1)), zP[mask], rcond=None)[0][0]
zP[np.isnan(zP)] = particles.z[np.isnan(zP)]*slope
mP = particles.m*slope


#mP = particles.m*zP/particles.z

#plt.hist(mP, bins = 100)
#plt.show()

# velocities

Tw = 4 # in hours
# v = w x r (units of R_earth/hour)  
vx = np.zeros(mP.shape[0])
vy = np.zeros(mP.shape[0])
vz = np.zeros(mP.shape[0])

wz = 2*np.pi/Tw 
for i in range(mP.shape[0]):
    vx[i] = -particles.y[i]*wz
    vy[i] = particles.x[i]*wz
    
vx = vx*R_earth/hour_to_s
vy = vy*R_earth/hour_to_s

# model densities and internal energy
rho = np.zeros((mP.shape[0]))
u = np.zeros((mP.shape[0]))

x = particles.x
y = particles.y

for k in range(mP.shape[0]):
    rc = np.sqrt(x[k]*x[k] + y[k]*y[k])
    rho[k] = rho_model.ev(rc, zP[k])
    u[k] = spipgen.ucold(rho[k], spipgen.granite, 10000) + spipgen.granite[11]*Ts
    
## Smoothing lengths, crudely estimated from the densities
num_ngb = 48    # Desired number of neighbours
w_edge  = 2     # r/h at which the kernel goes to zero
A1_h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge

A1_P = np.ones((mP.shape[0],))
A1_id = np.arange(mP.shape[0])
A1_mat_id = np.ones((mP.shape[0],))*Di_mat_id['Til_granite']

swift_to_SI = Conversions(M_earth, R_earth, 1)

# save profile
filename = 'init_test_linear.hdf5'
with h5py.File(filename, 'w') as f:
    save_picle_data(f, np.array([x, y, zP]).T, np.array([vx, vy, vz]).T,
                    mP, A1_h, rho, A1_P, u, A1_id, A1_mat_id,
                    4*Rs, swift_to_SI)  

#######
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 8
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
plt.scatter(x, zP, alpha=0.2, s = 2.5)
plt.scatter(1, 1, s = 0)
plt.scatter(-1, -1, s = 0)
plt.show()

