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

R_earth = 6371000;
##################
##################
def _eq_density(x, y, z, radii, densities):
    rho = np.zeros(x.shape[0])
    r = np.sqrt(x*x + y*y + z*z)
    dr = radii[1] - radii[0]
    for i in range(x.shape[0]):
        k = int(r[i]/dr)
        rho[i] = densities[k] + (densities[k + 1] - densities[k])*(r[i] - k*dr)/dr 
    return rho


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
################## particle by particle
# model densities
data = pd.read_csv("1layer_n10k.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]

rho = np.load("profile_parallel.npy")
r_array = np.load('r_array.npy')
z_array = np.load('z_array.npy')
rho_grid = rho[-1,:,:]
Re = np.max(r_array[rho_grid[:,0] > 0])

rho_model = RectBivariateSpline(r_array, z_array, rho_grid, kx = 1, ky = 1)

spipgen_plot.plotrho(rho[-1], r_array, z_array)
"""
# plot the model density
rM = np.linspace(0, r_array[-1], 1000)
zM = np.linspace(0, z_array[-1], 1000)
X, Y = np.meshgrid(rM, zM)
Z = rho_model.ev(X, Y)

plt.contour(X, Y, Z, levels = np.arange(0, 8000, 500))
plt.show()
"""
# Get equatorial profile
radii = np.arange(0, Re, Re/100000)
densities = rho_model.ev(radii,0)

# Generate seagen sphere
N = 10**5
particles = seagen.GenSphere(N, radii[1:], densities[1:])

rho_seagen = _eq_density(particles.x, particles.y, particles.z, radii, densities)
#rho_seagen = particles.A1_rho

zP = np.zeros(particles.m.shape[0])

for i in range(particles.m.shape[0]):
    rc = np.sqrt(particles.x[i]**2 + particles.y[i]**2)
    rho_spherical = rho_seagen[i]
    z = particles.z[i]
    f = lambda z: rho_model.ev(rc, np.abs(z)) - rho_spherical
    if z < 0:
        zP[i] = _bisection(f, z, 0.)
        
    else:
        zP[i] = _bisection(f, 0., z)

z_fine_r = np.load('z_fine_r.npy')
z_P_fine_r = np.load('zP_fine_r.npy')
z_fine = np.load('z_array_fine.npy')
z_P_fine = np.load('z_array_p_fine.npy')
"""
np.isnan(zP).sum()

slope = np.linalg.lstsq(particles.z.reshape((-1,1)), zP, rcond=None)[0][0]
plt.scatter(particles.z/R_earth, zP/R_earth, s = 5, alpha = 0.15)
plt.plot(particles.z/R_earth, particles.z/R_earth, c = 'r', linewidth = 0.1)
plt.plot(particles.z/R_earth, slope*particles.z/R_earth, c = 'g', linewidth = 0.3)
plt.axes().axhline(color = 'black')
plt.xlabel(r"$z$ $[R_{earth}]$")
plt.ylabel(r"$z'$ $[R_{earth}]$")
plt.show()


plt.scatter(z_fine_r/R_earth, z_P_fine_r/z_fine_r, s = 0.1, label = 'fine-r-grid and normal', c = 'green')
plt.scatter(particles.z/R_earth, zP/particles.z, s = 0.1, label = 'fine-z-grid', c = 'red')
plt.xlabel(r"$z$ $[R_{earth}]$")
plt.ylabel(r"$z'/z$")
plt.legend()
plt.show()

np.sum(zP/particles.z < 0.8)/N*100

plt.scatter(particles.z/R_earth, particles.z/zP, s =1)
plt.show()
"""
mP = particles.m*zP/particles.z

#plt.hist(mP, bins = 100)
#plt.show()

# velocities

Tw = 4 # in hours
# v = w x r (units of R_earth/hour)  
vx = np.zeros(mP.shape[0])
vy = np.zeros(mP.shape[0])
vz = np.zeros(mP.shape[0])

wz = 2*np.pi/Tw/hour_to_s 
for i in range(mP.shape[0]):
    vx[i] = -particles.y[i]*wz
    vy[i] = particles.x[i]*wz
    
# model densities and internal energy
rho = np.zeros((mP.shape[0]))
u = np.zeros((mP.shape[0]))

x = particles.x
y = particles.y

for k in range(mP.shape[0]):
    rc = np.sqrt(x[k]*x[k] + y[k]*y[k])
    rho[k] = rho_model.ev(rc, np.abs(zP[k]))
    u[k] = spipgen.ucold(rho[k], spipgen.granite, 10000) + spipgen.granite[11]*Ts
    
## Smoothing lengths, crudely estimated from the densities
num_ngb = 48    # Desired number of neighbours
w_edge  = 2     # r/h at which the kernel goes to zero
A1_h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge

A1_P = np.ones((mP.shape[0],))
A1_id = np.arange(mP.shape[0])
A1_mat_id = np.ones((mP.shape[0],))*Di_mat_id['Til_granite']

swift_to_SI = Conversions(1, 1, 1)

# save profile
filename = 'init_test_linear.hdf5'
with h5py.File(filename, 'w') as f:
    save_picle_data(f, np.array([x, y, zP]).T, np.array([vx, vy, vz]).T,
                    mP, A1_h, rho, A1_P, u, A1_id, A1_mat_id,
                    4*Rs*R_earth, swift_to_SI)  

#######
####### layer by layer
    
data = pd.read_csv("1layer_n200.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]
dr = data.R[0] - data.R[1]

rho = np.load("profile_n200.npy")
rho_grid = rho[-1,:,:]
Re = (np.sum(rho_grid[:, 0] > 0) - 1)*dr

r = np.arange(0, rho_grid.shape[0]*dr, dr)
z = np.arange(0, rho_grid.shape[1]*dr, dr)
rho_model = RectBivariateSpline(r, z, rho_grid, kx = 1, ky = 1)

# Get equatorial profile
radii = np.arange(0, Re, dr/100000)
densities = rho_model.ev(radii,0)

# Generate seagen sphere
N = 10**5
particles = seagen.GenSphere(N, radii[1:], densities[1:])

rho_seagen = _eq_density(particles.x, particles.y, particles.z, radii, densities)
#rho_seagen = particles.A1_rho

zP = np.zeros(particles.m.shape[0])
zPP = np.zeros(particles.m.shape[0])

for i in range(particles.m.shape[0]):
    rc = np.sqrt(particles.x[i]**2 + particles.y[i]**2)
    rho_spherical = rho_seagen[i]
    z = particles.z[i]
    f = lambda z: rho_model.ev(rc, np.abs(z)) - rho_spherical
    if z < 0:
        zP[i] = _bisection(f, z, 0.)
        
    else:
        zP[i] = _bisection(f, 0., z)

for r in np.unique(particles.A1_r):      
    mask = np.logical_and(particles.A1_r == r, np.abs(particles.z) >= 0.9*np.abs(particles.z[0])) 
    slope = np.linalg.lstsq(particles.z[mask].reshape((-1,1)), zP[mask], rcond=None)[0][0]
    mask = particles.A1_r == r
    zPP[mask] = particles.z[mask]*slope
    if r == np.unique(particles.A1_r)[1]:
        mask = particles.A1_r == particles.A1_r[0]
        zPP[mask] = particles.z[mask]*slope

"""
mask = np.isfinite(zP)
slope = np.linalg.lstsq(particles.z[mask].reshape((-1,1)), zP[mask], rcond=None)[0][0]
#plt.scatter(particles.z, zP, s = 1, alpha = 0.15)
plt.scatter(particles.z, zPP, s = 1, alpha = 1, color = 'green')
plt.plot(particles.z, particles.z, c = 'r', linewidth = 0.1)
plt.plot(particles.z, slope*particles.z, c = 'black', linewidth = 0.3)
plt.axes().axhline(color = 'black')
plt.xlabel(r"$z$ $[R_{earth}]$")
plt.ylabel(r"$z'$ $[R_{earth}]$")
plt.show()
"""
zPPP = np.zeros(particles.m.shape[0])
for i in range(zP.shape[0]):
    if np.isnan(zP[i]) or np.abs(zP[i]) < dr:
        zPPP[i] = zPP[i]
    else:
        zPPP[i] = zP[i]

"""
plt.scatter(particles.z, particles.z/zPP, s =1, color = 'red', alpha = 0.5)
plt.scatter(particles.z, particles.z/zP, s =1, color = 'green', alpha = 0.5)
plt.scatter(particles.z, particles.z/zPPP, s =1, color = 'blue', alpha = 0.5)
plt.show()
"""

mP = particles.m*zPPP/particles.z
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
    rho[k] = rho_model.ev(rc, zPPP[k])
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
filename = 'init_test_linear3.hdf5'
with h5py.File(filename, 'w') as f:
    save_picle_data(f, np.array([x, y, zPPP]).T, np.array([vx, vy, vz]).T,
                    mP, A1_h, rho, A1_P, u, A1_id, A1_mat_id,
                    4*Rs, swift_to_SI) 


#######
###### test: spherical case
data = pd.read_csv("1layer_n10k.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]
dr = data.R[0] - data.R[1]

radii = np.flip(np.array(data.R))
densities = np.flip(np.array(data.rho))

# Generate seagen sphere
N = 10**5
particles = seagen.GenSphere(N, radii[1:]*R_earth, densities[1:])

vx = np.zeros(particles.m.shape[0])
vy = np.zeros(particles.m.shape[0])
vz = np.zeros(particles.m.shape[0])

A1_P = np.ones((particles.m.shape[0],))
A1_id = np.arange(particles.m.shape[0])
A1_mat_id = np.ones((particles.m.shape[0],))*Di_mat_id['Til_granite']

u = np.zeros((particles.m.shape[0]))
for k in range(particles.m.shape[0]):
    rho_spherical = particles.A1_rho[k]
    u[k] = spipgen.ucold(rho_spherical, spipgen.granite, 10000) + spipgen.granite[11]*Ts
## Smoothing lengths, crudely estimated from the densities

num_ngb = 48    # Desired number of neighbours
w_edge  = 2     # r/h at which the kernel goes to zero
A1_h    = np.cbrt(num_ngb * particles.m / (4/3*np.pi * particles.rho)) / w_edge

# units

swift_to_SI = Conversions(1, 1, 1)

filename = 'init_test_spherical.hdf5'
with h5py.File(filename, 'w') as f:
    save_picle_data(f, np.array([particles.x, particles.y, particles.z]).T, np.array([vz, vz, vz]).T,
                    particles.m, A1_h, particles.rho, A1_P, u, A1_id, A1_mat_id,
                    4*Rs*R_earth, swift_to_SI) 


############ plot
data = pd.read_csv("1layer_n200.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]
dr = data.R[0] - data.R[1]

rho = np.load("profile_n200.npy")
rho_grid = rho[-1,:,:]
Re = (np.sum(rho_grid[:, 0] > 0) - 1)*dr

r = np.arange(0, rho_grid.shape[0]*dr, dr)
z = np.arange(0, rho_grid.shape[1]*dr, dr)
rho_model = RectBivariateSpline(r, z, rho_grid, kx = 1, ky = 1)    

# Get equatorial profile
radii = np.arange(0, Re, dr/100000)
densities = rho_model.ev(radii,0)

# Generate seagen sphere
N = 10**5
particles = seagen.GenSphere(N, radii[1:]*R_earth, densities[1:])

rho_seagen = _eq_density(particles.x/R_earth, particles.y/R_earth, particles.z/R_earth, radii, densities)
#rho_seagen = particles.A1_rho

zP = np.zeros(particles.m.shape[0])

for i in range(particles.m.shape[0]):
    rc = np.sqrt(particles.x[i]**2 + particles.y[i]**2)/R_earth
    rho_spherical = rho_seagen[i]
    z = particles.z[i]/R_earth
    f = lambda z: rho_model.ev(rc, np.abs(z)) - rho_spherical
    if z < 0:
        zP[i] = _bisection(f, z, 0.)
        
    else:
        zP[i] = _bisection(f, 0., z)
        
mask = np.isfinite(zP)
plt.scatter(particles.z/R_earth, zP, s = 1, alpha = 0.15, color = 'blue', label = 'model grid 400x200')
plt.axes().axhline(color = 'black')
plt.xlabel(r"$z$ $[R_{earth}]$")
plt.ylabel(r"$z'$ $[R_{earth}]$")
##
data = pd.read_csv("1layer_n100.csv", header=0)
Rs = data.R[0]
Ts = data['T'][0]
dr = data.R[0] - data.R[1]

rho = np.load("profile_n100.npy")
rho_grid = rho[-1,:,:]
Re = (np.sum(rho_grid[:, 0] > 0) - 1)*dr

r = np.arange(0, rho_grid.shape[0]*dr, dr)
z = np.arange(0, rho_grid.shape[1]*dr, dr)
rho_model = RectBivariateSpline(r, z, rho_grid, kx = 1, ky = 1)    

# Get equatorial profile
radii = np.arange(0, Re, dr/100000)
densities = rho_model.ev(radii,0)

# Generate seagen sphere
N = 10**5
particles = seagen.GenSphere(N, radii[1:]*R_earth, densities[1:])

rho_seagen = _eq_density(particles.x/R_earth, particles.y/R_earth, particles.z/R_earth, radii, densities)
#rho_seagen = particles.A1_rho

zP = np.zeros(particles.m.shape[0])

for i in range(particles.m.shape[0]):
    rc = np.sqrt(particles.x[i]**2 + particles.y[i]**2)/R_earth
    rho_spherical = rho_seagen[i]
    z = particles.z[i]/R_earth
    f = lambda z: rho_model.ev(rc, np.abs(z)) - rho_spherical
    if z < 0:
        zP[i] = _bisection(f, z, 0.)
        
    else:
        zP[i] = _bisection(f, 0., z)
        
mask = np.isfinite(zP)
plt.scatter(particles.z/R_earth, zP, s = 1, alpha = 0.15, color = 'red', label = 'model grid 200x100')
lgnd = plt.legend(loc="lower left", scatterpoints=1, fontsize=10)
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]
plt.show()