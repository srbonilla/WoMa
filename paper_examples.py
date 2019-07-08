#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:11:10 2019

@author: Sergio Ruiz-Bonilla
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import woma
import swift_io
import h5py

R_earth = 6371000
M_earth = 5.972E24

###############################################################################
########################## Initial set up #####################################
###############################################################################

# Function to plot results for spherical profile

def plot_spherical_profile(planet):
    
    fig, ax = plt.subplots(2, 2, figsize=(7,7))
    
    ax[0,0].plot(planet.A1_r/R_earth, planet.A1_rho)
    ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
    ax[0,0].set_yscale("log")
    ax[0,0].set_xlim(0, None)
    
    ax[1,0].plot(planet.A1_r/R_earth, planet.A1_m_enc/M_earth)
    ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")
    ax[1,0].set_xlim(0, None)
    
    ax[0,1].plot(planet.A1_r/R_earth, planet.A1_P)
    ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,1].set_ylabel(r"$P$ $[Pa]$")
    ax[0,1].set_xlim(0, None)
    
    ax[1,1].plot(planet.A1_r/R_earth, planet.A1_T)
    ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,1].set_ylabel(r"$T$ $[K]$")
    ax[1,1].set_xlim(0, None)
    
    plt.tight_layout()
    plt.show()
    
# Function to plot results for spining profile
    
def plot_spin_profile(spin_planet):
    
    sp = spin_planet
    
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].scatter(sp.A1_r/R_earth, sp.A1_rho, label = 'original', s = 0.5)
    ax[0].scatter(sp.A1_r_equator/R_earth, sp.A1_rho_equator, label = 'equatorial profile', s = 1)
    ax[0].scatter(sp.A1_r_pole/R_earth, sp.A1_rho_pole, label = 'polar profile', s = 1)
    ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
    ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
    ax[0].legend()
    
    
    r_array_coarse = np.linspace(0, np.max(sp.A1_r_equator), 100)
    z_array_coarse = np.linspace(0, np.max(sp.A1_r_pole), 100)
    rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
    for i in range(rho_grid.shape[0]):
        radius = r_array_coarse[i]
        for j in range(rho_grid.shape[1]):
            z = z_array_coarse[j]
            rho_grid[i,j] = woma.rho_rz(radius, z,
                                        sp.A1_r_equator, sp.A1_rho_equator,
                                        sp.A1_r_pole, sp.A1_rho_pole)
    
    X, Y = np.meshgrid(r_array_coarse/R_earth, z_array_coarse/R_earth)
    Z = rho_grid.T
    levels = np.arange(1000, 15000, 1000)
    ax[1].set_aspect('equal')
    CS = plt.contour(X, Y, Z, levels = levels)
    ax[1].clabel(CS, inline=1, fontsize=10)
    ax[1].set_xlabel(r"$r$ [$R_{earth}$]")
    ax[1].set_ylabel(r"$z$ [$R_{earth}$]")
    ax[1].set_title('Density (Kg/m^3)')
        
    plt.tight_layout()
    plt.show()
    
# Example 1 layer
l1_test = woma.Planet(
    name            = "prof_pE",
    A1_mat_layer    = ['Til_granite'],
    A1_T_rho_type   = [1],
    A1_T_rho_args   = [[None, 0.]],
    A1_R_layer      = [R_earth],
    M               = 0.8*M_earth,
    P_s             = 0,
    T_s             = 300
    )

l1_test.M_max = M_earth

l1_test.gen_prof_L1_fix_M_given_R()

plot_spherical_profile(l1_test)

l1_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l1_test,
    Tw           = 3,
    R_e          = 1.3*R_earth,
    R_p          = 1.1*R_earth
    )

l1_test_sp.spin()

plot_spin_profile(l1_test_sp)

particles = woma.GenSpheroid(
    name        = 'picles_spin',
    spin_planet = l1_test_sp,
    N_particles = 1e5)

positions = np.array([particles.A1_picle_x, particles.A1_picle_y, particles.A1_picle_z]).T
velocities = np.array([particles.A1_picle_vx, particles.A1_picle_vy, particles.A1_picle_vz]).T

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '1layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, positions, velocities,
                             particles.A1_picle_m, particles.A1_picle_h,
                             particles.A1_picle_rho, particles.A1_picle_P, particles.A1_picle_u,
                             particles.A1_picle_id, particles.A1_picle_mat_id,
                             4*R_earth, swift_to_SI) 
    
np.save('r_array', l1_test_sp.A1_r_equator)
np.save('z_array', l1_test_sp.A1_r_pole)
np.save('rho_e', l1_test_sp.A1_rho_equator)
np.save('rho_p', l1_test_sp.A1_rho_pole)
    
# Example 2 layer

l2_test = woma.Planet(
    name            = "prof_pE",
    A1_mat_layer    = ['Til_iron', 'Til_granite'],
    A1_T_rho_type   = [1, 1],
    A1_T_rho_args   = [[None, 0.], [None, 0.]],
    A1_R_layer      = [None, R_earth],
    M               = M_earth,
    P_s             = 0,
    T_s             = 300
    )

l2_test.gen_prof_L2_fix_R1_given_R_M()

plot_spherical_profile(l2_test)

l2_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l2_test,
    Tw           = 2.6,
    R_e          = 1.45*R_earth,
    R_p          = 1.1*R_earth
    )

l2_test_sp.spin()

plot_spin_profile(l2_test_sp)

particles = woma.GenSpheroid(
    name        = 'picles_spin',
    spin_planet = l2_test_sp,
    N_particles = 1e5)

positions = np.array([particles.A1_picle_x, particles.A1_picle_y, particles.A1_picle_z]).T
velocities = np.array([particles.A1_picle_vx, particles.A1_picle_vy, particles.A1_picle_vz]).T

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '2layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, positions, velocities,
                             particles.A1_picle_m, particles.A1_picle_h,
                             particles.A1_picle_rho, particles.A1_picle_P, particles.A1_picle_u,
                             particles.A1_picle_id, particles.A1_picle_mat_id,
                             4*R_earth, swift_to_SI) 
    
np.save('r_array', l2_test_sp.A1_r_equator)
np.save('z_array', l2_test_sp.A1_r_pole)
np.save('rho_e', l2_test_sp.A1_rho_equator)
np.save('rho_p', l2_test_sp.A1_rho_pole)

# Squash and compute SPH density#############################
import seagen
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d

N = 100000
#radii     = np.linspace(0, R_earth, 100000)
#densities = np.ones(100000)*1000       # uniform density
#densities = np.linspace(6000, 1000, 100000)

#radii     = prof_pE.A1_r
#densities = prof_pE.A1_rho

radii     = l1_test.A1_r_equator
densities = l1_test.A1_rho_equator

rho_model = interp1d(radii, densities)
f = 0.5

particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)

zP = particles.z*f
mP = particles.m*f

x_reshaped  = particles.x.reshape((-1,1))
y_reshaped  = particles.y.reshape((-1,1))
zP_reshaped = zP.reshape((-1,1))

X = np.hstack((x_reshaped, y_reshaped, zP_reshaped))

del x_reshaped, y_reshaped, zP_reshaped

w_edge  = 2     # r/h at which the kernel goes to zero
N_neig  = 48
particles_r = np.sqrt(particles.x**2 + particles.y**2 + particles.z**2)
rho     = rho_model(particles.r)
h       = np.cbrt(N_neig * mP / (4/3*np.pi * rho)) / w_edge

nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
nbrs.fit(X)

distances, indices = nbrs.kneighbors(X)
M = woma._generate_M(indices, mP)
rho_sph = woma.SPH_density(M, distances, h)

diff = (rho_sph - rho)/rho

plt.figure()
plt.scatter(zP/R_earth, diff, s = 0.5, alpha=0.5)
plt.xlabel(r"z [$R_{earth}$]")
plt.ylabel(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
plt.show()

####################
N = 100000
r_array = l1_test.A1_r_equator
z_array = l1_test.A1_r_pole
rho_e = l1_test.A1_rho_equator
rho_p = l1_test.A1_rho_pole

rho_e_model = interp1d(r_array, rho_e)
rho_p_model_inv = interp1d(rho_p, z_array)

Re = np.max(r_array[rho_e > 0])

radii = np.arange(0, Re, Re/1000000)
densities = rho_e_model(radii)

particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)

particles_r = np.sqrt(particles.x**2 + particles.y**2 + particles.z**2)
rho = rho_e_model(particles_r)

R = particles.A1_r.copy()
rho_layer = rho_e_model(R)
Z = rho_p_model_inv(rho_layer)

f = Z/R
zP = particles.z*f

mP = particles.m*f

# Compute velocities (T_w in hours)
# =============================================================================
# vx = np.zeros(mP.shape[0])
# vy = np.zeros(mP.shape[0])
# vz = np.zeros(mP.shape[0])
# 
# Tw = 4
# hour_to_s = 3600
# wz = 2*np.pi/Tw/hour_to_s
# 
# vx = -particles.y*wz
# vy = particles.x*wz
# =============================================================================

# internal energy
# =============================================================================
# u = np.zeros((mP.shape[0]))
# 
# x = particles.x
# y = particles.y
# 
# P = np.zeros((mP.shape[0],))
# 
# for k in range(mP.shape[0]):
#     T = weos.T_rho(rho[k], T_rho_type_L1, T_rho_args_L1, mat_id_L1)
#     u[k] = weos._find_u(rho[k], mat_id_L1, T, u_cold_array_L1)
#     P[k] = weos.P_EoS(u[k], rho[k], mat_id_L1)
# =============================================================================

#print("Internal energy u computed\n")
# Smoothing lengths, crudely estimated from the densities
w_edge  = 2     # r/h at which the kernel goes to zero
h       = np.cbrt(N_neig * mP / (4/3*np.pi * rho)) / w_edge

# =============================================================================
# A1_id     = np.arange(mP.shape[0])
# A1_mat_id = np.ones((mP.shape[0],))*mat_id_L1
# =============================================================================

mP = particles.m*f
unique_R = np.unique(R)

x_reshaped  = particles.x.reshape((-1,1))
y_reshaped  = particles.y.reshape((-1,1))
zP_reshaped = zP.reshape((-1,1))

X = np.hstack((x_reshaped, y_reshaped, zP_reshaped))

del x_reshaped, y_reshaped, zP_reshaped

nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
nbrs.fit(X)

distances, indices = nbrs.kneighbors(X)
M = woma._generate_M(indices, mP)
rho_sph = woma.SPH_density(M, distances, h)

diff = (rho_sph - rho)/rho

plt.figure()
plt.scatter(zP/R_earth, diff, s = 0.5, alpha=0.5)
plt.xlabel(r"z [$R_{earth}$]")
plt.ylabel(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
plt.show()

# paper plots #################################################################
# spherical profiles
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(1, 1, figsize=(9,9))
    
ax.scatter(l1_test.A1_r/R_earth, l1_test.A1_rho, label='1 layer test', s=5)
ax.scatter(l2_test.A1_r/R_earth, l2_test.A1_rho, label='2 layer test', s=5)
ax.set_xlabel(r"$r$ $[R_{earth}]$")
ax.set_ylabel(r"$\rho$ $[kg/m^3]$")
ax.set_xlim(0, None)
#ax.set_yscale("log")
plt.legend(markerscale=3)
plt.tight_layout()
#plt.savefig('Fig1' + ".pdf", dpi=400)
plt.show()

# spining profiles
plt.rcParams.update({'font.size': 20})

fig, ax = plt.subplots(2, 1, figsize=(9,12), sharex=True)
    
mask_e = l1_test_sp.A1_rho_equator > 0
mask_p = l1_test_sp.A1_rho_pole > 0
ax[0].scatter(l1_test_sp.A1_r_equator[mask_e]/R_earth, l1_test_sp.A1_rho_equator[mask_e],
              label = 'equatorial profile', s = 5)
ax[0].scatter(l1_test_sp.A1_r_pole[mask_p]/R_earth, l1_test_sp.A1_rho_pole[mask_p],
              label = 'polar profile', s = 5)
#ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
#ax[0].set_yscale("log")
ax[0].legend(markerscale=3, loc='upper right')

mask_e = l2_test_sp.A1_rho_equator > 0
mask_p = l2_test_sp.A1_rho_pole > 0
ax[1].scatter(l2_test_sp.A1_r_equator[mask_e]/R_earth, l2_test_sp.A1_rho_equator[mask_e],
              label = 'equatorial profile', s = 5)
ax[1].scatter(l2_test_sp.A1_r_pole[mask_p]/R_earth, l2_test_sp.A1_rho_pole[mask_p],
              label = 'polar profile', s = 5)
ax[1].set_xlabel(r"$r$ [$R_{earth}$]")
ax[1].set_ylabel(r"$\rho$ [$kg/m^3$]")
ax[1].legend(markerscale=3)

#ax[1].set_yscale("log")

plt.tight_layout()
plt.show()
plt.savefig('Fig2' + ".pdf", dpi=400)

# Convergence of the method
plt.rcParams.update({'font.size': 20})

r1 = []
rho_e1 = []
r2 = []
rho_e2 = []

l1_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l1_test,
    Tw           = 3,
    R_e          = 1.3*R_earth,
    R_p          = 1.1*R_earth
    )

l2_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l2_test,
    Tw           = 2.6,
    R_e          = 1.45*R_earth,
    R_p          = 1.1*R_earth
    )

for i in range(20):
    l1_test_sp.num_attempt = i
    l2_test_sp.num_attempt = i
    
    l1_test_sp.spin()
    r1.append(l1_test_sp.A1_r_equator)
    rho_e1.append(l1_test_sp.A1_rho_equator)
    
    l2_test_sp.spin()
    r2.append(l2_test_sp.A1_r_equator)
    rho_e2.append(l2_test_sp.A1_rho_equator)
    
fig, ax = plt.subplots(2, 1, figsize=(9,12), sharex=True)

for i in np.arange(0,11,2):
    mask = rho_e1[i] > 0
    ax[0].scatter(r1[i][mask]/R_earth, rho_e1[i][mask], label=str(i),
                  s=1, cmap='viridis')
    
    mask = rho_e2[i] > 0
    ax[1].scatter(r2[i][mask]/R_earth, rho_e2[i][mask], label=str(i),
                  s=1, cmap='viridis')

#ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
ax[1].set_xlabel(r"$r$ [$R_{earth}$]")
ax[0].set_ylabel(r"$\rho_{equator}$ [$kg/m^3$]")
ax[1].set_xlabel(r"$\rho_{equator}$ [$kg/m^3$]")


# produce a legend with the unique colors from the scatter
ax[0].legend(title='Iteration', markerscale=3)
ax[1].legend(title='Iteration', markerscale=3)

plt.tight_layout()
plt.show()