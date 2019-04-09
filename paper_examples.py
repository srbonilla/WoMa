#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:36:24 2019

@author: Sergio Ruiz-Bonilla
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################

path = '/home/sergio/Documents/SpiPGen/'     # Set local project folder
import os
os.chdir(path)
import sys
sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt
import woma
import eos

import swift_io
import h5py

R_earth = 6371000;
M_earth = 5.972E24;

###############################################################################
########################## Initial set up #####################################
###############################################################################

# Only need to run once
#woma.set_up()

###############################################################################
########################## 1 layer planet #####################################
###############################################################################

N                  = 10000                       # Number of integration steps
R                  = R_earth                     # Radius of the planet 
M_max              = M_earth                     # Upper bound for the mass of the planet
Ts                 = 300.                        # Temperature at the surface
Ps                 = 0.                          # Pressure at the surface
mat_id_core        = 101                         # Material id (see mat_id.txt)
T_rho_id_core      = 1.                          # Relation between density and temperature (T_rho_id.txt)
T_rho_args_core    = [np.nan, 0.]                # Extra arguments for the above relation 
rhos_min           = 2000.                       # Lower bound for the density at the surface
rhos_max           = 3000.                       # Upper bound for the density at the surface

# Load precomputed values of cold internal energy
ucold_array_core = eos.load_ucold_array(mat_id_core) 

# Find the correct mass for the planet
M = woma.find_mass_1layer(N, R, M_max, Ps, Ts, mat_id_core, T_rho_id_core, T_rho_args_core,
                          rhos_min, rhos_max, ucold_array_core)

# Compute the whole profile
r, m, P, T, rho, u, mat =  \
    woma.integrate_1layer(N, R, M, Ps, Ts,
                          mat_id_core, T_rho_id_core, T_rho_args_core,
                          rhos_min, rhos_max, ucold_array_core)

# Plot the results
# =============================================================================
# fig, ax = plt.subplots(2,2, figsize=(12,12))
# 
# ax[0,0].plot(r/R_earth, rho)
# ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
# 
# ax[1,0].plot(r/R_earth, m/M_earth)
# ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")
# 
# ax[0,1].plot(r/R_earth, P)
# ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[0,1].set_ylabel(r"$P$ $[Pa]$")
# 
# ax[1,1].plot(r/R_earth, T)
# ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[1,1].set_ylabel(r"$T$ $[K]$")
# 
# plt.tight_layout()
# plt.show()
# =============================================================================

# Let's convet it to a spining profile
iterations = 30                              # Iterations to convergence
r_array    = np.linspace(0, 1.2*R, 2000)     # Points at equatorial profile to find the solution
z_array    = np.linspace(0, 1.1*R, 2000)     # Points at equatorial profile to find the solution
Tw         = 4                               # Period of the planet [hours]

P_c   = P[-1]                                # Pressure at the center
P_s   = P[0]                                 # Pressure at the surface
rho_c = rho[-1]                              # Density at the center
rho_s = rho[0]                               # Density at the surface

# Compute equatorial and polar profiles
profile_e, profile_p = woma.spin1layer(iterations, r_array, z_array, r, rho, Tw,
                                       P_c, P_s, rho_c, rho_s,
                                       mat_id_core, T_rho_id_core, T_rho_args_core,
                                       ucold_array_core)

# Keep last iteration of the computation
rho_e = profile_e[-1]
rho_p = profile_p[-1]
# =============================================================================
# 
# np.save('r_array', r_array)
# np.save('z_array', z_array)
# np.save('rho_e', rho_e)
# np.save('rho_p', rho_p)
# 
# =============================================================================
# Particle placement and save data

N_picles = 1e5    # Number of particles
N_neig   = 48     # Number of neighbors
x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id =                      \
woma.picle_placement_1layer(r_array, rho_e, z_array, rho_p, Tw, N_picles,
                            mat_id_core, T_rho_id_core, T_rho_args_core,
                            ucold_array_core, N_neig)

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '1layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, np.array([x, y, z]).T, np.array([vx, vy, vz]).T,
                             m, h, rho, P, u, picle_id, mat_id,
                             4*R_earth, swift_to_SI) 

###############################################################################
########################## 2 layer planet #####################################
###############################################################################

# Create spherical profile (determine mass of the planet given radius and boundary)

N                 = 10000                  # Number of integration steps
R                 = R_earth                # Radius of the planet 
M_max             = 2*M_earth              # Upper bound for the mass of the planet
Ts                = 300.                   # Temperature at the surface
Ps                = 0.                     # Pressure at the surface
mat_id_core       = 100                    # Material id for the core (see mat_id.txt)
T_rho_id_core     = 1.                     # Relation between density and temperature for the core (T_rho_id.txt)
T_rho_args_core   = [np.nan, 0.]           # Extra arguments for the above relation 
mat_id_mantle     = 101                    # Material id for the mantle (see mat_id.txt)
T_rho_id_mantle   = 1.                     # Relation between density and temperature for the mantle (T_rho_id.txt)
T_rho_args_mantle = [np.nan, 0.]           # Extra arguments for the above relation 
rhos_min          = 2000.                  # Lower bound for the density at the surface
rhos_max          = 3000.                  # Upper bound for the density at the surface
b_cm              = 0.426*R_earth          # Boundary core-mantle

# Load precomputed values of cold internal energy
ucold_array_core   = woma.load_ucold_array(mat_id_core)
ucold_array_mantle = woma.load_ucold_array(mat_id_mantle)

# Find correct mass for the planet
M = woma.find_mass_2layer(N, R, M_max, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

# Compute the whole profile
r, m, P, T, rho, u, mat = \
    woma.integrate_2layer(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

# Plot the results
# =============================================================================
# fig, ax = plt.subplots(2,2, figsize=(12,12))
# 
# ax[0,0].plot(r/R_earth, rho)
# ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
# 
# ax[1,0].plot(r/R_earth, m/M_earth)
# ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")
# 
# ax[0,1].plot(r/R_earth, P)
# ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[0,1].set_ylabel(r"$P$ $[Pa]$")
# 
# ax[1,1].plot(r/R_earth, T)
# ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
# ax[1,1].set_ylabel(r"$T$ $[K]$")
# 
# plt.tight_layout()
# plt.show()
# =============================================================================

# Let's convet it to a spining profile
iterations = 30                              # Iterations to convergence
r_array    = np.linspace(0, 1.2*R, 1000)     # Points at equatorial profile to find the solution
z_array    = np.linspace(0, 1.1*R, 1000)     # Points at equatorial profile to find the solution
Tw         = 4                               # Period of the planet [hours]

P_c   = P[-1]                                # Pressure at the center
P_s   = P[0]                                 # Pressure at the surface
rho_c = rho[-1]                              # Density at the center
rho_s = rho[0]                               # Density at the surface

rho_i = 10000                                          # Density at boundary
P_i   = (P[rho > rho_i][0] + P[rho < rho_i][-1])/2.    # Pressure at the boundary

# Compute equatorial and polar profiles
profile_e, profile_p = woma.spin2layer(iterations, r_array, z_array, r, rho, Tw,
                                       P_c, P_i, P_s, rho_c, rho_s,
                                       mat_id_core, T_rho_id_core, T_rho_args_core,
                                       mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                       ucold_array_core, ucold_array_mantle)

# Keep last iteration of the computation
rho_e = profile_e[-1]
rho_p = profile_p[-1]

# Save results
# =============================================================================
# np.save('r_array', r_array)
# np.save('z_array', z_array)
# np.save('rho_e', rho_e)
# np.save('rho_p', rho_p)
# 
# =============================================================================
# Particle placement and save data

N_picles = 1e5    # Number of particles
N_neig   = 48     # Number of neighbors
x, y, z, vx, vy, vz, m, h, rho, P, u, mat_id, picle_id =                      \
woma.picle_placement_2layer(r_array, rho_e, z_array, rho_p, Tw, N_picles, rho_i,
                            mat_id_core, T_rho_id_core, T_rho_args_core,
                            mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                            ucold_array_core, ucold_array_mantle, N_neig)

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '2layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, np.array([x, y, z]).T, np.array([vx, vy, vz]).T,
                             m, h, rho, P, u, picle_id, mat_id,
                             4*R_earth, swift_to_SI) 
