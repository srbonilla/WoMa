#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:11:10 2019

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

R_earth = 6371000;
M_earth = 5.972E24;

###############################################################################
########################## Initial set up #####################################
###############################################################################

# Only need to run once

woma.set_up()

###############################################################################
########################## 1 layer planet #####################################
###############################################################################

########################## Example 1.1 ########################################

# Create spherical profile (determine mass of the planet given radius)

N             = 10000                       # Number of integration steps
R             = R_earth                     # Radius of the planet 
M_max         = M_earth                     # Upper bound for the mass of the planet
Ts            = 300.                        # Temperature at the surface
Ps            = 0.                          # Pressure at the surface
mat_id        = 101                         # Material id (see mat_id.txt)
T_rho_id      = 1.                          # Relation between density and temperature (T_rho_id.txt)
T_rho_args    = [np.nan, 0.]                # Extra arguments for the above relation 
rhos_min      = 2000.                       # Lower bound for the density at the surface
rhos_max      = 3000.                       # Upper bound for the density at the surface

# Load precomputed values of cold internal energy
ucold_array = woma.load_ucold_array(mat_id) 

# Find the correct mass for the planet
M = woma.find_mass_1layer(N, R, M_max, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                          rhos_min, rhos_max, ucold_array)

# Compute the whole profile
r, m, P, T, rho, u, mat = woma.integrate_1layer(N, R, M, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                                           rhos_min, rhos_max, ucold_array)

# Plot the results
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m/M_earth)
ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")

ax[0,1].plot(r/R_earth, P)
ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,1].set_ylabel(r"$P$ $[Pa]$")

ax[1,1].plot(r/R_earth, T)
ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,1].set_ylabel(r"$T$ $[K]$")

plt.tight_layout()
plt.show()

# Let's convet it to a spining profile
iterations = 10                              # Iterations to convergence
r_array    = np.linspace(0, 1.2*R, 1000)     # Points at equatorial profile to find the solution
z_array    = np.linspace(0, 1.2*R, 1000)     # Points at equatorial profile to find the solution
Tw         = 4                               # Period of the planet [hours]

P_c   = P[-1]                                # Pressure at the center
P_s   = P[0]                                 # Pressure at the surface
rho_c = rho[-1]                              # Density at the center
rho_s = rho[0]                               # Density at the surface

# Compute equatorial and polar profiles
profile_e, profile_p = woma.spin1layer(iterations, r_array, z_array, r, rho, Tw,
                                       P_c, P_s, rho_c, rho_s,
                                       mat_id, T_rho_id, T_rho_args,
                                       ucold_array)

# Keep last iteration of the computation
rho_e = profile_e[-1]
rho_p = profile_p[-1]

# Plot the results
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].scatter(r/R_earth, rho, label = 'original', s = 0.5)
ax[0].scatter(r_array/R_earth, rho_e, label = 'equatorial profile', s = 1)
ax[0].scatter(z_array/R_earth, rho_p, label = 'polar profile', s = 1)
ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
ax[0].legend()


r_array_coarse = np.arange(0, np.max(r_array), np.max(r_array)/100)
z_array_coarse = np.arange(0, np.max(z_array), np.max(z_array)/100)
rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
for i in range(rho_grid.shape[0]):
    radius = r_array_coarse[i]
    for j in range(rho_grid.shape[1]):
        z = z_array_coarse[j]
        rho_grid[i,j] = woma.rho_rz(radius, z, r_array, rho_e, z_array, rho_p)

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

###############################################################################
########################## 2 layer planet #####################################
###############################################################################

########################## Example 2.1 ########################################

# Create spherical profile (determine boundary of the planet given radius and mass)

N                 = 10000                  # Number of integration steps
R                 = R_earth                # Radius of the planet 
M                 = M_earth                # Mass of the planet
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

# Load precomputed values of cold internal energy
ucold_array_core   = woma.load_ucold_array(mat_id_core)
ucold_array_mantle = woma.load_ucold_array(mat_id_mantle)

# Find correct boundary (core-mantle) for the planet
b_cm = woma.find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

# Slighly tweek the mass of the planet to avoid peeks at the center
M_tweek = woma.find_mass_2layer(N, R, 2*M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

# Compute the whole profile
r, m, P, T, rho, u, mat = \
    woma.integrate_2layer(N, R, M_tweek, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

# Plot the results
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m/M_earth)
ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")

ax[0,1].plot(r/R_earth, P)
ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,1].set_ylabel(r"$P$ $[Pa]$")

ax[1,1].plot(r/R_earth, T)
ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,1].set_ylabel(r"$T$ $[K]$")

plt.tight_layout()
plt.show()

########################## Example 2.2 ########################################

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
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m/M_earth)
ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")

ax[0,1].plot(r/R_earth, P)
ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,1].set_ylabel(r"$P$ $[Pa]$")

ax[1,1].plot(r/R_earth, T)
ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,1].set_ylabel(r"$T$ $[K]$")

plt.tight_layout()
plt.show()

# Let's convet it to a spining profile
iterations = 10                               # Iterations to convergence
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

# Plot the results
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].scatter(r/R_earth, rho, label = 'original', s = 0.5)
ax[0].scatter(r_array/R_earth, rho_e, label = 'equatorial profile', s = 1)
ax[0].scatter(z_array/R_earth, rho_p, label = 'polar profile', s = 1)
ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
ax[0].legend()


r_array_coarse = np.arange(0, np.max(r_array), np.max(r_array)/100)
z_array_coarse = np.arange(0, np.max(z_array), np.max(z_array)/100)
rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
for i in range(rho_grid.shape[0]):
    radius = r_array_coarse[i]
    for j in range(rho_grid.shape[1]):
        z = z_array_coarse[j]
        rho_grid[i,j] = woma.rho_rz(radius, z, r_array, rho_e, z_array, rho_p)

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

###############################################################################
########################## 3 layer planet #####################################
###############################################################################

########################## Example 3.1 ########################################

# Create spherical profile (determine boundaries of the planet given mass and moment of inertia)

N                 = 10000                     # Number of integration steps
R                 = R_earth                   # Radius of the planet 
M                 = M_earth                   # Mass of the planet
Ts                = 300.                      # Temperature at the surface
Ps                = 0.                        # Pressure at the surface
mat_id_core       = 100                       # Material id for the core (see mat_id.txt)
T_rho_id_core     = 1.                        # Relation between density and temperature for the core (T_rho_id.txt)
T_rho_args_core   = [np.nan, 0.]              # Extra arguments for the above relation 
mat_id_mantle     = 101                       # Material id for the mantle (see mat_id.txt)
T_rho_id_mantle   = 1.                        # Relation between density and temperature for the mantle (T_rho_id.txt)
T_rho_args_mantle = [np.nan, 0.]              # Extra arguments for the above relation 
mat_id_atm        = 102                       # Material id for the atmosphere (see mat_id.txt)
T_rho_id_atm      = 1.                        # Relation between density and temperature for the atmosphere (T_rho_id.txt)
T_rho_args_atm    = [np.nan, 0.]              # Extra arguments for the above relation 
rhos_min          = 800.                      # Lower bound for the density at the surface
rhos_max          = 1000.                     # Upper bound for the density at the surface
MoI               = 0.3*M_earth*(R_earth**2)  # Moment of inertia

# Load precomputed values of cold internal energy
ucold_array_core   = woma.load_ucold_array(mat_id_core)
ucold_array_mantle = woma.load_ucold_array(mat_id_mantle)
ucold_array_atm    = woma.load_ucold_array(mat_id_atm)

# Find correct boundaries (core-mantle and mantle-atmosphere) for the planet
b_cm, b_ma = woma.find_boundaries_3layer(N, R, M, Ps, Ts, MoI, 
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Slighly tweek the mass of the planet to avoid peeks at the center
M_tweek = woma.find_mass_3layer(N, R, 2*M, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Compute the whole profile
r, m, P, T, rho, u, mat = \
    woma.integrate_3layer(N, R, M_tweek, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Plot the results
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m/M_earth)
ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")

ax[0,1].plot(r/R_earth, P)
ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,1].set_ylabel(r"$P$ $[Pa]$")

ax[1,1].plot(r/R_earth, T)
ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,1].set_ylabel(r"$T$ $[K]$")

plt.tight_layout()
plt.show()

########################## Example 3.2 ########################################

# Create spherical profile (determine boundary mantle-atmosphere of the planet 
# given mass and boundary core-mantle)

N                 = 10000                     # Number of integration steps
R                 = R_earth                   # Radius of the planet 
M                 = M_earth                   # Mass of the planet
Ts                = 300.                      # Temperature at the surface
Ps                = 0.                        # Pressure at the surface
mat_id_core       = 100                       # Material id for the core (see mat_id.txt)
T_rho_id_core     = 1.                        # Relation between density and temperature for the core (T_rho_id.txt)
T_rho_args_core   = [np.nan, 0.]              # Extra arguments for the above relation 
mat_id_mantle     = 101                       # Material id for the mantle (see mat_id.txt)
T_rho_id_mantle   = 1.                        # Relation between density and temperature for the mantle (T_rho_id.txt)
T_rho_args_mantle = [np.nan, 0.]              # Extra arguments for the above relation 
mat_id_atm        = 102                       # Material id for the atmosphere (see mat_id.txt)
T_rho_id_atm      = 1.                        # Relation between density and temperature for the atmosphere (T_rho_id.txt)
T_rho_args_atm    = [np.nan, 0.]              # Extra arguments for the above relation 
rhos_min          = 800.                      # Lower bound for the density at the surface
rhos_max          = 1000.                     # Upper bound for the density at the surface
b_cm              = 0.49*R_earth              # Boundary core-mantle

# Load precomputed values of cold internal energy
ucold_array_core   = woma.load_ucold_array(mat_id_core)
ucold_array_mantle = woma.load_ucold_array(mat_id_mantle)
ucold_array_atm    = woma.load_ucold_array(mat_id_atm)

# Find correct boundary mantle atmosphere for the planet
b_ma = woma.find_b_ma_3layer(N, R, M, Ps, Ts, b_cm, 
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Compute the whole profile
r, m, P, T, rho, u, mat = \
    woma.integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Plot the results
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m/M_earth)
ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")

ax[0,1].plot(r/R_earth, P)
ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,1].set_ylabel(r"$P$ $[Pa]$")

ax[1,1].plot(r/R_earth, T)
ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,1].set_ylabel(r"$T$ $[K]$")

plt.tight_layout()
plt.show()

########################## Example 3.3 ########################################

# Create spherical profile (determine mass of the planet given the boundaries)

N                 = 10000                     # Number of integration steps
R                 = R_earth                   # Radius of the planet 
M_max             = 2*M_earth                 # Mass of the planet
Ts                = 300.                      # Temperature at the surface
Ps                = 0.                        # Pressure at the surface
mat_id_core       = 100                       # Material id for the core (see mat_id.txt)
T_rho_id_core     = 1.                        # Relation between density and temperature for the core (T_rho_id.txt)
T_rho_args_core   = [np.nan, 0.]              # Extra arguments for the above relation 
mat_id_mantle     = 101                       # Material id for the mantle (see mat_id.txt)
T_rho_id_mantle   = 1.                        # Relation between density and temperature for the mantle (T_rho_id.txt)
T_rho_args_mantle = [np.nan, 0.]              # Extra arguments for the above relation 
mat_id_atm        = 102                       # Material id for the atmosphere (see mat_id.txt)
T_rho_id_atm      = 1.                        # Relation between density and temperature for the atmosphere (T_rho_id.txt)
T_rho_args_atm    = [np.nan, 0.]              # Extra arguments for the above relation 
rhos_min          = 800.                      # Lower bound for the density at the surface
rhos_max          = 1000.                     # Upper bound for the density at the surface
b_cm              = 0.49*R_earth              # Boundary core-mantle
b_ma              = 0.96*R_earth              # Boundary mantle-atmosphere

# Load precomputed values of cold internal energy
ucold_array_core   = woma.load_ucold_array(mat_id_core)
ucold_array_mantle = woma.load_ucold_array(mat_id_mantle)
ucold_array_atm    = woma.load_ucold_array(mat_id_atm)

# Find the correct mass
M = woma.find_mass_3layer(N, R, M_max, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Compute the whole profile
r, m, P, T, rho, u, mat = \
    woma.integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Plot the results
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m/M_earth)
ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")

ax[0,1].plot(r/R_earth, P)
ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,1].set_ylabel(r"$P$ $[Pa]$")

ax[1,1].plot(r/R_earth, T)
ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
ax[1,1].set_ylabel(r"$T$ $[K]$")

plt.tight_layout()
plt.show()

# Let's convet it to a spining profile
iterations = 10                              # Iterations to convergence
r_array    = np.linspace(0, 1.2*R, 1000)     # Points at equatorial profile to find the solution
z_array    = np.linspace(0, 1.1*R, 1000) # Points at equatorial profile to find the solution
Tw         = 4                               # Period of the planet [hours]

P_c   = P[-1]                                # Pressure at the center
P_s   = P[0]                                 # Pressure at the surface
rho_c = rho[-1]                              # Density at the center
rho_s = rho[0]                               # Density at the surface

rho_cm = 10000                                          # Density at boundary core-mantle
P_cm   = (P[rho > rho_cm][0] + P[rho < rho_cm][-1])/2.  # Pressure at the boundary core-mantle
rho_ma = 2000                                           # Density at boundary core-mantle
P_ma   = (P[rho > rho_ma][0] + P[rho < rho_ma][-1])/2.  # Pressure at the boundary core-mantle

# Compute equatorial and polar profiles
profile_e, profile_p = woma.spin3layer(iterations, r_array, z_array, r, rho, Tw,
                                       P_c, P_cm, P_ma, P_s, rho_c, rho_s,
                                       mat_id_core, T_rho_id_core, T_rho_args_core,
                                       mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                       mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                                       ucold_array_core, ucold_array_mantle, ucold_array_atm)

# Keep last iteration of the computation
rho_e = profile_e[-1]
rho_p = profile_p[-1]

# Plot the results
fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].scatter(r/R_earth, rho, label = 'original', s = 0.5)
ax[0].scatter(r_array/R_earth, rho_e, label = 'equatorial profile', s = 1)
ax[0].scatter(z_array/R_earth, rho_p, label = 'polar profile', s = 1)
ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
ax[0].legend()

r_array_coarse = np.arange(0, np.max(r_array), np.max(r_array)/100)
z_array_coarse = np.arange(0, np.max(z_array), np.max(z_array)/100)
rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
for i in range(rho_grid.shape[0]):
    radius = r_array_coarse[i]
    for j in range(rho_grid.shape[1]):
        z = z_array_coarse[j]
        rho_grid[i,j] = woma.rho_rz(radius, z, r_array, rho_e, z_array, rho_p)

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



