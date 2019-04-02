#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:11:10 2019

@author: Sergio Ruiz-Bonilla
"""

###############################################################################
####################### Libraries and constants ###############################
###############################################################################

path = '/home/sergio/Documents/SpiPGen/'
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(path)
import sys
sys.path.append(path)
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

R             = R_earth                     # Radius of the planet 
M_max         = M_earth                     # Upper bound for the mass of the planet
N             = 10000                       # Number of integration steps
Ts            = 300.                        # Temperature at the surface
Ps            = 0.                          # Pressure at the surface
T_rho_id      = 1.                          # Relation between density and temperature (T_rho_id.txt)
T_rho_args    = [np.nan, 0.]                # Extra arguments for the above relation 
mat_id        = 101                         # Material id (see mat_id.txt)
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
b_cm              = 0.42*R_earth           # Boundary core-mantle

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
M_max             = 200*M_earth                 # Mass of the planet
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
