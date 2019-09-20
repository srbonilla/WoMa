#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:48:11 2019

@author: sergio
"""

import seagen
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import woma
from scipy.interpolate import interp1d
from tqdm import tqdm

def vol_spheroid(R, Z):
    return 4*np.pi*R*R*Z/3

def integrand(theta, R_l, Z_l, R_h, Z_h):
    
    r_h = np.sin(theta)**2/R_h/R_h + np.cos(theta)**2/Z_h/Z_h
    r_h = np.sqrt(1/r_h)
    
    r_l = np.sin(theta)**2/R_l/R_l + np.cos(theta)**2/Z_l/Z_l
    r_l = np.sqrt(1/r_l)
    
    I = 2*np.pi*(r_h**3 - r_l**3)*np.sin(theta)/3
    
    return I
    
def V_theta(theta_0, theta_1, shell_config):
    
    R_l, Z_l = shell_config[0]
    R_h, Z_h = shell_config[1]
    
    assert R_h >= R_l
    assert Z_h >= Z_l
    assert theta_1 > theta_0
    
    V = integrate.quad(integrand, theta_0, theta_1, args=(R_l, Z_l, R_h, Z_h))
    
    return V[0]

# set up example
R_earth = 6371000
M_earth = 5.972E24

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

l1_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l1_test,
    Tw           = 3,
    R_e          = 1.3*R_earth,
    R_p          = 1.1*R_earth
    )

l1_test_sp.spin()

r_array = l1_test_sp.A1_r_equator
z_array = l1_test_sp.A1_r_pole
rho_e = l1_test_sp.A1_rho_equator
rho_p = l1_test_sp.A1_rho_pole
Tw = l1_test_sp.Tw
T_rho_type_L1 = l1_test_sp.A1_T_rho_type[0]
T_rho_args_L1 = l1_test_sp.A1_T_rho_args[0]
mat_id_L1     = l1_test_sp.A1_mat_id_layer[0]
N_neig = 48
N = 100000

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
m_sph = particles.m

# =============================================================================
# # example that works fine
# R_shell = 2691421
# Z_shell = 1957661
# R_shell_low  = 2455195
# R_shell_high = 2928477
# Z_shell_low  = 1786932
# Z_shell_high = 2128634
# rho_shell = 6644
# =============================================================================

# example that doesn't works fine
i = 10
delta = 1
R_shell = np.unique(R)[-i]
Z_shell = np.unique(Z)[-i]
R_shell_low  = np.unique(R)[-i - delta]
Z_shell_low = np.unique(Z)[-i - delta]
R_shell_high = np.unique(R)[-i + delta]
Z_shell_high = np.unique(Z)[-i + delta]
rho_shell = np.unique(rho_layer)[i - 1]
N_picle_shell = 2000000
#N_picle_shell = (R == R_shell).sum()

R_shell_low = np.mean([R_shell, R_shell_low])
Z_shell_low = np.mean([Z_shell, Z_shell_low])

R_shell_high = np.mean([R_shell, R_shell_high])
Z_shell_high = np.mean([Z_shell, Z_shell_high])

shell_config = [[R_shell_low, Z_shell_low], [R_shell_high, Z_shell_high]]

V_shell = vol_spheroid(R_shell_high, Z_shell_high) - vol_spheroid(R_shell_low, Z_shell_low)
m_shell = V_shell*rho_shell
particles = seagen.GenShell(N_picle_shell, R_shell)

x = particles.A1_x
y = particles.A1_y
z = particles.A1_z*Z_shell/R_shell
r = np.sqrt(x**2 + y**2 + z**2)

m_picle = m_shell/N_picle_shell # to be changed?

theta = np.arccos(z/r)

theta_bins = np.linspace(0, np.pi, 200)

m_picle = np.ones(N_picle_shell)*m_picle
m_bins_picle = np.zeros_like(theta_bins)
m_bins_theory = np.zeros_like(theta_bins)


for i in range(theta_bins.shape[0] - 1):
    low = theta_bins[i]
    high = theta_bins[i + 1]
    
    # particle distribution mass
    mask = np.logical_and(theta>=low, theta<high)
    m_bins_picle[i] = m_picle[mask].sum()
    
    # analitical solution mass
    m_bins_theory[i] = rho_shell*V_theta(low, high, shell_config)


plt.figure()
plt.scatter(theta_bins[:-1], m_bins_theory[:-1], alpha = 0.5, label='theory', s=5)
plt.scatter(theta_bins[:-1], m_bins_picle[:-1], alpha = 0.5, label='picle placement', s=5)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$m(\theta)$")
plt.yscale('log')
plt.legend()
plt.show()

# compute n of theta spherical
x = particles.A1_x
y = particles.A1_y
z = particles.A1_z
r = np.sqrt(x**2 + y**2 + z**2)

theta = np.arccos(z/r)
theta_sorted = np.sort(theta)

assert len(theta) == len(theta_sorted)

n_theta_sph = np.arange(1, N_picle_shell + 1)

p = 0.1
N = n_theta_sph.shape[0]
random_mask = np.random.binomial(1, p, N) > 0

plt.figure()
plt.scatter(theta_sorted[random_mask], n_theta_sph[random_mask],
            alpha = 0.5, label='spherical', s=5)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$n(\theta)$")
plt.legend()
plt.show()

# compute n of theta elipsoid
n_theta_elip = np.zeros_like(theta_bins)
n_theta_elip[1:] = m_bins_theory[:- 1]/(m_picle[0])
for i in range(1, n_theta_elip.shape[0]):
    n_theta_elip[i] = n_theta_elip[i] + n_theta_elip[i - 1]

plt.figure()
plt.scatter(theta_bins, n_theta_elip, alpha = 0.5, label='eliptical - theory', s=5)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$n(\theta)$")
plt.legend()
plt.show()

# combine
plt.figure()
plt.scatter(theta_sorted[random_mask], n_theta_sph[random_mask]/N_picle_shell,
            alpha = 0.5, label='spherical', s=5)
plt.scatter(theta_bins, n_theta_elip/N_picle_shell, alpha = 0.5, label='eliptical - theory', s=5)
plt.xlabel(r"$\theta$")
plt.ylabel(r"cumulative $n(\theta) [\%]$")
plt.legend()
plt.show()
