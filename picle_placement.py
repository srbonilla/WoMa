#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:37:46 2019

@author: sergio
"""

import woma
import numpy as np
import utils_spin as us
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import eos
from tqdm import tqdm
import seagen
from T_rho import T_rho
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from numba import njit

#auxiliary functions
@njit
def vol_spheroid(R, Z):
    return 4*np.pi*R*R*Z/3

@njit
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

@njit
def cart_to_spher(x, y, z):
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    
    return r, theta, phi

@njit
def spher_to_cart(r, theta, phi):
    
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    
    return x, y ,z


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

M = l1_test_sp.M
r_array = l1_test_sp.A1_r_equator
z_array = l1_test_sp.A1_r_pole
rho_e = l1_test_sp.A1_rho_equator
rho_p = l1_test_sp.A1_rho_pole
Tw = l1_test_sp.Tw
T_rho_type_L1 = l1_test_sp.A1_T_rho_type[0]
T_rho_args_L1 = l1_test_sp.A1_T_rho_args[0]
mat_id_L1     = l1_test_sp.A1_mat_id_layer[0]
N_neig = 48
N = 1000000

# function
rho_e_model = interp1d(r_array, rho_e)
rho_e_model_inv = interp1d(rho_e, r_array)
rho_p_model_inv = interp1d(rho_p, z_array)

rho_min = min(rho_e)
R_min = rho_e_model_inv(rho_min)
Z_min = rho_p_model_inv(rho_min)

Re = np.max(r_array[rho_e > 0])

radii = np.arange(0, Re, Re/1000000)
densities = rho_e_model(radii)

particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)

particles_r = np.sqrt(particles.x**2 + particles.y**2 + particles.z**2)
rho = rho_e_model(particles_r)

R = particles.A1_r.copy()
rho_shell = rho_e_model(R)
Z = rho_p_model_inv(rho_shell)

f = Z/R

R_shell = np.unique(R)
Z_shell = np.unique(Z)
rho_shell = np.unique(rho_shell)
rho_shell = -np.sort(-rho_shell)
N_shell_original = np.zeros_like(rho_shell, dtype='int')
M_shell = np.zeros_like(rho_shell)

for i in range(N_shell_original.shape[0]):
    N_shell_original[i] = int((R == R_shell[i]).sum())
    
alpha = M/particles.A1_m.sum()
m_picle = alpha*np.median(particles.A1_m)

for i in range(M_shell.shape[0] - 1):
    if i == 0:
        M_shell[i] = 4*m_picle
        
    else:
        R_l = R_shell[i - 1]
        Z_l = Z_shell[i - 1]
        R_h = R_shell[i + 1]
        Z_h = Z_shell[i + 1]
        R_0 = R_shell[i]
        Z_0 = Z_shell[i]
        
        R_l = (R_l + R_0)/2
        Z_l = (Z_l + Z_0)/2
        R_h = (R_h + R_0)/2
        Z_h = (Z_h + Z_0)/2
        
        M_shell[i] = rho_shell[i]*V_theta(0, np.pi, [[R_l, Z_l], [R_h, Z_h]])

#M_shell[-1] = rho_shell[-1]*V_theta(0, np.pi, [[R_h, Z_h], [(R_min + R_shell[-1])/2, (Z_min + Z_shell[-1])/2]])
#M_shell[-1] = rho_shell[-1]*V_theta(0, np.pi, [[R_h, Z_h], [R_shell[-1], Z_shell[-1]]])
M_shell[-1] = M - M_shell.sum()
N_shell = np.round(M_shell/m_picle).astype(int)

m_picle_shell = M_shell/N_shell

# n of theta for spherical shell
particles = seagen.GenShell(2000000, 1)

x = particles.A1_x
y = particles.A1_y
z = particles.A1_z
r = np.sqrt(x**2 + y**2 + z**2)

theta = np.arccos(z/r)
theta_sph = np.sort(theta)

assert len(theta) == len(theta_sph)

N_theta_sph_model = 2000000
n_theta_sph = np.arange(1, N_theta_sph_model + 1)/N_theta_sph_model

# Generate shells and make adjustments
A1_x = []
A1_y = []
A1_z = []
A1_rho = []
A1_m = []
A1_R = []
A1_Z = []

theta_bins = np.linspace(0, np.pi, 10000)
delta_theta = (theta_bins[1] - theta_bins[0])/2
theta_elip = theta_bins[:-1] + delta_theta

n_theta_elip = np.zeros_like(theta_bins)

for i in tqdm(range(N_shell.shape[0] - 1)):
    
    # first and last layers
    if i == 0:
        particles = seagen.GenShell(N_shell[i], R_shell[i])
        A1_x.append(particles.A1_x)
        A1_y.append(particles.A1_y)
        A1_z.append(particles.A1_z*Z_shell[i]/R_shell[i])
        A1_rho.append(rho_shell[i]*np.ones(N_shell[i]))
        A1_m.append(m_picle_shell[i]*np.ones(N_shell[i]))
        A1_R.append(R_shell[i]*np.ones(N_shell[i]))
        A1_Z.append(Z_shell[i]*np.ones(N_shell[i]))
        
# =============================================================================
#         particles = seagen.GenShell(N_shell[-1], R_shell[-1])
#         A1_x.append(particles.A1_x)
#         A1_y.append(particles.A1_y)
#         A1_z.append(particles.A1_z*Z_shell[-1]/R_shell[-1])
#         A1_rho.append(rho_shell[-1]*np.ones(N_shell[-1]))
#         A1_m.append(m_picle_shell[-1]*np.ones(N_shell[-1]))
#         A1_R.append(R_shell[-1]*np.ones(N_shell[-1]))
#         A1_Z.append(Z_shell[-1]*np.ones(N_shell[-1]))
# =============================================================================
        
    else:
        
        theta_elip = theta_bins[:-1] + delta_theta
        n_theta_elip = np.zeros_like(theta_bins)
        particles = seagen.GenShell(N_shell[i], R_shell[i])
            
        R_0 = R_shell[i]
        Z_0 = Z_shell[i]
        R_l = R_shell[i - 1]
        Z_l = Z_shell[i - 1]
        R_h = R_shell[i + 1]
        Z_h = Z_shell[i + 1]
            
        R_l = (R_l + R_0)/2
        Z_l = (Z_l + Z_0)/2
        R_h = (R_h + R_0)/2
        Z_h = (Z_h + Z_0)/2
        
        shell_config = [[R_l, Z_l], [R_h, Z_h]]
        
        for j in range(theta_bins.shape[0] - 1):
            low = theta_bins[j]
            high = theta_bins[j + 1]
            
            # analitical solution particle number
            n_theta_elip[j] = rho_shell[i]*V_theta(low, high, shell_config)/m_picle_shell[i]
            
        for j in range(1, n_theta_elip.shape[0]):
            n_theta_elip[j] = n_theta_elip[j] + n_theta_elip[j - 1]
            
        n_theta_elip = n_theta_elip[:-1]/N_shell[i]
            
        random_mask = np.random.binomial(1, 0.1, N_theta_sph_model) > 0
# =============================================================================
#         plt.figure()
#         plt.scatter(theta_sph[random_mask], n_theta_sph[random_mask],
#                     alpha = 0.5, label='spherical', s=5)
#         plt.scatter(theta_elip, n_theta_elip, alpha = 0.5, label='eliptical - theory', s=5)
#         plt.xlabel(r"$\theta$")
#         plt.ylabel(r"cumulative $n(\theta) [\%]$")
#         plt.legend()
#         plt.show()
# =============================================================================
        
        n_theta_sph_model = interp1d(theta_sph, n_theta_sph)
        theta_elip_n_model = interp1d(n_theta_elip, theta_elip)
        
        x = particles.A1_x
        y = particles.A1_y
        z = particles.A1_z
        
        r, theta, phi = cart_to_spher(x, y, z)
        
        theta = theta_elip_n_model(n_theta_sph_model(theta))
        
        x, y, z = spher_to_cart(r, theta, phi)
        alpha = np.sqrt(1/(x*x/R_0/R_0 + y*y/R_0/R_0 + z*z/Z_0/Z_0))
        x = alpha*x
        y = alpha*y
        z = alpha*z
        
# =============================================================================
#         plt.figure()
#         plt.scatter(np.sqrt(x**2 + y**2), z, s=1)
#         plt.scatter(np.sqrt(x1**2 + y1**2), z1, s=1)
#         plt.show()
# =============================================================================
        
        A1_x.append(x)
        A1_y.append(y)
        A1_z.append(z)

        A1_rho.append(rho_shell[i]*np.ones(N_shell[i]))
        A1_m.append(m_picle_shell[i]*np.ones(N_shell[i]))
        A1_R.append(R_shell[i]*np.ones(N_shell[i]))
        A1_Z.append(Z_shell[i]*np.ones(N_shell[i]))
        
# last shell
particles = seagen.GenShell(N_shell[-1], R_shell[-1])
R_0 = R_shell[-1]
Z_0 = Z_shell[-1]

x = particles.A1_x
y = particles.A1_y
z = particles.A1_z

r, theta, phi = cart_to_spher(x, y, z)

theta = theta_elip_n_model(n_theta_sph_model(theta))

x, y, z = spher_to_cart(r, theta, phi)
alpha = np.sqrt(1/(x*x/R_0/R_0 + y*y/R_0/R_0 + z*z/Z_0/Z_0))
x = alpha*x
y = alpha*y
z = alpha*z

# =============================================================================
#         plt.figure()
#         plt.scatter(np.sqrt(x**2 + y**2), z, s=1)
#         plt.scatter(np.sqrt(x1**2 + y1**2), z1, s=1)
#         plt.show()
# =============================================================================

A1_x.append(x)
A1_y.append(y)
A1_z.append(z)

A1_rho.append(rho_shell[i]*np.ones(N_shell[-1]))
A1_m.append(m_picle_shell[i]*np.ones(N_shell[-1]))
A1_R.append(R_shell[i]*np.ones(N_shell[-1]))
A1_Z.append(Z_shell[i]*np.ones(N_shell[-1]))
        
A1_x = np.concatenate(A1_x)
A1_y = np.concatenate(A1_y)
A1_z = np.concatenate(A1_z)
A1_rho = np.concatenate(A1_rho)
A1_m = np.concatenate(A1_m)
A1_R = np.concatenate(A1_R)
A1_Z = np.concatenate(A1_Z)
        
        
# Recompute final rho sph
x_reshaped = A1_x.reshape((-1,1))
y_reshaped = A1_y.reshape((-1,1))
z_reshaped = A1_z.reshape((-1,1))

X = np.hstack((x_reshaped, y_reshaped, z_reshaped))

del x_reshaped, y_reshaped, z_reshaped

nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
nbrs.fit(X)

distances, indices = nbrs.kneighbors(X)

w_edge = 2
h = np.max(distances, axis=1)/w_edge
M_sph = us._generate_M(indices, A1_m)
rho_sph_ps = us.SPH_density(M_sph, distances, h)

delta_rho_ps = (rho_sph_ps - A1_rho)/A1_rho
        
rc_ps = np.sqrt(A1_x**2 + A1_y**2)

plt.figure(figsize=(12, 12))
plt.scatter(rc_ps/R_earth, np.abs(A1_z)/R_earth, s = 40, alpha = 0.5, c = delta_rho_ps, 
            marker='.', edgecolor='none', cmap = 'coolwarm')
plt.xlabel(r"$r_c$ $[R_{earth}]$")
plt.ylabel(r"$z$ $[R_{earth}]$")
cbar = plt.colorbar()
cbar.set_label(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
plt.clim(-0.05, 0.05)
plt.axes().set_aspect('equal')
plt.show()
        

plt.figure()
plt.scatter(A1_z/R_earth, delta_rho_ps, s=1, alpha=0.5)
plt.show()   
        
plt.figure()
plt.hist(A1_m/np.mean(A1_m), bins = 100)
plt.show()      
        
        