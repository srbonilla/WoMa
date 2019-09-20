#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:49:29 2019

@author: sergio
"""

# pseudo-simulation
import woma
import numpy as np
import utils_spin as us
from scipy.interpolate import interp1d
import eos
from tqdm import tqdm
import seagen
from T_rho import T_rho
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from numba import njit

@njit
def compute_F(delta_rho_sph_0, delta_rho_sph_j, dist_0_j, kappa=1):
    assert(len(delta_rho_sph_j) == len(dist_0_j))
    if delta_rho_sph_0 <= 0.:
        F = np.zeros_like(dist_0_j)
        return F
    F = kappa*(delta_rho_sph_0 + delta_rho_sph_j)/dist_0_j/dist_0_j
    F[delta_rho_sph_0*delta_rho_sph_j <= 0] = 0.
    F[delta_rho_sph_j <= 0] = 0.
    
    return F

@njit
def compute_dir(x0, y0, z0, xj, yj, zj):
    
    assert(xj.shape[0] == yj.shape[0])
    assert(xj.shape[0] == zj.shape[0])
    
    dist = np.sqrt((x0 - xj)**2 + (y0 - yj)**2 + (z0 - zj)**2)
    
    vec_x = (x0 - xj)/dist
    vec_y = (y0 - yj)/dist
    vec_z = (z0 - zj)/dist
    
    return vec_x, vec_y, vec_z

@njit
def compute_Fx_Fy_Fz(delta_rho_sph_0, x0, y0, z0, delta_rho_sph_j, xj, yj, zj, kappa=1):
    
    assert(len(delta_rho_sph_j) == len(xj))
    assert(len(delta_rho_sph_j) == len(yj))
    assert(len(delta_rho_sph_j) == len(zj))
    
    dist = np.sqrt((x0 - xj)**2 + (y0 - yj)**2 + (z0 - zj)**2)
    
    F = compute_F(delta_rho_sph_0, delta_rho_sph_j, dist, kappa)
    
    x_dir, y_dir, z_dir = compute_dir(x0, y0, z0, xj, yj, zj)
    
    return np.sum(F*x_dir), np.sum(F*y_dir), np.sum(F*z_dir)

def compute_Fr_Fy_Fz_for_all(x, y, z, delta_rho, distances, indices, kappa=1):

    assert(len(x) == len(y))
    assert(len(x) == len(z))
    assert(len(x) == len(delta_rho))
    
    Fx = np.zeros_like(delta_rho)
    Fy = np.zeros_like(delta_rho)
    Fz = np.zeros_like(delta_rho)
    
    for i in range(delta_rho.shape[0]):
        j = indices[i]
        Fx_i, Fy_i, Fz_i = compute_Fx_Fy_Fz(delta_rho[i], x[i], y[i], z[i],
                                            delta_rho[j][1:], x[j][1:], y[j][1:], z[j][1:],
                                            kappa)
        Fx[i] = Fx_i
        Fy[i] = Fy_i
        Fz[i] = Fz_i
        
    return Fx, Fy, Fz

@njit
def update_x_y_z_vx_vy_vz(x, y, z, vx, vy, vz, Fx, Fy, Fz, dt=0.01):
    
    N_picle = len(x)
    assert(len(y) == N_picle)
    assert(len(z) == N_picle)
    assert(len(vx) == N_picle)
    assert(len(vy) == N_picle)
    assert(len(vz) == N_picle)
    assert(len(Fx) == N_picle)
    assert(len(Fy) == N_picle)
    assert(len(Fz) == N_picle)
    
# =============================================================================
#     vx_new = vx + Fx*dt
#     vy_new = vy + Fy*dt
#     vz_new = vz + Fz*dt
#     
#     x_new = x + vx*dt
#     y_new = y + vy*dt
#     z_new = z + vz*dt
# =============================================================================
    x_new = x + Fx*dt
    y_new = y + Fy*dt
    z_new = z + Fz*dt
    
    vx_new = np.zeros_like(x)
    vy_new = np.zeros_like(y)
    vz_new = np.zeros_like(z)
    
    return x_new, y_new, z_new, vx_new, vy_new, vz_new

@njit
def update_x_y_z_vx_vy_vz_inshell(x, y, z, vx, vy, vz, Fx, Fy, Fz, R, Z, dt=0.1):
    
    N_picle = len(x)
    assert(len(y) == N_picle)
    assert(len(z) == N_picle)
    assert(len(vx) == N_picle)
    assert(len(vy) == N_picle)
    assert(len(vz) == N_picle)
    assert(len(Fx) == N_picle)
    assert(len(Fy) == N_picle)
    assert(len(Fz) == N_picle)
    assert(len(R) == N_picle)
    assert(len(Z) == N_picle)
    
    x_new, y_new, z_new, vx_new, vy_new, vz_new = \
        update_x_y_z_vx_vy_vz(x, y, z, vx, vy, vz, Fx, Fy, Fz, dt)
    
    alpha = (x_new**2 + y_new**2)/R/R + z_new**2/Z/Z
    alpha = np.sqrt(1/alpha)
    x_new = alpha*x_new
    y_new = alpha*y_new
    z_new = alpha*z_new
    
    return x_new, y_new, z_new, vx_new, vy_new, vz_new

@njit
def cart_to_spher_vector_transf(vx, vy, vz, theta, phi):
    vr = vx*np.sin(theta)*np.cos(phi) + \
         vy*np.sin(theta)*np.sin(phi) + \
         vz*np.cos(theta)
    vtheta = vx*np.cos(theta)*np.cos(phi) + \
             vy*np.cos(theta)*np.sin(phi) - \
             vz*np.sin(theta)
    vphi = - vx*np.sin(phi) + vy*np.cos(phi)

    return vr, vtheta, vphi

@njit
def spher_to_cart_vector_transf(vr, vtheta, vphi, theta, phi):
    vx = vr*np.sin(theta)*np.cos(phi) + \
         vtheta*np.cos(theta)*np.cos(phi) - \
         vphi*np.sin(phi)
    vy = vr*np.sin(theta)*np.sin(phi) + \
         vtheta*np.cos(theta)*np.sin(phi) + \
         vphi*np.cos(phi)
    vz = vr*np.cos(theta) - vtheta*np.sin(theta)

    return vx, vy, vz

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

#set-up
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

# function
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
x = particles.x
y = particles.y
z = particles.z*f

#mP = particles.m*f
mP = particles.m*np.min(f)

# Compute velocities (T_w in hours)
vx = np.zeros(mP.shape[0])
vy = np.zeros(mP.shape[0])
vz = np.zeros(mP.shape[0])

hour_to_s = 3600
wz = 2*np.pi/Tw/hour_to_s

vx = -particles.y*wz
vy = particles.x*wz

# internal energy and pressure
u = np.zeros((mP.shape[0]))
P = np.zeros((mP.shape[0],))

for k in range(mP.shape[0]):
    T = T_rho(rho[k], T_rho_type_L1, T_rho_args_L1, mat_id_L1)
    u[k] = eos.u_rho_T(rho[k], T, mat_id_L1)
    P[k] = eos.P_u_rho(u[k], rho[k], mat_id_L1)

#print("Internal energy u computed\n")
# Smoothing lengths, crudely estimated from the densities
w_edge  = 2     # r/h at which the kernel goes to zero
#h2       = np.cbrt(N_neig * mP / (4/3*np.pi * rho)) / w_edge

A1_id     = np.arange(mP.shape[0])
A1_mat_id = np.ones((mP.shape[0],))*mat_id_L1

# initialize pseudo simulation positions and velocities
vx_ps = np.zeros_like(x)
vy_ps = np.zeros_like(x)
vz_ps = np.zeros_like(x)

x_ps = x.copy()
y_ps = y.copy()
z_ps = z.copy()

# simulation
for i in range(600):
    # Find neigbors and compute SPH density
    x_reshaped = x_ps.reshape((-1,1))
    y_reshaped = y_ps.reshape((-1,1))
    z_reshaped = z_ps.reshape((-1,1))
    
    X = np.hstack((x_reshaped, y_reshaped, z_reshaped))
    
    del x_reshaped, y_reshaped, z_reshaped
    
    nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=16)
    nbrs.fit(X)
    
    distances, indices = nbrs.kneighbors(X)
    
    h_ps = np.max(distances, axis=1)/w_edge
    M = us._generate_M(indices, mP)
    rho_sph = us.SPH_density(M, distances, h_ps)
    
    # Compute force
    delta_rho = (rho_sph - rho)/rho
    print("Iteration: " + str(i) + "")
    print("Sum abs delta rho sph: " + str(np.sum(np.abs(delta_rho[R != np.unique(R)[-1]]))))
    # set delta = 0 for boundaries
    delta_rho[R == np.max(R)] = 0.
    
    # Compute forces
    Fx, Fy, Fz = compute_Fr_Fy_Fz_for_all(x_ps, y_ps, z_ps, delta_rho, distances, indices, kappa=1e15)
    
    # Supress F in r, phi coordinates
    r_ps = np.sqrt(x_ps**2 + y_ps**2 + z_ps**2)
    theta = (np.arccos(z_ps/r_ps) + 2*np.pi) % (2*np.pi)
    phi = (np.arctan2(x_ps, y_ps) + 2*np.pi) % (2*np.pi)
    
    Fr, Ftheta, Fphi = cart_to_spher_vector_transf(Fx, Fy, Fz, theta, phi)
    
    Fr = 0.0001*Fr
    Fphi = 0.0001*Fphi
    
    Fx, Fy, Fz = spher_to_cart_vector_transf(Fr, Ftheta, Fphi, theta, phi)
    
    #Fz = 0.0001*Fz
    
    # let move only last shells
    Fx[R == np.unique(R)[-1]] = 0.
    Fy[R == np.unique(R)[-1]] = 0.
    Fz[R == np.unique(R)[-1]] = 0.
    
    #print(np.abs(delta_rho[R == np.unique(R)[-3]]).sum())

    # Update position
    x_ps, y_ps, z_ps, vx_ps, vy_ps, vz_ps = \
        update_x_y_z_vx_vy_vz_inshell(x_ps, y_ps, z_ps, vx_ps, vy_ps, vz_ps, Fx, Fy, Fz, R, Z, dt=1)
        
# =============================================================================
#     limit = 1000
#     vx_ps[vx_ps > limit] = limit
#     vy_ps[vy_ps > limit] = limit
#     vz_ps[vz_ps > limit] = limit
#     vx_ps[vx_ps < -limit] = -limit
#     vy_ps[vy_ps < -limit] = -limit
#     vz_ps[vz_ps < -limit] = -limit
#     
#     if i % 500 == 0:
#         vx_ps = np.zeros_like(x)
#         vy_ps = np.zeros_like(y)
#         vz_ps = np.zeros_like(z)
# =============================================================================

# Recompute final rho sph
x_reshaped = x_ps.reshape((-1,1))
y_reshaped = y_ps.reshape((-1,1))
z_reshaped = z_ps.reshape((-1,1))

X = np.hstack((x_reshaped, y_reshaped, z_reshaped))

del x_reshaped, y_reshaped, z_reshaped

nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
nbrs.fit(X)

distances, indices = nbrs.kneighbors(X)

h_ps = np.max(distances, axis=1)/w_edge
M = us._generate_M(indices, mP)
rho_sph_ps = us.SPH_density(M, distances, h_ps)

delta_rho_ps = (rho_sph_ps - rho)/rho

print("\nDone!")


rc_ps = np.sqrt(x_ps**2 + y_ps**2)

plt.figure(figsize=(12, 12))
plt.scatter(rc_ps/R_earth, np.abs(z_ps)/R_earth, s = 40, alpha = 0.5, c = delta_rho_ps, 
            marker='.', edgecolor='none', cmap = 'coolwarm')
plt.xlabel(r"$r_c$ $[R_{earth}]$")
plt.ylabel(r"$z$ $[R_{earth}]$")
cbar = plt.colorbar()
cbar.set_label(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
plt.clim(-0.1, 0.1)
plt.axes().set_aspect('equal')
plt.show()

plt.figure()
plt.scatter(z_ps/R_earth, delta_rho_ps, s=1, alpha=0.5)
plt.show()

plt.figure(figsize=(12, 12))
plt.scatter(rc_ps/R_earth, np.abs(z_ps)/R_earth, s = 40, alpha = 0.5, c = np.sqrt(Fx**2 + Fy**2), 
            marker='.', edgecolor='none', cmap = 'coolwarm')
plt.xlabel(r"$r_c$ $[R_{earth}]$")
plt.ylabel(r"$z$ $[R_{earth}]$")
cbar = plt.colorbar()
cbar.set_label(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
#plt.clim(-0.1,0.1)
plt.axes().set_aspect('equal')
plt.show()

plt.figure(figsize=(12, 12))
plt.scatter(rc_ps/R_earth, z_ps/R_earth, s = 40, alpha = 0.5, c = Fz, 
            marker='.', edgecolor='none', cmap = 'coolwarm')
plt.xlabel(r"$r_c$ $[R_{earth}]$")
plt.ylabel(r"$z$ $[R_{earth}]$")
cbar = plt.colorbar()
cbar.set_label(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
#plt.clim(-0.1,0.1)
plt.axes().set_aspect('equal')
plt.show()