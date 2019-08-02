#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:45:53 2019

@author: sergio
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from numba import njit
import matplotlib.animation as animation
from tqdm import tqdm
from scipy.special import ellipe

# [[R1, Z1], ..., [RN, ZN]]
shell_config = [[1, 1], [2, 1.5]]

def plot_config(shell_config):
    
    plt.figure()
    plt.scatter([0],[0])
    theta = np.linspace(0, 2*np.pi, 1000)
    for shell in shell_config:
        R, Z = shell
        r2 = 1/(R*R) + 1/((np.tan(theta)**2)*(Z**2))
        r2 = 1/r2
        r = np.abs(np.sqrt(r2))
        z = np.abs(np.sqrt(Z*Z*(1 - r*r/R/R)))
        r[theta > np.pi] = -r[theta > np.pi]
        z[np.logical_and(theta > np.pi/2, theta < 3*np.pi/2)] = \
        -z[np.logical_and(theta > np.pi/2, theta < 3*np.pi/2)]
        plt.scatter(r, z)
        
    plt.show()
        
plot_config(shell_config)

def place_picles(n_picle_shell, m_shell, shell_config):
    
    assert(len(n_picle_shell) == len(m_shell))
    assert(len(n_picle_shell) == len(shell_config))
    
    r_list = []
    z_list = []
    R_list = []
    Z_list = []
    m_list = []
    
    N = np.sum(n_picle_shell)
    
    for n_shell, shell in enumerate(shell_config):
        R_i, Z_i = shell
        R = np.ones(n_picle_shell[n_shell])*R_i
        Z = np.ones(n_picle_shell[n_shell])*Z_i
        m = m_shell[n_shell]/n_picle_shell[n_shell]
        m = m*np.ones(n_picle_shell[n_shell])
        
        theta = np.linspace(0, 2*np.pi, n_picle_shell[n_shell], endpoint=False)
        epsilon = np.random.uniform(0, 2*np.pi)
        theta = (theta + epsilon) % (2*np.pi)
        r2 = 1/(R_i*R_i) + 1/((np.tan(theta)**2)*(Z_i**2))
        r2 = 1/r2
        r = np.abs(np.sqrt(r2))
        z = np.abs(np.sqrt(Z_i*Z_i*(1 - r*r/R_i/R_i)))
        z[np.isnan(z)] = 0
        r[theta > np.pi] = -r[theta > np.pi]
        z[np.logical_and(theta > np.pi/2, theta < 3*np.pi/2)] = \
        -z[np.logical_and(theta > np.pi/2, theta < 3*np.pi/2)]
        
        r_list.append(r)
        z_list.append(z)
        R_list.append(R)
        Z_list.append(Z)
        m_list.append(m)

    r_list = np.concatenate(r_list)
    z_list = np.concatenate(z_list)
    R_list = np.concatenate(R_list)
    Z_list = np.concatenate(Z_list)
    m_list = np.concatenate(m_list)
    
    assert(len(r_list) == N)
    assert(len(z_list) == N)
    assert(len(R_list) == N)
    assert(len(Z_list) == N)
    assert(len(m_list) == N)
    
    return r_list, z_list, R_list, Z_list, m_list
    
n_picle_shell = [100, 100, 100]
m_shell = [10, 10, 10]
shell_config = [[1, 1], [2, 1.5], [3, 2]]

n_picle_shell = [10]
m_shell = [10]
shell_config = [[1, 1]]

r, z, R, Z, m = place_picles(n_picle_shell, m_shell, shell_config)
plt.figure()
plt.scatter(r, z)
plt.show()

# negative gravity
@njit
def compute_F(M, m, dist):
    assert(len(m) == len(dist))
    return M*m/(dist*dist)

@njit
def compute_dir(r0, z0, ri, zi):
    
    assert(ri.shape[0] == zi.shape[0])
    
    dist = np.sqrt((r0 - ri)**2 + (z0 - zi)**2)
    
    vec_r = (r0 - ri)/dist
    vec_z = (z0 - zi)/dist
    
    return vec_r, vec_z

@njit
def compute_Fr_Fz(m0, r0, z0, mi, ri, zi):
    
    assert(len(mi) == len(ri))
    assert(len(mi) == len(zi))
    
    dist = np.sqrt((r0 - ri)**2 + (z0 - zi)**2)
    
    F = compute_F(m0, mi, dist)
    
    r_dir, z_dir = compute_dir(r0, z0, ri, zi)
    
    return np.sum(F*r_dir), np.sum(F*z_dir)

def compute_Fr_Fz_for_all(m, r, z, N_neig=5):

    r_reshaped  = r.reshape((-1,1))
    z_reshaped  = z.reshape((-1,1))
    
    X = np.hstack((r_reshaped, z_reshaped))
    
    del r_reshaped, z_reshaped
    
    nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
    nbrs.fit(X)
    
    distances, indices = nbrs.kneighbors(X)
    
    Fr = np.zeros_like(m)
    Fz = np.zeros_like(m)
    
    for i in range(m.shape[0]):
        Fr_i, Fz_i = compute_Fr_Fz(m[i], r[i], z[i],
                                   m[indices[i]][1:], r[indices[i]][1:], z[indices[i]][1:])
        Fr[i] = Fr_i
        Fz[i] = Fz_i
        
    return Fr, Fz

@njit
def update_r_z_vr_vz(m, r, z, vr, vz, Fr, Fz, dt=0.1):
    
    N_picle = len(r)
    assert(len(z) == N_picle)
    assert(len(vr) == N_picle)
    assert(len(vz) == N_picle)
    assert(len(Fr) == N_picle)
    assert(len(Fz) == N_picle)
    
    vr_new = vr + Fr*dt/m
    vz_new = vz + Fz*dt/m
    
    r_new = r + vr*dt
    z_new = z + vz*dt
    
    return r_new, z_new, vr_new, vz_new

@njit
def update_r_z_vr_vz_inshell(m, r, z, vr, vz, Fr, Fz, R, Z, dt=0.1):
    
    N_picle = len(r)
    assert(len(z) == N_picle)
    assert(len(vr) == N_picle)
    assert(len(vz) == N_picle)
    assert(len(Fr) == N_picle)
    assert(len(Fz) == N_picle)
    assert(len(R) == N_picle)
    assert(len(Z) == N_picle)
    
    r_new, z_new, vr_new, vz_new = update_r_z_vr_vz(m, r, z, vr, vz, Fr, Fz, dt)
    
    r_new = 1/R/R + (z_new/r_new)**2/Z/Z
    r_new = 1/r_new
    r_new = np.sqrt(r_new)
    
    z_new = np.sqrt(Z*Z*(1 - r_new*r_new/R/R))
    z_new[np.isnan(z_new)] = 0.
    
    r_new = np.abs(r_new)*np.sign(r)
    z_new = np.abs(z_new)*np.sign(z)
    
    return r_new, z_new, vr_new, vz_new

@njit
def upper_limit_Fr_Fz(Fr, Fz, limit=1):
    
    Fr[np.abs(Fr) > limit] = limit*np.sign(Fr[np.abs(Fr) > limit])
    Fz[np.abs(Fz) > limit] = limit*np.sign(Fz[np.abs(Fz) > limit])
    
    return Fr, Fz

def perimeter_elipsoid(R, Z):
    #assert(R >= Z)
    e = np.sqrt(1 - Z*Z/R/R)
    return 4*R*ellipe(e*e)
    
# initial set-up
n_picle_shell = [60]
m_shell = [10]
shell_config = [[1, 0.4]]

n_picle_shell = [40, 60, 60]
m_shell = 10*np.array([1, 1, 1])
shell_config = [[1, 1], [2, 1.5], [3, 1.8]]

n_picle_shell = [30, 50, 60, 60, 80]
m_shell = 10*np.array([1, 1, 1, 1, 1])
shell_config = [[1, 1], [1.5, 1.2], [2, 1.5], [2.5, 1.8], [3, 2]]
shell_config = [[1.5, 1], [1.5, 1], [1.5, 1], [1.5, 1], [1.5, 1]]

shell_config = np.array(shell_config)
shell_config[:, 0] *= np.linspace(1, 3, 5)
shell_config[:, 1] *= np.linspace(1, 3, 5)

r, z, R, Z, m = place_picles(n_picle_shell, m_shell, shell_config)

vr = np.zeros_like(r)
vz = np.zeros_like(r)

plt.figure()
plt.scatter(r, z)
plt.xlim((-1.1*max(r), 1.1*max(r)))
plt.ylim((-1.1*max(r), 1.1*max(r)))
plt.show()

# animation
fig = plt.figure()
ims = []
for i in tqdm(range(4000)):
    Fr, Fz = compute_Fr_Fz_for_all(m, r, z, N_neig=20)
    Fr, Fz = upper_limit_Fr_Fz(Fr, Fz, limit=1)
    #r, z, vr, vz = update_r_z_vr_vz(m, r, z, vr, vz, Fr, Fz, dt=0.1)
    r, z, vr, vz = update_r_z_vr_vz_inshell(m, r, z, vr, vz, Fr, Fz, R, Z, dt=0.01)
    # stop system every x frames to converg faster
    if i % 5 == 0:
        vr = np.zeros_like(vr)
        vz = np.zeros_like(vz)
    im = plt.scatter(r, z, animated=True, color = 'black')
    plt.xlim((-1.1*max(r), 1.1*max(r)))
    plt.ylim((-1.1*max(r), 1.1*max(r)))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True,
                                repeat_delay=1000)
#ani.save('antigravity_toy.gif')
plt.show()

# with a real config
import woma
from scipy.interpolate import interp1d
import seagen
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

N = 100000
r_array = l1_test_sp.A1_r_equator
z_array = l1_test_sp.A1_r_pole
rho_e = l1_test_sp.A1_rho_equator
rho_p = l1_test_sp.A1_rho_pole

rho_e_model = interp1d(r_array, rho_e)
rho_p_model_inv = interp1d(rho_p, z_array)

Re = np.max(r_array[rho_e > 0])

radii = np.arange(0, Re, Re/1000000)
densities = rho_e_model(radii)

particles = seagen.GenSphere(N, radii[1:], densities[1:], verb=0)

particles_r = np.sqrt(particles.x**2 + particles.y**2 + particles.z**2)
rho = rho_e_model(particles_r)

R_picle = particles.A1_r.copy()
rho_picle = rho_e_model(R_picle)
Z_picle = rho_p_model_inv(rho_picle)

R_shell = np.unique(R_picle)
Z_shell = np.unique(Z_picle)
rho_shell = np.zeros_like(R_shell)

for i in range(rho_shell.shape[0]):
    rho_shell[i] = np.mean(rho_picle[R_picle == R_shell[i]])
    

shell_config = np.hstack((R_shell.reshape(-1,1), Z_shell.reshape(-1,1)))
L = perimeter_elipsoid(R_shell, Z_shell)
m_shell = L*rho_shell

def compute_n_picle_shell(n_picle_shell_0, m_shell):
    n_picle_shell = [n_picle_shell_0]
    m_picle = m_shell[0]/n_picle_shell[0]
    
    for i in range(1, len(m_shell)):
        n = m_shell[i]/m_picle
        n_picle_shell.append(int(n))
        
    n_picle_shell =  np.array(n_picle_shell)
    n_picle_shell = n_picle_shell.astype('int')
    return n_picle_shell

n_picle_shell = compute_n_picle_shell(3, m_shell)

r, z, R, Z, m = place_picles(n_picle_shell, m_shell, shell_config)

vr = np.zeros_like(r)
vz = np.zeros_like(r)

plt.figure()
plt.scatter(r, z, s=5)
plt.xlim((-1.1*max(r), 1.1*max(r)))
plt.ylim((-1.1*max(r), 1.1*max(r)))
plt.show()

# animation
fig = plt.figure()
ims = []
for i in tqdm(range(1000)):
    
    Fr, Fz = compute_Fr_Fz_for_all(m, r, z, N_neig=20)
    Fr = 50*Fr
    Fz = 50*Fz
    #Fr, Fz = upper_limit_Fr_Fz(Fr, Fz, limit=1)
    #r, z, vr, vz = update_r_z_vr_vz(m, r, z, vr, vz, Fr, Fz, dt=0.1)
    r, z, vr, vz = update_r_z_vr_vz_inshell(m, r, z, vr, vz, Fr, Fz, R, Z, dt=5)
    # stop system every x frames to converg faster
    if i % 10 == 0:
        vr = np.zeros_like(vr)
        vz = np.zeros_like(vz)
    im = plt.scatter(r, z, animated=True, color = 'black', s=5)
    plt.xlim((-1.1*max(r), 1.1*max(r)))
    plt.ylim((-1.1*max(r), 1.1*max(r)))
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True,
                                repeat_delay=1000)
#ani.save('antigravity_toy.gif')
plt.show()