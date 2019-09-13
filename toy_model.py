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
import seagen

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
        
# =============================================================================
#         r2 = 1/(R_i*R_i) + 1/((np.tan(theta)**2)*(Z_i**2))
#         r2 = 1/r2
#         r = np.abs(np.sqrt(r2))
#         z = np.abs(np.sqrt(Z_i*Z_i*(1 - r*r/R_i/R_i)))
#         z[np.isnan(z)] = 0
#         r[theta > np.pi] = -r[theta > np.pi]
#         z[np.logical_and(theta > np.pi/2, theta < 3*np.pi/2)] = \
#         -z[np.logical_and(theta > np.pi/2, theta < 3*np.pi/2)]
# =============================================================================
        
        r = R*np.sin(theta)
        z = Z*np.cos(theta)
        
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

#n_picle_shell = [10]
#m_shell = [10]
#shell_config = [[1, 1]]

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


    
# initial set-up
n_picle_shell = [100]
m_shell = [1]
shell_config = [[1, 0.6]]

n_picle_shell = [40, 60, 60]
m_shell = 10*np.array([1, 1, 1])
shell_config = [[1, 1], [2, 1.5], [3, 1.8]]

n_picle_shell = [60, 40, 100]
m_shell = 10*np.array([1, 1, 1])
shell_config = [[1, 1], [2, 1.9], [3, 2.8]]
#shell_config = [[1.5, 1], [1.5, 1], [1.5, 1]]

#shell_config = np.array(shell_config)
#shell_config[:, 0] *= np.linspace(1, 3, 5)
#shell_config[:, 1] *= np.linspace(1, 3, 5)

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
    Fr, Fz = upper_limit_Fr_Fz(Fr, Fz, limit=0.1)
    #r, z, vr, vz = update_r_z_vr_vz(m, r, z, vr, vz, Fr, Fz, dt=0.1)
    r, z, vr, vz = update_r_z_vr_vz_inshell(m, r, z, vr, vz, Fr, Fz, R, Z, dt=0.01)
    # stop system every x frames to converg faster
    if i % 3 == 0:
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

n_picle_shell = compute_n_picle_shell(2, m_shell)

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
for i in tqdm(range(3000)):
    
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

###############################################################################
# toy model 2
def perimeter_elipsoid(R, Z):
    #assert(R >= Z)
    e = np.sqrt(1 - Z*Z/R/R)
    return 4*R*ellipe(e*e)

def get_dist_between_2_shells(theta, shell_config):
    
    assert len(shell_config) == 2
    
    R1, Z1 = shell_config[0]
    R2, Z2 = shell_config[1]
    
    d1 = np.sin(theta)**2/R1/R1 + np.cos(theta)**2/Z1/Z1
    d1 = 1/d1
    d1 = np.sqrt(d1)
    
    d2 = np.sin(theta)**2/R2/R2 + np.cos(theta)**2/Z2/Z2
    d2 = 1/d2
    d2 = np.sqrt(d2)
    
    assert (d2 > d1).sum() == len(theta)
    
    return d2 - d1

def get_density_between_2_shells(theta, m, shell_config):
    
    assert len(shell_config) == 2
    assert len(m) == len(theta)
    
    d = get_dist_between_2_shells(theta, shell_config)
    density = m/d
    
    return density

def vol_spheroid(R, Z):
    return 4*np.pi*R*R*Z/3


    
theta_d = np.linspace(0, 2*np.pi, num=100)
shell_config_1 = [[1, 0.5], [1., 0.5]]
shell_config_1 = np.array(shell_config_1)*np.array([[0.5, 0.5], [1.5, 1.5]])
shell_config_1 = [[2455195, 1786932], [2928477, 2128634]]
#shell_config_2 = [[1, 1], [2, 1.9]]
d_1 = get_dist_between_2_shells(theta_d, shell_config_1)
#d_2 = get_dist_between_2_shells(theta, shell_config_2)

# =============================================================================
# plt.figure()
# plt.scatter(theta_d, d_1)
# #plt.scatter(theta, 1/d_2)
# #plt.scatter(theta, 0.25*(np.cos(2*theta) + 1)/2 + 1)
# plt.xlabel(r"$\theta$")
# plt.ylabel(r"$d$")
# plt.show()
# =============================================================================

# =============================================================================
# plt.figure()
# plt.scatter(theta, d_1)
# #plt.scatter(theta, d_2)
# plt.show()
# 
# =============================================================================

n_picle_shell = [200000]
#m_shell = [1]
#shell_config = [[1,0.5]]
R_test = 2691421
Z_test = 1957661
R_test_low  = 2455195
R_test_high = 2928477
Z_test_low  = 1786932
Z_test_high = 2128634
shell_config = [[2691421, 1957661]]
#rho_shell = m_shell[0]/perimeter_elipsoid(1, 0.5)
rho_shell = 6644
m_shell = [(vol_spheroid(R_test_high, Z_test_high) - vol_spheroid(R_test_low, Z_test_low))*rho_shell]

r, z, R, Z, m = place_picles(n_picle_shell, m_shell, shell_config)
#plt.figure()
#plt.scatter(r, z, s=1)
#plt.show()

theta_bins = np.linspace(0, 2*np.pi, 1000)
theta = (np.arctan2(r, z) + 2*np.pi) % (2*np.pi)

m_bins = np.zeros_like(theta_bins)
for i in range(theta_bins.shape[0] - 1):
    low = theta_bins[i]
    high = theta_bins[i + 1]
    
    mask = np.logical_and(theta>low, theta<high)
    m_bins[i] = m[mask].sum()
    
d_theta = theta_bins[1] - theta_bins[0]

plt.figure()
plt.scatter(theta_bins[:-1], m_bins[:-1], label='sum m/dtheta/rho0')
plt.scatter(theta_d, d_1*rho_shell*d_theta, label='dist')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$[m]$")
plt.legend()
plt.show()

# toy model 3 #######################################################
def V_theta(theta, d_theta, shell_config):
    
    R_l, Z_l = shell_config[0]
    R_h, Z_h = shell_config[1]
    
    assert R_h >= R_l
    assert Z_h >= Z_l
    
    rho_l = np.sin(theta)**2/R_l/R_l + np.cos(theta)**2/Z_l/Z_l
    rho_l = np.sqrt(1/rho_l)
    
    rho_h = np.sin(theta)**2/R_h/R_h + np.cos(theta)**2/Z_h/Z_h
    rho_h = np.sqrt(1/rho_h)
    
    V = 2*np.pi/3*(rho_h**3 - rho_l**3)*np.sin(theta)*d_theta
    
    return V

N_picle = 100000

# example that works fine
R_shell = 2691421
Z_shell = 1957661
R_shell_low  = 2455195
R_shell_high = 2928477
Z_shell_low  = 1786932
Z_shell_high = 2128634
rho_shell = 6644

# example that works fine
R_shell = 6948201
Z_shell = 4923112
R_shell_low  = 6674609
R_shell_high = 7227899
Z_shell_low  = 4743377
Z_shell_high = 5104052
rho_shell = 4049

R_shell_low = np.mean([R_shell, R_shell_low])
Z_shell_low = np.mean([Z_shell, Z_shell_low])

R_shell_high = np.mean([R_shell, R_shell_high])
Z_shell_high = np.mean([Z_shell, Z_shell_high])

shell_config = [[R_shell_low, Z_shell_low], [R_shell_high, Z_shell_high]]

m_shell = vol_spheroid(R_shell_high, Z_shell_high) - vol_spheroid(R_shell_low, Z_shell_low)
m_shell = m_shell*rho_shell
particles = seagen.GenShell(N_picle, R_shell)

zP = particles.A1_z*Z_shell/R_shell

m_picle = m_shell/N_picle

r_cyl = np.sqrt(particles.x**2 + particles.y**2)
theta = -np.arctan2(zP, r_cyl) + np.pi/2
theta = (theta + 2*np.pi) % (2*np.pi)

d_shells = get_dist_between_2_shells(theta, shell_config)

theta_bins = np.linspace(0, np.pi, 1000)
d_theta = theta_bins[1] - theta_bins[0]

m = np.ones(N_picle)*m_picle
m_bins = np.zeros_like(theta_bins)
for i in range(theta_bins.shape[0] - 1):
    low = theta_bins[i]
    high = theta_bins[i + 1]
    
    mask = np.logical_and(theta>=low, theta<high)
    m_bins[i] = m[mask].sum()




plt.figure()
plt.scatter(theta, rho_shell*V_theta(theta, d_theta, shell_config), label='theory', s=1)
plt.scatter(theta_bins[:-1] + d_theta/2, m_bins[:-1], alpha = 0.5, label='picle placement', s=1)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$m(\theta)$")
plt.yscale('log')
plt.legend()
plt.show()




########################################
import woma

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

l2_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l2_test,
    Tw           = 2.6,
    R_e          = 1.45*R_earth,
    R_p          = 1.1*R_earth
    )

l2_test_sp = woma.SpinPlanet(
    name         = 'sp_planet',
    planet       = l2_test,
    Tw           = 2*2.6,
    R_e          = 1.1*R_earth,
    R_p          = 1.01*R_earth
    )

l2_test_sp.spin()

from scipy.interpolate import interp1d
R_earth = 6371000

planet = l2_test_sp

model_e = interp1d(planet.A1_rho_equator, planet.A1_r_equator)
model_p = interp1d(planet.A1_rho_pole, planet.A1_r_pole)

rho_min_1 = np.min(planet.A1_rho_equator[planet.A1_rho_equator > 0])
rho_min_2 = np.min(planet.A1_rho_pole[planet.A1_rho_pole > 0])
rho_min = np.max([rho_min_1, rho_min_2]) + 1
rho_max = np.max([planet.A1_rho_equator, planet.A1_rho_pole]) - 1
rho = np.linspace(rho_min, rho_max, 1000)
R = model_e(rho)
Z = model_p(rho)

plt.figure()
plt.scatter(R/R_earth, Z/R)
plt.xlabel(r"$R$ $[R_{earth}]$")
plt.ylabel(r"$Z/R$")
plt.show()

######################################################
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
zP = particles.z*f

#mP = particles.m*f
mP = particles.m*np.mean(f)

# Compute velocities (T_w in hours)
vx = np.zeros(mP.shape[0])
vy = np.zeros(mP.shape[0])
vz = np.zeros(mP.shape[0])

hour_to_s = 3600
wz = 2*np.pi/Tw/hour_to_s

vx = -particles.y*wz
vy = particles.x*wz

# internal energy
u = np.zeros((mP.shape[0]))

x = particles.x
y = particles.y

P = np.zeros((mP.shape[0],))

for k in range(mP.shape[0]):
    T = T_rho(rho[k], T_rho_type_L1, T_rho_args_L1, mat_id_L1)
    u[k] = eos.u_rho_T(rho[k], T, mat_id_L1)
    P[k] = eos.P_u_rho(u[k], rho[k], mat_id_L1)

#print("Internal energy u computed\n")
# Smoothing lengths, crudely estimated from the densities
w_edge  = 2     # r/h at which the kernel goes to zero
h       = np.cbrt(N_neig * mP / (4/3*np.pi * rho)) / w_edge

A1_id     = np.arange(mP.shape[0])
A1_mat_id = np.ones((mP.shape[0],))*mat_id_L1

unique_R = np.unique(R)

x_reshaped  = x.reshape((-1,1))
y_reshaped  = y.reshape((-1,1))
zP_reshaped = zP.reshape((-1,1))

X = np.hstack((x_reshaped, y_reshaped, zP_reshaped))

del x_reshaped, y_reshaped, zP_reshaped

nbrs = NearestNeighbors(n_neighbors=N_neig, algorithm='kd_tree', metric='euclidean', leaf_size=15)
nbrs.fit(X)

N_mem = int(1e6)

print("Finding neighbors of all particles...")
distances, indices = nbrs.kneighbors(X)

for _ in tqdm(range(1), desc="Tweaking mass of every particle"):

    M = us._generate_M(indices, mP)

    rho_sph = us.SPH_density(M, distances, h)

    #diff = (rho_sph - rho)/rho
    #mP_next = (1 - diff)*mP
    #mP_next[R == unique_R[-1]] = mP[R == unique_R[-1]] # do not change mass of boundary layers

    #mP = mP_next

print("\nDone!")

diff = (rho_sph - rho)/rho
rc = np.sqrt(x**2 + y**2)

plt.figure(figsize=(12, 12))
plt.scatter(rc/R_earth, np.abs(zP)/R_earth, s = 40, alpha = 0.5, c = diff, 
            marker='.', edgecolor='none', cmap = 'coolwarm')
plt.xlabel(r"$r_c$ $[R_{earth}]$")
plt.ylabel(r"$z$ $[R_{earth}]$")
cbar = plt.colorbar()
cbar.set_label(r"$(\rho_{\rm SPH} - \rho_{\rm model}) / \rho_{\rm model}$")
plt.clim(-0.1,0.1)
plt.axes().set_aspect('equal')

plt.show()