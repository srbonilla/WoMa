#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:13:21 2019

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
import swift_io
import seagen
import spipgen_v2
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
from numba import jit, njit
import h5py

R_earth = 6371000;


@jit(nopython=False)
def _eq_density(x, y, z, radii, densities):
    rho = np.zeros(x.shape[0])
    r = np.sqrt(x*x + y*y + z*z)
    f = interp1d(radii, densities, kind = 'linear')
    for i in range(x.shape[0]):
        rho[i] = f(r[i])
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

def _picle_placement_1layer(rho_grid, r_array, z_array, Tw, N,
                            mat_id_core, T_rho_id_core, T_rho_args_core):

    Re = np.max(r_array[rho_grid[:,0] > 0])
    rho_model = RectBivariateSpline(r_array, z_array, rho_grid, kx = 1, ky = 1)

    radii = np.arange(0, Re, Re/1000000)
    densities = rho_model.ev(radii,0)

    particles = seagen.GenSphere(N, radii[1:], densities[1:])

    #rho_seagen = _eq_density(particles.x, particles.y, particles.z, radii, densities)
    
    zpz_grid = np.zeros((rho_grid.shape))

    # interpolate equatorial profile
    f = interp1d(radii, densities, kind = 'linear')
    
    #fill zpz grid
    for i in range(0, zpz_grid.shape[0]):
        for j in range(1, zpz_grid.shape[1]):
            rc = r_array[i]
            z = z_array[j]
            
            r = np.sqrt(rc*rc + z*z)
            
            if r > np.max(radii):
                rho_spherical = 0
            else:
                rho_spherical = f(r)
                
            rho_slice = rho_model.ev(np.ones((z_array.shape))*rc, z_array)
            k = (rho_slice >= rho_spherical).sum() - 1
            
            if k < z_array.shape[0] - 1:
                zpz_grid[i, j] = (z_array[k] - z_array[k + 1])*(rho_spherical - rho_slice[k])
                zpz_grid[i, j] = zpz_grid[i, j]/(rho_slice[k] - rho_slice[k + 1]) + z_array[k]
                zpz_grid[i, j] = zpz_grid[i, j]/z
                
    # Substitute first 2 rows
    zpz_grid[:, 0] = zpz_grid[:, 3]
    zpz_grid[:, 1] = zpz_grid[:, 3]
    zpz_grid[:, 2] = zpz_grid[:, 3]
    
    # fill 0's with boundary
    for i in range(zpz_grid.shape[0] - 1):
        for j in range(zpz_grid.shape[1] - 1):
            if zpz_grid[i, j + 1] == 0:
                zpz_grid[i, j + 1] = zpz_grid[i, j]
            if zpz_grid[i + 1, j] == 0:
                zpz_grid[i + 1, j] = zpz_grid[i, j]
                
    # Compute z' using the grid
    zP = np.zeros(particles.m.shape[0])
    zpz_model = RectBivariateSpline(r_array, z_array, zpz_grid, kx = 1, ky = 1)
        
    rc = np.sqrt(particles.x**2 + particles.y**2)
    z = particles.z
    zP = np.abs(zpz_model.ev(rc, np.abs(z))*z)*np.sign(z)
    """ 
    plt.scatter(particles.z/R_earth, zP/particles.z, s = 0.01, c = 'red')
    plt.xlabel(r"$z$ $[R_{earth}]$")
    plt.ylabel(r"$z'/z$")
    plt.show()
    """
    # Tweek masses
    mP = particles.m*zP/particles.z
    print("\nx, y, z, and m computed\n")
    
    # Compute velocities (T_w in hours)
    vx = np.zeros(mP.shape[0])
    vy = np.zeros(mP.shape[0])
    vz = np.zeros(mP.shape[0])
    
    hour_to_s = 3600
    wz = 2*np.pi/Tw/hour_to_s 
        
    vx = -particles.y*wz
    vy = particles.x*wz
    
    # model densities and internal energy
    rho = np.zeros((mP.shape[0]))
    u = np.zeros((mP.shape[0]))
    
    x = particles.x
    y = particles.y
    
    print("vx, vy, and vz computed\n")
    
    try:
        if mat_id_core == 100:
            ucold_array_core = np.load('ucold_array_100.npy')
        elif mat_id_core == 101:
            ucold_array_core = np.load('ucold_array_101.npy')
        elif mat_id_core == 102:
            ucold_array_core = np.load('ucold_array_102.npy')  
            
    except ImportError:
        return False
    
    #ucold_array_core = spipgen_v2._create_ucold_array(mat_id_core)
    c_core = spipgen_v2._spec_c(mat_id_core)
    
    rc = np.sqrt(x*x + y*y)
    rho = rho_model.ev(rc, np.abs(zP))
    for k in range(mP.shape[0]):
        u[k] = spipgen_v2._ucold_tab(rho[k], ucold_array_core)
        
    u = u + c_core*spipgen_v2.T_rho(rho, T_rho_id_core, T_rho_args_core)
    
    print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = 48    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    A1_h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_P = np.ones((mP.shape[0],)) # not implemented (not necessary)
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = np.ones((mP.shape[0],))*mat_id_core
    
    return x, y, zP, vx, vy, vz, mP, A1_h, rho, A1_P, u, A1_id, A1_mat_id

# example 1 layer
rho = np.load("profile_parallel_1l.npy")
r_array = np.load('r_array_1l.npy')
z_array = np.load('z_array_1l.npy')
rho_grid = rho[-1,:,:]
mat_id_core = 101
Tw = 4
T_rho_id_core = 1
T_rho_args_core = [300, 0]
N = 10**5

x, y, z, vx, vy, vz, m, h, rho, P, u, picle_id, mat_id =                      \
_picle_placement_1layer(rho_grid, r_array, z_array, Tw, N,
                        mat_id_core, T_rho_id_core, T_rho_args_core)

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '1layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, np.array([x, y, z]).T, np.array([vx, vy, vz]).T,
                             m, h, rho, P, u, picle_id, mat_id,
                             4*R_earth, swift_to_SI)  
    

        
###################################################################
    
def _picle_placement_2layer(rho_grid, r_array, z_array, Tw, N, rho_i,
                            mat_id_core, T_rho_id_core, T_rho_args_core,
                            mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle):

    Re = np.max(r_array[rho_grid[:,0] > 0])
    rho_model = RectBivariateSpline(r_array, z_array, rho_grid, kx = 1, ky = 1)

    radii = np.arange(0, Re, Re/1000000)
    densities = rho_model.ev(radii,0)

    particles = seagen.GenSphere(N, radii[1:], densities[1:])

    #rho_seagen = _eq_density(particles.x, particles.y, particles.z, radii, densities)
    
    zpz_grid = np.zeros((rho_grid.shape))

    # interpolate equatorial profile
    f = interp1d(radii, densities, kind = 'linear')
    
    #fill zpz grid
    for i in range(0, zpz_grid.shape[0]):
        for j in range(1, zpz_grid.shape[1]):
            rc = r_array[i]
            z = z_array[j]
            
            r = np.sqrt(rc*rc + z*z)
            
            if r > np.max(radii):
                rho_spherical = 0
            else:
                rho_spherical = f(r)
                
            rho_slice = rho_model.ev(np.ones((z_array.shape))*rc, z_array)
            k = (rho_slice >= rho_spherical).sum() - 1
            
            if k < z_array.shape[0] - 1:
                zpz_grid[i, j] = (z_array[k] - z_array[k + 1])*(rho_spherical - rho_slice[k])
                zpz_grid[i, j] = zpz_grid[i, j]/(rho_slice[k] - rho_slice[k + 1]) + z_array[k]
                zpz_grid[i, j] = zpz_grid[i, j]/z
                
    # Substitute first 2 rows
    zpz_grid[zpz_grid < 0] = 0
    zpz_grid[:, 0] = zpz_grid[:, 2]
    zpz_grid[:, 1] = zpz_grid[:, 2]
    
    # fill 0's with boundary
    for i in range(zpz_grid.shape[0] - 1):
        for j in range(zpz_grid.shape[1] - 1):
            if zpz_grid[i, j + 1] == 0:
                zpz_grid[i, j + 1] = zpz_grid[i, j]
            if zpz_grid[i + 1, j] == 0:
                zpz_grid[i + 1, j] = zpz_grid[i, j]
                
    # Compute z' using the grid
    zP = np.zeros(particles.m.shape[0])
    zpz_model = RectBivariateSpline(r_array, z_array, zpz_grid, kx = 1, ky = 1)
        
    rc = np.sqrt(particles.x**2 + particles.y**2)
    z = particles.z
    zP = np.abs(zpz_model.ev(rc, np.abs(z))*z)*np.sign(z)
    """ 
    plt.scatter(particles.z/R_earth, zP/particles.z, s = 0.01, c = 'red')
    plt.xlabel(r"$z$ $[R_{earth}]$")
    plt.ylabel(r"$z'/z$")
    plt.show()
    """
    # Tweek masses
    mP = particles.m*zP/particles.z
    print("\nx, y, z, and m computed\n")
    
    # Compute velocities (T_w in hours)
    vx = np.zeros(mP.shape[0])
    vy = np.zeros(mP.shape[0])
    vz = np.zeros(mP.shape[0])
    
    hour_to_s = 3600
    wz = 2*np.pi/Tw/hour_to_s 
        
    vx = -particles.y*wz
    vy = particles.x*wz
    
    # model densities and internal energy
    rho = np.zeros((mP.shape[0]))
    u = np.zeros((mP.shape[0]))
    
    x = particles.x
    y = particles.y
    
    print("vx, vy, and vz computed\n")
    
    try:
        
        if mat_id_core == 100:
            ucold_array_core = np.load('ucold_array_100.npy')
        elif mat_id_core == 101:
            ucold_array_core = np.load('ucold_array_101.npy')
        elif mat_id_core == 102:
            ucold_array_core = np.load('ucold_array_102.npy')
            
        if mat_id_mantle == 100:
            ucold_array_mantle = np.load('ucold_array_100.npy')
        elif mat_id_mantle == 101:
            ucold_array_mantle = np.load('ucold_array_101.npy')
        elif mat_id_mantle == 102:
            ucold_array_mantle = np.load('ucold_array_102.npy')
            
    except ImportError:
        return False
    
    #ucold_array_core = spipgen_v2._create_ucold_array(mat_id_core)
    c_core = spipgen_v2._spec_c(mat_id_core)
    c_mantle = spipgen_v2._spec_c(mat_id_mantle)
    
    rc = np.sqrt(x*x + y*y)
    rho = rho_model.ev(rc, np.abs(zP))
    for k in range(mP.shape[0]):
        if rho[k] > rho_i:
            u[k] = spipgen_v2._ucold_tab(rho[k], ucold_array_core)
            u[k] = u[k] + c_core*spipgen_v2.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
        else:
            u[k] = spipgen_v2._ucold_tab(rho[k], ucold_array_mantle)
            u[k] = u[k] + c_mantle*spipgen_v2.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
    
    print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = 48    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    A1_h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_P = np.ones((mP.shape[0],)) # not implemented (not necessary)
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_i)*mat_id_core + (rho <= rho_i)*mat_id_mantle
    
    return x, y, zP, vx, vy, vz, mP, A1_h, rho, A1_P, u, A1_id, A1_mat_id
    
    
# example 2 layer
rho = np.load("profile_parallel_2l.npy")
rho_grid = rho[-1]
r_array = np.load('r_array_2l.npy')
z_array = np.load('z_array_2l.npy')
Tw = 4
N = 10**5
rho_i = 10000
mat_id_core = 100
T_rho_id_core = 1
T_rho_args_core = [300, 0]
mat_id_mantle = 101
T_rho_id_mantle = 1
T_rho_args_mantle = [300, 0]

x, y, z, vx, vy, vz, m, h, rho, P, u, picle_id, mat_id =                      \
_picle_placement_2layer(rho_grid, r_array, z_array, Tw, N, rho_i,
                        mat_id_core, T_rho_id_core, T_rho_args_core,
                        mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle)

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '2layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, np.array([x, y, z]).T, np.array([vx, vy, vz]).T,
                             m, h, rho, P, u, picle_id, mat_id,
                             4*R_earth, swift_to_SI)   
    
#######################################
########################################
    

def _picle_placement_2layer(rho_grid, r_array, z_array, Tw, N, rho_i,
                            mat_id_core, T_rho_id_core, T_rho_args_core,
                            mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle):

    
    rho_e = rho_grid[:,0]
    rho_p = rho_grid[0,:]
    
    rho_e_model = interp1d(r_array, rho_e)
    #rho_p_model = interp1d(z_array, rho_p)
    
    rho_e_model_inv = interp1d(rho_e, r_array)
    rho_p_model_inv = interp1d(rho_p, z_array)

    Re = np.max(r_array[rho_e > 0])
    #rho_model = RectBivariateSpline(r_array, z_array, rho_grid, kx = 1, ky = 1)

    radii = np.arange(0, Re, Re/1000000)
    densities = rho_e_model(radii)

    particles = seagen.GenSphere(N, radii[1:], densities[1:])
    
    particles_r = np.sqrt(particles.x**2 + particles.y**2 + particles.z**2)
    particles_rc = np.sqrt(particles.x**2 + particles.y**2)
    particles_rho = rho_e_model(particles_r)
    
    R = rho_e_model_inv(particles_rho)
    Z = rho_p_model_inv(particles_rho)
    
    zP = np.sqrt(Z**2*(1 - (particles_rc/R)**2))*np.sign(particles.z)

    """ 
    plt.scatter(particles.z/R_earth, zP/particles.z, s = 0.01, c = 'red')
    plt.xlabel(r"$z$ $[R_{earth}]$")
    plt.ylabel(r"$z'/z$")
    plt.show()
    """
    # Tweek masses
    mP = particles.m*zP/particles.z
    print("\nx, y, z, and m computed\n")
    
    # Compute velocities (T_w in hours)
    vx = np.zeros(mP.shape[0])
    vy = np.zeros(mP.shape[0])
    vz = np.zeros(mP.shape[0])
    
    hour_to_s = 3600
    wz = 2*np.pi/Tw/hour_to_s 
        
    vx = -particles.y*wz
    vy = particles.x*wz
    
    # internal energy
    rho = particles_rho
    u = np.zeros((mP.shape[0]))
    
    x = particles.x
    y = particles.y
    
    print("vx, vy, and vz computed\n")
    
    try:
        
        if mat_id_core == 100:
            ucold_array_core = np.load('ucold_array_100.npy')
        elif mat_id_core == 101:
            ucold_array_core = np.load('ucold_array_101.npy')
        elif mat_id_core == 102:
            ucold_array_core = np.load('ucold_array_102.npy')
            
        if mat_id_mantle == 100:
            ucold_array_mantle = np.load('ucold_array_100.npy')
        elif mat_id_mantle == 101:
            ucold_array_mantle = np.load('ucold_array_101.npy')
        elif mat_id_mantle == 102:
            ucold_array_mantle = np.load('ucold_array_102.npy')
            
    except ImportError:
        return False
    
    #ucold_array_core = spipgen_v2._create_ucold_array(mat_id_core)
    c_core = spipgen_v2._spec_c(mat_id_core)
    c_mantle = spipgen_v2._spec_c(mat_id_mantle)
    
    for k in range(mP.shape[0]):
        if particles_rho[k] > rho_i:
            u[k] = spipgen_v2._ucold_tab(particles_rho[k], ucold_array_core)
            u[k] = u[k] + c_core*spipgen_v2.T_rho(particles_rho[k], T_rho_id_core, T_rho_args_core)
        else:
            u[k] = spipgen_v2._ucold_tab(rho[k], ucold_array_mantle)
            u[k] = u[k] + c_mantle*spipgen_v2.T_rho(particles_rho[k], T_rho_id_mantle, T_rho_args_mantle)
    
    print("Internal energy u computed\n")
    ## Smoothing lengths, crudely estimated from the densities
    num_ngb = 48    # Desired number of neighbours
    w_edge  = 2     # r/h at which the kernel goes to zero
    A1_h    = np.cbrt(num_ngb * mP / (4/3*np.pi * rho)) / w_edge
    
    A1_P = np.ones((mP.shape[0],)) # not implemented (not necessary)
    A1_id = np.arange(mP.shape[0])
    A1_mat_id = (rho > rho_i)*mat_id_core + (rho <= rho_i)*mat_id_mantle
    
    return x, y, zP, vx, vy, vz, mP, A1_h, rho, A1_P, u, A1_id, A1_mat_id
    
    
   
# example 2 layer new idea
rho = np.load("profile_parallel_2l.npy")
rho_grid = rho[-1]
r_array = np.load('r_array_2l.npy')
z_array = np.load('z_array_2l.npy')
Tw = 4
N = 10**5
rho_i = 10000
mat_id_core = 100
T_rho_id_core = 1
T_rho_args_core = [300, 0]
mat_id_mantle = 101
T_rho_id_mantle = 1
T_rho_args_mantle = [300, 0]

x, y, z, vx, vy, vz, m, h, rho, P, u, picle_id, mat_id =                      \
_picle_placement_2layer(rho_grid, r_array, z_array, Tw, N, rho_i,
                        mat_id_core, T_rho_id_core, T_rho_args_core,
                        mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle)

swift_to_SI = swift_io.Conversions(1, 1, 1)

filename = '2layer_10e5.hdf5'
with h5py.File(filename, 'w') as f:
    swift_io.save_picle_data(f, np.array([x, y, z]).T, np.array([vx, vy, vz]).T,
                             m, h, rho, P, u, picle_id, mat_id,
                             4*R_earth, swift_to_SI)   
     
    
    
    
    