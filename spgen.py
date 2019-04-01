#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:18:01 2019

@author: sergio
"""
path = '/home/sergio/Documents/SpiPGen/'
import numpy as np
from numba import jit
import os
os.chdir(path)
import sys
sys.path.append(path)
import eos
import matplotlib.pyplot as plt

# Global constants
G = 6.67408E-11;
R_earth = 6371000;
M_earth = 5.972E24;

# 1 layer
def load_ucold_array(mat_id):
    
    if mat_id == 100:
        ucold_array = np.load('ucold_array_100.npy')
    elif mat_id == 101:
        ucold_array = np.load('ucold_array_101.npy')
    elif mat_id == 102:
        ucold_array = np.load('ucold_array_102.npy')
        
    return ucold_array
    
@jit(nopython=True)
def integrate_1layer(N, R, M, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                     rhos_min, rhos_max, ucold_array):
    
    r   = np.linspace(R, 0, int(N))
    m   = np.zeros(r.shape)
    P   = np.zeros(r.shape)
    T   = np.zeros(r.shape)
    rho = np.zeros(r.shape)
    u   = np.zeros(r.shape)
    mat = np.ones(r.shape)*mat_id
        
    rhos = eos._find_rho_fixed_T(Ps, mat_id, Ts, rhos_min, rhos_max, ucold_array)
    
    c = eos._spec_c(mat_id)
    
    if rhos == rhos_min or rhos == rhos_max or rhos == (rhos_min + rhos_max)/2.:
        print("Could not find rho surface in that interval")
        return r, m, P, T, rho, u, mat
    
    else:
        us = eos.ucold(rhos, mat_id, 10000) + c*Ts
        if T_rho_id == 1:
            T_rho_args[0] = Ts*rhos**(-T_rho_args[1])
    
    dr = r[0] - r[1]
    
    m[0]    = M
    P[0]    = Ps
    T[0]    = Ts
    rho[0]  = rhos
    u[0]    = us
    
    for i in range(1, r.shape[0]):
            
        m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
        P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
        rho[i] = eos._find_rho(P[i], mat_id, T_rho_id, T_rho_args,
                               rho[i - 1], 1.1*rho[i - 1], ucold_array)
        T[i]   = eos.T_rho(rho[i], T_rho_id, T_rho_args)
        u[i]   = eos._ucold_tab(rho[i], ucold_array) + c*T[i]
        
        if m[i] < 0:
            
            return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat

@jit(nopython=True)
def find_mass_1layer(N, R, M_max, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                     rhos_min, rhos_max, ucold_array):
    
    M_min = 0.
    #tol   = 1e-7
    
    r, m, P, T, rho, u, mat = integrate_1layer(N, R, M_max, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                                               rhos_min, rhos_max, ucold_array)
    
    if m[-1] > 0.:
        
        for i in range(30):
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_1layer(N, R, M_try, Ps, Ts, mat_id,
                                   T_rho_id, T_rho_args,
                                   rhos_min, rhos_max, ucold_array)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max
        
# test 1 layer
R             = R_earth
M_max         = M_earth
N             = 10000
Ts            = 300.
Ps            = 0.
T_rho_id      = 1.
T_rho_args    = [np.nan, 0.]
mat_id        = 101
rhos_min      = 2000.
rhos_max      = 3000.

ucold_array = load_ucold_array(mat_id)

M = find_mass_1layer(N, R, M_max, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                     rhos_min, rhos_max, ucold_array)

r, m, P, T, rho, u, mat = integrate_1layer(N, R, M, Ps, Ts, mat_id, T_rho_id, T_rho_args,
                                           rhos_min, rhos_max, ucold_array)

# plotting
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m)
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

# 2 layer
@jit(nopython=True)
def integrate_2layer(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle):
    
    r   = np.linspace(R, 0, int(N))
    m   = np.zeros(r.shape)
    P   = np.zeros(r.shape)
    T   = np.zeros(r.shape)
    rho = np.zeros(r.shape)
    u   = np.zeros(r.shape)
    mat = np.zeros(r.shape)
        
    rhos = eos._find_rho_fixed_T(Ps, mat_id_mantle, Ts,
                                 rhos_min, rhos_max, ucold_array_mantle)
    
    c_core   = eos._spec_c(mat_id_core)
    c_mantle = eos._spec_c(mat_id_mantle)
    
    if rhos == rhos_min or rhos == rhos_max or rhos == (rhos_min + rhos_max)/2.:
        print("Could not find rho surface in that interval")
        return r, m, P, T, rho, u, mat
    
    else:
        us = eos.ucold(rhos, mat_id_mantle, 10000) + c_mantle*Ts
        if T_rho_id_mantle == 1:
            T_rho_args_mantle[0] = Ts*rhos**(-T_rho_args_mantle[1])
    
    dr = r[0] - r[1]
    
    m[0]    = M
    P[0]    = Ps
    T[0]    = Ts
    rho[0]  = rhos
    u[0]    = us
    mat[0]  = mat_id_mantle 
    
    for i in range(1, r.shape[0]):
            
        # mantle
        if r[i] > b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_mantle)
            T[i]   = eos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary core mantle
        elif r[i] <= b_cm and r[i - 1] > b_cm:
            
            rho_transition = eos._find_rho_fixed_T(P[i - 1], mat_id_core, T[i - 1],
                                                   rho[i - 1], 5*rho[i - 1], ucold_array_core)
            
            if T_rho_id_core == 1:
                T_rho_args_core[0] = T[i - 1]*rho_transition**(-T_rho_args_core[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
        # core  
        elif r[i] <= b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array) + c_core*T[i]
            mat[i] = mat_id_core
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat

@jit(nopython=True)
def find_mass_2layer(N, R, M_max, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle):
    
    M_min = 0.
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M_max, Ps, Ts, b_cm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
    
    if m[-1] > 0.:
        
        for i in range(30):
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_2layer(N, R, M_try, Ps, Ts, b_cm,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

@jit(nopython=True)
def find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle):
    
    b_min = 0.
    b_max = R 
    
    r, m, P, T, rho, u, mat = \
        integrate_2layer(N, R, M_max, Ps, Ts, b_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
    
    if m[-1] == 0.:
        
        for i in range(40):
            
            b_try = (b_min + b_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_2layer(N, R, M, Ps, Ts, b_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle)
            
            if m[-1] > 0.:
                b_min = b_try
            else:
                b_max = b_try
                
    else:
        print("R is too low, ran out of mass in first iteration")
        return 0.
        
    return b_try

# test 2 layer (find mass given boundary)

N                 = 10000   
R                 = R_earth
M_max             = M_earth
Ts                = 300.
Ps                = 0.
mat_id_core       = 100
T_rho_id_core     = 1.
T_rho_args_core   = [np.nan, 0.]
mat_id_mantle     = 101
T_rho_id_mantle   = 1.
T_rho_args_mantle = [np.nan, 0.]
rhos_min          = 2000.
rhos_max          = 3000.
b_cm              = 0.4262*R_earth

ucold_array_core   = load_ucold_array(mat_id_core)
ucold_array_mantle = load_ucold_array(mat_id_mantle)

M = find_mass_2layer(N, R, M_max, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

r, m, P, T, rho, u, mat = \
    integrate_2layer(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

# plotting
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m)
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


# test 2 layer (find boundary given mass)

N                 = 10000   
R                 = R_earth
M                 = M_earth
Ts                = 300.
Ps                = 0.
mat_id_core       = 100
T_rho_id_core     = 1.
T_rho_args_core   = [np.nan, 0.]
mat_id_mantle     = 101
T_rho_id_mantle   = 1.
T_rho_args_mantle = [np.nan, 0.]
rhos_min          = 2000.
rhos_max          = 3000.

ucold_array_core   = load_ucold_array(mat_id_core)
ucold_array_mantle = load_ucold_array(mat_id_mantle)

b_cm = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

r, m, P, T, rho, u, mat = \
    integrate_2layer(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)

# plotting
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m)
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

# 3 layer
@jit(nopython=True)
def integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    
    r   = np.linspace(R, 0, int(N))
    m   = np.zeros(r.shape)
    P   = np.zeros(r.shape)
    T   = np.zeros(r.shape)
    rho = np.zeros(r.shape)
    u   = np.zeros(r.shape)
    mat = np.zeros(r.shape)
        
    rhos = eos._find_rho_fixed_T(Ps, mat_id_atm, Ts,
                                 rhos_min, rhos_max, ucold_array_atm)
    
    c_core   = eos._spec_c(mat_id_core)
    c_mantle = eos._spec_c(mat_id_mantle)
    c_atm    = eos._spec_c(mat_id_atm)
    
    if rhos == rhos_min or rhos == rhos_max or rhos == (rhos_min + rhos_max)/2.:
        print("Could not find rho surface in that interval")
        return r, m, P, T, rho, u, mat
    
    else:
        us = eos.ucold(rhos, mat_id_atm, 10000) + c_atm*Ts
        if T_rho_id_atm == 1:
            T_rho_args_atm[0] = Ts*rhos**(-T_rho_args_atm[1])
    
    dr = r[0] - r[1]
    
    m[0]    = M
    P[0]    = Ps
    T[0]    = Ts
    rho[0]  = rhos
    u[0]    = us
    mat[0]  = mat_id_atm
    
    for i in range(1, r.shape[0]):
            
        # atm
        if r[i] > b_ma:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_atm)
            T[i]   = eos.T_rho(rho[i], T_rho_id_atm, T_rho_args_atm)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_atm) + c_atm*T[i]
            mat[i] = mat_id_atm
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary mantle atm
        elif r[i] <= b_ma and r[i - 1] > b_ma:
            
            rho_transition = eos._find_rho_fixed_T(P[i - 1], mat_id_mantle, T[i - 1],
                                                   rho[i - 1], 5*rho[i - 1], ucold_array_mantle)
            
            if T_rho_id_mantle == 1:
                T_rho_args_mantle[0] = T[i - 1]*rho_transition**(-T_rho_args_mantle[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_mantle)
            T[i]   = eos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
        # mantle
        elif r[i] > b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_mantle)
            T[i]   = eos.T_rho(rho[i], T_rho_id_mantle, T_rho_args_mantle)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_mantle) + c_mantle*T[i]
            mat[i] = mat_id_mantle
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
        # boundary core mantle
        elif r[i] <= b_cm and r[i - 1] > b_cm:
            
            rho_transition = eos._find_rho_fixed_T(P[i - 1], mat_id_core, T[i - 1],
                                                   rho[i - 1], 5*rho[i - 1], ucold_array_core)
            
            if T_rho_id_core == 1:
                T_rho_args_core[0] = T[i - 1]*rho_transition**(-T_rho_args_core[1])
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho_transition*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho_transition/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho_transition, ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
            mat[i] = mat_id_core
            
        # core  
        elif r[i] <= b_cm:
            
            m[i]   = m[i - 1] - 4*np.pi*r[i - 1]*r[i - 1]*rho[i - 1]*dr
            P[i]   = P[i - 1] + G*m[i - 1]*rho[i - 1]/(r[i - 1]**2)*dr
            rho[i] = eos._find_rho(P[i], mat_id_core, T_rho_id_core, T_rho_args_core,
                                   rho[i - 1], 1.1*rho[i - 1], ucold_array_core)
            T[i]   = eos.T_rho(rho[i], T_rho_id_core, T_rho_args_core)
            u[i]   = eos._ucold_tab(rho[i], ucold_array) + c_core*T[i]
            mat[i] = mat_id_core
            
            if m[i] < 0: 
                return r, m, P, T, rho, u, mat
        
    return r, m, P, T, rho, u, mat


@jit(nopython=True)
def find_mass_3layer(N, R, M_max, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    
    M_min = 0.
    
    r, m, P, T, rho, u, mat = \
        integrate_3layer(N, R, M_max, Ps, Ts, b_cm, b_ma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    
    if m[-1] > 0.:
        
        for i in range(40):
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M_try, Ps, Ts, b_cm, b_ma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

@jit(nopython=True)
def _moi(r, rho):
    
    dr = np.abs(r[0] - r[1])
    r4 = np.power(r, 4)
    
    return 2*np.pi*(4/3)*np.sum(r4*rho)*dr

@jit(nopython=True)
def find_b_ma_3layer(N, R, M_max, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    
    b_ma_min = b_cm
    b_ma_max = R
    
    r, m, P, T, rho, u, mat = \
        integrate_3layer(N, R, M_max, Ps, Ts, b_cm, b_ma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    
    if m[-1] > 0.:
        
        for i in range(40):
            
            M_try = (M_min + M_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M_try, Ps, Ts, b_cm, b_ma,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if m[-1] > 0.:
                M_max = M_try
            else:
                M_min = M_try
                
    else:
        print("M_max is too low, ran out of mass in first iteration")
        return 0.
        
    return M_max

# test 3 layer (find mass given boundaries)

N                 = 10000   
R                 = R_earth
M_max             = M_earth
Ts                = 300.
Ps                = 0.
mat_id_core       = 100
T_rho_id_core     = 1.
T_rho_args_core   = [np.nan, 0.]
mat_id_mantle     = 101
T_rho_id_mantle   = 1.
T_rho_args_mantle = [np.nan, 0.]
mat_id_atm        = 102
T_rho_id_atm      = 1.
T_rho_args_atm    = [np.nan, 0.]
rhos_min          = 800.
rhos_max          = 1000.
b_cm              = 0.4*R_earth
b_ma              = 0.9*R_earth

ucold_array_core   = load_ucold_array(mat_id_core)
ucold_array_mantle = load_ucold_array(mat_id_mantle)
ucold_array_atm    = load_ucold_array(mat_id_atm)

M = find_mass_3layer(N, R, M_max, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

r, m, P, T, rho, u, mat = \
    integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm)

# plotting
fig, ax = plt.subplots(2,2, figsize=(12,12))

ax[0,0].plot(r/R_earth, rho)
ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")

ax[1,0].plot(r/R_earth, m)
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
