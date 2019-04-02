#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:18:01 2019

@author: Sergio Ruiz-Bonilla
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
        ucold_array = np.load('data/ucold_array_100.npy')
    elif mat_id == 101:
        ucold_array = np.load('data/ucold_array_101.npy')
    elif mat_id == 102:
        ucold_array = np.load('data/ucold_array_102.npy')
        
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
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
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
        integrate_2layer(N, R, M, Ps, Ts, b_max,
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
        
    return b_min

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
            u[i]   = eos._ucold_tab(rho[i], ucold_array_core) + c_core*T[i]
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
def moi(r, rho):
    
    dr = np.abs(r[0] - r[1])
    r4 = np.power(r, 4)
    
    return 2*np.pi*(4/3)*np.sum(r4*rho)*dr

@jit(nopython=True)
def find_b_ma_3layer(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    
    b_cm_min = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)
    
    b_cm_max = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_atm)
    
    if b_cm > b_cm_max:
        print("value of b_cm is too high,")
        print("maximum value available for this configuration is:")
        print(b_cm_max/R_earth, "R_earth")
        return -1
        
    elif b_cm < b_cm_min:
        print("value of b_cm is too low,")
        print("minimum value available for this configuration is:")
        print(b_cm_min/R_earth, "R_earth")
        return -1
        
    b_ma_min = R
    b_ma_max = b_cm
    
    r, m_min, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_min,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    r, m_max, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    
    if m_max[-1] > 0. and m_min[-1] == 0:
        
        for i in range(30):
            
            b_ma_try = (b_ma_min + b_ma_max)/2.
            
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if m[-1] > 0.:
                b_ma_max = b_ma_try
            else:
                b_ma_min = b_ma_try
                
    else:
        print("Something went wrong")
        return 0.
        
    return b_ma_max


@jit(nopython=True)
def _find_b_ma_3layer_nocheck(N, R, M, Ps, Ts, b_cm,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    
    b_ma_min = R
    b_ma_max = b_cm
    
    r, m_min, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_min,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    r, m_max, P, T, rho, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
    

        
    for i in range(30):
            
        b_ma_try = (b_ma_min + b_ma_max)/2.
            
        r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M, Ps, Ts, b_cm, b_ma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
        if m[-1] > 0.:
            b_ma_max = b_ma_try
        else:
            b_ma_min = b_ma_try
                

    return b_ma_max



@jit(nopython=True)
def find_boundaries_3layer(N, R, M, Ps, Ts, MoI,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle, ucold_array_atm):
    
    b_cm_max = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_mantle)
    
    b_cm_min = find_boundary_2layer(N, R, M, Ps, Ts,
                     mat_id_core, T_rho_id_core, T_rho_args_core,
                     mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                     rhos_min, rhos_max,
                     ucold_array_core, ucold_array_atm)
        
    b_ma_max = R
    b_ma_min = b_cm_min
    
    r_min, m, P, T, rho_min, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm_min, b_ma_min,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    r_max, m, P, T, rho_max, u, mat = \
        integrate_3layer(N, R, M, Ps, Ts, b_cm_max, b_ma_max,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
        
    moi_min = moi(r_min, rho_min)
    moi_max = moi(r_max, rho_max)
    
    if MoI > moi_min and  MoI < moi_max:
        
        for i in range(10):
            
            b_cm_try = (b_cm_min + b_cm_max)/2.
            
            b_ma_try = _find_b_ma_3layer_nocheck(N, R, M, Ps, Ts, b_cm_try,
                             mat_id_core, T_rho_id_core, T_rho_args_core,
                             mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                             mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                             rhos_min, rhos_max,
                             ucold_array_core, ucold_array_mantle, ucold_array_atm)
                    
            r, m, P, T, rho, u, mat = \
                  integrate_3layer(N, R, M, Ps, Ts, b_cm_try, b_ma_try,
                         mat_id_core, T_rho_id_core, T_rho_args_core,
                         mat_id_mantle, T_rho_id_mantle, T_rho_args_mantle,
                         mat_id_atm, T_rho_id_atm, T_rho_args_atm,
                         rhos_min, rhos_max,
                         ucold_array_core, ucold_array_mantle, ucold_array_atm)
            
            if moi(r,rho) < MoI:
                b_cm_min = b_cm_try
            else:
                b_cm_max = b_cm_try
                
    elif MoI > moi_max:
        print("Moment of interia is too high,")
        print("maximum value is:")
        print(moi_max/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        b_cm_try = 0.
        b_ma_try = 0.
    
    elif MoI < moi_min:
        print("Moment of interia is too low,")
        print("minimum value is:")
        print(moi_min/M_earth/R_earth/R_earth,"[M_earth R_earth^2]")
        b_cm_try = 0.
        b_ma_try = 0.
        
    else:
        print("Something went wrong")
        b_cm_try = 0.
        b_ma_try = 0.
        
    return b_cm_try, b_ma_try

def set_up():
    
    ucold_array_100 = eos._create_ucold_array(100)
    ucold_array_101 = eos._create_ucold_array(101)
    ucold_array_102 = eos._create_ucold_array(102)
    
    np.save("data/ucold_array_100", ucold_array_100)
    np.save("data/ucold_array_101", ucold_array_101)
    np.save("data/ucold_array_102", ucold_array_102)
    
