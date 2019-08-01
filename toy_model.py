#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 12:45:53 2019

@author: sergio
"""

import numpy as np
import matplotlib.pyplot as plt

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
        
        theta = np.linspace(0, 2*np.pi, n_picle_shell[n_shell])
        r2 = 1/(R_i*R_i) + 1/((np.tan(theta)**2)*(Z_i**2))
        r2 = 1/r2
        r = np.abs(np.sqrt(r2))
        z = np.abs(np.sqrt(Z_i*Z_i*(1 - r*r/R_i/R_i)))
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

r, z, R, Z, m = place_picles(n_picle_shell, m_shell, shell_config)
plt.figure()
plt.scatter(r, z)
plt.show()