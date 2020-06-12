""" 
WoMa examples
"""

import numpy as np
import matplotlib.pyplot as plt
import woma

R_earth = 6.371e6 # m
M_earth = 5.972e24 # kg


def plot_planet_profiles(planet):
    fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    
    ax[0, 0].plot(planet.A1_r/R_earth, planet.A1_rho)
    ax[0, 0].set_xlabel(r"Radius $[R_\oplus]$")
    ax[0, 0].set_ylabel(r"Density [kg m$^{-3}$]")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlim(0, None)
    
    ax[1, 0].plot(planet.A1_r/R_earth, planet.A1_m_enc/M_earth)
    ax[1, 0].set_xlabel(r"Radius $[R_\oplus]$")
    ax[1, 0].set_ylabel(r"Enclosed Mass $[M_\oplus]$")
    ax[1, 0].set_xlim(0, None)
    ax[1, 0].set_ylim(0, None)
        
    ax[0, 1].plot(planet.A1_r/R_earth, planet.A1_P)
    ax[0, 1].set_xlabel(r"Radius $[R_\oplus]$")
    ax[0, 1].set_ylabel(r"Pressure [Pa]")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set_xlim(0, None)
    
    ax[1, 1].plot(planet.A1_r/R_earth, planet.A1_T*1e-3)
    ax[1, 1].set_xlabel(r"Radius $[R_\oplus]$")
    ax[1, 1].set_ylabel(r"Temperature [$10^3$ K]")
    ax[1, 1].set_xlim(0, None)
    ax[1, 1].set_ylim(0, None)
    
    plt.tight_layout()


# ============================================================================ #
#                       Simple examples                                        #
# ============================================================================ #




# ============================================================================ #
#                       Full examples                                          #
# ============================================================================ #
# Each example here demonstrates:
#   1. Building a spherical planet profile 
#   2. Placing particles to model the spherical planet
#   3. Creating a spinning planet profile 
#   4. Placing particles to model the spinning planet 
# 
# These cover both a wide variety of planets and a wide range of input options,
# as described in the comments of each example function.

def simple_basalt_earth():
    # Create a basalt somewhat Earth-like planet.
    # Spherical profile:
    #   Choose the mass, let woma find the radius.
    #   Tillotson basalt, isothermal
    
    basalt_earth = woma.Planet(
        name            = "basalt_earth",
        A1_mat_layer    = ["Til_basalt"],
        A1_T_rho_type   = ["power=0."],
        M               = M_earth,
        T_s             = 500,
        P_s             = 1e5,
        )

    basalt_earth.gen_prof_L1_find_R_given_M()

    plot_planet_profiles(basalt_earth)
    


# ============================================================================ #


if __name__ == "__main__":
    # Run an example
    
    plt.show()
