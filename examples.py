""" 
WoMa examples
"""

import numpy as np
import matplotlib.pyplot as plt
import woma
import woma.spin_funcs.utils_spin as us

R_earth = 6.371e6 # m
M_earth = 5.972e24 # kg


def plot_planet_profile(planet):
    
    if isinstance(planet, woma.Planet):
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
        
    if isinstance(planet, woma.SpinPlanet):
        sp = planet
    
        fig, ax = plt.subplots(1,2, figsize=(12,6))
        ax[0].scatter(sp.A1_r/R_earth, sp.A1_rho, label = 'original', s = 0.5)
        ax[0].scatter(sp.A1_r_equator/R_earth, sp.A1_rho_equator, label = 'equatorial profile', s = 1)
        ax[0].scatter(sp.A1_r_pole/R_earth, sp.A1_rho_pole, label = 'polar profile', s = 1)
        ax[0].set_xlabel(r"$r$ [$R_{earth}$]")
        ax[0].set_ylabel(r"$\rho$ [$kg/m^3$]")
        ax[0].legend()
        
        
        r_array_coarse = np.linspace(0, np.max(sp.A1_r_equator), 100)
        z_array_coarse = np.linspace(0, np.max(sp.A1_r_pole), 100)
        rho_grid = np.zeros((r_array_coarse.shape[0], z_array_coarse.shape[0]))
        for i in range(rho_grid.shape[0]):
            radius = r_array_coarse[i]
            for j in range(rho_grid.shape[1]):
                z = z_array_coarse[j]
                rho_grid[i,j] = us.rho_rz(radius, z,
                                          sp.A1_r_equator, sp.A1_rho_equator,
                                          sp.A1_r_pole, sp.A1_rho_pole)
        
        X, Y = np.meshgrid(r_array_coarse/R_earth, z_array_coarse/R_earth)
        Z = rho_grid.T
        levels = np.arange(1000, 15000, 1000)
        ax[1].set_aspect('equal')
        CS = plt.contour(X, Y, Z, levels = levels)
        ax[1].clabel(CS, inline=1, fontsize=10)
        ax[1].set_xlabel(r"$r$ [$R_{earth}$]")
        ax[1].set_ylabel(r"$z$ [$R_{earth}$]")
        ax[1].set_title('Density (Kg/m^3)')
            
        plt.tight_layout()
        
    plt.show()


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
        R_max           = 2*R_earth,
    )
    
    # determine profile
    basalt_earth.gen_prof_L1_find_R_given_M()

    # plot results
    plot_planet_profile(basalt_earth)
    
    # particle placement
    particles = woma.ParticleSet(basalt_earth, 1e6)
    
    # save particles
    # particles.save("basalt_earth.hdf5")
    
    # spin planet
    spin_basalt_earth = woma.spin_planet_fix_M(
        planet     = basalt_earth,
        period     = 24,
    )
    
    # plot results
    plot_planet_profile(spin_basalt_earth)
    
    # particle placement
    particles = woma.ParticleSet(spin_basalt_earth, 1e6)
    
    # save particles
    # particles.save("spin_basalt_earth.hdf5")
    
def aneos_earth():
    # Create a Earth-like planet using ANEOS EoS.
    # Spherical profile:
    #   Choose the mass, let woma find the radius.
    #   Tillotson basalt, isothermal
    
    aneos_earth = woma.Planet(
        name            = "aneos_earth",
        A1_mat_layer    = ["ANEOS_Fe85Si15", "ANEOS_forsterite"],
        A1_T_rho_type   = ["adiabatic", "adiabatic"],
        M               = M_earth,
        A1_R_layer      = [None, R_earth],
        T_s             = 500,
        P_s             = 1e5,
        
    )
    
    # determine profile
    aneos_earth.gen_prof_L2_find_R1_given_R_M()
    
    # print boundary radius
    print("Boundary at", aneos_earth.A1_R_layer[0] / R_earth, "R_earth")

    # plot results
    plot_planet_profile(aneos_earth)
    
    # particle placement
    particles = woma.ParticleSet(aneos_earth, 1e7)
    
    # save particles
    # particles.save("basalt_earth.hdf5")
    
    # spin planet
    spin_aneos_earth = woma.spin_planet_fix_M(
        planet     = aneos_earth,
        period     = 5,
    )
    
    # plot results
    plot_planet_profile(spin_aneos_earth)
    
    # particle placement
    particles = woma.ParticleSet(spin_basalt_earth, 1e7)
    
    # save particles
    # particles.save("spin_basalt_earth.hdf5")
    
def uranus():
    # Create a Uranus-like planet.
    # Spherical profile:
    #   Choose the mass, let woma find the radius.
    #   Tillotson basalt, isothermal
    
    uranus = woma.Planet(
        name            = "Uranus",
        A1_mat_layer    = ["HM80_rock", "HM80_ice", "HM80_HHe"],
        A1_T_rho_type   = ["power=0.", "power=0.9", "adiabatic"],
        A1_M_layer      = np.array([2.02, 11.68, 0.84]) * M_earth,
        R_max           = 5 * R_earth,
        P_s             = 1e5,
        T_s             = 60,
        )

    # determine profile
    uranus.gen_prof_L3_find_R_R1_R2_given_M_M1_M2() # todo
    
    # plot results
    plot_planet_profile(uranus)
    
    # particle placement
    particles = woma.ParticleSet(uranus, 1e7)
    
    # save particles
    # particles.save("basalt_earth.hdf5")
    
    # spin planet
    spin_uranus = woma.spin_planet_fix_M(
        planet     = uranus,
        period     = 17.,
    )
    
    # plot results
    plot_planet_profile(spin_uranus)
    
    # particle placement
    particles = woma.ParticleSet(spin_uranus, 1e7)
    
    # save particles
    # particles.save("spin_basalt_earth.hdf5")

def super_earth():
    # Create a super Earth-like planet.
    # Spherical profile:
    #   Choose the mass, let woma find the radius.
    #   Tillotson basalt, isothermal
    
    super_earth = woma.Planet(
        name            = "super_earth",
        A1_mat_layer    = ["Til_iron", "SESAME_basalt", "HM80_HHe"],
        A1_T_rho_type   = ["power=0.", "adiabatic", "adiabatic"],
        A1_M_layer      = np.array([0.3, 0.7, 0.1]) * 2 * M_earth,
        R_max           = 8 * R_earth,
        P_s             = 1e5,
        T_s             = 1000,
    )
    
    # determine profile
    super_earth.gen_prof_L3_find_R_R1_R2_given_M_M1_M2()
    
    # print boundary radius
    print("Boundary at", super_earth.A1_R_layer[0] / R_earth, "R_earth")

    # plot results
    plot_planet_profile(super_earth)
    
    # particle placement
    particles = woma.ParticleSet(super_earth, 1e7)
    
    # save particles
    # particles.save("basalt_earth.hdf5")
    
    # spin planet
    spin_super_earth = woma.spin_planet_fix_M(
        planet     = super_earth,
        period     = 5,
    )
    
    # plot results
    plot_planet_profile(spin_super_earth)
    
    # particle placement
    particles = woma.ParticleSet(spin_super_earth, 1e7)
    
    # save particles
    # particles.save("spin_basalt_earth.hdf5")

# ============================================================================ #


if __name__ == "__main__":
    # Run an example
    
    plt.show()
