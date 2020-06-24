""" 
WoMa examples
"""

import numpy as np
import matplotlib.pyplot as plt
import woma

R_earth = 6.371e6 # m
M_earth = 5.972e24 # kg


# ============================================================================ #
#                       Plotting functions                                     #
# ============================================================================ #


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


def plot_planet_profiles_alternate(planet, fig=None, ax=None):
    if fig == None:
        fig, ax = plt.subplots(2, 2, figsize=(7, 7))
    
    ax[0, 0].plot(planet.A1_r/R_earth, planet.A1_rho)
    ax[0, 0].set_xlabel(r"Radius $[R_\oplus]$")
    ax[0, 0].set_ylabel(r"Density [kg m$^{-3}$]")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlim(0, None)
            
    ax[1, 0].plot(planet.A1_r/R_earth, planet.A1_P)
    ax[1, 0].set_xlabel(r"Radius $[R_\oplus]$")
    ax[1, 0].set_ylabel(r"Pressure [Pa]")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlim(0, None)
    
    ax[0, 1].plot(planet.A1_r/R_earth, planet.A1_T*1e-3)
    ax[0, 1].set_xlabel(r"Radius $[R_\oplus]$")
    ax[0, 1].set_ylabel(r"Temperature [$10^3$ K]")
    ax[0, 1].set_xlim(0, None)
    ax[0, 1].set_ylim(0, None)
    
    ax[1, 1].plot(np.log(planet.A1_rho/5), np.log(planet.A1_T))
    ax[1, 1].set_xlabel(r"ln( $\rho / 5$ kg m$^{-3}$ )")
    ax[1, 1].set_ylabel(r"ln( Temperature [K] )")
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================ #
#                       Spherical profile examples                             #
# ============================================================================ #

import woma
def demo_gen_prof_L1_find_R_given_M():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_granite"],
        A1_T_rho_type   = ["power=0."],
        A1_R_layer      = [0.988 * R_earth],
        M               = 0.8*M_earth,
        P_s             = 0,
        T_s             = 300
        )

    planet.R_max = R_earth

    planet.gen_prof_L1_find_R_given_M()

    plot_planet_profiles(planet)


def demo_gen_prof_L1_find_M_given_R():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_granite"],
        A1_T_rho_type   = ["power=0."],
        A1_R_layer      = [R_earth],
        P_s             = 0,
        T_s             = 300,
        )

    planet.M_max = M_earth

    planet.gen_prof_L1_find_M_given_R()

    plot_planet_profiles(planet)


def demo_gen_prof_L2_find_R1_given_R_M():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = ["power=0.", "power=0."],
        A1_R_layer      = [None, R_earth],
        M               = M_earth,
        P_s             = 0,
        T_s             = 300,
        )

    planet.gen_prof_L2_find_R1_given_R_M()

    plot_planet_profiles_alternate(planet)


def demo_gen_prof_L2_find_R_given_M_R1():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = ["power=0.", "power=0."],
        A1_R_layer      = [0.40*R_earth, R_earth],
        M               = M_earth,
        P_s             = 0,
        T_s             = 300,
        )

    planet.R_max = 2*R_earth
    planet.gen_prof_L2_find_R_given_M_R1()

    plot_planet_profiles(planet)


def demo_gen_prof_L2_find_M_given_R1_R():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = ["power=0.", "power=0."],
        A1_T_rho_args   = [[None, 0.], [None, 0.]],
        A1_R_layer      = [0.40*R_earth, R_earth],
        P_s             = 0,
        T_s             = 300,
        )

    planet.M_max = 2*M_earth
    planet.gen_prof_L2_find_M_given_R1_R()

    plot_planet_profiles(planet)


def demo_gen_prof_L3_given_prof_L2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = ["power=0.", "power=0."],
        A1_R_layer      = [None, R_earth],
        M               = 0.887*M_earth,
        P_s             = 1e5,
        T_s             = 2000,
        num_attempt     = 10
        )

    planet.gen_prof_L2_find_R1_given_R_M()

    mat_id_atm = "idg_N2"
    T_rho_type_atm = woma.gv.type_rho_pow
    T_rho_args_atm = [None, 0]

    planet.gen_prof_L3_given_prof_L2(
        mat_id_atm,
        T_rho_type_atm,
        T_rho_args_atm,
        rho_min=1e-6
        )

    plot_planet_profiles(planet)


def demo_gen_prof_L3_find_R1_R2_given_R_M_I():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = ["power=0.", "power=0.", "power=0."],
        A1_R_layer      = [None, None, R_earth],
        P_s             = 0,
        T_s             = 300,
        I_MR2           = 0.3*M_earth*R_earth**2,
        M               = M_earth,
        num_attempt     = 5,
        num_attempt_2   = 5
        )

    planet.gen_prof_L3_find_R1_R2_given_R_M_I()

    plot_planet_profiles(planet)


def demo_gen_prof_L3_find_R2_given_R_M_R1():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "SESAME_steam"],
        A1_T_rho_type   = ["power=0.", "power=0.", "power=0."],
        A1_R_layer      = [0.55*R_earth, None, R_earth],
        P_s             = 1e5,
        T_s             = 300,
        M               = M_earth
        )

    planet.gen_prof_L3_find_R2_given_R_M_R1()

    plot_planet_profiles(planet)


def demo_gen_prof_L3_find_R1_given_R_M_R2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = ["power=0.", "power=0.", "power=0."],
        A1_R_layer      = [None, 0.9*R_earth, R_earth],
        P_s             = 0,
        T_s             = 300,
        M               = M_earth
        )

    planet.gen_prof_L3_find_R1_given_R_M_R2()

    plot_planet_profiles(planet)


def demo_gen_prof_L3_find_M_given_R_R1_R2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = ["power=0.", "power=0.", "power=0."],
        A1_R_layer      = [0.5*R_earth, 0.9*R_earth, R_earth],
        P_s             = 0,
        T_s             = 300,
        M_max           = 2*M_earth
        )

    planet.gen_prof_L3_find_M_given_R_R1_R2()

    plot_planet_profiles(planet)


def demo_gen_prof_L3_find_R_given_M_R1_R2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = ["power=0.", "power=0.", "power=0."],
        A1_R_layer      = [0.5*R_earth, 0.9*R_earth, None],
        P_s             = 0,
        T_s             = 300,
        M               = M_earth,
        R_max           = 2*R_earth
        )

    planet.gen_prof_L3_find_R_given_M_R1_R2()

    plot_planet_profiles(planet)


def demo_gen_uranus_prof():
    
    planet = woma.Planet(
        name            = "Uranus",
        A1_mat_layer    = ["HM80_rock", "HM80_ice", "HM80_HHe"],
        A1_T_rho_type   = ["power=0.", "power=0.9", "adiabatic"],
        # M               = 14.536 * M_earth,
        M_max           = 14.7 * M_earth,
        A1_R_layer      = [1.0 * R_earth, 3.1 * R_earth, 3.98 * R_earth],
        P_s             = 1e5,
        T_s             = 60,
        )

    planet.gen_prof_L3_find_M_given_R_R1_R2()

    plot_planet_profiles_alternate(planet)


def demo_gen_earth_adiabatic():

    planet = woma.Planet(
        name            = "Earth",
        A1_mat_layer    = ["Til_iron", "ANEOS_forsterite"],
        A1_T_rho_type   = ["power=2.", "adiabatic"],
        A1_R_layer      = [None, R_earth],
        M               = M_earth,
        P_s             = 1e5,
        T_s             = 500,
        )
    
    planet.gen_prof_L2_find_R1_given_R_M()
    
    plot_planet_profiles_alternate(planet)
    

# ============================================================================ #


if __name__ == "__main__":
    # Run an example
    demo_gen_prof_L2_find_R1_given_R_M()
    
    plt.show()
