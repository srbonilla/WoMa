""" WoMa Examples
"""

# ============================================================================ #
# ===================== Libraries and constants ============================== #
# ============================================================================ #

import numpy as np
import matplotlib.pyplot as plt
import woma
import weos

R_earth = 6371000
M_earth = 5.972E24

# ============================================================================ #

def plot_planet_profiles(planet):
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    
    ax[0,0].plot(planet.A1_r/R_earth, planet.A1_rho)
    ax[0,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,0].set_ylabel(r"$\rho$ $[kg/m^3]$")
    ax[0,0].set_yscale("log")
    ax[0,0].set_xlim(0, None)
    
    ax[1,0].plot(planet.A1_r/R_earth, planet.A1_m_enc/M_earth)
    ax[1,0].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,0].set_ylabel(r"$M$ $[M_{earth}]$")
    ax[1,0].set_xlim(0, None)
    
    ax[0,1].plot(planet.A1_r/R_earth, planet.A1_P)
    ax[0,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[0,1].set_ylabel(r"$P$ $[Pa]$")
    ax[0,1].set_xlim(0, None)
    
    ax[1,1].plot(planet.A1_r/R_earth, planet.A1_T)
    ax[1,1].set_xlabel(r"$r$ $[R_{earth}]$")
    ax[1,1].set_ylabel(r"$T$ $[K]$")
    ax[1,1].set_xlim(0, None)
    
    plt.tight_layout()

# ============================================================================ #

def test_gen_prof_L1_fix_R_given_M():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_granite"],
        A1_T_rho_type   = [1],
        A1_T_rho_args   = [[None, 0.]],
        A1_R_layer      = [0.988 * R_earth],
        M               = 0.8*M_earth,
        P_s             = 0,
        T_s             = 300
        )

    planet.R_max = R_earth

    planet.gen_prof_L1_fix_R_given_M()

    plot_planet_profiles(planet)

def test_gen_prof_L1_fix_M_given_R():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_granite"],
        A1_T_rho_type   = [1],
        A1_T_rho_args   = [[None, 0.]],
        A1_R_layer      = [R_earth],
        P_s             = 0,
        T_s             = 300,
        )

    planet.M_max = M_earth

    planet.gen_prof_L1_fix_M_given_R()

    plot_planet_profiles(planet)

def test_gen_prof_L2_fix_R1_given_R_M():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = [1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.]],
        A1_R_layer      = [None, R_earth],
        M               = M_earth,
        P_s             = 0,
        T_s             = 300,
        )

    planet.gen_prof_L2_fix_R1_given_R_M()

    plot_planet_profiles(planet)

def test_gen_prof_L2_fix_R_given_M_R1():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = [1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.]],
        A1_R_layer      = [0.40*R_earth, R_earth],
        M               = M_earth,
        P_s             = 0,
        T_s             = 300,
        )

    planet.R_max = 2*R_earth
    planet.gen_prof_L2_fix_R_given_M_R1()

    plot_planet_profiles(planet)

def test_gen_prof_L2_fix_M_given_R1_R():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = [1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.]],
        A1_R_layer      = [0.40*R_earth, R_earth],
        P_s             = 0,
        T_s             = 300,
        )

    planet.M_max = 2*M_earth
    planet.gen_prof_L2_fix_M_given_R1_R()

    plot_planet_profiles(planet)

def test_gen_prof_L3_given_prof_L2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite"],
        A1_T_rho_type   = [1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.]],
        A1_R_layer      = [None, R_earth],
        M               = 0.887*M_earth,
        P_s             = 1e5,
        T_s             = 2000,
        num_attempt     = 10
        )

    planet.gen_prof_L2_fix_R1_given_R_M()

    mat_id_atm = "idg_N2"
    T_rho_type_atm = 1
    T_rho_args_atm = [None, 0]

    planet.gen_prof_L3_given_prof_L2(
        mat_id_atm,
        T_rho_type_atm,
        T_rho_args_atm,
        rho_min=1e-6
        )

    plot_planet_profiles(planet)

def test_gen_prof_L3_fix_R1_R2_given_R_M_I():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = [1, 1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
        A1_R_layer      = [None, None, R_earth],
        P_s             = 0,
        T_s             = 300,
        I_MR2           = 0.3*M_earth*R_earth**2,
        M               = M_earth,
        num_attempt     = 5,
        num_attempt_2   = 5
        )

    planet.gen_prof_L3_fix_R1_R2_given_R_M_I()

    plot_planet_profiles(planet)

def test_gen_prof_L3_fix_R2_given_R_M_R1():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "SESAME_steam"],
        A1_T_rho_type   = [1, 1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
        A1_R_layer      = [0.55*R_earth, None, R_earth],
        P_s             = 1e2,
        T_s             = 300,
        M               = M_earth
        )

    planet.gen_prof_L3_fix_R2_given_R_M_R1()

    plot_planet_profiles(planet)

def test_gen_prof_L3_fix_R1_given_R_M_R2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = [1, 1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
        A1_R_layer      = [None, 0.9*R_earth, R_earth],
        P_s             = 0,
        T_s             = 300,
        M               = M_earth
        )

    planet.gen_prof_L3_fix_R1_given_R_M_R2()

    plot_planet_profiles(planet)

def test_gen_prof_L3_fix_M_given_R_R1_R2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = [1, 1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
        A1_R_layer      = [0.5*R_earth, 0.9*R_earth, R_earth],
        P_s             = 0,
        T_s             = 300,
        M_max           = 2*M_earth
        )

    planet.gen_prof_L3_fix_M_given_R_R1_R2()

    plot_planet_profiles(planet)

def test_gen_prof_L3_fix_R_given_M_R1_R2():
    planet = woma.Planet(
        name            = "planet",
        A1_mat_layer    = ["Til_iron", "Til_granite", "Til_water"],
        A1_T_rho_type   = [1, 1, 1],
        A1_T_rho_args   = [[None, 0.], [None, 0.], [None, 0.]],
        A1_R_layer      = [0.5*R_earth, 0.9*R_earth, None],
        P_s             = 0,
        T_s             = 300,
        M               = M_earth,
        R_max           = 2*R_earth
        )

    planet.gen_prof_L3_fix_R_given_M_R1_R2()

    plot_planet_profiles(planet)

# ============================================================================ #

if __name__ == "__main__":    
    # Run some standard tests
    test_gen_prof_L1_fix_R_given_M()
    test_gen_prof_L2_fix_R1_given_R_M()
    test_gen_prof_L3_given_prof_L2()
    
    plt.show()
