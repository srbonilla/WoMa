"""
WoMa SESAME, ANEOS, and other (in ~SESAME-style tables) equations of state
"""

import numpy as np
from numba import njit

from woma.misc import glob_vars as gv
from woma.misc import utils as ut


@njit
def find_index_and_interp(x, A1_x):
    """Return the index and interpolation factor of a value in an array.

    Allows x outside A1_x. If so then intp will be < 0 or > 1.

    Parameters
    ----------
    x : float
        The value to find.

    A1_x : [float]
        The array to search.

    Returns
    -------
    idx : int
        The index of the last array element smaller than the value.

        0               If x is below A1_x.
        len(A1_x) - 2   If x is above A1_x.

    intp : float
        The interpolation factor for how far the values is from the
        indexed array value to the next.

        < 0     If x is below A1_x.
        > 1     If x is above A1_x.
    """
    # assert np.all(np.sort(A1_x) == A1_x)

    idx = np.searchsorted(A1_x, x, side="right") - 1
    # Return error values if outside the array
    if idx == -1:
        idx = 0
    elif idx >= len(A1_x) - 1:
        idx = len(A1_x) - 2

    # Check for duplicate elements
    if A1_x[idx + 1] != A1_x[idx]:
        intp = (x - A1_x[idx]) / (A1_x[idx + 1] - A1_x[idx])
    else:
        intp = 1.0

    return np.array([idx, intp])


def prepare_table_SESAME(A1_rho, A1_T, A2_P, A2_u, A2_s, verbosity=0):
    """Prepare SESAME-like tables to be used.

    ##Needs checking and tidying

    Parameters
    ----------
    A1_rho : [float]
        1d array of densities (kg m^-3).
    A1_T : [float]
        1d array of temperatures (K).
    A2_P : [float]
        2d array of pressures (Pa).
    A2_u : [float]
        2d array of internal energies (J kg^-1 ).
    A2_s : [float]
        2d array of specific entropies (J kg^-1 K^-1).
    verbosity : int, optional
        Printing options. The default is 0.

    Returns
    -------
    None.

    """

    # Basic dimension checks
    n_row = A1_rho.shape[0]
    n_col = A1_T.shape[0]
    assert A2_P.shape[0] == n_row
    assert A2_u.shape[0] == n_row
    assert A2_s.shape[0] == n_row
    assert A2_P.shape[1] == n_col
    assert A2_u.shape[1] == n_col
    assert A2_s.shape[1] == n_col

    # first element of A1_rho and A1_T cannot be == 0
    # because interpolation is in log rho, log T
    ### but we discard those in the interpolation functions anyway?
    small = A1_rho[1] * 0.0001
    if A1_rho[0] <= 0:
        A1_rho[0] = small
    if A1_T[0] <= 0:
        A1_T[0] = small

    # Non-negative elements
    assert np.all(A1_rho > 0)
    assert np.all(A1_T > 0)

    # Sorted arrays
    assert np.all(np.sort(A1_rho) == A1_rho)
    assert np.all(np.sort(A1_T) == A1_T)

    # Avoid negative pressures
    A2_P[A2_P < 0] = 0
    assert np.all(A2_P >= 0)

    # Negative u?
    # assert np.all(A2_u >= 0)

    # Negative u?
    # assert np.all(A2_s >= 0)

    # partial P / partial rho at fixed T must be >= 0
    count = 0
    for j, T in enumerate(A1_T):
        for i, rho in enumerate(A1_rho[:-1]):
            if A2_P[i + 1, j] < A2_P[i, j]:
                A2_P[i + 1, j] = A2_P[i, j]
                count += 1

    if verbosity >= 1:
        print("partial P / partial rho at fixed T must be >= 0")
        print(
            "count of modified values:", count, ", total table entries:", n_row * n_col
        )
        print("fraction:", count / n_row / n_col)

    # partial u / partial T at fixed rho must be >= 0
    count = 0
    for j, T in enumerate(A1_T[:-1]):
        for i, rho in enumerate(A1_rho):
            if A2_u[i, j + 1] < A2_u[i, j]:
                A2_u[i, j + 1] = A2_u[i, j]
                count += 1

    if verbosity >= 1:
        print("partial u / partial T at fixed rho must be >= 0")
        print(
            "count of modified values:", count, ", total table entries:", n_row * n_col
        )
        print("fraction:", count / n_row / n_col)


def load_table_SESAME(Fp_table):
    """Load and return the table file data.

    File contents: see write_table_SESAME().

    Parameters
    ----------
    Fp_table : str
        The table file path.

    Returns
    -------
    A1_rho, A1_T : [float]
        Density (kg m^-3) and temperature (K) arrays.

    A2_u, A2_P, A2_c, A2_s : [[float]]
        Table arrays of sp. int. energy (J kg^-1), pressure (Pa), sound speed
        (m s^-1), and sp. entropy (J K^-1 kg^-1).

    A1_log_*, A2_log_* : [float], [[float]]
        The natural log versions of the same arrays.
    """
    # Load
    Fp_table = ut.check_end(Fp_table, ".txt")
    with open(Fp_table) as f:
        # Skip the header
        for i in range(12):
            f.readline()

        # Skip the version date
        f.readline()

        num_rho, num_T = np.array(f.readline().split(), dtype=int)
        A2_u = np.empty((num_rho, num_T))
        A2_P = np.empty((num_rho, num_T))
        A2_c = np.empty((num_rho, num_T))
        A2_s = np.empty((num_rho, num_T))

        A1_rho = np.array(f.readline().split(), dtype=float)
        A1_T = np.array(f.readline().split(), dtype=float)

        for i_T in range(num_T):
            for i_rho in range(num_rho):
                (
                    A2_u[i_rho, i_T],
                    A2_P[i_rho, i_T],
                    A2_c[i_rho, i_T],
                    A2_s[i_rho, i_T],
                ) = np.array(f.readline().split(), dtype=float)

    # Prepare table
    prepare_table_SESAME(A1_rho, A1_T, A2_P, A2_u, A2_s, verbosity=0)

    return (
        A1_rho,
        A1_T,
        A2_u,
        A2_P,
        A2_c,
        A2_s,
        np.log(A1_rho),
        np.log(A1_T),
        np.log(A2_u),
        np.log(A2_P),
        np.log(A2_c),
        np.log(A2_s),
    )


def load_phase_table_ANEOS_forsterite():
    """Load and return the 2D array of KPA flag phase IDs.

    See https://github.com/ststewart/aneos-forsterite-2019 etc.

    ###WIP
    """
    import os, sys

    this_dir, this_file = os.path.split(__file__)
    path = os.path.join(this_dir, "../data/aneos-forsterite-2019/")
    sys.path.append(path)
    import eostable as eost

    MODELNAME = "Forsterite-ANEOS-SLVTv1.0G1"
    # Header information must all be compatible with float format
    MATID = 1.0  # MATID number
    DATE = 190802.0  # Date as a single 6-digit number YYMMDD
    VERSION = 0.1  # ANEOS Parameters Version number
    FMN = 70.0  # Formula weight in atomic numbers for Mg2SiO4
    FMW = 140.691  # Formula molecular weight (g/cm3) for Mg2SiO4
    # The following define the default initial state for material in the 201 table
    R0REF = 3.22  # g/cm3 *** R0REF is inserted into the density array
    K0REF = 1.1e12  # dynes/cm2
    T0REF = 298.0  # K -- *** T0REF is inserted into the temperature array
    P0REF = 1.0e6  # dynes/cm2 -- this defines the principal Hugoniot calculated below

    NewEOS = eost.extEOStable()  # FIRST make new empty EOS object
    NewEOS.loadextsesame(
        path + "NEW-SESAME-EXT.TXT"
    )  # LOAD THE EXTENDED 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.loadstdsesame(
        path + "NEW-SESAME-STD-NOTENSION.TXT"
    )  # LOAD THE STANDARD 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.MODELNAME = MODELNAME  # string set above in user input
    NewEOS.MDQ = np.zeros((NewEOS.NT, NewEOS.ND))  # makes the empty MDQ array
    # Units: g/cm3, K, GPa, MJ/kg, MJ/kg, MJ/K/kg, cm/s, MJ/K/kg, KPA flag. 2D arrays are (NT,ND).

    # Add the header info to the table. This could be done during the loading.
    # if made from this notebook, these values are set in the user-input above.
    # ** MAKE SURE THEY MATCH ANEOS.INPUT **
    NewEOS.MATID = MATID
    NewEOS.DATE = DATE
    NewEOS.VERSION = VERSION
    NewEOS.FMN = FMN
    NewEOS.FMW = FMW
    NewEOS.R0REF = R0REF
    NewEOS.K0REF = K0REF
    NewEOS.T0REF = T0REF
    NewEOS.P0REF = P0REF
    # Load the information from ANEOS.INPUT and ANEOS.OUTPUT
    NewEOS.loadaneos(
        aneosinfname=path + "ANEOS.INPUT",
        aneosoutfname=path + "ANEOS.OUTPUT",
        silent=True,
    )

    return NewEOS.KPA.T


# ========
# Awkwardly initialise SESAME-style tables as global variables needed for numba
# ========
# SESAME
(
    A1_rho_SESAME_iron,
    A1_T_SESAME_iron,
    A2_u_SESAME_iron,
    A2_P_SESAME_iron,
    A2_c_SESAME_iron,
    A2_s_SESAME_iron,
    A1_log_rho_SESAME_iron,
    A1_log_T_SESAME_iron,
    A2_log_u_SESAME_iron,
    A2_log_P_SESAME_iron,
    A2_log_c_SESAME_iron,
    A2_log_s_SESAME_iron,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_SESAME_basalt,
    A1_T_SESAME_basalt,
    A2_u_SESAME_basalt,
    A2_P_SESAME_basalt,
    A2_c_SESAME_basalt,
    A2_s_SESAME_basalt,
    A1_log_rho_SESAME_basalt,
    A1_log_T_SESAME_basalt,
    A2_log_u_SESAME_basalt,
    A2_log_P_SESAME_basalt,
    A2_log_c_SESAME_basalt,
    A2_log_s_SESAME_basalt,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_SESAME_water,
    A1_T_SESAME_water,
    A2_u_SESAME_water,
    A2_P_SESAME_water,
    A2_c_SESAME_water,
    A2_s_SESAME_water,
    A1_log_rho_SESAME_water,
    A1_log_T_SESAME_water,
    A2_log_u_SESAME_water,
    A2_log_P_SESAME_water,
    A2_log_c_SESAME_water,
    A2_log_s_SESAME_water,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_SS08_water,
    A1_T_SS08_water,
    A2_u_SS08_water,
    A2_P_SS08_water,
    A2_c_SS08_water,
    A2_s_SS08_water,
    A1_log_rho_SS08_water,
    A1_log_T_SS08_water,
    A2_log_u_SS08_water,
    A2_log_P_SS08_water,
    A2_log_c_SS08_water,
    A2_log_s_SS08_water,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)

# ANEOS
(
    A1_rho_ANEOS_forsterite,
    A1_T_ANEOS_forsterite,
    A2_u_ANEOS_forsterite,
    A2_P_ANEOS_forsterite,
    A2_c_ANEOS_forsterite,
    A2_s_ANEOS_forsterite,
    A1_log_rho_ANEOS_forsterite,
    A1_log_T_ANEOS_forsterite,
    A2_log_u_ANEOS_forsterite,
    A2_log_P_ANEOS_forsterite,
    A2_log_c_ANEOS_forsterite,
    A2_log_s_ANEOS_forsterite,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
# A2_phase_ANEOS_forsterite = load_phase_table_ANEOS_forsterite()
(
    A1_rho_ANEOS_iron,
    A1_T_ANEOS_iron,
    A2_u_ANEOS_iron,
    A2_P_ANEOS_iron,
    A2_c_ANEOS_iron,
    A2_s_ANEOS_iron,
    A1_log_rho_ANEOS_iron,
    A1_log_T_ANEOS_iron,
    A2_log_u_ANEOS_iron,
    A2_log_P_ANEOS_iron,
    A2_log_c_ANEOS_iron,
    A2_log_s_ANEOS_iron,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_ANEOS_Fe85Si15,
    A1_T_ANEOS_Fe85Si15,
    A2_u_ANEOS_Fe85Si15,
    A2_P_ANEOS_Fe85Si15,
    A2_c_ANEOS_Fe85Si15,
    A2_s_ANEOS_Fe85Si15,
    A1_log_rho_ANEOS_Fe85Si15,
    A1_log_T_ANEOS_Fe85Si15,
    A2_log_u_ANEOS_Fe85Si15,
    A2_log_P_ANEOS_Fe85Si15,
    A2_log_c_ANEOS_Fe85Si15,
    A2_log_s_ANEOS_Fe85Si15,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)

# AQUA
(
    A1_rho_AQUA,
    A1_T_AQUA,
    A2_u_AQUA,
    A2_P_AQUA,
    A2_c_AQUA,
    A2_s_AQUA,
    A1_log_rho_AQUA,
    A1_log_T_AQUA,
    A2_log_u_AQUA,
    A2_log_P_AQUA,
    A2_log_c_AQUA,
    A2_log_s_AQUA,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)

# CMS19
(
    A1_rho_CMS19_H,
    A1_T_CMS19_H,
    A2_u_CMS19_H,
    A2_P_CMS19_H,
    A2_c_CMS19_H,
    A2_s_CMS19_H,
    A1_log_rho_CMS19_H,
    A1_log_T_CMS19_H,
    A2_log_u_CMS19_H,
    A2_log_P_CMS19_H,
    A2_log_c_CMS19_H,
    A2_log_s_CMS19_H,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_CMS19_He,
    A1_T_CMS19_He,
    A2_u_CMS19_He,
    A2_P_CMS19_He,
    A2_c_CMS19_He,
    A2_s_CMS19_He,
    A1_log_rho_CMS19_He,
    A1_log_T_CMS19_He,
    A2_log_u_CMS19_He,
    A2_log_P_CMS19_He,
    A2_log_c_CMS19_He,
    A2_log_s_CMS19_He,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_CD21_HHe,
    A1_T_CD21_HHe,
    A2_u_CD21_HHe,
    A2_P_CD21_HHe,
    A2_c_CD21_HHe,
    A2_s_CD21_HHe,
    A1_log_rho_CD21_HHe,
    A1_log_T_CD21_HHe,
    A2_log_u_CD21_HHe,
    A2_log_P_CD21_HHe,
    A2_log_c_CD21_HHe,
    A2_log_s_CD21_HHe,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)

# Generic user-provided custom tables
(
    A1_rho_custom_0,
    A1_T_custom_0,
    A2_u_custom_0,
    A2_P_custom_0,
    A2_c_custom_0,
    A2_s_custom_0,
    A1_log_rho_custom_0,
    A1_log_T_custom_0,
    A2_log_u_custom_0,
    A2_log_P_custom_0,
    A2_log_c_custom_0,
    A2_log_s_custom_0,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_custom_1,
    A1_T_custom_1,
    A2_u_custom_1,
    A2_P_custom_1,
    A2_c_custom_1,
    A2_s_custom_1,
    A1_log_rho_custom_1,
    A1_log_T_custom_1,
    A2_log_u_custom_1,
    A2_log_P_custom_1,
    A2_log_c_custom_1,
    A2_log_s_custom_1,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_custom_2,
    A1_T_custom_2,
    A2_u_custom_2,
    A2_P_custom_2,
    A2_c_custom_2,
    A2_s_custom_2,
    A1_log_rho_custom_2,
    A1_log_T_custom_2,
    A2_log_u_custom_2,
    A2_log_P_custom_2,
    A2_log_c_custom_2,
    A2_log_s_custom_2,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_custom_3,
    A1_T_custom_3,
    A2_u_custom_3,
    A2_P_custom_3,
    A2_c_custom_3,
    A2_s_custom_3,
    A1_log_rho_custom_3,
    A1_log_T_custom_3,
    A2_log_u_custom_3,
    A2_log_P_custom_3,
    A2_log_c_custom_3,
    A2_log_s_custom_3,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)
(
    A1_rho_custom_4,
    A1_T_custom_4,
    A2_u_custom_4,
    A2_P_custom_4,
    A2_c_custom_4,
    A2_s_custom_4,
    A1_log_rho_custom_4,
    A1_log_T_custom_4,
    A2_log_u_custom_4,
    A2_log_P_custom_4,
    A2_log_c_custom_4,
    A2_log_s_custom_4,
) = (
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros(1),
    np.zeros(1),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
    np.zeros((2, 2)),
)

# ========
# Generic
# ========
@njit
def Z_rho_T(rho, T, mat_id, Z_choice):
    """Compute an equation of state parameter from the density and temperature.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Z_choice : str
        The parameter to calculate, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag.

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    # Unpack the arrays of Z, log(rho), and log(T)
    if mat_id == gv.id_SESAME_iron:
        A1_log_rho, A1_log_T = (A1_log_rho_SESAME_iron, A1_log_T_SESAME_iron)
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_iron
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_iron
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_iron
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_rho, A1_log_T = (A1_log_rho_SESAME_basalt, A1_log_T_SESAME_basalt)
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_basalt
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_basalt
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_basalt
    elif mat_id == gv.id_SESAME_water:
        A1_log_rho, A1_log_T = (A1_log_rho_SESAME_water, A1_log_T_SESAME_water)
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_water
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_water
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_water
    elif mat_id == gv.id_SS08_water:
        A1_log_rho, A1_log_T = (A1_log_rho_SS08_water, A1_log_T_SS08_water)
        if Z_choice == "P":
            A2_Z = A2_P_SS08_water
        elif Z_choice == "u":
            A2_Z = A2_u_SS08_water
        elif Z_choice == "s":
            A2_Z = A2_s_SS08_water
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_rho, A1_log_T = (A1_log_rho_ANEOS_forsterite, A1_log_T_ANEOS_forsterite)
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_forsterite
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_forsterite
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_forsterite
        # elif Z_choice == "phase":
        #     A2_Z = A2_phase_ANEOS_forsterite
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_rho, A1_log_T = (A1_log_rho_ANEOS_iron, A1_log_T_ANEOS_iron)
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_iron
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_iron
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_iron
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_rho, A1_log_T = (A1_log_rho_ANEOS_Fe85Si15, A1_log_T_ANEOS_Fe85Si15)
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_Fe85Si15
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_Fe85Si15
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_Fe85Si15
    elif mat_id == gv.id_AQUA:
        A1_log_rho, A1_log_T = (A1_log_rho_AQUA, A1_log_T_AQUA)
        if Z_choice == "P":
            A2_Z = A2_P_AQUA
        elif Z_choice == "u":
            A2_Z = A2_u_AQUA
        elif Z_choice == "s":
            A2_Z = A2_s_AQUA
    elif mat_id == gv.id_CMS19_H:
        A1_log_rho, A1_log_T = (A1_log_rho_CMS19_H, A1_log_T_CMS19_H)
        if Z_choice == "P":
            A2_Z = A2_P_CMS19_H
        elif Z_choice == "u":
            A2_Z = A2_u_CMS19_H
        elif Z_choice == "s":
            A2_Z = A2_s_CMS19_H
    elif mat_id == gv.id_CMS19_He:
        A1_log_rho, A1_log_T = (A1_log_rho_CMS19_He, A1_log_T_CMS19_He)
        if Z_choice == "P":
            A2_Z = A2_P_CMS19_He
        elif Z_choice == "u":
            A2_Z = A2_u_CMS19_He
        elif Z_choice == "s":
            A2_Z = A2_s_CMS19_He
    elif mat_id == gv.id_CD21_HHe:
        A1_log_rho, A1_log_T = (A1_log_rho_CD21_HHe, A1_log_T_CD21_HHe)
        if Z_choice == "P":
            A2_Z = A2_P_CD21_HHe
        elif Z_choice == "u":
            A2_Z = A2_u_CD21_HHe
        elif Z_choice == "s":
            A2_Z = A2_s_CD21_HHe
    elif mat_id == gv.id_custom_0:
        A1_log_rho, A1_log_T = (A1_log_rho_custom_0, A1_log_T_custom_0)
        if Z_choice == "P":
            A2_Z = A2_P_custom_0
        elif Z_choice == "u":
            A2_Z = A2_u_custom_0
        elif Z_choice == "s":
            A2_Z = A2_s_custom_0
    elif mat_id == gv.id_custom_1:
        A1_log_rho, A1_log_T = (A1_log_rho_custom_1, A1_log_T_custom_1)
        if Z_choice == "P":
            A2_Z = A2_P_custom_1
        elif Z_choice == "u":
            A2_Z = A2_u_custom_1
        elif Z_choice == "s":
            A2_Z = A2_s_custom_1
    elif mat_id == gv.id_custom_2:
        A1_log_rho, A1_log_T = (A1_log_rho_custom_2, A1_log_T_custom_2)
        if Z_choice == "P":
            A2_Z = A2_P_custom_2
        elif Z_choice == "u":
            A2_Z = A2_u_custom_2
        elif Z_choice == "s":
            A2_Z = A2_s_custom_2
    elif mat_id == gv.id_custom_3:
        A1_log_rho, A1_log_T = (A1_log_rho_custom_3, A1_log_T_custom_3)
        if Z_choice == "P":
            A2_Z = A2_P_custom_3
        elif Z_choice == "u":
            A2_Z = A2_u_custom_3
        elif Z_choice == "s":
            A2_Z = A2_s_custom_3
    elif mat_id == gv.id_custom_4:
        A1_log_rho, A1_log_T = (A1_log_rho_custom_4, A1_log_T_custom_4)
        if Z_choice == "P":
            A2_Z = A2_P_custom_4
        elif Z_choice == "u":
            A2_Z = A2_u_custom_4
        elif Z_choice == "s":
            A2_Z = A2_s_custom_4
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T * 1)  # why is numba so weird?

    # 2D interpolation (bilinear with log(rho), log(T)) to find Z(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T = find_index_and_interp(log_T, A1_log_T)
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    # Table values
    Z_1 = A2_Z[idx_rho, idx_T]
    Z_2 = A2_Z[idx_rho, idx_T + 1]
    Z_3 = A2_Z[idx_rho + 1, idx_T]
    Z_4 = A2_Z[idx_rho + 1, idx_T + 1]

    # Choose the nearest table value, no interpolation
    if Z_choice == "phase":
        if intp_rho < 0.5:
            if intp_T < 0.5:
                return Z_1
            else:
                return Z_2
        else:
            if intp_T < 0.5:
                return Z_3
            else:
                return Z_4

    # Check for non-positive values
    if Z_choice in ["u", "s"]:
        # If more than two table values are non-positive then return zero
        num_non_pos = np.sum(np.array([Z_1, Z_2, Z_3, Z_4]) < 0)
        if num_non_pos > 2:
            return 0.0

        # If just one or two are non-positive then replace them with a tiny value
        # Unless already trying to extrapolate in which case return zero
        if num_non_pos > 0:
            if intp_rho < 0 or intp_T < 0:
                return 0.0
            else:
                Z_tiny = np.amin(np.abs(A2_Z)) * 1e-3
                if Z_1 <= 0:
                    Z_1 = Z_tiny
                if Z_2 <= 0:
                    Z_2 = Z_tiny
                if Z_3 <= 0:
                    Z_3 = Z_tiny
                if Z_4 <= 0:
                    Z_4 = Z_tiny

    # Interpolate with the log values
    Z_1 = np.log(Z_1)
    Z_2 = np.log(Z_2)
    Z_3 = np.log(Z_3)
    Z_4 = np.log(Z_4)

    # Z(rho, T)
    Z = (1 - intp_rho) * ((1 - intp_T) * Z_1 + intp_T * Z_2) + intp_rho * (
        (1 - intp_T) * Z_3 + intp_T * Z_4
    )

    # Convert back from log
    return np.exp(Z)


@njit
def Z_rho_Y(rho, Y, mat_id, Z_choice, Y_choice):
    """Compute an equation of state parameter from the density and another
    parameter.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    Y : float
        The chosen input parameter (SI).

    mat_id : int
        Material id.

    Z_choice, Y_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    assert Z_choice != Y_choice

    if Y_choice == "T":
        return Z_rho_T(rho, Y, mat_id, Z_choice)

    # Unpack the arrays of Z, log(rho), and log(Y)
    if mat_id == gv.id_SESAME_iron:
        A1_log_rho = A1_log_rho_SESAME_iron
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_iron
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_iron
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_iron
        if Y_choice == "P":
            A2_log_Y = A2_log_P_SESAME_iron
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_SESAME_iron
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_SESAME_iron
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_rho = A1_log_rho_SESAME_basalt
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_basalt
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_basalt
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_basalt
        if Y_choice == "P":
            A2_log_Y = A2_log_P_SESAME_basalt
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_SESAME_basalt
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_SESAME_basalt
    elif mat_id == gv.id_SESAME_water:
        A1_log_rho = A1_log_rho_SESAME_water
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_water
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_water
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_water
        if Y_choice == "P":
            A2_log_Y = A2_log_P_SESAME_water
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_SESAME_water
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_SESAME_water
    elif mat_id == gv.id_SS08_water:
        A1_log_rho = A1_log_rho_SS08_water
        if Z_choice == "P":
            A2_Z = A2_P_SS08_water
        elif Z_choice == "u":
            A2_Z = A2_u_SS08_water
        elif Z_choice == "s":
            A2_Z = A2_s_SS08_water
        if Y_choice == "P":
            A2_log_Y = A2_log_P_SS08_water
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_SS08_water
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_SS08_water
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_rho = A1_log_rho_ANEOS_forsterite
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_forsterite
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_forsterite
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_forsterite
        # elif Z_choice == "phase":
        #     A2_Z = A2_phase_ANEOS_forsterite
        if Y_choice == "P":
            A2_log_Y = A2_log_P_ANEOS_forsterite
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_ANEOS_forsterite
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_ANEOS_forsterite
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_rho = A1_log_rho_ANEOS_iron
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_iron
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_iron
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_iron
        if Y_choice == "P":
            A2_log_Y = A2_log_P_ANEOS_iron
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_ANEOS_iron
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_ANEOS_iron
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_rho = A1_log_rho_ANEOS_Fe85Si15
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_Fe85Si15
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_Fe85Si15
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_Fe85Si15
        if Y_choice == "P":
            A2_log_Y = A2_log_P_ANEOS_Fe85Si15
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_ANEOS_Fe85Si15
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_ANEOS_Fe85Si15
    elif mat_id == gv.id_AQUA:
        A1_log_rho = A1_log_rho_AQUA
        if Z_choice == "P":
            A2_Z = A2_P_AQUA
        elif Z_choice == "u":
            A2_Z = A2_u_AQUA
        elif Z_choice == "s":
            A2_Z = A2_s_AQUA
        if Y_choice == "P":
            A2_log_Y = A2_log_P_AQUA
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_AQUA
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_AQUA
    elif mat_id == gv.id_CMS19_H:
        A1_log_rho = A1_log_rho_CMS19_H
        if Z_choice == "P":
            A2_Z = A2_P_CMS19_H
        elif Z_choice == "u":
            A2_Z = A2_u_CMS19_H
        elif Z_choice == "s":
            A2_Z = A2_s_CMS19_H
        if Y_choice == "P":
            A2_log_Y = A2_log_P_CMS19_H
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_CMS19_H
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_CMS19_H
    elif mat_id == gv.id_CMS19_He:
        A1_log_rho = A1_log_rho_CMS19_He
        if Z_choice == "P":
            A2_Z = A2_P_CMS19_He
        elif Z_choice == "u":
            A2_Z = A2_u_CMS19_He
        elif Z_choice == "s":
            A2_Z = A2_s_CMS19_He
        if Y_choice == "P":
            A2_log_Y = A2_log_P_CMS19_He
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_CMS19_He
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_CMS19_He
    elif mat_id == gv.id_CD21_HHe:
        A1_log_rho = A1_log_rho_CD21_HHe
        if Z_choice == "P":
            A2_Z = A2_P_CD21_HHe
        elif Z_choice == "u":
            A2_Z = A2_u_CD21_HHe
        elif Z_choice == "s":
            A2_Z = A2_s_CD21_HHe
        if Y_choice == "P":
            A2_log_Y = A2_log_P_CD21_HHe
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_CD21_HHe
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_CD21_HHe
    elif mat_id == gv.id_custom_0:
        A1_log_rho = A1_log_rho_custom_0
        if Z_choice == "P":
            A2_Z = A2_P_custom_0
        elif Z_choice == "u":
            A2_Z = A2_u_custom_0
        elif Z_choice == "s":
            A2_Z = A2_s_custom_0
        if Y_choice == "P":
            A2_log_Y = A2_log_P_custom_0
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_custom_0
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_custom_0
    elif mat_id == gv.id_custom_1:
        A1_log_rho = A1_log_rho_custom_1
        if Z_choice == "P":
            A2_Z = A2_P_custom_1
        elif Z_choice == "u":
            A2_Z = A2_u_custom_1
        elif Z_choice == "s":
            A2_Z = A2_s_custom_1
        if Y_choice == "P":
            A2_log_Y = A2_log_P_custom_1
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_custom_1
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_custom_1
    elif mat_id == gv.id_custom_2:
        A1_log_rho = A1_log_rho_custom_2
        if Z_choice == "P":
            A2_Z = A2_P_custom_2
        elif Z_choice == "u":
            A2_Z = A2_u_custom_2
        elif Z_choice == "s":
            A2_Z = A2_s_custom_2
        if Y_choice == "P":
            A2_log_Y = A2_log_P_custom_2
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_custom_2
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_custom_2
    elif mat_id == gv.id_custom_3:
        A1_log_rho = A1_log_rho_custom_3
        if Z_choice == "P":
            A2_Z = A2_P_custom_3
        elif Z_choice == "u":
            A2_Z = A2_u_custom_3
        elif Z_choice == "s":
            A2_Z = A2_s_custom_3
        if Y_choice == "P":
            A2_log_Y = A2_log_P_custom_3
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_custom_3
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_custom_3
    elif mat_id == gv.id_custom_4:
        A1_log_rho = A1_log_rho_custom_4
        if Z_choice == "P":
            A2_Z = A2_P_custom_4
        elif Z_choice == "u":
            A2_Z = A2_u_custom_4
        elif Z_choice == "s":
            A2_Z = A2_s_custom_4
        if Y_choice == "P":
            A2_log_Y = A2_log_P_custom_4
        elif Y_choice == "u":
            A2_log_Y = A2_log_u_custom_4
        elif Y_choice == "s":
            A2_log_Y = A2_log_s_custom_4
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Ignore the first elements of rho = 0, T = 0
    A2_Z = A2_Z[1:, 1:]
    A2_log_Y = A2_log_Y[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_Y = np.log(Y)

    # 2D interpolation (bilinear with log(rho), log(Y)) to find Z(rho, Y).
    # If rho and/or Y are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Y (in this and the next density slice of the 2D Y array)
    idx_Y_1_intp_Y_1 = find_index_and_interp(log_Y, A2_log_Y[idx_rho])
    idx_Y_1 = int(idx_Y_1_intp_Y_1[0])
    intp_Y_1 = idx_Y_1_intp_Y_1[1]
    idx_Y_2_intp_Y_2 = find_index_and_interp(log_Y, A2_log_Y[idx_rho + 1])
    idx_Y_2 = int(idx_Y_2_intp_Y_2[0])
    intp_Y_2 = idx_Y_2_intp_Y_2[1]

    # Table values
    Z_1 = A2_Z[idx_rho, idx_Y_1]
    Z_2 = A2_Z[idx_rho, idx_Y_1 + 1]
    Z_3 = A2_Z[idx_rho + 1, idx_Y_2]
    Z_4 = A2_Z[idx_rho + 1, idx_Y_2 + 1]

    # Choose the nearest table value, no interpolation
    if Z_choice == "phase":
        if intp_rho < 0.5:
            if intp_Y_1 < 0.5:
                return Z_1
            else:
                return Z_2
        else:
            if intp_Y_2 < 0.5:
                return Z_3
            else:
                return Z_4

    # If below the minimum Y at this rho then just use the lowest table values
    if Y_choice in ["u", "s"]:
        if idx_rho >= 0 and (intp_Y_1 < 0 or intp_Y_2 < 0 or Z_1 > Z_2 or Z_3 > Z_4):
            intp_Y_1 = 0
            intp_Y_2 = 0

    # Check for non-positive values
    if Z_choice in ["u", "s"]:
        # If more than two table values are non-positive then return zero
        num_non_pos = np.sum(np.array([Z_1, Z_2, Z_3, Z_4]) < 0)
        if num_non_pos > 2:
            return 0.0

        # If just one or two are non-positive then replace them with a tiny value
        # Unless already trying to extrapolate in which case return zero
        if num_non_pos > 0:
            if intp_rho < 0 or intp_Y_1 < 0 or intp_Y_2 < 0:
                return 0.0
            else:
                # Z_tiny  = np.amin(A2_Z[A2_Z > 0]) * 1e-3
                Z_tiny = np.amin(np.abs(A2_Z)) * 1e-3
                if Z_1 <= 0:
                    Z_1 = Z_tiny
                if Z_2 <= 0:
                    Z_2 = Z_tiny
                if Z_3 <= 0:
                    Z_3 = Z_tiny
                if Z_4 <= 0:
                    Z_4 = Z_tiny

    # Interpolate with the log values
    Z_1 = np.log(Z_1)
    Z_2 = np.log(Z_2)
    Z_3 = np.log(Z_3)
    Z_4 = np.log(Z_4)

    # Z(rho, Y)
    Z = (1 - intp_rho) * ((1 - intp_Y_1) * Z_1 + intp_Y_1 * Z_2) + intp_rho * (
        (1 - intp_Y_2) * Z_3 + intp_Y_2 * Z_4
    )

    # Convert back from log
    return np.exp(Z)


@njit
def Z_X_T(X, T, mat_id, Z_choice, X_choice):
    """Compute an equation of state parameter from another parameter and the
    temperature.

    Warning: Not all of the 2D X arrays are monotonic with density along a
    temperature slice, which can break the attempted interpolation.

    Parameters
    ----------
    X : float
        The chosen input parameter (SI).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Z_choice, X_choice : str
        The parameter to calculate, and the other input parameter, choose from:
            P       Pressure.
            u       Specific internal energy.
            s       Specific entropy.
            phase   Phase KPA flag (Z_choice only).

    Returns
    -------
    Z : float
        The chosen parameter (SI).
    """
    assert Z_choice != X_choice

    if X_choice == "rho":
        return Z_rho_T(X, T, mat_id, Z_choice)

    # Unpack the arrays of Z and log(X)
    if mat_id == gv.id_SESAME_iron:
        A1_log_T = A1_log_T_SESAME_iron
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_iron
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_iron
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_iron
        if X_choice == "P":
            A2_log_X = A2_log_P_SESAME_iron
        elif X_choice == "u":
            A2_log_X = A2_log_u_SESAME_iron
        elif X_choice == "s":
            A2_log_X = A2_log_s_SESAME_iron
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_T = A1_log_T_SESAME_basalt
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_basalt
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_basalt
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_basalt
        if X_choice == "P":
            A2_log_X = A2_log_P_SESAME_basalt
        elif X_choice == "u":
            A2_log_X = A2_log_u_SESAME_basalt
        elif X_choice == "s":
            A2_log_X = A2_log_s_SESAME_basalt
    elif mat_id == gv.id_SESAME_water:
        A1_log_T = A1_log_T_SESAME_water
        if Z_choice == "P":
            A2_Z = A2_P_SESAME_water
        elif Z_choice == "u":
            A2_Z = A2_u_SESAME_water
        elif Z_choice == "s":
            A2_Z = A2_s_SESAME_water
        if X_choice == "P":
            A2_log_X = A2_log_P_SESAME_water
        elif X_choice == "u":
            A2_log_X = A2_log_u_SESAME_water
        elif X_choice == "s":
            A2_log_X = A2_log_s_SESAME_water
    elif mat_id == gv.id_SS08_water:
        A1_log_T = A1_log_T_SS08_water
        if Z_choice == "P":
            A2_Z = A2_P_SS08_water
        elif Z_choice == "u":
            A2_Z = A2_u_SS08_water
        elif Z_choice == "s":
            A2_Z = A2_s_SS08_water
        if X_choice == "P":
            A2_log_X = A2_log_P_SS08_water
        elif X_choice == "u":
            A2_log_X = A2_log_u_SS08_water
        elif X_choice == "s":
            A2_log_X = A2_log_s_SS08_water
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_T = A1_log_T_ANEOS_forsterite
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_forsterite
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_forsterite
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_forsterite
        # elif Z_choice == "phase":
        #     A2_Z = A2_phase_ANEOS_forsterite
        if X_choice == "P":
            A2_log_X = A2_log_P_ANEOS_forsterite
        elif X_choice == "u":
            A2_log_X = A2_log_u_ANEOS_forsterite
        elif X_choice == "s":
            A2_log_X = A2_log_s_ANEOS_forsterite
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_T = A1_log_T_ANEOS_iron
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_iron
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_iron
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_iron
        if X_choice == "P":
            A2_log_X = A2_log_P_ANEOS_iron
        elif X_choice == "u":
            A2_log_X = A2_log_u_ANEOS_iron
        elif X_choice == "s":
            A2_log_X = A2_log_s_ANEOS_iron
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_T = A1_log_T_ANEOS_Fe85Si15
        if Z_choice == "P":
            A2_Z = A2_P_ANEOS_Fe85Si15
        elif Z_choice == "u":
            A2_Z = A2_u_ANEOS_Fe85Si15
        elif Z_choice == "s":
            A2_Z = A2_s_ANEOS_Fe85Si15
        if X_choice == "P":
            A2_log_X = A2_log_P_ANEOS_Fe85Si15
        elif X_choice == "u":
            A2_log_X = A2_log_u_ANEOS_Fe85Si15
        elif X_choice == "s":
            A2_log_X = A2_log_s_ANEOS_Fe85Si15
    elif mat_id == gv.id_AQUA:
        A1_log_T = A1_log_T_AQUA
        if Z_choice == "P":
            A2_Z = A2_P_AQUA
        elif Z_choice == "u":
            A2_Z = A2_u_AQUA
        elif Z_choice == "s":
            A2_Z = A2_s_AQUA
        if X_choice == "P":
            A2_log_X = A2_log_P_AQUA
        elif X_choice == "u":
            A2_log_X = A2_log_u_AQUA
        elif X_choice == "s":
            A2_log_X = A2_log_s_AQUA
    elif mat_id == gv.id_CMS19_H:
        A1_log_T = A1_log_T_CMS19_H
        if Z_choice == "P":
            A2_Z = A2_P_CMS19_H
        elif Z_choice == "u":
            A2_Z = A2_u_CMS19_H
        elif Z_choice == "s":
            A2_Z = A2_s_CMS19_H
        if X_choice == "P":
            A2_log_X = A2_log_P_CMS19_H
        elif X_choice == "u":
            A2_log_X = A2_log_u_CMS19_H
        elif X_choice == "s":
            A2_log_X = A2_log_s_CMS19_H
    elif mat_id == gv.id_CMS19_He:
        A1_log_T = A1_log_T_CMS19_He
        if Z_choice == "P":
            A2_Z = A2_P_CMS19_He
        elif Z_choice == "u":
            A2_Z = A2_u_CMS19_He
        elif Z_choice == "s":
            A2_Z = A2_s_CMS19_He
        if X_choice == "P":
            A2_log_X = A2_log_P_CMS19_He
        elif X_choice == "u":
            A2_log_X = A2_log_u_CMS19_He
        elif X_choice == "s":
            A2_log_X = A2_log_s_CMS19_He
    elif mat_id == gv.id_CD21_HHe:
        A1_log_T = A1_log_T_CD21_HHe
        if Z_choice == "P":
            A2_Z = A2_P_CD21_HHe
        elif Z_choice == "u":
            A2_Z = A2_u_CD21_HHe
        elif Z_choice == "s":
            A2_Z = A2_s_CD21_HHe
        if X_choice == "P":
            A2_log_X = A2_log_P_CD21_HHe
        elif X_choice == "u":
            A2_log_X = A2_log_u_CD21_HHe
        elif X_choice == "s":
            A2_log_X = A2_log_s_CD21_HHe
    elif mat_id == gv.id_custom_0:
        A1_log_T = A1_log_T_custom_0
        if Z_choice == "P":
            A2_Z = A2_P_custom_0
        elif Z_choice == "u":
            A2_Z = A2_u_custom_0
        elif Z_choice == "s":
            A2_Z = A2_s_custom_0
        if X_choice == "P":
            A2_log_X = A2_log_P_custom_0
        elif X_choice == "u":
            A2_log_X = A2_log_u_custom_0
        elif X_choice == "s":
            A2_log_X = A2_log_s_custom_0
    elif mat_id == gv.id_custom_1:
        A1_log_T = A1_log_T_custom_1
        if Z_choice == "P":
            A2_Z = A2_P_custom_1
        elif Z_choice == "u":
            A2_Z = A2_u_custom_1
        elif Z_choice == "s":
            A2_Z = A2_s_custom_1
        if X_choice == "P":
            A2_log_X = A2_log_P_custom_1
        elif X_choice == "u":
            A2_log_X = A2_log_u_custom_1
        elif X_choice == "s":
            A2_log_X = A2_log_s_custom_1
    elif mat_id == gv.id_custom_2:
        A1_log_T = A1_log_T_custom_2
        if Z_choice == "P":
            A2_Z = A2_P_custom_2
        elif Z_choice == "u":
            A2_Z = A2_u_custom_2
        elif Z_choice == "s":
            A2_Z = A2_s_custom_2
        if X_choice == "P":
            A2_log_X = A2_log_P_custom_2
        elif X_choice == "u":
            A2_log_X = A2_log_u_custom_2
        elif X_choice == "s":
            A2_log_X = A2_log_s_custom_2
    elif mat_id == gv.id_custom_3:
        A1_log_T = A1_log_T_custom_3
        if Z_choice == "P":
            A2_Z = A2_P_custom_3
        elif Z_choice == "u":
            A2_Z = A2_u_custom_3
        elif Z_choice == "s":
            A2_Z = A2_s_custom_3
        if X_choice == "P":
            A2_log_X = A2_log_P_custom_3
        elif X_choice == "u":
            A2_log_X = A2_log_u_custom_3
        elif X_choice == "s":
            A2_log_X = A2_log_s_custom_3
    elif mat_id == gv.id_custom_4:
        A1_log_T = A1_log_T_custom_4
        if Z_choice == "P":
            A2_Z = A2_P_custom_4
        elif Z_choice == "u":
            A2_Z = A2_u_custom_4
        elif Z_choice == "s":
            A2_Z = A2_s_custom_4
        if X_choice == "P":
            A2_log_X = A2_log_P_custom_4
        elif X_choice == "u":
            A2_log_X = A2_log_u_custom_4
        elif X_choice == "s":
            A2_log_X = A2_log_s_custom_4
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_T) == 1:
        raise ValueError(
            "Please load the corresponding EoS table. See woma.load_eos_tables()."
        )

    # Ignore the first elements of rho = 0, T = 0
    A2_Z = A2_Z[1:, 1:]
    A2_log_X = A2_log_X[1:, 1:]

    # Convert to log
    log_T = np.log(T)
    log_X = np.log(X)

    # 2D interpolation (bilinear with log(X), log(T)) to find Z(X, T).
    # If T and/or X are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Temperature
    idx_T_intp_T = find_index_and_interp(log_T, A1_log_T[1:])
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    # X (in this and the next temperature slice of the 2D X array)
    idx_X_1_intp_X_1 = find_index_and_interp(log_X, A2_log_X[:, idx_T])
    idx_X_1 = int(idx_X_1_intp_X_1[0])
    intp_X_1 = idx_X_1_intp_X_1[1]
    idx_X_2_intp_X_2 = find_index_and_interp(log_X, A2_log_X[:, idx_T + 1])
    idx_X_2 = int(idx_X_2_intp_X_2[0])
    intp_X_2 = idx_X_2_intp_X_2[1]

    # Table values
    Z_1 = A2_Z[idx_X_1, idx_T]
    Z_2 = A2_Z[idx_X_1 + 1, idx_T]
    Z_3 = A2_Z[idx_X_2, idx_T + 1]
    Z_4 = A2_Z[idx_X_2 + 1, idx_T + 1]

    # Choose the nearest table value, no interpolation
    if Z_choice == "phase":
        if intp_T < 0.5:
            if intp_X_1 < 0.5:
                return Z_1
            else:
                return Z_2
        else:
            if intp_X_2 < 0.5:
                return Z_3
            else:
                return Z_4

    # Check for non-positive values
    if Z_choice in ["u", "s"]:
        # If more than two table values are non-positive then return zero
        num_non_pos = np.sum(np.array([Z_1, Z_2, Z_3, Z_4]) < 0)
        if num_non_pos > 2:
            return 0.0

        # If just one or two are non-positive then replace them with a tiny value
        # Unless already trying to extrapolate in which case return zero
        if num_non_pos > 0:
            if intp_T < 0 or intp_X_1 < 0 or intp_X_2 < 0:
                return 0.0
            else:
                # Z_tiny  = np.amin(A2_Z[A2_Z > 0]) * 1e-3
                Z_tiny = np.amin(np.abs(A2_Z)) * 1e-3
                if Z_1 <= 0:
                    Z_1 = Z_tiny
                if Z_2 <= 0:
                    Z_2 = Z_tiny
                if Z_3 <= 0:
                    Z_3 = Z_tiny
                if Z_4 <= 0:
                    Z_4 = Z_tiny

    # Interpolate with the log values
    Z_1 = np.log(Z_1)
    Z_2 = np.log(Z_2)
    Z_3 = np.log(Z_3)
    Z_4 = np.log(Z_4)

    # Z(X, T)
    Z = (1 - intp_T) * ((1 - intp_X_1) * Z_1 + intp_X_1 * Z_2) + intp_T * (
        (1 - intp_X_2) * Z_3 + intp_X_2 * Z_4
    )

    # Convert back from log
    return np.exp(Z)


# ========
# Pressure
# ========
@njit
def P_u_rho(u, rho, mat_id):
    """Compute the pressure from the internal energy and density.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    P : float
        Pressure (Pa).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A2_log_u_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_log_u_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_log_u_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SS08_water,
            A1_log_rho_SS08_water,
            A2_log_u_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_log_u_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_log_u_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_log_u_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_P, A1_log_rho, A2_log_u = (A2_P_AQUA, A1_log_rho_AQUA, A2_log_u_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_log_u_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_log_u_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A2_log_u_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_0,
            A1_log_rho_custom_0,
            A2_log_u_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_1,
            A1_log_rho_custom_1,
            A2_log_u_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_2,
            A1_log_rho_custom_2,
            A2_log_u_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_3,
            A1_log_rho_custom_3,
            A2_log_u_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_4,
            A1_log_rho_custom_4,
            A2_log_u_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Ignore the first elements of rho = 0, T = 0
    A2_P = A2_P[1:, 1:]
    A2_log_u = A2_log_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_u = np.log(u)

    # 2D interpolation (bilinear with log(rho), log(u)) to find P(rho, u).
    # If rho and/or u are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # u (in this and the next density slice of the 2D u array)
    idx_u_1_intp_u_1 = find_index_and_interp(log_u, A2_log_u[idx_rho])
    idx_u_1 = int(idx_u_1_intp_u_1[0])
    intp_u_1 = idx_u_1_intp_u_1[1]
    idx_u_2_intp_u_2 = find_index_and_interp(log_u, A2_log_u[idx_rho + 1])
    idx_u_2 = int(idx_u_2_intp_u_2[0])
    intp_u_2 = idx_u_2_intp_u_2[1]

    P_1 = A2_P[idx_rho, idx_u_1]
    P_2 = A2_P[idx_rho, idx_u_1 + 1]
    P_3 = A2_P[idx_rho + 1, idx_u_2]
    P_4 = A2_P[idx_rho + 1, idx_u_2 + 1]

    # If below the minimum u at this rho then just use the lowest table values
    if idx_rho >= 0 and (intp_u_1 < 0 or intp_u_2 < 0 or P_1 > P_2 or P_3 > P_4):
        intp_u_1 = 0
        intp_u_2 = 0

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([P_1, P_2, P_3, P_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_u_1 < 0 or intp_u_2 < 0:
            return 0.0
        else:
            # P_tiny  = np.amin(A2_P[A2_P > 0]) * 1e-3
            P_tiny = np.amin(np.abs(A2_P)) * 1e-3
            if P_1 <= 0:
                P_1 = P_tiny
            if P_2 <= 0:
                P_2 = P_tiny
            if P_3 <= 0:
                P_3 = P_tiny
            if P_4 <= 0:
                P_4 = P_tiny

    # Interpolate with the log values
    P_1 = np.log(P_1)
    P_2 = np.log(P_2)
    P_3 = np.log(P_3)
    P_4 = np.log(P_4)

    # P(rho, u)
    P = (1 - intp_rho) * ((1 - intp_u_1) * P_1 + intp_u_1 * P_2) + intp_rho * (
        (1 - intp_u_2) * P_3 + intp_u_2 * P_4
    )

    # Convert back from log
    return np.exp(P)


@njit
def P_T_rho(T, rho, mat_id):
    """Compute the internal energy from the density and temperature.

    Parameters
    ----------
    T : float
        Temperature (K).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    P : float
        Pressure (Pa).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A1_log_T_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A1_log_T_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_SESAME_water,
            A1_log_rho_SESAME_water,
            A1_log_T_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_SS08_water,
            A1_log_rho_SS08_water,
            A1_log_T_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A1_log_T_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A1_log_T_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A1_log_T_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_P, A1_log_rho, A1_log_T = (A2_P_AQUA, A1_log_rho_AQUA, A1_log_T_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_CMS19_H,
            A1_log_rho_CMS19_H,
            A1_log_T_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_CMS19_He,
            A1_log_rho_CMS19_He,
            A1_log_T_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A1_log_T_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_custom_0,
            A1_log_rho_custom_0,
            A1_log_T_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_custom_1,
            A1_log_rho_custom_1,
            A1_log_T_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_custom_2,
            A1_log_rho_custom_2,
            A1_log_T_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_custom_3,
            A1_log_rho_custom_3,
            A1_log_T_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A2_P, A1_log_rho, A1_log_T = (
            A2_P_custom_4,
            A1_log_rho_custom_4,
            A1_log_T_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T * 1)  # why is numba so weird?

    # 2D interpolation (bilinear with log(rho), log(T)) to find u(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T = find_index_and_interp(log_T, A1_log_T)
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    P_1 = A2_P[idx_rho, idx_T]
    P_2 = A2_P[idx_rho, idx_T + 1]
    P_3 = A2_P[idx_rho + 1, idx_T]
    P_4 = A2_P[idx_rho + 1, idx_T + 1]

    # Interpolate with the log values
    P_1 = np.log(P_1)
    P_2 = np.log(P_2)
    P_3 = np.log(P_3)
    P_4 = np.log(P_4)

    # P(rho, T)
    P = (1 - intp_rho) * ((1 - intp_T) * P_1 + intp_T * P_2) + intp_rho * (
        (1 - intp_T) * P_3 + intp_T * P_4
    )

    # Convert back from log
    return np.exp(P)


# ========
# Temperature
# ========
@njit
def T_rho_s(rho, s, mat_id):
    """Compute the temperature from the density and specific entropy.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    s : float
        Specific entropy (J kg^-1 K^-1).

    mat_id : int
        Material id.

    Returns
    -------
    T : float
        Temperature (K).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A2_s_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_s_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_s_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_SS08_water,
            A1_log_rho_SS08_water,
            A2_s_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_s_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_s_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_s_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A1_log_T, A1_log_rho, A2_s = (A1_log_T_AQUA, A1_log_rho_AQUA, A2_s_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_s_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_s_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A2_s_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_custom_0,
            A1_log_rho_custom_0,
            A2_s_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_custom_1,
            A1_log_rho_custom_1,
            A2_s_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_custom_2,
            A1_log_rho_custom_2,
            A2_s_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_custom_3,
            A1_log_rho_custom_3,
            A2_s_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A1_log_T, A1_log_rho, A2_s = (
            A1_log_T_custom_4,
            A1_log_rho_custom_4,
            A2_s_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Convert to log
    log_rho = np.log(rho)

    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # s (in this and the next density slice of the 2D s array)
    idx_s_1_intp_s_1 = find_index_and_interp(s, A2_s[idx_rho])
    idx_s_1 = int(idx_s_1_intp_s_1[0])
    intp_s_1 = idx_s_1_intp_s_1[1]
    idx_s_2_intp_s_2 = find_index_and_interp(s, A2_s[idx_rho + 1])
    idx_s_2 = int(idx_s_2_intp_s_2[0])
    intp_s_2 = idx_s_2_intp_s_2[1]

    # Normal interpolation
    log_T = (1 - intp_rho) * (
        (1 - intp_s_1) * A1_log_T[idx_s_1] + intp_s_1 * A1_log_T[idx_s_1 + 1]
    ) + intp_rho * (
        (1 - intp_s_2) * A1_log_T[idx_s_2] + intp_s_2 * A1_log_T[idx_s_2 + 1]
    )

    # Convert back from log
    T = np.exp(log_T)
    if T < 0:
        T = 0

    return T


@njit
def T_u_rho(u, rho, mat_id):
    """Compute the temperature from the density and specific entropy.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    T : float
        Temperature (K).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A2_u_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_u_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_u_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_SS08_water,
            A1_log_rho_SS08_water,
            A2_u_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_u_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_u_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_u_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A1_log_T, A1_log_rho, A2_u = (A1_log_T_AQUA, A1_log_rho_AQUA, A2_u_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_u_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_u_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A2_u_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_custom_0,
            A1_log_rho_custom_0,
            A2_u_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_custom_1,
            A1_log_rho_custom_1,
            A2_u_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_custom_2,
            A1_log_rho_custom_2,
            A2_u_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_custom_3,
            A1_log_rho_custom_3,
            A2_u_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A1_log_T, A1_log_rho, A2_u = (
            A1_log_T_custom_4,
            A1_log_rho_custom_4,
            A2_u_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Convert to log
    log_rho = np.log(rho)

    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho)
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # s (in this and the next density slice of the 2D u array)
    idx_u_1_intp_u_1 = find_index_and_interp(u, A2_u[idx_rho])
    idx_u_1 = int(idx_u_1_intp_u_1[0])
    intp_u_1 = idx_u_1_intp_u_1[1]
    idx_u_2_intp_u_2 = find_index_and_interp(u, A2_u[idx_rho + 1])
    idx_u_2 = int(idx_u_2_intp_u_2[0])
    intp_u_2 = idx_u_2_intp_u_2[1]

    # Normal interpolation
    log_T = (1 - intp_rho) * (
        (1 - intp_u_1) * A1_log_T[idx_u_1] + intp_u_1 * A1_log_T[idx_u_1 + 1]
    ) + intp_rho * (
        (1 - intp_u_2) * A1_log_T[idx_u_2] + intp_u_2 * A1_log_T[idx_u_2 + 1]
    )

    # Convert back from log
    T = np.exp(log_T)
    if T < 0:
        T = 0

    return T


# ========
# Specific internal energy
# ========
@njit
def u_rho_T(rho, T, mat_id):
    """Compute the internal energy from the density and temperature.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    u : float
        Specific internal energy (J kg^-1).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A1_log_T_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A1_log_T_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SESAME_water,
            A1_log_rho_SESAME_water,
            A1_log_T_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_SS08_water,
            A1_log_rho_SS08_water,
            A1_log_T_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A1_log_T_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A1_log_T_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A1_log_T_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_u, A1_log_rho, A1_log_T = (A2_u_AQUA, A1_log_rho_AQUA, A1_log_T_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_CMS19_H,
            A1_log_rho_CMS19_H,
            A1_log_T_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_CMS19_He,
            A1_log_rho_CMS19_He,
            A1_log_T_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A1_log_T_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_custom_0,
            A1_log_rho_custom_0,
            A1_log_T_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_custom_1,
            A1_log_rho_custom_1,
            A1_log_T_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_custom_2,
            A1_log_rho_custom_2,
            A1_log_T_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_custom_3,
            A1_log_rho_custom_3,
            A1_log_T_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A2_u, A1_log_rho, A1_log_T = (
            A2_u_custom_4,
            A1_log_rho_custom_4,
            A1_log_T_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Ignore the first elements of rho = 0, T = 0
    A2_u = A2_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T * 1)  # why is numba so weird?

    # 2D interpolation (bilinear with log(rho), log(T)) to find u(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T = find_index_and_interp(log_T, A1_log_T[1:])
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    u_1 = A2_u[idx_rho, idx_T]
    u_2 = A2_u[idx_rho, idx_T + 1]
    u_3 = A2_u[idx_rho + 1, idx_T]
    u_4 = A2_u[idx_rho + 1, idx_T + 1]

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([u_1, u_2, u_3, u_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_T < 0:
            return 0.0
        else:
            u_tiny = np.amin(np.abs(A2_u)) * 1e-3
            if u_1 <= 0:
                u_1 = u_tiny
            if u_2 <= 0:
                u_2 = u_tiny
            if u_3 <= 0:
                u_3 = u_tiny
            if u_4 <= 0:
                u_4 = u_tiny

    # Interpolate with the log values
    u_1 = np.log(u_1)
    u_2 = np.log(u_2)
    u_3 = np.log(u_3)
    u_4 = np.log(u_4)

    # u(rho, T)
    u = (1 - intp_rho) * ((1 - intp_T) * u_1 + intp_T * u_2) + intp_rho * (
        (1 - intp_T) * u_3 + intp_T * u_4
    )

    # Convert back from log
    return np.exp(u)


# ========
# Specific entropy
# ========
@njit
def s_u_rho(u, rho, mat_id):
    """Compute the specific entropy from the internal energy and density.

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    s : float
        Specific entropy (J K^-1 kg^-1).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        raise ValueError("No entropy values for this material")
    elif mat_id == gv.id_SESAME_basalt:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_log_u_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_log_u_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_SS08_water,
            A1_log_rho_SS08_water,
            A2_log_u_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_log_u_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_log_u_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_log_u_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_s, A1_log_rho, A2_log_u = (A2_s_AQUA, A1_log_rho_AQUA, A2_log_u_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_log_u_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_log_u_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A2_log_u_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_custom_0,
            A1_log_rho_custom_0,
            A2_log_u_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_custom_1,
            A1_log_rho_custom_1,
            A2_log_u_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_custom_2,
            A1_log_rho_custom_2,
            A2_log_u_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_custom_3,
            A1_log_rho_custom_3,
            A2_log_u_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A2_s, A1_log_rho, A2_log_u = (
            A2_s_custom_4,
            A1_log_rho_custom_4,
            A2_log_u_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Ignore the first elements of rho = 0, T = 0
    A2_s = A2_s[1:, 1:]
    A2_log_u = A2_log_u[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_u = np.log(u)

    # 2D interpolation (bilinear with log(rho), log(u)) to find s(rho, u).
    # If rho and/or u are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # u (in this and the next density slice of the 2D u array)
    idx_u_1_intp_u_1 = find_index_and_interp(log_u, A2_log_u[idx_rho])
    idx_u_1 = int(idx_u_1_intp_u_1[0])
    intp_u_1 = idx_u_1_intp_u_1[1]
    idx_u_2_intp_u_2 = find_index_and_interp(log_u, A2_log_u[idx_rho + 1])
    idx_u_2 = int(idx_u_2_intp_u_2[0])
    intp_u_2 = idx_u_2_intp_u_2[1]

    s_1 = A2_s[idx_rho, idx_u_1]
    s_2 = A2_s[idx_rho, idx_u_1 + 1]
    s_3 = A2_s[idx_rho + 1, idx_u_2]
    s_4 = A2_s[idx_rho + 1, idx_u_2 + 1]

    # If below the minimum u at this rho then just use the lowest table values
    if idx_rho >= 0 and (intp_u_1 < 0 or intp_u_2 < 0 or s_1 > s_2 or s_3 > s_4):
        intp_u_1 = 0
        intp_u_2 = 0

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([s_1, s_2, s_3, s_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_u_1 < 0 or intp_u_2 < 0:
            return 0.0
        else:
            s_tiny = np.amin(np.abs(A2_s)) * 1e-3
            if s_1 <= 0:
                s_1 = s_tiny
            if s_2 <= 0:
                s_2 = s_tiny
            if s_3 <= 0:
                s_3 = s_tiny
            if s_4 <= 0:
                s_4 = s_tiny

    # Interpolate with the log values
    s_1 = np.log(s_1)
    s_2 = np.log(s_2)
    s_3 = np.log(s_3)
    s_4 = np.log(s_4)

    # s(rho, u)
    s = (1 - intp_rho) * ((1 - intp_u_1) * s_1 + intp_u_1 * s_2) + intp_rho * (
        (1 - intp_u_2) * s_3 + intp_u_2 * s_4
    )

    # Convert back from log
    return np.exp(s)


@njit
def s_rho_T(rho, T, mat_id):
    """Compute the specific entropy from the density and temperature.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    T : float
        Temperature (K).

    mat_id : int
        Material id.

    Returns
    -------
    s : float
        Specific entropy (J kg^-1 K^-1).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        raise ValueError("No entropy values for this material")
    elif mat_id == gv.id_SESAME_basalt:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A1_log_T_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SESAME_water,
            A1_log_rho_SESAME_water,
            A1_log_T_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_SS08_water,
            A1_log_rho_SS08_water,
            A1_log_T_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A1_log_T_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A1_log_T_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A1_log_T_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_s, A1_log_rho, A1_log_T = (A2_s_AQUA, A1_log_rho_AQUA, A1_log_T_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_CMS19_H,
            A1_log_rho_CMS19_H,
            A1_log_T_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_CMS19_He,
            A1_log_rho_CMS19_He,
            A1_log_T_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A1_log_T_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_custom_0,
            A1_log_rho_custom_0,
            A1_log_T_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_custom_1,
            A1_log_rho_custom_1,
            A1_log_T_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_custom_2,
            A1_log_rho_custom_2,
            A1_log_T_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_custom_3,
            A1_log_rho_custom_3,
            A1_log_T_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A2_s, A1_log_rho, A1_log_T = (
            A2_s_custom_4,
            A1_log_rho_custom_4,
            A1_log_T_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Ignore the first elements of rho = 0, T = 0
    A2_s = A2_s[1:, 1:]

    # Convert to log
    log_rho = np.log(rho)
    log_T = np.log(T)

    # 2D interpolation (bilinear with log(rho), log(T)) to find s(rho, T).
    # If rho and/or T are below or above the table, then use the interpolation
    # formula to extrapolate using the edge and edge-but-one values.

    # Density
    idx_rho_intp_rho = find_index_and_interp(log_rho, A1_log_rho[1:])
    idx_rho = int(idx_rho_intp_rho[0])
    intp_rho = idx_rho_intp_rho[1]

    # Temperature
    idx_T_intp_T = find_index_and_interp(log_T, A1_log_T[1:])
    idx_T = int(idx_T_intp_T[0])
    intp_T = idx_T_intp_T[1]

    s_1 = A2_s[idx_rho, idx_T]
    s_2 = A2_s[idx_rho, idx_T + 1]
    s_3 = A2_s[idx_rho + 1, idx_T]
    s_4 = A2_s[idx_rho + 1, idx_T + 1]

    # If more than two table values are non-positive then return zero
    num_non_pos = np.sum(np.array([s_1, s_2, s_3, s_4]) < 0)
    if num_non_pos > 2:
        return 0.0

    # If just one or two are non-positive then replace them with a tiny value
    # Unless already trying to extrapolate in which case return zero
    if num_non_pos > 0:
        if intp_rho < 0 or intp_T < 0:
            return 0.0
        else:
            s_tiny = np.amin(np.abs(A2_s)) * 1e-3
            if s_1 <= 0:
                s_1 = s_tiny
            if s_2 <= 0:
                s_2 = s_tiny
            if s_3 <= 0:
                s_3 = s_tiny
            if s_4 <= 0:
                s_4 = s_tiny

    # s(rho, T)
    s = (1 - intp_rho) * ((1 - intp_T) * s_1 + intp_T * s_2) + intp_rho * (
        (1 - intp_T) * s_3 + intp_T * s_4
    )

    return s


# ========
# Density
# ========
@njit
def rho_u_P(u, P, mat_id, rho_ref):
    """Compute the density from the specific internal energy and pressure.
    Start search for roots at the reference density. If there are are multiple
    roots, return the root with the smallest abs(log(root_rho) - log(rho_ref)).
    Note: this algorithm is slightly different from the one used in SWIFT

    Parameters
    ----------
    u : float
        Specific internal energy (J kg^-1).

    P : float
        Pressure (Pa).

    mat_id : int
        Material id.

    rho_ref : float
        Reference density. Pick root closest to this value.

    Returns
    -------
    rho : float
        Density (kg m^-3).
    """
    # Unpack the parameters
    if mat_id == gv.id_SESAME_iron:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_iron,
            A1_log_rho_SESAME_iron,
            A2_log_u_SESAME_iron,
        )
    elif mat_id == gv.id_SESAME_basalt:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_basalt,
            A1_log_rho_SESAME_basalt,
            A2_log_u_SESAME_basalt,
        )
    elif mat_id == gv.id_SESAME_water:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SESAME_water,
            A1_log_rho_SESAME_water,
            A2_log_u_SESAME_water,
        )
    elif mat_id == gv.id_SS08_water:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_SS08_water,
            A1_log_rho_SS08_water,
            A2_log_u_SS08_water,
        )
    elif mat_id == gv.id_ANEOS_forsterite:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_forsterite,
            A1_log_rho_ANEOS_forsterite,
            A2_log_u_ANEOS_forsterite,
        )
    elif mat_id == gv.id_ANEOS_iron:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_iron,
            A1_log_rho_ANEOS_iron,
            A2_log_u_ANEOS_iron,
        )
    elif mat_id == gv.id_ANEOS_Fe85Si15:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_ANEOS_Fe85Si15,
            A1_log_rho_ANEOS_Fe85Si15,
            A2_log_u_ANEOS_Fe85Si15,
        )
    elif mat_id == gv.id_AQUA:
        A2_P, A1_log_rho, A2_log_u = (A2_P_AQUA, A1_log_rho_AQUA, A2_log_u_AQUA)
    elif mat_id == gv.id_CMS19_H:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CMS19_H,
            A1_log_rho_CMS19_H,
            A2_log_u_CMS19_H,
        )
    elif mat_id == gv.id_CMS19_He:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CMS19_He,
            A1_log_rho_CMS19_He,
            A2_log_u_CMS19_He,
        )
    elif mat_id == gv.id_CD21_HHe:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_CD21_HHe,
            A1_log_rho_CD21_HHe,
            A2_log_u_CD21_HHe,
        )
    elif mat_id == gv.id_custom_0:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_0,
            A1_log_rho_custom_0,
            A2_log_u_custom_0,
        )
    elif mat_id == gv.id_custom_1:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_1,
            A1_log_rho_custom_1,
            A2_log_u_custom_1,
        )
    elif mat_id == gv.id_custom_2:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_2,
            A1_log_rho_custom_2,
            A2_log_u_custom_2,
        )
    elif mat_id == gv.id_custom_3:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_3,
            A1_log_rho_custom_3,
            A2_log_u_custom_3,
        )
    elif mat_id == gv.id_custom_4:
        A2_P, A1_log_rho, A2_log_u = (
            A2_P_custom_4,
            A1_log_rho_custom_4,
            A2_log_u_custom_4,
        )
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(A1_log_rho) == 1:
        raise ValueError(
            "Please load the corresponding SESAME table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    # Ignore the first elements of rho = 0, T = 0
    A2_P = A2_P[1:, 1:]
    A1_log_rho = A1_log_rho[1:]
    A2_log_u = A2_log_u[1:, 1:]

    # Convert inputs to log
    log_u = np.log(u)
    log_P = np.log(P)
    log_rho_ref = np.log(rho_ref)

    # Find rounded down index of reference density. This is where we start our search
    idx_rho_intp_rho_ref = find_index_and_interp(log_rho_ref, A1_log_rho)
    idx_rho_ref = int(idx_rho_intp_rho_ref[0])

    # If no roots are found in the current search range, we increase search range
    # by search_factor_log_rho above and below the reference density each iteration.
    search_factor_log_rho = np.log(2)

    # Initialise the minimum and maximum densities we're searching to at the
    # reference density. These will change before the first iteration.
    log_rho_min = log_rho_ref
    log_rho_max = log_rho_ref

    # When searching, we increase the range above/below with every iteration if no
    # roots are found in the current range. We search from index idx_rho_above_min
    # to idx_rho_above_max when searching above. idx_rho_above_min is the closest
    # index above the reference density which we haven't seached in a previous iteration
    # and idx_rho_above_max is the rounded up index associated with log_rho_max
    # When searching below, we search from idx_rho_below_max to idx_rho_below_min.
    # idx_rho_below_max is the closest index below the reference density which we
    # haven't seached in a previous iteration and idx_rho_below_min is the rounded down
    # index associated with log_rho_min

    # Initialise search indices around rho_ref
    idx_rho_below_max = idx_rho_ref
    idx_rho_above_min = idx_rho_ref

    # If we find a root, it will get stored as closest_root
    closest_root = 0

    # Counters will stop us getting stuck in a while loop.
    max_counter = 100
    counter1 = 0

    # Initialise these because of numba
    P_above_upper = 0
    P_below_lower = 0

    # Start search for roots
    while closest_root == 0 and counter1 < max_counter:
        # Increase search range by search_factor_log_rho
        log_rho_max += search_factor_log_rho
        idx_rho_above_max = int(find_index_and_interp(log_rho_max, A1_log_rho)[0])
        log_rho_min -= search_factor_log_rho
        idx_rho_below_min = int(find_index_and_interp(log_rho_min, A1_log_rho)[0])

        counter2 = 0
        # If table densities have large enough increments that increasing search range
        # by search_factor_log_rho doesn't change the table indices we're looking between,
        # increase search range by search_factor_log_rho until we're between new indices
        while (
            idx_rho_below_min > idx_rho_below_max
            or idx_rho_above_max < idx_rho_above_min
        ) and counter2 < max_counter:
            log_rho_max += search_factor_log_rho
            idx_rho_above_max = int(find_index_and_interp(log_rho_max, A1_log_rho)[0])
            log_rho_min -= search_factor_log_rho
            idx_rho_below_min = int(find_index_and_interp(log_rho_min, A1_log_rho)[0])
            counter2 += 1

        # When searching above/below, we are looking for where the pressure P(rho, u)
        # of the table densities changes from being less than to more than, or vice versa,
        # the desired pressure. If this is the case, there is a root between these
        # table values of rho.

        # First look for roots above rho_ref
        for idx_rho in range(idx_rho_above_min, idx_rho_above_max + 1):

            # This is similar to P_u_rho, but we're not interest in intp_rho,
            # but instead calculate the pressure for both intp_rho=0 and intp_rho=1

            # u (in this and the next density slice of the 2D u array)
            idx_u_1_intp_u_1 = find_index_and_interp(log_u, A2_log_u[idx_rho])
            idx_u_1 = int(idx_u_1_intp_u_1[0])
            intp_u_1 = idx_u_1_intp_u_1[1]
            idx_u_2_intp_u_2 = find_index_and_interp(log_u, A2_log_u[idx_rho + 1])
            idx_u_2 = int(idx_u_2_intp_u_2[0])
            intp_u_2 = idx_u_2_intp_u_2[1]

            P_1 = A2_P[idx_rho, idx_u_1]
            P_2 = A2_P[idx_rho, idx_u_1 + 1]
            P_3 = A2_P[idx_rho + 1, idx_u_2]
            P_4 = A2_P[idx_rho + 1, idx_u_2 + 1]

            # If below the minimum u at this rho then just use the lowest table values
            if idx_rho >= 0 and (
                intp_u_1 < 0 or intp_u_2 < 0 or P_1 > P_2 or P_3 > P_4
            ):
                intp_u_1 = 0
                intp_u_2 = 0

            # If more than two table values are non-positive then return zero
            num_non_pos = np.sum(np.array([P_1, P_2, P_3, P_4]) < 0)
            if num_non_pos > 2:
                return 0.0

            # If just one or two are non-positive then replace them with a tiny value
            # Unless already trying to extrapolate in which case return zero
            if num_non_pos > 0:
                if intp_u_1 < 0 or intp_u_2 < 0:
                    return 0.0
                else:
                    # P_tiny  = np.amin(A2_P[A2_P > 0]) * 1e-3
                    P_tiny = np.amin(np.abs(A2_P)) * 1e-3
                    if P_1 <= 0:
                        P_1 = P_tiny
                    if P_2 <= 0:
                        P_2 = P_tiny
                    if P_3 <= 0:
                        P_3 = P_tiny
                    if P_4 <= 0:
                        P_4 = P_tiny

            # Interpolate with the log values
            P_1 = np.log(P_1)
            P_2 = np.log(P_2)
            P_3 = np.log(P_3)
            P_4 = np.log(P_4)

            # Pressure for intp_rho = 0
            P_above_lower = np.exp((1 - intp_u_1) * P_1 + intp_u_1 * P_2)

            # Because of linear interpolation, pressures are not exactly continuous
            # as we go from one side of a grid point to another. See if there is
            # a root between the last P_above_upper and the new P_above_lower,
            # which are approx the same.
            if idx_rho != idx_rho_ref:
                if (P_above_lower - P) * (P_above_upper - P) <= 0:
                    closest_root = np.exp(A1_log_rho[idx_rho])
                    break

            # Pressure for intp_rho = 1
            P_above_upper = np.exp((1 - intp_u_2) * P_3 + intp_u_2 * P_4)

            # Does the pressure of the adjacent table densities switch from being
            # above to below the desired pressure, or vice versa? If so, there is a root.
            if (P_above_lower - P) * (P_above_upper - P) <= 0:

                # If there is a root, interpolate between the table values:
                intp_rho = (log_P - ((1 - intp_u_1) * P_1 + intp_u_1 * P_2)) / (
                    ((1 - intp_u_2) * P_3 + intp_u_2 * P_4)
                    - ((1 - intp_u_1) * P_1 + intp_u_1 * P_2)
                )

                closest_root = np.exp(
                    A1_log_rho[idx_rho]
                    + intp_rho * (A1_log_rho[idx_rho + 1] - A1_log_rho[idx_rho])
                )

                # If the root is between the same table values as the reference value, then this is the closest root,
                # so we can return it without further searching
                if idx_rho == idx_rho_ref:
                    return closest_root
                break

        # If we found a root above, we can decrease our search range below,
        # since we're only looking for the closest root.
        if closest_root:
            log_rho_min = log_rho_ref - (np.log(closest_root) - log_rho_ref)
            idx_rho_below_min = int(find_index_and_interp(log_rho_min, A1_log_rho)[0])

        # Now look for roots below rho_ref
        for idx_rho in range(idx_rho_below_max, idx_rho_below_min - 1, -1):
            # u (in this and the next density slice of the 2D u array)
            idx_u_1_intp_u_1 = find_index_and_interp(log_u, A2_log_u[idx_rho])
            idx_u_1 = int(idx_u_1_intp_u_1[0])
            intp_u_1 = idx_u_1_intp_u_1[1]
            idx_u_2_intp_u_2 = find_index_and_interp(log_u, A2_log_u[idx_rho + 1])
            idx_u_2 = int(idx_u_2_intp_u_2[0])
            intp_u_2 = idx_u_2_intp_u_2[1]

            P_1 = A2_P[idx_rho, idx_u_1]
            P_2 = A2_P[idx_rho, idx_u_1 + 1]
            P_3 = A2_P[idx_rho + 1, idx_u_2]
            P_4 = A2_P[idx_rho + 1, idx_u_2 + 1]

            # If below the minimum u at this rho then just use the lowest table values
            if idx_rho >= 0 and (
                intp_u_1 < 0 or intp_u_2 < 0 or P_1 > P_2 or P_3 > P_4
            ):
                intp_u_1 = 0
                intp_u_2 = 0

            # If more than two table values are non-positive then return zero
            num_non_pos = np.sum(np.array([P_1, P_2, P_3, P_4]) < 0)
            if num_non_pos > 2:
                return 0.0

            # If just one or two are non-positive then replace them with a tiny value
            # Unless already trying to extrapolate in which case return zero
            if num_non_pos > 0:
                if intp_u_1 < 0 or intp_u_2 < 0:
                    return 0.0
                else:
                    # P_tiny  = np.amin(A2_P[A2_P > 0]) * 1e-3
                    P_tiny = np.amin(np.abs(A2_P)) * 1e-3
                    if P_1 <= 0:
                        P_1 = P_tiny
                    if P_2 <= 0:
                        P_2 = P_tiny
                    if P_3 <= 0:
                        P_3 = P_tiny
                    if P_4 <= 0:
                        P_4 = P_tiny

            # Interpolate with the log values
            P_1 = np.log(P_1)
            P_2 = np.log(P_2)
            P_3 = np.log(P_3)
            P_4 = np.log(P_4)

            # Pressure for intp_rho = 1
            P_below_upper = np.exp(((1 - intp_u_2) * P_3 + intp_u_2 * P_4))

            # Because of linear interpolation, pressures are not exactly continuous
            # as we go from one side of a grid point to another. See if there is
            # a root between the last P_below_lower and the new P_below_upper,
            # which are approx the same.
            if idx_rho != idx_rho_ref:
                if (P_below_lower - P) * (P_below_upper - P) <= 0:
                    closest_root = np.exp(A1_log_rho[idx_rho + 1])
                    break

            # Pressure for intp_rho = 0
            P_below_lower = np.exp(((1 - intp_u_1) * P_1 + intp_u_1 * P_2))

            # Does the pressure of the adjacent table densities switch from being
            # above to below the desired pressure, or vice versa? If so, there is a root.
            if (P_below_lower - P) * (P_below_upper - P) <= 0:

                # If there is a root, interpolate between the table values:
                intp_rho = (log_P - ((1 - intp_u_1) * P_1 + intp_u_1 * P_2)) / (
                    ((1 - intp_u_2) * P_3 + intp_u_2 * P_4)
                    - ((1 - intp_u_1) * P_1 + intp_u_1 * P_2)
                )

                root_rho_below = np.exp(
                    A1_log_rho[idx_rho]
                    + intp_rho * (A1_log_rho[idx_rho + 1] - A1_log_rho[idx_rho])
                )

                # If we found a root above, which one is closer to the reference rho?
                if closest_root:
                    if np.abs(np.log(root_rho_below) - np.log(rho_ref)) < np.abs(
                        np.log(closest_root) - np.log(rho_ref)
                    ):
                        closest_root = root_rho_below
                else:
                    closest_root = root_rho_below

                break

        # Return the root if we found one
        if closest_root:
            return closest_root

        # If we didn't find a root, get ready to extend the search range
        idx_rho_below_max = idx_rho_below_min
        idx_rho_above_min = idx_rho_above_max
        counter1 += 1

        # If we cover the whole EoS table and don't find a root, give an error
        if idx_rho_below_max == 0 and idx_rho_above_min == len(A1_log_rho) - 2:
            raise ValueError("rho_u_P failed to converge on root within EoS table")

    # If we don't find a root before we reach max_counter, give an error
    raise ValueError("rho_u_P failed to converge on root after many iterations")
