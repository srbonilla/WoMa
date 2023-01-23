""" 
WoMa Tillotson equations of state
"""

import numpy as np
from numba import njit
import os

from woma.misc import glob_vars as gv


def load_u_cold_array(mat_id):
    """Load precomputed values of cold internal energy.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    u_cold_array : [float]
        Precomputed values of cold internal energy from function
        _create_u_cold_array() (J kg^-1).
    """
    if mat_id == gv.id_Til_iron:
        u_cold_array = np.load(gv.Fp_u_cold_Til_iron)
    elif mat_id == gv.id_Til_granite:
        u_cold_array = np.load(gv.Fp_u_cold_Til_granite)
    elif mat_id == gv.id_Til_basalt:
        u_cold_array = np.load(gv.Fp_u_cold_Til_basalt)
    elif mat_id == gv.id_Til_water:
        u_cold_array = np.load(gv.Fp_u_cold_Til_water)
    else:
        raise ValueError("Invalid material ID")

    return u_cold_array


# Set None values for cold internal energy arrays
A1_u_cold_iron = np.zeros(1)
A1_u_cold_granite = np.zeros(1)
A1_u_cold_basalt = np.zeros(1)
A1_u_cold_water = np.zeros(1)


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

    # Material constants for Tillotson EoS (SI)
    # mat_id, rho_0, a, b, A, B, u_0, u_iv, u_cv, alpha, beta, eta_min, P_min, eta_zero
    iron = np.array(
        [
            gv.id_Til_iron,
            7800,
            0.5,
            1.5,
            1.28e11,
            1.05e11,
            9.5e6,
            2.4e6,
            8.67e6,
            5,
            5,
            0,
            0,
            0,
        ]
    )
    granite = np.array(
        [
            gv.id_Til_granite,
            2680,
            0.5,
            1.3,
            1.8e10,
            1.8e10,
            1.6e7,
            3.5e6,
            1.8e7,
            5,
            5,
            0,
            0,
            0,
        ]
    )
    basalt = np.array(
        [
            gv.id_Til_basalt,
            2700,
            0.5,
            1.5,
            2.67e10,
            2.67e10,
            4.87e8,
            4.72e6,
            1.82e7,
            5,
            5,
            0,
            0,
            0,
        ]
    )
    water = np.array(
        [
            gv.id_Til_water,
            998,
            0.7,
            0.15,
            2.18e9,
            1.325e10,
            7.0e6,
            4.19e5,
            2.69e6,
            10,
            5,
            0.925,
            0,
            0.875,
        ]
    )

    if mat_id == gv.id_Til_iron:
        material = iron
    elif mat_id == gv.id_Til_granite:
        material = granite
    elif mat_id == gv.id_Til_basalt:
        material = basalt
    elif mat_id == gv.id_Til_water:
        material = water
    else:
        raise ValueError("Invalid material ID")

    # Unpack the parameters
    (
        rho_0,
        a,
        b,
        A,
        B,
        u_0,
        u_iv,
        u_cv,
        alpha,
        beta,
        eta_min,
        P_min,
        eta_zero,
    ) = material[1:]

    eta = rho / rho_0
    eta_sq = eta * eta
    mu = eta - 1.0
    nu = 1.0 / eta - 1.0
    w = u / (u_0 * eta_sq) + 1.0
    w_inv = 1.0 / w

    P_c = 0.0
    P_e = 0.0
    P = 0.0

    # Condensed or cold
    P_c = (a + b * w_inv) * rho * u + A * mu + B * mu * mu

    if eta < eta_zero:
        P_c = 0.0
    elif eta < eta_min:
        P_c *= (eta - eta_zero) / (eta_min - eta_zero)

    # Expanded and hot
    P_e = a * rho * u + (b * rho * u * w_inv + A * mu * np.exp(-beta * nu)) * np.exp(
        -alpha * nu * nu
    )

    # Condensed or cold state
    if (1.0 < eta) or (u < u_iv):
        P = P_c

    # Expanded and hot state
    elif (eta < 1) and (u_cv < u):
        P = P_e

    # Hybrid state
    else:
        P = ((u - u_iv) * P_e + (u_cv - u) * P_c) / (u_cv - u_iv)

    # Minimum pressure
    if P < P_min:
        P = P_min

    return P


@njit
def C_V_Til(mat_id):
    """Return the specific heat capacity.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    C_V : float
        Specific heat capacity (J kg^-1 K^-1).
    """
    if mat_id == gv.id_Til_iron:
        return 449.0
    elif mat_id == gv.id_Til_granite:
        return 790.0
    elif mat_id == gv.id_Til_basalt:
        return 790.0
    elif mat_id == gv.id_Til_water:
        return 4186.0
    else:
        raise ValueError("Invalid material ID")


@njit
def _rho_0(mat_id):
    """Return the density for which the cold internal energy is zero.

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    rho_0 : float
        Density (kg m^-3).
    """
    if mat_id == gv.id_Til_iron:
        return 7800.0
    elif mat_id == gv.id_Til_granite:
        return 2680.0
    elif mat_id == gv.id_Til_basalt:
        return 2700.0
    elif mat_id == gv.id_Til_water:
        return 998.0
    else:
        raise ValueError("Invalid material ID")


@njit
def u_cold(rho, mat_id, N):
    """Compute the cold internal energy from the density.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    N : int
        Number of subdivisions for the numerical integral.

    Returns
    -------
    u_cold : float
        Cold specific internal energy (J kg^-1).
    """
    assert rho >= 0
    mat_type = mat_id // gv.type_factor
    if mat_type == gv.type_Til:
        rho_0 = _rho_0(mat_id)
        drho = (rho - rho_0) / N
        x = rho_0
        u_cold = 1e-9

        for j in range(N):
            x += drho
            u_cold += P_u_rho(u_cold, x, mat_id) * drho / x**2

    else:
        raise ValueError("Invalid material ID")

    return u_cold


@njit
def _create_u_cold_array(mat_id):
    """Compute tabulated values of the cold internal energy.

    Ranges from density = 100 to 100000 kg/m^3

    Parameters
    ----------
    mat_id : int
        Material id.

    Returns
    -------
    u_cold_array : [float]
        Array of cold specific internal energies (J kg^-1).
    """
    N_row = 10000
    u_cold_array = np.zeros((N_row,))
    rho_min = 100
    rho_max = 100000
    N_u_cold = 10000

    rho = rho_min
    drho = (rho_max - rho_min) / (N_row - 1)

    for i in range(N_row):
        u_cold_array[i] = u_cold(rho, mat_id, N_u_cold)
        rho = rho + drho

    return u_cold_array


@njit
def u_cold_tab(rho, mat_id):
    """Compute the cold internal energy using premade tabulated values.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    mat_id : int
        Material id.

    Returns
    -------
    u_cold : float
        Cold specific internal energy (J kg^-1).
    """

    if mat_id == gv.id_Til_iron:
        u_cold_array = A1_u_cold_iron
    elif mat_id == gv.id_Til_granite:
        u_cold_array = A1_u_cold_granite
    elif mat_id == gv.id_Til_basalt:
        u_cold_array = A1_u_cold_basalt
    elif mat_id == gv.id_Til_water:
        u_cold_array = A1_u_cold_water
    else:
        raise ValueError("Invalid material ID")

    # Check necessary data loaded
    if len(u_cold_array) == 1:
        raise ValueError(
            "Please load the corresponding Tillotson table.\n"
            + "Use the woma.load_eos_tables function.\n"
        )

    N_row = u_cold_array.shape[0]
    rho_min = 100
    rho_max = 100000

    drho = (rho_max - rho_min) / (N_row - 1)

    a = int(((rho - rho_min) / drho))
    b = a + 1

    if a >= 0 and a < (N_row - 1):
        u_cold = u_cold_array[a]
        u_cold += ((u_cold_array[b] - u_cold_array[a]) / drho) * (
            rho - rho_min - a * drho
        )

    elif rho < rho_min:
        u_cold = u_cold_array[0]
    else:
        u_cold = u_cold_array[int(N_row - 1)]
        u_cold += (
            (u_cold_array[int(N_row - 1)] - u_cold_array[int(N_row) - 2]) / drho
        ) * (rho - rho_max)

    return u_cold


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

    mat_type = mat_id // gv.type_factor

    if mat_type == gv.type_Til:
        cv = C_V_Til(mat_id)

        u = u_cold_tab(rho, mat_id) + cv * T

    else:
        raise ValueError("Invalid material ID")

    return u


@njit
def P_T_rho(T, rho, mat_id):
    """Compute the pressure from the density and temperature.

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
    u : float
        Specific internal energy (J kg^-1).
    """

    mat_type = mat_id // gv.type_factor

    if mat_type == gv.type_Til:

        cv = C_V_Til(mat_id)
        u = u_cold_tab(rho, mat_id) + cv * T
        P = P_u_rho(u, rho, mat_id)

    else:
        raise ValueError("Invalid material ID")

    return P


@njit
def T_u_rho(u, rho, mat_id):
    """Compute the pressure from the density and temperature.

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

    mat_type = mat_id // gv.type_factor

    if mat_type == gv.type_Til:

        cv = C_V_Til(mat_id)
        u_cold = u_cold_tab(rho, mat_id)
        T = (u - u_cold) / cv

        if T < 0:
            T = 0.0

    else:
        raise ValueError("Invalid material ID")

    return T


@njit
def NR_iter(rho_iter, curve, P, u, material):

    """Carries out one Newton-Raphson iteration.

    Parameters
    ----------
    rho_iter : float
        Previous iteration density (kg m^-3).

    curve : str
        The Tillotson curve that is being considered.

    P : float
        Pressure (Pa).

    u : float
        Specific internal energy (J kg^-1).

    material : (float)
        Tillotson material parameters.

    Returns
    -------
    rho_iter : float
        Next iteration density (kg m^-3).

    P_fraction : float
        (P_iter - P)/P. Measure of how close we are to converging
    """

    (
        rho_0,
        a,
        b,
        A,
        B,
        u_0,
        u_iv,
        u_cv,
        alpha,
        beta,
        eta_min,
        P_min,
        eta_zero,
    ) = material

    eta_iter = rho_iter / rho_0
    eta_iter_sq = eta_iter * eta_iter
    mu_iter = eta_iter - 1.0
    nu_iter = 1.0 / eta_iter - 1.0
    w_iter = u / (u_0 * eta_iter_sq) + 1.0
    w_iter_inv = 1.0 / w_iter
    exp1 = np.exp(-beta * nu_iter)
    exp2 = np.exp(-alpha * nu_iter * nu_iter)

    # Derivatives
    dw_inv_drho_iter = (2 * u_0 * u * eta_iter / rho_0) / (u + u_0 * eta_iter_sq) ** 2
    dmu_drho_iter = 1 / rho_0
    dmu_sq_drho_iter = 2 * rho_iter / rho_0**2 - 2 / rho_0
    dexp1_drho_iter = beta * rho_0 * exp1 / rho_iter**2
    dexp2_drho_iter = 2 * alpha * rho_0 * (rho_0 - rho_iter) * exp2 / rho_iter**3

    if curve == "cold" or curve == "hybrid":
        P_c_iter = (
            (a + b * w_iter_inv) * rho_iter * u
            + A * mu_iter
            + B * mu_iter * mu_iter
            - P
        )
        dP_c_drho_iter = (
            (a + b * w_iter_inv) * u
            + b * u * rho_iter * dw_inv_drho_iter
            + A * dmu_drho_iter
            + B * dmu_sq_drho_iter
        )
        P_fraction = P_c_iter / P

    elif curve == "cold_min" or curve == "hybrid_min":
        P_c_iter = (
            (a + b * w_iter_inv) * rho_iter * u + A * mu_iter + B * mu_iter * mu_iter
        ) * (eta_iter - eta_zero) / (eta_min - eta_zero) - P
        dP_c_drho_iter = (
            (a + b * w_iter_inv) * u
            + b * u * rho_iter * dw_inv_drho_iter
            + A * dmu_drho_iter
            + B * dmu_sq_drho_iter
        ) * (eta_iter - eta_zero) / (eta_min - eta_zero) + (
            (a + b * w_iter_inv) * rho_iter * u + A * mu_iter + B * mu_iter * mu_iter
        ) * (
            1 / (rho_0 * (eta_min - eta_zero))
        )
        P_fraction = P_c_iter / P

    if curve == "hot" or curve == "hybrid" or curve == "hybrid_min":
        P_h_iter = (
            a * rho_iter * u
            + (b * rho_iter * u * w_iter_inv + A * mu_iter * exp1) * exp2
            - P
        )
        dP_h_drho_iter = (
            a * u
            + (
                b * u * w_iter_inv
                + b * u * rho_iter * dw_inv_drho_iter
                + A * mu_iter * dexp1_drho_iter
                + A * exp1 * dmu_drho_iter
            )
            * exp2
            + (b * rho_iter * u * w_iter_inv + A * mu_iter * exp1) * dexp2_drho_iter
        )
        P_fraction = P_h_iter / P

    if curve == "cold" or curve == "cold_min":
        rho_iter -= P_c_iter / dP_c_drho_iter

    elif curve == "hot":
        rho_iter -= P_h_iter / dP_h_drho_iter

    elif curve == "hybrid" or curve == "hybrid_min":
        P_hybrid_iter = ((u - u_iv) * P_h_iter + (u_cv - u) * P_c_iter) / (u_cv - u_iv)
        dP_hybrid_drho_iter = (
            (u - u_iv) * dP_h_drho_iter + (u_cv - u) * dP_c_drho_iter
        ) / (u_cv - u_iv)
        rho_iter -= P_hybrid_iter / dP_hybrid_drho_iter
        P_fraction = P_hybrid_iter / P

    return rho_iter, P_fraction


@njit
def curve_finder(eta, eta_min, u, u_iv, u_cv):
    """Finds possible curves given u. Orders based on likelihood of valid root, given eta.

     Parameters
     ----------
     eta : float
         Density / rho_0.

      u : float
         Specific internal energy (J kg^-1).

      eta_min, u_iv, u_cv : floats
         Tillotson material parameters.

     Returns
     -------
     possible_curves : [str]
         Ordered list of possible curves. Possible curves are given by:

     u REGION 1
     u < u_iv:
         eta < eta_min:           cold_min
         eta_min < eta:           cold

    u REGION 2
    u_iv < u < u_cv
         eta < eta_min:           hybrid_min
         eta_min < eta < 1:       hybrid
         1 < eta:                 cold

     u REGION 3
     u_cv < u
         eta < 1:                 hot
         1 < eta:                 cold

     NOTE: for a lot of EoS, eta_min = 0, so search this region last if given the option to save time for most EoS
    """

    # u REGION 1
    if u <= u_iv:
        if eta <= eta_min:
            possible_curves = ["cold_min", "cold"]
        else:
            possible_curves = ["cold", "cold_min"]
    # u REGION 2
    elif u <= u_cv:
        if eta <= eta_min:
            possible_curves = ["hybrid_min", "hybrid", "cold"]
        elif eta <= 1:
            possible_curves = ["hybrid", "cold", "hybrid_min"]
        else:
            possible_curves = ["cold", "hybrid", "hybrid_min"]

    # u REGION 3
    else:
        if eta <= 1:
            possible_curves = ["hot", "cold"]
        else:
            possible_curves = ["cold", "hot"]

    return possible_curves


@njit
def constrain_to_curve(rho, curve, rho_0, eta_min, u, u_iv, u_cv):
    """Constrains rho to the valid region of a curve.

    Parameters
    ----------
    rho : float
        Density (kg m^-3).

    curve : str
        The Tillotson curve that is being considered.

    rho_0, eta_min, u, u_iv, u_cv : floats
        Tillotson material parameters.

    Returns
    -------
    rho : float
        Density (kg m^-3) constrained to curve.
    """

    # u REGION 1
    if u <= u_iv:
        if curve == "cold_min":
            rho = min(rho, eta_min * rho_0)
        elif curve == "cold":
            rho = max(rho, eta_min * rho_0)
    # u REGION 2
    elif u <= u_cv:
        if curve == "hybrid_min":
            rho = min(rho, eta_min * rho_0)
        elif curve == "hybrid":
            rho = max(rho, eta_min * rho_0)
            rho = min(rho, rho_0)
        elif curve == "cold":
            rho = max(rho, rho_0)

    # u REGION 3
    else:
        if curve == "hot":
            rho = min(rho, rho_0)
        elif curve == "cold":
            rho = max(rho, rho_0)

    return rho


@njit
def rho_u_P(u, P, mat_id, rho_ref):
    """Compute the density from the internal energy and pressure.

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

    # Material constants for Tillotson EoS (SI)
    # mat_id, rho_0, a, b, A, B, u_0, u_iv, u_cv, alpha, beta, eta_min, P_min, eta_zero
    iron = np.array(
        [
            gv.id_Til_iron,
            7800,
            0.5,
            1.5,
            1.28e11,
            1.05e11,
            9.5e6,
            2.4e6,
            8.67e6,
            5,
            5,
            0,
            0,
            0,
        ]
    )
    granite = np.array(
        [
            gv.id_Til_granite,
            2680,
            0.5,
            1.3,
            1.8e10,
            1.8e10,
            1.6e7,
            3.5e6,
            1.8e7,
            5,
            5,
            0,
            0,
            0,
        ]
    )
    basalt = np.array(
        [
            gv.id_Til_basalt,
            2700,
            0.5,
            1.5,
            2.67e10,
            2.67e10,
            4.87e8,
            4.72e6,
            1.82e7,
            5,
            5,
            0,
            0,
            0,
        ]
    )
    water = np.array(
        [
            gv.id_Til_water,
            998,
            0.7,
            0.15,
            2.18e9,
            1.325e10,
            7.0e6,
            4.19e5,
            2.69e6,
            10,
            5,
            0.925,
            0,
            0.875,
        ]
    )

    if mat_id == gv.id_Til_iron:
        material = iron
    elif mat_id == gv.id_Til_granite:
        material = granite
    elif mat_id == gv.id_Til_basalt:
        material = basalt
    elif mat_id == gv.id_Til_water:
        material = water
    else:
        raise ValueError("Invalid material ID")

    # Unpack the parameters
    (
        rho_0,
        a,
        b,
        A,
        B,
        u_0,
        u_iv,
        u_cv,
        alpha,
        beta,
        eta_min,
        P_min,
        eta_zero,
    ) = material[1:]

    assert u_iv <= u_cv

    if P <= P_min or u == 0:
        return rho_ref

    # We start search on the same curve as rho_ref, since this is most likely curve to find rho on
    eta_ref = rho_ref / rho_0

    # Given u, what are the possible curves we can be on?
    # The output is ordered based on how likely the root is to be on that curve given eta_ref
    possible_curves = curve_finder(eta_ref, eta_min, u, u_iv, u_cv)

    # Newton-Raphson
    max_iter = 10
    tol = 1e-5

    # Loop over all possible curves until we find a valid root
    for i, curve in enumerate(possible_curves):

        # Our first rho_iter must be constrained to the valid region of the curve
        rho_iter = constrain_to_curve(rho_ref, curve, rho_0, eta_min, u, u_iv, u_cv)
        # set this to an arbitrary number so we definitely dont't think we converge straigt away
        last_rho_iter = -1e5
        for j in range(max_iter):

            # Carry out a Newton-Raphson iteration
            rho_iter, P_fraction = NR_iter(rho_iter, curve, P, u, material[1:])
            # Constrain the next iteration density to the valid region of the curve
            rho_iter = constrain_to_curve(
                rho_iter, curve, rho_0, eta_min, u, u_iv, u_cv
            )

            # Either we've converged ...
            if np.abs(P_fraction) < tol:
                return rho_iter
            # ... or we're stuck at the boundary ...
            elif rho_iter == last_rho_iter:
                break

            # ... or we loop again
            last_rho_iter = rho_iter

    raise ValueError("rho_u_P failed to converge")
    return rho_ref
