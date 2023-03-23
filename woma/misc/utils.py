"""
WoMa miscellaneous utilities
"""

import numpy as np
from numba import njit
from woma.misc.glob_vars import G
from woma.misc import glob_vars as gv
from woma.eos import eos
import sys
from importlib import reload


def _print_banner():
    print("\n")
    print("#  WoMa - World Maker")
    print("#  sergio.ruiz-bonilla@durham.ac.uk")
    print("\n")


def check_end(string, end):
    """Check that a string ends with the required characters, append them if not.

    Parameters
    ----------
    string : str
        The string to check.

    end : str
        The required ending.

    Returns
    -------
    string : str
        The string with the required ending.
    """
    if string[-len(end) :] != end:
        string += end

    return string


def format_array_string(array, format):
    """Return a print-ready string of an array's contents in a given format.

    Parameters
    ----------
    array :[e.g. float, int, str]
        An array of values that can be printed with the given format.

    format : str
        A printing format, e.g. "%d", "%.5g".

        Custom options:
        "string":   Include quotation marks around each string.
        "dorf":     Int or float if decimal places are needed.

    Returns
    -------
    string : str
        The formatted string.
    """
    string = ""

    # Append each element
    # 1D
    if len(np.shape(array)) == 1:
        for x in array:
            if x is None:
                string += "None, "
            else:
                # Custom formats
                if format == "string":
                    string += '"%s", ' % x
                elif format == "dorf":
                    string += "{0}, ".format(str(round(x, 1) if x % 1 else int(x)))
                # Standard formats
                else:
                    string += "%s, " % (format % x)
    # Recursive for higher dimensions
    else:
        for arr in array:
            string += "%s, " % format_array_string(arr, format)

    # Add brackets and remove the trailing comma
    return "[%s]" % string[:-2]


def add_whitespace(string, space):
    """Return a padded string for aligned printing with adjusted spaces.

    e.g.
        >>> asdf = 123
        >>> qwerty = 456
        >>> print("%s = %d \n""%s = %d" %
                  (add_whitespace("asdf", 12), asdf,
                   add_whitespace("qwerty", 12), qwerty))
        asdf         = 123
        qwerty       = 456

    Parameters
    ----------
    string : str
        The input string.

    space : int
        The required length for alignment.

    Returns
    -------
    string : str
        The padded string.
    """
    return "%s" % string + " " * (space - len("%s" % string))


@njit
def moi(A1_r, A1_rho):
    """Compute the moment of inertia for a planet with spherical symmetry.

    Parameters
    ----------
    A1_r : [float]
        Radii of the planet (m).

    A1_rho : [float]
        Densities at each radii (kg m^-3).

    Returns
    ----------
    MoI : float
        Moment of inertia (kg m^2).
    """
    dr = np.abs(A1_r[0] - A1_r[1])
    r4 = np.power(A1_r, 4)
    MoI = 2 * np.pi * (4 / 3) * np.sum(r4 * A1_rho) * dr

    return MoI


class Conversions:
    """Simple conversions from one set of units to another, derived using the
        base mass-, length-, and time-unit relations.

    Usage e.g.
    ----------
    cgs_to_SI   = Conversions(1e-3, 1e-2, 1)
    SI_to_cgs   = cgs_to_SI.inv()

    rho_SI  = rho_cgs * cgs_to_SI.rho
    G_cgs   = 6.67e-11 * SI_to_cgs.G

    Parameters
    ----------
    m : float
       Value to convert mass from the first units to the second.

    l : float
       Value to convert length from the first units to the second.

    t : float
       Value to convert time from the first units to the second.

    Attributes (all : float)
    m           Mass
    l           Length
    t           Time
    v           Velocity
    a           Acceleration
    rho         Density
    drho_dt     Rate of change of density
    P           Pressure
    u           Specific energy
    du_dt       Rate of change of specific energy
    E           Energy
    s           Specific entropy
    G           Gravitational constant
    """

    def __init__(self, m, l, t):
        # Input conversions
        self.m = m
        self.l = l
        self.t = t
        # Derived conversions
        self.v = l * t**-1
        self.a = l * t**-2
        self.rho = m * l**-3
        self.drho_dt = m * l**-4
        self.P = m * l**-1 * t**-2
        self.u = l**2 * t**-2
        self.du_dt = l**2 * t**-3
        self.E = m * l**2 * t**-2
        self.s = l**2 * t**-2
        self.G = m**-1 * l**3 * t**-2

    def inv(self):
        """Return the inverse to this conversion"""
        return Conversions(1 / self.m, 1 / self.l, 1 / self.t)


# Standard unit conversions
SI_to_SI = Conversions(1, 1, 1)  # No-op
cgs_to_SI = Conversions(1e-3, 1e-2, 1)
SI_to_cgs = cgs_to_SI.inv()


def impact_pos_vel_b_v_c_r(
    b, v_c, r, R_t, R_i, M_t, M_i, units_b="b", units_v_c="m/s", return_t=False
):
    """
    Calculate the intial position and velocity of an impactor to result in the
    desired scenario at contact, rotated such that the velocity at contact is
    in the negative x direction.

    As described in Appendix A of in Kegerreis et al. (2020) ApJ 897:161, with
    the erroneous factors of 2 removed from Eq.(A8) to set the rotation angle.

    Parameters
    ----------
    b : float
        The impact parameter b=sin(B), or the impact angle B if units_b is "B".

    v_c : float
        The impactor's speed at contact, in units given by units_v_c.
        e.g. v_c = 1 v_esc --> v = 0 at infinity and a parabolic orbit.

    r : float
        The initial distance between body centres (m).

    R_t, R_i : float
        The radii of the target and impactor (m).

    M_t, M_i : float
        The masses of the target and impactor (kg).

    units_b : str (opt.)
        The units of the impact parameter/angle: "b" (default) for the
        dimensionless impact parameter, or "B" for the impact angle in degrees.

    units_v_c : str (opt.)
        The units of the contact speed: "m/s" (default), "v_esc" for the
        mutual escape speed, or "v_inf" for the velocity at infinity.

    return_t : bool (opt.)
        If True, then also return the time until contact, e.g. to be used by
        impact_pos_vel_b_v_c_t().

    Returns
    -------
    A1_pos : [float]
        The impactor's initial x, y, z coordinates (m).

    A1_vel : [float]
        The impactor's initial x, y, z velocities (m s^-1).

    t : float (opt.)
        If return_t is True, then also return the time taken from the
        initial position until contact (s).
    """
    mu = G * (M_t + M_i)
    v_esc = np.sqrt(2 * mu / (R_t + R_i))
    r_c = R_t + R_i

    # Convert to b and v_c (m/s) if necessary
    if units_b == "b":
        pass
    elif units_b == "B":
        b = np.sin(b * np.pi / 180)
    else:
        raise ValueError("Invalid units_b:", units_b)
    if units_v_c == "m/s":
        pass
    elif units_v_c == "v_esc":
        v_c *= v_esc
    elif units_v_c == "v_inf":
        v_inf = v_c
        v_c = np.sqrt(v_inf**2 + 2 * mu / r_c)
    else:
        raise ValueError("Invalid units_v_c:", units_v_c)

    # Contact position
    y_c = b * r_c
    if r < r_c:
        raise ValueError(
            "Invalid r = %g m for body radii %g + %g = %g m" % (r, R_t, R_i, r_c)
        )

    # Parabola
    if v_c == v_esc:
        # Initial speed and position
        v = np.sqrt(2 * mu / r)
        y = v_c * y_c / v
        x = np.sqrt(r**2 - y**2)

        # True anomalies (actually the complementary angles)
        theta_c = np.pi - np.arccos(y_c**2 * v_c**2 / (mu * r_c) - 1)
        theta = np.pi - np.arccos(y_c**2 * v_c**2 / (mu * r) - 1)

        # Velocity angle at contact
        alpha_ = theta_c / 2
    # Ellipse or hyperbola
    else:
        # Semimajor axis
        a = 1 / (2 / r_c - v_c**2 / mu)

        # Initial speed and position
        v = np.sqrt(mu * (2 / r - 1 / a))
        y = v_c * y_c / v
        x = np.sqrt(r**2 - y**2)

        # Periapsis and eccentricity
        r_p = min(
            abs(a + np.sqrt(a**2 - a * v_c**2 * y_c**2 / mu)),
            abs(a - np.sqrt(a**2 - a * v_c**2 * y_c**2 / mu)),
        )
        e = 1 - r_p / a

        # Check requested separation is valid
        r_a = 2 * a - r_p
        if v_c < v_esc and r > r_a:
            raise Exception(
                "Invalid r = %g m for bound orbit (v_esc = %g m/s) with "
                "apoapsis = %g m, period/2 = %.1f s"
                % (r, v_esc, r_a, 0.5 * np.sqrt(a**3 * 4 * np.pi**2 / mu))
            )

        # True anomalies (actually the complementary angles)
        theta_c = np.arccos((1 - a * (1 - e**2) / r_c) / e)
        theta = np.arccos((1 - a * (1 - e**2) / r) / e)

        # Velocity angle at contact
        alpha_ = np.arcsin(np.sqrt(a**2 * (1 - e**2) / (2 * a * r_c - r_c**2)))

    # Rotate
    if b == 0:
        phi = 0
    else:
        # A=sin^-1(y/r), B=theta-A, C=theta_c-B, D=pi-C, phi=pi-alpha_-D
        phi = alpha_ - theta_c + theta - np.arcsin(y / r)

    x_ = x * np.cos(phi) - y * np.sin(phi)
    y_ = x * np.sin(phi) + y * np.cos(phi)
    v_x_ = -abs(v * np.cos(phi))
    v_y_ = abs(v * np.sin(phi))

    # Time until contact, if requested
    if return_t:
        # Radial
        if b == 0:
            # Parabolic
            if v_c == v_esc:
                # Time until point-masses contact
                t = np.sqrt(2 * r**3 / (9 * mu))
                t_c = np.sqrt(2 * r_c**3 / (9 * mu))
            # Elliptical
            elif 0 < a:
                # Standard constants
                w = 1 / r - v**2 / (2 * mu)
                w_c = 1 / r_c - v_c**2 / (2 * mu)
                wr = w * r
                wr_c = w_c * r_c
                # Time until point-masses contact
                t = (np.arcsin(np.sqrt(wr)) - np.sqrt(wr * (1 - wr))) / np.sqrt(
                    2 * mu * w**3
                )
                t_c = (np.arcsin(np.sqrt(wr_c)) - np.sqrt(wr_c * (1 - wr_c))) / np.sqrt(
                    2 * mu * w_c**3
                )
            # Hyperbolic
            else:
                # Standard constants
                w = abs(1 / r - v**2 / (2 * mu))
                w_c = abs(1 / r_c - v_c**2 / (2 * mu))
                wr = w * r
                wr_c = w_c * r_c
                # Time until point-masses contact
                t = (np.sqrt(wr**2 + wr) - np.log(np.sqrt(wr) + np.sqrt(1 + wr))) / (
                    np.sqrt(2 * mu * w**3)
                )
                t_c = (
                    np.sqrt(wr_c**2 + wr_c)
                    - np.log(np.sqrt(wr_c) + np.sqrt(1 + wr_c))
                ) / (np.sqrt(2 * mu * w**3))
        # Not radial
        else:
            # Parabolic
            if v_c == v_esc:
                # Eccentric anomaly
                E = np.tan(0.5 * (np.pi - theta))
                E_c = np.tan(0.5 * (np.pi - theta_c))
                # Mean anomaly
                M = E + E**3 / 3
                M_c = E_c + E_c**3 / 3
                # Periapsis
                r_p = mu * (1 + np.cos(np.pi - theta)) / v**2
                # Time until periapsis
                t = np.sqrt(2 * r_p**3 / mu) * M
                t_c = np.sqrt(2 * r_p**3 / mu) * M_c
            # Elliptical
            elif 0 < a:
                # Eccentric anomaly
                E = np.arccos(
                    (e + np.cos(np.pi - theta)) / (1 + e * np.cos(np.pi - theta))
                )
                E_c = np.arccos(
                    (e + np.cos(np.pi - theta_c)) / (1 + e * np.cos(np.pi - theta_c))
                )
                # Mean anomaly
                M = E - e * np.sin(E)
                M_c = E_c - e * np.sin(E_c)
                # Time until periapsis
                t = np.sqrt(a**3 / mu) * M
                t_c = np.sqrt(a**3 / mu) * M_c
            # Hyperbolic
            else:
                # Eccentric anomaly
                E = np.arccosh(
                    (e + np.cos(np.pi - theta)) / (1 + e * np.cos(np.pi - theta))
                )
                E_c = np.arccosh(
                    (e + np.cos(np.pi - theta_c)) / (1 + e * np.cos(np.pi - theta_c))
                )
                # Mean anomaly
                M = -E + e * np.sinh(E)
                M_c = -E_c + e * np.sinh(E_c)
                # Time until periapsis
                t = np.sqrt(-(a**3) / mu) * M
                t_c = np.sqrt(-(a**3) / mu) * M_c

        return np.array([x_, y_, 0]), np.array([v_x_, v_y_, 0]), t - t_c
    else:
        return np.array([x_, y_, 0]), np.array([v_x_, v_y_, 0])


def impact_pos_vel_b_v_c_t(
    b, v_c, t, R_t, R_i, M_t, M_i, units_b="b", units_v_c="m/s", r_max_factor=100
):
    """
    Calculate the intial position and velocity of an impactor to result in the
    desired scenario at contact, rotated such that the velocity at contact is
    in the negative x direction.

    Find the initial distance between the body centres, r, that yields the
    desired time until contact by iteratively calling impact_pos_vel_b_v_c_r().

    As described in Appendix A of in Kegerreis et al. (2020) ApJ 897:161.

    Parameters
    ----------
    b : float
        The impact parameter, b = sin(B), or the impact angle B (degrees) if
        units_b is "B".

    v_c : float
        The impactor's speed at contact, in units given by units_v_c.
        e.g. v_c = 1 v_esc --> v = 0 at infinity and a parabolic orbit.

    t : float
        The time taken from the initial position until contact (s).

    R_t, R_i : float
        The radii of the target and impactor (m).

    M_t, M_i : float
        The masses of the target and impactor (kg).

    units_b : str (opt.)
        The units of the impact parameter/angle: "b" (default) for the
        dimensionless impact parameter, or "B" for the impact angle in degrees.

    units_v_c : str (opt.)
        The units of the contact speed: "m/s" (default), "v_esc" for the
        mutual escape speed, or "v_inf" for the velocity at infinity.

    r_max_factor : float (opt.)
        This times the sum of the body radii sets the maximum initial
        separation of the body centres for the bisection search. Default 100.

    Returns
    -------
    A1_pos : [float]
        The impactor's initial x, y, z coordinates (m).

    A1_vel : [float]
        The impactor's initial x, y, z velocities (m s^-1).
    """
    # Boundary guesses for the initial separation
    r_min = R_t + R_i
    r_max = r_min * r_max_factor

    # Bisection to find the separation to give the desired time to impact
    i = 0
    i_max = 1e2
    t_ = 0
    tol = 1e-6
    while tol < abs(t_ - t) / t:
        r = 0.5 * (r_min + r_max)
        i += 1

        # Catch if r too big for a bound orbit
        try:
            t_ = impact_pos_vel_b_v_c_r(
                b,
                v_c,
                r,
                R_t,
                R_i,
                M_t,
                M_i,
                units_b=units_b,
                units_v_c=units_v_c,
                return_t=True,
            )[2]
        except Exception:
            # Reduce r next time
            t_ = t * 2
            # Raise the error anyway if out of iterations
            if i >= i_max:
                raise

        # Bisect
        if t_ < t:
            r_min = r
        else:
            r_max = r

        if i >= i_max:
            raise RuntimeError("Failed to find r(t) after %d iterations" % (i))

    return impact_pos_vel_b_v_c_r(
        b, v_c, r, R_t, R_i, M_t, M_i, units_b=units_b, units_v_c=units_v_c
    )


def check_loaded_eos_tables():
    """###"""
    A1_mat = gv.Di_mat_id.keys()
    A1_mat = list(A1_mat)

    # discard idg materials
    A1_idg = []
    for material in A1_mat:
        if material.startswith("idg"):
            A1_idg.append(material)
    for material in A1_idg:
        A1_mat.remove(material)

    # Check Tillotson
    if len(eos.tillotson.A1_u_cold_iron) == 1:
        A1_mat.remove("Til_iron")
    if len(eos.tillotson.A1_u_cold_granite) == 1:
        A1_mat.remove("Til_granite")
    if len(eos.tillotson.A1_u_cold_basalt) == 1:
        A1_mat.remove("Til_basalt")
    if len(eos.tillotson.A1_u_cold_water) == 1:
        A1_mat.remove("Til_water")

    # Check HM80
    if len(eos.hm80.A2_log_P_HM80_HHe) == 1:
        A1_mat.remove("HM80_HHe")
    if len(eos.hm80.A2_log_P_HM80_ice) == 1:
        A1_mat.remove("HM80_ice")
    if len(eos.hm80.A2_log_P_HM80_rock) == 1:
        A1_mat.remove("HM80_rock")

    # Check SESAME
    if len(eos.sesame.A1_rho_SESAME_iron) == 1:
        A1_mat.remove("SESAME_iron")
    if len(eos.sesame.A1_rho_SESAME_basalt) == 1:
        A1_mat.remove("SESAME_basalt")
    if len(eos.sesame.A1_rho_SESAME_water) == 1:
        A1_mat.remove("SESAME_water")
    if len(eos.sesame.A1_rho_SS08_water) == 1:
        A1_mat.remove("SS08_water")

    if len(eos.sesame.A1_rho_ANEOS_forsterite) == 1:
        A1_mat.remove("ANEOS_forsterite")
    if len(eos.sesame.A1_rho_ANEOS_iron) == 1:
        A1_mat.remove("ANEOS_iron")
    if len(eos.sesame.A1_rho_ANEOS_Fe85Si15) == 1:
        A1_mat.remove("ANEOS_Fe85Si15")

    if len(eos.sesame.A1_rho_AQUA) == 1:
        A1_mat.remove("AQUA")

    if len(eos.sesame.A1_rho_CMS19_H) == 1:
        A1_mat.remove("CMS19_H")
    if len(eos.sesame.A1_rho_CMS19_He) == 1:
        A1_mat.remove("CMS19_He")
    if len(eos.sesame.A1_rho_CD21_HHe) == 1:
        A1_mat.remove("CD21_HHe")

    # Check custom
    if len(eos.sesame.A1_rho_custom_0) == 1:
        A1_mat.remove("custom_0")
    if len(eos.sesame.A1_rho_custom_1) == 1:
        A1_mat.remove("custom_1")
    if len(eos.sesame.A1_rho_custom_2) == 1:
        A1_mat.remove("custom_2")
    if len(eos.sesame.A1_rho_custom_3) == 1:
        A1_mat.remove("custom_3")
    if len(eos.sesame.A1_rho_custom_4) == 1:
        A1_mat.remove("custom_4")

    return A1_mat


def load_eos_tables(A1_mat_input=None):
    """Load necessary tables for eos computations.

    Parameters
    ----------
    A1_mat_input : [str] or str
        List of the materials (or just one) to be loaded. Default None loads all
        materials available. See Di_mat_id in `misc/glob_vars.py`.
    """
    # Load all tables (default) (except custom)
    if A1_mat_input is None:
        A1_mat = list(gv.Di_mat_id.keys())
        for material in ["custom_%d" % i for i in range(5)]:
            A1_mat.remove(material)
    elif not hasattr(A1_mat_input, "copy"):
        # Put a single input into a list
        A1_mat_input = [A1_mat_input]
        A1_mat = A1_mat_input.copy()
    else:
        A1_mat = A1_mat_input.copy()
    # Discard idg materials
    A1_idg = []
    for material in A1_mat:
        if material.startswith("idg"):
            A1_idg.append(material)
    for material in A1_idg:
        A1_mat.remove(material)

    # Check A1_mat elements are available eos
    for material in A1_mat:
        if material not in gv.Di_mat_id.keys():
            raise ValueError(
                "%s not available. Check misc/glob_vars.py for available eos."
                % (material)
            )

    # Check if tables are already loaded
    A1_mat_loaded = check_loaded_eos_tables()

    A1_mat_loaded = sorted(A1_mat_loaded)
    A1_mat = sorted(A1_mat)
    if A1_mat_loaded == A1_mat:
        return None
    if all(x in A1_mat_loaded for x in A1_mat):
        return None

    # print("Loading eos tables...")

    # Reload woma modules, need to recompile for numba
    to_reload = []
    for k, v in sys.modules.items():
        if "woma" in k:
            to_reload.append(k)
    for k in to_reload:
        reload(sys.modules[k])

    # Tillotson
    if "Til_iron" in A1_mat and len(eos.tillotson.A1_u_cold_iron) == 1:
        eos.tillotson.A1_u_cold_iron = eos.tillotson.load_u_cold_array(gv.id_Til_iron)
    if "Til_granite" in A1_mat and len(eos.tillotson.A1_u_cold_granite) == 1:
        eos.tillotson.A1_u_cold_granite = eos.tillotson.load_u_cold_array(
            gv.id_Til_granite
        )
    if "Til_basalt" in A1_mat and len(eos.tillotson.A1_u_cold_basalt) == 1:
        eos.tillotson.A1_u_cold_basalt = eos.tillotson.load_u_cold_array(
            gv.id_Til_basalt
        )
    if "Til_water" in A1_mat and len(eos.tillotson.A1_u_cold_water) == 1:
        eos.tillotson.A1_u_cold_water = eos.tillotson.load_u_cold_array(gv.id_Til_water)

    # Hubbard & MacFarlane (1980) Uranus/Neptune
    if "HM80_HHe" in A1_mat and len(eos.hm80.A2_log_P_HM80_HHe) == 1:
        (
            eos.hm80.log_rho_min_HM80_HHe,
            eos.hm80.log_rho_max_HM80_HHe,
            eos.hm80.num_rho_HM80_HHe,
            eos.hm80.log_rho_step_HM80_HHe,
            eos.hm80.log_u_min_HM80_HHe,
            eos.hm80.log_u_max_HM80_HHe,
            eos.hm80.num_u_HM80_HHe,
            eos.hm80.log_u_step_HM80_HHe,
            eos.hm80.A2_log_P_HM80_HHe,
            eos.hm80.A2_log_T_HM80_HHe,
        ) = eos.hm80.load_table_HM80(gv.Fp_HM80_HHe)
    if "HM80_ice" in A1_mat and len(eos.hm80.A2_log_P_HM80_ice) == 1:
        eos.hm80.A1_u_cold_HM80_ice = eos.hm80.load_u_cold_array(gv.id_HM80_ice)
        (
            eos.hm80.log_rho_min_HM80_ice,
            eos.hm80.log_rho_max_HM80_ice,
            eos.hm80.num_rho_HM80_ice,
            eos.hm80.log_rho_step_HM80_ice,
            eos.hm80.log_u_min_HM80_ice,
            eos.hm80.log_u_max_HM80_ice,
            eos.hm80.num_u_HM80_ice,
            eos.hm80.log_u_step_HM80_ice,
            eos.hm80.A2_log_P_HM80_ice,
            eos.hm80.A2_log_T_HM80_ice,
        ) = eos.hm80.load_table_HM80(gv.Fp_HM80_ice)
    if "HM80_rock" in A1_mat and len(eos.hm80.A2_log_P_HM80_rock) == 1:
        eos.hm80.A1_u_cold_HM80_rock = eos.hm80.load_u_cold_array(gv.id_HM80_rock)
        (
            eos.hm80.log_rho_min_HM80_rock,
            eos.hm80.log_rho_max_HM80_rock,
            eos.hm80.num_rho_HM80_rock,
            eos.hm80.log_rho_step_HM80_rock,
            eos.hm80.log_u_min_HM80_rock,
            eos.hm80.log_u_max_HM80_rock,
            eos.hm80.num_u_HM80_rock,
            eos.hm80.log_u_step_HM80_rock,
            eos.hm80.A2_log_P_HM80_rock,
            eos.hm80.A2_log_T_HM80_rock,
        ) = eos.hm80.load_table_HM80(gv.Fp_HM80_rock)

    # SESAME
    if "SESAME_iron" in A1_mat and len(eos.sesame.A1_rho_SESAME_iron) == 1:
        (
            eos.sesame.A1_rho_SESAME_iron,
            eos.sesame.A1_T_SESAME_iron,
            eos.sesame.A2_u_SESAME_iron,
            eos.sesame.A2_P_SESAME_iron,
            eos.sesame.A2_c_SESAME_iron,
            eos.sesame.A2_s_SESAME_iron,
            eos.sesame.A1_log_rho_SESAME_iron,
            eos.sesame.A1_log_T_SESAME_iron,
            eos.sesame.A2_log_u_SESAME_iron,
            eos.sesame.A2_log_P_SESAME_iron,
            eos.sesame.A2_log_c_SESAME_iron,
            eos.sesame.A2_log_s_SESAME_iron,
        ) = eos.sesame.load_table_SESAME(gv.Fp_SESAME_iron)
    if "SESAME_basalt" in A1_mat and len(eos.sesame.A1_rho_SESAME_basalt) == 1:
        (
            eos.sesame.A1_rho_SESAME_basalt,
            eos.sesame.A1_T_SESAME_basalt,
            eos.sesame.A2_u_SESAME_basalt,
            eos.sesame.A2_P_SESAME_basalt,
            eos.sesame.A2_c_SESAME_basalt,
            eos.sesame.A2_s_SESAME_basalt,
            eos.sesame.A1_log_rho_SESAME_basalt,
            eos.sesame.A1_log_T_SESAME_basalt,
            eos.sesame.A2_log_u_SESAME_basalt,
            eos.sesame.A2_log_P_SESAME_basalt,
            eos.sesame.A2_log_c_SESAME_basalt,
            eos.sesame.A2_log_s_SESAME_basalt,
        ) = eos.sesame.load_table_SESAME(gv.Fp_SESAME_basalt)
    if "SESAME_water" in A1_mat and len(eos.sesame.A1_rho_SESAME_water) == 1:
        (
            eos.sesame.A1_rho_SESAME_water,
            eos.sesame.A1_T_SESAME_water,
            eos.sesame.A2_u_SESAME_water,
            eos.sesame.A2_P_SESAME_water,
            eos.sesame.A2_c_SESAME_water,
            eos.sesame.A2_s_SESAME_water,
            eos.sesame.A1_log_rho_SESAME_water,
            eos.sesame.A1_log_T_SESAME_water,
            eos.sesame.A2_log_u_SESAME_water,
            eos.sesame.A2_log_P_SESAME_water,
            eos.sesame.A2_log_c_SESAME_water,
            eos.sesame.A2_log_s_SESAME_water,
        ) = eos.sesame.load_table_SESAME(gv.Fp_SESAME_water)
    if "SS08_water" in A1_mat and len(eos.sesame.A1_rho_SS08_water) == 1:
        (
            eos.sesame.A1_rho_SS08_water,
            eos.sesame.A1_T_SS08_water,
            eos.sesame.A2_u_SS08_water,
            eos.sesame.A2_P_SS08_water,
            eos.sesame.A2_c_SS08_water,
            eos.sesame.A2_s_SS08_water,
            eos.sesame.A1_log_rho_SS08_water,
            eos.sesame.A1_log_T_SS08_water,
            eos.sesame.A2_log_u_SS08_water,
            eos.sesame.A2_log_P_SS08_water,
            eos.sesame.A2_log_c_SS08_water,
            eos.sesame.A2_log_s_SS08_water,
        ) = eos.sesame.load_table_SESAME(gv.Fp_SS08_water)
    if "ANEOS_forsterite" in A1_mat and len(eos.sesame.A1_rho_ANEOS_forsterite) == 1:
        (
            eos.sesame.A1_rho_ANEOS_forsterite,
            eos.sesame.A1_T_ANEOS_forsterite,
            eos.sesame.A2_u_ANEOS_forsterite,
            eos.sesame.A2_P_ANEOS_forsterite,
            eos.sesame.A2_c_ANEOS_forsterite,
            eos.sesame.A2_s_ANEOS_forsterite,
            eos.sesame.A1_log_rho_ANEOS_forsterite,
            eos.sesame.A1_log_T_ANEOS_forsterite,
            eos.sesame.A2_log_u_ANEOS_forsterite,
            eos.sesame.A2_log_P_ANEOS_forsterite,
            eos.sesame.A2_log_c_ANEOS_forsterite,
            eos.sesame.A2_log_s_ANEOS_forsterite,
        ) = eos.sesame.load_table_SESAME(gv.Fp_ANEOS_forsterite)
    if "ANEOS_iron" in A1_mat and len(eos.sesame.A1_rho_ANEOS_iron) == 1:
        (
            eos.sesame.A1_rho_ANEOS_iron,
            eos.sesame.A1_T_ANEOS_iron,
            eos.sesame.A2_u_ANEOS_iron,
            eos.sesame.A2_P_ANEOS_iron,
            eos.sesame.A2_c_ANEOS_iron,
            eos.sesame.A2_s_ANEOS_iron,
            eos.sesame.A1_log_rho_ANEOS_iron,
            eos.sesame.A1_log_T_ANEOS_iron,
            eos.sesame.A2_log_u_ANEOS_iron,
            eos.sesame.A2_log_P_ANEOS_iron,
            eos.sesame.A2_log_c_ANEOS_iron,
            eos.sesame.A2_log_s_ANEOS_iron,
        ) = eos.sesame.load_table_SESAME(gv.Fp_ANEOS_iron)
    if "ANEOS_Fe85Si15" in A1_mat and len(eos.sesame.A1_rho_ANEOS_Fe85Si15) == 1:
        (
            eos.sesame.A1_rho_ANEOS_Fe85Si15,
            eos.sesame.A1_T_ANEOS_Fe85Si15,
            eos.sesame.A2_u_ANEOS_Fe85Si15,
            eos.sesame.A2_P_ANEOS_Fe85Si15,
            eos.sesame.A2_c_ANEOS_Fe85Si15,
            eos.sesame.A2_s_ANEOS_Fe85Si15,
            eos.sesame.A1_log_rho_ANEOS_Fe85Si15,
            eos.sesame.A1_log_T_ANEOS_Fe85Si15,
            eos.sesame.A2_log_u_ANEOS_Fe85Si15,
            eos.sesame.A2_log_P_ANEOS_Fe85Si15,
            eos.sesame.A2_log_c_ANEOS_Fe85Si15,
            eos.sesame.A2_log_s_ANEOS_Fe85Si15,
        ) = eos.sesame.load_table_SESAME(gv.Fp_ANEOS_Fe85Si15)
    if "AQUA" in A1_mat and len(eos.sesame.A1_rho_AQUA) == 1:
        (
            eos.sesame.A1_rho_AQUA,
            eos.sesame.A1_T_AQUA,
            eos.sesame.A2_u_AQUA,
            eos.sesame.A2_P_AQUA,
            eos.sesame.A2_c_AQUA,
            eos.sesame.A2_s_AQUA,
            eos.sesame.A1_log_rho_AQUA,
            eos.sesame.A1_log_T_AQUA,
            eos.sesame.A2_log_u_AQUA,
            eos.sesame.A2_log_P_AQUA,
            eos.sesame.A2_log_c_AQUA,
            eos.sesame.A2_log_s_AQUA,
        ) = eos.sesame.load_table_SESAME(gv.Fp_AQUA)
    if "CMS19_H" in A1_mat and len(eos.sesame.A1_rho_CMS19_H) == 1:
        (
            eos.sesame.A1_rho_CMS19_H,
            eos.sesame.A1_T_CMS19_H,
            eos.sesame.A2_u_CMS19_H,
            eos.sesame.A2_P_CMS19_H,
            eos.sesame.A2_c_CMS19_H,
            eos.sesame.A2_s_CMS19_H,
            eos.sesame.A1_log_rho_CMS19_H,
            eos.sesame.A1_log_T_CMS19_H,
            eos.sesame.A2_log_u_CMS19_H,
            eos.sesame.A2_log_P_CMS19_H,
            eos.sesame.A2_log_c_CMS19_H,
            eos.sesame.A2_log_s_CMS19_H,
        ) = eos.sesame.load_table_SESAME(gv.Fp_CMS19_H)
    if "CMS19_He" in A1_mat and len(eos.sesame.A1_rho_CMS19_He) == 1:
        (
            eos.sesame.A1_rho_CMS19_He,
            eos.sesame.A1_T_CMS19_He,
            eos.sesame.A2_u_CMS19_He,
            eos.sesame.A2_P_CMS19_He,
            eos.sesame.A2_c_CMS19_He,
            eos.sesame.A2_s_CMS19_He,
            eos.sesame.A1_log_rho_CMS19_He,
            eos.sesame.A1_log_T_CMS19_He,
            eos.sesame.A2_log_u_CMS19_He,
            eos.sesame.A2_log_P_CMS19_He,
            eos.sesame.A2_log_c_CMS19_He,
            eos.sesame.A2_log_s_CMS19_He,
        ) = eos.sesame.load_table_SESAME(gv.Fp_CMS19_He)
    if "CD21_HHe" in A1_mat and len(eos.sesame.A1_rho_CD21_HHe) == 1:
        (
            eos.sesame.A1_rho_CD21_HHe,
            eos.sesame.A1_T_CD21_HHe,
            eos.sesame.A2_u_CD21_HHe,
            eos.sesame.A2_P_CD21_HHe,
            eos.sesame.A2_c_CD21_HHe,
            eos.sesame.A2_s_CD21_HHe,
            eos.sesame.A1_log_rho_CD21_HHe,
            eos.sesame.A1_log_T_CD21_HHe,
            eos.sesame.A2_log_u_CD21_HHe,
            eos.sesame.A2_log_P_CD21_HHe,
            eos.sesame.A2_log_c_CD21_HHe,
            eos.sesame.A2_log_s_CD21_HHe,
        ) = eos.sesame.load_table_SESAME(gv.Fp_CD21_HHe)

    # Custom
    if "custom_0" in A1_mat and len(eos.sesame.A1_rho_custom_0) == 1:
        (
            eos.sesame.A1_rho_custom_0,
            eos.sesame.A1_T_custom_0,
            eos.sesame.A2_u_custom_0,
            eos.sesame.A2_P_custom_0,
            eos.sesame.A2_c_custom_0,
            eos.sesame.A2_s_custom_0,
            eos.sesame.A1_log_rho_custom_0,
            eos.sesame.A1_log_T_custom_0,
            eos.sesame.A2_log_u_custom_0,
            eos.sesame.A2_log_P_custom_0,
            eos.sesame.A2_log_c_custom_0,
            eos.sesame.A2_log_s_custom_0,
        ) = eos.sesame.load_table_SESAME(gv.Fp_custom_0)
    if "custom_1" in A1_mat and len(eos.sesame.A1_rho_custom_1) == 1:
        (
            eos.sesame.A1_rho_custom_1,
            eos.sesame.A1_T_custom_1,
            eos.sesame.A2_u_custom_1,
            eos.sesame.A2_P_custom_1,
            eos.sesame.A2_c_custom_1,
            eos.sesame.A2_s_custom_1,
            eos.sesame.A1_log_rho_custom_1,
            eos.sesame.A1_log_T_custom_1,
            eos.sesame.A2_log_u_custom_1,
            eos.sesame.A2_log_P_custom_1,
            eos.sesame.A2_log_c_custom_1,
            eos.sesame.A2_log_s_custom_1,
        ) = eos.sesame.load_table_SESAME(gv.Fp_custom_1)
    if "custom_2" in A1_mat and len(eos.sesame.A1_rho_custom_2) == 1:
        (
            eos.sesame.A1_rho_custom_2,
            eos.sesame.A1_T_custom_2,
            eos.sesame.A2_u_custom_2,
            eos.sesame.A2_P_custom_2,
            eos.sesame.A2_c_custom_2,
            eos.sesame.A2_s_custom_2,
            eos.sesame.A1_log_rho_custom_2,
            eos.sesame.A1_log_T_custom_2,
            eos.sesame.A2_log_u_custom_2,
            eos.sesame.A2_log_P_custom_2,
            eos.sesame.A2_log_c_custom_2,
            eos.sesame.A2_log_s_custom_2,
        ) = eos.sesame.load_table_SESAME(gv.Fp_custom_2)
    if "custom_3" in A1_mat and len(eos.sesame.A1_rho_custom_3) == 1:
        (
            eos.sesame.A1_rho_custom_3,
            eos.sesame.A1_T_custom_3,
            eos.sesame.A2_u_custom_3,
            eos.sesame.A2_P_custom_3,
            eos.sesame.A2_c_custom_3,
            eos.sesame.A2_s_custom_3,
            eos.sesame.A1_log_rho_custom_3,
            eos.sesame.A1_log_T_custom_3,
            eos.sesame.A2_log_u_custom_3,
            eos.sesame.A2_log_P_custom_3,
            eos.sesame.A2_log_c_custom_3,
            eos.sesame.A2_log_s_custom_3,
        ) = eos.sesame.load_table_SESAME(gv.Fp_custom_3)
    if "custom_4" in A1_mat and len(eos.sesame.A1_rho_custom_4) == 1:
        (
            eos.sesame.A1_rho_custom_4,
            eos.sesame.A1_T_custom_4,
            eos.sesame.A2_u_custom_4,
            eos.sesame.A2_P_custom_4,
            eos.sesame.A2_c_custom_4,
            eos.sesame.A2_s_custom_4,
            eos.sesame.A1_log_rho_custom_4,
            eos.sesame.A1_log_T_custom_4,
            eos.sesame.A2_log_u_custom_4,
            eos.sesame.A2_log_P_custom_4,
            eos.sesame.A2_log_c_custom_4,
            eos.sesame.A2_log_s_custom_4,
        ) = eos.sesame.load_table_SESAME(gv.Fp_custom_4)

    return None
