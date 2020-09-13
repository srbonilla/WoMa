"""
WoMa miscellaneous utilities
"""

import numpy as np
from numba import njit
from woma.misc.glob_vars import G


def _print_banner():
    print("\n")
    print("#  WoMa - World Maker")
    print("#  sergio.ruiz-bonilla@durham.ac.uk")
    print("\n")


def check_end(string, end):
    """ Check that a string ends with the required characters, append them if not.
    
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
    """ Return a print-ready string of an array's contents in a given format.
    
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
    """ Return a padded string for aligned printing with adjusted spaces.

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
    """ Compute the moment of inertia for a planet with spherical symmetry.

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
    """ Class to store conversions from one set of units to another, derived
        using the base mass-, length-, and time-unit relations.

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
        self.v = l * t ** -1
        self.a = l * t ** -2
        self.rho = m * l ** -3
        self.drho_dt = m * l ** -4
        self.P = m * l ** -1 * t ** -2
        self.u = l ** 2 * t ** -2
        self.du_dt = l ** 2 * t ** -3
        self.E = m * l ** 2 * t ** -2
        self.s = l ** 2 * t ** -2
        self.G = m ** -1 * l ** 3 * t ** -2

    def inv(self):
        """ Return the inverse to this conversion """
        return Conversions(1 / self.m, 1 / self.l, 1 / self.t)


# Standard unit conversions
SI_to_SI = Conversions(1, 1, 1)  # No-op
cgs_to_SI = Conversions(1e-3, 1e-2, 1)
SI_to_cgs = cgs_to_SI.inv()
