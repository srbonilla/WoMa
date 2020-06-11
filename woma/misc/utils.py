"""
WoMa miscellaneous utilities
"""

import numpy as np
from numba import njit


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
