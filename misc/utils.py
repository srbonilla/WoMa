#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:28:08 2019

@author: sergio
"""

import numpy as np
from numba import njit

def _print_banner():
    print("\n")
    print("#  WoMa - World Maker")
    print("#  sergio.ruiz-bonilla@durham.ac.uk")
    print("\n")

# Misc utilities (move to separate file...)
def check_end(string, end):
    """ Check that a string ends with the required characters and append them
        if not.
    """
    if string[-len(end):] != end:
        string  += end

    return string

def format_array_string(array, format):
    """ Return a print-ready string of an array's contents in a given format.

        Args:
            array ([])
                An array of values that can be printed with the given format.

            format (str)
                A printing format, e.g. "%d", "%.5g".

                Custom options:
                    "string":   Include quotation marks around each string.
                    "dorf":     Int or float if decimal places are needed.

        Returns:
            string (str)
                The formatted string.
    """
    string  = ""

    # Append each element
    # 1D
    if len(np.shape(array)) == 1:
        for x in array:
            if x is None:
                string += "None, "
            else:
                # Custom formats
                if format == "string":
                    string += "\"%s\", " % x
                elif format == "dorf":
                    string += "{0}, ".format(
                        str(round(x, 1) if x % 1 else int(x)))
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
    """ Return a string for aligned printing with adjusted spaces to account for
        the length of the input.

        e.g.
            >>> asdf = 123
            >>> qwerty = 456
            >>> print("%s = %d \n""%s = %d" %
                      (add_whitespace("asdf", 12), asdf,
                       add_whitespace("qwerty", 12), qwerty))
            asdf         = 123
            qwerty       = 456
    """
    return "%s" % string + " " * (space - len("%s" % string))

@njit
def moi(A1_r, A1_rho):
    """ Computes moment of inertia for a planet with spherical symmetry.

        Args:
            A1_r ([float]):
                Radii of the planet (SI).

            A1_rho ([float]):
                Densities asociated with the radii (SI)

        Returns:
            MoI (float):
                Moment of inertia (SI).
    """
    dr  = np.abs(A1_r[0] - A1_r[1])
    r4  = np.power(A1_r, 4)
    MoI = 2*np.pi*(4/3)*np.sum(r4*A1_rho)*dr

    return MoI