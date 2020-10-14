"""
WoMa equation of state set up
"""

import os
import numpy as np
import sys

from woma.misc import glob_vars as gv
from woma.eos import tillotson

# Set up equation of state data
def set_up(verbosity=1):
    """Fetch or create equation of state files if they don't exist.

    Saved in the `data/` directory.
    """
    # Make the directory if it doesn't already exist
    if not os.path.isdir("data"):
        os.mkdir("data")

    # Make the files if they don't already exist

    # Internal energy cold curves for Tillotson mateials
    if not os.path.isfile(gv.Fp_u_cold_Til_iron):
        if verbosity >= 1:
            print("Creating u cold curve for material Til_iron... ", end="")
            sys.stdout.flush()

        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_iron)
        np.save(gv.Fp_u_cold_Til_iron, u_cold_array)
        del u_cold_array

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_u_cold_Til_granite):
        if verbosity >= 1:
            print("Creating u cold curve for material Til_granite... ", end="")
            sys.stdout.flush()

        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_granite)
        np.save(gv.Fp_u_cold_Til_granite, u_cold_array)
        del u_cold_array

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_u_cold_Til_basalt):
        if verbosity >= 1:
            print("Creating u cold curve for material Til_basalt... ", end="")
            sys.stdout.flush()

        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_basalt)
        np.save(gv.Fp_u_cold_Til_basalt, u_cold_array)
        del u_cold_array

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_u_cold_Til_water):
        if verbosity >= 1:
            print("Creating u cold curve for material Til_water... ", end="")
            sys.stdout.flush()

        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_water)
        np.save(gv.Fp_u_cold_Til_water, u_cold_array)
        del u_cold_array

        if verbosity >= 1:
            print("Done")

    # SESAME tables
    ###TO DO
    if not os.path.isfile(gv.Fp_SESAME_iron):
        if verbosity >= 1:
            print("Downloading SESAME iron table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_SESAME_basalt):
        if verbosity >= 1:
            print("Downloading SESAME basalt table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_SESAME_water):
        if verbosity >= 1:
            print("Downloading SESAME water table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_SS08_water):
        if verbosity >= 1:
            print("Downloading SS08 water table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_SESAME_H2):
        if verbosity >= 1:
            print("Downloading SESAME H2 table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_SESAME_N2):
        if verbosity >= 1:
            print("Downloading SESAME N2 table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_SESAME_steam):
        if verbosity >= 1:
            print("Downloading SESAME steam table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")

    if not os.path.isfile(gv.Fp_SESAME_CO2):
        if verbosity >= 1:
            print("Downloading SESAME CO2 table... ", end="")
            sys.stdout.flush()

        # Do stuff

        if verbosity >= 1:
            print("Done")


set_up()
