"""
WoMa equation of state set up
"""

import os
import numpy as np
import sys

from woma.misc import glob_vars as gv
from woma.eos import tillotson

# Set up equation of state data
def set_up():
    """ Fetch or create equation of state files if they don't exist.
    
    Saved in the `data/` directory.
    """
    # Make the directory if it doesn't already exist
    if not os.path.isdir("data"):
        os.mkdir("data")

    # Make the files if they don't already exist

    # Internal energy cold curves for Tillotson mateials
    if not os.path.isfile(gv.Fp_u_cold_Til_iron):
        print("Creating u cold curve for material Til_iron... ", end="")
        sys.stdout.flush()
        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_iron)
        np.save(gv.Fp_u_cold_Til_iron, u_cold_array)
        del u_cold_array
        print("Done")

    if not os.path.isfile(gv.Fp_u_cold_Til_granite):
        print("Creating u cold curve for material Til_granite... ", end="")
        sys.stdout.flush()
        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_granite)
        np.save(gv.Fp_u_cold_Til_granite, u_cold_array)
        del u_cold_array
        print("Done")

    if not os.path.isfile(gv.Fp_u_cold_Til_basalt):
        print("Creating u cold curve for material Til_basalt... ", end="")
        sys.stdout.flush()
        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_basalt)
        np.save(gv.Fp_u_cold_Til_basalt, u_cold_array)
        del u_cold_array
        print("Done")

    if not os.path.isfile(gv.Fp_u_cold_Til_water):
        print("Creating u cold curve for material Til_water... ", end="")
        sys.stdout.flush()
        u_cold_array = tillotson._create_u_cold_array(gv.id_Til_water)
        np.save(gv.Fp_u_cold_Til_water, u_cold_array)
        del u_cold_array
        print("Done")

    # SESAME tables
    ###TO DO
    if not os.path.isfile(gv.Fp_SESAME_iron):
        print("Downloading SESAME iron table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")

    if not os.path.isfile(gv.Fp_SESAME_basalt):
        print("Downloading SESAME basalt table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")

    if not os.path.isfile(gv.Fp_SESAME_water):
        print("Downloading SESAME water table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")

    if not os.path.isfile(gv.Fp_SS08_water):
        print("Downloading SS08 water table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")

    if not os.path.isfile(gv.Fp_SESAME_H2):
        print("Downloading SESAME H2 table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")

    if not os.path.isfile(gv.Fp_SESAME_N2):
        print("Downloading SESAME N2 table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")

    if not os.path.isfile(gv.Fp_SESAME_steam):
        print("Downloading SESAME steam table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")

    if not os.path.isfile(gv.Fp_SESAME_CO2):
        print("Downloading SESAME CO2 table... ", end="")
        sys.stdout.flush()
        # Do stuff
        print("Done")


set_up()