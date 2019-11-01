WoMa
======

A python solver of the differential equations of a planet in hydrostatic
equilibrium for rotating and non-rotating cases, and it's corresponding
particle placement for smooth particle hydrodynamics (SPH) simulations.

It includes SEAGen (Kegerreis et al. 2019) and a new algorithm
to suit initial conditions for spining planets in SPH simulations.

Sergio Ruiz-Bonilla: sergio.ruiz-bonilla@durham.ac.uk  

This program has been tested for a wide range of cases but not exhaustively. If
you find any bugs, potential improvements, or features worth adding, then please
let us know!


Contents
--------
+ `/eos` Equations of state and relation temperature-density.
+ `/data` Data forder for EoS tables. 
+ `/misc` Miscellaneus functions.
+ `/spherical_funcs` Miscellaneus functions for spherical planets.
+ `/spin_funcs` Miscellaneus functions for spinning planets.
+ `woma.py` The main program classes and functions.
+ `examples.py` Examples to demonstrate how to use the WoMa module.
+ `LICENSE.txt` GNU general public license v3+.


Basic Usage
-----------
+ See the doc strings in `woma.py` and `eos.py` for all the details.
+ See `tutorial.ipynb` and `examples.py` for full working examples.


Requirements
------------
+ Python 3 (tested with 3.6.0).


Notation etc.
-------------
+ PEP8 is followed in most cases apart from some indentation alignment.
+ Arrays are explicitly labelled with a prefix `A1_`, or `An_` for an
    `n`-dimensional array.
+ Particle is abbreviated to `picle`.


To Do etc. (Jacob thoughts in progress)
---------------------------------------
+ Change the iterations to be for some tolerance not a number of iterations
+ Move the add-L3 function out of the class, like the others
+ Add documentation to examples and basic instructions to readme
+ Make into a PyPI package
+ Add all output variables to the hdf5 file
+ Add doc strings and prints to the examples, explain what they're trying to do
    and how, so that someone can copy them and understand.
+ Change T_rho_type options to string not int for input (and internal too?)
