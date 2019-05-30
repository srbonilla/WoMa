WoMa
======

A python solver of the differential equations of a planet in hydrostatic
equilibrium for rotating and non-rotating cases.

It includes SEAGen (Kegerreis et al. 2019) and a new algorithm
to suit initial conditions for spining planets in SPH simulations.

Sergio Ruiz-Bonilla: sergio.ruiz-bonilla@durham.ac.uk  

This program has been tested for a wide range of cases but not exhaustively. If
you find any bugs, potential improvements, or features worth adding, then please
let us know!


Contents
--------
+ `woma.py` The main program classes and functions.
+ `seagen.py` The main program classes and functions of the SEAGen project.
+ `examples.py` Examples to demonstrate how to use the WoMa module.
+ `eos.py` Functions related equations of states.
+ `LICENSE.txt` GNU general public license v3+.


Basic Usage
-----------
+ See the doc strings in `woma.py` and `eos.py` for all the details.
+ See `examples.py` for full working examples.


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
+ Replace Tillotson (and all) EoS parameter arrays with simple class objects.
+ Import SEAGen as a normal module instead of copying the file?
+ Add documentation to examples and basic instructions to readme.
+ Why are some `jit`s commented out?
+ Replace `np.nan` with `None` for default function arguments
+ Use something more generic than core/mantle/atmosphere for the layer names
+ Simpler way to choose which input parameters are fixed
+ Allow fixing the mass ratios of each layer instead of e.g. outer radius 
+ Make into a PyPI package
+ Sort out eos.py's name to not conflict with GIHR
+ Apparently numba can't handle dictionaries?
+ Zero axes in example plots and non-symbol labels
+ Examples are using non-Earth materials?
+ Tidy line lengths for where I've messed things up
+ Edit/remove Planet() function. Could even just use the main class instead
+ Print output variables like boundaries when done
+ Add all output variables to the hdf5 file
+ Rename array variables e.g. r -> A1_r (or whatever) everywhere
+ Change the iterations to be for some tolerance not a number of iterations
+ Add units to hdf5 file
+ Set A1_r_layer and all other things that can be set after calculating profiles
+ Move the add-L3 function out of the class, like the others
