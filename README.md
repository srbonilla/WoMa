WoMa
====

Create models of rotating and non-rotating planets by solving the differential 
equations for hydrostatic equilibrium, and create initial conditions for 
smoothed particle hydrodynamics (SPH) simulations by placing particles that
precisely match the planet's profiles.

See the `tutorial.ipynb` notebook for a full tutorial and examples,
with additional documentation in this file, below.

<!-- Presented in Ruiz-Bonilla et al. (2020), MNRAS..., https://doi.org/... -->

Includes SEAGen (https://github.com/jkeger/seagen; Kegerreis et al. 2019, MNRAS 
487:4) with modifications for spinning planets.

Sergio Ruiz-Bonilla: sergio.ruiz-bonilla@durham.ac.uk  
Jacob Kegerreis: jacob.kegerreis@durham.ac.uk

<!-- Visit https://github.com/.../woma to download the code including examples
and for support. -->

This program has been tested for a wide range of cases but not exhaustively. If
you find any bugs, potential improvements, or features worth adding, then please
let us know!


Files
-----
+ `tutorial.ipynb` Jupyter notebook tutorial and examples.
+ `woma/` Code directory.
    + `main.py` The main program classes and functions.
    + `data/` Data folder for equation of state (EoS) tables. 
    + `eos/` EoS and temperature-density relations.
    + `spherical_funcs/` Functions for spherical planets.
    + `spin_funcs/` Functions for spinning planets.
    + `misc/` Miscellaneous functions.
+ `README.md` This file. General info and documentation.
+ `LICENSE.txt` GNU general public license v3+.


Requirements
------------
+ Python 3 (tested with 3.6.0).


Notation etc.
-------------
+ Formatted with [black](https://github.com/psf/black).
+ Arrays are explicitly labelled with a prefix `A1_`, or `An_` for an
    `n`-dimensional array.
    

Dev Notes
---------
+ Format all files with black (except the examples):  
    `black woma/`
+ Comment out numba for debugging:  
    `find ./ -type f -name "*.py" -exec sed -i "s/@njit/#@njit/g" {} \;`  
    Revert: `find ./ -type f -name "*.py" -exec sed -i "s/#@njit/@njit/g" {} \;`




Documentation
=============
See the `tutorial.ipynb` notebook for a full tutorial and examples.
The basic usage explained there is not repeated here.
This documentation summarises the different options available.

Full documentation is provided in the class and function docstrings.

Most functions take an optional `verbosity` argument 
that controls the amount of printed information. 
Set `0` for no printing, `1` for standard output, or `2` for extra details.

### Contents
1. Spherical profiles  
2. Spinning profiles  
3. Particle placement


1. Spherical profiles 
---------------------
All profiles require 2 of the 3 surface values: `P_s`, `T_s`, `rho_s`,
plus the material and temperature-density relation for each layer (see below):
`A1_mat_layer` and `A1_T_rho_type`.


### Equations of state (EoS) 
It is important to check that the EoS you use are appropriate 
for your use case.

So far, we have implemented several Tillotson, ANEOS, SESAME, 
and Hubbard & MacFarlane (1980) materials, with more on the way.

Materials are set for each layer with `A1_mat_layer`
using the following material names,
which are converted internally into material IDs,
set by a base type ID (multiplied by 100) plus a minor type:

+ Tillotson (Melosh 2007; Benz & Asphaug 1999): `1`
    + Iron: `Til_iron` : `100`
    + Granite: `Til_granite` : `101`
    + Water: `Til_water` : `102`
    + Basalt: `Til_basalt` : `103`
+ Hubbard & MacFarlane (1980): `2`
    + Hydrogen-helium atmosphere: `HM80_HHe` : `200`
    + Ice H20-CH4-NH3 mix: `HM80_ice` : `201`
    + Rock SiO2-MgO-FeS-FeO mix: `HM80_rock` : `202`
+ SESAME (and similar): `3`
    + Iron (2140): `SESAME_iron` : `300`
    + Basalt (7530): `SESAME_basalt` : `301`
    + Water (7154): `SESAME_water` : `302`
    + Senft & Stewart (2008) water in a SESAME-style table: `SS08_water` : `303`
+ ANEOS (in SESAME-style tables): `4`
    + Forsterite (Stewart et al. 2019): `ANEOS_forsterite` : `400`
    + Iron (Stewart, zenodo.org/record/3866507): `ANEOS_iron` : `401`
    + Fe85Si15 (Stewart, zenodo.org/record/3866550): `ANEOS_Fe85Si15` : `402`

These are defined in `woma/misc/glob_vars.py`.


### Temperature-density relations
These relations are set for each layer with `A1_T_rho_type`:

+ `"adiabatic"`: Adiabatic, only available for some EoS.
+ `"power=a"` where `a` is a float: A power law T ~ rho^`a`. 
    Use `"power=0"` for isothermal.


### Profile generation
There are several options for which additional parameters are set 
and which unknowns are found, depending on the number of layers in the planet.

Most of these functions are simple iterative bisection searches 
over the unknown parameter(s)
to find a valid planet in hydrostatic equilibrium 
that satisfies the set values.
The maximum parameters (e.g. `R_max`) just set the upper bound 
for these iterations so do not need to be precise.

If the outer radii or masses of some but not all layers are required as inputs, 
then the unknown elements in the input arrays can be left as `None`, e.g.:
`A1_R_layer = [3.14, None]` or `A1_M_layer = [None, 3.14, 1.59]`.

#### 1 layer
+ `gen_prof_L1_find_R_given_M()`, requires:
    + Total mass: `M`
    + Maximum radius: `R_max`
+ `gen_prof_L1_find_M_given_R()`, requires:
    + Total radius: `R`
    + Maximum mass: `M_max`
+ `gen_prof_L1_given_R_M()`, requires: \#\#\#why?
    + Total radius: `R`
    + Total mass: `M`

#### 2 layers
+ `gen_prof_L1_find_R_given_M()`, requires:
    + Total mass: `M`
    + Maximum radius: `R_max`
+ `gen_prof_L2_find_R1_given_R_M()`, requires: \#\#\#why?
    + Total radius: `R`
    + Total mass: `M`
+ `gen_prof_L2_find_M_given_R1_R()`, requires:
    + Total radius: `R`
    + Layer 1 outer radius: `A1_R_layer[0]`
    + Maximum mass: `M_max`
+ `gen_prof_L2_find_R_given_M_R1()`, requires:
    + Total mass: `M`
    + Layer 1 outer radius: `A1_R_layer[0]`
    + Maximum radius: `R_max`
+ `gen_prof_L2_find_R1_R_given_M1_M2()`, requires:
    + Layer 1 and 2 masses: `A1_M_layer`
    + Maximum radius: `R_max`
+ `gen_prof_L2_given_R_M_R1()`, requires: \#\#\#why?
    + Total mass: `M`
    + Total radius: `R`
    + Layer 1 outer radius: `A1_R_layer[0]`
+ `gen_prof_L2_find_R1_given_M1_add_L2()`,
    first make the inner 1 layer, then add a second layer on top
    (e.g. atmosphere), integrating outwards. 
    The `_s` "surface" parameters set the conditions 
    at the outer edge of the inner layer.
    Requires:
    + Layer 1 mass: `A1_M_layer[0]`
    + Layer 1 maximum radius: `R_max`
    + Minimum density for outer layer: `rho_min`
+ `gen_prof_L2_find_M1_given_R1_add_L2()` (see above), requires:
    + Layer 1 outer radius: `A1_R_layer[0]`
    + Layer 1 maximum mass: `M_max`
    + Minimum density for outer layer: `rho_min`

#### 3 layers
+ `gen_prof_L3_find_R2_given_R_M_R1()`, requires:
    + Total mass: `M`
    + Total radius: `R`
    + Layer 1 outer radius: `A1_R_layer[0]`
+ `gen_prof_L3_find_R1_given_R_M_R2()`, requires:
    + Total mass: `M`
    + Total radius: `R`
    + Layer 2 outer radius: `A1_R_layer[1]`
+ `gen_prof_L3_find_M_given_R_R1_R2()`, requires:
    + Layer 1, 2, and 3 outer radii: `A1_R_layer`
+ `gen_prof_L3_find_R_given_M_R1_R2()`, requires:
    + Total mass: `M`
    + Layer 1 and 2 outer radii: `A1_R_layer[0]`, `[1]`
+ `gen_prof_L3_given_R_M_R1_R2()`, requires: \#\#\#why?
    + Total mass: `M`
    + Total radius: `R`
    + Layer 1 and 2 outer radii: `A1_R_layer[0]`, `[1]`
+ `gen_prof_L3_find_R1_R2_given_M1_M2_add_L3()` 
    (see `gen_prof_L2_find_R1_given_M1_add_L2()`), requires:
    + Layer 1 and 2 masses: `A1_M_layer[0]`, `[1]`
    + Minimum density for outer layer: `rho_min`


2. Spinning profiles  
--------------------


3. Particle placement
---------------------
See `tutorial.ipynb` for the main usage:
```python
planet = woma.Planet( . . . ) 
# or 
planet = woma.SpinPlanet( . . . )

N_particles = 1e7
N_ngb = 48 # Optional number of neighbours for approximate SPH smoothing lengths

particles = woma.ParticleSet(planet, N_particles, N_ngb=N_ngb)
``` 

The output attributes available from the `particles` object 
are documented in the `ParticleSet` class docstring in `woma/main.py`.