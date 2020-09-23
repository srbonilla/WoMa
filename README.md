![WoMa](http://astro.dur.ac.uk/~cklv53/woma_logo.png "WoMa")

Create models of rotating and non-rotating planets by solving the differential 
equations for hydrostatic equilibrium, and create initial conditions for e.g. 
smoothed particle hydrodynamics (SPH) simulations by placing particles that
precisely match the planet's profiles.

See the `tutorial.ipynb` notebook for a full tutorial and examples 
(https://github.com/srbonilla/WoMa), with additional documentation in this 
file, below.

Presented in Ruiz-Bonilla et al. (2020, https://arxiv.org/abs/2007.02965).

Includes SEAGen (https://github.com/jkeger/seagen; Kegerreis et al. 2019, 
[MNRAS 487:4](https://doi.org/10.1093/mnras/stz1606)) 
with modifications for spinning planets.

Sergio Ruiz-Bonilla: sergio.ruiz-bonilla@durham.ac.uk  
Jacob Kegerreis: jacob.kegerreis@durham.ac.uk

If you find any bugs, potential improvements, or features worth adding, then 
please let us know!


Files
-----
+ `tutorial.ipynb` Jupyter notebook tutorial and examples.
+ `woma/` Code directory.
    + `main.py` The main program classes and functions.
    + `data/` Data for equation of state (EoS) tables. 
    + `eos/` EoS and temperature-density relations.
    + `spherical_funcs/` Functions for spherical profiles.
    + `spin_funcs/` Functions for spinning profiles.
    + `misc/` Miscellaneous functions and constants.
+ `README.md` This file. General info and documentation.
+ `LICENSE.txt` GNU general public license v3+.
+ `setup.py`, `setup.cfg`, `MANIFEST.in` PyPI package files.


Installation and requirements
-----------------------------
+ Install the package with `pip install woma`, see https://pypi.org/project/woma/
+ Python 3 (tested with 3.6.9)
	  + seagen>=1.4.1, numpy, numba>=0.50.1, h5py


Notation etc.
-------------
+ Formatted with [black](https://github.com/psf/black).
+ Arrays are explicitly labelled with a prefix `A1_`, or `An_` for an
    `n`-dimensional array.




Documentation
=============
See the `tutorial.ipynb` notebook for a full tutorial and examples.
This documentation summarises the different options available.

Full documentation is provided in the class and function docstrings.

Most functions take an optional `verbosity` argument 
that controls the amount of printed information. 
Set `0` for no printing, `1` for standard output, or `2` for extra details.

### Contents
1. Spherical profiles  
2. Spinning profiles  
3. Particle placement




## 1. Spherical profiles 
All profiles require the temperature and either the pressure or density at the 
surface: `T_s` and `P_s` or `rho_s`,
plus the material and temperature-density relation for each layer (see below):
`A1_mat_layer` and `A1_T_rho_type`.

The optional parameter `num_prof` sets the number of profile integration steps.
Default 1000.


### Equations of state (EoS) 
It is important to check that the EoS you use are appropriate 
for your application.

So far, we have implemented several Tillotson, ANEOS, SESAME, 
and Hubbard & MacFarlane (1980) materials, with more on the way.

Materials are set for each layer with `A1_mat_layer`
using the following material names,
which are converted internally into material IDs,
set by a base type ID (multiplied by 100) plus a minor type:

+ Tillotson (Melosh 1989; Benz & Asphaug 1999): `1`
    + Iron: `Til_iron` : `100`
    + Granite: `Til_granite` : `101`
    + Water: `Til_water` : `102`
    + Basalt: `Til_basalt` : `103`
+ Hubbard & MacFarlane (1980): `2`
    + Hydrogen-helium atmosphere: `HM80_HHe` : `200`
    + Ice H20-CH4-NH3 mix: `HM80_ice` : `201`
    + Rock SiO2-MgO-FeS-FeO mix: `HM80_rock` : `202`
+ SESAME (Lyon & Johnson 1992) and similar: `3`
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
    So e.g. `"power=0"` for isothermal.


### Profile generation
There are several options for which additional parameters are set and which 
unknowns are found, depending on the number of layers in the planet.

Most of these functions are simple iterative bisection searches over the unknown 
parameter(s) to find a valid planet in hydrostatic equilibrium that satisfies 
the set attribute values.

The additional function arguments like `R_max` set things like the upper bound
for an iteration so usually do not need to be precise.

Optional arguments for these functions (in addition to the `verbosity`) set:
+ `tol`: The tolerance for finding unknown parameters as a fractional 
    difference between two consecutive iterations. Default usually 0.001, 
    depending on the method.
+ `num_attempt`: The maximum number of iteration attempts if the tolerance has 
    still not been reached. Default usually 40, depending on the method.

If the outer radii or masses of some but not all layers are required as inputs, 
then the unknown elements in the input arrays can be left as `None`, e.g.:
`A1_R_layer = [3.1415, None]` or `A1_M_layer = [None, 1.23, 4.56]`.

#### 1 layer
+ `gen_prof_L1_find_R_given_M()`, requires:
    + Total mass: `self.M`
    + Maximum radius: `R_max`
+ `gen_prof_L1_find_M_given_R()`, requires:
    + Total radius: `self.R`
    + Maximum mass: `M_max`

#### 2 layers
+ `gen_prof_L2_find_R1_given_M_R()`, requires:
    + Total radius: `self.R`
    + Total mass: `self.M`
+ `gen_prof_L2_find_M_given_R_R1()`, requires:
    + Total radius: `self.R`
    + Layer 1 outer radius: `self.A1_R_layer[0]`
    + Maximum mass: `M_max`
+ `gen_prof_L2_find_R_given_M_R1()`, requires:
    + Total mass: `self.M`
    + Layer 1 outer radius: `self.A1_R_layer[0]`
    + Maximum radius: `R_max`
+ `gen_prof_L2_find_R_R1_given_M1_M2()`, requires:
    + Layer 1 and 2 masses: `self.A1_M_layer`
    + Minimum and maximum radii: `R_min`, `R_max`
    
#### 3 layers
+ `gen_prof_L3_find_M_given_R_R1_R2()`, requires:
    + Layer 1, 2, and 3 outer radii: `self.A1_R_layer`
+ `gen_prof_L3_find_R1_given_M_R_R2()`, requires:
    + Total mass: `self.M`
    + Total radius: `self.R`
    + Layer 2 outer radius: `self.A1_R_layer[1]`
+ `gen_prof_L3_find_R2_given_M_R_R1()`, requires:
    + Total mass: `self.M`
    + Total radius: `self.R`
    + Layer 1 outer radius: `self.A1_R_layer[0]`
+ `gen_prof_L3_find_R_given_M_R1_R2()`, requires:
    + Total mass: `self.M`
    + Layer 1 and 2 outer radii: `self.A1_R_layer[0]`, `[1]`
+ `gen_prof_L3_find_R1_R2_given_M_R_I()`, requires:
    + Total mass: `self.M`

#### Adding layers
+ `gen_prof_given_inner_prof()`:
    After generating an initial planet, a new layer can be added on top by 
    integrating outwards. Requires:
    + Name of the material in the new layer: `mat`
    + Temperature-density relation in the new layer: `T_rho_type`
    + Minimum density at which the new layer will stop: `rho_min`
    + Minimum pressure at which the new layer will stop: `P_min`

#### Additional parameters
See the `main.py` and other docstrings for full details.

The class's `num_prof` parameter sets the number of radial profile steps,
while the profile generating functions take arguments like `tol`
and/or `num_attempt` that control the convergence criterion and 
maximum number of iterations to do to find the unknown parameters.




## 2. Spinning profiles  
See `tutorial.ipynb` for the main usage:
```python
spherical_planet = woma.Planet( . . . ) 

spin_planet = woma.SpinPlanet(
    planet = spherical_planet,
    period = 24,  # hours
)
```

The output attributes available from the `spin_planet` object 
are documented in the `SpinPlanet` class docstring in `woma/main.py`.

The primary outputs are the arrays of properties of the nested spheroids,
including their equatorial and polar radii (semi-major and semi-minor axes), 
`A1_R` and `A1_Z`, 
and for example their masses, densities, pressures, and temperatures, 
`A1_m`, `A1_rho`, `A1_P`, and `A1_T`.

Additional parameters are similar to those in the spherical case mentioned 
above. See the docstrings for full details.



## 3. Particle placement
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
are documented in the `ParticlePlanet` class docstring in `woma/main.py`.
