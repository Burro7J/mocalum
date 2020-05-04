<h3 align="center">MOCALUM</h3>
<h4 align="center">Monte Carlo based Lidar Uncertainty Model</h4>

---

<p align="center"> A Python package for Monte Carlo based lidar uncertainty modeling.
    <br>
</p>

## Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Contributors](#authors)
- [How to contribute](#contributing)
- [How to cite](#cite)
- [Acknowledgments](#acknowledgement)
<!-- - [TODO](../TODO.md) -->

## About <a name = "about"></a>
`mocalum` is a python package for Monte Carlo based lidar uncertainty modeling. It has following features:
 - Slick and super fast Monte Carlo uncertainty modeling
 - Simulation of single or multi lidar configuration
 - Configuration of arbitrary trajectories for single and multi lidars
 - Configuration of `IVAP` (sector-scan) trajectory for single lidar
 - 3D or 4D / uniform or turbulent flow field generation
 - Sampling of correlated or uncorrelated uncertainty terms
 - Built-in 2nd order kinematic model for calculation of trajectory timing
 - 3D or 4D interpolation/projection of flow on lidar(s) line-of-sight(s)
 - [xarray](http://xarray.pydata.org/en/stable/#) datasets enriched with metadata

## Getting Started <a name = "getting_started"></a>

### Prerequisite <a name = "required"></a>
You must have `conda` or `anaconda` installed on your computer.


### Installing
Make a new `conda` environment:
```
conda create -n mc_test python=3.7
```

Be sure that you are in the previously made [conda environment](#required):
```
conda activate mc_test
```

Install `mocalum` in the new environment and you are ready to go:
```
pip install git+https://gitlab-internal.windenergy.dtu.dk/e-windlidar/mocalum.git
```

## Running the test <a name = "tests"></a>
<!-- While in `mocalum` folder go to `test` subfolder
```
cd test
```
and execute script:
```
python test_workflow.py
``` -->

## Usage <a name="usage"></a>

## Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/) - Languange
- [xarray](http://xarray.pydata.org/en/stable/#) - Package
- [numpy](https://numpy.org/) - Package
- [pandas]() - Package
- [netCDF4]() - Package
- [tqdm]() - Package
- [scipy]() - Package
- [pyconturb](https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb) - Package


## Contributors <a name = "authors"></a>

### Author
- [Nikola Vasiljevic](@niva) - design, development and testing (DevOps) of `mocalum`

### Contributors
- [Andrea Vignaroli](@andv) - initial wrapper around [PyConTurb](https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb), method for conversion of 3D to 4D turbulence box
- [Bjarke Tobias Olsen](@btol) - hints on how to speed up advance interpolation using [xarray](http://xarray.pydata.org/en/stable/interpolation.html#advanced-interpolation)
- [Anders Tegtmeier Pedersen](@antp) - `matlab` script to sample correlated uncertainties

## How to cite <a name = "cite"></a>

## Contributing <a name = "contributing"></a>

## Acknowledgements <a name = "acknowledgement"></a>

