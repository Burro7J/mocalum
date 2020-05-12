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
Ideally you should have `conda` or `anaconda` installed on your computer so you can build an isolated `python` environment in which you will install `mocalum`.


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
pip install git+https://github.com/niva83/mocalum/mocalum.git
```

## Usage <a name="usage"></a>

In the folder [examples](./examples) you will find [jupyter](https://jupyter.org/) notebook tutorial on how to use `mocalum`. The purpose of the tutorials is to familiarize users with [mocalum](https://github.com/niva83/mocalum) and enable them to quickly build there own workflows with it. The tutorials cover various usage of [mocalum](https://github.com/niva83/mocalum). The tutorials are described in a dedicated [README](./examples/README.md).

## Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/) - Languange
- [xarray](http://xarray.pydata.org/en/stable/#) - Package
- [numpy](https://numpy.org/) - Package
- [pandas](https://pandas.pydata.org/) - Package
- [netCDF4](http://unidata.github.io/netcdf4-python/netCDF4/index.html) - Package
- [scipy](https://www.scipy.org/) - Package
- [pyconturb](https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb) - Package


## Contributors <a name = "authors"></a>

### Author
- [Nikola Vasiljevic](https://orbit.dtu.dk/en/persons/nikola-vasiljevic) - design, development and testing (DevOps) of `mocalum`

### Contributors
- [Andrea Vignaroli](https://orbit.dtu.dk/en/persons/andrea-vignaroli) - initial wrapper around [PyConTurb](https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb), method for conversion of 3D to 4D turbulence box
- [Bjarke Tobias Olsen](https://orbit.dtu.dk/en/persons/bjarke-tobias-olsen) - hints on how to speed up advance interpolation using [xarray](http://xarray.pydata.org/en/stable/interpolation.html#advanced-interpolation)
- [Anders Tegtmeier Pedersen](https://orbit.dtu.dk/en/persons/anders-tegtmeier-pedersen) - `matlab` script to sample correlated uncertainties

## How to cite <a name = "cite"></a>


[![DOI](https://zenodo.org/badge/262975742.svg)](https://zenodo.org/badge/latestdoi/262975742)


## Contributing <a name = "contributing"></a>
If you want to take an active part in the further development of `mocalum` make a pull request or post an issue in this repository.

