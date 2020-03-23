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
- [Contributing](#contributing)
- [Authors](#authors)
- [How to cite](#cite)
- [Acknowledgments](#acknowledgement)
<!-- - [TODO](../TODO.md) -->

## About <a name = "about"></a>


## Getting Started <a name = "getting_started"></a>

### Prerequisite <a name = "required"></a>
Setup `conda` enviroment with all necessary dependencies needed to run `mocalum` (copy/paste in terminal and hit ENTER):

```
conda create -c conda-forge -n mocalum --strict-channel-priority python=3.7 tqdm jupyter pytest netcdf4 xarray pylint matplotlib nbval scipy
```

### Installing
Clone repository:
```
git clone https://gitlab-internal.windenergy.dtu.dk/e-windlidar/mocalum.git
```

Afterwards, enter folder of `mocalum`
```
cd mocalum
```

Be sure that you are in the previously made [conda environment](#required):
```
conda activate mocalum
```

Now install `mocalum` in the conda environment with the same name:
```
pip install .
```

## Running the test <a name = "tests"></a>
While in `mocalum` folder go to `test` subfolder
```
cd test
```
and execute script:
```
python test_workflow.py
```

## Usage <a name="usage"></a>

## Built Using <a name = "built_using"></a>
- [Python](https://www.python.org/) - Languange
- [xarray](http://xarray.pydata.org/en/stable/#) - Package
- [numpy](https://numpy.org/) - Package

## Authors <a name = "authors"></a>
- [@niva]() - idea and work
- [@smbd]() - idea and work

## How to cite <a name = "cite"></a>

## Contributing <a name = "contributing"></a>

## Acknowledgements <a name = "acknowledgement"></a>

