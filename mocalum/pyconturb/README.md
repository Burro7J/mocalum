[![pipeline status](https://gitlab.windenergy.dtu.dk/rink/pyconturb/badges/master/pipeline.svg)](https://gitlab.windenergy.dtu.dk/rink/pyconturb/commits/master)
[![coverage report](https://gitlab.windenergy.dtu.dk/rink/pyconturb/badges/master/coverage.svg)](https://gitlab.windenergy.dtu.dk/rink/pyconturb/commits/master)

![PyConTurb](https://gitlab.windenergy.dtu.dk/rink/pyconturb/raw/master/docs/logo.png)

# PyConTurb: Constrained Stochastic Turbulence for Wind Energy Applications

This Python package uses a novel method to generate stochastic turbulence boxes
that are constrained by one or more measured time series. Details on the theory
can be found in [this paper from Torque 2016](https://iopscience.iop.org/article/10.1088/1742-6596/1037/6/062032/meta).

Despite the package's name, the main function, `gen_turb` can be used with or
without constraining time series. Without the constraining time series, it is
the Veers simulation method.

## Installation, Examples, Bug Reporting and More

Please see the [documentation website](https://pyconturb.pages.windenergy.dtu.dk/pyconturb/).
