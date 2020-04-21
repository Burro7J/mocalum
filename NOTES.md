# Developer notes

## Conda dev enviroment setup
Setup `conda` development enviroment `mocalum_dev` using following command (ctrl+c in terminal):

```
conda create -c conda-forge -n mocalum_dev --strict-channel-priority python=3.7 pylint pytest nbval jupyter numpy pandas xarray netCDF4 matplotlib tqdm scipy
```

## Use of submodules
`MOCALUM` use [`PyConTurb`](https://pyconturb.pages.windenergy.dtu.dk/pyconturb/index.html) as a submodule.
The submodule is configure using following `git` commands:
```
git submodule add https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git modules/pyconturb
git submodule update --init --recursive
```

