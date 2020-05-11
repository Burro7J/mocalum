# Overview

`mocalum` is a python package for Monte Carlo based lidar uncertainty modeling. It has following features:

- Fast Monte Carlo uncertainty modeling
- Simulation of single or multi lidar configurations
- Configuration of arbitrary trajectories for single and multi lidars
- Configuration of [IVAP](https://journals.ametsoc.org/doi/10.1175/JTECH2047.1) (sector-scan) trajectory for single lidar
- 3D or 4D / uniform or turbulent flow field generation
- Sampling of correlated or uncorrelated uncertainty terms
- Built-in 2nd order kinematic model for calculation of trajectory timing
- 3D or 4D interpolation/projection of flow on lidar(s) line-of-sight(s)
- [xarray](http://xarray.pydata.org/en/stable/#) datasets enriched with metadata

## `mocalum` workflow

Typical `mocalum` workflow includes following steps (depict in the figure below):

1. Creating `mocalum` object
1. Adding lidar(s) and configuring uncertainty contributors to the object
2. Setting up measurement scenario for previously added lidar(s)
2. Sampling uncertainties
3. Generating flow field entailing measurement points
4. Project flow field on line-of-sight(s)
5. Reconstruct wind vector(s)
6. Perform statistical analysis of reconstructed wind vectors

![mocalum workflow](./assets/workflow.png)

**Figure 1.** `mocalum` workflow

### `mocalum` object
`mocalum` is developed considering [object-oriented paradigm](https://realpython.com/python3-object-oriented-programming/). Therefore, all calculations and access to generated data are through an instance of `mocalum` object on which various methods are applied. Once an instance of `mocalum` object is created users have access to following methods:

- `add_lidar`
- `generate_complex_trajectory`
- `generate_PPI_scan`
- `generate_uncertainties`
- `generate_flow_field`
- `project_to_los`
- `reconstruct_wind`

as well to `data` that will be created as a result of applying the above listed methods.

### Adding lidar

`add_lidar` method is used to add lidar(s) to the `mocalum` object instance. This method requires following input parameters:

- `id` of lidar
- `lidar position` as `numpy` array of triplets (x, y, z)
- `uncertainty dictionary` containing configuration of uncertainty contributors

The `uncertainty dictionary` contains values as standard uncertainties of each contributor to the total wind reconstruction uncertainty. The uncertainty contributors are:

- estimation uncertainty of radial velocity (in m/s)
- ranging uncertainty (in m)
- azimuth uncertainty (in deg)
- elevation uncertainty (in deg)

The uncertainty contributors are in more details described in [Uncertainty model for dual-Doppler retrievals of wind speed and wind direction](https://www.overleaf.com/project/5d26f91c09b8aa33a4702c4e).

Here is an example on how a `mocalum` object is created, and then a lidar named `koshava` added to the object:

```
import numpy as np
import mocalum as mc

mc_test = mc.Mocalum()

lidar_pos = np.array([0,0,0])

unc_cfg = {'unc_az'   : 0.1, # standard uncertainty for azimuth in deg
           'unc_el'   : 0.1, # standard uncertainty for elevation in deg
           'unc_rng'  : 5,   # standard uncertainty for ranging in m
           'unc_est'  : 0.1, # standard uncertainty for radial velocity estimation in m/s
           'corr_coef':0}    # correlation coefficient

mc_test.add_lidar('koshava', lidar_pos, unc_cfg)
```

One can access the underlying data that have been created by adding a lidar to the mocalum `object` by simply accessing the measurement configuration dictionary stored inside `data` sub-object:

```
mc_test.data.meas_cfg['koshava']
```

which returns:

```
{'position': array([0, 0, 0]),
 'uncertainty': {'unc_az': {'mu': 0, 'std': 0.1, 'units': 'deg'},
  'unc_el': {'mu': 0, 'std': 0.1, 'units': 'deg'},
  'unc_rng': {'mu': 0, 'std': 5, 'units': 'm'},
  'unc_est': {'mu': 0, 'std': 0.1, 'units': 'm.s^-1'},
  'corr_coef': 0},
 'config': {}}
```



### Setting up measurement scenario

Following the lidar placement and uncertainty configuration an user configures the measurement scenario which lidar will 'virtually' perform. Two methods are available to this, either the user can configure `PPI` or complex trajectory scans, using methods `generate_PPI_scan` and `generate_complex_trajectory` respectively.

`generate_PPI_scan` method requires following input parameters:

- lidar id for which the PPI scan is configured
- PPI scan dictionary containing following required parameters:

    - `sector_size` : size of the scanned sector in degrees
    - `azimuth_mid` : central azimuth angle of the PPI scanned arc
    - `angular_step` : Incremental step performed to complete a PPI scan
    - `acq_time` : acquisition time to complete an angular step, thus acquire LOS measurements
    - `elevation` : elevation angle of PPI scan
    - `range` : range at which measurements should take place
    - `no_scans` : number of PPI scans must be equal or bigger than 1

while following parameters are optional:

- `max_speed` : maximum permitted angular speed
- `max_acc` : maximum permitted angular acceleration


Similarly, `generate_complex_trajectory` method requires following parameters:

- lidar ids provided as a list of strings for which the complex trajectory will be generated
- Scan configuration as a dictionary containing with following key-value pairs:

    - `points` : `numpy` array of measurement points provided as triplets of coordinates (x, y, z)
    - `no_scans` : number of scans through the points
    - `acq_time` : acquisition time of LOS measurements

while following parameters are optional:

- `max_speed` : max permitted angular speed
- `max_acc` : max permitted angular acceleration
- `sync` : indicates whether to synchronize ('sync':True) or not ('sync':False) motion among multiple lidars

Here is an example on how a `PPI` scan is configure and assigned to the lidar `koshava`:
```
meas_height = 100 # in m
meas_range = 1000 # in m
PPI_cfg = {
    'no_scans' : 10000,
    'range' : meas_range,
    'elevation' : np.degrees(np.arcsin(meas_height / meas_range)), # to assure measurements at 100 m agl
    'angular_step' : 1,  # in deg
    'acq_time' : 1,      # in s
    'azimuth_mid' : 90,  # central azimuth angle in deg
    'sector_size' : 30,  # deg
}

mc_test.generate_PPI_scan('koshava', PPI_cfg)

```

In the above example we configured `koshava` to peform PPI scans of 30 degrees centered at the azimuth angle of 90 degrees at the range of 1 km over 100 m the above ground level. The lidar will perform 10000 virtual `PPI` scans. In this example we omitted supplying any information on kinematic limits of the scanner head. By 'attaching' measurement scenario to the lidar the aforementioned measurement configuration dictionary becomes updated:

```
{'position': array([0, 0, 0]),
 'uncertainty': {'unc_az': {'mu': 0, 'std': 0.1, 'units': 'deg'},
  'unc_el': {'mu': 0, 'std': 0.1, 'units': 'deg'},
  'unc_rng': {'mu': 0, 'std': 5, 'units': 'm'},
  'unc_est': {'mu': 0, 'std': 0.1, 'units': 'm.s^-1'},
  'corr_coef': 0},
 'config': {'scan_type': 'PPI',
  'max_scn_speed': 50,
  'max_scn_acc': 100,
  'scn_speed': 1.0,
  'no_los': 30,
  'no_scans': 10000,
  'sectrsz': 30,
  'scn_tm': 30.0,
  'rtn_tm': 0,
  'az': array([ 75.,  76.,  77.,  78.,  79.,  80.,  81.,  82.,  83.,  84.,  85.,
          86.,  87.,  88.,  89.,  90.,  91.,  92.,  93.,  94.,  95.,  96.,
          97.,  98.,  99., 100., 101., 102., 103., 104.]),
  'el': array([5.73917048, 5.73917048, 5.73917048, 5.73917048, 5.73917048,
         5.73917048, 5.73917048, 5.73917048, 5.73917048, 5.73917048,
         5.73917048, 5.73917048, 5.73917048, 5.73917048, 5.73917048,
         5.73917048, 5.73917048, 5.73917048, 5.73917048, 5.73917048,
         5.73917048, 5.73917048, 5.73917048, 5.73917048, 5.73917048,
         5.73917048, 5.73917048, 5.73917048, 5.73917048, 5.73917048]),
  'rng': array([1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.,
         1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.,
         1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000., 1000.,
         1000., 1000., 1000.])}}
```


Besides the update of the measurement configuration dictionary, a probing [xarray dataset](http://xarray.pydata.org/en/stable/data-structures.html#dataset) was created in this step in the background which fused all information together. It is accessible using the following command:

```
mc_test.data.probing['koshava']
```

Visually our measurement scenario is shown in the figure below.

![mocalum workflow](./assets/sd_scan_perfect.png)

**Figure 2.** Lidar `koshava` performing a virtual `PPI` scan.


### Sampling uncertainties
At this point in the workflow we have added a lidar to the `mocalum` object and configure it to perform virtual measurement scenario. Even though we described the lidar with values of the standard deviation of each uncertainty contributor we still did not actually 'inject' the uncertainties in our measurement scenario. To this, and basically sample uncertainties we need to execute the method `generate_uncertainties`:

```
mc_test.generate_uncertainties('koshava')
```

If we now visually represent the measurement scenario (see Figure 3) we can notice that the previously perfect arc scan (Figure 2) became 'crooked' because the uncertainties in the laser beam positioning have been injected.

![mocalum workflow](./assets/sd_scan.png)

**Figure 2.** Lidar `koshava` performing a virtual `PPI` scan.


As we sample and inject the uncertainties into the probing datasets the actual volume within which the probing takes place will enlarge. The information about bounding box around the measurement points is saved as a python `dictionary`. As such, it represents a convenient way of fetching the information about the dimensions to generate the flow field by internal `mocalum` method. Here is an example of bounding box dict for `koshava` which is accessible via `.data.bbox_meas_pts[lidar_id]`:

```
mc_test.data.bbox_meas_pts['koshava']
```

```
{'CRS': {'x': 'Absolute coordinate, corresponds to Easting in m',
  'y': 'Absolute coordinate, corresponds to Northing in m',
  'z': 'Absolute coordinate, corresponds to height above sea level in m',
  'rot_matrix': array([[ 1.,  0.],
         [-0.,  1.]])},
 'x': {'min': 919.0619309735225,
  'max': 1042.6979161781533,
  'offset': 0,
  'res': 25},
 'y': {'min': -273.7636264604222,
  'max': 292.0006588353198,
  'offset': 0,
  'res': 25},
 'z': {'min': 86.55328669932726,
  'max': 113.27967440963353,
  'offset': 0,
  'res': 5},
 't': {'min': 1.0, 'max': 300000.0, 'offset': 0, 'res': 1.0}}
```

### Generating flow field
Currently, `mocalum`, through the method `generate_flow_field`, provides means to generate:
- `uniform` flow field assuming that wind is only changing with height according to the [power law](https://en.wikipedia.org/wiki/Wind_profile_power_law)
- `turbulent` flow field which is generated using a wrapper around [pyconturb](https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb)

Based on the measurement scenario `mocalum` calculates necessary dimensions and resolution of flow field box (3D for uniform or 4D for turbulent), generates the flow field data and saves them as `xarray` dataset. For this tutorial we will generate `uniform` flow field. The method `generate_flow_field` requires following input parameters:

- `lidar_id` : string or list of string corresponding to lidar ids
- `atmo_cfg` : dictionary describing mean flow parameters, containing following keys:

    - `wind_speed` : mean horizontal wind speed in m/s
    - `upward_velocity` : vertical wind speed in m/s
    - `wind_from_direction` : mean wind direction in deg
    - `reference_height` : height at which wind speed is given in m
    - `shear_exponent` : vertical wind shear exponent

- `flow_type` : flow field type to be generated, it can be `uniform` or `turbulent`

When the method `generate_flow_field` is executed it takes bounding box information, in this case for one lidar, and creates the appropriate size of 3D or 4D data structure, runs the flow model, and populates the data structure with wind vector information. Let's start with `uniform` flow field generation in which the flow only changes with height according to the [power law wind profile](https://en.wikipedia.org/wiki/Wind_profile_power_law). Even though the generated flow field dataset could contain only one dimensional coordinate (i.e, height above the ground level), for the purpose of keeping the `mocalum` backend more generic a 3D dataset is created.

If we visualize the extent of the flow field dataset in 2D we can see that it entails all the measurement points:

![mocalum workflow](./assets/bbox.png)

**Figure 3.** Bounding box around measurement points, red arrow indicates wind direction.


As mentioned earlier `turbulent` flow fields are generated using `pyconturb`. By default, `pyconturb` can generate 3D turbulence box aligned with the mean wind direction, which coordinates are: `time` , height and `y'` which is orthogonal to the wind direction. Usually the length of the generated turbulence box is 600s ~ 10 min. Directly we cannot use the `pyconturb` turbulence box in `mocalum`, since it requires either 3D spatially structured flow field data or 4D (space and time). That's the reason why a wrapper, containing the data wrangler which restructures the `pyconturb` output, was made in `mocalum`. This wrapper converts `pyconturb` 3D turbulence box, which contains a mixture of spatial and time coordinates, first to 3D spatial datasets and then into 4D dataset. The conversion from 3D to 4D is done considering the *Taylor Frozen Turbulence Hypothesis*. Basically, we can view the `time` coordinate as an `x'` coordinate which is inline with the wind direction (see figure below).



![turbulence box](./assets/turb_start.png)
**Figure 4.** Turbulence box with respect to absolute coordinates

The time steps of 3D turbulence box are converted to `x'` coordinates considering the following expressions:

<img src="https://latex.codecogs.com/svg.latex?\Large&space; x'_{i}= V_{mean} * t_{i}" title="x coordinate" />
<br>
<img src="https://latex.codecogs.com/svg.latex?\Large&space; t_{i} = i*\Delta t, i=0,1,..,N" title="x coordinate" />

If we have a long enough turbulence box we can perform a sliding window slicing, where the window size is sufficient to cover the measurement points, and convert 3D into 4D turbulence box:


![3D into 4D turbulence box](./assets/3D_to_4d.png)

**Figure 5.** From 3D to 4D dataset


This is exactly what `mocalum` is doing. Prequel to the data wrangling, `mocalum` considers the bounding boxes around the measurement points and efficiently configures `pyconturb` to generate the initial turbulence box.

### Wind reconstruction
### Uncertainty analysis