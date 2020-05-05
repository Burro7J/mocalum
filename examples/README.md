# **mocalum examples**

Here you will find examples on how to use [mocalum](https://gitlab-internal.windenergy.dtu.dk/e-windlidar/mocalum) provided as [jupyter](https://jupyter.org/) notebooks. The purpose of the examples are to familiarize users with [mocalum](https://gitlab-internal.windenergy.dtu.dk/e-windlidar/mocalum) and enable them to quickly build there own workflows with it. The examples cover various usage of [mocalum](https://gitlab-internal.windenergy.dtu.dk/e-windlidar/mocalum) and they are provided as 6 tutorials.


## **Tutorial 1: getting started with `mocalum`**
This [tutorial](./tutorial-01) introduces the basic building of `mocalum`, such as:

 - Adding a lidar to the `mocalum` object
 - Configuring lidar measurement scenario
 - Configuring and generating lidar uncertainty contributors
 - Configuring and generating flow field
 - Calculating radial velocity
 - Reconstructing wind speed
 - Accessing generated `mocalum` specific `xarray` datasets

## **Tutorial 2: multi lidars**
In [tutorial 1](./tutorial-01) we worked with a single lidar performing aPPI scan. This [tutorial](./tutorial-02) explains how to configure `mocalum` to work with multi lidars. Specifically it covers following topics:
 - Adding multiple lidars to the `mocalum` object
 - Configuring measurement scenario for multi lidars
 - Generation of turbulent flow field for multi lidars
 - Calculating radial velocity for multi lidars
 - Wind speed reconstruction for multi lidars
 - Simple plotting of results


## **Tutorial 3: generation of uniform and turbulent flow field**
This [tutorial](./tutorial-03) explains:
 - Background on generation of uniform and turbulent flow fields
 - Access to the generated flow fields
 - Validation of flow field data


## **Tutorial 4: Monte-Carlo simulation for single-Doppler configuration** <a name = "single-Doppler"></a>
This [tutorial](./tutorial-04) explains:
 - Configuration of `IVAP` measurement scenario
 - Generation of 4D turbulent flow field
 - Calculation of wind speed uncertainty for sector scanning lidar considering correlated uncertainty terms
 - Calculation of wind speed uncertainty for sector scanning lidar considering uncorrelated uncertainty terms

## **Tutorial 5: Monte-Carlo simulation for dual-Doppler configuration** <a name = "dual-Doppler"></a>
This [tutorial](./tutorial-05) explains:
 - Configuration of dual-Doppler measurement scenario
 - Generation of 4D turbulent flow field
 - Calculation of wind speed uncertainty for sector scanning lidar considering correlated uncertainty terms
 - Calculation of wind speed uncertainty for sector scanning lidar considering uncorrelated uncertainty terms

## **Tutorial 6: Comparison of single- and dual- Doppler configurations**
This [tutorial](./tutorial-06) joins [tutorial 4](./tutorial-04) and [tutorial 5](./tutorial-05), thus showing which lidar configuration is more accurate when comes to wind speed retrieval.
