# **mocalum examples**

Here you will find examples on how to use [mocalum]() provided as [jupyter](https://jupyter.org/) notebooks. The purpose of the examples are to familiarize users with [mocalum]() and enable them to quickly build there own workflows with it. The examples cover various usage of [mocalum]() and they are provided as `N` tutorials.


## **Tutorial 1: getting started with `mocalum`**
This [tutorial](./tutorial-01) introduces the basic building of `mocalum`, such as:

  - Adding lidars to the `mocalum` object
  - Configuring lidar measurement scenario
  - Configuring and generating lidar uncertainty contributors
  - Configuring and generating flow field
  - Calculating radial velocity
  - Reconstructing wind speed
  - Accessing generated `mocalum` specific `xarray` datasets

## **Tutorial 2: single and multi lidars**
This [tutorial](./tutorial-02) explains how to configure `mocalum` to work with single and multi lidars. Specifically it covers following topics:
- Adding lidars to the `mocalum` object
- Configuring measurement scenario for single and multi lidars
- Generation of flow field for single and multi lidars
- Calculating radial velocity for single and multi lidars
- Wind speed reconstruction for single and multi lidars
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

## **Tutorial 6: Comparison of Monte-Carlo simulation for single- and dual- Doppler configuration**
This [tutorial](./tutorial-06) joins [tutorial 4](./tutorial-04) and [tutorial 5](./tutorial-05), thus showing which lidar configuration is more accurate when comes to wind speed retrieval.
