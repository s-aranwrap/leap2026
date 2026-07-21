# ML-based convective parameterization from km-scale DYAMOND CESM runs 

## Introduction

Atmospheric convection, the vertical movement of heat and moisture in the atmosphere, plays a critical role in large-scale circulation, extreme weather, and global climate change (Lin et al., 2022). However, convection is notoriously difficult to represent in global climate models due to the scale mismatch between convective processes (\~1–10 km) and the resolution of these global models (\~50–100 km). Atmospheric models rely on parametrization schemes to approximate the bulk effect of these subgrid-scale convective processes on each coarse gridcell. Despite decades of convective scheme development, large biases attributable to parametrization choices persist across state-of-the-art models (Huang et al., 2018). Today, convection remains one of the largest sources of uncertainty in climate projections (Bony et al., 2015; Sherwood et al., 2014).

Machine learning offers a promising alternative approach to convective parametrization. Rather than encoding physical assumptions by hand, machine learning models can be trained on high-resolution simulation data to learn the relationships between small-scale convective processes and the large-scale atmospheric state, then used in place of conventional parametrization schemes. This approach has been validated in idealised aquaplanet settings (Brenowitz & Bretherton, 2018, 2019; Yuval & O'Gorman, 2020; Yuval et al., 2021) and more recently extended to realistic geography (Watt-Meyer et al., 2024). These schemes typically involve training on coarse-grained data from km-scale simulations that explicitly simulate convective processes, learning corrections to the temperature and moisture tendencies at each coarse gridcell. A particularly encouraging finding is that this can be achieved with relatively short training datasets, on the order of a few months of high-resolution simulation data. Since these simulations are far too expensive to run over the timescales needed for climate projection, machine learning offers a way to harness their advantages at a fraction of the cost.

Data generated for the DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains (DYAMOND) project could be valuable in training such a model. This project takes advantage of the data generated from the Community Earth System Model (CESM)'s km-scale DYAMOND simulations to learn coarse-grained temperature and precipitation tendencies.



## Methods

### CESM DYAMOND Simulations

We use data from the km-scale CESM run for DYnamics of the Atmospheric general circulation Modeled On Non-hydrostatic Domains (DYAMOND) global storm-resolving model comparison project. The atmospheric model used in this simulation is the Community Atmosphere Model (CAM7), which runs at a resolution of 3.75 km on 58 atmospheric levels with modified CAM physics (no ZM convection scheme, modified CLUBB). This includes 2(3??) runs: DYAMOND1 is a 40-day run starting on 1 August 2016, DYAMOND2 is a 40-day run starting on 20 January 2020, and DYAMOND3 (not used here) is a 1-year run starting on 1 March 2020. These are initialized with ERA5 and coupled to high-resolution (3.75 km) land, ocean, and sea ice.

Tendencies are calculated for each 3-hour interval. 

### Coarse Graining

Atmospheric state variables and tendencies coarse-grained from a resolution of 3.75 km to 1˚ by taking the weighted average by area within each 1˚ x 1˚ box. 
???

### Data

???

Mechanically, mirroring YOG20: at each timestep you use, keep all 180 latitudes, and for each latitude randomly draw a subset of the 360 longitudes, say on the order of 20–25, with the random draw refreshed each timestep so you're not always sampling the same meridians.


### Machine Learning Architecture

???

## Results

???

## References

[ https://www.nature.com/articles/s41467-020-17142-3#Abs1 ]


Bony et al., 2015 [https://www.nature.com/articles/ngeo2398]

Huang et al., 2018 [https://doi.org/10.1007/s00704-017-2078-9]

Lin et al., 2022 [https://www.tandfonline.com/doi/full/10.1080/07055900.2022.2082915#abstract]

Sherwood et al., 2014 [https://www.nature.com/articles/nature12829]


Watt-Meyer et al., 2024 [https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003668]



[https://www.cesm.ucar.edu/sites/default/files/2025-06/2025cesmherrington_0.pdf]
[https://pcmdi.llnl.gov/research/DYAMOND3/]


