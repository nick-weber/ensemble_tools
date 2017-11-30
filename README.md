ensemble_tools
==============

Tools for downloading, processing, and analyzing operational ensemble forecasts.

Contents:
---------

- **``ensemble.py``**: contains the ``GlobalEnsemble`` and ``RegionalEnsemble`` classes, which are [``xarray.Dataset``](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html) wrapper classes that point to multidimensional [member, time, latitude, longitude] ensemble forecast data. The classes contain a wide range of functions for manipulating, displaying, and performing computations with the model fields.

- **``ens_utilities``**: module containing a functions that are used by a number of modules in this package.

- **``plotting.py``**: contains the ``Plotter`` class, which is used to create ensemble graphics using data stored the ``GlobalEnsemble`` and ``RegionalEnsemble`` objects.

- **``uw_wrf/``**: package containing tools used specifically for working with the University of Washington operational regional WRF ensemble.

   - **``plumes.py``**: contains the ``Plumes`` class, which is for loading/plotting UW WRF ensemble plumes for numerous stations. Data for this class is loaded from netCDFs (created with a Fortran script written by David Ovens) which contain WRF forecast data interpolated to 169 Pacific Northwest station sites.

   - **``tools.py``**: a module of functions used to load UW WRF ensemble data.

   - **``rename_vars.py``**: simply contains a dictionary used to rename WRF variables using more readable names.

   - **``members.txt``**: text file containing the names of the UW WRF ensemble members.

   - **``new_wrf_plotting_functions/``**: [package written by Luke Madaus](https://github.com/lmadaus/new_wrf_plotting_functions) for computing variables using WRF fields.

- **``tracking/``**: package for tracking extratropical cyclones using global ensemble fields.

   - **``cyclonetracks.py``**: consists of four List/Dictionary-like classes (``CycloneCenter``, ``CycloneTrack``, ``TrackCluster``, and ``EnsembleTrackClusters``) used to store information about cyclones, tracks, and ensemble clusters of tracks.

   - **``tracker.py``**: module containing the cyclone tracking algorithm (adaped from [Flaounas et al. 2017](https://www.geosci-model-dev.net/7/1841/2014/gmd-7-1841-2014.pdf).

- **``process_cfsv2.py``**: used for downloading and converting [CFSv2](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/climate-forecast-system-version2-cfsv2) grib files.

- **``cfs.table``**: text file used by the [wgrib2 utility](http://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/) in ``process_cfsv2.py`` to convert grib files to netCDFs.

- **``stids.csv``**: text file containing ASOS station identifiers and their associated lat/lon coordinates (for creating ensemble plumes etc.). 
