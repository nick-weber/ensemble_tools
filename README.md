ensemble_tools
==============

Tools for downloading, processing, and analyzing operational ensemble forecasts.

--> process_cfsv2.py --> Contains functions for downloading CFSv2 ensemble forecast gribs and converting them to netcdfs.

--> cfs.table --> A nc_table text file used in process_cfsv2.py during the wgrib2 conversion of gribs to netcdfs.

--> ensemble.py --> Contains the Ensemble class for storing and manipulating ensemble forecast data in a netcdf-like format.

--> plotting_cfs.py --> Contains the Plotter class which is used, in conjuntion with the Ensemble class, to plot CFSv2 ensemble forecast graphics.

--> stids.csv --> CSV file containing lat/lon locations of wx stations.
