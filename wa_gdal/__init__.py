# -*- coding: utf-8 -*-
"""
Authors: Gonzalo E. Espinoza-Dávalos, Wim G.M. Bastiaanssen, Boaz Bett, and
         Xueliang Cai
         IHE Delft 2017
Contact: g.espinoza@un-ihe.org
Repository: https://github.com/gespinoza/hants
Module: hants

Description:
This module is an implementation of the Harmonic ANalysis of Time Series
(HANTS) applied to geographic data. This module can be used to perform the
HANTS analysis to a collection of time-variable raster data at each pixel.

There are two equivalent options to run HANTS:
- gdal: for this option use 'from hants import wa_gdal'
- arcpy: for this option use 'from hants import wa_arcpy'
The only difference between the two options is the underlying library to
process the geographic data. The HANTS algorithm is the same.

Example 1:
from hants import wa_gdal
# Data parameters
rasters_path = r'C:\example\data'
name_format = 'PROBAV_S1_TOC_{0}_100M_V001.tif'
start_date = '2015-08-01'
end_date = '2016-07-28'
latlim = [11.4505, 11.4753]
lonlim = [108.8605, 108.8902]
cellsize = 0.00099162627
nc_path = r'C:\example\ndvi_probav.nc'
rasters_path_out = r'C:\example\output_rasters'
# HANTS parameters
nb = 365
nf = 3
low = -1
high = 1
HiLo = 'Lo'
fet = 0.05
delta = 0.25
dod = 1
# Run
wa_gdal.run_HANTS(rasters_path, name_format,
                  start_date, end_date, latlim, lonlim, cellsize, nc_path,
                  nb, nf, HiLo, low, high, fet, dod, delta,
                  4326, -9999.0, rasters_path_out)
# Check fit
point = [108.87, 11.47]
ylim = [-1, 1]
wa_gdal.plot_point(nc_path, point, ylim)

Example 2:
from hants import wa_gdal
# Create netcdf file
rasters_path = r'C:\example\data'
name_format = 'PROBAV_S1_TOC_{0}_100M_V001.tif'
start_date = '2015-08-01'
end_date = '2016-07-28'
latlim = [11.4505, 11.4753]
lonlim = [108.8605, 108.8902]
cellsize = 0.00099162627
nc_path = r'C:\example\ndvi_probav.nc'
wa_gdal.create_netcdf(rasters_path, name_format, start_date, end_date,
                      latlim, lonlim, cellsize, nc_path)
# Run HANTS
nb = 365
nf = 3
low = -1
high = 1
HiLo = 'Lo'
fet = 0.05
delta = 0.25
dod = 1
wa_gdal.HANTS_netcdf(nc_path, nb, nf, HiLo, low, high, fet, dod, delta)
# Check fit
point = [108.87, 11.47]
ylim = [-1, 1]
wa_gdal.plot_point(nc_path, point, ylim)
# Export rasters
rasters_path_out = r'C:\example\output_rasters'
wa_gdal.export_tiffs(rasters_path_out, nc_path, name_format)

"""

from .main import (run_HANTS, create_netcdf, HANTS_netcdf, HANTS,
                   export_tiffs, plot_point)

__all__ = ['run_HANTS', 'create_netcdf', 'HANTS_netcdf', 'HANTS',
           'export_tiffs', 'plot_point']

__version__ = '0.1'
