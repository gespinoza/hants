# -*- coding: utf-8 -*-
"""
Authors: Gonzalo E. Espinoza-Dávalos
Contact: g.espinoza@un-ihe.org, gespinoza@utexas.edu
Repository: https://github.com/gespinoza/davgis
Module: davgis

Description:
This module is a python wrapper to simplify scripting and automation of common
GIS workflows used in water resources.

"""

from .functions import (Buffer, Feature_to_Raster, List_Fields,
                        Raster_to_Array, Resample, Array_to_Raster, Clip,
                        Raster_to_Points, Add_Field, Spatial_Reference,
                        List_Datasets, NetCDF_to_Raster, Apply_Filter,
                        Extract_Band, Get_Extent, Interpolation_Default,
                        Kriging_Interpolation_Points,
                        Kriging_Interpolation_Array)

__all__ = ['Buffer', 'Feature_to_Raster', 'List_Fields', 'Raster_to_Array',
           'Resample', 'Array_to_Raster', 'Clip', 'Raster_to_Points',
           'Add_Field', 'Spatial_Reference', 'List_Datasets',
           'NetCDF_to_Raster', 'Apply_Filter', 'Extract_Band', 'Get_Extent',
           'Interpolation_Default', 'Kriging_Interpolation_Points',
           'Kriging_Interpolation_Array']

__version__ = '0.1'
