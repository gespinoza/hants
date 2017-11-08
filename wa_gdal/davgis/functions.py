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

from __future__ import division
import os
import math
import tempfile
import warnings
import ogr
import osr
import gdal
import pandas as pd
import netCDF4
from scipy.interpolate import griddata
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

np = pd.np


def Buffer(input_shp, output_shp, distance):
    """
    Creates a buffer of the input shapefile by a given distance
    """

    # Input
    inp_driver = ogr.GetDriverByName('ESRI Shapefile')
    inp_source = inp_driver.Open(input_shp, 0)
    inp_lyr = inp_source.GetLayer()
    inp_lyr_defn = inp_lyr.GetLayerDefn()
    inp_srs = inp_lyr.GetSpatialRef()

    # Output
    out_name = os.path.splitext(os.path.basename(output_shp))[0]
    out_driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_shp):
        out_driver.DeleteDataSource(output_shp)
    out_source = out_driver.CreateDataSource(output_shp)

    out_lyr = out_source.CreateLayer(out_name, inp_srs, ogr.wkbPolygon)
    out_lyr_defn = out_lyr.GetLayerDefn()

    # Add fields
    for i in range(inp_lyr_defn.GetFieldCount()):
        field_defn = inp_lyr_defn.GetFieldDefn(i)
        out_lyr.CreateField(field_defn)

    # Add features
    for i in range(inp_lyr.GetFeatureCount()):
        feature_inp = inp_lyr.GetNextFeature()
        geometry = feature_inp.geometry()
        feature_out = ogr.Feature(out_lyr_defn)

        for j in range(0, out_lyr_defn.GetFieldCount()):
            feature_out.SetField(out_lyr_defn.GetFieldDefn(j).GetNameRef(),
                                 feature_inp.GetField(j))

        feature_out.SetGeometry(geometry.Buffer(distance))
        out_lyr.CreateFeature(feature_out)
        feature_out = None

    # Save and/or close the data sources
    inp_source = None
    out_source = None

    # Return
    return output_shp


def Feature_to_Raster(input_shp, output_tiff,
                      cellsize, field_name=False, NoData_value=-9999):
    """
    Converts a shapefile into a raster
    """

    # Input
    inp_driver = ogr.GetDriverByName('ESRI Shapefile')
    inp_source = inp_driver.Open(input_shp, 0)
    inp_lyr = inp_source.GetLayer()
    inp_srs = inp_lyr.GetSpatialRef()

    # Extent
    x_min, x_max, y_min, y_max = inp_lyr.GetExtent()
    x_ncells = int((x_max - x_min) / cellsize)
    y_ncells = int((y_max - y_min) / cellsize)

    # Output
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)
    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, gdal.GDT_Int16)

    out_source.SetGeoTransform((x_min, cellsize, 0, y_max, 0, -cellsize))
    out_source.SetProjection(inp_srs.ExportToWkt())
    out_lyr = out_source.GetRasterBand(1)
    out_lyr.SetNoDataValue(NoData_value)

    # Rasterize
    if field_name:
        gdal.RasterizeLayer(out_source, [1], inp_lyr,
                            options=["ATTRIBUTE={0}".format(field_name)])
    else:
        gdal.RasterizeLayer(out_source, [1], inp_lyr, burn_values=[1])

    # Save and/or close the data sources
    inp_source = None
    out_source = None

    # Return
    return output_tiff


def List_Fields(input_lyr):
    """
    Lists the field names of input layer
    """
    # Input
    if isinstance(input_lyr, str):
        inp_driver = ogr.GetDriverByName('ESRI Shapefile')
        inp_source = inp_driver.Open(input_lyr, 0)
        inp_lyr = inp_source.GetLayer()
        inp_lyr_defn = inp_lyr.GetLayerDefn()
    elif isinstance(input_lyr, ogr.Layer):
        inp_lyr_defn = input_lyr.GetLayerDefn()

    # List
    names_ls = []

    # Loop
    for j in range(0, inp_lyr_defn.GetFieldCount()):
        field_defn = inp_lyr_defn.GetFieldDefn(j)
        names_ls.append(field_defn.GetName())

    # Save and/or close the data sources
    inp_source = None

    # Return
    return names_ls


def Raster_to_Array(input_tiff, ll_corner, x_ncells, y_ncells,
                    values_type='float32'):
    """
    Loads a raster into a numpy array
    """
    # Input
    inp_lyr = gdal.Open(input_tiff)
    inp_srs = inp_lyr.GetProjection()
    inp_transform = inp_lyr.GetGeoTransform()
    inp_band = inp_lyr.GetRasterBand(1)
    inp_data_type = inp_band.DataType

    cellsize_x = inp_transform[1]
    rot_1 = inp_transform[2]
    rot_2 = inp_transform[4]
    cellsize_y = inp_transform[5]
    NoData_value = inp_band.GetNoDataValue()

    ll_x = ll_corner[0]
    ll_y = ll_corner[1]

    top_left_x = ll_x
    top_left_y = ll_y - cellsize_y*y_ncells

    # Change start point
    temp_path = tempfile.mkdtemp()
    temp_driver = gdal.GetDriverByName('GTiff')
    temp_tiff = os.path.join(temp_path, os.path.basename(input_tiff))
    temp_source = temp_driver.Create(temp_tiff, x_ncells, y_ncells,
                                     1, inp_data_type)
    temp_source.GetRasterBand(1).SetNoDataValue(NoData_value)
    temp_source.SetGeoTransform((top_left_x, cellsize_x, rot_1,
                                 top_left_y, rot_2, cellsize_y))
    temp_source.SetProjection(inp_srs)

    # Snap
    gdal.ReprojectImage(inp_lyr, temp_source, inp_srs, inp_srs,
                        gdal.GRA_Bilinear)
    temp_source = None

    # Read array
    d_type = pd.np.dtype(values_type)
    out_lyr = gdal.Open(temp_tiff)
    array = out_lyr.ReadAsArray(0, 0, out_lyr.RasterXSize,
                                out_lyr.RasterYSize).astype(d_type)
    array[pd.np.isclose(array, NoData_value)] = pd.np.nan
    out_lyr = None

    return array


def Resample(input_tiff, output_tiff, cellsize, method=None,
             NoData_value=-9999):
    """
    Resamples a raster to a different spatial resolution
    """
    # Input
    inp_lyr = gdal.Open(input_tiff)
    inp_srs = inp_lyr.GetProjection()
    inp_transform = inp_lyr.GetGeoTransform()
    inp_band = inp_lyr.GetRasterBand(1)
    inp_data_type = inp_band.DataType

    top_left_x = inp_transform[0]
    cellsize_x = inp_transform[1]
    rot_1 = inp_transform[2]
    top_left_y = inp_transform[3]
    rot_2 = inp_transform[4]
    cellsize_y = inp_transform[5]
    # NoData_value = inp_band.GetNoDataValue()

    x_tot_n = inp_lyr.RasterXSize
    y_tot_n = inp_lyr.RasterYSize

    x_ncells = int(math.floor(x_tot_n * (cellsize_x/cellsize)))

    y_ncells = int(math.floor(y_tot_n * (-cellsize_y/cellsize)))

    # Output
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)
    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, inp_data_type)
    out_source.GetRasterBand(1).SetNoDataValue(NoData_value)
    out_source.SetGeoTransform((top_left_x, cellsize, rot_1,
                                top_left_y, rot_2, -cellsize))
    out_source.SetProjection(inp_srs)

    # Resampling
    method_dict = {'NearestNeighbour': gdal.GRA_NearestNeighbour,
                   'Bilinear': gdal.GRA_Bilinear,
                   'Cubic': gdal.GRA_Cubic,
                   'CubicSpline': gdal.GRA_CubicSpline,
                   'Lanczos': gdal.GRA_Lanczos,
                   'Average': gdal.GRA_Average,
                   'Mode': gdal.GRA_Mode}

    if method in range(6):
        method_sel = method
    elif method in method_dict.keys():
        method_sel = method_dict[method]
    else:
        warnings.warn('Using default interpolation method: Nearest Neighbour')
        method_sel = 0

    gdal.ReprojectImage(inp_lyr, out_source, inp_srs, inp_srs, method_sel)

    # Save and/or close the data sources
    inp_lyr = None
    out_source = None

    # Return
    return output_tiff


def Array_to_Raster(input_array, output_tiff, ll_corner, cellsize,
                    srs_wkt):
    """
    Saves an array into a raster file
    """
    # Output
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)
    y_ncells, x_ncells = input_array.shape
    gdal_datatype = gdaltype_from_dtype(input_array.dtype)

    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, gdal_datatype)
    out_band = out_source.GetRasterBand(1)
    out_band.SetNoDataValue(-9999)

    out_top_left_x = ll_corner[0]
    out_top_left_y = ll_corner[1] + cellsize*y_ncells

    out_source.SetGeoTransform((out_top_left_x, cellsize, 0,
                                out_top_left_y, 0, -cellsize))
    out_source.SetProjection(str(srs_wkt))
    out_band.WriteArray(input_array)

    # Save and/or close the data sources
    out_source = None

    # Return
    return output_tiff


def Clip(input_tiff, output_tiff, bbox):
    """
    Clips a raster given a bounding box
    """
    # Input
    inp_lyr = gdal.Open(input_tiff)
    inp_srs = inp_lyr.GetProjection()
    inp_transform = inp_lyr.GetGeoTransform()
    inp_band = inp_lyr.GetRasterBand(1)
    inp_array = inp_band.ReadAsArray()
    inp_data_type = inp_band.DataType

    top_left_x = inp_transform[0]
    cellsize_x = inp_transform[1]
    rot_1 = inp_transform[2]
    top_left_y = inp_transform[3]
    rot_2 = inp_transform[4]
    cellsize_y = inp_transform[5]
    NoData_value = inp_band.GetNoDataValue()

    x_tot_n = inp_lyr.RasterXSize
    y_tot_n = inp_lyr.RasterYSize

    # Bounding box
    xmin, ymin, xmax, ymax = bbox

    # Get indices, number of cells, and top left corner
    x1 = max([0, int(math.floor((xmin - top_left_x)/cellsize_x))])
    x2 = min([x_tot_n, int(math.ceil((xmax - top_left_x)/cellsize_x))])
    y1 = max([0, int(math.floor((ymax - top_left_y)/cellsize_y))])
    y2 = min([y_tot_n, int(math.ceil((ymin - top_left_y)/cellsize_y))])

    x_ncells = x2 - x1
    y_ncells = y2 - y1

    out_top_left_x = top_left_x + x1*cellsize_x
    out_top_left_y = top_left_y + y1*cellsize_y

    # Output
    out_array = inp_array[y1:y2, x1:x2]
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)
    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, inp_data_type)
    out_band = out_source.GetRasterBand(1)
    out_band.SetNoDataValue(NoData_value)
    out_source.SetGeoTransform((out_top_left_x, cellsize_x, rot_1,
                                out_top_left_y, rot_2, cellsize_y))
    out_source.SetProjection(inp_srs)
    out_band.WriteArray(out_array)

    # Save and/or close the data sources
    inp_lyr = None
    out_source = None

    # Return
    return output_tiff


def Raster_to_Points(input_tiff, output_shp):
    """
    Converts a raster to a point shapefile
    """
    # Input
    inp_lyr = gdal.Open(input_tiff)
    inp_srs = inp_lyr.GetProjection()
    transform = inp_lyr.GetGeoTransform()
    inp_band = inp_lyr.GetRasterBand(1)

    top_left_x = transform[0]
    cellsize_x = transform[1]
    top_left_y = transform[3]
    cellsize_y = transform[5]
    NoData_value = inp_band.GetNoDataValue()

    x_tot_n = inp_lyr.RasterXSize
    y_tot_n = inp_lyr.RasterYSize

    top_left_x_center = top_left_x + cellsize_x/2.0
    top_left_y_center = top_left_y + cellsize_y/2.0

    # Read array
    array = inp_lyr.ReadAsArray(0, 0, x_tot_n, y_tot_n)  # .astype(pd.np.float)
    array[pd.np.isclose(array, NoData_value)] = pd.np.nan

    # Output
    out_srs = osr.SpatialReference()
    out_srs.ImportFromWkt(inp_srs)
    out_name = os.path.splitext(os.path.basename(output_shp))[0]
    out_driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_shp):
        out_driver.DeleteDataSource(output_shp)
    out_source = out_driver.CreateDataSource(output_shp)

    out_lyr = out_source.CreateLayer(out_name, out_srs, ogr.wkbPoint)
    ogr_field_type = ogrtype_from_dtype(array.dtype)
    Add_Field(out_lyr, "RASTERVALU", ogr_field_type)
    out_lyr_defn = out_lyr.GetLayerDefn()

    # Add features
    for xi in range(x_tot_n):
        for yi in range(y_tot_n):
            value = array[yi, xi]
            if ~pd.np.isnan(value):
                feature_out = ogr.Feature(out_lyr_defn)

                feature_out.SetField2(0, value)

                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(top_left_x_center + xi*cellsize_x,
                               top_left_y_center + yi*cellsize_y)

                feature_out.SetGeometry(point)
                out_lyr.CreateFeature(feature_out)

                feature_out = None

    # Save and/or close the data sources
    inp_lyr = None
    out_source = None

    # Return
    return output_shp


def Add_Field(input_lyr, field_name, ogr_field_type):
    """
    Add a field to a layer using the following ogr field types:
    0 = ogr.OFTInteger
    1 = ogr.OFTIntegerList
    2 = ogr.OFTReal
    3 = ogr.OFTRealList
    4 = ogr.OFTString
    5 = ogr.OFTStringList
    6 = ogr.OFTWideString
    7 = ogr.OFTWideStringList
    8 = ogr.OFTBinary
    9 = ogr.OFTDate
    10 = ogr.OFTTime
    11 = ogr.OFTDateTime
    """

    # List fields
    fields_ls = List_Fields(input_lyr)

    # Check if field exist
    if field_name in fields_ls:
        raise Exception('Field: "{0}" already exists'.format(field_name))

    # Create field
    inp_field = ogr.FieldDefn(field_name, ogr_field_type)
    input_lyr.CreateField(inp_field)

    return inp_field


def Spatial_Reference(epsg, return_string=True):
    """
    Obtain a spatial reference from the EPSG parameter
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    if return_string:
        return srs.ExportToWkt()
    else:
        return srs


def List_Datasets(path, ext):
    """
    List the data sets in a folder
    """
    datsets_ls = []
    for f in os.listdir(path):
        if os.path.splitext(f)[1][1:] == ext:
            datsets_ls.append(f)
    return datsets_ls


def NetCDF_to_Raster(input_nc, output_tiff, ras_variable,
                     x_variable='longitude', y_variable='latitude',
                     crs={'variable': 'crs', 'wkt': 'crs_wkt'}, time=None):
    """
    Extract a layer from a netCDF file and save it as a raster file.

    For temporal netcdf files, use the 'time' parameter as:
    t = {'variable': 'time_variable', 'value': '30/06/2017'}
    """
    # Input
    inp_nc = netCDF4.Dataset(input_nc, 'r')
    inp_values = inp_nc.variables[ras_variable]
    x_index = inp_values.dimensions.index(x_variable)
    y_index = inp_values.dimensions.index(y_variable)

    if not time:
        inp_array = inp_values[:]
    else:
        time_variable = time['variable']
        time_value = time['value']
        t_index = inp_values.dimensions.index(time_variable)
        time_index = list(inp_nc.variables[time_variable][:]).index(time_value)
        if t_index == 0:
            inp_array = inp_values[time_index, :, :]
        elif t_index == 1:
            inp_array = inp_values[:, time_index, :]
        elif t_index == 2:
            inp_array = inp_values[:, :, time_index]
        else:
            raise Exception("The array has more dimensions than expected")

    # Transpose array if necessary
    if y_index > x_index:
        inp_array = pd.np.transpose(inp_array)

    # Additional parameters
    gdal_datatype = gdaltype_from_dtype(inp_array.dtype)
    NoData_value = inp_nc.variables[ras_variable]._FillValue

    if type(crs) == str:
        srs_wkt = crs
    else:
        crs_variable = crs['variable']
        crs_wkt = crs['wkt']
        exec('srs_wkt = str(inp_nc.variables["{0}"].{1})'.format(crs_variable,
                                                                 crs_wkt))

    inp_x = inp_nc.variables[x_variable]
    inp_y = inp_nc.variables[y_variable]

    cellsize_x = abs(pd.np.mean([inp_x[i] - inp_x[i-1]
                                 for i in range(1, len(inp_x))]))
    cellsize_y = -abs(pd.np.mean([inp_y[i] - inp_y[i-1]
                                  for i in range(1, len(inp_y))]))

    # Output
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)

    y_ncells, x_ncells = inp_array.shape

    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, gdal_datatype)
    out_band = out_source.GetRasterBand(1)
    out_band.SetNoDataValue(pd.np.asscalar(NoData_value))

    out_top_left_x = inp_x[0] - cellsize_x/2.0
    if inp_y[-1] > inp_y[0]:
        out_top_left_y = inp_y[-1] - cellsize_y/2.0
        inp_array = pd.np.flipud(inp_array)
    else:
        out_top_left_y = inp_y[0] - cellsize_y/2.0

    out_source.SetGeoTransform((out_top_left_x, cellsize_x, 0,
                                out_top_left_y, 0, cellsize_y))
    out_source.SetProjection(srs_wkt)
    out_band.WriteArray(inp_array)
    out_band.ComputeStatistics(True)

    # Save and/or close the data sources
    inp_nc.close()
    out_source = None

    # Return
    return output_tiff


def Apply_Filter(input_tiff, output_tiff, number_of_passes):
    """
    Smooth a raster by replacing cell value by the average value of the
    surrounding cells
    """
    # Input
    inp_lyr = gdal.Open(input_tiff)
    inp_srs = inp_lyr.GetProjection()
    inp_transform = inp_lyr.GetGeoTransform()
    inp_band = inp_lyr.GetRasterBand(1)
    inp_array = inp_band.ReadAsArray()
    inp_data_type = inp_band.DataType

    top_left_x = inp_transform[0]
    cellsize_x = inp_transform[1]
    rot_1 = inp_transform[2]
    top_left_y = inp_transform[3]
    rot_2 = inp_transform[4]
    cellsize_y = inp_transform[5]
    NoData_value = inp_band.GetNoDataValue()

    x_ncells = inp_lyr.RasterXSize
    y_ncells = inp_lyr.RasterYSize

    # Filter
    inp_array[inp_array == NoData_value] = pd.np.nan
    out_array = array_filter(inp_array, number_of_passes)

    # Output
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)
    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, inp_data_type)
    out_band = out_source.GetRasterBand(1)
    out_band.SetNoDataValue(NoData_value)
    out_source.SetGeoTransform((top_left_x, cellsize_x, rot_1,
                                top_left_y, rot_2, cellsize_y))
    out_source.SetProjection(inp_srs)
    out_band.WriteArray(out_array)

    # Save and/or close the data sources
    inp_lyr = None
    out_source = None

    # Return
    return output_tiff


def Extract_Band(input_tiff, output_tiff, band_number=1):
    """
    Extract and save a raster band into a new raster
    """
    # Input
    inp_lyr = gdal.Open(input_tiff)
    inp_srs = inp_lyr.GetProjection()
    inp_transform = inp_lyr.GetGeoTransform()
    inp_band = inp_lyr.GetRasterBand(band_number)
    inp_array = inp_band.ReadAsArray()
    inp_data_type = inp_band.DataType

    NoData_value = inp_band.GetNoDataValue()

    x_ncells = inp_lyr.RasterXSize
    y_ncells = inp_lyr.RasterYSize

    # Output
    out_driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(output_tiff):
        out_driver.Delete(output_tiff)
    out_source = out_driver.Create(output_tiff, x_ncells, y_ncells,
                                   1, inp_data_type)
    out_band = out_source.GetRasterBand(1)
    out_band.SetNoDataValue(NoData_value)
    out_source.SetGeoTransform(inp_transform)
    out_source.SetProjection(inp_srs)
    out_band.WriteArray(inp_array)

    # Save and/or close the data sources
    inp_lyr = None
    out_source = None

    # Return
    return output_tiff


def Get_Extent(input_lyr):
    """
    Obtain the input layer extent (xmin, ymin, xmax, ymax)
    """
    # Input
    filename, ext = os.path.splitext(input_lyr)
    if ext.lower() == '.shp':
        inp_driver = ogr.GetDriverByName('ESRI Shapefile')
        inp_source = inp_driver.Open(input_lyr)
        inp_lyr = inp_source.GetLayer()
        x_min, x_max, y_min, y_max = inp_lyr.GetExtent()
        inp_lyr = None
        inp_source = None
    elif ext.lower() == '.tif':
        inp_lyr = gdal.Open(input_lyr)
        inp_transform = inp_lyr.GetGeoTransform()
        x_min = inp_transform[0]
        x_max = x_min + inp_transform[1] * inp_lyr.RasterXSize
        y_max = inp_transform[3]
        y_min = y_max + inp_transform[5] * inp_lyr.RasterYSize
        inp_lyr = None
    else:
        raise Exception('The input data type is not recognized')
    return (x_min, y_min, x_max, y_max)


def Interpolation_Default(input_shp, field_name, output_tiff,
                          method='nearest', cellsize=None):
    '''
    Interpolate point data into a raster

    Available methods: 'nearest', 'linear', 'cubic'
    '''
    # Input
    inp_driver = ogr.GetDriverByName('ESRI Shapefile')
    inp_source = inp_driver.Open(input_shp, 0)
    inp_lyr = inp_source.GetLayer()
    inp_srs = inp_lyr.GetSpatialRef()
    inp_wkt = inp_srs.ExportToWkt()

    # Extent
    x_min, x_max, y_min, y_max = inp_lyr.GetExtent()
    ll_corner = [x_min, y_min]
    if not cellsize:
        cellsize = min(x_max - x_min, y_max - y_min)/25.0
    x_ncells = int((x_max - x_min) / cellsize)
    y_ncells = int((y_max - y_min) / cellsize)

    # Feature points
    x = []
    y = []
    z = []
    for i in range(inp_lyr.GetFeatureCount()):
        feature_inp = inp_lyr.GetNextFeature()
        point_inp = feature_inp.geometry().GetPoint()

        x.append(point_inp[0])
        y.append(point_inp[1])
        z.append(feature_inp.GetField(field_name))

    x = pd.np.array(x)
    y = pd.np.array(y)
    z = pd.np.array(z)

    # Grid
    X, Y = pd.np.meshgrid(pd.np.linspace(x_min + cellsize/2.0,
                                         x_max - cellsize/2.0,
                                         x_ncells),
                          pd.np.linspace(y_min + cellsize/2.0,
                                         y_max - cellsize/2.0,
                                         y_ncells))

    # Interpolate
    out_array = griddata((x, y), z, (X, Y), method=method)
    out_array = pd.np.flipud(out_array)

    # Save raster
    Array_to_Raster(out_array, output_tiff, ll_corner, cellsize, inp_wkt)

    # Return
    return output_tiff


def Kriging_Interpolation_Points(input_shp, field_name, output_tiff, cellsize,
                                 bbox=None):
    """
    Interpolate point data using Ordinary Kriging

    Reference: https://cran.r-project.org/web/packages/automap/automap.pdf
    """
    # Spatial reference
    inp_driver = ogr.GetDriverByName('ESRI Shapefile')
    inp_source = inp_driver.Open(input_shp, 0)
    inp_lyr = inp_source.GetLayer()
    inp_srs = inp_lyr.GetSpatialRef()
    srs_wkt = inp_srs.ExportToWkt()
    inp_source = None
    # Temp folder
    temp_dir = tempfile.mkdtemp()
    temp_points_tiff = os.path.join(temp_dir, 'points_ras.tif')
    # Points to raster
    Feature_to_Raster(input_shp, temp_points_tiff,
                      cellsize, field_name, -9999)
    # Raster extent
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        ll_corner = [xmin, ymin]
        x_ncells = int(math.ceil((xmax - xmin)/cellsize))
        y_ncells = int(math.ceil((ymax - ymin)/cellsize))
    else:
        temp_lyr = gdal.Open(temp_points_tiff)
        x_min, x_max, y_min, y_max = temp_lyr.GetExtent()
        ll_corner = [x_min, y_min]
        x_ncells = temp_lyr.RasterXSize
        y_ncells = temp_lyr.RasterYSize
        temp_lyr = None
    # Raster to array
    points_array = Raster_to_Array(temp_points_tiff, ll_corner,
                                   x_ncells, y_ncells, values_type='float32')
    # Run kriging
    x_vector = np.arange(xmin + cellsize/2, xmax + cellsize/2, cellsize)
    y_vector = np.arange(ymin + cellsize/2, ymax + cellsize/2, cellsize)
    out_array = Kriging_Interpolation_Array(points_array, x_vector, y_vector)
    # Save array as raster
    Array_to_Raster(out_array, output_tiff, ll_corner, cellsize, srs_wkt)
    # Return
    return output_tiff


def Kriging_Interpolation_Array(input_array, x_vector, y_vector):
    """
    Interpolate data in an array using Ordinary Kriging

    Reference: https://cran.r-project.org/web/packages/automap/automap.pdf
    """
    # Total values in array
    n_values = np.isfinite(input_array).sum()
    # Load function
    pandas2ri.activate()
    robjects.r('''
                library(gstat)
                library(sp)
                library(automap)
                kriging_interpolation <- function(x_vec, y_vec, values_arr,
                                                  n_values){
                  # Parameters
                  shape <- dim(values_arr)
                  counter <- 1
                  df <- data.frame(X=numeric(n_values),
                                   Y=numeric(n_values),
                                   INFZ=numeric(n_values))
                  # Save values into a data frame
                  for (i in seq(shape[2])) {
                    for (j in seq(shape[1])) {
                      if (is.finite(values_arr[j, i])) {
                        df[counter,] <- c(x_vec[i], y_vec[j], values_arr[j, i])
                        counter <- counter + 1
                      }
                    }
                  }
                  # Grid
                  coordinates(df) = ~X+Y
                  int_grid <- expand.grid(x_vec, y_vec)
                  names(int_grid) <- c("X", "Y")
                  coordinates(int_grid) = ~X+Y
                  gridded(int_grid) = TRUE
                  # Kriging
                  krig_output <- autoKrige(INFZ~1, df, int_grid)
                  # Array
                  values_out <- matrix(krig_output$krige_output$var1.pred,
                                       nrow=length(y_vec),
                                       ncol=length(x_vec),
                                       byrow = TRUE)
                  return(values_out)
                }
                ''')
    kriging_interpolation = robjects.r['kriging_interpolation']
    # Execute kriging function and get array
    r_array = kriging_interpolation(x_vector, y_vector, input_array, n_values)
    array_out = np.array(r_array)
    # Return
    return array_out


def get_neighbors(x, y, nx, ny, cells=1):
    """
    Get a list of neighboring cells
    """
    neighbors_ls = [(xi, yi)
                    for xi in range(x - 1 - cells + 1, x + 2 + cells - 1)
                    for yi in range(y - 1 - cells + 1, y + 2 + cells - 1)
                    if (-1 < x <= nx - 1 and -1 < y <= ny - 1 and
                        (x != xi or y != yi) and
                        (0 <= xi <= nx - 1) and (0 <= yi <= ny - 1))]
    return neighbors_ls


def get_mean_neighbors(array, index, include_cell=False):
    """
    Get the mean value of neighboring cells
    """
    xi, yi = index
    nx, ny = array.shape
    stay = True
    cells = 1
    while stay:
        neighbors_ls = get_neighbors(xi, yi, nx, ny, cells)
        if include_cell:
            neighbors_ls = neighbors_ls + [(xi, yi)]
        values_ls = [array[i] for i in neighbors_ls]
        if pd.np.isnan(values_ls).all():
            cells += 1
        else:
            value = pd.np.nanmean(values_ls)
            stay = False
    return value


def array_filter(array, number_of_passes=1):
    """
    Smooth cell values by replacing each cell value by the average value of the
    surrounding cells
    """
    while number_of_passes >= 1:
        ny, nx = array.shape
        arrayf = pd.np.empty(array.shape)
        arrayf[:] = pd.np.nan
        for j in range(ny):
            for i in range(nx):
                arrayf[j, i] = get_mean_neighbors(array, (j, i), True)
        array[:] = arrayf[:]
        number_of_passes -= 1
    return arrayf


def ogrtype_from_dtype(d_type):
    """
    Return the ogr data type from the numpy dtype
    """
    # ogr field type
    if 'float' in d_type.name:
        ogr_data_type = 2
    elif 'int' in d_type.name:
        ogr_data_type = 0
    elif 'string' in d_type.name:
        ogr_data_type = 4
    elif 'bool' in d_type.name:
        ogr_data_type = 8
    else:
        raise Exception('"{0}" is not recognized'.format(d_type))
    return ogr_data_type


def gdaltype_from_dtype(d_type):
    """
    Return the gdal data type from the numpy dtype
    """
    # gdal field type
    if 'int8' == d_type.name:
        gdal_data_type = 1
    elif 'uint16' == d_type.name:
        gdal_data_type = 2
    elif 'int16' == d_type.name:
        gdal_data_type = 3
    elif 'uint32' == d_type.name:
        gdal_data_type = 4
    elif 'int32' == d_type.name:
        gdal_data_type = 5
    elif 'float32' == d_type.name:
        gdal_data_type = 6
    elif 'float64' == d_type.name:
        gdal_data_type = 7
    elif 'bool' in d_type.name:
        gdal_data_type = 1
    elif 'int' in d_type.name:
        gdal_data_type = 5
    elif 'float' in d_type.name:
        gdal_data_type = 7
    elif 'complex' == d_type.name:
        gdal_data_type = 11
    else:
        warnings.warn('"{0}" is not recognized. '
                      '"Unknown" data type used'.format(d_type))
        gdal_data_type = 0
    return gdal_data_type
