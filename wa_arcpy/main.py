# -*- coding: utf-8 -*-
"""
Authors: Gonzalo E. Espinoza-Dávalos, Wim G.M. Bastiaanssen, Boaz Bett, and
         Xueliang Cai
         IHE Delft 2017
Contact: g.espinoza@un-ihe.org
Repository: https://github.com/gespinoza/hants
Module: hants
"""

from __future__ import division
import netCDF4
import pandas as pd
import math
import arcpy
import os
import tempfile
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings


def run_HANTS(rasters_path_inp, name_format,
              start_date, end_date, latlim, lonlim, cellsize, nc_path,
              nb, nf, HiLo, low, high, fet, dod, delta,
              epsg=4326, fill_val=-9999.0,
              rasters_path_out=None, export_hants_only=False):
    '''
    This function runs the python implementation of the HANTS algorithm. It
    takes a folder with geotiffs raster data as an input, creates a netcdf
    file, and optionally export the data back to geotiffs.
    '''
    create_netcdf(rasters_path_inp, name_format, start_date, end_date,
                  latlim, lonlim, cellsize, nc_path,
                  epsg, fill_val)
    HANTS_netcdf(nc_path, nb, nf, HiLo, low, high, fet, dod, delta,
                 fill_val)
    if rasters_path_out:
        export_tiffs(rasters_path_out, nc_path, name_format, export_hants_only)
    return nc_path


def create_netcdf(rasters_path, name_format, start_date, end_date,
                  latlim, lonlim, cellsize, nc_path,
                  epsg=4326, fill_val=-9999.0):
    '''
    This function creates a netcdf file from a folder with geotiffs rasters to
    be used to run HANTS.
    '''
    # Latitude and longitude
    lat_ls = pd.np.arange(latlim[0] + 0.5*cellsize, latlim[1] + 0.5*cellsize,
                          cellsize)
    lat_ls = lat_ls[::-1]  # ArcGIS numpy
    lon_ls = pd.np.arange(lonlim[0] + 0.5*cellsize, lonlim[1] + 0.5*cellsize,
                          cellsize)
    lat_n = len(lat_ls)
    lon_n = len(lon_ls)
    spa_ref = arcpy.SpatialReference(4326).exportToString()
    ll_corner = arcpy.Point(lonlim[0], latlim[0] + cellsize/2.0)  # Note: ----
    # The '+ cellsize/2.0' fixes an arcpy offset on the y-axis on arcgis 10.4.1

    # Rasters
    dates_dt = pd.date_range(start_date, end_date, freq='D')
    dates_ls = [d.strftime('%Y%m%d') for d in dates_dt]
    arcpy.env.workspace = rasters_path
    ras_ls = arcpy.ListRasters()

    # Cell code
    temp_ll_ls = [pd.np.arange(x, x + lon_n)
                  for x in range(1, lat_n*lon_n, lon_n)]
    code_ls = pd.np.array(temp_ll_ls)

    empty_vec = pd.np.empty((lat_n, lon_n))
    empty_vec[:] = fill_val

    # Create netcdf file
    print 'Creating netCDF file...'
    nc_file = netCDF4.Dataset(nc_path, 'w', format="NETCDF4")

    # Create Dimensions
    lat_dim = nc_file.createDimension('latitude', lat_n)
    lon_dim = nc_file.createDimension('longitude', lon_n)
    time_dim = nc_file.createDimension('time', len(dates_ls))

    # Create Variables
    crs_var = nc_file.createVariable('crs', 'i4')
    crs_var.grid_mapping_name = 'latitude_longitude'
    crs_var.crs_wkt = spa_ref

    lat_var = nc_file.createVariable('latitude', 'f8', ('latitude'),
                                     fill_value=fill_val)
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'

    lon_var = nc_file.createVariable('longitude', 'f8', ('longitude'),
                                     fill_value=fill_val)
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'

    time_var = nc_file.createVariable('time', 'l', ('time'),
                                      fill_value=fill_val)
    time_var.standard_name = 'time'
    time_var.calendar = 'gregorian'

    code_var = nc_file.createVariable('code', 'i4', ('latitude', 'longitude'),
                                      fill_value=fill_val)

    outliers_var = nc_file.createVariable('outliers', 'i4',
                                          ('latitude', 'longitude', 'time'),
                                          fill_value=fill_val)
    outliers_var.long_name = 'outliers'

    original_var = nc_file.createVariable('original_values', 'f8',
                                          ('latitude', 'longitude', 'time'),
                                          fill_value=fill_val)
    original_var.long_name = 'original values'

    hants_var = nc_file.createVariable('hants_values', 'f8',
                                       ('latitude', 'longitude', 'time'),
                                       fill_value=fill_val)
    hants_var.long_name = 'hants values'

    combined_var = nc_file.createVariable('combined_values', 'f8',
                                          ('latitude', 'longitude', 'time'),
                                          fill_value=fill_val)
    combined_var.long_name = 'combined values'

    print '\tVariables created'

    # Load data
    lat_var[:] = lat_ls
    lon_var[:] = lon_ls
    time_var[:] = dates_ls
    code_var[:] = code_ls

    # temp folder
    temp_dir = tempfile.mkdtemp()
    bbox = "{0} {1} {2} {3}".format(lonlim[0], latlim[0], lonlim[1], latlim[1])

    # Raster loop
    print '\tExtracting data from rasters...'
    for tt in range(len(dates_ls)):

        # Raster
        ras = name_format.format(dates_ls[tt])

        if ras in ras_ls:
            # Resample
            ras_resampled = os.path.join(temp_dir, 'r_' + ras)
            arcpy.management.Resample(os.path.join(rasters_path, ras),
                                      ras_resampled, cellsize)
            # Clip
            ras_clipped = os.path.join(temp_dir, 'c_' + ras)
            arcpy.management.Clip(ras_resampled, bbox, ras_clipped)

            # Raster to Array
            array = arcpy.RasterToNumPyArray(ras_resampled,
                                             ll_corner, lon_n, lat_n)
            # Store values
            original_var[:, :, tt] = array

        else:
            # Store values
            original_var[:, :, tt] = empty_vec

    # Close file
    nc_file.close()
    print 'NetCDF file created'

    # Return
    return nc_path


def HANTS_netcdf(nc_path, nb, nf, HiLo, low, high, fet, dod, delta,
                 fill_val=-9999.0):
    '''
    This function runs the python implementation of the HANTS algorithm. It
    takes the input netcdf file and fills the 'hants_values',
    'combined_values', and 'outliers' variables.
    '''
    # Read netcdfs
    nc_file = netCDF4.Dataset(nc_path, 'r+')

    time_var = nc_file.variables['time'][:]
    original_values = nc_file.variables['original_values'][:]
    lat_values = nc_file.variables['latitude'][:]
    lon_values = nc_file.variables['longitude'][:]

    [rows, cols, ztime] = original_values.shape
    size_st = cols*rows

    values_hants = pd.np.empty((rows, cols, ztime))
    outliers_hants = pd.np.empty((rows, cols, ztime))

    values_hants[:] = pd.np.nan
    outliers_hants[:] = pd.np.nan

    # Additional parameters
    ni = len(time_var)
    ts = range(ni)

    # Loop
    counter = 1
    print 'Running HANTS...'
    for m in range(rows):
        for n in range(cols):
            print '\t{0}/{1}\tlat: {2}\tlon: {3}'.format(counter, size_st,
                                                         lat_values[m],
                                                         lon_values[n])

            y = pd.np.array(original_values[m, n, :])

            y[~pd.np.isfinite(y)] = fill_val

            [yr, outliers] = HANTS(ni, nb, nf, y, ts, HiLo,
                                   low, high, fet, dod, delta, fill_val)

            values_hants[m, n, :] = yr
            outliers_hants[m, n, :] = outliers

            counter = counter + 1

    nc_file.variables['hants_values'][:] = values_hants
    nc_file.variables['outliers'][:] = outliers_hants
    nc_file.variables['combined_values'][:] = pd.np.where(outliers_hants,
                                                          values_hants,
                                                          original_values)
    # Close netcdf file
    nc_file.close()


def HANTS_singlepoint(nc_path, point, nb, nf, HiLo, low, high, fet, dod,
                      delta, fill_val=-9999.0):
    '''
    This function runs the python implementation of the HANTS algorithm for a
    single point (lat, lon). It plots the fit and returns a data frame with
    the 'original' and the 'hants' time series.
    '''
    # Location
    lonx = point[0]
    latx = point[1]

    nc_file = netCDF4.Dataset(nc_path, 'r')

    time = [pd.to_datetime(i, format='%Y%m%d')
            for i in nc_file.variables['time'][:]]

    lat = nc_file.variables['latitude'][:]
    lon = nc_file.variables['longitude'][:]

    # Check that the point falls within the extent of the netcdf file
    lon_max = max(lon)
    lon_min = min(lon)
    lat_max = max(lat)
    lat_min = min(lat)
    if not (lon_min < lonx < lon_max) or not (lat_min < latx < lat_max):
        warnings.warn('The point lies outside the extent of the netcd file. '
                      'The closest cell is plotted.')
        if lonx > lon_max:
            lonx = lon_max
        elif lonx < lon_min:
            lonx = lon_min
        if latx > lat_max:
            latx = lat_max
        elif latx < lat_min:
            latx = lat_min

    # Get lat-lon index in the netcdf file
    lat_closest = lat.flat[pd.np.abs(lat - latx).argmin()]
    lon_closest = lon.flat[pd.np.abs(lon - lonx).argmin()]

    lat_i = pd.np.where(lat == lat_closest)[0][0]
    lon_i = pd.np.where(lon == lon_closest)[0][0]

    # Read values
    original_values = nc_file.variables['original_values'][lat_i, lon_i, :]

    # Additional parameters
    ni = len(time)
    ts = range(ni)

    # HANTS
    y = pd.np.array(original_values)

    y[~pd.np.isfinite(y)] = fill_val

    [hants_values, outliers] = HANTS(ni, nb, nf, y, ts, HiLo, low, high, fet,
                                     dod, delta, fill_val)
    # Plot
    top = 1.15*max(pd.np.nanmax(original_values),
                   pd.np.nanmax(hants_values))
    bottom = 1.15*min(pd.np.nanmin(original_values),
                      pd.np.nanmin(hants_values))
    ylim = [bottom, top]

    plt.plot(time, hants_values, 'r-', label='HANTS')
    plt.plot(time, original_values, 'b.', label='Original data')

    plt.ylim(ylim[0], ylim[1])
    plt.legend(loc=4)
    plt.xlabel('time')
    plt.ylabel('values')
    plt.gcf().autofmt_xdate()
    plt.axes().set_title('Point: lon {0:.2f}, lat {1:.2f}'.format(lon_closest,
                                                                  lat_closest))
    plt.axes().set_aspect(0.5*(time[-1] - time[0]).days/(ylim[1] - ylim[0]))

    plt.show()

    # Close netcdf file
    nc_file.close()

    # Data frame
    df = pd.DataFrame({'time': time,
                       'original': original_values,
                       'hants': hants_values})

    # Return
    return df


def HANTS(ni, nb, nf, y, ts, HiLo, low, high, fet, dod, delta, fill_val):
    '''
    This function applies the Harmonic ANalysis of Time Series (HANTS)
    algorithm originally developed by the Netherlands Aerospace Centre (NLR)
    (http://www.nlr.org/space/earth-observation/).

    This python implementation was based on two previous implementations
    available at the following links:
    https://codereview.stackexchange.com/questions/71489/harmonic-analysis-of-time-series-applied-to-arrays
    http://nl.mathworks.com/matlabcentral/fileexchange/38841-matlab-implementation-of-harmonic-analysis-of-time-series--hants-
    '''
    # Arrays
    mat = pd.np.zeros((min(2*nf+1, ni), ni))
    # amp = np.zeros((nf + 1, 1))

    # phi = np.zeros((nf+1, 1))
    yr = pd.np.zeros((ni, 1))
    outliers = pd.np.zeros((1, len(y)))

    # Filter
    sHiLo = 0
    if HiLo == 'Hi':
        sHiLo = -1
    elif HiLo == 'Lo':
        sHiLo = 1

    nr = min(2*nf+1, ni)
    noutmax = ni - nr - dod
    # dg = 180.0/math.pi
    mat[0, :] = 1.0

    ang = 2*math.pi*pd.np.arange(nb)/nb
    cs = pd.np.cos(ang)
    sn = pd.np.sin(ang)

    i = pd.np.arange(1, nf+1)
    for j in pd.np.arange(ni):
        index = pd.np.mod(i*ts[j], nb)
        mat[2 * i-1, j] = cs.take(index)
        mat[2 * i, j] = sn.take(index)

    p = pd.np.ones_like(y)
    bool_out = (y < low) | (y > high)
    p[bool_out] = 0
    outliers[bool_out.reshape(1, y.shape[0])] = 1
    nout = pd.np.sum(p == 0)

    if nout > noutmax:
        if pd.np.isclose(y, fill_val).any():
            ready = pd.np.array([True])
            yr = y
            outliers = pd.np.zeros((y.shape[0]), dtype=int)
            outliers[:] = fill_val
        else:
            raise Exception('Not enough data points.')
    else:
        ready = pd.np.zeros((y.shape[0]), dtype=bool)

    nloop = 0
    nloopmax = ni

    while ((not ready.all()) & (nloop < nloopmax)):

        nloop += 1
        za = pd.np.matmul(mat, p*y)

        A = pd.np.matmul(pd.np.matmul(mat, pd.np.diag(p)),
                         pd.np.transpose(mat))
        A = A + pd.np.identity(nr)*delta
        A[0, 0] = A[0, 0] - delta

        zr = pd.np.linalg.solve(A, za)

        yr = pd.np.matmul(pd.np.transpose(mat), zr)
        diffVec = sHiLo*(yr-y)
        err = p*diffVec

        err_ls = list(err)
        err_sort = deepcopy(err)
        err_sort.sort()

        rankVec = [err_ls.index(f) for f in err_sort]

        maxerr = diffVec[rankVec[-1]]
        ready = (maxerr <= fet) | (nout == noutmax)

        if (not ready):
            i = ni - 1
            j = rankVec[i]
            while ((p[j]*diffVec[j] > 0.5*maxerr) & (nout < noutmax)):
                p[j] = 0
                outliers[0, j] = 1
                nout += 1
                i -= 1
                if i == 0:
                    j = 0
                else:
                    j = 1

    return [yr, outliers]


def export_tiffs(rasters_path_out, nc_path, name_format,
                 export_hants_only=False):
    '''
    This function exports the output of the HANTS analysis.
    If 'export_hants_only' is False (default), the output rasters have the best
    value available. Therefore, the cells in the output rasters will have the
    original value for the cells that are not outliers and the hants values for
    the cells that are outliers or the cells where data is not available.
    If 'export_hants_only' is True, the exported rasters have the values
    obtained by the HANTS algorithm disregarding of the original values.
    '''
    # Print
    print 'Exporting...'

    # Create folders
    if not os.path.exists(rasters_path_out):
        os.makedirs(rasters_path_out)
    # Read time data
    nc_file = netCDF4.Dataset(nc_path, 'r')
    time_var = nc_file.variables['time'][:]
    nc_file.close()

    # Output type
    if export_hants_only:
        variable_selected = 'hants_values'
    else:
        variable_selected = 'combined_values'

    # Loop through netcdf file
    for yyyymmdd in time_var:
        print '\t{0}'.format(yyyymmdd)
        output_name = rasters_path_out + os.sep + name_format.format(yyyymmdd)

        temp_lyr_name = 'ras_{0}'.format(yyyymmdd)
        arcpy.MakeNetCDFRasterLayer_md(nc_path, variable_selected,
                                       'longitude', 'latitude',
                                       temp_lyr_name, '#',
                                       'time {0}'.format(yyyymmdd), 'BY_VALUE')
        output_ras = arcpy.Raster(temp_lyr_name)
        output_ras.save(output_name)
        arcpy.management.Delete(temp_lyr_name)

    # Return
    print 'Done'
    return rasters_path_out


def plot_point(nc_path, point, ylim=None):
    '''
    This function plots the original time series and the HANTS time series.
    It can be used to assess the fit.
    '''
    # Location
    lonx = point[0]
    latx = point[1]

    nc_file = netCDF4.Dataset(nc_path, 'r')

    time = [pd.to_datetime(i, format='%Y%m%d')
            for i in nc_file.variables['time'][:]]

    lat = nc_file.variables['latitude'][:]
    lon = nc_file.variables['longitude'][:]

    # Check that the point falls within the extent of the netcdf file
    lon_max = max(lon)
    lon_min = min(lon)
    lat_max = max(lat)
    lat_min = min(lat)
    if not (lon_min < lonx < lon_max) or not (lat_min < latx < lat_max):
        warnings.warn('The point lies outside the extent of the netcd file. '
                      'The closest cell is plotted.')
        if lonx > lon_max:
            lonx = lon_max
        elif lonx < lon_min:
            lonx = lon_min
        if latx > lat_max:
            latx = lat_max
        elif latx < lat_min:
            latx = lat_min

    # Get lat-lon index in the netcdf file
    lat_closest = lat.flat[pd.np.abs(lat - latx).argmin()]
    lon_closest = lon.flat[pd.np.abs(lon - lonx).argmin()]

    lat_i = pd.np.where(lat == lat_closest)[0][0]
    lon_i = pd.np.where(lon == lon_closest)[0][0]

    # Read values
    values_o = nc_file.variables['original_values'][lat_i, lon_i, :]
    values_h = nc_file.variables['hants_values'][lat_i, lon_i, :]

    if not ylim:
        top = 1.15*max(pd.np.nanmax(values_o),
                       pd.np.nanmax(values_h))
        bottom = 1.15*min(pd.np.nanmin(values_o),
                          pd.np.nanmin(values_h))
        ylim = [bottom, top]

    # Plot
    plt.plot(time, values_h, 'r-', label='HANTS')
    plt.plot(time, values_o, 'b.', label='Original data')

    plt.ylim(ylim[0], ylim[1])
    plt.legend(loc=4)
    plt.xlabel('time')
    plt.ylabel('values')
    plt.gcf().autofmt_xdate()
    plt.axes().set_title('Point: lon {0:.2f}, lat {1:.2f}'.format(lon_closest,
                                                                  lat_closest))
    plt.axes().set_aspect(0.5*(time[-1] - time[0]).days/(ylim[1] - ylim[0]))

    plt.show()

    # Close netcdf file
    nc_file.close()

    # Return
    return True


def makediag3d(M):
    '''
    Computing diagonal for each row of a 2d array.
    Reference: http://stackoverflow.com/q/27214027/2459096
    '''
    b = pd.np.zeros((M.shape[0], M.shape[1]*M.shape[1]))
    b[:, ::M.shape[1]+1] = M
    # Return
    return b.reshape(M.shape[0], M.shape[1], M.shape[1])
