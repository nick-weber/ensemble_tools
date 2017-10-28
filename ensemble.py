#!/usr/bin/env python
import numpy as np
import xarray
from datetime import datetime, timedelta
from copy import deepcopy
import os
from netCDF4 import date2num, num2date


class Ensemble(xarray.Dataset):
    """
    xarray Dataset wrapper class to store and manipulate CFSv2 operational ensemble forecast data
    """
    @classmethod
    def from_netcdfs(cls, idate, ncdir, filetag='cfsv2ens*.nc', model='CFSv2', chunks={'time':28}):
        """
        Initializes an Ensemble class from a series of netcdf files, each containing forecast
        data for an individual member of the 16-member CFSv2 operational ensemble
        
        Requires:
        idate ----> the day on which all ensemble members were initialized (datetime object)
        ncdir ----> directory containing the processed netcdf files
        filetag --> ls wildcard used to list the proper netcdf files in the ncdir
        chunks ---> dictionary describing how to chunk the data (xarray implements dask)
        
        Returns:
        ensemble -> an instance of this Ensemble class
        """
        # Load the forecasts from the individual member files and combine along a new dimension 'ens'
        ensemble = xarray.open_mfdataset('{}/{}'.format(ncdir, filetag), concat_dim='ens', 
                                    chunks=chunks, autoclose=True, decode_cf=False)
        ensemble = ensemble.assign_coords(ens=np.arange(ensemble.dims['ens'])+1)
        # Change precipitation from mm/s to mm/h
        if 'prate1h' in ensemble.variables.keys():
            ensemble['prate1h'] *= 3600.
            ensemble.update(ensemble.assign(prate1d=ensemble.variables['prate1h']*24.))
        # Change temperature from K to C
        if 't2m' in ensemble.variables.keys():
            ensemble['t2m'] -= 273.15
        # Assign attributes
        ensemble.attrs.update(idate=idate, dt=6, workdir=ncdir, model=model)
        ensemble.__class__ = cls
        return ensemble
    
    @classmethod
    def from_ensemble_netcdf(cls, ncfile, chunks={'time':28}):
        """
        Initializes an Ensemble class from a single netcdf file that was produced using
        the save_to_disk() method in this Ensemble class
        
        Requires:
        ncfile ---> the netcdf file to be loaded (full path)
        chunks ---> dictionary describing how to chunk the data (xarray implements dask)
        
        Returns:
        ensemble -> an instance of this Ensemble class
        """
        # Open the netcdf file
        ensemble = xarray.open_dataset(ncfile, chunks=chunks, autoclose=True, decode_cf=False)
        # Convert the 'idate' attribute from a float to a datetime object
        ensemble.attrs['idate'] = num2date(ensemble.attrs['idate'], 'hours since 1800-01-01')
        ensemble.__class__ = cls
        return ensemble
        
    #==== Functions to get various useful attributes ==========================
    def model(self):
        return self.attrs['model']
    def idate(self):
        return self.attrs['idate']
    def dt(self):
        return self.attrs['dt']
    def workdir(self):
        return self.attrs['workdir']
    def dx(self):
        return self['lon'].values[1] - self['lon'].values[0]
    def dy(self):
        return self['lat'].values[1] - self['lat'].values[0]
    def ntimes(self):    
        return self.dims['time']
    def ny(self):
        return self.dims['latitude']
    def nx(self):
        return self.dims['longitude']
    def nmems(self):
        return self.dims['ens']
    def vars(self):
        return [x for x in self.variables.keys() if x not in ['latitude', 'longitude', 'time', 'p']]
    def nvars(self):
        return len(self.vars())
    def vdates(self):
        return np.array([self.idate() +  timedelta(days=1) + timedelta(hours=int(t*self.dt())) for t in range(self.ntimes())])
    def leadtimes(self):
        return [timedelta_hours(self.idate(), d) for d in self.vdates()]
        
#==== Get the lat/lon grid and area weights ==================================
    def latlons(self):
        """ Returns 1D lat and lon grids """
        return self['latitude'].values, self['longitude'].values
    
    def area_weights(self, asdataarray=False):
        if asdataarray:
            return np.cos(np.radians(self['latitude']))
        else:
            return np.cos(np.radians(self['latitude'].values))
        
#==== Calculate some ensemble metrics ========================================
    def ens_mean(self, field=None):
        """ Calculates the ensemble mean """
        if field is None:
            return self.mean(dim='ens')
        else:
            return self[field].mean(dim='ens')
        
    def ens_stdv(self, field=None):
        """ Calculates the ensemble standard deviation """
        if field is None:
            return self.std(dim='ens')
        else:
            return self[field].std(dim='ens')
        
    def ens_var(self, field=None):
        """ Calculates the ensemble variance """
        if field is None:
            return self.var(dim='ens')
        else:
            return self[field].var(dim='ens')
        
#==== Project coordinates for plotting on a map ==============================
    def project_coordinates(self, m):
        """
        Projects the lat-lons onto a map projection
        """
        lo, la = np.meshgrid(self['longitude'].values[:], self['latitude'].values[:])
        return m(lo, la)

#==== Calculate the wind speed at a particular level =========================
    def calculate_windspeed(self, lev='10m'):
        # if a number was given for the level, then use that pressure level
        if type(lev)==int or type(lev)==float:
            u = self['uzonal_{}hPa'.format(lev)].values
            v = self['umeridional_{}hPa'.format(lev)].values
            levstring = '{}hPa'.format(lev)
        # otherwise, assume we want 10-meter winds
        else:
            u = self['u10'].values
            v = self['v10'].values
            levstring = '10m'
        # calculate the wind speed
        wndvar = xarray.DataArray(np.sqrt(u**2+v**2), dims=self['u10'].dims)
        assignvar = { 'wndspd_{}'.format(levstring) : wndvar }
        self.update(self.assign(**assignvar))
            

#==== Calculate the relative vorticity from the U, V winds ===================
    def calculate_relvor(self, lev=850):
        from windspharm.standard import VectorWind

        # Load the full U and V fields
        u_full = self['uzonal_{}hPa'.format(lev)].values
        v_full = self['umeridional_{}hPa'.format(lev)].values
        relvor = np.zeros(np.shape(u_full))

        # Loop through all the valid times and ensemble members
        for i1 in range(np.shape(u_full)[0]):
            for i2 in range(np.shape(u_full)[1]):
                # Create the spherical harmonics vector object
                u = u_full[i1, i2, ::-1, :]  # lats must go from N to S
                v = v_full[i1, i2, ::-1, :]  # lats must go from N to S
                wnd = VectorWind(u, v, gridtype='regular')
                # Calculate the relative vorticity
                relvor[i1, i2, :, :] = wnd.vorticity()[::-1, :]

        # Assign relative vorticity as a new variable
        vorvar = xarray.DataArray(relvor, dims=self['uzonal_{}hPa'.format(lev)].dims)
        assignvar = { 'relvor_{}hPa'.format(lev) : vorvar }
        self.update(self.assign(**assignvar))
    
#==== Resamples the fields temporally and returns the coarsened xarray =======
    def coarsen_temporally(self, new_dt):
        """
        Resamples the dataset at a new, coarser temporal frequency
        """
        assert new_dt % self.dt() == 0
        dt_ratio = int(new_dt / self.dt())
        new_obj = self.isel(time=np.arange(self.ntimes())[::dt_ratio])
        new_obj.attrs['dt'] = new_dt
        return new_obj
    
#==== Meridionally average a field ==========================================
    def hovmoller(self, field=None, lat_i=-15., lat_f=15.):
        """
        Computes and returns a meridionally averaged field or the full dataset
        """
        lats = self['latitude'].values
        yi = nearest_ind(lats, lat_i)
        yf = nearest_ind(lats, lat_f) + 1
        # Either average/return the entire dataset, or just one field
        if field is None:
            latband = self.isel(latitude=range(yi,yf)) * self.area_weights(asdataarray=True)
        else:
            latband = self[field].isel(latitude=range(yi,yf)) * self.area_weights()[None, yi:yf, None]
        return latband.mean(dim='latitude', keep_attrs=True)
        
#==== Average all fields or a single field between two times ==================
    def compute_timemean(self, field=None, dt_i=None, dt_f=None):
        """
        Computes and returns a temporally averaged field or the full dataset
        """
        # If no times are provided, average over the entire time dimension
        if dt_i is None or dt_f is None:
            if field is None:  return self.mean(dim='time', keep_attrs=True)
            else:              return self[field].mean(dim='time', keep_attrs=True)
        # Otherwise, average between the two desired times
        else:
            ti = nearest_ind(self.vdates(), dt_i)
            tf = nearest_ind(self.vdates(), dt_f) + 1
            if field is None:  return self.isel(time=range(ti,tf)).mean(dim='time', keep_attrs=True)
            else:              return self.isel(time=range(ti,tf))[field].mean(dim='time', keep_attrs=True)
        
#==== Average the data to a coarser timescale (e.g., daily, weekly) ===========
    def temporal_average(self, timescale):
        """
        Computes and returns a new Ensemble that has been averaged at a coareser
        temporal scale
        
        'timescale' should be in hours 
        """
        assert timescale % self.dt() == 0
        indiv_times = []
        vdates = self.vdates()
        ntsteps = int(timescale/self.dt())
        # Use the compute_timemean function above to average the data every [timescale] hours
        for t in np.arange(0, self.ntimes()-1, ntsteps):
            avg_1time = self.compute_timemean(dt_i=vdates[t], 
                                              dt_f=vdates[t]+timedelta(hours=timescale-self.dt()))
            indiv_times.append(avg_1time)
        # Combine into one Dataset and assign the updated [dt] attribute
        avgd_data = xarray.concat(indiv_times, dim='time', data_vars='different')
        avgd_data.__class__ = self.__class__
        avgd_data.attrs.update(dt=timescale)
        return avgd_data
        
#==== Fetch the data from a subset of the grid ===============================
    def subset(self, field=None, ll=(-91, -181), ur=(91, 361), aw=False):
        """
        Returns a spatial subset of a field or the entire dataset
        """
        # Get the indices for the spatial subdomain
        lats, lons = self.latlons()
        lats = np.round(lats,1); lons=np.round(lons,1)
        y_inds = np.where((lats>=ll[0])*(lats<=ur[0]))[0]
        x_inds = np.where((lons>=ll[1])*(lons<=ur[1]))[0]
        # Either return the whole dataset, or just one field
        if field is None:
            subset = self.isel(latitude=y_inds, longitude=x_inds)
            subset.__class__ = self.__class__
        else:
            subset = self[field].isel(latitude=y_inds, longitude=x_inds)
        # Optionally apply a latitude-dependent area-weighting to the data
        if aw:
            weights = self.area_weights()
            return subset, weights[y_inds]
        else:
            return subset
        
#==== Average a field within some spatial domain =============================
    def spatial_average(self, field, slat=-91, nlat=91, wlon=-181, elon=361):
        """
        Computes and returns a spatially averaged field
        Default: global mean
        """
        subset, weights = self.subset(field, ll=(slat, wlon), ur=(nlat, elon), aw=True)
        return np.average(subset.values, axis=(-2,-1), 
                          weights=np.tile(weights[:,None],(np.shape(subset)[0],1,np.shape(subset)[-1])))
    
#==== Get the timeseries of a given field at the desired lat/lon =============
    def get_timeseries(self, field, loc, method='nearest', verbose=False):
        """
        Uses NN or linear interpolation to get timeseries of the desired field at a
        specified lat/lon location
        """
        from scipy.interpolate import griddata
        
        lat, lon = loc
        lats, lons = self.latlons()
        # Find the point nearest to the desired location
        if method=='nearest':
            lat_ind = nearest_ind(lats, lat)
            lon_ind = nearest_ind(lons, lon)
            if verbose:
                print('Fetching data at {:.02f}N {:.02f}E'.format(lats[lat_ind], lons[lon_ind]))
            # Return the data at that point
            return self[field].isel(latitude=lat_ind, longitude=lon_ind).values
        
        # OR use the interpolation function for better accuracy
        elif method=='linear':
            if verbose: print('loading full data array for interpolation...')
            # ONLY load the points surrounding the desired point
            xi = nearest_ind(lons, lon)-4
            xf = nearest_ind(lons, lon)+5
            yi = nearest_ind(lats, lat)-4
            yf = nearest_ind(lats, lat)+5
            data = self[field].isel(longitude=range(xi,xf), latitude=range(yi,yf)).values
            sublats = lats[yi:yf]
            sublons = lons[xi:xf]
            time = self.leadtimes()
            # If this is a 4D dataset, do the interpolation for each member separately
            if 'ens' in self.dims:
                ens = np.arange(self.nmems())
                d_interp = np.zeros(np.shape(data)[:2])
                for mem in np.arange(self.dims['ens']):
                    if verbose: print('  interpolating member-{}'.format(int(mem)+1))
                    la, t, lo = np.meshgrid(sublats, time, sublons)
                    if self.dims['ens']==np.shape(d_interp)[0]:
                        d_interp[mem, :] = griddata((t.flatten(), la.flatten(), lo.flatten()), 
                                                    data[mem, :, :, :].flatten(), (list(time), [lat], [lon]))
                    else:
                        d_interp[:, mem] = griddata((t.flatten(), la.flatten(), lo.flatten()), 
                                                    data[:, mem, :, :].flatten(), (list(time), [lat], [lon]))

            # If this is just a single member (or an ensemble mean) do the full interpolation:
            else:
                la, t, lo = np.meshgrid(lats, time, lons)
                d_interp = griddata((t.flatten(), la.flatten(), lo.flatten()), data.flatten(), 
                                           (list(time), [lat], [lon]))
            return d_interp
        else:
            raise IOError('Interpolation method "{}" is invalid!'.format(method))
        
#==== Apply spatial filter to a desired field  ================================
    def spatial_filter(self, field, N=3):
        """Currently only applies a uniform 1-1-1 smoother"""
        # Load the field into memory
        data = self[field].values
        filtdata = np.zeros(np.shape(data))
        
        # Now add together the N adjacent values on all sides
        for xx in range(-N, N+1):
            for yy in range(-N, N+1):
                filtdata += np.roll(data, shift=(yy,xx), axis=(2,3))
        # Now divide by the total number of grid boxes to get the average
        filtdata /= (2*N + 1)**2
        # NaN out the N latitudes closest to the poles (where the roll is non-continuous)
        filtdata[:, :, :N+1, :] = np.nan
        filtdata[:, :, -N:, :] = np.nan
        
        # Now create a new variable with the new filtered data
        assignvar = {'filt_{}'.format(field) : (self[field].dims, filtdata)}
        self.update(self.assign(**assignvar))
        

#==== Bandpass filter a desired field  ========================================
    def bandpass_filter(self, field, freq_i=1/2400., freq_f=1/480., 
                        wavenumbers=None, dim='time'):
        """
        Applies a spatial or temporal bandpass filter to a field
        """
        from numpy.fft import rfft, irfft, fftfreq
        
        # Find the index and interval for the dimension we are filtering over
        dimnum = self[field].dims.index(dim)
        if dim=='time':
            ds = self.dt()
        elif dim=='latitude':
            ds = self['latitude'].values[1] - self['latitude'].values[0]
        elif dim=='longitude':
            ds = self['longitude'].values[1] - self['longitude'].values[0]
        else:
            raise ValueError('invalid dimension {}'.format(dim))
        
        # Take the fft of the desired field
        signal = self[field].values
        W = fftfreq(self[field].shape[dimnum], d=ds)
        f_signal = rfft(signal, axis=dimnum)

        # Zero out the power spectrum outside the desired wavenumber/frequency band   
        cut_f_signal = f_signal.copy()
        if wavenumbers is not None and dim=='longitude':
            cut = np.zeros(np.shape(cut_f_signal))
            cut[:, :, wavenumbers] = 1
            cut_f_signal *= cut
        elif dimnum==0:
            print([(w**-1)/24 for w in W])
            cut_f_signal[(W < freq_i) + (W > freq_f), :, :] = 0
        elif dimnum==1:
            cut_f_signal[:, (W < freq_i) + (W > freq_f), :] = 0
        elif dimnum==2:
            cut_f_signal[:, :, (W < freq_i) + (W > freq_f)] = 0
        else:
            raise ValueError('Invalid dimenion number {}'.format(dimnum))

        # Assign a new variable, containing the filtered data, to the Dataset
        assignvar = {'{}_{}filt'.format(field, dim) : (('time','latitude','longitude'), irfft(cut_f_signal, axis=dimnum))}
        self.update(self.assign(**assignvar))
        
#==== Function to save the xarray Dataset to a netcdf file ====================
    def save_to_disk(self, filename=None):
        # Dump this object to disk
        if filename is None:
            filename = '{}/cfsv2_ensemble_{:%Y%m%d%H}.nc'.format(self.workdir(), self.idate())
        self.attrs['idate'] = date2num(self.idate(), 'hours since 1800-01-01')
        self.to_netcdf(filename)
        self.attrs['idate'] = num2date(self.idate(), 'hours since 1800-01-01')
        
        
        
###############################################################################################
# extra utilities
###############################################################################################

def timedelta_hours(dt_i, dt_f):
    """ Find the number of hours between two dates """
    return int((dt_f-dt_i).days*24 + (dt_f-dt_i).seconds/3600)

def nearest_ind(array, value):
    return int((np.abs(array-value)).argmin())