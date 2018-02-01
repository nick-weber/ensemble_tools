#!/usr/bin/env python
import numpy as np
import xarray
from datetime import datetime, timedelta
from copy import deepcopy
import os
from netCDF4 import date2num, num2date
import ens_utilities as ut

###############################################################################################
# Class for a global model on a regular grid
###############################################################################################

class GlobalEnsemble(xarray.Dataset):
    """
    xarray Dataset wrapper class to store and manipulate global operational ensemble forecast data
    """
    @classmethod
    def from_NCEP_netcdfs(cls, idate, ncdir, filetag='cfsv2ens*.nc', model='CFSv2', chunks={'time':28}):
        """
        Initializes a GlobalEnsemble class from a series of netcdf files, each containing forecast
        data for an individual member an NCEP operational ensemble (e.g., GEFS, CFSv2)
        
        Requires:
        idate ----> the day on which all ensemble members were initialized (datetime object)
        ncdir ----> directory containing the processed netcdf files
        filetag --> ls wildcard used to list the proper netcdf files in the ncdir
        chunks ---> dictionary describing how to chunk the data (xarray implements dask)
        
        Returns:
        ensemble -> an instance of this GlobalEnsemble class
        """
        # Load the forecasts from the individual member files and combine along a new dimension 'ens'
        ensemble = xarray.open_mfdataset('{}/{}'.format(ncdir, filetag), concat_dim='ens', 
                                    chunks=chunks, autoclose=True, decode_cf=False)
        ensemble = ensemble.assign_coords(ens=np.arange(ensemble.dims['ens'])+1)
        # Perform necessary calculation(s) to ensure precip rate is in mm/h
        if 'prate1h' in ensemble.variables.keys() and model=='CFSv2':
            ensemble['prate1h'] *= 3600.
            ensemble.update(ensemble.assign(prate1d=ensemble.variables['prate1h']*24.))
        elif model=='GEFS':
            ensemble.update(ensemble.assign(prate1h=ensemble.variables['ptotal']/6.))
        # Assign attributes
        ensemble.attrs.update(idate=idate, dt=6, workdir=ncdir, model=model)
        ensemble.__class__ = cls
        return ensemble
    
    @classmethod
    def from_ensemble_netcdf(cls, ncfile, chunks={'time':28}):
        """
        Initializes a GlobalEnsemble class from a single netcdf file that was produced using
        the save_to_disk() method in this GlobalEnsemble class
        
        Requires:
        ncfile ---> the netcdf file to be loaded (full path)
        chunks ---> dictionary describing how to chunk the data (xarray implements dask)
        
        Returns:
        ensemble -> an instance of this GlobalEnsemble class
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
        return self['longitude'].values[1] - self['longitude'].values[0]
    def dy(self):
        return self['latitude'].values[1] - self['latitude'].values[0]
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
        if self.model()=='CFSv2':
            return np.array([self.idate() +  timedelta(days=1) + timedelta(hours=int(t*self.dt())) \
                             for t in range(self.ntimes())])
        else:
            return np.array([self.idate() + timedelta(hours=int(t*self.dt())) for t in range(self.ntimes())])
    def leadtimes(self):
        return [ut.timedelta_hours(self.idate(), d) for d in self.vdates()]
        
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
            u = self['u_{}hPa'.format(lev)].values
            v = self['v_{}hPa'.format(lev)].values
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
        u_full = self['u_{}hPa'.format(lev)].values
        v_full = self['v_{}hPa'.format(lev)].values
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
        vorvar = xarray.DataArray(relvor, dims=self['u_{}hPa'.format(lev)].dims)
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
        yi = ut.nearest_ind(lats, lat_i)
        yf = ut.nearest_ind(lats, lat_f) + 1
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
            ti = ut.nearest_ind(self.vdates(), dt_i)
            tf = ut.nearest_ind(self.vdates(), dt_f) + 1
            if field is None:  return self.isel(time=range(ti,tf)).mean(dim='time', keep_attrs=True)
            else:              return self.isel(time=range(ti,tf))[field].mean(dim='time', keep_attrs=True)
        
#==== Average the data to a coarser timescale (e.g., daily, weekly) ===========
    def temporal_average(self, timescale):
        """
        Computes and returns a new GlobalEnsemble that has been averaged at a coareser
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
            lat_ind = ut.nearest_ind(lats, lat)
            lon_ind = ut.nearest_ind(lons, lon)
            if verbose:
                print('Fetching data at {:.02f}N {:.02f}E'.format(lats[lat_ind], lons[lon_ind]))
            # Return the data at that point
            return self[field].isel(latitude=lat_ind, longitude=lon_ind).values
        
        # OR use the interpolation function for better accuracy
        elif method=='linear':
            if verbose: print('loading full data array for interpolation...')
            # ONLY load the points surrounding the desired point
            xi = ut.nearest_ind(lons, lon)-4
            xf = ut.nearest_ind(lons, lon)+5
            yi = ut.nearest_ind(lats, lat)-4
            yf = ut.nearest_ind(lats, lat)+5
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
            # We want the 'ens' dimension to be the first dimension
            if self.dims['ens']==np.shape(d_interp)[1]:
                d_interp = np.swap_axes(d_interp, 0, 1)
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
            filename = '{}/{}_ensemble_{:%Y%m%d%H}.nc'.format(self.model(), self.workdir(), self.idate())
        self.attrs['idate'] = date2num(self.idate(), 'hours since 1800-01-01')
        self.to_netcdf(filename)
        self.attrs['idate'] = num2date(self.idate(), 'hours since 1800-01-01')
        
        
###############################################################################################
# Class for a regional model on an irregular grid
###############################################################################################

class RegionalEnsemble(xarray.Dataset):
    """
    xarray Dataset wrapper class to store and manipulate regional operational ensemble forecast data
    """
    @classmethod
    def from_wrfouts(cls, idate, wrfdir, domain=1, memberfile='uw_wrf/members.txt', 
                     dt=1, verbose=False):
        """
        Initializes a RegionalEnsemble class from an ensemble of WRF output files, with each member
        stored in a separate subdirectory within [wrfdir]
        
        Requires:
        idate ------> the day on which all ensemble members were initialized (datetime object)
        wrfdir -----> directory containing the ensemble member subdirectories
        domain -----> WRF domain identifies (int)
        memberfile -> name of the text file containing the names of the ensemble members
        chunks -----> dictionary describing how to chunk the data (xarray implements dask)
        
        Returns:
        ensemble -> an instance of this RegionalEnsemble class
        """
        from uw_wrf.tools import load_ensemble_xr, load_ensemble_allhrs
        
        # Get the ensemble member directories from a text file
        mems = np.genfromtxt(memberfile, dtype=str)
        
        # Load the desired WRF forecasts as xarray objects
        try:
            ensemble, missing_mems = load_ensemble_allhrs(idate, wrfdir, mems, dom=domain, returnmissing=True, verbose=verbose)
        except IOError as err:
            if verbose: print('ERROR in "load_ensemble_allhrs()":', err.args[0])
            ensemble, missing_mems = load_ensemble_xr(wrfdir, mems, dom=domain, returnmissing=True, verbose=verbose)
        
        # Assign attributes
        ensemble.attrs.update(idate=idate, dt=dt, workdir=wrfdir, model='WRF', domain=domain,
                              missing_mems=missing_mems)
        ensemble.__class__ = cls
        ensemble.attrs.update(bmap=ensemble.get_map_projection())
        
        ## Calculate the rain RATE and snow RATE
        #if verbose: print('Computing precip/snow rates...')
        #precip = ensemble['rainnc'] + ensemble['rainc']
        #print(precip)
        #ensemble = ensemble.assign(precip=precip)
        #prate = (ensemble['precip'] - ensemble['precip'].shift(Time=1)) / ensemble.dt()  #  mm/hr
        #srate = 10.* ((ensemble['snownc'] - ensemble['snownc'].shift(Time=1)) / ensemble.dt()) / 25.4 # in/hr
        #ensemble = ensemble.assign(prate=prate, srate=srate)
        
        return ensemble
        
    #==== Functions to get various useful attributes ==========================
    def bmap(self):
        return self.attrs['bmap']
    def domain(self):
        return self.attrs['domain']
    def model(self):
        return self.attrs['model']
    def idate(self):
        return self.attrs['idate']
    def dt(self):
        return self.attrs['dt']
    def workdir(self):
        return self.attrs['workdir']
    def dx(self):
        return self.attrs['DX']
    def dy(self):
        return self.attrs['DY']
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
        if self.model()=='CFSv2':
            return np.array([self.idate() +  timedelta(days=1) + timedelta(hours=int(t*self.dt())) \
                             for t in range(self.ntimes())])
        else:
            return np.array([self.idate() + timedelta(hours=int(t*self.dt())) for t in range(self.ntimes())])
    def leadtimes(self):
        return [ut.timedelta_hours(self.idate(), d) for d in self.vdates()]
        
#==== Get the lat/lon grid and area weights ==================================
    def latlons(self, staggering=None):
        """ Returns 1D lat and lon grids """
        if staggering in ['U', 'u']:
            return self['latitude_ustag'].values, self['longitude_ustag'].values
        elif staggering in ['V', 'v']:
            return self['latitude_vstag'].values, self['longitude_vstag'].values
        return self['latitude'].values, self['longitude'].values
    
    def area_weights(self, asdataarray=False):
        if asdataarray:
            return np.cos(np.radians(self['latitude']))
        else:
            return np.cos(np.radians(self['latitude'].values))
        
    def get_stag(self, field):
        """ Determine the staggering of the given field """
        if 'longitude_stag' in self[field].dims:
            return 'U'
        elif 'latitude_stag' in self[field].dims:
            return 'V'
        return None
        
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
        
#==== Create a Basemap projection identical to the desired model domain ======
    def get_map_projection(self, ax=None):
        """
        Gets a map projection identical to the Lambert Conformal projection
        specified in this object's attributes (i.e., the WRF domain)
        
        Requires:
        ax --> axis object to link this projection to
        
        Returns:
        m ---> Basemap projection
        """
        from mpl_toolkits.basemap import Basemap
        m = Basemap(width=(self.nx()-2)*self.dx(), height=(self.ny()-2)*self.dy(),
                    resolution='l',area_thresh=1000.,projection='lcc', ax=ax,
                    lat_1=self.attrs['TRUELAT1'], lat_2=self.attrs['TRUELAT2'],
                    lat_0=self.attrs['CEN_LAT'], lon_0=self.attrs['CEN_LON'])
        return m
        
#==== Project coordinates for plotting on a map ==============================
    def project_coordinates(self, m):
        """
        Projects the lat-lons onto a map projection
        """
        la, lo = self.latlons()
        return m(lo, la)

#==== Retrieve or calculate a field from this dataset ========================
    def get_field(self, field, lev=None):
        
        # Use Luke's WRF functions to compute new variables
        import uw_wrf.new_wrf_plotting_functions.wrf_metfunctions as wrf
        from itertools import product as iprod

        # ANY RAW WRF FIELD
        if field in self.variables.keys():
            return self[field].values

        # MEAN SEA LEVEL PRESSURE
        elif field=='mslp':
            PB = self['pressure_base'].values; P = self['pressure_pert'].values
            PHB = self['height_base'].values; PH = self['height_pert'].values
            T = self['theta_pert'].values; QVAPOR = self['qv'].values
            T0 = self.attrs['t0']
            if len(P.shape) == 3:
                slp = wrf.slp(PB, P, PHB, PH, T, QVAPOR, TBASE=T0)
            elif len(P.shape) == 4:
                slp = np.zeros(self['psfc'].shape) # should be 3-d
                for m in range(P.shape[0]):
                    slp[m,:,:] = wrf.slp(PB, P[m,:,:,:], PHB, PH[m,:,:,:], T[m,:,:,:], 
                                         QVAPOR[m,:,:,:], TBASE=T0)[0]
            elif len(P.shape) == 5:
                slp = np.zeros(self['psfc'].shape) # should be 4-d
                for n,m in list(iprod(range(P.shape[0]), range(P.shape[1]))):
                    slp[n,m,:,:] = wrf.slp(PB, P[n,m,:,:,:], PHB, PH[n,m,:,:,:], T[n,m,:,:,:], 
                                           QVAPOR[n,m,:,:,:], TBASE=T0)[0]
            else: raise ValueError('Bad shape of PB:',PB.shape)
            return slp
        
        # HOURLY PRECIPITATION RATE (assumes hourly output)
        elif field=='prate':
            RAINC = self['rainc'].values; RAINNC = self['rainnc'].values
            if len(RAINNC.shape) == 3:
                prate = wrf.calculate_preciprate(RAINNC, RAINC=RAINC, t_axis=0)
            elif len(RAINNC.shape) == 4:
                prate = wrf.calculate_preciprate(RAINNC, RAINC=RAINC, t_axis=1)
            else: raise ValueError('Bad shape of RAINNC:',RAINNC.shape)
            return prate
        
        # TOTAL ACCUMULATED PRECIPITATION
        elif field=='precip':
            RAINC = self['rainc'].values; RAINNC = self['rainnc'].values
            return RAINC + RAINNC
        
        # HOURLY MODEL SNOWFALL RATE (assumes hourly output)
        elif field=='srate':
            SNOWNC = self['snownc'].values
            if len(SNOWNC.shape) == 3:
                srate = wrf.calculate_preciprate(SNOWNC, t_axis=0)
            elif len(SNOWNC.shape) == 4:
                srate = wrf.calculate_preciprate(SNOWNC, t_axis=1)
            else: raise ValueError('Bad shape of SNOWNC:',SNOWNC.shape)
            return srate * 10. # assumes 10:1 ratio for snowfall:SWE
        
        # TOTAL ACCUMULATED SNOWFALL
        elif field=='snow':
            return self['snownc'].values * 10. # assumes 10:1 ratio for snowfall:SWE
        
        # 10-METER WIND SPEEDS
        elif field=='wnd10m':
            U10 = self['u10'].values; V10 = self['v10'].values
            return np.sqrt(U10**2 + V10**2)

        else: raise ValueError('Unknown field: "{}"'.format(field))
        
#==== Get the timeseries of a given field at the desired lat/lon =============
    def get_timeseries(self, field, loc, method='nearest', verbose=False):
        """
        Uses NN or linear interpolation to get timeseries of the desired field at a
        specified lat/lon location
        """
        from scipy.interpolate import griddata
        
        lat, lon = loc
        # We're assuming that the desired field is a surface/unstaggered field
        # e.g., t2m, wnd10m, precip, etc.  (u10 and v10 ARE unstaggered!)
        lats, lons = self.latlons()

        # Find the index of the nearest point to the desired lat/lon
        diffs = np.abs(lats-lat) + np.abs(lons-lon)
        yn, xn = np.unravel_index(diffs.argmin(), diffs.shape)
        if method=='nearest':
            if verbose: print('Fetching data at {:.02f}N {:.02f}E'.format(lats[yn,xn], lons[yn,xn]))
            # Return the data at that point
            return self.isel(latitude=[yn], longitude=[xn]).get_field(field).squeeze()
        
        if verbose: print('performing linear interpolation...')
        # ONLY load the points surrounding the desired point
        xi = xn - 4
        xf = xn + 5
        yi = yn - 4
        yf = yn + 5
        data = self.isel(longitude=range(xi,xf), latitude=range(yi,yf)).get_field(field)
        
        sublats = lats[yi:yf, xi:xf]
        sublons = lons[yi:yf, xi:xf]
        time = np.array(self.leadtimes())
        # Make a 3D time-lat-lon meshgrid
        t = np.broadcast_to(time[:,None,None], (len(time), sublats.shape[0], sublats.shape[1]))
        la = np.broadcast_to(sublats[None,:,:], (len(time), sublats.shape[0], sublats.shape[1]))
        lo = np.broadcast_to(sublons[None,:,:], (len(time), sublons.shape[0], sublons.shape[1]))
        
        # If this is a 4D dataset, do the interpolation for each member separately
        if 'ens' in self.dims:
            ens = np.arange(self.nmems())
            d_interp = np.zeros(np.shape(data)[:2])
            for mem in np.arange(self.dims['ens']):
                if verbose: print('  interpolating member-{}'.format(int(mem)+1))
                if self.dims['ens']==np.shape(d_interp)[0]:
                    d_interp[mem, :] = griddata((t.flatten(), la.flatten(), lo.flatten()), 
                                                data[mem, :, :, :].flatten(), (list(time), [lat], [lon]))
                else:
                    d_interp[:, mem] = griddata((t.flatten(), la.flatten(), lo.flatten()), 
                                                    data[:, mem, :, :].flatten(), (list(time), [lat], [lon]))

        # If this is just a single member (or an ensemble mean) do the full interpolation:
        else:
            d_interp = griddata((t.flatten(), la.flatten(), lo.flatten()), data.flatten(), 
                                (list(time), [lat], [lon]))
        # We want the 'ens' dimension to be the first dimension
        if self.dims['ens']==np.shape(d_interp)[1]:
            d_interp = np.swap_axes(d_interp, 0, 1)
        return d_interp
    
#==== Get the timeseries of a given field at the desired lat/lon =============
    def get_multiple_timeseries(self, field, locs, verbose=False):
        """
        Uses NN or linear interpolation to get timeseries of the desired field at a
        specified lat/lon location
        """
        from scipy.interpolate import griddata
        
        # Separete the desired points into latitudes and longitudes
        latlocs = [loc[0] for loc in locs]
        lonlocs = [loc[1] for loc in locs]
        
        # We're assuming that the desired field is a surface/unstaggered field
        # e.g., t2m, wnd10m, precip, etc.  (u10 and v10 ARE unstaggered!)
        lats, lons = self.latlons()

        
        # Load in the dataset for the desired field
        print('Loading full "{}" field into memory...'.format(field))
        data = self.get_field(field) # shape = (t, lat, lon) OR (ens, t, lat, lon)
        
        time = np.array(self.leadtimes())
        # Make a 3D time-lat-lon meshgrid
        t = np.broadcast_to(time[:,None,None], (len(time), lats.shape[0], lats.shape[1]))
        la = np.broadcast_to(lats[None,:,:], (len(time), lats.shape[0], lats.shape[1]))
        lo = np.broadcast_to(lons[None,:,:], (len(time), lons.shape[0], lons.shape[1]))
        
        # If this is a 4D dataset, do the interpolation for each member separately
        if verbose: print('performing linear interpolation...')
        if 'ens' in self.dims:
            ens = np.arange(self.nmems())
            d_interp = np.zeros(np.shape(data)[:2])
            for mem in np.arange(self.dims['ens']):
                if verbose: print('  interpolating member-{}'.format(int(mem)+1))
                if self.dims['ens']==np.shape(d_interp)[0]:
                    d_interp[mem, :] = griddata((t.flatten(), la.flatten(), lo.flatten()), 
                                                data[mem, :, :, :].flatten(), (list(time), [latlocs], [lonlocs]))
                else:
                    d_interp[:, mem] = griddata((t.flatten(), la.flatten(), lo.flatten()), 
                                                    data[:, mem, :, :].flatten(), (list(time), [latlocs], [lonlocs]))

        # If this is just a single member (or an ensemble mean) do the full interpolation:
        else:
            d_interp = griddata((t.flatten(), la.flatten(), lo.flatten()), data.flatten(), 
                                (list(time), [latlocs], [lonlocs]))
        # We want the 'ens' dimension to be the first dimension
        if self.dims['ens']==np.shape(d_interp)[1]:
            d_interp = np.swap_axes(d_interp, 0, 1)
        return d_interp
    
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
            ti = ut.nearest_ind(self.vdates(), dt_i)
            tf = ut.nearest_ind(self.vdates(), dt_f) + 1
            if field is None:  return self.isel(time=range(ti,tf)).mean(dim='time', keep_attrs=True)
            else:              return self.isel(time=range(ti,tf))[field].mean(dim='time', keep_attrs=True)
        
#==== Average the data to a coarser timescale (e.g., daily, weekly) ===========
    def temporal_average(self, timescale):
        """
        Computes and returns a new GlobalEnsemble that has been averaged at a coareser
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
        
#==== Function to save the xarray Dataset to a netcdf file ====================
    def save_to_disk(self, filename=None):
        # Dump this object to disk
        if filename is None:
            filename = '{}/{}_ensemble_{:%Y%m%d%H}.nc'.format(self.model(), self.workdir(), self.idate())
        self.attrs['idate'] = date2num(self.idate(), 'hours since 1800-01-01')
        self.to_netcdf(filename)
        self.attrs['idate'] = num2date(self.idate(), 'hours since 1800-01-01')
        
        
###############################################################################################
# extra utilities
###############################################################################################