#!/usr/bin/env python
"""
Tools for loading WRF output files into xarray format
"""

import xarray
import numpy as np
from .rename_vars import vardict

constvars = ['XLAT', 'XLONG', 'XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V', 'F', 'HGT', 'PB', 'PHB', 'SST']

def load_ensemble_xr(wrfensdir, memberdirs, dom=1, chunks=None, returnmissing=False, verbose=False):
    """
    Loads wrfout files from subdirectories within [wrfensdir] into an xarray Dataset
    
    Requires:
    wrfensdir --> directory containing the ensemble member subdirectories (string)
    memberdirs -> list of strings; each is the name of a subdirectory containing wrfouts 
                  for one ensemble member
    dom --------> identifier for the WRF domain (integer)
    
    Returns:
    ens_dset ---> an xarray Dataset object containing all members, times, locations, and desired variables
    """
    import os
    
    # Use the load_member_xr function to create a list of xarray ensemble members
    mem_xrs = []
    missing_mems = []
    for mem in memberdirs:
        try:
            mem_xrs.append(load_member_xr('{}/{}'.format(wrfensdir, mem), dom=dom, verbose=verbose))
        except OSError as err:   # this means that this member does not exist!
            if verbose: print('Member "{}" is missing!'.format(mem))
            missing_mems.append(mem)
    
    # We want to concatentate the variables that are not constant
    concat_vars = [vrbl for vrbl in mem_xrs[0].variables.keys() if (vrbl not in constvars+['XTIME'])]
    
    # Use xarray.concat to concatenate our list of xarray Datasets
    if verbose: print('Concatenating member xarrays...')
    ens_dset = xarray.concat(mem_xrs, dim='ens', data_vars=concat_vars).rename(vardict)
    
    
    ## Re-chunk the dataset
    #if verbose: print('Re-chunking...')
    #if chunks is None: chunks = {dim : ens_dset.dims[dim] for dim in ens_dset.dims}
    #ens_dset = ens_dset.chunk(chunks)
    if returnmissing: return ens_dset, missing_mems
    else: return ens_dset
    
    
def load_member_xr(wrfoutdir, dom=1, verbose=False):
    """
    Loads wrfout files from a directory (one ensemble member) into an xarray Dataset
    
    Requires:
    wrfoutdir -> directory containing the wrfout* netcdf files (string)
    dom -------> identifier for the WRF domain (integer)
    
    Returns:
    dset ------> an xarray Dataset object containing all times, locations, and desired variables
    """
    # Some attributes specifying constant variables (CAN BE CHANGED)
    attribs = {'ptop' : 10000., 'p0' : 100000., 't0' : 290.}
    
    # Variables to skip over and not store within the xarray object (CAN BE CHANGED)
    dropvars = ['Times', 'LU_INDEX', 'ZNU', 'ZNW', 'ZS', 'DZS', 'VAR_SSO', 'LAP_HGT', 
                'MU', 'MUB', 'NEST_POS', 'FNM', 'FNP', 'RDNW', 'RDN', 'DNW', 'DN', 'CFN', 'CFN1',
                'THIS_IS_AN_IDEAL_RUN', 'RDX', 'RDY', 'RESM', 'ZETATOP', 'CF1', 'CF2', 'CF3', 'ITIMESTEP',
                'SNOALB', 'TSLB', 'SMOIS', 'SH2O', 'SMCREL', 'SEAICE', 'SMSTAV', 'SMSTOT', 'SFROFF', 
                'UDROFF', 'GRDFLX', 'VAR', 'MAPFAC_M', 'MAPFAC_U', 'MAPFAC_V', 'MAPFAC_MX', 'MAPFAC_UX',
                'MAPFAC_VX', 'MAPFAC_MY', 'MAPFAC_UY', 'MAPFAC_VY', 'MF_VX_INV', 'SINALPHA', 'COSALPHA',
                'TLP', 'TISO', 'MAX_MSTFX', 'MAX_MSTFY', 'ALBEDO', 'EMISS', 'NOAHRES', 'QSNOWXY', 'RUNSF',
                'RUNSB', 'TMN', 'ZNT', 'UST', 'SR', 'SAVE_TOPO_FROM_REAL', 'ISEEDARR_RAND_PERTURB',
                'ISEEDARR_SPPT', 'ISEEDARR_SKEBS', 'XLAND','LAKEMASK', 'SST_INPUT', 'P_TOP', 'T00', 'P00',
                'QNICE', 'QNRAIN', 'Z0', 'E', 'RAINCV', 'RAINNCV', 'CLDFRA', 'SNOWC', 'PREC_ACC_C',
                'PREC_ACC_NC', 'SNOW_ACC_NC', 'ACSNOM']
    
    # Use wildcard identifier to select all the desired wrf output files
    wrfouts = '{}/wrfout_d{:01d}.*0000'.format(wrfoutdir, dom)
    
    # Open/concatenate/store all the wrfouts for the desired domain
    if verbose: print('Loading', wrfouts)
    dset = xarray.open_mfdataset(wrfouts, drop_variables=dropvars, concat_dim='Time', 
                                 autoclose=True, decode_cf=False)
    
    # For variables that are constant in time, just store the values at one time
    for vrbl in constvars:
        #if 'Time' not in dset[vrbl].dims: continue
        dset[vrbl] = dset[vrbl].isel(Time=0)
    
    # Assign the attributes above and return the Dataset
    dset = dset.assign_attrs(**attribs)
    return dset

def load_ensemble_allhrs(idate, wrfensdir, memberdirs, dom=1, returnmissing=False, 
                         chunks=None, verbose=False):
    """
    Loads *combined* wrfout files (e.g., "wrfout*allhrs.nc"; one per each ensemble member) into
    an ensemble xarray Dataset.
    
    Requires:
    idate ------> the ensemble initialization time (datetime object)
    wrfensdir --> directory containing the ensemble member subdirectories (string)
    memberdirs -> list of strings; each is the name of a subdirectory containing wrfouts 
                  for one ensemble member
    dom -------> identifier for the WRF domain (integer)
    
    Returns:
    dset ------> an xarray Dataset object containing all times, locations, and desired variables
    """
    import os
    
    # Some attributes specifying constant variables (CAN BE CHANGED)
    attribs = {'ptop' : 10000., 'p0' : 100000., 't0' : 290.}
    
    # Variables to skip over and not store within the xarray object (CAN BE CHANGED)
    dropvars = ['Times', 'LU_INDEX', 'ZNU', 'ZNW', 'ZS', 'DZS', 'VAR_SSO', 'LAP_HGT', 
                'MU', 'MUB', 'NEST_POS', 'FNM', 'FNP', 'RDNW', 'RDN', 'DNW', 'DN', 'CFN', 'CFN1',
                'THIS_IS_AN_IDEAL_RUN', 'RDX', 'RDY', 'RESM', 'ZETATOP', 'CF1', 'CF2', 'CF3', 'ITIMESTEP',
                'SNOALB', 'TSLB', 'SMOIS', 'SH2O', 'SMCREL', 'SEAICE', 'SMSTAV', 'SMSTOT', 'SFROFF', 
                'UDROFF', 'GRDFLX', 'VAR', 'MAPFAC_M', 'MAPFAC_U', 'MAPFAC_V', 'MAPFAC_MX', 'MAPFAC_UX',
                'MAPFAC_VX', 'MAPFAC_MY', 'MAPFAC_UY', 'MAPFAC_VY', 'MF_VX_INV', 'SINALPHA', 'COSALPHA',
                'TLP', 'TISO', 'MAX_MSTFX', 'MAX_MSTFY', 'ALBEDO', 'EMISS', 'NOAHRES', 'QSNOWXY', 'RUNSF',
                'RUNSB', 'TMN', 'ZNT', 'UST', 'SR', 'SAVE_TOPO_FROM_REAL', 'ISEEDARR_RAND_PERTURB',
                'ISEEDARR_SPPT', 'ISEEDARR_SKEBS', 'XLAND','LAKEMASK', 'SST_INPUT', 'P_TOP', 'T00', 'P00',
                'QNICE', 'QNRAIN', 'Z0', 'E', 'RAINCV', 'RAINNCV', 'CLDFRA', 'SNOWC', 'PREC_ACC_C',
                'PREC_ACC_NC', 'SNOW_ACC_NC', 'ACSNOM']
    
    # Point to the "wrfout*allhrs.nc" file in each ensemble directory
    all_wrfouts = ['{}/{}/wrfout_d{:01d}.{:%Y%m%d%H}.allhrs.nc'.format(wrfensdir, mem, dom, idate) for mem in memberdirs]
    wrfouts = []
    missing_mems = []
    for w, wrfout in enumerate(all_wrfouts):
        if os.path.isfile(wrfout): wrfouts.append(wrfout)
        else:                      missing_mems.append(memberdirs[w])
    # If less than half of the members have an "allhrs.nc" file, raise an error
    if len(wrfouts) < len(all_wrfouts)/2:
        raise IOError('Only {} of the {} members have an "allhrs.nc" file!'.format(len(wrfouts), len(all_wrfouts)))
    
    # Open/concatenate/store all the wrfouts for the desired domain
    if verbose: print('Loading {} "allhrs.nc" files...'.format(len(wrfouts)))
    if chunks is None: chunks = {'Time': 24, 'ens' : len(wrfouts)}
    dset = xarray.open_mfdataset(wrfouts, drop_variables=dropvars, concat_dim='ens', 
                                 autoclose=True, decode_cf=False)
    
    # For variables that are constant for all times/members, just store the values at one time/member
    for vrbl in constvars:
        if 'Time' in dset[vrbl].dims: dset[vrbl] = dset[vrbl].isel(Time=0)
        if 'ens' in dset[vrbl].dims:  dset[vrbl] = dset[vrbl].isel(ens=0)
    
    # Assign the attributes above, rename the variables, and return the Dataset
    dset = dset.assign_attrs(**attribs).rename(vardict)
    if returnmissing: return dset, missing_mems
    else: return dset

def load_plumes(wrfrundir, mems, dom=1, returnmissing=False, verbose=False):
    """
    Loads pre-processed netcdfs containing WRF meteorogram data for a number of stations in 
    the Pacific Northwest. Each netcdf contains meteorogram data for one ensemble member
    and all stations/times. Data is loaded/returned as an xarray Dataset.
    
    Requires:
    wrfrundir -----> directory containing the member subdirectories (string)
    mems ----------> list of names of the member subdirectories (strings)
    dom -----------> identifier for the WRF domain (integer)
    returnmissing -> if True, a list of the missing members is also returned
    
    Returns:
    dset ---------> an xarray Dataset object containing all members, times, sites, and variables
    missing_mems -> (Optional) a list of the missing ensemble members
    """
    import os
    
    # Variables to skip over and not store within the xarray object (CAN BE CHANGED)
    dropvars = ['U', 'V', 'QVAPOR', 'P', 'PB', 'T', 'U10', 'V10', 'dir', 'spd', 't', 'td', 'rh', 'ptotal',
                'Q2', 'RAINC', 'RAINNC', 'SNOWNC', 'ACSNOW', 'acsnow']
    
    # Get the pre-processed "plumes" file for each ensemble member
    ncfiles = ['{}/{}/plumes.{}.d{:01d}.nc'.format(wrfrundir, mem, mem, dom) for mem in mems]
    # Skip over any members whose plume files do not exist
    availfiles = []
    missing_mems = []
    for n, ncfile in enumerate(ncfiles):
        if os.path.isfile(ncfile): 
            availfiles.append(ncfile)
        else: 
            missing_mems.append(mems[n])
            if verbose: print(ncfile, 'not available!')
    
    # Open/concatenate/store all the ensemble plumes 
    if verbose: print('Loading {} plume netcdfs...'.format(len(availfiles)))
    dset = xarray.open_mfdataset(availfiles, drop_variables=dropvars, concat_dim='ens', 
                                 autoclose=True, decode_cf=True)
    
    # For variables that are the same for all members, just store the values for one member
    for vrbl in ['times', 'sites', 'lat', 'lon', 'HGT']:
        dset[vrbl] = dset[vrbl].isel(ens=0)
    dset['HGT'] = dset['HGT'].isel(time=0)
        
    # Rename the variables for readability
    rename = {'HGT':'elev', 'PSFC':'psfc', 'T2':'t2m', 'dir_10':'wdir10m', 'rain':'precip',
              'spd_10':'wspd10m', 'td_2':'td2m', 'rh_2':'rh2m', 'slp':'mslp', 'time':'Time'}
    dset = dset.rename(rename)
    if returnmissing: return dset, missing_mems
    else:             return dset


#################################################################################################################
# Tools for calculating atmospheric variables on WRF grids
#################################################################################################################
