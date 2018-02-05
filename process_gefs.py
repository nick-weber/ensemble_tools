#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Contains functions for downloading, processing, and making verification figures 
with operational CFSv2 forecasts
"""

import os
from datetime import datetime, timedelta
import numpy as np
from time import time
import multiprocessing
from functools import partial
from contextlib import contextmanager


# Global variable: a list of GEFS ensemble member names (strings)
members = ['gec00'] + ['gep{:02d}'.format(x+1) for x in range(20)]


@contextmanager
def poolcontext(*args, **kwargs):
    """Allows the passing of additional arguments to a function in a MP pool"""
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
    
    
def make_wget_text_file(fileprefix, runhr, fcsthr):
    """
    Makes text file for fast wget command (one forecast lead time).
    
    Requires:
    fileprefix -> string describing the gefs file type (e.g., 'pgrb2ap5')
    runhr ------> 2-char string describing the forecast init hour (e.g., '00' or '12')
    fcsthr -----> 3-char string describing the forecast lead hour (e.g., '000' or '024')
    
    Returns:
    The name of the wget text file
    """
    # Open the text file for writing
    wgetfile = 'wget_list_f'+fcsthr+'.txt'
    
    # Get a list of filenames and see if they already exist locally
    filenames = ['{}.t{}z.{}.0p50.f{}'.format(mem, runhr, fileprefix[0:-2], fcsthr) for mem in members]
    # If all these files exist, don't write the text file and return None
    if np.array([os.path.isfile('./'+fn) for fn in filenames]).all():
        return None
    
    with open(wgetfile,'w') as ens_file:
        # Download the members!
        for filename in filenames:
            ens_file.write(filename+'\n')
    return wgetfile



def download_gefs_mp(t, idate, getdir, fileprefix='pgrb2ap5', verbose=False):
    '''
    Function capable of downloading one timestep of GEFS data. Can run in parallel.
    
    Requires:
    t ----------> forecast lead hour (int)
    idate ------> forecast initialization date (datetime object)
    getdir -----> directory to download the forecasts to (string)
    fileprefix -> string describing the gefs file type (e.g., 'pgrb2ap5')
    '''
    # Set some constants
    hh = idate.strftime('%H')
    fxxx = '{0:0=3d}'.format(t)
    ftp_masterdir = 'ftp://ftp.ncep.noaa.gov/pub/data/nccf/com/gens/prod/'
    
    # Get the full ftp path and the full path to where we will put the data
    ftpdir = '{}gefs.{:%Y%m%d}/{}/{}/'.format(ftp_masterdir, idate, hh, fileprefix)

    if os.getcwd() != getdir:
        os.chdir(getdir)  # move to where we will download the data
    
    # Write a text file with all the filenames for wget
    wf = make_wget_text_file(fileprefix, hh, fxxx)
    
    # If these files all exist already, do not execute wget
    if wf is None:
        if verbose: print('SKIPPING lead {}: grib files already downloaded!'.format(fxxx))
        return
    
    # Otherwise, download the data!
    if verbose: print('Downloading gribs for time:', fxxx)
    os.system('wget -q -nc --user=anonymous  --password=jzagrod@uw.edu --base={} -i ./{}'.format(ftpdir, wf))
    os.system('rm -rf ' + wf) # clean up
    return

              
              
def get_gefs(idate, gefsdir, fileprefix='pgrb2ap5', nprocs=16, fhours=range(0,337,6), verbose=False):
    """
    Downloads operational GEFS ensemble forecasts (initialized on idate)
    Currently set to download pgrb2ap5 (0.5 degree resolution w/ most common parameters)

    Requires:
    idate ------> the day on which all ensemble members were initialized (datetime object)
    gefsdir ----> directory to store the downloaded grib files
    fileprefix--> list of GEFS products to download (supports pgrb2ap5,pgrb2bp5 for sure)
    nprocs -----> number of processors to use for downloading
    fhours -----> list (ints) of desired forecast hours; default: first two weeks (6-hrly)
    """
    if verbose: print('\n==== Downloading operational GEFS forecasts ====')
    if verbose: print('============= using {:02d} processors =============='.format(nprocs))
    if verbose: print('======= Initialization: {:%Y-%m-%d %H:00} ======='.format(idate))
    start = time()
     
    # Create the directory to store the GEFS data
    if not os.path.isdir(gefsdir): os.makedirs(gefsdir)
        
    # The control + 20 members are downloaded for each hour
    with poolcontext(processes=nprocs) as pool:   # use a multiprocessing pool!
        pool.map(partial(download_gefs_mp, idate=idate, getdir=gefsdir, verbose=verbose), fhours)
        
    if verbose: print('Download time: {:.2f} min'.format((time()-start)/60.))
    return



def grbs2nc_mp(filetuple, matchtag, tablefile, verbose=False):
    """
    Uses wgrib2 to combine grib files into one netcdf file.
    Implementable with multiprocessing.
    """
    # Unpack the tuple
    grbfiles = filetuple[0] # list of gribs to be converted
    ncfile = filetuple[1]   # name of the netcdf to convert to
    mem = filetuple[2]
    # Make an inventory file so everything doesn't go to stdout
    inv = 'inv_{}.log'.format(mem)
    
    # No need to convert anything if the netcdf file already exists!
    if os.path.isfile(ncfile):
        if verbose: print('SKIPPING {} -- file already exists!'.format(ncfile))
        return
    print('Combining {} gribs into file: {}'.format(len(grbfiles), ncfile))
    
    #Loop through files and convert to one netcdf
    for g, gfile in enumerate(grbfiles):
        if g==0:   # the netcdf file has not been created yet
            cmd = 'wgrib2 {} -inv {} {} -nc_table {} -netcdf {}'.format(gfile, inv, matchtag, tablefile, ncfile)
        else:      # since the netcdf exists now, add the '-append' tag
            cmd = 'wgrib2 {} -inv {} {} -append -nc_table {} -netcdf {}'.format(gfile, inv, matchtag, tablefile, ncfile)
        
        os.system(cmd)
        os.system('rm -f ' + gfile)
    return
    
    
    
def convert_gefs_grb2nc(idate, gefsdir, tablefile='cfs.table', nprocs=16, verbose=False):
    """
    Converts downloaded GEFS gribs to netcdf, retaining only the dates, lead times, 
    and variables designated in the nctable file and the -match keywords.
    
    Converts the grib files into one netcdf file (all lead times) per ensemble member.
    
    Currently won't work if <fileprefix> was not 'pgrb2ap5' for the downloaded gribs.
    
    *Utilizes the wgrib2 utility
    
    Requires:
    idate ---------> the date/time at which all ensemble members were initialized (datetime object)
    gefsdir -------> grib directory (also where the netcdfs will go)
    tablefile -----> the nc_table file used in the wgrib2 conversion (not using yet but will later)
    vrbls ---------> list of desired variable for conversion (if None, all variables are read)
    nprocs --------> number of processors to use for file conversion
    """
    from subprocess import check_output, Popen
    
    if verbose: print('\n== Combining GEFS gribs into 1-member netcdfs ==')
    if verbose: print('============= using {:02d} processors =============='.format(nprocs))
    if verbose: print('======= Initialization: {:%Y-%m-%d %H:00} ======='.format(idate))
    start = time()

    # Point to the nc_table text file (full path)
    if tablefile[0] != '/': tablefile = '/home/disk/p/njweber2/pypackages/ensemble_tools/{}'.format(tablefile)
    assert os.path.isfile(tablefile)
    if os.getcwd() != gefsdir: os.chdir(gefsdir)
    
    # Want one netcdf per ensemble member, so let's get a list of all the grib files for each member
    # (this will yield a list of tuples that we can feed to the multiprocessing function)
    memfiles = []
    for mem in members:    # <members> is defined at the top of this module
        lscom = 'ls -1a {}/{}*'.format(gefsdir, mem)
        grbfiles = check_output([lscom], shell=True).split()
        grbfiles = [g.decode("utf-8") for g in grbfiles]
        # Name the corresponding netcdf file that we will make
        ncfile = os.path.join(gefsdir, '{}_{:%Y%m%d%H}.nc'.format(mem, idate))
        # Add tuple to the list: contains a list of grib files and the name of the (to-be) netcdf file
        memfiles.append((grbfiles, ncfile, mem))
        

    # Get the -match keyword from the .table file
    with open(tablefile, "r") as tfile:
        for l, line in enumerate(tfile):
            if l==14:
                matchtag = line[1:].rstrip()
                break
                
    # Loop through the file lists and convert to netcdfs (one per member)
    with poolcontext(processes=nprocs) as pool:   # use a multiprocessing pool!
        pool.map(partial(grbs2nc_mp, matchtag=matchtag, 
                         tablefile=tablefile, verbose=verbose), memfiles)
    if verbose: print('Conversion time: {:.2f} min'.format((time()-start)/60.))
    return



def check_ncfiles(gefsdir, idate):
    """
    Returns True is the netcdfs for this initialization date have already been created.
    """
    ncfiles = [os.path.join(gefsdir, '{}_{:%Y%m%d%H}.nc'.format(mem, idate)) for mem in members]
    iscreated = [os.path.isfile(ncfile) for ncfile in ncfiles]
    return np.array(iscreated).all()



def main(cycle=0, return_ens_args=False, verbose=False):
    """
    Function for the operational downloading/processing of today's GEFS run.
    
    Requires:
    cycle -----------> UTC hour (int) of cycle initialization (i.e., 0 or 12)
    return_ens_args -> if True, will return <args> and <kwargs> to be fed into 
                       ensemble.GlobalEnsemble.from_NCEP_netcdfs() to create a
                       GlobalEnsemble instance from this GEFS run
    """
    t0 = time()
    # Get today's date
    today = datetime.today().replace(hour=cycle, minute=0, second=0, microsecond=0)
    
    # Point to where we want to download the data
    gefsdir = '/home/disk/anvil2/ensembles/gefs/forecasts/{:%Y%m%d%H}'.format(today)
    
    if check_ncfiles(gefsdir, today):
        if verbose: print('netcdfs for {:%Y%m%d%H} exist!'.format(today))
    
    else:
        # Download the ensemble gribs
        get_gefs(today, gefsdir, verbose=verbose)

        # Convert/combine the gribs to netcdfs (one per ensemble member)
        convert_gefs_grb2nc(today, gefsdir, verbose=verbose)

        if verbose: print('\n ==== TOTAL ELAPSED TIME: {:.2f} min ====\n'.format((time()-t0)/60.))
    
    if return_ens_args:
        return (today, gefsdir), {'filetag':'g*.nc', 'model':'GEFS'}
    else: return



if __name__ == '__main__':
    main()

