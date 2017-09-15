# -*- coding: utf-8 -*-
"""
Contains functions for downloading, processing, and making verification figures 
with operational CFSv2 forecasts
"""

import os
from datetime import datetime, timedelta
import numpy as np
import time
import glob

cfs_vars = ['prate', 'prmsl', 'tmp2m', 'wnd10m', 'wnd200', 'wnd850', 'z500']

def download_cfsv2_16mem(idate, cfsdir, verbose=False):
    """
    Downloads operational CFSv2 ensemble forecasts (initalized on idate)
    
    Requires:
    idate ---> the day on which all ensemble members were initialized (datetime object)
    cfsdir --> directory to store the downloaded grib files
    """
    from urllib.request import urlretrieve
    import time

    if not os.path.isdir(cfsdir):
        os.system('mkdir {}'.format(cfsdir))
    
    # Point to the proper location on the NOMADS server for downloading CFSv2 forecasts
    if verbose: print('\n==== Downloading operational CFSv2 forecasts ====')
    nomads = 'http://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs/cfs.{:%Y%m%d}'.format(idate)
    start = time.time()
    
    # loop through all init times (00,06,12,18) and ensemble members (1,2,3,4)
    for hh in ['00','06','12','18']:
        for mem in ['01','02','03','04']:
            for var in cfs_vars:
                # Find the url for the forecast of each desired variable
                url = '{}/{}/time_grib_{}/{}.{}.{:%Y%m%d}{}.daily.grb2'.format(nomads, hh, mem, var, mem, idate, hh)
                localfile = '{}/fcst.{}.cfs.{:%Y%m%d}{}_{}.grb2'.format(cfsdir, var, idate, hh, mem)
                if os.path.isfile(localfile):
                    if verbose: print('File already exists:\n{}'.format(localfile))
                    continue
                # Download the forecast
                if verbose: print('Downloading {}...'.format(url))
                urlretrieve(url, localfile)
    end = time.time()
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

def download_gefs(idate,gefsdir,verbose=False):
    """
    Downloads operational GEFS ensemble forecasts (initialized on idate)
    Currently set to download pgrb2ap5 (0.5 degree resolution w/ most common parameters)

    Requires:
    idate ---> the day on which all ensemble members were initialized (datetime object)
    gefsdir --> directory to store the downloaded grib files
    """
    gefs_gribs = ['control','members','ens_mean','ens_spread']
    
    if not os.path.isdir(cfsdir):
        os.system('mkdir {}'.format(gefsdir))

    # Point to the proper location on the NOMADS server for downloading CFSv2 forecasts
    if verbose: print('\n==== Downloading operational GEFS forecasts ====')
    ftpdir = 'ftp://ftp.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs.{:%Y%m%d}'.format(idate)
    start = time.time()
    tstep = range(0,195,3)+range(198,390,6)
    
    # loop through all init times (00,06,12,18)
    for hh in ['00','06','12','18']:
    #for hh in ['12']:
        #control run
        os.chdir(gefsdir)
        os.system('mkdir -p '+hh)
        os.chdir(hh)
        os.system('mkdir -p control')
        os.chdir('control')
        ftpdir = ftpdir+'/'+hh+'/pgrb2ap5/'
        for t in tstep:
             fxxx = "{0:0=3d}".format(t)
             filename = 'gec00.t'+hh+'z.pgrb2a.0p50.f'+fxxx
             os.system('wget -nc --user=anonymous  --password=jzagrod@uw.edu '+ ftpdir+filename)

        convert_gefs_grb2nc(gefsdir+'/'+hh+'/control',idate,int(hh),fileprefix='gec',filetype='control')

        #ensemble members
        for mem in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20']:
            os.chdir(gefsdir+'/'+hh)
            os.system('mkdir -p '+mem)
            os.chdir(mem)
            for t in tstep:
                fxxx = "{0:0=3d}".format(t)
                filename = 'gep'+mem+'.t'+hh+'z.pgrb2a.0p50.f'+fxxx
                os.system('wget -nc --user=anonymous  --password=jzagrod@uw.edu '+ ftpdir+filename)

            convert_gefs_grb2nc(gefsdir+'/'+hh+'/'+mem,idate,int(hh),fileprefix='gep'+mem,filetype='mem'+mem)

        #ensemble mean
        os.chdir(gefsdir+'/'+hh)
        os.system('mkdir -p ens_mean')
        os.chdir('ens_mean')
        for t in tstep:
             fxxx = "{0:0=3d}".format(t)
             filename = 'geavg.t'+hh+'z.pgrb2a.0p50.f'+fxxx
             os.system('wget -nc --user=anonymous  --password=jzagrod@uw.edu '+ ftpdir+filename)

        convert_gefs_grb2nc(gefsdir+'/'+hh+'/'+mem,idate,int(hh),fileprefix='geavg',filetype='ens_mean')

        #ensemble spread
        os.chdir(gefsdir+'/'+hh)
        os.system('mkdir -p ens_spread')
        os.chdir('ens_spread')
        for t in tstep:
             fxxx = "{0:0=3d}".format(t)
             filename = 'gespr.t'+hh+'z.pgrb2a.0p50.f'+fxxx
             os.system('wget -nc --user=anonymous  --password=jzagrod@uw.edu '+ ftpdir+filename)

        convert_gefs_grb2nc(gefsdir+'/'+hh+'/'+mem,idate,int(hh),fileprefix='gespr',filetype='ens_spread')

    end = time.time()
    if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

def convert_gefs_grb2nc(grbdir,iday,ihr, tablefile='cfs.table', fileprefix='gec', filetype='control',vrbls=None, verbose=False):
    """
    Converts downloaded GEFS gribs (forecasts or analyses) to netcdf, 
    retaining only the dates, lead times, and variables designated 
    in the nctable file and the -match keywords
    
    *Utilizes the wgrib2 utility

    Currently assumes that all timesteps are being saved, variables can be adjusted in the code below.
    I can make a table later so that it is prettier. 
    
    Requires:
    grbdir --------> grib directory (also where the netcdfs will go)
    iday ----------> the day on which all ensemble members were initialized (datetime object)
    ihr------------> the hour of the initialization run (00, 06, 12, or 18)
    tablefile -----> the nc_table file used in the wgrib2 conversion (not using yet but will later)
    fileprefix ----> prefix for the grib files so we know what type they are 
    filetype-------> prefix for the output file
    vrbls ---------> list of desired variable for convection (if None, all variables are read)
    """
    from subprocess import check_output, Popen
    import time
    ncoutfile = grbdir+'test.nc'
    
    # Point to the nc_table (full path)
    if tablefile[0] != '/': tablefile = '/home/disk/meso-home/jzagrod/Models/ensemble_tools/{}'.format(tablefile)
    assert os.path.isfile(tablefile)
    assert iday.hour==0
    
    #Get list of grib files
    grbfiles = sorted(glob.glob(grbdir+'/'+fileprefix+'*'))

    #Change init time to correct hour
    #itime = iday.replace(hour=ihr)

    # Get the -match keyword from the .table file
    '''
    with open(tablefile, "r") as tfile:
        for l, line in enumerate(tfile):
            if l==14:
                matchtag = line[1:].rstrip()
                break
    '''
    #temporary matchtag for testing
    matchtag = ' -match ":(PRMSL|APCP):"'
                
    #Loop through files and convert to netCDF
    for gind in range(0,len(grbfiles)):
        grboutfile = grbdir+'/'+'gefs_'+filetype+'.nc'
        os.system('wgrib2 '+grbfiles[gind]+matchtag+' -append -nc_table '+tablefile+' -netcdf '+grboutfile)
    #the loop has to be run twice because wgrib doesn't like the uneven timesteps the first time
    #but if you run it a second time it is happy
    for gind in range(0,len(grbfiles)):
        grboutfile = grbdir+'/'+'gefs_'+filetype+'.nc'
        os.system('wgrib2 '+grbfiles[gind]+matchtag+' -append -nc_table '+tablefile+' -netcdf '+grboutfile)
        os.system('rm -rf '+grbfiles[gind])
    print('Successfully converted to ncdf '+filetype)
    return


def convert_ensemble_grb2nc(datadir, iday, ndays, tablefile='cfs.table', outfileprefix='cfsv2ens',
                            inittimes=[0,6,12,18], mems=[1,2,3,4], vrbls=None, verbose=False):
    """
    Converts downloaded CFS gribs (forecasts or analyses) to netcdf, 
    retaining only the dates, lead times, and variables designated 
    in the nctable file and the -match keywords
    
    The gribs are all interpolated to a 0.5-degree lat-lon grid
    before conversion to netCDF (to ensure compatibility for combination)
    
    *Utilizes the wgrib2 utility
    
    Requires:
    datadir -------> directory containing the grib files (also where the netcdfs will go)
    iday ----------> the day on which all ensemble members were initialized (datetime object)
    ndays ---------> desired number of days of ensemble forecast data
    tablefile -----> the nc_table file used in the wgrib2 conversion
    outfileprefix -> prefix for the netcdf files
    inittimes -----> list of initialization times (ints) for the ensemble members
    mems ----------> number of ensemble members at each initialization time
    vrbls ---------> list of desired variable for convection (if None, all variables are read)
    """
    from subprocess import check_output, Popen
    import time
    
    # Point to the nc_table (full path)
    if tablefile[0] != '/': tablefile = '/home/disk/user_www/njweber2/nobackup/cfs/{}'.format(tablefile)
    assert os.path.isfile(tablefile)
    assert iday.hour==0
    
    # We want to make one netcdf file for each ensemble member, so loop through inittimes and mems
    for timenum, inittime in enumerate(inittimes):
        mem_idate = datetime(iday.year, iday.month, iday.day, inittime)
        for mem in mems:
            member = mem + 4*timenum  # value from 1 to 16
    
            # Name the output nc file for this member
            ncoutfile = '{}/{}_mem{:02d}.nc'.format(datadir, outfileprefix, member)
            if os.path.isfile(ncoutfile):
                if verbose: print('{} already exists!'.format(ncoutfile))
                continue

            # List all of the grib files for this member
            lscom = 'ls -1a {}/fcst*.cfs.{:%Y%m%d}{:02d}_{:02d}.grb2'.format(datadir, iday, inittime, mem)
            grbfiles = check_output([lscom], shell=True).split()
            grbfiles = [g.decode("utf-8") for g in grbfiles]

            # Only convert the desired variables
            if vrbls is not None:
                grbfiles = [grb for grb in grbfiles if any([vrbl in grb for vrbl in vrbls])]

            # Our first grib file *must* be for a variable that is not temporally averaged (like precip/OLR)
            if any([vrbl in grbfiles[0] for vrbl in ['prate', 'ulwtoa']]):
                # Find the first grb with any other variable
                i = 0
                for g, grbfile in enumerate(grbfiles):
                    if not any([vrbl in grbfile for vrbl in ['prate', 'ulwtoa']]):
                        i = g
                        break
                # Swap the first file with the non-OLR/prate file
                grbfiles[0], grbfiles[i] = grbfiles[i], grbfiles[0]

            # Get the -match keyword from the .table file
            with open(tablefile, "r") as tfile:
                for l, line in enumerate(tfile):
                    if l==14:
                        matchtag = line[1:].rstrip()
                        break
                        
            # Use the -match keyword to select the desired dates
            # We only want dates starting on the day FOLLOWING initialization
            # (e.g., if this member was init. on Jan. 1 @ 18:00, we want only dates starting at Jan 2, 00:00)

            # Create a -match tag for the desired date forecast hours
            startlead = timedelta_hours(mem_idate, iday+timedelta(days=1))
            endlead = timedelta_hours(mem_idate, iday+timedelta(days=1+ndays))
            matchtag2 = '-match ":('
            for h in range(startlead, endlead+1, 6):
                matchtag2 += '{} hour fcst'.format(h)
                if h < endlead: matchtag2 += '|'
            matchtag2 += '):"'
            matchtag = ' '.join(matchtag, matchtag2)      

            # Interpolate each grib file to a 0.5-deg lat-lon grid and then convert/append
            # to one netcdf file
            start = time.time()
            for g, grbfile in enumerate(grbfiles):
                print('processing grib {} of {}...'.format(g+1, len(grbfiles)))
                print(grbfile)
                print('  interpolating...')
                interpcomm = 'wgrib2 {} {} -new_grid_vectors U:V -new_grid latlon 0:720:0.5 -90:361:0.5 temp.grb2'
                interpcomm = interpcomm.format(grbfile, matchtag)
                Popen([interpcomm], shell=True).wait()
                convertcomm = 'wgrib2 temp.grb2 -nc_table {} -netcdf {}'
                convertcomm = convertcomm.format(tablefile, ncoutfile)

                print('  converting to netcdf...')
                if os.path.isfile(ncoutfile):
                    splits = convertcomm.split('-nc_table')
                    convertcomm = splits[0] + '-append -nc_table' + splits[1]
                Popen([convertcomm], shell=True).wait()
                if interp: Popen(['rm -f temp.grb2'], shell=True).wait()
            end = time.time()
            if verbose: print('Elapsed time: {:.2f} min'.format((end-start)/60.))
    return

def timedelta_hours(dt_i, dt_f):
    """ Find the number of hours between two dates """
    return int((dt_f-dt_i).days*24 + (dt_f-dt_i).seconds/3600)

if __name__ == '__main__':
    t1 = time.time()
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    print('\n', today)                                      
    cfsdir = '/home/disk/vader2/njweber2/cfs4website/forecasts/{:%Y%m%d%H}'.format(today)
    #driver(today)
    #download_cfsv2_16mem(today, cfsdir, verbose=True)
    #convert_ensemble_grb2nc(cfsdir, today, 28, verbose=True)

    gefsdir = '/home/disk/anvil2/ensembles/gefs/forecasts/{:%Y%m%d}'.format(today)
    download_gefs(today,gefsdir,verbose=True)

    t2 = time.time()
    print('====== TOTAL TIME: {:.2f} min ======\n'.format((t2-t1)/60.))

    
