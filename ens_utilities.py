#!/usr/bin/env python
"""
Utilities for dealing with and plotting geophysical data
"""
import numpy as np

def checkdir(direc):
    """ Creates the directory [direc] if it does not already exist """
    import os
    if not os.path.isdir(direc):
            os.mkdir(direc)
            
def to_localtime(dt):
    """ Converts a datetime from UTC to West Coast (Pacific) time """
    import pytz
    dt_utc = dt.replace(tzinfo=pytz.utc)
    return dt_utc.astimezone(pytz.timezone('America/Los_Angeles'))

def highlight_localtime(ax, dts, x, addtext=True):
    import calendar
    
    # Convert the datetimes to local (Pacific) time
    dts_local = [to_localtime(date) for date in dts]
    
    # Find all of the indices for midnight localtime
    hrs_local = np.array([date.hour for date in dts_local])
    inds = np.where(hrs_local == 0)[0]
    
    # On the axis, highlight every other local day in gray
    for i, ind in enumerate(inds):
        if i%2 != 0: continue
        if i+1 >= len(inds): next_ind = len(x)-1
        else:                next_ind = inds[i+1]
        ax.axvspan(x[ind], x[next_ind], alpha=0.17, color='grey')
        
    # Add text labels for the day of the week
    if addtext:
        for ind in inds:
            ax.text(x[ind]+1, ax.get_ylim()[-1], calendar.day_name[dts_local[ind].weekday()][:3], 
                    ha='left', va='top', color='.15')
            
def station_latlon(stid, stidfile='/home/disk/p/njweber2/pypackages/ensemble_tools/stids.csv'):
    # Read lat/lon for this station from the text file stidfile
    stids = np.genfromtxt(stidfile, delimiter=',', usecols=0, dtype='str')
    if stid in stids:
        locs = np.genfromtxt(stidfile, delimiter=',', usecols=(1,2))
        st_ind = np.where(stids==stid)[0][0]
        lat, lon = locs[st_ind]
        if lon > 180: lon -= 360.
    else:
        raise ValueError('station id "{}" not in file!'.format(stid))
    return lat, lon

def plotting_units(field, data, newunit='F'):
    if field in ['t2m', 'td2m']:   # K to Celsius or Fahrenheit
        if newunit=='C': return data - 273.15
        else: return (data-273.15)*1.8 + 32.
    elif field in ['wnd10m', 'wspd10m']: # m/s to knots
        return data*1.94384
    elif field in ['srate', 'snow']: # mm to inches
        return data*0.0393701
    else:
        return data
    
def timedelta_hours(dt_i, dt_f):
    """ Find the number of hours between two dates """
    return int((dt_f-dt_i).days*24 + (dt_f-dt_i).seconds/3600)

def nearest_ind(array, value):
    return int((np.abs(array-value)).argmin())