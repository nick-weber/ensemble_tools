#!/usr/bin/env python
"""
This module contains functions to plot gefs operational ensemble data
Currently does not use ensemble.py 
It needs to be updated to use ensemble.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import pyart
import netCDF4 as nc4

from matplotlib import cm, colors

class Plotter:
    #class that pulls in the correct GEFS
    def __init__(self,initday='20170914',inithour='00',figdir='/home/disk/anvil2/ensembles/gefs/plots',
                gefsdir='/home/disk/anvil2/ensembles/gefs/forecasts'):
        self.initday = initday
        self.inithour = inithour
        self.figdir = figdir
        self.gefsdir = gefsdir+'/'+initday+'/'+inithour+'/'

        os.chdir(self.figdir)
        os.system('mkdir -p '+self.initday)

    def init_basemap(self,ax):   
        #initialize basemap
        #instead of plotzone can eventually feed this function preset domains from another script
        plotzone=[-180,-100,20,55]
        lon0 = -150.4
        lat0 = 47.53

        map = Basemap(projection='stere',resolution='h',area_thresh=10000,
            lon_0 = lon0,
	    lat_0 = lat0,
            llcrnrlat=plotzone[2],urcrnrlat=plotzone[3],\
            llcrnrlon=plotzone[0],urcrnrlon=plotzone[1],\
            ax=ax) #width = plotzone,height=plotzone

        map.drawcoastlines()
        map.drawstates()
        map.drawcountries()
        map.drawparallels(np.arange(-80,81,5),labels=[1,0,0,0], size=8, linewidth=1.0)
        map.drawmeridians(np.arange(0,360,10),labels=[0,0,0,1], size=8, linewidth=1.0)
        map.fillcontinents()
        return map

    def plot_mslp_contours(self):
        #plots MSLP contours
        #1000 hpa solid line
        #members 1-20 in different colors

        #make figdirs
        os.system('mkdir -p '+self.figdir+'/'+self.initday)
        os.system('mkdir -p '+self.figdir+'/'+self.initday+'/'+self.inithour)
        os.system('mkdir -p '+self.figdir+'/'+self.initday+'/'+self.inithour+'/mslp_contours')
        outdir = self.figdir+'/'+self.initday+'/'+self.inithour+'/mslp_contours/'
        datadir = self.gefsdir+'/'+self.initday+'/'

        #set levels
        levels = np.arange(950,1006,4)
        #loop through 97 timesteps. For each time step, plot all 20 members
        for t in range(0,97): 
            #initialize figure
            fig = plt.figure()
            fig.set_size_inches(18,8)
            ax = fig.add_subplot(111)
            m = self.init_basemap(ax)
            for mem in range(0,21): #21
                if mem == 0:
                    #control run
                    nc_file = self.gefsdir+'/control/gefs_control.nc'
                    nc_data = nc4.Dataset(nc_file,'r')
                else:
                    nc_file = self.gefsdir+"{0:0=2d}".format(mem)+'/gefs_mem'+"{0:0=2d}".format(mem)+'.nc'
                    nc_data = nc4.Dataset(nc_file,'r')
                #loop through times
                if t == 0:
                    #get lons/lats
                    lons,lats = np.meshgrid(nc_data.variables['longitude'],nc_data.variables['latitude'])
                    lons[np.where((lons > 180))]-=360
                    #get datetime of this step
                    steptime = datetime.datetime.utcfromtimestamp(nc_data.variables['time'][t])
                    #get lat/lon
                    mlons,mlats = m(lons,lats)
                mslp = nc_data.variables['mslp'][t,:,:]/100.
                if mem != 0:
                    prmsl_mem = m.contour(mlons,mlats,mslp,levels,clabels=levels,linewidths=0.65,cmap='plasma_r',
                                        vmin=levels[0],vmax=levels[-1],alpha=0.4)
                else:
                    prmsl_ctl = m.contour(mlons,mlats,mslp,levels,clabels=levels,linewidths=1.2,cmap='plasma_r',
                                        vmin=levels[0],vmax=levels[-1])

            #colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="2%", pad=0.15)
            cbar = plt.colorbar(prmsl_ctl, cax=cax, ticks=levels)
            cbar.lines[0].set_linewidths(10)
            cbar.outline.set_linewidth(0)

            #titles and saving plot
            initdatetime = datetime.datetime.utcfromtimestamp(nc_data.variables['time'].getncattr('reference_time'))
            initstring = initdatetime.strftime('%a %HZ %d-%b-%Y')
            stepstring = steptime.strftime('%a %HZ %d-%b-%Y')
            outstring = 'gefs_mslp_contours_'+initdatetime.strftime('%Y%m%d%H')+'_F'+ "{0:0=3d}".format(t*3)
            plottitle = 'GEFS MSLP Ctl+20 Mem Spread | Init: '+initstring+' | F'+ "{0:0=3d}".format(t*3)+' | Valid: '+stepstring
            ax.set_title(plottitle) 
            plt.savefig(outdir+outstring+'.png', bbox_inches='tight')
            plt.close('all')
            print ('Plotted '+outstring)
        
#test
initday='20170914'
pplot = Plotter(initday=initday,inithour='12')
pplot.plot_mslp_contours()
