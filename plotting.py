#!/usr/bin/env python
"""
This module contains functions to plot operational ensemble data
"""
import numpy as np
from time import time
import os
import matplotlib.pyplot as plt
from color_maker.color_maker import color_map
import ens_utilities as ut
import seaborn as sns
sns.reset_orig()

# A dictionary of variables' units (for plot labeling),  {varname : units}
units = {'t2m' : 'F', 'wnd10m' : 'kts', 'prate' : 'mm/h', 'precip' : 'mm', 
         'srate' : 'in/h', 'snow' : 'inches', 'mslp': 'hPa'}

class Plotter:
    """
    Class for storing and plotting ensemble forecast data
    """
    
    def __init__(self, ensemble, figdir='/home/disk/p/njweber2/ensemble_figures', 
                 csvdir='/home/disk/p/njweber2/ensemble_csv_data'):
        """
        Requires:
        ensemble -> an Ensemble class instance storing the forecast data
        figdir ---> directory to save figures to
        csvdir ---> directory to store intermediate text files in
        """
        # Assign class variables
        self.ens = ensemble
        self.figdir = '{}/{:%Y%m%d%H}'.format(figdir, ensemble.idate())
        self.csvdir = '{}/{:%Y%m%d%H}'.format(csvdir, ensemble.idate())
        # Create our output directories if they do not exist
        for direc in [self.figdir, self.csvdir]:
            if not os.path.isdir(direc):
                os.system('mkdir {}'.format(direc))
                
    #==============================================================================================
    #======= UTILITY FUNCTIONS ====================================================================
    #==============================================================================================
    def get_plumes(self, field, pt, interp='linear', verbose=False):
        start = time()
        # Use the get_timeseries function in the Ensemble class
        if verbose: print('Fetching/calculating/interpolating field: "{}"'.format(field))
        data = self.ens.get_timeseries(field, pt, method=interp)
        assert data.shape[0] == self.ens.nmems()
        # Convert the units
        data = ut.plotting_units(field, data, newunit=units[field])
        if verbose: print('Calculation/interpolation time: {:.02f} min'.format((time()-start)/60.))
        return data
    
    def format_ts_axis(self, ax, field, x, tick_intvl, style='ticks', ticklabels=True, rot=30):
        
        intvl = int(tick_intvl/self.ens.dt())
        # Format the x and y axes
        ax.set_ylabel('{} [{}]'.format(field, units[field]))
        ax.set_xlim([x[0]-0.5,x[-1]+0.5])
        ax.set_xticks(x[::intvl])
            
        # Stylize the "ticks" plot
        if style=='ticks':
            sns.despine(ax=ax, offset=10, trim=True)
            ax.yaxis.grid()

        # Add date labels on the bottom axis only
        if ticklabels:
            ax.set_xticklabels(['{:%m-%d_%H}Z'.format(dt) for dt in self.ens.vdates()][::intvl])
            for tick in ax.get_xticklabels():
                tick.set_rotation(rot)
                tick.set_horizontalalignment("right")
        else:
            ax.set_xticklabels([])
        
    ###############################################################################################
    #### PLOTTING FUNCTIONS #######################################################################
    ###############################################################################################
    
    
    #======= PROBABILITY MAPS =====================================================================
    
    def plot_prob_thresh(self, field, thresh, dt1, dt2, title=None, cmap=plt.cm.viridis_r,
                         greaterthan=True, m=None, show=True, savedir=None, 
                         probability=False, verbose=False):

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10,6))
        fig.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.95)

        # Draw onto the Basemap object
        if m is None:
            if self.ens.model()=='WRF': m = self.ens.bmap()
            else: raise ValueError('Need to supply a Basemap projection!')
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        m.drawparallels(np.arange(-90,91,5),labels=[1,0,0,0],fontsize=10, dashes=[5,3])
        m.drawmeridians(np.arange(0,361,10),labels=[0,0,0,1],fontsize=10, dashes=[5,3])

        # Project lats/lons
        x, y, = self.ens.project_coordinates(m)

        # Find the indices corresponding to the desired time period
        t1 = ut.nearest_ind(self.ens.vdates(), dt1)
        t2 = ut.nearest_ind(self.ens.vdates(), dt2)+1

        # Get the data for the desired time period
        if verbose: print('Loading {} data...'.format(field))
        if field in ['prate','srate']:
            if t1==0: 
                data = self.ens.isel(time=range(t1,t2+1)).get_field(field)[:,:-1,:,:]
            else:    
                data = self.ens.isel(time=range(t1-1,t2+1)).get_field(field)[:,1:-1,:,:]
        else:
            data = self.ens.isel(time=range(t1,t2)).get_field(field)
        data = ut.plotting_units(field, data, newunit=units[field])

        # Find the probability of exceedance of the given threshold
        if greaterthan:
            gtlt = '>'
            exceed = (data > thresh).astype(int)
        else:
            gtlt = '<'
            exceed = (data < thresh).astype(int)
        if verbose: print('Calculating prob( {} {} {:.01f} )...'.format(field, gtlt, thresh))
        # FIRST we just want to see if ANY of the times between dt1 and dt2 exceeded the threshold
        exceed = (np.sum(exceed, axis=1)>0).astype(int)
        # NOW we get the probability at each grid point
        prob = np.sum(exceed, axis=0)
        problevs = np.arange(self.ens.dims['ens'])+1
        probunits = '# of members'
        if probability: 
            prob = (prob / self.ens.dims['ens']) * 100.
            problevs = np.arange(10, 101, 5)
            probunits = '%'

        # Contour-fill the probability of exceedance values
        if verbose: print('Plotting probabilities...')
        cs = ax.contourf(x, y, prob, levels=problevs, cmap=cmap, antialiasing=False)
        # Colorbar and contour labels
        cax = fig.add_axes([0.89, 0.05, 0.03, 0.9])
        cb = plt.colorbar(cs, cax=cax, label=probunits)
        cb.set_ticks(problevs)
        cb.set_ticklabels(problevs)

        # Put model info in upper-right corner
        if self.ens.model()=='WRF': model = 'WRF_d{:02d}'.format(self.ens.domain())
        else:                       model = self.ens.model()
        txt = ax.text(0.98, 0.975, model, color='k', ha='right', va='top', 
                      transform=ax.transAxes, fontsize=14)
        txt.set_bbox(dict(facecolor='white', alpha=0.65, edgecolor='k'))


        # Title
        if title is not None:
            ax.set_title(title, loc='left')
        else:
            if dt1==dt2:
                titledate = ' on {:%Y-%m-%d_%H}'.format(dt1)
            else:
                titledate = ' btw. {:%Y-%m-%d_%H} and {:%Y-%m-%d_%H}'.format(dt1, dt2)
            title = 'P( {} {} {:.01f} {} ){}'
            ax.text(0., 1.01, title.format(field, gtlt, thresh, units[field], titledate), 
                    ha='left', va='bottom', transform=ax.transAxes, fontsize=12)
            ax.text(1., 1.01, 'init: {:%Y-%m-%d %H:00}'.format(self.ens.idate()), 
                    ha='right', va='bottom', transform=ax.transAxes, fontsize=12)
        if savedir is not None:
            plt.savefig('{}/{}_prob-{}-{:.01f}_{:%Y%m%d%H}-{:%Y%m%d%H}.png'.format(savedir, model, field, thresh,dt1,dt2))
        if show: plt.show()
        else: plt.close()
            
    
    #======= ENSEMBLE PLUMES ======================================================================
    
    def plot_plumes(self, ax, x, data, col, plotmean=True):
        """
        Plots ensemble plumes onto a given axis.
        
        Requires:
        ax -------> the matplotlib axis object to plot onto
        x --------> values corresponding to the x (time) axis (array-like, 1D)
        data -----> 2D data of shape (# members, # times)
        col ------> color to plot the plume lines in (string)
        plotmean -> plot the ensemble mean in bold? (bool)
        """
        for mem in range(np.shape(data)[0]):
            # Plot each ensemble member with a thin line
            ax.plot(x, data[mem,:], color=col, linewidth=0.7, alpha=0.7, zorder=1)
        if plotmean:
            # Plot the ensemble mean in *bold*
            ax.plot(x, np.nanmean(data, axis=0), color=col, linewidth=3, zorder=2)
    
    def plumes(self, stid, field, interp='linear', showfig=False, style='ticks', tick_intvl=6, 
               col=sns.color_palette()[0], savefile=None, savefig=True, verbose=False):
        """
        Plots an ensemble plume forecast of a single field at a specific station.

        Requires:
        stid -------> station ID (string); must be in the stidfile (see below)
        field ------> name of the atmospheric variables (string)
        interp -----> interpolation method to get ensemble forecast at point ('nearest' or 'linear')
        showfig ----> display the figure? If false, figure is closed after saving.
        style ------> seaborn plotting style
        tick_intvl -> number of hours between each x-axis tick (int)
        col --------> color to plot the ensemble plume lines in
        savefile ---> name of the .png file to save the figure to (automatically generated if None)
        savefig ----> do we want to save this figure to a .png file? (bool)
        """
        # Read lat/lon for this station from the text file stidfile
        lat, lon = ut.station_latlon(stid)
        title = '{} ({}$^\circ$, {}$^\circ$)'.format(stid, lat, lon)

        # Get the ensemble forecast for this location using NN or linear interpolation
        data = self.get_plumes(field, (lat,lon), interp=interp, verbose=verbose)   

        # Use seaborn to set the style
        if verbose: print('Plotting...')
        sns.set(style=style)
        sns.set_context("notebook", font_scale=1.2)
        
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(10,5))
        plt.subplots_adjust(left=0.13, right=0.98, bottom=0.2, top=0.94)

        # Plot the individual ensemble members
        x = np.arange(1, np.shape(data)[1]+1)
        self.plot_plumes(ax, x, data, col, plotmean=True)

        # Stylize/format the axes
        self.format_ts_axis(ax, field, x, tick_intvl, style=style)

        # Title the figure
        if self.ens.model()=='WRF': model = '{}_d{:02d}'.format(self.ens.model(), self.ens.domain())
        else:                       model = self.ens.model()
        ax.set_title('{} ensemble plumes for {}'.format(model, title), loc='left')

        # Save the figure, then close or display it
        if savefile is None: savefile = '{}/{}_{}_plumes_{}.png'.format(self.figdir, model, field, stid)
        if savefig: plt.savefig(savefile)
        if showfig: plt.show()
        else: plt.close()
        sns.reset_orig()

    def meteorogram(self, stid, interp='linear', showfig=False, style='ticks', tick_intvl=6,
                    savefile=None, savefig=True, verbose=False):
        """
        Plots a multi-panel ensemble plume forecast at a specific station.
        Currently plots 2m temperature, 10m wind speeds, and precipitation rate.
        
        Requires:
        stid -------> station ID (string); must be in the        (see below)
        interp -----> interpolation method to get ensemble forecast at point ('nearest' or 'linear')
        showfig ----> display the figure? If false, figure is closed after saving.
        tick_intvl -> number of hours between each x-axis tick (int)
        savefile ---> name of the .png file to save the figure to (automatically generated if None)
        savefig ----> do we want to save this figure to a .png file? (bool)
        """
        # Some dictionaries for plotting (e.g., boxplot shading based on ens mean)
        vrbls = ['t2m','wnd10m','prate']
        cmaps = {'t2m'    : color_map('WhiteBlueGreenYellowRed'),
                 'wnd10m' : plt.cm.plasma,
                 'prate'  : color_map('GMT_drywet')}
        cmaplims = {'t2m' : (25,85), 'wnd10m' : (0,25), 'prate' : (0,7)}
        
        # Read lat/lon for this station from the text file stidfile
        lat, lon = ut.station_latlon(stid)
        title = '{} ({}$^\circ$, {}$^\circ$)'.format(stid, lat, lon)
        
        # Get the ensemble forecast for this location using NN or linear interpolation
        plumes = [self.get_plumes(var, (lat,lon), interp=interp, verbose=verbose) for var in vrbls]
        
        # Use seaborn to stylize the figure
        if verbose: print('Plotting...')
        sns.set(style=style)
        sns.set_context("notebook", font_scale=1.2)
        
        # Create the figure and axes
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,10))
        plt.subplots_adjust(hspace=0.1, left=0.1, right=0.95, bottom=0.15, top=0.95)
        intvl = int(tick_intvl/self.ens.dt())  # x-axis labeling interval

        # Loop through each axis to plot the data
        x = range(1,len(plumes[0][0,:])+1)
        for var, ax, data in zip(vrbls, axes, plumes):
            assert data.shape[0] == self.ens.nmems()
            
            # Plot the boxplots for each variable on each axis
            bp = ax.boxplot(data, widths=0.7, patch_artist=True, zorder=2)
            
            # Plot the individual ensemble members
            self.plot_plumes(ax, x, data, '.15', plotmean=False)
            
            # Make the box lines gray and shade each boxplot acoording to its ens. mean
            for ensmean, b, m in zip(np.nanmean(data,axis=0), bp['boxes'], bp['medians']):
                # Color the lines dark grey
                b.set(color='.15', linewidth=1.4)
                m.set(color='.15', linewidth=1.4)
                
                # Make the fill color correspond to the ens mean value
                minv, maxv = cmaplims[var]
                frac = (ensmean - minv) / (maxv - minv)
                if frac<0: frac = 0.
                if frac>1: frac = 1.
                b.set(facecolor=cmaps[var](frac))

            # Color the whiskers and caps dark grey as well
            for w,c in zip(bp['whiskers'], bp['caps']):
                w.set(color='.15', linewidth=1.4)
                c.set(color='.15', linewidth=1.4)
            
            # Stylize/format the axes
            self.format_ts_axis(ax, var, x, tick_intvl, style=style, ticklabels=(ax==axes[-1]))
            # Highlight days in local (Pacific) time
            ut.highlight_localtime(ax, self.ens.vdates(), x, addtext=(ax==axes[0]))

        # Title the figure
        if self.ens.model()=='WRF': model = '{}_d{:02d}'.format(self.ens.model(), self.ens.domain())
        else:                       model = self.ens.model()
        axes[0].set_title('{} ensemble meteorogram for {}'.format(model, title), loc='left')

        # Save the figure, then close or display it
        if savefile is None: savefile = '{}/{}_meteorogram_{}.png'.format(self.figdir, model, stid)
        if savefig: plt.savefig(savefile)
        if showfig: plt.show()
        else: plt.close()
        sns.reset_orig()
    
    
##############################################################################################################################
#### OTHER UTILITIES #########################################################################################################
##############################################################################################################################