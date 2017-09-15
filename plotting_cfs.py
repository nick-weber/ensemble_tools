#!/usr/bin/env python
"""
This module contains functions to plot CFSv2 operational ensemble data
<<<<<<< HEAD
Hi Nick! 
=======
Hi Nick!
>>>>>>> 958d80ad33198cbccb48111bd2d7c4e572240765
"""
import numpy as np
from time import time
import os
import matplotlib.pyplot as plt
from color_maker.color_maker import color_map
import seaborn as sns
sns.reset_orig()

class Plotter:
    """
    Class for storing and plotting ensemble forecast data
    """
    
    def __init__(self, ensemble, figdir='/glade/u/home/njweber2/cfs/figures', 
                 csv_dir='/glade/u/home/njweber2/cfs/csv_data'):
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
    #======= PLOTTING FUNCTIONS ===================================================================
    #==============================================================================================

    def meteorogram(self, stid, interp='linear', showfig=False, style='ticks', 
                    savefile=None, verbose=False):
        """
        Plots an ensemble plume (daily-averaged) forecast at a specific station.
        Currently plots 2m temperature, 10m wind speeds, and precipitation rate.
        
        Requires:
        stid -----> station ID (string); must be in the stidfile (see below)
        interp ---> interpolation method to get ensemble forecast at point ('nearest' or 'linear')
        showfig --> display the figure? If false, figure is closed after saving.
        style ----> seaborn plotting style
        savefile -> name of the .png file to save the figure to (automatically generated if None)
        """
        stidfile = '/home/disk/user_www/njweber2/nobackup/cfs/stids.csv' # contains station lat/lons
        
        # Read lat/lon for this station from the text file stidfile
        stids = np.genfromtxt(stidfile, delimiter=',', usecols=0, dtype='str')
        if stid in stids:
            locs = np.genfromtxt(stidfile, delimiter=',', usecols=(1,2))
            st_ind = np.where(stids==stid)[0][0]
            lat, lon = locs[st_ind]
        else:
            raise ValueError('station id "{}" not in file!'.format(stid))
        title = '{} ({}$^\circ$, {}$^\circ$)'.format(stid, lat, lon)

        # Get the ensemble forecast for this location using NN or linear interpolation
        start = time()
        if verbose: print('Interpolating data to get timeseries...')
        # Use the get_timeseries function in the Ensemble class
        t2m = self.ens.get_timeseries('t2m', (lat, lon), method=interp)
        wnd10m = np.sqrt(self.ens.get_timeseries('u10', (lat, lon), method=interp)**2 + \
                         self.ens.get_timeseries('v10', (lat, lon), method=interp)**2)
        prate = self.ens.get_timeseries('prate1d', (lat, lon), method=interp)
        print('Interpolation time: {:.02f} min'.format((time()-start)/60.))    

        # Some dictionaries for plotting (e.g., boxplot shading based on ens mean)
        vrbls = ['t2m','wnd10m','prate']
        units = {'t2m' : 'C', 'wnd10m' : 'm s$^{-1}$', 'prate' : 'mm d$^{-1}$'}
        cmaps = {'t2m'    : color_map('WhiteBlueGreenYellowRed'),
                 'wnd10m' : plt.cm.plasma,
                 'prate'  : color_map('GMT_drywet')}
        cmaplims = {'t2m' : (-10,25), 'wnd10m' : (-4,10), 'prate' : (3,30)}

        # Use seaborn to set the style
        if verbose: print('Plotting...')
        sns.set(style=style)
        sns.set_context("notebook", font_scale=1.2)
        # Create the figure and axes
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12,12))
        plt.subplots_adjust(hspace=0.07, left=0.05, right=0.95, bottom=0.5, top=0.95)

        # Loop through each axis to plot the data
        for var, ax, data in zip(vrbls, axes, [t2m, wnd10m, prate]):
            # Plot the boxplots for each variable on each axis
            bp = ax.boxplot(data.T, widths=0.7, patch_artist=True, zorder=2)
            
            # Plot the individual ensemble members
            for mem in range(np.shape(data)[-1]):
                ax.plot(range(1,len(data[:,mem])+1), data[:, mem], color='.15', linewidth=0.7, alpha=0.7, zorder=1)
            
            # Make the box lines gray and shade each boxplot
            for ensmean, b, m in zip(np.nanmean(data,axis=-1), bp['boxes'], bp['medians']):
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
            
            # Label the y-axis appropriately
            ax.set_ylabel('{} [{}]'.format(var, units[var]))

        # Stylize the plot by removing unnecessary axis lines
        if style=='ticks': sns.despine(offset=10, trim=True)

        # Add date labels on the bottom axis only
        axes[0].set_xticklabels([])
        axes[1].set_xticklabels([])
        axes[2].set_xticklabels(['{:%b-%d}'.format(dt) for dt in self.ens.vdates()])
        for tick in axes[2].get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")

        # Title the figure
        axes[0].set_title('CFSv2 ensemble meteorogram for {}'.format(title), loc='left')

        # Save the figure, then close or display it
        if savefile is None:
            plt.savefig('{}/meteorogram{}.png'.format(self.figdir, stid))
        else:
            plt.savefig(savefile)
        if showfig: plt.show()
        else: plt.close()
        sns.reset_orig()
    
    
