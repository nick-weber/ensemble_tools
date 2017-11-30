#!/usr/bin/env python
"""
Module for loading and plotting UW WRF ensemble plumes at different locations
"""

import numpy as np
import xarray
from datetime import datetime
import matplotlib.pyplot as plt
import ens_utilities as ut
import os
import seaborn as sns
sns.reset_orig()


# A dictionary of variables' units (for plot labeling),  {varname : units}
units = {'t2m' : 'F', 'td2m' : 'F', 'wspd10m' : 'kts', 'wdir10m' : 'deg. from N', 'precip' : 'in', 
         'snow' : 'in', 'prate' : 'mm hr$^{-1}$', 'srate' : 'in hr$^{-1}$', 'mslp': 'hPa',
         'rh2m' : '%'}

# A dictionary of colors for each WRF ensemble member
memcols = {'cmcg':'firebrick', 'gasp2':'fuchsia', 'gefs01':'darkblue', 'gefs02':'dodgerblue',
           'gefs03':'dodgerblue', 'gefs04':'dodgerblue', 'gefs05':'dodgerblue', 'gefs06':'dodgerblue',
           'gefs07':'dodgerblue', 'gefs08':'dodgerblue', 'gefs09':'dodgerblue', 'gefs10':'dodgerblue',
           'jmag2':'forestgreen', 'ngps':'gold', 'ukmo':'darkorange'}


class Plumes(xarray.Dataset):
    
    @classmethod
    def from_netcdfs(cls, wrfdir, figdir, domain=1, memberfile=None, verbose=False):
        """
        Initializes a Plumes instance with the ensemble members in [wrfdir] using
        tools.load_plumes().
        
        Requires:
        wrfdir -----> directory containing the ensemble member subdirectories
        domain -----> WRF domain identifies (int)
        memberfile -> name of the text file containing the names of the ensemble members
        
        Returns:
        ensplumes --> a new Plumes instance (wrapped xarray Dataset)
        """
        from .tools import load_plumes
        
        # Get the ensemble member directories from a text file
        if memberfile is None:
            memberfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'members.txt')
        mems = np.genfromtxt(memberfile, dtype=str)
        
        # Load the plumes from the pre-processed netcdfs as an xarray Dataset
        ensplumes, missingmems = load_plumes(wrfdir, mems, dom=domain, returnmissing=True, verbose=verbose)
        mems = [mem for mem in mems if mem not in missingmems]
        assert len(mems) == ensplumes.dims['ens']
        
        # Make sure the figure directory exists
        ut.checkdir(figdir)
            
        # Assign attributes
        ensplumes.attrs.update(workdir=wrfdir, model='WRF', domain=domain, figdir=figdir,
                               mems=mems, missing_mems=missingmems)
        ensplumes.__class__ = cls
        
        # Calculate the rain RATE and snow RATE
        prate = ((ensplumes['precip'] - ensplumes['precip'].shift(Time=1)) / ensplumes.dt()) * 25.4 # in/hr -> mm/hr
        srate = (ensplumes['snow'] - ensplumes['snow'].shift(Time=1)) / ensplumes.dt() # in/hr
        ensplumes = ensplumes.assign(prate=prate, srate=srate)
        
        return ensplumes
    
    
    #==== Functions to get various useful attributes/dimensions ==========================
    def domain(self):
        return self.attrs['domain']
    def model(self):
        return self.attrs['model']
    def idate(self):
        return self.vdates()[0]
    def dt(self):
        return ut.timedelta_hours(self.idate(), self.vdates()[1])
    def workdir(self):
        return self.attrs['workdir']
    def figdir(self):
        return self.attrs['figdir']
    def ntimes(self):    
        return self.dims['Time']
    def mems(self):
        return self.attrs['mems']
    def nmems(self):
        return self.dims['ens']
    def vars(self):
        return [x for x in self.variables.keys() if x not in ['times', 'sites', 'lat', 'lon']]
    def nvars(self):
        return len(self.vars())
    def stations(self):
        return [site.decode('utf-8').strip() for site in self['sites'].values]
    def sites(self):
        return self.stations()
    def vdates(self):
        return [datetime.strptime(dt.decode('utf-8'), '%Y-%m-%d_%H:00:00') for dt in self['times'].values]
    def leadtimes(self):
        return [ut.timedelta_hours(self.idate(), d) for d in self.vdates()]
    def missing_mems(self):
        return self.attrs['missing_mems']
    def n_missing_mems(self):
        return len(self.missing_mems())
    
    
    #==== Miscellaneous utilities ========================================================
    def get_site_latlon(self, stid):
        """ Return the lat and lon of a desired site """
        assert stid in self.sites()
        site_ind = np.where(np.array(self.sites())==stid)[0][0]
        return self['lat'].values[site_ind], self['lon'].values[site_ind]
    
    def get_site_elev(self, stid):
        """ Return the elevation (m) of a desired site """
        assert stid in self.sites()
        site_ind = np.where(np.array(self.sites())==stid)[0][0]
        return self['elev'].values[site_ind]
    
    def get_site_plumes(self, stid, field, asdataarray=True):
        """
        Gets ensemble plumes at a specific site for desired field.
        
        Requires:
        stid
        field
        asdataarray
        
        Returns:
        siteplumes
        """
        assert stid in self.sites()
        assert field in self.vars()
        site_ind = np.where(np.array(self.sites())==stid)[0][0]
        siteplumes = self[field].isel(location=site_ind)
        
        # Modify the units, if necessary
        if field not in ['snow','srate']: siteplumes = ut.plotting_units(field, siteplumes, newunit=units[field])
            
        if asdataarray: return siteplumes
        else:           return siteplumes.values
        
    def format_ts_axis(self, ax, field, x, tick_intvl, style='ticks', trim=True, 
                       ticklabels=True, rot=30):
        
        intvl = int(tick_intvl/self.dt())
        # Format the x and y axes
        ax.text(-0.13, 0.5, '{} [{}]'.format(field, units[field]), rotation=90, va='center', 
                ha='right', transform=ax.transAxes)
        ax.set_xlim([x[0]-0.5,x[-1]+0.5])
        ax.set_xticks(x[::intvl])
            
        # Stylize the "ticks" plot
        if style=='ticks':
            sns.despine(ax=ax, offset=10, trim=trim)
            ax.yaxis.grid()

        # Add date labels on the bottom axis only
        if ticklabels:
            ax.set_xticklabels(['{:%m-%d_%H}Z'.format(dt) for dt in self.vdates()][::intvl])
            for tick in ax.get_xticklabels():
                tick.set_rotation(rot)
                tick.set_horizontalalignment("right")
        else:
            ax.set_xticklabels([])
    
    
    
    #==== Functions plot the ensemble plumes in various ways ============================
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
            
            
    def plot_singlefield(self, stid, field, showfig=False, style='ticks', tick_intvl=6, 
                         col=sns.color_palette()[0], savefile=None, savefig=True, 
                         stid_subdir=True, verbose=False):
        """
        Plots an ensemble plume forecast of a single field at a specific station.

        Requires:
        stid -------> station ID (string); must be in the stidfile (see below)
        field ------> name of the atmospheric variable (string)
        showfig ----> display the figure? If false, figure is closed after saving.
        style ------> seaborn plotting style
        tick_intvl -> number of hours between each x-axis tick (int)
        col --------> color to plot the ensemble plume lines in
        savefile ---> name of the .png file to save the figure to (automatically generated if None)
        savefig ----> do we want to save this figure to a .png file? (bool)
        stid_subdir -> should we create/use a figure subdirectory for this station? (bool)
        """
        # Read lat/lon for this station
        lat, lon = self.get_site_latlon(stid)
        hgt = int(self.get_site_elev(stid) * 3.28084)
        title = '{}  ({:.01f}$^\circ$, {:.01f}$^\circ$) - elevation: {} ft'.format(stid, lat, lon, hgt)

        # Get the ensemble forecast for this location
        if verbose: print('Acquiring plumes for {}...'.format(stid))
        data = self.get_site_plumes(stid, field)   

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
        # Highlight days in local (Pacific) time
        ut.highlight_localtime(ax, self.vdates(), x)

        # Title the figure
        model = '{}_d{:02d}'.format(self.model(), self.domain())
        ax.text(0., 1.01, title, ha='left', va='bottom', transform=ax.transAxes)
        ax.text(1., 1.01, model, ha='right', va='bottom', transform=ax.transAxes)

        # Save the figure, then close or display it
        figdir = self.figdir()
        if stid_subdir:
            figdir = os.path.join(figdir, stid)
            ut.checkdir(figdir)
        if savefile is None: savefile = '{}/plumes_{}_{}_{}.png'.format(figdir, model, field, stid)
        if savefig: plt.savefig(savefile)
        if showfig: plt.show()
        else: plt.close()
        sns.reset_orig()
        
        
        
    def plot_singlefield_wcols(self, stid, field, showfig=False, style='ticks', tick_intvl=6, 
                               savefile=None, savefig=True, stid_subdir=True, verbose=False):
        """
        Plots an ensemble plume forecast of a single field at a specific station, with the
        individual ensemble members highlighted in different colors.

        Requires:
        stid --------> station ID (string); must be in the stidfile (see below)
        field -------> name of the atmospheric variable (string)
        showfig -----> display the figure? If false, figure is closed after saving.
        style -------> seaborn plotting style
        tick_intvl --> number of hours between each x-axis tick (int)
        savefile ----> name of the .png file to save the figure to (automatically generated if None)
        savefig -----> do we want to save this figure to a .png file? (bool)
        stid_subdir -> should we create/use a figure subdirectory for this station? (bool)
        """
        # Read lat/lon for this station
        lat, lon = self.get_site_latlon(stid)
        hgt = int(self.get_site_elev(stid) * 3.28084)
        title = '{}  ({:.01f}$^\circ$, {:.01f}$^\circ$) - elevation: {} ft'.format(stid, lat, lon, hgt)

        # Get the ensemble forecast for this location
        if verbose: print('Acquiring plumes for {}...'.format(stid))
        data = self.get_site_plumes(stid, field)   

        # Use seaborn to set the style
        if verbose: print('Plotting...')
        sns.set(style=style)
        sns.set_context("notebook", font_scale=1.2)
        
        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(10,5))
        plt.subplots_adjust(left=0.13, right=0.85, bottom=0.2, top=0.94)

        # Plot the plumes!
        x = np.arange(1, np.shape(data)[1]+1)
        for m, mem in enumerate(self.mems()):
            # Plot each ensemble member with a thin line
            if mem=='gefs01': lw = 2.
            else: lw = 1.
            if mem=='gefs02': 
                label = 'gefs02-10'
                zo = 1
            elif ('gefs' in mem) and (mem not in ['gefs01', 'gefs02']): 
                label = '_nolegend_'
                zo = 1
            else: 
                label = mem
                zo = 2
            ax.plot(x, data[m,:], color=memcols[mem], linewidth=lw, zorder=zo, label=label)
        # Plot the ensemble mean in *bold*
        ax.plot(x, np.nanmean(data, axis=0), color='0.15', linewidth=3, alpha=0.65, zorder=3, label='MEAN')
        
        # Create the legend
        ax.legend(bbox_to_anchor=(1., 0.5), loc='center left', ncol=1)

        # Stylize/format the axes
        self.format_ts_axis(ax, field, x, tick_intvl, style=style)
        # Highlight days in local (Pacific) time
        ut.highlight_localtime(ax, self.vdates(), x)

        # Title the figure
        model = '{}_d{:02d}'.format(self.model(), self.domain())
        ax.text(0., 1.01, title, ha='left', va='bottom', transform=ax.transAxes)
        ax.text(1., 1.01, model, ha='right', va='bottom', transform=ax.transAxes)

        # Save the figure, then close or display it
        figdir = self.figdir()
        if stid_subdir:
            figdir = os.path.join(figdir, stid)
            ut.checkdir(figdir)
        if savefile is None: savefile = '{}/plumesMC_{}_{}_{}.png'.format(figdir, model, field, stid)
        if savefig: plt.savefig(savefile)
        if showfig: plt.show()
        else: plt.close()
        sns.reset_orig()
        
        
        
    def plot_multipanel(self, stid, showfig=False, style='ticks', tick_intvl=6, 
                        savefile=None, savefig=True, stid_subdir=True, adjust_ylims=True, verbose=False,
                        fields=['t2m', 'td2m', 'mslp', 'wspd10m', 'rain', 'snow'],
                        cols=['tomato', 'royalblue', 'indigo', 'darkorange',
                              'forestgreen', 'royalblue']):
        """
        Plots an ensemble plume forecast of multiple fields at a specific station.

        Requires:
        stid --------> station ID (string); must be in the stidfile (see below)
        showfig -----> display the figure? If false, figure is closed after saving.
        style -------> seaborn plotting style
        tick_intvl --> number of hours between each x-axis tick (int)
        savefile ----> name of the .png file to save the figure to (automatically generated if None)
        savefig -----> do we want to save this figure to a .png file? (bool)
        stid_subdir -> should we create/use a figure subdirectory for this station? (bool)
        fields ------> list of the atmospheric variables to plot (strings)
        cols --------> list of colors (one per field) to plot the ensemble plume lines in (strings)
        """
        # Read lat/lon for this station
        lat, lon = self.get_site_latlon(stid)
        hgt = int(self.get_site_elev(stid) * 3.28084)
        title = '{}  ({:.01f}$^\circ$, {:.01f}$^\circ$) - elevation: {} ft'.format(stid, lat, lon, hgt)

        # Use seaborn to set the style
        sns.set(style=style)
        sns.set_context("notebook", font_scale=1.2)
        
        # If both temp and dewpoint are in [fields], we will plot them on the same axis
        t_and_td = 't2m' in fields and 'td2m' in fields
        if t_and_td:
            tdi = fields.index('td2m')
            tdcol = cols[tdi]
            fields = [f for f in fields if f != 'td2m']
            cols = [col for c, col in enumerate(cols) if c != tdi]
        
        # Create the figure and axes
        fig, axes = plt.subplots(nrows=len(fields), figsize=(8,11))
        plt.subplots_adjust(left=0.13, right=0.98, bottom=0.1, top=0.95)
        
        x = np.arange(1, self.dims['Time']+1)
        for ax, field, col in zip(axes, fields, cols):
            # Get the ensemble forecast for this location
            if verbose: print('Acquiring {} plumes for {}...'.format(field, stid))
            data = self.get_site_plumes(stid, field)
            
            # Load/plot the dewpoint if we're plotting temperature
            if field=='t2m' and t_and_td:
                tddata = self.get_site_plumes(stid, 'td2m')
                self.plot_plumes(ax, x, tddata, tdcol, plotmean=True)
                
            # Plot the individual ensemble members
            self.plot_plumes(ax, x, data, col, plotmean=True)
            
            # Adjust the ylims so that they *exclude* the initialization time (which can be faulty)
            if adjust_ylims and field not in ['rain', 'snow']: 
                ax.set_ylim(np.min(data[:,1:]), np.max(data[:,1:]))

            # Stylize/format the axes
            self.format_ts_axis(ax, field, x, tick_intvl, style=style, trim=False)
            # Highlight days in local (Pacific) time
            ut.highlight_localtime(ax, self.vdates(), x, addtext=(ax==axes[0]))
            if ax != axes[-1]: ax.set_xticklabels([])

        # Title the figure
        model = '{}_d{:02d}'.format(self.model(), self.domain())
        axes[0].text(0., 1.02, title, ha='left', va='bottom', transform=axes[0].transAxes)
        axes[0].text(1., 1.02, model, ha='right', va='bottom', transform=axes[0].transAxes)
        
        # Save the figure, then close or display it
        figdir = self.figdir()
        if stid_subdir:
            figdir = os.path.join(figdir, stid)
            ut.checkdir(figdir)
        if savefile is None: savefile = '{}/multiplumes_{}_{}.png'.format(figdir, model, stid)
        if savefig: plt.savefig(savefile)
        if showfig: plt.show()
        else: plt.close()
        sns.reset_orig()
        
        
        
        
#### Main driver function for operational use #######################################################################

def main(vrbls=['t2m', 'td2m', 'rh2m', 'wspd10m', 'mslp', 'precip', 'snow', 'prate', 'srate'],
         idate=None , verbose=True):
    """
    Uses the class above to plot WRF ensemble plumes for all available stations. Uses plot_singlefield_wcols() to
    make one plume figure per station, per variable, with the different plume lines colored by ensemble member.
    
    Requires
    vrbls --> list of variables to plot ensemble plumes of
    idate --> WRF model initialization date (datetime); default=today
    """
    
    from time import time
    import warnings
    warnings.filterwarnings("ignore") # to suppress the RuntimeWarnings from dividing by zero

    # Get our model initialization date, data directory, and figure directory (default = TODAY)
    if idate is None: idate = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    ensdir = '/home/disk/sage4/mm5rt/nobackup/ensembles/{:%Y%m%d%H}'.format(idate)
    figdir = '/home/disk/p/njweber2/ensemble_figures/{:%Y%m%d%H}'.format(idate)

    # Create Plumes object with these data
    if verbose: print('Loading WRF ensemble plume data...')
    start = time()
    plumes = Plumes.from_netcdfs(ensdir, figdir, domain=3, verbose=verbose)
    #print(plumes)
    plumes.load() # load everything into memory for faster plotting operations
    if verbose: print('{:.02f} min\n'.format((time()-start)/60.))
    
    # Make an individual, member-colored plumes plot for all stations and several meteorological variables
    if verbose: print('Plotting ensemble plumes for {} stations:'.format(plumes.dims['location']))
    start = time()
    for s, site in enumerate(plumes.sites()):
        if ((s+1)%10==0 or s==0) and verbose: print('  on station {}'.format(int(s+1)))
        for vrbl in vrbls:
            plumes.plot_singlefield_wcols(site, vrbl, showfig=False, savefig=True)
    if verbose: print('{:.02f} min\nFinished.\n'.format((time()-start)/60.))
        
        
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ioff()
    main()
    