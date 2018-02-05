#!/usr/bin/env python
"""
Module containing classes for cyclone centers (at one time), cyclone tracks,
ensemble cyclone track clusters, and full forecast sets of track clusters.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_orig()

class CycloneCenter:
    """
    Contains the location, maximum vorticity, effective radius, and more for a tracked cyclone
    center at one time
    """
    def __init__(self, ens, indices, time, mem, vort, radius, lat, lon):
        """
        Variable descriptions given in the comments below
        """
        self.ens = ens            # an Ensemble object from whose data this CycloneCenter is derived
        self.indices = indices    # indices in the full lat/lon grid  (tuple)
        self.time = time          # time step index from initialization (int)
        self.mem = mem            # member number (int)
        self.vort = vort          # cyclone maximum filtered vorticity (float)
        self.radius = radius      # radius of cyclone, in # of grid cells (int)
        self.lat = lat            # latitude of cyclone center (float)
        self.lon = lon            # longitude of cyclone center (float)
        
    # Functions for retrieving infomation about other fields within the cyclone's radius
    def field_at_center(self, field):
        """Returns the value of an atmospheric variable at the cyclone's center (where vort=max)"""
        vrbl = self.ens[field].isel(time=self.time, ens=self.mem)
        return vrbl.isel(latitude=self.indices[0], longitude=self.indices[1])
    
    def field_in_rad(self, field):
        """Returns an array of all values of an atmospheric variable within
           this cyclone's radius"""
        vrbl = self.ens[field].isel(time=self.time, ens=self.mem-1)
        x2d, y2d = np.meshgrid(np.arange(np.shape(vrbl)[1]), np.arange(np.shape(vrbl)[0]))
        incircle = (x2d - self.indices[1])**2 + (y2d - self.indices[0])**2 <= self.radius**2
        return vrbl.values[incircle]
        
    
    def max_field(self, field):
        """Returns the maximum value of an atmospheric field within this cyclone's radius"""
        return np.max(self.field_in_rad(field))
    
    def min_field(self, field):
        """Returns the minimum value of an atmospheric field within this cyclone's radius"""
        return np.min(self.field_in_rad(field))
    
    def mean_field(self, field):
        """Returns the average value of an atmospheric field within this cyclone's radius"""
        return np.nanmean(self.field_in_rad(field))
                                    
        
    def meets_conditions(self, track, before=True):
        """
        Returns true if this cyclone center meets the conditions to be added onto the given cyclone track
        See Flaounas et al. 2017 for more details.
        
        Conditions:  (1) the location of the next center must be within a 5-deg-lat, 10-deg-lon range
                     (2) the maximum vorticity between the two consecutive steps must not differ by >50%
                     (3) if the sum of two consecutive displacements is more than 3-deg
                         long, then the angle between these displacements must be greater than 90-deg
        """
        from .tracker import angle_btw_pts
        
        if before:  # we wish to PREPEND this cyclone to the track
            # Is this cyclone at the preceding time?
            if self.time != track.times()[0]-1:
                return False
            trackcyclone = track.centers[0]
            threecenters = [self] + list(track.centers[:2])
        else:       # we wish to APPEND this cyclone to the track
            # Is this cyclone at the preceding time?
            if self.time != track.times()[-1]+1:
                return False
            trackcyclone = track.centers[-1]
            threecenters = list(track.centers[-2:]) + [self]
        
        # CONDITION 1
        dlat = abs(self.lat-trackcyclone.lat)
        dlon = abs(self.lon-trackcyclone.lon)
        if dlon > 180.: dlon -= 180  # account for wrapping around 0/360 degree meridian
        if dlat > 5 or dlon > 10:
            return False
        # CONDITION 2
        if not ( 0.5 <= self.vort/trackcyclone.vort <= 1.5 ):
            return False
        # CONDITION 3
        if len(threecenters)==3:  # i.e., if this isn't the first point being appended to the track
            totaldistance = threecenters[0].distance_from(threecenters[1]) + \
                            threecenters[1].distance_from(threecenters[2])
            if totaldistance<=3:
                return True
            angle = angle_btw_pts([threecenters[0].lat,threecenters[0].lon], 
                                  [threecenters[1].lat,threecenters[1].lon],
                                  [threecenters[2].lat,threecenters[2].lon])
            if angle <= 90.:
                return False
        # If we passed all the tests, then this is a valid addition to the track!
        return True
            
    def distance_from(self, othercenter):
        """Returns the distance from another CycloneCenter in degrees"""
        dlat = self.lat - othercenter.lat
        dlon = self.lon - othercenter.lon
        # account for wrapping around longitude 0/360
        if  dlon > 180:   dlon -= 360.
        elif dlon < -180: dlon += 360.
        return np.sqrt(dlat**2 + dlon**2)
        
################################################################################################################
        
class CycloneTrack:
    """
    Primarily a list of CycloneCenter objects, this class contains information about a cyclone
    track across numerous times (for a single forecast, i.e. one member).
    """
    def __init__(self, centers):
        """
        Variables/attributes described in comments below
        """
        # The following is a list of length N
        # where N = number of points in this cyclone track
        self.centers = centers # list of CycloneCenter objects
        
        # Other attributes
        self.finished = False # if True, this track cannot be extended anymore
        
    # Functions to get cyclone attributes along its track
    def member(self):
        return self.centers[0].mem
    def ntimes(self):
        return len(self.centers)
    def length(self):
        return len(self.centers)
    def center_inds(self):
        return [c.indices for c in self.centers]
    def times(self, asarr=False):
        if asarr: return np.array([c.time for c in self.centers])
        else: return [c.time for c in self.centers]
    def vorts(self, asarr=False):
        if asarr: return np.array([c.vort for c in self.centers])
        else: return [c.vort for c in self.centers]
    def radii(self, asarr=False):
        if asarr: return np.array([c.radius for c in self.centers])
        else: return [c.radius for c in self.centers]
    def lats(self, asarr=False):
        if asarr: return np.array([c.lat for c in self.centers])
        else: return [c.lat for c in self.centers]
    def lons(self, asarr=False):
        if asarr: return np.array([c.lon for c in self.centers])
        else: return [c.lon for c in self.centers]
    def latlons(self, asarr=False):
        return self.lats(asarr=asarr), self.lons(asarr=asarr)
        
    # Function to get a model field along the cyclone track
    def field_along_track(self, field, value='max'):
        """
        Returns a timeseries of the maximum, minimum, average, or center-value of an 
        atmospheric variable along this cyclone track.
        
        Maximum, minimum, and average values are computed from a list of points encompassed
        by the cyclone's time-dependent radius.
        """
        if value=='max':
            return np.array([c.max_field(field) for c in self.centers])
        elif value=='min':
            return np.array([c.min_field(field) for c in self.centers])
        elif value in ['mean', 'avg']:
            return np.array([c.mean_field(field) for c in self.centers])
        elif value in ['center', 'atcenter']:
            return np.array([c.field_at_center(field) for c in self.centers])
        else:
            raise ValueError('invalid value "{}"'.format(value))
            
################################################################################################################
        
class TrackCluster:
    """
    Primarily a list of CycloneTrack objects, this class stores and displays data from an ensemble
    of cyclone tracks (i.e., an ensemble forecast of one storm).
    """
    def __init__(self, tracks, cluster_id, full_id=None, color='red'):
        """
        Requires:
        tracks -----> a list of CycloneTrack objects, representing an ensemble of tracks for one cyclone
        cluster_id -> a integer ID for this cluster (output from HBDSCAN algorithm)
        color ------> the color to use when plotting this cluster of cyclone tracks (string)
        """
        
        # Make sure we got rid of all the duplicate members
        if cluster_id != -1: 
            mems = np.array([t.member() for t in tracks])
            assert len(mems)==len(set(mems))
            
        # Now let's just sort the tracks by member before storing them
        tracks.sort(key=lambda x: x.member())
        self.tracks = tracks
        # And let's give this cluster of tracks an ID and a plotting color
        self.cluster_id = cluster_id
        if full_id is None and cluster_id!=-1: full_id = 'cyc{:02d}'.format(cluster_id)
        elif full_id is None and cluster_id==-1: full_id = 'noclust'
        self.full_id = full_id
        self.color = color
        
    # Functions to acquire simple attributes
    def ntracks(self):
        return len(self.tracks)
    def nmems(self):
        return len(self.tracks)
    
    #==== Some plotting functions ==================================================
    def plot_plumes(self, field, value='max', ax=None, title=True, plotmean=True, 
                    plotenscount=True, factor=1., unit='', **plot_args):
        """
        Plots ensemble plumes over all forecast hours for a desired model field.
        
        Requires:
        field ----> name of the model field (string)
        value ----> 'max', 'min', 'mean', or 'center'
        ax -------> axis object to plot onto
        title ----> if True, will generate/display a title
        plotmean -> if True, will plot the ensemble mean in a darker hue
        factor ---> factor to multiply the timeseries by (float; e.g., .01 to get from Pa to hPa)
        unit -----> units for the model field (string)
        
        Returns:
        ax -------> the axis object used for plotting
        fig ------> (optional) the new, created figure (if input ax was None)
        """
        from matplotlib.ticker import AutoMinorLocator

        # Create the figure, axis if we need to
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.despine(offset=10, ax=ax)
            ax.yaxis.grid()
        # How many times are in the forecast? How many members are in this cluster?
        nt = self.tracks[0].centers[0].ens.ntimes()
        nm = self.nmems()
        ally = np.empty((nm, nt))
        ally[:,:] = np.nan
        # How many hours are between each timestep?
        dt = self.tracks[0].centers[0].ens.dt()
        # Plot each track
        min_x = 100; max_x = 0  # these are for creating ticks later
        for t, track in enumerate(self.tracks):
            x = track.times(asarr=True)
            if min(x) < min_x: min_x = min(x)
            if max(x) > max_x: max_x = max(x)
            # get the field data to plot
            y = track.field_along_track(field, value=value)
            ax.plot(x*dt, y*factor, color=self.color, alpha=0.6, zorder=1, **plot_args)
            ally[t, x] = y  # store values to compute mean later
        # Compute/plot the ensemble mean
        if plotmean:
            # We only want to average at times where >50% of the members have data
            ybar = np.array([np.nanmean(ally[:,t]) if len(np.where(~np.isnan(ally[:,t]))[0])>0.5*nm \
                             else np.nan for t in range(nt)])
            darkercolor = tuple([rgb*0.8 for rgb in self.color])
            ax.plot(np.arange(nt)*dt, ybar*factor, color=darkercolor, linewidth=4, zorder=2, **plot_args)
        # Set major ticks every 24h and minor ticks every 6h
        ax.set_xticks(np.arange(min_x-min_x%24, max_x*dt+24, 24))
        ax.set_xticklabels(['F{:02d}d'.format(int(lead/24)) for lead in ax.get_xticks()], 
                           rotation=20, ha='right')
        ax.xaxis.set_minor_locator(AutoMinorLocator(int(24/dt)))
        # Plot the ensemble member count
        if plotenscount and self.cluster_id != -1:
            ax.set_ylabel(field)
            ax2 = ax.twinx()
            count = np.array([len(np.where(~np.isnan(ally[:,t]))[0]) for t in range(nt)])
            inds = np.where(count>0)[0]
            ax2.bar(np.arange(nt)[inds]*dt, count[inds], dt, color='grey', zorder=0, alpha=0.25)
            ax2.set_yticks(np.arange(nm)+1)
            ax2.yaxis.tick_right()
            sns.despine(right=False, left=True, offset=10, ax=ax2)
            ax2.set_ylabel('# of members')
            
        # Create a title
        if title:
            model = self.tracks[0].centers[0].ens.model()
            idate = self.tracks[0].centers[0].ens.idate()
            if unit != '': unit = ' [{}]'.format(unit)
            titl = '{} {} {}{} plumes -- init: {:%Y-%m-%d %H:00}'.format(model, value, field, unit, idate)
            # main title on the left
            ax.text(0, 1.02, titl, transform=ax.transAxes, ha='left', va='bottom')
            # cluster ID on the right
            ax.text(1, 1.02, self.full_id, transform=ax.transAxes, ha='right', va='bottom', color=self.color)
        try: return fig, ax  # if no axis was provided, return the new figure and its axis
        except: return ax    # otherwise, just return the supplied axis
        
################################################################################################################
        
class EnsembleTrackClusters(dict):
    """
    Essentially a dictionary of TrackCluster objects, this class computes (using machine learning) and stores
    clusters of cyclones from an ensemble forecast system.
    
    The HBDSCAN clustering algorithm (by Campello, Moulavi, and Sander) is used to cluster the cyclones tracks:
    http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb
    """
    # A list of colors to use to distinguish the cyclone track clusters:
    palette = sns.xkcd_palette(['purple', 'blue', 'green', 'pink', 'brown', 'red', 'mustard', 'teal', 'orange', 
                                'light green', 'magenta', 'yellow', 'sky blue', 'lime green', 'puke', 'turquoise', 
                                'lavender', 'bluish', 'tan', 'aqua', 'mauve', 'olive']*5)
    
    def __init__(self, alltracks, min_cluster_size=6, latbounds=(20, 70), lonbounds=(150, 240)):
        """
        Requires:
        alltracks --------> one long list of CycloneTrack objects in no particular order (includes all members)
        min_cluster_size -> the minimum number of ensemble members in a cluster (int)
        latbounds --------> latitude bounds for the plotting domain (tuple of ints/floats)
        lonbounds --------> longitude bounds for the plotting domain (tuple of ints/floats)
        """
        import hdbscan
        from collections import Counter
        
        # 1) GET DATA TO CLUSTER THE TRACKS:
        #    we would like to achieve one cluster per storm, where each cluster contains the cyclone
        #    track forecasts from each ensemble member
        #    NOTE: not every cluster will contain every ensemble member!
        
        # Let's cluster using 5 variables: lat0, lon0, time0, dlat, and dlon
        # "0" denotes first timestep; dlat and dlon are computed between timesteps 1 and 2
        tracksdata = np.zeros((len(alltracks), 5))
        tracksdata[:,0] = [track.centers[0].lat for track in alltracks]  # lat0
        tracksdata[:,1] = [track.centers[0].lon for track in alltracks]  # lon0
        tracksdata[:,2] = [track.centers[0].time for track in alltracks] # time0
        tracksdata[:,3] = [track.centers[1].lat-track.centers[0].lat for track in alltracks] # dlat
        tracksdata[:,4] = [track.centers[1].lon-track.centers[0].lon for track in alltracks] # dlon
        
        # 2) RUN THE CLUSTERING ALGORITHM
        #    here we use HBDSCAN, an unsupervised clustering algorithm that accounts for variable-density clusters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        # acquire cluster labels (integers), ranging from 0 to (# of clusters - 1)
        # a label value of -1 means that track was not clustered
        cluster_labels = clusterer.fit_predict(tracksdata)
        assert -1 in cluster_labels # need this for the following loop to function properly
        
        # 3) CREATE A TrackCluster OBJECT FOR EACH CLUSTER
        for label in np.unique(cluster_labels)[::-1]:  # reverse order, so that unsorted (-1) tracks are last
            # Get some labeling information
            if label==-1: # <--- the "no cluster" label
                color = (0,0,0)
                key = 'noclust'
            else:
                color = self.palette[label]
                key = 'cyc{:02d}'.format(label)
                
            # Find all the tracks within this cluster
            trackinds = np.where(np.array(cluster_labels)==label)[0]
            clustertracks = [alltracks[ind] for ind in trackinds]
            
            # If two or more cyclone tracks in this cluster are from the same member, keep the one whose
            # total integrated vorticity most resembles the rest of the cluster
            mems = np.array([t.member() for t in clustertracks])
            # check for duplicate members
            duplicate_members = [item for item, count in Counter(mems).items() if count > 1]
            for dupmem in duplicate_members:
                if label==-1: continue
                # find the tracks that are from this member
                inds = np.where(mems==dupmem)[0]
                # get their integrated vorticity
                allvorts = np.array([np.sum(t.vorts()) for t in clustertracks])
                meanvort = np.mean(allvorts)
                vort_err = np.abs(allvorts[inds] - meanvort) # absolute total vorticity error from the cluster mean
                # only keep the track with the minimum vorticity error
                dropinds = inds[np.where(vort_err>np.min(vort_err))[0]]
                clustertracks = [track for t, track in enumerate(clustertracks) if t not in dropinds]
                mems = np.array([t.member() for t in clustertracks])
                # categorize the dropped tracks as "unclustered"
                cluster_labels[trackinds[dropinds]] = -1
            
            # Store this cluster of tracks
            self[key] = TrackCluster(clustertracks, label, color=color)
         
        # 4) Store the latitude and longitude bounds for the cyclone tracking domain
        self.latbounds = latbounds
        self.lonbounds = lonbounds
            
    # Functions to acquire simple attributes
    def labels(self):
        return list(self.keys())[::-1]
    def clusters(self):
        return list(self.values())[::-1]
    def iteritems(self):
        return [(l,c) for l,c in zip(self.labels(), self.clusters())]
    def show_palette(self):
        sns.palplot(self.palette)
        
    #==== Some plotting functions ==================================================
    def create_domain_map_proj(self, ax, landfill=True, dlat=10, dlon=20):
        """
        Creates a cylindrical Basemap projection object for the track clustering domain.
        
        Requires:
        ax -------> an axis object to draw the map on
        landfill -> if True, landmasses will be filled with a light grey color
        dlat -----> interval (in degrees) for latitude labels
        dlon -----> interval (in degrees) for longitude labels
        
        Returns:
        m --------> the new Basemap object
        """
        from mpl_toolkits.basemap import Basemap

        m = Basemap(ax=ax, llcrnrlon=self.lonbounds[0],llcrnrlat=self.latbounds[0],
                    urcrnrlon=self.lonbounds[1],urcrnrlat=self.latbounds[1],
                    projection='cyl', resolution ='l')
        m.drawcoastlines()
        m.drawcountries()
        if landfill: m.fillcontinents(color='lightgrey')
        m.drawparallels(np.arange(0,90,dlat),labels=[1,0,0,0], dashes=[4,4])
        m.drawmeridians(np.arange(0,360,dlon),labels=[0,0,0,1], dashes=[4,4])
        return m

    def plot_tracks_on_map(self, m, t_ind, show_unclustered=True, only_unclustered=False):
        """
        Plots the clustered CycloneTracks onto a map projection.
        
        Requires:
        m ----------------> a Basemap projection object
        t_ind ------------> index denoting the final time at which to plot tracks
        show_unclustered -> if True, will also plot unclustered tracks in a dim grey
        only_unclustered -> if True, will ONLY plot unclustered tracks
        """
        # Loop through each cluster
        for clabel, cluster in self.iteritems():
            # Loop through each track in the cluster
            for t, track in enumerate(cluster.tracks):
                # Get the lat/lons of this track
                tracklats, tracklons = track.latlons(asarr=True)
                tracklons = tracklons[np.where(track.times(asarr=True)<=t_ind)[0]]
                tracklats = tracklats[np.where(track.times(asarr=True)<=t_ind)[0]]
                # Project onto map coordinates
                x, y = m(tracklons, tracklats)
                # Assign a plot label, but only to one (the first) track, to keep the legend clean
                if t==0: lab = clabel
                else:    lab = None
                # Plot the ensemble clusters in their respective color
                if clabel != 'noclust' and not only_unclustered:
                    m.plot(x, y, zorder=6, color=cluster.color, linewidth=0.6, marker='o', markersize=2, label=lab)
                # Plot the unclustered tracks in a faded black
                elif clabel=='noclust' and (show_unclustered or only_unclustered):
                    m.plot(x, y, zorder=6, color=cluster.color, linewidth=0.6, alpha=0.6, label=lab)
                    
            
            
    