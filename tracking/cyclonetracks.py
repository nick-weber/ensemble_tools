#!/usr/bin/env python
"""
Module containing classes cyclone centers (at one time) and cyclone tracks
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
        See Flaounas et al. 2017
        
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
    track across numerous times.
    """
    def __init__(self, centers):
        """
        Variables/attributes described in comments below
        """
        # The following is a list of length N
        # N = number of points in this cyclone track
        self.centers = centers # list of CycloneCenter objects
        
        # Other attributes
        self.finished = False # this track cannot be extended anymore
        
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
        
    # Function to get a field along the cyclone track
    def field_along_track(self, field, value='max'):
        """Returns a timeseries of the maximum, minimum, average, or center-value of an 
           atmospheric variable along this cyclone track"""
        if value=='max':
            return np.array([c.max_field(field) for c in self.centers])
        elif value=='min':
            return np.array([c.min_field(field) for c in self.centers])
        elif value in ['mean', 'avg']:
            return np.array([c.mean_field(field) for c in self.centers])
        elif value=='atcenter':
            return np.array([c.field_at_center(field) for c in self.centers])
        else:
            raise ValueError('invalid value "{}"'.format(value))
            
################################################################################################################
        
class TrackCluster:
    """
    Primarily a list of CycloneTrack objects, this class stores and dislays data from an ensemble
    of cyclone tracks (i.e., an ensemble forecast of one storm).
    """
    def __init__(self, tracks, cluster_id, full_id=None, color='red'):
        """
        Requires:
        tracks -----> a list of CycloneTrack objects, representing an ensemble of tracks for one cyclone
        cluster_id -> a string ID for this cluster
        color ------> the color to use when plotting this cluster of cyclone tracks (string)
        """
        from collections import Counter
        
        # If two or more cyclone tracks in this cluster are from the same member, keep the one whose
        # total integrated vorticity most resembles the rest of the cluster
        mems = np.array([t.member() for t in tracks])
        duplicate_members = [item for item, count in Counter(mems).items() if count > 1]
        for dupmem in duplicate_members:
            if cluster_id==-1: continue
            # find the tracks that are from this member
            inds = np.where(mems==dupmem)[0]
            # get their integrated vorticity
            allvorts = np.array([np.sum(t.vorts()) for t in tracks])
            meanvort = np.mean(allvorts)
            vort_err = np.abs(allvorts[inds] - meanvort) # absolute total vorticity error from the cluster mean
            # drop the tracks whose duration errors are larger than the min duration error
            dropinds = inds[np.where(vort_err>np.min(vort_err))[0]]
            tracks = [track for t, track in enumerate(tracks) if t not in dropinds]
            mems = np.array([t.member() for t in tracks])
        # Make sure we got rid of all the duplicates
        if cluster_id != -1: assert len(mems)==len(set(mems))
            
        # Now let's just sort the remaining tracks by member before storing them
        tracks.sort(key=lambda x: x.member())
        self.tracks = tracks
        # And let's give this cluster of tracks an ID and a plotting color
        self.cluster_id = cluster_id
        if full_id is None: full_id = 'cyc{:02d}'.format(cluster_id)
        self.full_id = full_id
        self.color = color
        
    # Functions to acquire simple attributes
    def ntracks(self):
        return len(self.tracks)
    def nmems(self):
        return len(self.tracks)
    
    #==== Some plotting functions ==================================================
    def plot_plumes(self, field, value='max', ax=None, title=True, plotmean=True, 
                    factor=1., unit=None, **plot_args):
        from matplotlib.ticker import AutoMinorLocator

        # Create the figure, axis if we need to
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.despine()
            ax.yaxis.grid()
        # If we are going to compute the ensemble mean, we need to store all the members
        if plotmean:
            # How many times are in the forecast? How many members are in this cluster?
            nt = self.tracks[0].centers[0].ens.ntimes()
            nm = self.nmems()
            ally = np.empty((nm, nt))
            ally[:,:] = np.nan
        # How many hours are between each timestep?
        dt = self.tracks[0].centers[0].ens.dt()
        # Plot each track
        min_x = 100; max_x = 0
        for t, track in enumerate(self.tracks):
            x = track.times(asarr=True)
            if min(x) < min_x: min_x = min(x)
            if max(x) > max_x: max_x = max(x)
            y = track.field_along_track(field, value=value)
            ax.plot(x*dt, y*factor, color=self.color, alpha=0.6, **plot_args)
            if plotmean: ally[t, x] = y  # store values to compute mean
        # Compute/plot the ensemble mean
        if plotmean:
            # We only want to average at times where >50% of the members have data
            ybar = np.array([np.nanmean(ally[:,t]) if len(np.where(~np.isnan(ally[:,t]))[0])>0.5*nm \
                             else np.nan for t in range(nt)])
            darkercolor = tuple([rgb*0.8 for rgb in self.color])
            ax.plot(np.arange(nt)*dt, ybar*factor, color=darkercolor, linewidth=4, **plot_args)
        ax.set_xticks(np.arange(min_x-min_x%24, max_x*dt+24, 24))
        ax.set_xticklabels(['f{:02d}'.format(int(lead)) for lead in ax.get_xticks()])
        ax.xaxis.set_minor_locator(AutoMinorLocator(int(24/dt)))
        # Create a title
        if title:
            model = self.tracks[0].centers[0].ens.model()
            idate = self.tracks[0].centers[0].ens.idate()
            if unit is None:
                titl = '{} {} {} plumes -- init: {:%Y-%m-%d %H:00}'.format(model, value, field, idate)
            else:
                titl = '{} {} {} [{}] plumes -- init: {:%Y-%m-%d %H:00}'.format(model, value, field, unit, idate)
            ax.text(0, 1.02, titl, transform=ax.transAxes, ha='left', va='bottom')
            ax.text(1, 1.02, self.full_id, transform=ax.transAxes, ha='right', va='bottom', color=self.color)
        try: return fig, ax
        except: return ax
        
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
                                'lavender', 'bluish', 'tan', 'aqua', 'mauve', 'olive'])
    
    def __init__(self, alltracks, min_cluster_size=4):
        """
        Requires:
        alltracks --------> one long list of CycloneTrack objects in no particular order (includes all members)
        min_cluster_size -> the minimum number of ensemble members in a cluster
        """
        import hdbscan
        
        # 1) GET DATA TO CLUSTER THE TRACKS:
        #    we could like to achieve one cluster per storm, where each cluster contains the cyclone
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
        #    here we use HBDSCAN, an unsupervised clustering algorithm that accounts for variable-density
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        # acquire cluster labels (integers), ranging from 0 to (# of clusters - 1)
        # a label value of -1 means that track was not clustered
        cluster_labels = clusterer.fit_predict(tracksdata)
        
        # 3) CREATE A TrackCluster OBJECT FOR EACH CLUSTER
        for label in np.unique(cluster_labels):
            # Get some labeling information
            if label==-1: # <--- the "no cluster" label
                color = 'k'
                key = 'noclust'
            else:
                color = self.palette[label]
                key = 'cyc{:02d}'.format(label)
                
            # Find all the tracks within this cluster
            trackinds = np.where(np.array(cluster_labels)==label)[0]
            clustertracks = [alltracks[ind] for ind in trackinds]
            
            # Store this cluster of tracks
            self[key] = TrackCluster(clustertracks, label, color=color)
            
    # Functions to acquire simple attributes
    def labels(self):
        return list(self.keys())
    def clusters(self):
        return list(self.values())
    def iteritems(self):
        return [(l,c) for l,c in zip(self.labels(), self.clusters())]
    def show_palette(self):
        sns.palplot(self.palette)
            
    def plot_tracks_on_map(self, m, t_ind, show_unclustered=True, only_unclustered=False):
        # Loop through each cluster
        for clabel, cluster in self.iteritems():
            # Loop through each track in the cluster
            for t, track in enumerate(cluster.tracks):
                # Get the lat/lons of this track
                tracklons = track.lons(asarr=True)
                tracklats = track.lats(asarr=True)
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
                    
            
            
    