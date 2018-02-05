#!/usr/bin/env python
"""
Module containing functions used to locate and track cyclones (vorticity maxima) in 
space and time.
"""
import numpy as np

def driver(ens, filterscale=4, latbounds=(20, 70), lonbounds=(150, 240), verbose=False):
    """
    Uses the functions below, given an Ensemble object (and some other information), to identify, track,
    cluster, and store cyclones in an ensemble forecast system.
    
    Requires:
    ens ---------> the Ensemble object whose data is used to track these cyclones
    filterscale -> the number of points in each direction to be used in the spatial filter
                   e.g., 4 --> a 9x9 spatial filter
    latbounds ---> latitude bounds for the cyclone-tracking domain (tuple of ints/floats)
    lonbounds ---> longitude bounds for the cyclone-tracking domain (tuple of ints/floats)
    
    Returns:
    clusters ----> an EnsembleTrackClusters object containing all the cyclone tracks 
                   (and their cluster IDs) within the model spatiotemporal domain
    """
    from .cyclonetracks import EnsembleTrackClusters
    from time import time
    
    # 1) Calculate the spatially filtered 850 hPa vorticity
    if 'filt_relvor_850hPa' not in ens.variables.keys():
        #if verbose: print('Calculating 850hPa vorticity and applying spatial filter... ', end='')
        #start = time()
        #ens.calculate_relvor(lev=850)
        #ens.spatial_filter('relvor_850hPa', N=filterscale)
        #if verbose: print('{:.2f} min'.format((time()-start)/60.))
        print('Calculating 850hPa vorticity... ', end='')
        start = time()
        ens.calculate_relvor(lev=850)
        print('{:.2f} min'.format((time()-start)/60.))
            
        print('Applying spatial filter... ', end='')
        start = time()
        ens.spatial_filter('relvor_850hPa', N=filterscale)
        print('{:.2f} min'.format((time()-start)/60.))
            
    # 2) Find all possible cyclone tracks for each ensemble member
    if verbose: print('Finding cyclone centers and tracks for each ensemble member... ', end='')
    start = time()
    tracks = {} # this will be a list of lists of tracks
    for mem in range(ens.dims['ens']):
        # 2.A) Make a list of this member's cyclones at ALL times
        cyclones = []
        for t_ind in range(ens.dims['time']):
            cyclones_1time = locate_vort_maxima_1time(ens, 'filt_relvor_850hPa', t_ind, mem, 
                                                      latbounds=latbounds, lonbounds=lonbounds)
            cyclones.append(cyclones_1time)
        # 2.B) Use recursive function to find all valid cyclone tracks for this member
        tracks[mem] = track_cyclones(cyclones)
    if verbose: print('{:.2f} min'.format((time()-start)/60.))
    
    # 3) Cluster the tracks by spatiotemporal proximity (w/ machine learning)
    if verbose: print('Clustering cyclone tracks using HBDSCAN... ', end='')
    start = time()
    alltracks = [tr for sublist in tracks.values() for tr in sublist] # flatten the 2D track list
    clusters = EnsembleTrackClusters(alltracks, latbounds=latbounds, lonbounds=lonbounds)
    if verbose: print('{:.2f} min'.format((time()-start)/60.))
    
    # DONE
    return clusters

##############################################################################################################

def locate_vort_maxima_1time(ens, vortvar, t_ind, mem, latbounds=(-70, 70), lonbounds=(0, 360), 
                             vorthresh=3e-5, minrad=3, maxrad=20):
    """
    Locates all potential cyclone centers, or vorticity maxima, in a 2D vorticity field within a desired domain.
    Also computes each center's effective radius.
    
    methodology from: Flaounas et al. 2014
    
    Requires:
    ens --------> the Ensemble object whose data is used to track these cyclones
    vortvar ----> name of the vorticity variable used to locate cyclones (string; e.g., 'filt_relvor_850hPa')
    t_ind ------> time index
    mem --------> ensemble member index (integer from 0 to (#members - 1) )
    latbounds --> latitude bounds for the cyclone-tracking domain (tuple of ints/floats)
    lonbounds --> longitude bounds for the cyclone-tracking domain (tuple of ints/floats)
    vortthresh -> vorticity threshold used in cyclone detection
    minrad -----> minimum radius (in # grid cells) for a cyclone center
    maxrad -----> maximum radius (in # grid cells) for a cyclone center
    
    Returns:
    cyclones ---> a list of CycloneCenter objects for this ensemble member and model time
    """
    import itertools
    from .cyclonetracks import CycloneCenter
    
    # Get the latitude, longitude, and vorticity arrays for this member/time
    lats, lons = ens.latlons()
    vort = ens[vortvar].isel(time=t_ind, ens=mem).values
    assert np.shape(vort) == (len(lats), len(lons))
    
    # "define the local maximum as the maximum value of the central grid point 
    #  among its eight surrounding grid points" - Flaounas et al. 2014
    centers = []
    for j, lat in enumerate(lats):
        if not (lat>=latbounds[0] and lat<=latbounds[1]): continue
        for i, lon in enumerate(lons):
            if not (lon>=lonbounds[0] and lon<=lonbounds[1]): continue
            # If this point's vorticity is not above the threshold value, move on
            if vort[j,i] < vorthresh: continue
            # Get the eight neighbors of this point (actually nine including self)
            neigh_inds = [(j+jj, i+ii) for jj,ii in itertools.product([-1,0,1],[-1,0,1])]
            neigh_inds = [(jj,ii-len(lons)) if ii>=len(lons) else (jj,ii) for jj,ii in neigh_inds]
            neighbors = np.array([vort[jj, ii] for jj,ii in neigh_inds])
            # Make sure the neighbors are all above the threshold
            if not (neighbors>=vorthresh).all(): continue
            # If this point's vorticity surpasses all it's neighbors', then
            # it is a local maximum
            if (vort[j,i] >= neighbors).all():
                centers.append((j,i))
                
    # establish a circular radius around each maximum that grows until:
    # 1) the average vorticity within the disk is less than the threshold value
    # 2) the radius reaches a pre-defined maximum length (default: 20 pts / 10 deg) or
    # 3) a vorticity value greater than that of the cyclonic center is found within the area
    radii = np.zeros(len(centers))
    del_inds = []
    for c, center in enumerate(centers):
        j, i = center
        radius = minrad # grid points
        keepgrowing = True
        while keepgrowing:
            # Make a list of all the (j,i) points within this center's radius
            ptsinrad = itertools.product(range(-radius, radius+1),range(-radius, radius+1))
            ptsinrad = [(jj,ii) for jj,ii in ptsinrad if (jj**2+ii**2)<=radius**2]
            ptsinrad = [(j+jj,i+ii-len(lons)) if i+ii>=len(lons) else (j+jj,i+ii) for jj,ii in ptsinrad]
            # Get the vorticity values at all these points
            vorinrad = np.array([vort[jj, ii] for jj,ii in ptsinrad])
            # don't want storms that are too close (within 2 pts) to the poles
            if np.array([(jj<=2 or jj>=len(lats)-2) for jj,ii in ptsinrad]).any():
                keepgrowing = False
            # check: CONDITION 1
            elif np.nanmean(vorinrad) <= vorthresh:
                keepgrowing = False
            # check: CONDITION 2 
            elif radius >= maxrad:
                keepgrowing = False
            # check: CONDITION 3
            elif (vorinrad > vort[j,i]).any():
                keepgrowing = False
            # Otherwise, keep on growing!
            else: 
                radius += 1
        radii[c] = radius
        # We will delete any weaker cyclones that are found within this cyclone's radius
        for oc, othercenter in enumerate(centers):
            if oc==c: continue
            if othercenter in ptsinrad and vort[othercenter]<vort[center]:
                del_inds.append(oc)
    # Delete any weaker cyclones that are found within stronger cyclones' radii
    del_inds = list(set(del_inds))
    for index in sorted(del_inds, reverse=True):
        del centers[index]
    radii = np.delete(radii, del_inds)
        
    # Remove any centers that have too small a radius
    keepinds = np.where(radii>minrad)[0]
    centers = [centers[ki] for ki in keepinds]
    radii = list(radii[keepinds])
    
    # Create a list of CycloneCenter objects for this time
    cyclones = [CycloneCenter(ens, c, t_ind, mem+1, vort[c[0],c[1]], r, lats[c[0]], lons[c[1]]) for \
                c, r in zip(centers, radii)]
    return cyclones

##############################################################################################################

def track_cyclones(cyclones_alltimes, minlength=4):
    """
    Given a list of CycloneCenters (for one ensemble member) at many times, this function finds all possible
    cyclone tracks using a separate recursive function, then returns a list of only the tracks that
    optimize a vorticity/distance cost function.
    
    Requires:
    cyclones_alltimes --> a list of lists of CycloneCenter objects; parent list is of len(time), 
                          interior list of of len(# of cyclones at that time)
    minlength ----------> minimum required length (in timesteps) of a cyclone track
    
    Returns:
    besttracks ----------> a list of CycloneTrack objects for this ensemble member
    """
    from .cyclonetracks import CycloneTrack
    from itertools import compress
    
    # Make a giant (i.e., flattened) list of all the CycloneCenters and corresponding vorticity maxima
    allcyclones = [cyc for sublist in cyclones_alltimes for cyc in sublist]
    maxvorts = [cyc.vort for cyc in allcyclones]
    
    # Now create a list of CycloneCenters *sorted* from highest maxvort to lowest
    cyclones = [cyc for v,cyc in sorted(zip(maxvorts, allcyclones), reverse=True)]
    
    # Loop through our sorted potential cyclone centers and find the best track that goes through each
    used_cyclones = []
    besttracks = []
    for tc, thiscyclone in enumerate(cyclones):
        if thiscyclone in used_cyclones: continue
        used_cyclones.append(thiscyclone)
        
        # FIRST: Select all possible tracks that go through this cyclone center
        tracks = [CycloneTrack(centers=[thiscyclone])]
        tracks = recursive_pathfinder(tracks, cyclones, used_cyclones)
        # We will only consider tracks that span at least [minlength] time steps
        longenough = [track.ntimes() >= minlength for track in tracks]
        tracks = list(compress(tracks, longenough))
        if len(tracks)==0: continue
        
        # NEXT: Choose the track that minimizes the cost function (Flaounas et al. 2014):
        #  C = sum( (dist. btw. adjacent centers)*|vort. diff. btw. adjacent centers| ) / sum(dist. btw. adjacent centers)
        costs = np.zeros(len(tracks))
        for t, track in enumerate(tracks):
            dists = np.array([track.centers[i].distance_from(track.centers[i+1]) for i in range(track.ntimes()-1)])
            dvors = np.abs( np.array(track.vorts()) - np.roll(track.vorts(), -1))[:-1]
            costs[t] = np.sum(dists*dvors) / np.sum(dists)
        besttrack = tracks[np.argmin(costs)]
        
        # FINALLY: Remove all the chosen track points from the list of possible cyclone centers
        for cyclone in besttrack.centers:
            used_cyclones.append(cyclone)
        besttracks.append(besttrack)
    return besttracks
        
        
##############################################################################################################

def recursive_pathfinder(tracks, cyclones, used_cyclones):
    """
    Recursively expands the list of cyclone tracks given a list of candidate cyclones within a space/time
    domain. Candidate cyclones are added to a track if they meet the following criteria (discerned by
    the meets_conditions() method in the CycloneCenter class):
    Criteria:  (1) the location of the next center must be within 5-deg-lat, 10-deg-lon
               (2) the maximum vorticity between the two consecutive steps must not differ by >50%
               (3) if the sum of two consecutive displacements is more than 3-deg
                   long, then the angle between these displacements must be greater than 90-deg
    
    Requires:
    tracks --------> list of CycloneTrack objects that is recursively expanded
    cyclones ------> list of candidate CycloneCenter objects to be added to the tracks
    used_cyclones -> list of CycloneCenter objects that have already been used
    
    Returns:
    tracks --------> list of CycloneTrack objects (all possible tracks that meet the criteria)
    """
    from .cyclonetracks import CycloneTrack
    from itertools import compress
    
    allnewtracks = []
    
    # Let's see if we can expand ANY of the given tracks:
    for track in tracks:
        if track.finished: continue
        thisnewtracks = [] # a list of new tracks that branch off this one
        
        # 1) Loop through and see if we can PREPEND or APPEND any cyclones to this track
        prev_cents = []
        next_cents = []
        for othercyclone in cyclones:
            if othercyclone in used_cyclones: continue
                
            # If this center meets all the criteria and PREcedes, add it to the list of "previous centers"
            if othercyclone.meets_conditions(track, before=True):
                prev_cents.append(othercyclone)
            # If this center meets all the criteria and PROcedes, add it to the list of "next centers"
            elif othercyclone.meets_conditions(track, before=False):
                next_cents.append(othercyclone)
          
        # If there are no new centers to append/prepend, then this track is finished
        if len(prev_cents)==0 and len(next_cents)==0:
            track.finished = True
            continue
            
        # 2) PREPEND and APPEND all possible cyclone centers
        # PREceding centers
        for pc, prevcent in enumerate(prev_cents):
            if pc > 0: # Create new track!
                thisnewtracks.append(CycloneTrack(centers=[prevcent] + track.centers[1:]))
            else: # PREpend to this track
                track.centers = [prevcent] + track.centers
        # PROceding centers        
        for nc, nextcent in enumerate(next_cents):
            if nc > 0: # Create new tracks!
                thisnewtracks += [CycloneTrack(centers=t.centers + [nextcent]) for t in thisnewtracks \
                                  if nextcent.time > t.centers[-1].time]
                thisnewtracks.append(CycloneTrack(centers=track.centers[:-1] + [nextcent]))
            else: # APpend to this track
                track.centers = track.centers + [nextcent]
                for newtrack in thisnewtracks:
                    newtrack.centers = newtrack.centers + [nextcent]
        
        # Add the list of this track's new branching tracks to the list of ALL tracks
        allnewtracks += thisnewtracks
        
    
    # Add all the newly branched tracks to the list of all plausible tracks
    tracks += allnewtracks
    
    # Check Criteria #3 again! If a track was originally length-1 and increased to length-3 during the
    # append/prepend stage, it could have gotten through that check in the meets_conditions() function
    keepthistrack = [True for track in tracks]
    for t, track in enumerate(tracks):
        if track.ntimes()==3:
            cyc1 = track.centers[0]; cyc2 = track.centers[1]; cyc3 = track.centers[2]
            totaldistance = cyc1.distance_from(cyc2) + cyc2.distance_from(cyc3)
            angle = angle_btw_pts([cyc1.lat,cyc1.lon], [cyc2.lat,cyc2.lon], [cyc3.lat,cyc3.lon])
            if totaldistance > 3 and angle <= 90.:
                keepthistrack[t] = False
    tracks = list(compress(tracks, keepthistrack))
    
    # Recursively call the function if any of the tracks can still be expanded
    allfinished = all([t.finished for t in tracks])
    if not allfinished:
        tracks = recursive_pathfinder(tracks, cyclones, used_cyclones)
    return tracks

##############################################################################################################

def angle_btw_pts(a,b,c):
    """Computes the angle A->B->C in degrees"""
    # Vectors from the vertex B
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    # Handle wrapping around lon=0/360
    if ba[1]>180:    ba[1] -= 360
    elif ba[1]<-180: ba[1] += 360
    if bc[1]>180:    bc[1] -= 360
    elif bc[1]<-180: bc[1] += 360
    # Compute the angle and convert to degrees
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)