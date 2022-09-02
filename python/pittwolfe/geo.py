"""
    geo
    -----------
    A collection of routines for dealing with geographical coordinates
"""

import numpy as np
import scipy as sp

from .util import cosd, sind

def spherical_distance(p1, p2):
    '''
    Calculate distance on a unit sphere using the Vincenty formula, which is supposed to
    be resistant to rounding errors over all distances.

    Parameters
    ----------
    p1, p2 : array-like containing two floats or ndarrays
        Lat/Lon pairs in degrees. On of p1, p2 may be a pair of ndarrays

    Returns
    -------
    d : float
        distance on the unit sphere

    Notes
    -----
    '''

    phi1 = np.deg2rad(p1[0])
    phi2 = np.deg2rad(p2[0])
    lam1 = np.deg2rad(p1[1])
    lam2 = np.deg2rad(p2[1])
    dlam = lam2-lam1

    d = np.arctan(np.sqrt((np.cos(phi2)*np.sin(dlam))**2
                    + (np.cos(phi1)*np.sin(phi2) - np.sin(phi1)*np.cos(phi2)*np.cos(dlam))**2)
                    / (np.sin(phi1)*np.sin(phi2) + np.cos(phi1)*np.cos(phi2)*np.cos(dlam)))

    if isinstance(d, np.ndarray):
        d[d < 0] += pi
    else:
        if d < 0:
            d += pi

    return d


class geoPath:
    '''
    A class representing a path on the globe in lat/lon coordinates.
    '''
    def __init__(self, lon, lat, rEarth=6370):
        '''
        Create a new geoPath object.

        Parameters
        ----------
        lon : scalar or 1D array-like
            List of longitudes in degrees.
        lat : scalar or 1D array-like
            List of latitudes in degrees.
        rEarth : scalar, optional
            Radius of Earth. Setting this determines the length units of the geoPath

        '''
        lon = np.atleast_1d(np.array(lon))
        lat = np.atleast_1d(np.array(lat))

        self.N = len(lon)

        if self.N != len(lat):
            raise RuntimeError('lon and lat must have the same length')

        self.lon = lon
        self.lat = lat

        self.a = rEarth

        self.calc_arclength()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return geoPath(self.lon[idx], self.lat[idx])

    def __repr__(self):
        return 'length {:d} geoPath'.format(self.N)

    def copy(self):
        return geoPath(self.lon.copy(), self.lat.copy(), self.a)

    def calc_arclength(self):
        '''
        Calculate arclength using the Vincenty formula.
        '''
        if self.N == 1:
            self.s = 0
        else:
            self.ds = np.zeros(self.N-1)
            self.s = np.zeros(self.N)

            for n in range(self.N-1):
                p1 = (self.lat[n+1], self.lon[n+1])
                p2 = (self.lat[n], self.lon[n])
                self.ds[n] = self.a*spherical_distance(p1, p2)

            self.s[1:] = np.cumsum(self.ds)

    def distance(self, p):
        '''
        Distance from (lon, lat) point p to points on path.
        '''
        if isinstance(p, geoPath):
            lon = p.lon
            lat = p.lat
        else:
            lon = p[0]
            lat = p[1]

        return self.a*spherical_distance((lat, lon), (self.lat, self.lon))

    def reparameterize(self, s):
        '''
        Return geoPath with new arclength parameter

        Parameters
        ----------
        s : 1D np.array
            New arclength parameter. Must be within the range of the original arclength.

        Returns
        -------
          : geoPath
            The path with the points located at the nodes of the new arclength parameter.
        '''
        fxi = sp.interpolate.interp1d(self.s, self.lon, kind='cubic')
        fyi = sp.interpolate.interp1d(self.s, self.lat, kind='cubic')

        return geoPath(fxi(s), fyi(s), rEarth=self.a)

    def subdivide_segment(self, idx, divisions=1):
        '''
        Subdivide a geoPath segment.

        Parameters
        ----------
        idx : int
            Location of the beginning of the segment to subdivide.
        divisions : int, optional
            Number of times to subdivide. The segment is divided into divisions + 1 equal
            length segments.

        Returns
        -------
          : geoPath
            Original path with the segment subdivided.
        '''

        s0 = self.s[idx]
        s1 = self.s[idx+1]

        s_segment = np.linspace(s0, s1, divisions+2)[1:-1]

        lon_segment = (self.lon[idx]*(s1 - s_segment) + self.lon[idx+1]*(s_segment - s0))/(s1 - s0)
        lat_segment = (self.lat[idx]*(s1 - s_segment) + self.lat[idx+1]*(s_segment - s0))/(s1 - s0)

        return geoPath(np.hstack((self.lon[:idx+1], lon_segment, self.lon[idx+1:])),
                       np.hstack((self.lat[:idx+1], lat_segment, self.lat[idx+1:])),
                       rEarth=self.a)


    def midpoints(self):
        '''
        Arclength and values at segment midpoints.

        Returns
        -------
         : geoPath
            A path build out of the midpoints of the original path.
        '''

        ret = geoPath((self.lon[1:] + self.lon[:-1])/2,
                      (self.lat[1:] + self.lat[:-1])/2,
                      rEarth = self.a)
        ret.s = (self.s[1:] + self.s[:-1])/2
        ret.ds = np.diff(ret.s)

        return ret

    def unit_tangent(self, return_loc=True):
        '''
        Calculate unit tangent using the tangent plane approximation.

        Parameters
        ----------
        return_loc : bool, optional
            If also return location of the vector.

        Returns
        -------
        s : 1D np.array, optional
            Location of the vector. Only returned if return_loc == True
        t : 2D np.array
            The tangent vector

        Notes
        -----
        The tangent vector is "located" at the midpoints between lat/lon pairs.
        '''
        dx = self.a*cosd((self.lat[1:]+self.lat[:-1])/2)*np.deg2rad(np.diff(self.lon))
        dy = self.a*np.deg2rad(np.diff(self.lat))
        t = np.vstack((dx, dy))
        t /= np.sqrt(dx**2 + dy**2)
        t = t.T

        if return_loc:
            return self.midpoints(), t
        else:
            return t

    def unit_normal(self, return_loc=True):
        '''
        Calculate unit normal using the tangent plane approximation.

        Parameters
        ----------
        return_loc : bool, optional
            If also return location of the vector.

        Returns
        -------
        s : 1D np.array, optional
            Location of the vector. Only returned if return_loc == True
        n : 2D np.array
            The normal vector

        Notes
        -----
        The normal vector is "located" at the midpoints between lat/lon pairs and is is
        oriented to the left of the tangent vector looking toward greater arc length.
        '''
        dx = self.a*cosd((self.lat[1:]+self.lat[:-1])/2)*np.deg2rad(np.diff(self.lon))
        dy = self.a*np.deg2rad(np.diff(self.lat))
        n = np.vstack((-dy, dx))
        n /= np.sqrt(dx**2 + dy**2)
        n = n.T

        if return_loc:
            return self.midpoints(), n
        else:
            return n

    def to_vec(self):
        return np.vstack((self.lon, self.lat)).T



