"""
    plt
    -----------
    A collection of random plotting routines
"""
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely.geometry as sgeom
import os
import glob

package_directory = os.path.dirname(os.path.abspath(__file__))

##########################################################################################
# Styles and colors
##########################################################################################
linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (5, 2, 1, 2))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

def visualize_colors(colors):
    n = len(colors)
    ncols = 4
    nrows = n // ncols + 1

    fig, ax = plt.subplots(figsize=(8, 5))

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols

    for i, color in enumerate(colors):
        col = i % ncols
        row = i // ncols
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        ax.text(xi_text, y, '{:d}'.format(i), fontsize=12,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y + h * 0.1, xi_line, xf_line,
                  color=color, linewidth=(h * 0.6))

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)


# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


# These are the "Color blind 10" colors as RGB.
color_blind10 = [(0,107,164), (255,128,14), (171,171,171), (89,89,89),
             (95,158,209), (200,82,0), (137,137,137), (162,200,236),
             (255,188,121), (207,207,207)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(color_blind10)):
    r, g, b = color_blind10[i]
    color_blind10[i] = (r / 255., g / 255., b / 255.)


# These are the "Gray 5" colors as RGB.
gray5 = [(96,99,106), (165,172,175), (65,68,81), (143,135,130),
             (207,207,207)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(gray5)):
    r, g, b = gray5[i]
    gray5[i] = (r / 255., g / 255., b / 255.)

# NCL colormaps are available from http://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml#Ncview
def get_ncl_cmap(name, reverse=False):
    from matplotlib.colors import ListedColormap

    if name[-4:] != '.rgb':
        name = name + '.rgb'

    colors = np.loadtxt(
        os.path.join(package_directory, 'ncl_colormaps', name),
        skiprows=2)

    if colors.max() > 1:
        colors /= 256

    if reverse:
        colors = colors[::-1,:]
    return ListedColormap(colors)

def get_ncl_cmap_names():
    cmaps = glob.glob(os.path.join(package_directory, 'ncl_colormaps/*.rgb'))

    return [os.path.splitext(os.path.basename(cmap))[0] for cmap in cmaps]

def register_ncl_cmaps():
    from matplotlib.cm import register_cmap

    cmaps = glob.glob(os.path.join(package_directory, 'ncl_colormaps/*.rgb'))

    for cmap in cmaps:
        name = os.path.splitext(os.path.basename(cmap))[0]
        register_cmap('ncl.' + name, cmap=get_ncl_cmap(cmap))
        register_cmap('ncl.' + name + '_r', cmap=get_ncl_cmap(cmap, reverse=True))


##########################################################################################
# Maps
##########################################################################################
class LongitudeFormatter(object):
    def __init__(self,
                 direction_labels=True,
                 zero_direction_label=False,
                 dateline_direction_label=False,
                 degree_symbol=u'\u00B0',
                 number_format='g'):

        self.direction_labels = direction_labels
        self.zero_direction_label = zero_direction_label
        self.dateline_direction_label = dateline_direction_label
        self.degree_symbol = degree_symbol
        self.number_format = number_format

    def format(self, lons):
        labels = []

        try:
            N = len(lons)
        except TypeError:
            lons = (lons, )

        for lon in lons:
            if self.direction_labels:
                if lon == 0 and self.zero_direction_label:
                    direction_label = 'E'
                elif lon == 180 and self.dateline_direction_label:
                    direction_label = 'E'
                elif lon == -180 and self.dateline_direction_label:
                    direction_label = 'W'
                elif lon > 0:
                    direction_label = 'E'
                elif lon < 0:
                    direction_label = 'W'
                else:
                    direction_label = ''

                lon = np.abs(lon)
            else:
                direction_label = ''

            label_format = '{:' + self.number_format + '}' + self.degree_symbol + direction_label

            labels.append(label_format.format(lon))

        if len(lons) == 1:
            return labels[0]
        else:
            return labels


class LatitudeFormatter(object):
    def __init__(self,
                 direction_labels=True,
                 zero_direction_label=False,
                 degree_symbol=u'\u00B0',
                 number_format='g'):

        self.direction_labels = direction_labels
        self.zero_direction_label = zero_direction_label
        self.degree_symbol = degree_symbol
        self.number_format = number_format

    def format(self, lats):
        labels = []

        try:
            N = len(lats)
        except TypeError:
            lats = (lats, )

        for lat in lats:
            if self.direction_labels:
                if lat == 0 and self.zero_direction_label:
                    direction_label = 'N'
                elif lat > 0:
                    direction_label = 'N'
                elif lat < 0:
                    direction_label = 'S'
                else:
                    direction_label = ''

                lat = np.abs(lat)
            else:
                direction_label = ''

            label_format = '{:' + self.number_format + '}' + self.degree_symbol + direction_label

            labels.append(label_format.format(lat))

        if len(lats) == 1:
            return labels[0]
        else:
            return labels


def label_map(ax, lons, lats,
              xaxis_lat=None,
              yaxis_lon=None,
              lon_formatter=None,
              lat_formatter=None):

    if lon_formatter is None:
        lon_formatter = LongitudeFormatter(direction_labels=False)

    if lat_formatter is None:
        lat_formatter = LatitudeFormatter()

    extent = ax.get_extent(crs=ccrs.PlateCarree())
    if xaxis_lat is None:
        xaxis_lat = extent[2]
    if yaxis_lon is None:
        yaxis_lon = extent[0]

    # label bottom and left side
    lon_proj = ax.projection.transform_points(ccrs.PlateCarree(),
        lons, np.repeat(xaxis_lat, len(lons)))[:,0]
    lat_proj = ax.projection.transform_points(ccrs.PlateCarree(),
        np.repeat(yaxis_lon, len(lats)), lats)[:,1]

    ax.set_xticks(lon_proj, crs=ax.projection)
    ax.set_xticklabels(lon_formatter.format(lons))

    ax.set_yticks(lat_proj, crs=ax.projection)
    ax.set_yticklabels(lat_formatter.format(lats))



### Routines for labeling Lambert conformal maps (also works on Albers equal area maps)
### From https://nbviewer.jupyter.org/gist/ajdawson/dd536f786741e987ae4e
def find_side(ls, side):
    """
    Given a shapely LineString which is assumed to be rectangular, return the
    line corresponding to a given side of the rectangle.

    """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])


def lambert_xticks(ax, ticks):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])


def lambert_yticks(ax, ticks):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """Get the tick locations and labels for an axis of a Lambert Conformal projection."""

    from copy import copy

    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels

##########################################################################################
# Shading
##########################################################################################
def calc_unit_normal_vector(fld):
    '''
    Calculate the unit normal vector of a surface in 2D.

    Parameters
    ----------
    fld : 2D ndarray
        Height field of surface.

    Returns
    -------
    nx, ny, nz : 3 2D ndarrays
        Components of the unit normal.

    Notes
    -----
    The horizontal coordinates are assumed to be uniformly spaced.
    '''

    fldx = ma.zeros(fld.shape)
    fldx[:,1:] = np.diff(fld, axis=-1)
    fldx[:,:-1] = (fldx[:,:-1] + fldx[:,1:])/2

    fldy = ma.zeros(fld.shape)
    fldy[1:,:] = np.diff(fld, axis=0)
    fldy[:-1,:] = (fldy[:-1,:] + fldy[1:,:])/2

    norm = 1 + np.sqrt(fldx**2 + fldy**2)

    nx = -fldx/norm
    ny = -fldy/norm
    nz = 1/norm

    return nx, ny, nz


def hillshade(φ, θ, fld, vertical_exageration=1):
    '''
    Generate shading for a surface lit from a given angle.

    Parameters
    ----------
    φ : scalar number
        Azimuth of light source.
    θ : scalar number
        Zenith angle of light source.
    fld : 2D ndarray
        Height field for surface to be shaded.
    vertical_exageration : positive number, optional
        Degree of vertical exageration; >1 increases apparent height, <1 decreases apparent height.
        Default: 1

    Returns
    -------
    2D ndarray
        Shading intensity (between 0 and 1).

    Notes
    -----
    '''
    from .util import cosd, sind
    nx, ny, nz = calc_unit_normal_vector(vertical_exageration*fld)
    return np.maximum(nx*cosd(φ)*cosd(θ) + ny*sind(φ)*cosd(θ) + nz*sind(θ), 0)


def blend_shade(rgb, intensity, min_brightness=0, max_brightness=1,
                min_intensity=0, mid_intensity=0.5, max_intensity=1):
    '''
    Blend a shading field with an RGB image.

    Parameters
    ----------
    rbg : 2D rgb array
        Array of rgb values. Alpha channel ignored if present.
    intensity : 2D ndarray
        Intensity field for shading. Values should be between 0 and 1
    min_brightness : number between 0 and 1, optional
        Minimum brightness of final image. Default: 0
    max_brightness : number between 0 and 1, optional
        Maximum brightness of final image. Default: 1
    min_intensity : number between 0 and 1, optional
        Value of intensity to map to min_brightness. Default: 0
    mid_intensity : number between 0 and 1, optional
        Value of intensity to map to middle brightness. Default: 0.5
    max_intensity : number between 0 and 1, optional
        Value of intensity to map to max_brightness. Default: 0

    Returns
    -------
    2D rgb field
        Shaded rgb field.

    Notes
    -----
    '''
    intensity = intensity[..., np.newaxis]
    rgb = rgb[:,:,:3] # chop out alpha channel

    return np.where(intensity <= mid_intensity,
                    ((mid_intensity - intensity)*min_brightness + (intensity - min_intensity)*rgb)/(mid_intensity-min_intensity),
                    ((intensity - mid_intensity)*max_brightness + (max_intensity - intensity)*rgb)/(max_intensity-mid_intensity)
                   )
##########################################################################################
# Labels
##########################################################################################
def sublabel(ax, labels='abcdefghijklmnopqrstuvxyz', xoffset=0, yoffset=0, format='{:s})'):
    '''
    Add sublabels to axes.

    Parameters
    ----------
    ax : array_like
        Axes to label.
    labels : str, optional
        Labels. Default: 'abcdefghijklmnopqrstuvxyz'
    xoffset : float, optional
        Default: 0
    yoffset : float, optional
        Default: 0
    format : str, optional
        Format of label. Default: '{:s})'

    Returns
    -------
    anno : list
        list of annotation objects.

    Notes
    -----
    '''

    anno = []
    for a, label in zip(ax.flatten(), labels):
        anno.append(a.annotate(format.format(label), (-0.2 + xoffset, 0.95 + yoffset),
            xycoords='axes fraction'))

    return anno

def smooth_axis_boundary(xlim, ylim, interp_fac=40):
    '''
    Produces a smooth path to use as the boundary of a logically rectangular axis.

    Parameters
    ----------
    xlim : 2 element list
        The limits of the x axis.
    ylim : 2 element list
        The limits of the y axis.
    interp_fac : int, optional
        Number of times to subdivide each segment.

    Returns
    -------
    path : matplotlib.path.Path
        The smoothed boundary path.
    '''
    import matplotlib.path as mpath

    vertices = [
        [xlim[0], ylim[0]],
        [xlim[1], ylim[0]],
        [xlim[1], ylim[1]],
        [xlim[0], ylim[1]],
        [xlim[0], ylim[0]]
    ]

    return mpath.Path(vertices, closed=True).interpolated(interp_fac)
