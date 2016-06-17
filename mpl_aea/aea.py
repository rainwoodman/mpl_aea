""" 
    Native matplotlib support of frequently used 2d projections,
    for looking up to the sky.

    This file is initially developed as part of skymapper by Peter Melchior
    based on the example in matplotlib.

    It is later adopted by me (Yu Feng), and I will maintain a copy in
    imaginglss for easier access, also because I do plan to clean up
    the function signatures and variable naming (breaking compatibility with
    old skymapper code).

    The current version adds the ability to generate equal area histograms
    on HealPix pixels.

    It does not depend on healpy, there is a minimal python implementation of 
    healpix at the end of the file; imported in the javascript/lua style.
    
    The intention is one day we will submit a PR of this to matplotlib.

    What does not work:
        
        1. Panning.
        2. Color bar is sometimes in the wrong place
        3. Label locations are poorly calculated.

    What does work:
        Evertying else.

    Author: Yu Feng 
            Peter Melchior

"""
from __future__ import unicode_literals

import matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.tri.triangulation import Triangulation

from matplotlib.ticker import NullLocator, Formatter, FixedLocator, MaxNLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform, blended_transform_factory, Bbox
import matplotlib.spines as mspines
import matplotlib.axis as maxis

import numpy as np
from . import healpix

__author__ = "Yu Feng"
__email__ =  "rainwoodman@gmail.com"

class SkymapperAxes(Axes):
    """
    A base class for a Skymapper axes that takes in ra0, dec0, dec1, dec2.

    The base class takes care of clipping and interpolating with matplotlib.

    Subclass and override class method get_projection_class.

    """
    # The subclass projection must specify a name.  This will be used be the
    # user to select the projection.

    name = None

    @classmethod
    def get_projection_class(kls):
        raise NotImplementedError('Must implement this in subclass')

    def __init__(self, *args, **kwargs):
        self.ra0 = None
        self.dec0 = None
        self.dec1 = None
        self.dec2 = None

        Axes.__init__(self, *args, **kwargs)

        self.cla()

    def _init_axis(self):
        # Axes._init_axis() -- until HammerAxes.xaxis.cla() works.
        self.xaxis = maxis.XAxis(self)
        self.spines['bottom'].register_axis(self.xaxis)
        self.spines['top'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.spines['left'].register_axis(self.yaxis)
        self.spines['right'].register_axis(self.yaxis)
        self._update_transScale()

    def cla(self):
        """
        Override to set up some reasonable defaults.
        """
        # Don't forget to call the base class
        Axes.cla(self)

        # Turn off minor ticking altogether
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())

        self.xaxis.set_major_locator(MaxNLocator(5, prune='both'))
        self.yaxis.set_major_locator(MaxNLocator(5, prune='both'))

        # Do not display ticks -- we only want gridlines and text
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')

        self.set_center(None, None)

        # FIXME: probabaly want to override autoscale_view
        # to properly handle wrapping introduced by margin
        # and properlty wrap data. 
        # It doesn't make sense to have xwidth > 360. 
        self._tight = True

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # There are three important coordinate spaces going on here:
        #
        #    1. Data space: The space of the data itself
        #
        #    2. Axes space: The unit rectangle (0, 0) to (1, 1)
        #       covering the entire plot area.
        #
        #    3. Display space: The coordinates of the resulting image,
        #       often in pixels or dpi/inch.

        # This function makes heavy use of the Transform classes in
        # ``lib/matplotlib/transforms.py.`` For more information, see
        # the inline documentation there.

        # The goal of the first two transformations is to get from the
        # data space (in this case meridian and parallel) to axes
        # space.  It is separated into a non-affine and affine part so
        # that the non-affine part does not have to be recomputed when
        # a simple affine change to the figure has been made (such as
        # resizing the window or changing the dpi).

        # 1) The core transformation from data space into
        # rectilinear space defined in the HammerTransform class.
        self.transProjection = self.get_projection_class()()
        self.transProjection.set_center((180, 0))
        self.transProjection.set_dec1(-65)
        self.transProjection.set_dec2(80)

        # 2) The above has an output range that is not in the unit
        # rectangle, so scale and translate it so it fits correctly
        # within the axes.  The peculiar calculations of xscale and
        # yscale are specific to a Aitoff-Hammer projection, so don't
        # worry about them too much.

        # This will be updated after the xy limits are set.
        self.transAffine = Affine2D()

        # 3) This is the transformation from axes space to display
        # space.
        self.transAxes = BboxTransformTo(self.bbox)

        # Now put these 3 transforms together -- from data all the way
        # to display coordinates.  Using the '+' operator, these
        # transforms will be applied "in order".  The transforms are
        # automatically simplified, if possible, by the underlying
        # transformation framework.
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes

        self.transClip = \
            self.transProjection + \
            self.transAffine

        # The main data transformation is set up.  Now deal with
        # gridlines and tick labels.

        # Longitude gridlines and ticklabels.  The input to these
        # transforms are in display space in x and axes space in y.
        # Therefore, the input values will be in range (-xmin, 0),
        # (xmax, 1).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the equator.
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1.0, 180) \
            .translate(0.0, -90)

        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData

        self._xaxis_text1_transform = \
            self._xaxis_pretransform + \
            self.transData + \
            Affine2D().translate(0.0, -8.0)
        self._xaxis_text2_transform = \
            self._xaxis_pretransform+ \
            self.transData + \
            Affine2D().translate(0.0, -8.0)

        # Now set up the transforms for the parallel ticks.  The input to
        # these transforms are in axes space in x and display space in
        # y.  Therefore, the input values will be in range (0, -ymin),
        # (1, ymax).  The goal of these transforms is to go from that
        # space to display space.  The tick labels will be offset 4
        # pixels from the edge of the axes ellipse.
        self._yaxis_stretch = Affine2D().scale(360, 1.0).translate(0.0, 0.0)
        self._yaxis_stretch1 = Affine2D().scale(360, 1.0).translate(0.0, 0.0)
        self._yaxis_stretch2 = Affine2D().scale(360, 1.0).translate(0.0, 0.0)

        self._yaxis_transform = \
            self._yaxis_stretch + \
            self.transData

        self._yaxis_text1_transform = \
            self._yaxis_stretch1 + \
            self.transData
#            Affine2D().translate(-8.0, 0.0)

        self._yaxis_text2_transform = \
            self._yaxis_stretch2 + \
            self.transData
#            Affine2D().translate(8.0, 0.0)

    def _update_affine(self):
        # update the transformations and clip paths
        # after new lims are set.
        if self.ra0 is None:
            x0, x1 = self.viewLim.intervalx
            ra0 = 0.5 * (x0 + x1)
        else:
            ra0 = self.ra0
        if self.dec0 is None:
            y0, y1 = self.viewLim.intervaly
            dec0 = 0.5 * (y0 + y1)
        else:
            dec0 = self.dec0
        if self.dec1 is None:
            y0, y1 = self.viewLim.intervaly
            dec1 = y0 + (y1 - y0) / 12.
        else:
            dec1 = self.dec1
        if self.dec2 is None:
            y0, y1 = self.viewLim.intervaly
            dec2 = y1 - (y1 - y0) / 12.
        else:
            dec2 = self.dec2

        self.transProjection.set_center((ra0, dec0))
        self.transProjection.set_dec1(dec1)
        self.transProjection.set_dec2(dec2)

        self._yaxis_stretch\
            .clear() \
            .scale(self.viewLim.width, 1.0) \
            .translate(self.viewLim.x0, 0)

        self._yaxis_stretch1\
            .clear() \
            .scale(self.viewLim.width, 1.0) \
            .translate(self.viewLim.x0 - 0.00 * self.viewLim.width, 0)

        self._yaxis_stretch2\
            .clear() \
            .scale(self.viewLim.width, 1.0) \
            .translate(self.viewLim.x0 + 0.00 * self.viewLim.width, 0)

        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self.viewLim.height) \
            .translate(0.0, self.viewLim.y0)

        corners_data = np.array([[self.viewLim.x0, self.viewLim.y0],
                      [ra0,            self.viewLim.y0],
                      [self.viewLim.x1, self.viewLim.y0],
                      [self.viewLim.x1, self.viewLim.y1],
                      [self.viewLim.x0, self.viewLim.y1],])

        corners = self.transProjection.transform_non_affine(corners_data)

        x0 = corners[0][0]
        x1 = corners[2][0]

        # special case when x1 is wrapped back to x0
        # FIXME: I don't think we need it anymore.
        if x0 == x1: x1 = - x0

        y0 = corners[1][1]
        y1 = max([corners[3][1], corners[4][1]])

        xscale = x1 - x0
        yscale = y1 - y0

        self.transAffine.clear() \
            .translate( - (x0 + x1) * 0.5, - (y0 + y1) * 0.5) \
            .scale(0.95 / xscale, 0.95 / yscale)  \
            .translate(0.5, 0.5)

        # now update the clipping path
        path = Path(corners_data)
        path0 = self.transProjection.transform_path(path)
        path = self.transClip.transform_path(path)
        self.patch.set_xy(path.vertices)

    def get_xaxis_transform(self, which='grid'):
        """
        Override this method to provide a transformation for the
        x-axis grid and ticks.
        """
        assert which in ['tick1', 'tick2', 'grid']
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._xaxis_text1_transform, 'center', 'center'

    def get_xaxis_text2_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        secondary x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._xaxis_text2_transform, 'center', 'center'

    def get_yaxis_transform(self, which='grid'):
        """
        Override this method to provide a transformation for the
        y-axis grid and ticks.
        """
        assert which in ['tick1', 'tick2', 'grid']
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._yaxis_text1_transform, 'center', 'center'

    def get_yaxis_text2_transform(self, pixelPad):
        """
        Override this method to provide a transformation for the
        secondary y-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        return self._yaxis_text2_transform, 'center', 'center'

    def _gen_axes_patch(self):
        """
        ClipPath.

        Initially set to a size of 2 box in transAxes.

        After xlim and ylim are set, this will be changed to the actual
        region in transData.

        For unclear reason the very initial clip path is always applied
        to the grid. Therefore we set size to 2.0 to avoid bad clipping.
        """
        return Polygon([(0, 0), (2, 0), (2, 2), (0, 2)], fill=False)

    def _gen_axes_spines(self):
        d = {
            'left': mspines.Spine.linear_spine(self, spine_type='left'),
            'right': mspines.Spine.linear_spine(self, spine_type='right'),
            'top': mspines.Spine.linear_spine(self, spine_type='top'),
            'bottom': mspines.Spine.linear_spine(self, spine_type='bottom'),
        }
        d['left'].set_position(('axes', 0))
        d['right'].set_position(('axes', 1))
        d['top'].set_position(('axes', 0))
        d['bottom'].set_position(('axes', 1))
        #FIXME: these spines can be moved wit set_position(('axes', ?)) but
        # 'data' fails. Because the transformation is non-separatable,
        # and because spines / data makes that assumption, we probably
        # do not have a easy way to support moving spines via native matplotlib
        # api on data axis.

        # also the labels currently do not follow the spines. Likely because
        # they are not registered?

        return d

    # Prevent the user from applying scales to one or both of the
    # axes.  In this particular case, scaling the axes wouldn't make
    # sense, so we don't allow it.
    def set_xscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_xscale(self, *args, **kwargs)

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError
        Axes.set_yscale(self, *args, **kwargs)

    def set_center(self, ra0, dec0):
        """ Set the center of ra """
        self.ra0 = ra0
        self.dec0 = dec0
        self._update_affine()

    def set_parallels(self, dec1, dec2):
        """ Set the parallels """
        self.dec1 = dec1
        self.dec2 = dec2
        self._update_affine()

    # when xlim and ylim are updated, the transformation
    # needs to be updated too.
    def set_xlim(self, *args, **kwargs):
        Axes.set_xlim(self, *args, **kwargs)

        # FIXME: wrap x0 x1 to ensure they enclose ra0.
        x0, x1 = self.viewLim.intervalx
        if self.ra0 is not None:
            if not x0 <= self.transProjection.ra0 or \
               not x1 > self.transProjection.ra0:
                raise ValueError("The given limit in RA does not enclose ra0")

        self._update_affine()

    def set_ylim(self, *args, **kwargs):
        Axes.set_ylim(self, *args, **kwargs)
        self._update_affine()

    def _histmap(self, show, ra, dec, weights=None, nside=32, perarea=False, mean=False, range=None, **kwargs):
        r = histogrammap(ra, dec, weights, nside, perarea=perarea, range=range)

        if weights is not None:
            w, N = r
        else:
            w = r
        if mean:
            mask = N != 0
            w[mask] /= N[mask]
        else:
            mask = w > 0
        return w, mask, show(w, mask, nest=False, **kwargs)

    def histmap(self, ra, dec, weights=None, nside=32, perarea=False, mean=False, range=None, **kwargs):
        return self._histmap(self.mapshow, ra, dec, weights, nside, perarea, mean, range, **kwargs)

    def histcontour(self, ra, dec, weights=None, nside=32, perarea=False, mean=False, range=None, **kwargs):
        return self._histmap(self.mapcontour, ra, dec, weights, nside, perarea, mean, range, **kwargs)

    def mapshow(self, map, mask=None, nest=False, shading='flat', **kwargs):
        """ Display a healpix map """
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        defaults = dict(rasterized=True,
                    alpha=1.0,
                    linewidth=0)
        defaults.update(kwargs)
        if mask is None:
            mask = map == map

        if shading == 'flat':
            coll = HealpixCollection(map, mask, 
                    transform=self.transData, **defaults)
        else:
            coll = HealpixTriCollection(map, mask, transform=self.transData, **defaults)
        
        coll.set_clim(vmin=vmin, vmax=vmax)
        self.add_collection(coll)
        self._sci(coll)
        self.autoscale_view(tight=True)

        return coll

    def mapcontour(self, map, mask=None, nest=False, **kwargs):
        """ Display a healpix map as coutours. This is approximate. """
        if mask is None:
            mask = map == map

        ra, dec = healpix.pix2radec(healpix.npix2nside(len(map)), mask.nonzero()[0])
        im = self.tricontour(ra, dec, map[mask], **kwargs)
        self._sci(im)
        self.autoscale_view(tight=True)
        return im

    def format_coord(self, lon, lat):
        """
        Override this method to change how the values are displayed in
        the status bar.

        In this case, we want them to be displayed in degrees N/S/E/W.
        """
        lon = lon
        lat = lat
        if lat >= 0.0:
            ns = 'N'
        else:
            ns = 'S'
        if lon >= 0.0:
            ew = 'E'
        else:
            ew = 'W'
        # \u00b0 : degree symbol
        return '%f\u00b0%s, %f\u00b0%s' % (abs(lat), ns, abs(lon), ew)

    class DegreeFormatter(Formatter):
        """
        This is a custom formatter that converts the native unit of
        radians into (truncated) degrees and adds a degree symbol.
        """

        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            degrees = round(x / self._round_to) * self._round_to
            # \u00b0 : degree symbol
            return "%d\u00b0" % degrees

    def set_meridian_grid(self, degrees):
        """
        Set the number of degrees between each meridian grid.

        It provides a more convenient interface to set the ticking than set_xticks would.
        """
        # Set up a FixedLocator at each of the points, evenly spaced
        # by degrees.
        x0, x1 = self.get_xlim()
        number = abs((x1 - x0) / degrees) + 1
        self.xaxis.set_major_locator(
            FixedLocator(
                np.linspace(x0, x1, number, True)[1:-1]))
        # Set the formatter to display the tick labels in degrees,
        # rather than radians.
        self.xaxis.set_major_formatter(self.DegreeFormatter(degrees))

    def set_parallel_grid(self, degrees):
        """
        Set the number of degrees between each meridian grid.

        It provides a more convenient interface than set_yticks would.
        """
        # Set up a FixedLocator at each of the points, evenly spaced
        # by degrees.
        y0, y1 = self.get_ylim()
        number = ((y1 - y0) / degrees) + 1
        self.yaxis.set_major_locator(
            FixedLocator(
                np.linspace(y0, y1, number, True)[1:-1]))
        # Set the formatter to display the tick labels in degrees,
        # rather than radians.
        self.yaxis.set_major_formatter(self.DegreeFormatter(degrees))

    # Interactive panning and zooming is not supported with this projection,
    # so we override all of the following methods to disable it.
    def _in_axes(self, mouseevent):
        if hasattr(self._pan_trans):
            return True
        else:
            return Axes._in_axes(self, mouseevent)

    def can_zoom(self):
        """
        Return True if this axes support the zoom box
        """
        return True

    def start_pan(self, x, y, button):
        self._pan_trans = self.transAxes.inverted() + \
                blended_transform_factory(
                        self._yaxis_stretch,
                        self._xaxis_pretransform,)

    def end_pan(self):
        delattr(self, '_pan_trans')

    def drag_pan(self, button, key, x, y):
        pan1 = self._pan_trans.transform([(x, y)])[0]
        self.set_ra0(360 - pan1[0])
        self.set_dec0(pan1[1])
        self._update_affine()

# now define the Albers equal area axes

class AlbersEqualAreaAxes(SkymapperAxes):
    """
    A custom class for the Albers Equal Area projection.

    https://en.wikipedia.org/wiki/Albers_projection
    """

    name = 'aea'

    @classmethod
    def get_projection_class(kls):
        return kls.AlbersEqualAreaTransform

    # Now, the transforms themselves.
    class AlbersEqualAreaTransform(Transform):
        """
        The base Hammer transform.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, **kwargs):
            Transform.__init__(self, **kwargs)
            self.dec0 = 0
            self.ra0 = 180
            self.dec1 = -60
            self.dec2 = 30
            self._update()

        def set_center(self, center):
            ra0, dec0 = center
            self.ra0  = ra0
            self.dec0 = dec0
            self._update()

        def set_dec1(self, dec1):
            self.dec1 = dec1
            self._update()

        def set_dec2(self, dec2):
            self.dec2 = dec2
            self._update()

        def _update(self):
            self.n = 0.5 * (np.sin(np.radians(self.dec1)) 
                          + np.sin(np.radians(self.dec2)))

            self.C = np.cos(np.radians(self.dec1))**2 + 2 * self.n * np.sin(np.radians(self.dec1))
            self.rho0 = self.__rho__(self.dec0)

        def __rho__(self, dec):
            if self.n == 0:
                return np.sqrt(self.C - 2 * self.n * np.sin(np.radians(dec)))
            else:
                return np.sqrt(self.C - 2 * self.n * np.sin(np.radians(dec))) / self.n

        def transform_non_affine(self, ll):
            """
            Override the transform_non_affine method to implement the custom
            transform.

            The input and output are Nx2 numpy arrays.
            """
            ra = ll[:,0]
            dec = ll[:,1]
            ra0 = self.ra0
            ra_ = np.radians(ra - ra0) # Do not inverse for RA

            # FIXME: problem with the slices sphere: outer parallel needs to be dubplicated at the expense of the central one
            if self.n == 0:
                rt = np.array([
                    self.rho0 * (ra_),
                    - self.rho0 * (np.sin(np.radians(self.dec0) - np.sin(np.radians(dec)))),
                    ]).T
            else:
                theta = self.n * ra_
                rho = self.__rho__(dec)
                rt = np.array([
                       rho*np.sin(theta),
                       self.rho0 - rho*np.cos(theta)]).T
            #if np.isnan(rt).any(): 
            #    raise ValueError('nan occured : ll =%s' % (str(ll)))
            return rt

        # This is where things get interesting.  With this projection,
        # straight lines in data space become curves in display space.
        # This is done by interpolating new values between the input
        # values of the data.  Since ``transform`` must not return a
        # differently-sized array, any transform that requires
        # changing the length of the data array must happen within
        # ``transform_path``.
        def transform_path_non_affine(self, path):
            # Adaptive interpolation:
            # we keep adding control points, till all control points
            # have an error of less than 0.01 (about 1%)
            # or if the number of control points is > 80.
            ra0 = self.ra0
            path = path.cleaned(curves=False)
            v = path.vertices
            diff = v[:, 0] - v[0, 0]
            v00 = v[0][0] - ra0
            while v00 > 180: v00 -= 360
            while v00 < -180: v00 += 360
            v00 += ra0
            v[:, 0] = v00 + diff
            nonstop = path.codes > 0
            path = Path(v[nonstop], path.codes[nonstop])
            isteps = int(path._interpolation_steps * 1.5)
            if isteps < 10: isteps = 10
            while True:
                ipath = path.interpolated(isteps)
                tiv = self.transform(ipath.vertices)
                itv = Path(self.transform(path.vertices)).interpolated(isteps).vertices
                if np.mean(np.abs(tiv - itv)) < 0.01:
                    break
                if isteps > 20:
                    break
                isteps = int(isteps * 1.5)
            return Path(tiv, ipath.codes)

        transform_path_non_affine.__doc__ = \
            Transform.transform_path_non_affine.__doc__

        if matplotlib.__version__ < '1.2':
            transform = transform_non_affine
            transform_path = transform_path_non_affine
            transform_path.__doc__ = Transform.transform_path.__doc__

        def inverted(self):
            return AlbersEqualAreaAxes.InvertedAlbersEqualAreaTransform(self)
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedAlbersEqualAreaTransform(Transform):
        """ Inverted transform.

            This will always only give values in the prime ra0-180 ~ ra0+180 range, I believe.
            So it is inherently broken. I wonder when matplotlib actually calls this function,
            given that interactive is disabled.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, inverted, **kwargs):
            Transform.__init__(self, **kwargs)
            self.inverted = inverted

        def transform_non_affine(self, xy):
            x = xy[:,0]
            y = xy[:,1]
            inverted = self.inverted

            rho = np.sqrt(x**2 + (inverted.rho0 - y)**2)

            # make sure that the signs are correct
            if inverted.n == 0:
                rt = np.degrees(
                        [
                    np.radians(inverted.ra0) + x / inverted.rho0,
                    np.arcsin(y / inverted.rho0 + np.sin(np.radians(inverted.dec0)))
                        ]).T
                return rt
            elif inverted.n > 0:
                theta = np.degrees(np.arctan2(x, inverted.rho0 - y))
            else:
                theta = np.degrees(np.arctan2(-x, -(inverted.rho0 - y)))
            return np.degrees([np.radians(inverted.ra0) + theta/inverted.n,
                np.arcsin((inverted.C - (rho * inverted.n)**2)/(2*inverted.n))]).T

            transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        if matplotlib.__version__ < '1.2':
            transform = transform_non_affine

        def inverted(self):
            # The inverse of the inverse is the original transform... ;)
            return self.inverted

        inverted.__doc__ = Transform.inverted.__doc__

class HealpixCollection(PolyCollection):
    def __init__(self, map, mask, nest=False, **kwargs):
        nside = healpix.npix2nside(len(mask))
        self.v = pix2quad(nside, mask.nonzero()[0], nest)
        PolyCollection.__init__(self, self.v, array=map[mask], **kwargs)

    def get_datalim(self, transData):
        """ The data lim of a healpix collection.
        """ 
        # FIXME: it is currently set to the full sky.
        #    This could have been trimmed down. 
        #    We want to set xlim smartly such that the largest
        #    empty region is chopped off. I think it is possible, by
        #    doing a histogram in ra, for example. 
        vmin = (0, -90)
        vmax = (360, 90)
        return Bbox((vmin, vmax))

import matplotlib.transforms as mtransforms
import warnings
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.cbook as cbook
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import docstring
import matplotlib.transforms as transforms
import matplotlib.artist as artist
from matplotlib.artist import allow_rasterization
import matplotlib.backend_bases as backend_bases
import matplotlib.path as mpath
from matplotlib import _path
import matplotlib.mlab as mlab
import matplotlib.lines as mlines
from matplotlib.collections import Collection

class HealpixTriCollection(Collection):
    """
    Class for the efficient drawing of a triangular mesh using
    Gouraud shading.

    A triangular mesh is a :class:`~matplotlib.tri.Triangulation`
    object.
    """
    def __init__(self, map, mask, nest=False, **kwargs):
        Collection.__init__(self, **kwargs)
        nside = healpix.npix2nside(len(map))
        # remove the first axes
        verts = pix2tri(nside, mask.nonzero()[0]).reshape(-1, 3, 2)
        c = np.ones((verts.shape[0], verts.shape[1])) * np.repeat(map[mask][:, None], 2, axis=0)

        self._verts = verts
        self._shading = 'gouraud'
        self._is_filled = True
        self.set_array(c.reshape(-1))
        
    def get_paths(self):
        if self._paths is None:
            self.set_paths()
        return self._paths

    def set_paths(self):
        self._paths = self.convert_mesh_to_paths(self._verts)

    @staticmethod
    def convert_mesh_to_paths(verts):
        """
        Converts a given mesh into a sequence of
        :class:`matplotlib.path.Path` objects for easier rendering by
        backends that do not directly support meshes.

        This function is primarily of use to backend implementers.
        """
        Path = mpath.Path
        return [Path(x) for x in verts]

    @allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__)
        transform = self.get_transform()

        # Get a list of triangles and the color at each vertex.
        
        verts = self._verts
        
        self.update_scalarmappable()
        colors = self._facecolors.reshape(-1, 3, 4)
        
        oldshape = list(verts.shape)
        
        verts = transform.transform(verts.reshape(-1, 2)).reshape(oldshape)

        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_linewidth(self.get_linewidth()[0])
        renderer.draw_gouraud_triangles(gc, verts, colors, mtransforms.IdentityTransform())
        gc.restore()
        renderer.close_group(self.__class__.__name__)

    def get_datalim(self, transData):
        """ The data lim of a healpix collection.
        """ 
        # FIXME: it is currently set to the full sky.
        #    This could have been trimmed down. 
        #    We want to set xlim smartly such that the largest
        #    empty region is chopped off. I think it is possible, by
        #    doing a histogram in ra, for example. 
        vmin = (0, -90)
        vmax = (360, 90)
        return Bbox((vmin, vmax))


def _wrap360(phi, dir='left'):
    phi[np.abs(phi) < 1e-9] = 0
    if dir == 'left':
        ref = phi.min(axis=-1)
    else:
        ref = phi.max(axis=-1)
#    print('ref', ref, phi, ref % 360 - ref)
    diff = (ref % 360) - ref 
    phi = phi + diff[:, None]
    
    #diff = phi - ref[:, None] 
    #print('great', (diff > 180).sum())
    #diff[diff > 180] -= 360 
    #print('less', (diff < -180).sum())
    #diff[diff < -180] += 360
    #phi = ref[:, None] + diff
    return phi 

# a few helper functions talking to healpy/healpix.
def pix2quad(nside, pix, nest=False):
    """Generate healpix quad vertices for pixels where mask is True

    Args:
        pix: list of pixel numbers
        nest: nested or not
        nside: HealPix nside

    Returns:
        vertices
        vertices: (N,4,2), RA/Dec coordinates of 4 boundary points of cell
    """

    pix = np.asarray(pix)
    vertices = np.zeros((pix.size, 4, 2))

    theta, phi = healpix.vertices(nside, pix)
    theta = np.degrees(theta)
    phi = np.degrees(phi)

    vertices[:, :, 0] = phi
    vertices[:, :, 1] = 90.0 - theta

    # ensure objects are in the same image plane.
    vertices[:, :, 0] = _wrap360(phi, 'right')

    return vertices

def pix2tri(nside, pix, nest=False):
    """Generate healpix quad vertices for pixels where mask is True

    Args:
        pix: list of pixel numbers
        nest: nested or not
        nside: HealPix nside

    Returns:
        vertices
        vertices: (N,3,2,2), RA/Dec coordinates of 3 boundary points of 2 triangles
    """

    # each pixel contains 2 triangles.
    pix = np.asarray(pix)
    vertices = np.zeros((pix.size, 2, 3, 2))

    theta, phi = healpix.vertices(nside, pix)
    theta = np.degrees(theta)
    phi = np.degrees(phi)

    vertices[:, 0, :, 0] = _wrap360(phi[:, [0, 1, 3]], 'left')
    vertices[:, 0, :, 1] = 90.0 - theta[:, [0, 1, 3]]
    vertices[:, 1, :, 0] = _wrap360(phi[:, [1, 2, 3]], 'right')
    vertices[:, 1, :, 1] = 90.0 - theta[:, [1, 2, 3]]

    return vertices

def histogrammap(ra, dec, weights=None, nside=32, perarea=False, range=None):
    if range is not None:
        (ra1, ra2), (dec1, dec2) = range
        m  = (ra >= ra1)& (ra <= ra2)
        m &= (dec >= dec1)& (dec <= dec2)
        ra = ra[m]
        dec = dec[m]
        if weights is not None:
            weights = weights[m]

    ipix = healpix.ang2pix(nside, np.radians(90-dec), np.radians(ra))
    npix = healpix.nside2npix(nside)
    if perarea:
        npix = healpix.nside2npix(nside)
        sky = 360. ** 2 / np.pi
        area = 1. * (sky / npix)
    else:
        area = 1

    if weights is not None:
        w = np.bincount(ipix, weights=weights, minlength=npix)
        N = np.bincount(ipix, minlength=npix)
        w = w / area
        N = N / area
        return w, N
    else:
        w = 1.0 * np.bincount(ipix, minlength=npix)
        return w / area


