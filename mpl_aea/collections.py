
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
from matplotlib.collections import PolyCollection, TriMesh

from matplotlib.transforms import Affine2D, BboxTransformTo, Transform, blended_transform_factory, Bbox

from . import healpix

class HealpixQuadCollection(PolyCollection):
    def __init__(self, map, mask, nest=False, **kwargs):
        nside = healpix.npix2nside(len(mask))
        self.v, ind = pix2quad(nside, mask.nonzero()[0], nest)
        #print(len(ind), len(self.v), mask.sum())
        PolyCollection.__init__(self, self.v, array=map[ind], **kwargs)

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

class BaseHealpixTriCollection(Collection):
    """
    Class for the efficient drawing of a triangular mesh using
    Gouraud shading.

    A triangular mesh is a :class:`~matplotlib.tri.Triangulation`
    object.
    """
    def __init__(self, **kwargs):
        Collection.__init__(self, **kwargs)

        verts, c = self.get_verts()
        self.set_array(c.reshape(-1))

    def get_map(self):
        return map, mask

    def get_verts(self):
        map, mask = self.get_map()

        nside = healpix.npix2nside(len(map))

        verts, pix, pix_c = pix2tri(nside, mask)
        c = 0.5 * (map[pix] + map[pix_c])
        return verts, c

    @allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return
        renderer.open_group(self.__class__.__name__)
        transform = self.get_transform()
        # Get a list of triangles and the color at each vertex.

        verts, c = self.get_verts()
        self.set_array(c.reshape(-1))
        self.update_scalarmappable()
        colors = self._facecolors.reshape(-1, 3, 4)

        if hasattr(self.axes.transProjection, "vertices_into_view"):
            v = self.axes.transProjection.vertices_into_view(verts)
        else:
            v = verts

        verts = transform.transform(v.reshape(-1, 2)).reshape(v.shape)

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

class HealpixTriCollection(BaseHealpixTriCollection):
    def __init__(self, map, mask, nest=False, **kwargs):
        nside = healpix.npix2nside(len(map))

        verts, pix, pix_c = pix2tri(nside, mask)
        c = 0.5 * (map[pix] + map[pix_c])

        self._verts = verts
        self._c = c

        BaseHealpixTriCollection.__init__(self, **kwargs)

    def get_verts(self):
        return self._verts, self._c

class HealpixHistogram(BaseHealpixTriCollection):
    def __init__(self, ra, dec, weights=None, nside=None, perarea=False, mean=False, range=None, **kwargs):
        self.args = (ra, dec, weights, nside, perarea, mean, range)
        BaseHealpixTriCollection.__init__(self, **kwargs)

    def get_map(self):
        ra, dec, weights, nside, perarea, mean, range = self.args
        if hasattr(self, 'axes') and self.axes is not None:
            x0, x1 = np.radians(self.axes.get_xlim())
            y0, y1 = np.radians(self.axes.get_ylim())
            fraction = np.abs(((np.sin(y1) - np.sin(y0)) * (x1 - x0)) / (4 * np.pi))
            if fraction >= 0.01:
                newnside = 2 ** (int(np.log2(healpix.npix2nside(4096. / fraction) + 1)) + 1)
            else:
                newnside = 8
        else:
            newnside = 8

        if newnside != nside or not hasattr(self, '_w'):
            self.args = ra, dec, weights, newnside, perarea, mean, range

            r = healpix.histogrammap(ra, dec, weights, newnside, perarea=perarea, range=range)

            if weights is not None:
                w, N = r
            else:
                w = r
            if mean:
                mask = N != 0
                w[mask] /= N[mask]
            else:
                mask = w > 0

            self._w = w
            self._mask = mask

        return self._w, self._mask

    def get_datalim(self, transData):
        """ The data lim of a healpix collection.
        """ 
        # FIXME: it is currently set to the full sky.
        #    This could have been trimmed down. 
        #    We want to set xlim smartly such that the largest
        #    empty region is chopped off. I think it is possible, by
        #    doing a histogram in ra, for example. 
        ra, dec, weights, nside, perarea, mean, range = self.args
        vmin = (ra.min(), dec.min())
        vmax = (ra.max(), dec.max())
        return Bbox((vmin, vmax))

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
    theta, phi = healpix.vertices(nside, pix)
    theta = np.round(np.degrees(theta), 5)
    phi = np.round(np.degrees(phi), 5)
    #print 'here', _wrap360(np.array([[ -22.5  , 0. ,  22.5  , 0. ]]), 'right')
    #print 'here', _wrap360(_wrap360(np.array([[ -22.5  , 0. ,  22.5  , 0. ]]), 'right'), 'left')

    # ensure objects are in the same image plane.
    phi, ind = _wrap(phi)
    vertices = np.zeros((len(phi), 4, 2))

    vertices[:, :, 0] = phi
    vertices[:, :, 1] = 90.0 - theta[ind]

    return vertices, pix[ind]

def _wrap(phi):
    ind = np.arange(len(phi))
    wright = _wrap360(phi, 'right')
    wleft = _wrap360(phi, 'left')
    mask = (wright != wleft).any(axis=-1)

    return (np.concatenate([phi[~mask], wright[mask], wleft[mask]], axis=0),
            np.concatenate([ind[~mask], ind[mask], ind[mask]], axis=0))

def _wrap360(phi, dir='left'):
    phi = phi.copy() # make a copy
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


def pix2tri(nside, mask, nest=False):
    """Generate healpix quad vertices for pixels where mask is True

    Args:
        pix: list of pixel numbers
        nest: nested or not
        nside: HealPix nside

    Returns:
        vertices
        vertices: (N, 3,2), RA/Dec coordinates of 3 boundary points of 2 triangles
        pix, pixel id of each vertex
    """
    # each pixel contains 2 triangles.
    pix = mask.nonzero()[0]

    theta, phi = healpix.vertices(nside, pix)
    theta_n, phi_n = healpix.vertices(nside, pix, step=(1.0, 1.0, 1.0, 1.0))
    pix_n = healpix.ang2pix(nside, theta_n, phi_n).reshape(theta.shape)

    pix_c = np.empty_like(pix_n)
    pix_c[...] = pix[:, None]

    bad = ~mask[pix_n]
    pix_n[bad] = pix_c[bad]

    theta = np.degrees(theta)
    phi = np.degrees(phi)

    phi1, ind1 = _wrap(phi[:, [0, 1, 3]])
    phi2, ind2 = _wrap(phi[:, [1, 2, 3]])
    cen1 = pix_c[:, [0, 1, 3]][ind1]
    cen2 = pix_c[:, [1, 2, 3]][ind2]
    neigh1 = pix_n[:, [0, 1, 3]][ind1]
    neigh2 = pix_n[:, [1, 2, 3]][ind2]
    theta1 = theta[:, [0, 1, 3]][ind1]
    theta2 = theta[:, [1, 2, 3]][ind2]
    phi = np.concatenate([phi1, phi2], axis=0)
    ind = np.concatenate([ind1, ind2], axis=0)
    theta = np.concatenate([theta1, theta2], axis=0)
    neigh = np.concatenate([neigh1, neigh2], axis=0)
    cen = np.concatenate([cen1, cen2], axis=0)

    vertices = np.zeros((len(phi), 3, 2))

    vertices[:, :, 0] = phi
    vertices[:, :, 1] = 90.0 - theta

    return vertices, neigh, cen

