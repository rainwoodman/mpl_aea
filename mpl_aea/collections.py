
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
        self.plot_nside = None
        BaseHealpixTriCollection.__init__(self, **kwargs)

    def get_map(self):
        ra, dec, weights, nside, perarea, mean, range = self.args
        if nside is None:
            if hasattr(self, 'axes') and self.axes is not None:
                x0, x1 = np.radians(self.axes.get_xlim())
                y0, y1 = np.radians(self.axes.get_ylim())
                fraction = np.abs(((np.sin(y1) - np.sin(y0)) * (x1 - x0)) / (4 * np.pi))
                if fraction >= 0.01:
                    nside = 2 ** (int(np.log2(healpix.npix2nside(4096. / fraction) + 1)) + 1)
                else:
                    nside = 8
            else:
                nside = 8

        if self.plot_nside != nside or not hasattr(self, '_w'):

            r = healpix.histogrammap(ra, dec, weights, nside, perarea=perarea, range=range)

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
            self.plot_nside = nside

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
        ref = ra.min()
        ra = ra - ref
        while (ra > 180).any():
            ra[ra > 180] -= 360
        while (ra <= -180).any():
            ra[ra <= -180] += 360
        ra = ra + ref

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
    theta_c, phi_c = healpix.pix2ang(nside, pix)
    theta = np.concatenate([theta, theta_c[..., None]], axis=-1)
    phi = np.concatenate([phi, phi_c[..., None]], axis=-1)

    theta_n, phi_n = healpix.vertices(nside, pix, step=(1.0, 1.0, 1.0, 1.0))
    pix_n = healpix.ang2pix(nside, theta_n, phi_n).reshape(theta_n.shape)

    other_pix = np.concatenate([pix_n, pix[..., None]], axis=-1)

    pix_c = np.empty_like(other_pix)
    pix_c[...] = pix[:, None]
    bad = ~mask[other_pix]
    other_pix[bad] = pix_c[bad]


    theta = np.degrees(theta)
    phi = np.degrees(phi)

    r = []
    for tri in [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]:
        phi1, ind1 = _wrap(phi[:, tri])
        cen1 = pix_c[:, tri][ind1]
        neigh1 = other_pix[:, tri][ind1]
        theta1 = theta[:, tri][ind1]
        r.append((phi1, ind1, cen1, neigh1, theta1))

    phi, ind, cen, neigh, theta = [np.concatenate(i , axis=0) for i in zip(*r)]

    vertices = np.zeros((len(phi), 3, 2))

    vertices[:, :, 0] = phi
    vertices[:, :, 1] = 90.0 - theta

    return vertices, neigh, cen

