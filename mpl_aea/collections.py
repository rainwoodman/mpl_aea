
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
    wright = _wrap360(phi, 'right')
    wleft = _wrap360(phi, 'left')
    mask = (wright != wleft).any(axis=-1)

    vertices = np.zeros((len(pix) - mask.sum(), 4, 2))

    vertices[:, :, 0] = wright[~mask]
    vertices[:, :, 1] = 90.0 - theta[~mask]
    verticesl = np.zeros((mask.sum(), 4, 2))
    verticesr = np.zeros((mask.sum(), 4, 2))
    verticesl[:, :, 1] = 90.0 - theta[mask]
    verticesr[:, :, 1] = 90.0 - theta[mask]
    verticesl[:, :, 0] = wleft[mask]
    verticesr[:, :, 0] = wright[mask]

    return np.concatenate([vertices, verticesl, verticesr], axis=0), np.concatenate([pix[~mask], pix[mask], pix[mask]], axis=0)


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

