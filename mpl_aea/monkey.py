import matplotlib.transforms as mtransforms
import matplotlib.path as mpath
import matplotlib.artist as martist

# stolen from matplotlib head.
class TransformedPatchPath(mtransforms.TransformedPath):
    """
    A :class:`TransformedPatchPath` caches a non-affine transformed copy of
    the :class:`~matplotlib.path.Patch`. This cached copy is automatically
    updated when the non-affine part of the transform or the patch changes.
    """
    def __init__(self, patch):
        """
        Create a new :class:`TransformedPatchPath` from the given
        :class:`~matplotlib.path.Patch`.
        """
        mtransforms.TransformNode.__init__(self)

        transform = patch.get_transform()
        self._patch = patch
        self._transform = transform
        self.set_children(transform)
        self._path = patch.get_path()
        self._transformed_path = None
        self._transformed_points = None

    def _revalidate(self):
        patch_path = self._patch.get_path()
        # Only recompute if the invalidation includes the non_affine part of
        # the transform, or the Patch's Path has changed.
        if (self._transformed_path is None or self._path != patch_path or
                (self._invalid & self.INVALID_NON_AFFINE ==
                    self.INVALID_NON_AFFINE)):
            self._path = patch_path
            self._transformed_path = \
                self._transform.transform_path_non_affine(patch_path)
            self._transformed_points = \
                mpath.Path._fast_from_codes_and_verts(
                    self._transform.transform_non_affine(patch_path.vertices),
                    None,
                    {'interpolation_steps': patch_path._interpolation_steps,
                     'should_simplify': patch_path.should_simplify})
        self._invalid = 0

old_set_clip_path = martist.Artist.__dict__['set_clip_path']

def set_clip_path(self, path, transform=None):
    from matplotlib.patches import Patch

    success = False
    if transform is None:
        if isinstance(path, Patch):
            self._clippath = TransformedPatchPath(path)
            self.pchanged()
            self.stale = True
            return
    return old_set_clip_path(self, path, transform)

martist.Artist.set_clip_path = set_clip_path

