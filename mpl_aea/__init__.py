from matplotlib.projections import register_projection

from .aea import AlbersEqualAreaAxes
from .geo import MollweideAxes
# Now register the projection with matplotlib so the user can select
# it.
register_projection(AlbersEqualAreaAxes)
register_projection(MollweideAxes)

from .version import __version__

try:
    from matplotlib.transforms import TransformedPatchPath
except ImportError:
    from . import monkey

