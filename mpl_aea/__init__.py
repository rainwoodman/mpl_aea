from matplotlib.projections import register_projection

from .aea import AlbersEqualAreaAxes
from .geo import MollweideAxes
# Now register the projection with matplotlib so the user can select
# it.
register_projection(AlbersEqualAreaAxes)
register_projection(MollweideAxes)
