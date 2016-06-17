from matplotlib.projections import register_projection

from .aea import AlbersEqualAreaAxes
# Now register the projection with matplotlib so the user can select
# it.
register_projection(AlbersEqualAreaAxes)
