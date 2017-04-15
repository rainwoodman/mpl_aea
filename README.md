# mpl_aea

This is probably the most integrated astronomy (RA, DEC) plotting tool with
matplotlib. 

We add two new projections : ast.aea and ast.mollweide; these take RA and DEC
coordinates as x and y.

aea is AlbertEqualArea mapping. We by default use the reduced form where
dec1 == - dec2 as a Cylindrical mapping. (modify dec1, dec2 with `set_parallels`)
 mollweide is similiar to the one
used in healpy -- it does not allow zooming in.

In addition to the usual matplotlib methods (plot, scatter, ...),
we add a histmap method for visualizating density fields.
The histogram is done on Healpix pixels.

We also have a mapshow method for showing healpix maps.

We use matplotlib's Gouraud interpolation to improve the image quality.
Each healpix pixel is break into four triangles,
with the center set to the pixel value
and four corners set to the mean against the neighbouring pixel.

Sometimes it is necessary to manually specify a xlim and ylim, and shifting
the tick labels. See examples.ipynb for examples.

