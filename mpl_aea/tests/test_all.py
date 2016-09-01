import mpl_aea

def test_aea():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Now make a simple example using the custom projection.

    import numpy as np

    fig = Figure(figsize=(6, 6))

#    ra = np.random.uniform(size=100, low=0, high=360)
#    dec = np.random.uniform(size=100, low=-90, high=90)
    ra = np.linspace(0, 360, 100)
    dec = np.linspace(-90, 90, 100)
    ax = fig.add_subplot(111, projection="aea")
    ax.set_xlim(359, 0)
    ax.set_ylim(-70, 70)
    ax.set_parallels(-20, 60)
    ax.set_center(180, 0)
    ax.plot(ra, dec, '*')
    ax.axhline(-20)
    ax.axvline(140)

#    ax.plot(*pix2tri(8, [104, 105, 106]).reshape(-1, 2).T, color='k')

    ra = np.random.uniform(size=1000, low=30, high=60)
    dec = np.random.uniform(size=1000, low=-50, high=50)
    ax.histmap(ra, dec, nside=32, weights=ra * dec, mean=True)

    ra = np.random.uniform(size=1000, low=120, high=160)
    dec = np.random.uniform(size=1000, low=-50, high=50)
    ax.histcontour(ra, dec, weights=ra * dec, nside=32, mean=True)

    ax.tick_params(labelright=True, labeltop=True)

    ax.tripcolor(ra, dec, ra*dec)
    fig.colorbar(ax._gci())

    ra = np.random.uniform(size=1000, low=180, high=200)
    dec = np.random.uniform(size=1000, low=-50, high=50)

    ax.set_meridian_grid(30)
    ax.set_parallel_grid(30)
    ax.grid()
    canvas = FigureCanvasAgg(fig)
    fig.savefig('xxx-aea.png')

def test_moll():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Now make a simple example using the custom projection.

    import numpy as np

    fig = Figure(figsize=(6, 6))

#    ra = np.random.uniform(size=100, low=0, high=360)
#    dec = np.random.uniform(size=100, low=-90, high=90)
    ra = np.linspace(0, 360, 100)
    dec = np.linspace(-90, 90, 100)

    ra = np.random.uniform(size=1000, low=30, high=60)
    dec = np.random.uniform(size=1000, low=-50, high=50)

    ax = fig.add_subplot(111, projection="ast.mollweide")
    ax.plot(ra, dec, '*')
    ax.axhline(-20)
    ax.axvline(140)

    ra = np.random.uniform(size=1000, low=180, high=200)
    dec = np.random.uniform(size=1000, low=-50, high=50)

    ax.grid()
    canvas = FigureCanvasAgg(fig)
    fig.savefig('xxx-moll.png')
