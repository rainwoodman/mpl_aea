""" A pure python (numpy-based) version of key healpix functions.

    The ring scheme is implemented. 

    Depencency: numpy.

    It shall probably be self-hosted as an individual python package.

    Author: Yu Feng <rainwoodman@gmail.com>
"""

import numpy

def npix2nside(npix):
    # FIXME: this could be buggy for large npix
    nside2 = npix // 12
    nside = numpy.array(nside2 ** 0.5).astype('i8')
    return nside

def nside2npix(nside):
    return nside * nside * 12

def pix2radec(nside, pix):
    theta, phi = pix2ang(nside, pix)
    return numpy.degrees(phi), 90 - numpy.degrees(theta)

def radec2pix(nside, ra, dec):
    phi = numpy.radians(ra)
    theta = numpy.radians(90. - dec)
    return ang2pix(nside, theta, phi)

def ang2pix(nside, theta, phi):
    r"""Convert angle :math:`\theta` :math:`\phi` to pixel.

        This is translated from chealpix.c; but refer to Section 4.1 of
        http://adsabs.harvard.edu/abs/2005ApJ...622..759G
    """
    nside, theta, phi = numpy.lib.stride_tricks.broadcast_arrays(nside, theta, phi)
    
    def equatorial(nside, tt, z):
        t1 = nside * (0.5 + tt)
        t2 = nside * z * 0.75
        jp = (t1 - t2).astype('i8')
        jm = (t1 + t2).astype('i8')
        ir = nside + 1 + jp - jm # in {1, 2n + 1}
        kshift = 1 - (ir & 1) # kshift=1 if ir even, 0 odd 
 
        ip = (jp + jm - nside + kshift + 1) // 2 # in {0, 4n - 1}
        
        ip = ip % (4 * nside)
        return nside * (nside - 1) * 2 + (ir - 1) * 4 * nside + ip
        
    def polecaps(nside, tt, z, s):
        tp = tt - numpy.floor(tt)
        za = numpy.abs(z)
        tmp = nside * s / ((1 + za) / 3) ** 0.5
        mp = za > 0.99
        tmp[mp] = nside[mp] * (3 *(1-za[mp])) ** 0.5
        jp = (tp * tmp).astype('i8')
        jm = ((1 - tp) * tmp).astype('i8')
        ir = jp + jm + 1
        ip = (tt * ir).astype('i8')
        ip = ip % (4 * ir)

        r1 = 2 * ir * (ir - 1) 
        r2 = 2 * ir * (ir + 1)
 
        r = numpy.empty_like(r1)
        
        r[z > 0] = r1[z > 0] + ip[z > 0]
        r[z < 0] = 12 * nside[z < 0] * nside[z < 0] - r2[z < 0] + ip[z < 0]
        return r
    
    z = numpy.cos(theta)
    s = numpy.sin(theta)
    
    tt = (phi / (0.5 * numpy.pi) ) % 4 # in [0, 4]
    
    result = numpy.zeros(z.shape, dtype='i8')
    mask = (z < 2. / 3) & (z > -2. / 3)
  
    result[mask] = equatorial(nside[mask], tt[mask], z[mask])
    result[~mask] = polecaps(nside[~mask], tt[~mask], z[~mask], s[~mask])
    return result
    
def pix2ang(nside, pix):
    r"""Convert pixel to angle :math:`\theta` :math:`\phi`.

        nside and pix are broadcast with numpy rules.

        Returns: theta, phi

        This is translated from chealpix.c; but refer to Section 4.1 of
        http://adsabs.harvard.edu/abs/2005ApJ...622..759G
    """
    nside, pix = numpy.lib.stride_tricks.broadcast_arrays(nside, pix)
    
    ncap = nside * (nside - 1) * 2
    npix = 12 * nside * nside
    
    def northpole(pix, npix):
        iring = (1 + ((1 + 2 * pix) ** 0.5)).astype('i8') // 2
        iphi = (pix + 1) - 2 * iring * (iring - 1)
        z = 1.0 - (iring*iring) * 4. / npix
        phi = (iphi - 0.5) * 0.5 * numpy.pi / iring
        return z, phi
    
    def equatorial(pix, nside, npix, ncap):
        ip = pix - ncap
        iring = ip // (4 * nside) + nside
        iphi = ip % (4 * nside) + 1
        fodd = (((iring + nside) &1) + 1.) * 0.5
        z = (2 * nside - iring) * nside * 8.0 / npix
        phi = (iphi - fodd) * (0.5 * numpy.pi) / nside
        return z, phi
    
    def southpole(pix, npix):
        ip = npix - pix
        iring = (1 + ((2 * ip - 1)**0.5).astype('i8')) // 2
        iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1))
        z = -1 + (iring * iring) * 4. / npix
        phi = (iphi - 0.5 ) * 0.5 * numpy.pi / iring
        return z, phi
    
    mask1 = pix < ncap
    
    mask2 = (~mask1) & (pix < npix - ncap)
    mask3 = pix >= npix - ncap

    z = numpy.zeros(pix.shape, dtype='f8')
    phi = numpy.zeros(pix.shape, dtype='f8')
    z[mask1], phi[mask1] = northpole(pix[mask1], npix[mask1])
    z[mask2], phi[mask2] = equatorial(pix[mask2], nside[mask2], npix[mask2], ncap[mask2])
    z[mask3], phi[mask3] = southpole(pix[mask3], npix[mask3])
    return numpy.arccos(z), phi

def ang2xy(theta, phi):
    r"""Convert :math:`\theta` :math:`\phi` to :math:`x_s` :math:`y_s`.

        Returns: x, y

        Refer to Section 4.4 of http://adsabs.harvard.edu/abs/2005ApJ...622..759G
    """
    theta, phi = numpy.lib.stride_tricks.broadcast_arrays(theta, phi)
    z = numpy.cos(theta)
    x = numpy.empty(theta.shape, dtype='f8')
    y = numpy.empty(theta.shape, dtype='f8')
    def sigma(z):
        return numpy.sign(z) * (2 - (3 * (1- numpy.abs(z))) ** 0.5)
            
    def equatorial(z, phi):
        return phi, 3 * numpy.pi / 8 * z
    def polarcaps(z, phi):
        s = sigma(z)
        x = phi - (numpy.abs(s) - 1) * (phi % (0.5 * numpy.pi) - 0.25 * numpy.pi)
        y = 0.25 * numpy.pi * s
        return x, y
    
    mask = numpy.abs(z) < 2. / 3

    x[mask], y[mask] = equatorial(z[mask], phi[mask])
    x[~mask], y[~mask] = polarcaps(z[~mask], phi[~mask])
    return x, y

def xy2ang(x, y):
    r"""Convert :math:`x_s` :math:`y_s` to :math:`\theta` :math:`\phi`.
        
        Returns: theta, phi

        Refer to Section 4.4 of http://adsabs.harvard.edu/abs/2005ApJ...622..759G
    """
    x, y = numpy.lib.stride_tricks.broadcast_arrays(x, y)
    
    theta = numpy.empty(x.shape, dtype='f8')
    phi = numpy.empty(x.shape, dtype='f8')

    def equatorial(x, y):
        return numpy.arccos(8 * y / (3 * numpy.pi)), x
    
    def polarcaps(x, y):
        ya = numpy.abs(y)
        xt = x % (0.5 * numpy.pi)
        inv = ya - numpy.pi * 0.5
        inv[inv != 0] = 1 / inv[inv != 0]
        inv[inv == 0] = 0
        phi = x - (ya - numpy.pi * 0.25) * inv * (xt - 0.25 * numpy.pi)
        z = (1 - 1. / 3 * (2 - 4 * ya / numpy.pi)**2) * y / ya
        return numpy.arccos(z), phi
    
    mask = numpy.abs(y) < numpy.pi * 0.25
   
    theta[mask], phi[mask] = equatorial(x[mask], y[mask])
    theta[~mask], phi[~mask] = polarcaps(x[~mask], y[~mask])
    return theta, phi

def vertices(nside, pix, step=(0.5, 0.5, 0.5, 0.5)):
    r""" Calculate the vertices for pixels 
            3
         0     2
            1
        Returns: theta, phi
            for each (nside, pix) pair, a four-vector of theta, and
            a four-vector of phi is returned, corresponding to
            the theta, phi of each vertex of the pixel boundary.
            (left, bottom, right, top)
    """
    nside, pix = numpy.lib.stride_tricks.broadcast_arrays(nside, pix)
    x = numpy.zeros(nside.shape, dtype=('f8', 4))
    y = numpy.zeros(nside.shape, dtype=('f8', 4))
    theta, phi = pix2ang(nside, pix)
    xc, yc = ang2xy(theta, phi)
    xstep = numpy.pi / (2 * nside)
    ystep = numpy.pi / (2 * nside)
    x[..., 0] = xc - step[0] * xstep
    y[..., 0] = yc
    x[..., 1] = xc
    y[..., 1] = yc + step[1] * ystep
    x[..., 2] = xc + step[2] * xstep
    y[..., 2] = yc
    x[..., 3] = xc
    y[..., 3] = yc - step[3] * ystep

    theta, phi = xy2ang(x, y)
    return theta, phi

def histogrammap(ra, dec, weights=None, nside=32, perarea=False, range=None):
    if range is not None:
        (ra1, ra2), (dec1, dec2) = range
        m  = (ra >= ra1)& (ra <= ra2)
        m &= (dec >= dec1)& (dec <= dec2)
        ra = ra[m]
        dec = dec[m]
        if weights is not None:
            weights = weights[m]

    ipix = ang2pix(nside, numpy.radians(90-dec), numpy.radians(ra))
    npix = nside2npix(nside)
    if perarea:
        npix = nside2npix(nside)
        sky = 360. ** 2 / numpy.pi
        area = 1. * (sky / npix)
    else:
        area = 1

    if weights is not None:
        w = numpy.bincount(ipix, weights=weights, minlength=npix)
        N = numpy.bincount(ipix, minlength=npix)
        w = w / area
        N = N / area
        return w, N
    else:
        w = 1.0 * numpy.bincount(ipix, minlength=npix)
        return w / area


