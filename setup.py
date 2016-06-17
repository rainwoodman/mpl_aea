from numpy.distutils.core import setup, Extension
from numpy import get_include

setup(name="mpl_aea", version="0.0.1",
      author="Yu Feng",
      author_email="rainwoodman@gmail.com",
      description="AEA Projection for Matplotlib derived from skywalker ",
      url="http://github.com/rainwoodman/mpl_aea",
      zip_safe=True,
      package_dir = {'mpl_aea': 'mpl_aea'},
      packages = [
        'mpl_aea'
      ],
      requires=['numpy', 'matplotlib'],
)

