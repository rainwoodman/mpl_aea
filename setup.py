from setuptools import setup
import os

def find_version(path):
    import re
    # path shall be a plain ascii text file.
    s = open(path, 'rt').read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              s, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Version not found")

setup(name="mpl_aea", version=find_version("mpl_aea/version.py"),
      author="Yu Feng",
      author_email="rainwoodman@gmail.com",
      description="AEA Projection for Matplotlib derived from skywalker ",
      url="http://github.com/rainwoodman/mpl_aea",
      zip_safe=True,
      package_dir = {'mpl_aea': 'mpl_aea'},
      license='GPL3',
      packages = [
        'mpl_aea'
      ],
      test_suite='mpl_aea.tests.test_all',
      install_requires=['numpy', 'matplotlib'],
)

