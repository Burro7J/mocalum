# -*- coding: utf-8 -*-
"""Setup file for PyConTurb

See README.md for how to use this file.
"""
from setuptools import setup

exec(open('pyconturb/_version.py').read())  # get version and release


setup(name='pyconturb',
      version=__version__,
      description='An open-source constrained turbulence generator',
      url='https://gitlab.windenergy.dtu.dk/rink/pyconturb',
      author='Jenni Rinker',
      author_email='rink@dtu.dk',
      license='MIT',
      packages=['pyconturb',  # top-level package
                'pyconturb.io',  # file io
                ],
      install_requires=['numpy',  # numberic arrays
                        'pandas',  # column-labelled arrays
                        'scipy',  # interpolating profile functions
                        ],
      zip_safe=False)
