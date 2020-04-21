# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(

    name='mocalum',  # Required
    version='0.1.0',  # Required
    description='Monte Carlo based Lidar Uncertainty Model (MOCALUM): Python libpackage for lidar uncertainty assessment',  # Optional
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',  # Optional
    author='Nikola Vasiljevic, Andrea Vignaroli',  # Optional
    author_email='niva@dtu.dk',  # Optional
    classifiers=[  # Optional
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
    ],

    packages=['mocalum'],  # Required
    python_requires='>=3.6',
    install_requires=[
                      'numpy',
                      'pandas',
                      'xarray',
                      'netCDF4',
                      'matplotlib',
                      'tqdm',
                      'scipy',
                      'pyconturb @ git+https://gitlab.windenergy.dtu.dk/pyconturb/pyconturb.git@master'
                      ]
)