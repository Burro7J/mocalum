"""This module contains dictionaries of metadata information
related to MC_lum xarray datasets.
"""

# Dimensions metadata
DIMS = {}

basic_el = {'name':{'long_name':'',
                    'standard_name':'',
                    'units':'',
                    'URI':''}}

# Coordinates metadata
DIMS = {'rng':{'long_name':'Range of measurements',
                 'standard_name':'range',
                 'units':'m',
                 'URI':''},
          'cc':{'long_name':'Correletation coefficient',
                'standard_name':'correlation_coefficient',
                'units':'',
                'URI':''},
          'wdir':{'long_name':'Wind direction',
                  'standard_name':'wind_from_direction',
                  'units':'degree',
                  'URI':''},
          'sectrsz':{'long_name':'PPI sector size',
                     'standard_name':'scanned_sector_size',
                     'units':'degree',
                     'URI':''}
         }

VARS = {'ws':{'long_name':'Wind speed',
                 'standard_name':'wind_speed',
                 'units':'m.s-1',
                 'URI':''},
        'u_ws':{'long_name':'Wind speed uncertainty',
                'standard_name':'wind_speed_uncertainty',
                'units':'m.s-1',
                'URI':''},
        'wdir':{'long_name':'Wind direction',
                'standard_name':'wind_from_direction',
                'units':'degree',
                'URI':''}
         }         