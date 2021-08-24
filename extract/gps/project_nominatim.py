#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import requests
from json.decoder import JSONDecodeError

from extract.common import config


class NOMINATIM_API:

    def __init__(self):
        self._config = config()['web_sites']['NOMINATIM']
        self.url = self._config['url']

    def request_coordinates(self, address_city_state_zip):
        address_city_state_zip['LONGITUDE'] = None
        address_city_state_zip['LATITUDE'] = None
        for idx, row in address_city_state_zip.iterrows():
            address = '+'.join(str(row['ADDRESS']).strip().split())
            city = str(row['CITY']).strip()
            state = str(row['STATE']).strip()
            zip = str(row['ZIP']).strip()
            search_parameter = f'{address},+{city},+{state},+{zip}'
            query = self.url + f'/search?q={search_parameter}&format=json&addressdetails=1&limit=1&polygon_svg=1'
            result = requests.get(query)
            try:
                if result.status_code == 200:
                    address_city_state_zip.loc[idx, 'LATITUDE'] =\
                        float(result.json()[0]['lat'])
                    address_city_state_zip.loc[idx, 'LONGITUDE'] =\
                        float(result.json()[0]['lon'])
                else:
                    address_city_state_zip.loc[idx, 'LATITUDE'] = None
                    address_city_state_zip.loc[idx, 'LONGITUDE'] = None
            except (IndexError, JSONDecodeError):
                address_city_state_zip.loc[idx, 'LATITUDE'] = None
                address_city_state_zip.loc[idx, 'LONGITUDE'] = None
        return address_city_state_zip
