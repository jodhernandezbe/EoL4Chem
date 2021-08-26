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
            city = '+'.join(str(row['CITY']).strip().split())
            state = str(row['STATE']).strip()
            zip = str(row['ZIP']).strip()[0:5]
            query = self.url + f'/search.php?street={address}&city={city}&state={state}&postalcode={zip}&format=jsonv2'
            result = requests.get(query)
            try:
                if result.status_code == 200:
                    address_city_state_zip.loc[idx, 'LATITUDE'] =\
                        float(result.json()[0]['lat'])
                    address_city_state_zip.loc[idx, 'LONGITUDE'] =\
                        float(result.json()[0]['lon'])
                else:
                    lat, long = self.search_excluding_address(city, state, zip)
                    address_city_state_zip.loc[idx, 'LATITUDE'] = lat
                    address_city_state_zip.loc[idx, 'LONGITUDE'] = long
            except (IndexError, JSONDecodeError):
                lat, long = self.search_excluding_address(city, state, zip)
                address_city_state_zip.loc[idx, 'LATITUDE'] = lat
                address_city_state_zip.loc[idx, 'LONGITUDE'] = long
        return address_city_state_zip

    
    def search_excluding_address(self, city, state, zip):
        options = [f'/search.php?city={city}&state={state}&postalcode={zip}&format=jsonv2',
                   f'/search.php?postalcode={zip}&format=jsonv2']
        for idx, option in enumerate(options):
            query = self.url + option
            result = requests.get(query)
            try:
                if result.status_code == 200:
                        lat = float(result.json()[0]['lat'])
                        long = float(result.json()[0]['lon'])
                else:
                    lat = None
                    long = None
            except (IndexError, JSONDecodeError):
                lat = None
                long = None
            if (idx == 0) and (lat):
                break
        return lat, long
