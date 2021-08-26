#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import requests
import numpy as np
import os
import pandas as pd

from extract.common import config


class OSRM_API:

    def __init__(self):
        self._config = config()['web_sites']['OSRM']
        self.url = self._config['url']
        self._dir_path = os.path.dirname(os.path.realpath(__file__))

    def sea_port_lists(self, Lat, Long):
        Ports = pd.read_csv(self._dir_path + '/../../ancillary/others/Important_sea_ports_in_the_USA.csv')
        # Only taking the Container and Tonnage ports
        Ports = Ports.loc[Ports['Dry only'] != 'Yes']
        Ports.reset_index(drop=True, inplace=True)
        Ports['Distance'] = Ports.apply(lambda x:
                                        self.harvesine_formula(x['Latitude'],
                                                               x['Longitude'],
                                                               Lat, Long),
                                        axis=1)
        Ports = Ports.loc[Ports.Distance.idxmin()]
        return Ports['Latitude'], Ports['Longitude']

    def maritime_transport(self, Lat_1, Long_1, Lat_2, Long_2, N_Times):
        Lat_port_1, Lon_port_1 = self.sea_port_lists(Lat_1, Long_1)
        Lat_port_2, Lon_port_2 = self.sea_port_lists(Lat_2, Long_2)
        distance = self.harvesine_formula(Lat_port_1, Lon_port_1,
                                          Lat_port_2, Lon_port_2)
        Maritime_distance = distance
        Land_1, _ = self.request_directions(Lat_1, Long_1,
                                            Lat_port_1, Lon_port_1,
                                            Times=N_Times)
        distance = distance + Land_1
        Lan_2, _ = self.request_directions(Lat_port_2, Lon_port_2,
                                           Lat_2, Long_2,
                                           Times=N_Times)
        distance = distance + Lan_2
        Maritime_distance = round(Maritime_distance/distance, 4)
        return distance, Maritime_distance

    def request_directions(self, Lat_1, Long_1, Lat_2, Long_2, Times=0):
        coordinate_1 = f'{Long_1},{Lat_1}'
        coordinate_2 = f'{Long_2},{Lat_2}'
        service = 'route'
        version = 'v1'
        profile = 'driving'
        N_Times = Times
        if N_Times < 5:
            try:
                query = self.url + f'/{service}/{version}/{profile}/{coordinate_1};{coordinate_2}?overview=false'
                result = requests.get(query)
                if result.status_code == 200:
                    distance = round(result.json()['routes'][0]
                                     ['legs'][0]['distance']/1000, 4)
                    Maritime_distance = 0.0
                else:
                    N_Times += 1
                    distance, Maritime_distance = self.maritime_transport(
                                                     Lat_1, Long_1,
                                                     Lat_2, Long_2, N_Times)
            except IndexError:
                N_Times += 1
                distance, Maritime_distance = self.maritime_transport(
                                                    Lat_1, Long_1,
                                                    Lat_2, Long_2, N_Times)
        else:
            distance = self.harvesine_formula(Lat_1, Long_1, Lat_2, Long_2)
            Maritime_distance = None
        return [distance, Maritime_distance]

    def harvesine_formula(self, Lat_1, Long_1, Lat_2, Long_2):
        Average_earth_radius = 6371
        Lat_1, Long_1, Lat_2, Long_2 = map(np.radians, [Lat_1, Long_1, Lat_2, Long_2])
        dlon = Long_2 - Long_1
        dlat = Lat_2 - Lat_1
        a = np.sin(dlat/2)**2 + np.cos(Lat_1) * np.cos(Lat_2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = round(c * Average_earth_radius, 4) 
        return distance
