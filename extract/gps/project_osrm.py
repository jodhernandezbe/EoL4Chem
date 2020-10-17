#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import requests
import math
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from common import config


class OSRM_API:

    def __init__(self):
        self._config = config()['web_sites']['OSRM']
        self.url = self._config['url']
        self._dir_path = os.path.dirname(os.path.realpath(__file__))

    def sea_port_lists(self, Lat, Long):
        Ports = pd.read_csv(self._dir_path + '/../../ancillary/others/Important_sea_ports_in_the_USA.csv')
        # Only taking the Container and Tonnage ports
        Ports = Ports.loc[Ports['Dry only'] != 'Yes']
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
        phi1, phi2 = math.radians(Lat_1), math.radians(Lat_2)
        dphi = math.radians(Lat_2 - Lat_1)
        d = math.radians(Long_2 - Long_1)
        a = math.sin(dphi/2)**2+math.cos(phi1)*math.cos(phi2)*math.sin(d/2)**2
        distance = round(2*Average_earth_radius*math.atan2(math.sqrt(a),
                                                           math.sqrt(1 - a)),
                         4)
        return distance
