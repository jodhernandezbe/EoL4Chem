# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Importing libraries
import requests
import zipfile
import os, io

from extract.common import config

class FRS_Scrapper:

    def __init__(self):
        self._config = config()['web_sites']['FRS']
        self._dir_path = os.path.dirname(os.path.realpath(__file__)) # Working Directory
        #self._dir_path = os.getcwd() # if you are working on Jupyter Notebook


    def _visit(self):
        _url = self._config['url']
        self._r_file = requests.get(_url)


    def extracting_zip(self):
        list_to_extract = ['NATIONAL_ENVIRONMENTAL_INTEREST_FILE.CSV',
                            'NATIONAL_FACILITY_FILE.CSV',
                            'NATIONAL_ALTERNATIVE_NAME_FILE.CSV',
                            'NATIONAL_SIC_FILE.CSV',
                            'NATIONAL_NAICS_FILE.CSV']

        self._visit()
        print('Here')
        with zipfile.ZipFile(io.BytesIO(self._r_file.content)) as z:
            for filename in list_to_extract:
                print(filename)
                z.extract(filename, self._dir_path + '/csv')

if __name__ == '__main__':

    FRS_Scrapper = FRS_Scrapper()
    FRS_Scrapper.extracting_zip()
