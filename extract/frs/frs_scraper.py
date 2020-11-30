# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Importing libraries
import requests
import zipfile
import sys, os, io

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from common import config

class FRS_Scrapper:

    def __init__(self):
        self._config = config()['web_sites']['FRS']
        self._dir_path = os.path.dirname(os.path.realpath(__file__)) # Working Directory
        #self._dir_path = os.getcwd() # if you are working on Jupyter Notebook


    def _visit(self):
        _url = self._config['url']
        self._r_file = requests.get(_url)


    def extracting_zip(self):
        self._visit()
        with zipfile.ZipFile(io.BytesIO(self._r_file.content)) as z:
            z.extractall(self._dir_path + '/csv')

if __name__ == '__main__':

    FRS_Scrapper = FRS_Scrapper()
    FRS_Scrapper.extracting_zip()
