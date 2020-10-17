# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Importing libraries

import os, sys, io, shutil, errno, stat
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import zipfile
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
import chardet, codecs
import re
import time
import datetime
import csv
import math
import datetime
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
from common import config

class RCRAInfo_Scrapper:

    def __init__(self, Year):

        # Specification could be found in:
        # https://rcrainfopreprod.epa.gov/rcrainfo-help/application/publicHelp/index.htm
        # Date: 3/17/2020

        # List of tables:
        ### BR_REPORTING_2001
        ### BR_REPORTING_2003
        ### BR_REPORTING_2005
        ### BR_REPORTING_2007
        ### BR_REPORTING_2009
        ### BR_REPORTING_2011
        ### BR_REPORTING_2013
        ### BR_REPORTING_2015
        ### BR_REPORTING_2017

        self._dir_path = os.path.dirname(os.path.realpath(__file__)) # Working Directory
        #self._dir_path = os.getcwd() # if you are working on Jupyter Notebook
        self.Year = Year
        self._config = config()['web_sites']['RCRAInfo']
        self._queries = self._config['queries']
        self._url = self._config['url'] # Uniform Resource Locator (URL) of RCRAInfo Database


    def visit(self):
        regex = re.compile(r'(.+).zip\s?\(\d+.?\d*\s?[a-zA-Z]{2,}\)')
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-notifications')
        options.add_argument('--no-sandbox')
        options.add_argument('--verbose')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-software-rasterizer')
        options.add_argument("--log-level=3")
        options.add_argument('--hide-scrollbars')
        prefs = {'download.default_directory' : self._dir_path + '/ZIP',
                'download.prompt_for_download': False,
                'download.directory_upgrade': True,
                'safebrowsing_for_trusted_sources_enabled': False,
                'safebrowsing.enabled': False}
        options.add_experimental_option('prefs', prefs)
        browser = webdriver.Chrome(ChromeDriverManager().install(), options = options)
        #browser.maximize_window()
        browser.set_page_load_timeout(180)
        browser.get(self._url)
        time.sleep(30)
        Table_of_tables = browser.find_element_by_xpath(self._queries['Table_of_tables'])
        rows = Table_of_tables.find_elements_by_css_selector('tr')[1:] # Excluding header
        # Extracting zip files for Biennial Report Tables
        Links = {}
        for row in rows:
            loop = 'YES'
            while loop == 'YES':
                try:
                    Table_name = re.search(regex, row.find_elements_by_css_selector('td')[3].text).group(1)
                    Link = row.find_elements_by_css_selector('td')[3].find_elements_by_css_selector('a')[0].get_attribute('href')
                    Links.update({Table_name:Link})
                    loop = 'NO'
                except AttributeError:
                    loop = 'YES'
                    now = datetime.datetime.now()
                    print('AttributeError occurred with selenium due to not appropriate charging of website.\nHour: {}:{}:{}'.format(now.hour,now.minute,now.second))
        # Download the desired zip
        for Year in self.Year:
            browser.get(Links['BR_REPORTING_' + Year])
            condition = os.path.exists(self._dir_path + '/ZIP/BR_REPORTING_' + Year + '.zip')
            while condition is False:
                condition = os.path.exists(self._dir_path + '/ZIP/BR_REPORTING_' + Year + '.zip')
            time.sleep(5)
            self._extracting_files('BR_REPORTING_' + Year)
        browser.quit()


    def _extracting_files(self, ZIP):
        PATH_UNZIPPING = self._dir_path + '/TXT/' + ZIP
        PATH_ZIP = self._dir_path + '/ZIP/' + ZIP + '.zip'
        try:
            os.mkdir(PATH_UNZIPPING)
        except OSError:
            print('Creation of the directory {} failed'.format(PATH_UNZIPPING.replace('\\','/')))
        else:
            print('Successfully created the directory {}'.format(PATH_UNZIPPING.replace('\\','/')))
        with zipfile.ZipFile(PATH_ZIP) as z:
            z.extractall(PATH_UNZIPPING)
        os.remove(PATH_ZIP)


    def organizing_files(self):
        for Year in self.Year:
            Table = 'BR_REPORTING_' + Year
            PATH_CSV = self._dir_path + '/CSV'
            RCRA_TABLE_SPECIFICATIONS = pd.read_csv(self._dir_path + '/../../Ancillary/RCRAInfo/BR_REPORTING_SPECIFICATIONS.csv')
            TABLE_COLUMNS = RCRA_TABLE_SPECIFICATIONS['English Name']
            LENGHT_INFO = RCRA_TABLE_SPECIFICATIONS['Field Length'].astype(int)
            # Checking files unzipped
            Files = [file for file in os.listdir(self._dir_path + '/TXT/' + Table) if ((file.startswith(Table)) & file.endswith('.txt'))]
            Files.sort()
            # Concatenating files by year
            df_br = pd.DataFrame()
            PATH_TXT = self._dir_path + '/TXT/' + Table
            for File in Files:
                print('Processing file {}'.format(File))
                df = pd.read_fwf(PATH_TXT + '/' + File, widths = LENGHT_INFO, \
                             header = None, names = TABLE_COLUMNS)
                df['Reporting Cycle Year'] = Year
                df_br = pd.concat([df_br, df],  ignore_index = True,
                                      sort = True, axis = 0)
            PATH_DIRECTORY = PATH_CSV + '/' + Table + '.csv'
            df_br.to_csv(PATH_DIRECTORY, sep = ',',
                                            index = False)
            shutil.rmtree(PATH_TXT)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('Option',
                        help = 'What do you want to do:\
                        [A] Extract information from RCRAInfo.\
                        [B] Organize files for csv.',
                        type = str)

    parser.add_argument('-Y', '--Year', nargs = '+',
                        help = 'What Bienniarl report do you want?. Currently up to 2017',
                        required = False)

    args = parser.parse_args()

    if args.Option == 'A':
        Scrapper = RCRAInfo_Scrapper(args.Year)
        Scrapper.visit()
    elif args.Option == 'B':
        Scrapper = RCRAInfo_Scrapper(args.Year)
        Scrapper.organizing_files()
