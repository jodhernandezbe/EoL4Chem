# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from fake_useragent import UserAgent
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import time
import sys
import re

from ancillary.normalizing_naics.normalizing import normalizing_naics

def organizing_NPV(value):
    '''
    Function to support on the organization of the national aggregated producion volume from CDR
    '''

    list_values = [int(float(val)) for val in re.sub(r'[\<\>\-\,lb]', '', value).strip().split()]
    return max(list_values)


def organizing_national_production_volume(cdr_path, saving_path, SRS):
    '''
    Function to organize the national aggregated producion volume from CDR
    '''

    National_Production_Volume = pd.read_csv(cdr_path + '/National_Aggregate_Production_Volume.csv',
                                    low_memory=True)
    Columns_PPV = [col for col in National_Production_Volume.columns if 'PPV' in col]
    Nat_PPV = pd.DataFrame()
    for col in Columns_PPV:
        Nat_PPV_aux = National_Production_Volume[['STRIPPED_CHEMICAL_ID_NUMBER', col]]
        Nat_PPV_aux['YEAR'] = re.search(r'.+(\d{4})', col).group(1)
        Nat_PPV_aux.rename(columns={col: 'NAT_AGG_PPV', 'STRIPPED_CHEMICAL_ID_NUMBER': 'ID'}, inplace=True)
        Nat_PPV_aux['UNIT'] = 'lb'
        Nat_PPV_aux = Nat_PPV_aux.loc[Nat_PPV_aux['NAT_AGG_PPV'] != 'Withheld']
        Nat_PPV_aux['NAT_AGG_PPV'] = Nat_PPV_aux['NAT_AGG_PPV'].apply(lambda x: organizing_NPV(x))
        Nat_PPV = pd.concat([Nat_PPV, Nat_PPV_aux], sort=False, axis=0,
                                ignore_index=True)
        del Nat_PPV_aux
    Nat_PPV = pd.merge(Nat_PPV, SRS, how='inner', on='ID')
    Nat_PPV.to_csv(saving_path + '/National_Production_Volume.csv',
                   sep=',', index=False)


def searching_facilities(cdr_path, FRS):
    '''
    Function to search for the TSCA reporting facilities in the FRS
    '''

    Facilities = pd.DataFrame()
    list_files = ['Consumer_and_Commercial_Use', 'Industrial_Processing_and_Use',
                  'Manufacturing_Information']
    for file in list_files:
        Facilities_aux = pd.read_csv(cdr_path + '/' + file + '.csv',
                            low_memory=False,
                            usecols=['SITE_NAME', 'SITE_ADDRESS',
                            	     'SITE_CITY', 'SITE_COUNTY', 'SITE_STATE',
                                     'SITE_ZIP'])
        Facilities = pd.concat([Facilities, Facilities_aux],
                                axis=0,
                                ignore_index=True)
        Facilities.drop_duplicates(keep='first', inplace=True)
        del Facilities_aux
    Facilities = pd.merge(FRS, Facilities, how='inner',
                                left_on='ALTERNATIVE_NAME', right_on='SITE_NAME')
    Facilities.drop_duplicates(keep='first', inplace=True)
    Facilities.drop(inplace=True, columns=['ALTERNATIVE_NAME'])
    return Facilities


def searching_substances(cdr_path, SRS):
    '''
    Function to search on SRS the information for the TSCA reporting substances
    '''

    Substances = pd.DataFrame()
    list_files = ['Consumer_and_Commercial_Use', 'Industrial_Processing_and_Use',
                  'Manufacturing_Information']
    for file in list_files:
        Substances_aux = pd.read_csv(cdr_path + '/' + file + '.csv',
                            low_memory=False,
                            usecols=['STRIPPED_CHEMICAL_ID_NUMBER'])
        Substances = pd.concat([Substances, Substances_aux],
                                axis=0,
                                ignore_index=True)
        Substances.drop_duplicates(keep='first', inplace=True)
        del Substances_aux
    SRS.rename(columns={'ID': 'STRIPPED_CHEMICAL_ID_NUMBER'}, inplace=True)
    Substances = pd.merge(Substances, SRS, how='inner', on= 'STRIPPED_CHEMICAL_ID_NUMBER')
    return Substances


def organizing_information(cdr_path, Facilities, Substances, year, saving_path):
    '''
    Function to organize the information from CDR into an unique file
    '''

    Removing = ['Withheld', 'CBI']
    Columns = {'Consumer_and_Commercial_Use': ['STRIPPED_CHEMICAL_ID_NUMBER',
                                            'PHYSICAL_FORMS', 'SITE_NAME',
                                            'SITE_NAME', 'SITE_ADDRESS',
                                            'SITE_CITY', 'SITE_COUNTY',
                                            'SITE_STATE', 'SITE_ZIP',
                                            'PRODUCT_CATEGORY', 'CHILDREN_PRODUCTS',
                                            'CONS_COMM_OPTION', 'C_PCT_PROD_VOLUME',
                                            'C_MAX_CONCENTRATION', 'C_NUM_WORKERS'],
                'Industrial_Processing_and_Use': ['STRIPPED_CHEMICAL_ID_NUMBER',
                                            'PHYSICAL_FORMS', 'MAX_CONCENTRATION',
                                            'SITE_NAME',
                                            'SITE_NAME', 'SITE_ADDRESS',
                                            'SITE_CITY', 'SITE_COUNTY',
                                            'SITE_STATE', 'SITE_ZIP',
                                            'TYPE_PROCESS_USE', 'SECTOR',
                                            'FUNCTION_CATEGORY', 'I_PCT_PROD_VOLUME',
                                            'NUM_SITES', 'I_NUM_WORKERS'],
                'Manufacturing_Information': ['STRIPPED_CHEMICAL_ID_NUMBER',
                                            'SITE_NAME', 'SITE_ADDRESS',
                                            'SITE_CITY', 'SITE_COUNTY',
                                            'SITE_STATE', 'SITE_ZIP',
                                            'NUM_WORKERS', 'MAX_CONCENTRATION',
                                            'PHYSICAL_FORMS']}
    CDR = pd.DataFrame()
    for file, cols in Columns.items():
        CDR_aux = pd.read_csv(cdr_path + '/' + file + '.csv',
                            low_memory=False,
                            usecols=cols)
        if file == 'Consumer_and_Commercial_Use':
            CDR_aux.rename(columns={'C_PCT_PROD_VOLUME': 'PCT_PROD_VOLUME',
                                    'C_MAX_CONCENTRATION': 'MAX_CONCENTRATION',
                                    'C_NUM_WORKERS': 'NUM_WORKERS',
                                    'CONS_COMM_OPTION': 'OPTION'},
                        inplace=True)
            CDR_aux = CDR_aux.loc[pd.notnull(CDR_aux['OPTION'])]
            CDR_aux.loc[CDR_aux['OPTION'] == 'Both', 'OPTION'] = 'Consumer and Commercial'
        elif file == 'Industrial_Processing_and_Use':
            CDR_aux.rename(columns={'I_PCT_PROD_VOLUME': 'PCT_PROD_VOLUME',
                                    'I_NUM_WORKERS': 'NUM_WORKERS'},
                        inplace=True)
            CDR_aux['OPTION'] = 'Industrial'
            CDR_aux = CDR_aux.loc[pd.notnull(CDR_aux['TYPE_PROCESS_USE'])]
            CDR_aux['SECTOR'] = CDR_aux['SECTOR'].where(pd.notnull(CDR_aux['SECTOR']), None)
            CDR_aux['SECTOR'] = CDR_aux['SECTOR'].apply(lambda x: re.sub(r'[\.]','', x).capitalize() if x else x)
        elif file == 'Manufacturing_Information':
            CDR_aux['OPTION'] = 'Manufacturing'
        CDR_aux.replace(to_replace='NKRA', value='Not known or reasonably ascertainable', inplace=True)
        CDR = pd.concat([CDR, CDR_aux], sort=False, axis=0,
                                ignore_index=True)
        del CDR_aux
    # Adding SRS information
    CDR = pd.merge(CDR, Substances, on='STRIPPED_CHEMICAL_ID_NUMBER', how='inner')
    CDR.rename(columns={'Internal Tracking Number': 'INTERNAL_TRACKING_NUMBER',
                        'Substance Name': 'SUBSTANCE_NAME'},
                inplace=True)
    # Adding FRS information
    CDR = pd.merge(CDR, Facilities, on=['SITE_NAME', 'SITE_ADDRESS',
                                        'SITE_CITY', 'SITE_COUNTY', 'SITE_STATE',
                                        'SITE_ZIP'],
                    how='inner')
    CDR.sort_values(by=['SITE_NAME', 'STRIPPED_CHEMICAL_ID_NUMBER'], inplace=True)
    # Substances by facility
    Substances_by_facility = CDR[['INTERNAL_TRACKING_NUMBER',
                                    'STRIPPED_CHEMICAL_ID_NUMBER',
                                    'SUBSTANCE_NAME', 'REGISTRY_ID',
                                    'SITE_NAME', 'SITE_ADDRESS',
                                    'SITE_CITY', 'SITE_COUNTY',
                                    'SITE_STATE', 'SITE_ZIP']]
    Substances_by_facility.drop_duplicates(keep='first', inplace=True)
    # Removing
    CDR = CDR.where(~CDR.isin(Removing), None)
    # Organizing maximum concentration
    max_dictionary = {'90% +': 'At least 90% by weight',
                    '60% - < 90%': 'At least 60 but less than 90% by weight',
                    '30% - < 60%': 'At least 30 but less than 60% by weight',
                    '1% - < 30%': 'At least 1 but less than 30% by weight',
                    '< 1%': 'Less than 1% by weight'}
    CDR['MAX_CONCENTRATION'] = CDR['MAX_CONCENTRATION'].where(pd.notnull(CDR['MAX_CONCENTRATION']), None)
    CDR['MAX_CONCENTRATION'] = CDR['MAX_CONCENTRATION'].apply(lambda x: (max_dictionary[x] if x in max_dictionary.keys() else x.strip().capitalize()) if x else x)
    # Organizing workers
    worker_dictionary = {'< 10': 'Fewer than 10 workers',
            '24-Oct': 'At least 10 but fewer than 25 workers',
            '10 - 24': 'At least 10 but fewer than 25 workers',
            '25 - 49': 'At least 25 but fewer than 50 workers',
            '50 - 99': 'At least 50 but fewer than 100 workers',
            '100 - 499': 'At least 100 but fewer than 500 workers',
            '500 - 999': 'At least 500 but fewer than 1,000 workers',
            '1000 - 9999': 'At least 1,000 but fewer than 10,000 workers',
            '10,000+': 'At least 10,000 workers'}
    CDR['NUM_WORKERS'] = CDR['NUM_WORKERS'].where(pd.notnull(CDR['NUM_WORKERS']), None)
    CDR['NUM_WORKERS'] = CDR['NUM_WORKERS'].apply(lambda x: (worker_dictionary[x] if x in worker_dictionary.keys() else x.strip().capitalize()) if x else x)
    # Organizing sites
    site_dictionary = {'< 10': 'Fewer than 10 sites',
                    '10 - 24': 'At least 10 but fewer than 25 sites',
                    '25 - 99': 'At least 25 but fewer than 100 sites',
                    '100 - 249': 'At least 100 but fewer than 250 sites',
                    '250 - 999': 'At least 250 but fewer than 1,000 sites',
                    '1000 - 9999': 'At least 1,000 but fewer than 10,000 sites',
                    '10,000+': 'At least 10,000 sites'}
    CDR['NUM_SITES'] = CDR['NUM_SITES'].where(pd.notnull(CDR['NUM_SITES']), None)
    CDR['NUM_SITES'] = CDR['NUM_SITES'].apply(lambda x: (site_dictionary[x] if x in site_dictionary.keys() else x.strip().capitalize()) if x else x)
    # Organizing percentages
    func = lambda x: (x.strip().capitalize() if not re.search(r'[0-9]+\.?[0-9]*', str(x)) else (float(x)/10 if float(x) > 100 else float(x))) if x else x
    CDR['PCT_PROD_VOLUME'] = CDR['PCT_PROD_VOLUME'].where(pd.notnull(CDR['PCT_PROD_VOLUME']), None)
    CDR['PCT_PROD_VOLUME'] = CDR['PCT_PROD_VOLUME'].apply(lambda x: func(x))
    CDR['YEAR'] = year
    # Saving CDR Info
    Path_f = saving_path + '/Uses_information.csv'
    if os.path.exists(Path_f):
        df_f = pd.read_csv(Path_f, low_memory=False, dtype={'YEAR': int})
        df_f = pd.concat([df_f, CDR], sort=False, axis=0,
                                ignore_index=True)
        df_f.drop_duplicates(keep='first', inplace=True)
        df_f.sort_values(by=['SITE_NAME', 'STRIPPED_CHEMICAL_ID_NUMBER'], inplace=True)
        df_f.to_csv(Path_f, sep=',', index=False)
    else:
        CDR.to_csv(Path_f, sep=',', index=False)
    # Saving susbtances by facility
    Path_c = saving_path + '/Substances_by_facilities.csv'
    if os.path.exists(Path_c):
        df_c = pd.read_csv(Path_c, low_memory=False)
        df_c = pd.concat([df_c, Substances_by_facility], sort=False, axis=0,
                                ignore_index=True)
        df_c.drop_duplicates(keep='first', inplace=True)
        df_c.sort_values(by=['REGISTRY_ID', 'STRIPPED_CHEMICAL_ID_NUMBER'], inplace=True)
        df_c.to_csv(Path_c, sep=',', index=False)
    else:
        Substances_by_facility.to_csv(Path_c, sep=',', index=False)


def starting_browser():
    '''
    Function to start the web browser
    '''

    options = Options()
    ua = UserAgent()
    options.add_argument('--incognito')
    options.add_argument('--log-level=OFF')
    userAgent = ua.random
    options.headless = True
    options.add_argument(f'user-agent={userAgent}')
    browser = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    #browser.maximize_window()
    return browser


def searching_naics_code_using_web_scrapping(IS_list, browser):
    '''
    Function to search automatically on Google the NAICS code
    '''

    browser.get('https://www.google.com/')
    time.sleep(1.0)
    IS_text = ' '.join(IS_list)
    NAICS = None
    if re.search(r'naics[a-zA-Z\s\:\-\.]*([0-9]{2,6})', IS_text):
        NAICS = re.findall(r'naics[a-zA-Z\s\:\-\.]*([0-9]{2,6})', IS_text)[0]
    else:
        delay = 20
        try:
            searching_bar = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, '//input[@class="gLFyf gsfi"]')))
            searching_bar.send_keys('NAICS code for ' + IS_text) # searching bar
            searching_bar.submit()
            try:
                found_text = browser.find_element_by_xpath('//div[@class="kp-blk c2xzTb Wnoohf OJXvsb"]').text # getting text
                if re.search(r'NAICS[a-zA-Z\s\:\-\.]*([0-9]{2,6})', found_text):
                    NAICS = re.findall(r'NAICS[a-zA-Z\s\:\-\.]*([0-9]{2,6})', found_text)[0]
            except NoSuchElementException:
                try:
                    table_text = browser.find_element_by_xpath('//div[@class="webanswers-webanswers_table__webanswers-table"]').text
                    table_text = re.sub(r'20[0-9]{2} NAICS', '', table_text)
                    if re.search(r'([0-9]{2,6})', table_text):
                        NAICS = re.findall(r'([0-9]{2,6})', table_text)[0]
                        print(NAICS)
                except NoSuchElementException:
                    pass
        except TimeoutException:
            print('It took a long time to load')
            browser.close()
            browser = starting_browser()
            NAICS, browser = searching_naics_code_using_web_scrapping(IS_list, browser)
    browser.back()
    return (NAICS, browser)


def searching_naics_codes_for_downstream(cdr_path, saving_path):
    '''
    Function to find the NAICS code for the downstream industrial uses resported to CDR
    '''

    # https://www.epa.gov/sites/production/files/documents/replacingnaicswithis.pdf
    IS_NAICS = pd.read_csv(cdr_path + '/Industrial_Sector_(IS)_Codes_to_Replace_Five-Digit_NAICS.csv',
                low_memory=False)
    IS_NAICS['Title'] = IS_NAICS['Title'].apply(lambda x: re.sub(r'[\-\(\)]', '', x).capitalize().strip())
    # Reading industrial sectors
    CDR_INFO = pd.read_csv(saving_path + '/Uses_information.csv', low_memory=False)
    CDR = CDR_INFO['SECTOR'].to_frame(name='SECTOR')
    CDR = CDR.loc[pd.notnull(CDR['SECTOR'])]
    CDR.drop_duplicates(keep='first', inplace=True)
    CDR['SECTOR_SEARCH'] = CDR['SECTOR'].apply(lambda x: re.sub(r'[\-\(\)]', '', x).strip().lower())
    # Finding NAICS in IS_NAICS
    CDR = pd.merge(CDR, IS_NAICS, left_on='SECTOR', right_on='Title', how='left')
    CDR.drop(columns=['Title'], inplace=True)
    # Separating
    CDR_is = CDR.loc[pd.notnull(CDR['Industrial Sector Code'])]
    CDR_no_is = CDR.loc[~pd.notnull(CDR['Industrial Sector Code'])]
    del CDR
    # Tokenizing NLTK and removing punctuation
    CDR_no_is['SECTOR_SEARCH'] = CDR_no_is['SECTOR_SEARCH'].apply(lambda x: [word for word in word_tokenize(x) if word not in string.punctuation])
    # Lemmatizing
    lemmatizer = WordNetLemmatizer()
    CDR_no_is['SECTOR_SEARCH']  = CDR_no_is['SECTOR_SEARCH'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    # Deleting stopwords
    stop_words = set(stopwords.words('english'))
    CDR_no_is['SECTOR_SEARCH']  = CDR_no_is['SECTOR_SEARCH'].apply(lambda x: [word for word in x if not word in stop_words] )
    # Searching NAICS in Google
    browser = starting_browser()
    n_rows = 0
    for index, row in CDR_no_is.iterrows():
        if n_rows % 5 == 0:
            browser.close()
            browser = starting_browser()
        CDR_no_is.loc[index, '2007 NAICS US Code'], browser = searching_naics_code_using_web_scrapping(row['SECTOR_SEARCH'], browser)
        n_rows += 1
    browser.close()
    # Concatenating
    CDR = pd.concat([CDR_is, CDR_no_is], sort=False, axis=0,
                            ignore_index=True)
    del CDR_is, CDR_no_is
    CDR.drop(columns=['Industrial Sector Code', 'SECTOR_SEARCH'], inplace=True)
    CDR.rename(columns={'2007 NAICS US Code': 'NAICS'}, inplace=True)
    # Merging
    CDR_INFO = pd.merge(CDR_INFO, CDR, how='left', on='SECTOR')
    CDR_INFO.to_csv(saving_path + '/Uses_information.csv', sep=',', index=False)


def main_function(years):
    dir_path = os.path.dirname(os.path.realpath(__file__)) # Working Directory
    # dir_path = os.getcwd() # if you are working on Jupyter Notebook
    saving_path = dir_path + '/csv'
    cdr_path = dir_path + '/../../ancillary/cdr'
    srs_path = dir_path + '/../../ancillary/others/TSCA_Nonconfidential_Inventory.csv'

    # Calling SRS
    SRS = pd.read_csv(srs_path, low_memory=False,
                      usecols=['Internal Tracking Number',
                               'ID', 'Substance Name'])
    SRS['ID'] = SRS['ID'].apply(lambda x: int(float(re.sub(r'\-', '', x).strip())))

    # Calling FRS alternative names
    FRS_NAMES = pd.read_csv(dir_path + '/../../extract/frs/csv/NATIONAL_ALTERNATIVE_NAME_FILE.CSV',
                            low_memory=False,
                            dtype={'POSTAL_CODE': 'object', 'REGISTRY_ID': 'int'},
                            usecols=['REGISTRY_ID', 'PGM_SYS_ACRNM',
                                      'ALTERNATIVE_NAME'])
    FRS_NAMES = FRS_NAMES.loc[FRS_NAMES['PGM_SYS_ACRNM'] == 'TSCA']
    FRS_NAMES.drop_duplicates(keep='first', inplace=True)
    FRS_NAMES.drop(columns=['PGM_SYS_ACRNM'], inplace=True)
    FRS_NAMES['ALTERNATIVE_NAME'] = FRS_NAMES['ALTERNATIVE_NAME'].apply(lambda x: str(x).upper())

    # Calling FRS NAICS by program
    FRS_NAICS = pd.read_csv(dir_path + '/../../extract/frs/csv/NATIONAL_NAICS_FILE.CSV',
                            low_memory=False,
                            usecols=['REGISTRY_ID', 'NAICS_CODE'],
                            dtype={'NAICS_CODE': 'int', 'REGISTRY_ID': 'int'})
    FRS_NAICS.drop_duplicates(keep='first', inplace=True)

    # Normalizing NAICS without known year
    Exploring_years = [1987, 2002, 2007, 2012]
    for year in Exploring_years:
        FRS_NAICS['REPORTING YEAR'] = year
        FRS_NAICS = normalizing_naics(FRS_NAICS,
                                      naics_column='NAICS_CODE',
                                      column_year='REPORTING YEAR')
        FRS_NAICS.drop(columns=['REPORTING YEAR'], inplace=True)

    # Calling SIC codes
    FRS_SIC = pd.read_csv(dir_path + '/../../extract/frs/csv/NATIONAL_SIC_FILE.CSV',
                          low_memory=False,
                          usecols=['REGISTRY_ID', 'SIC_CODE'],
                          dtype={'NAICS_CODE': 'int', 'REGISTRY_ID': 'int'})
    FRS_SIC.drop_duplicates(keep='first', inplace=True)

    # Calling concordance SIC to NAICS
    SIC_to_NAICS = pd.read_csv(dir_path + '/../../ancillary/others/1987_SIC_to_2002_NAICS.csv',
                               usecols=['1987 SIC', '2002 NAICS'],
                               dtype={'2002 NAICS': 'int',
                                      '1987 SIC': 'int'})
    SIC_to_NAICS.rename(columns={'2002 NAICS': 'NAICS_CODE',
                                 '1987 SIC': 'SIC_CODE'},
                        inplace=True)

    # Converting SIC to NAICS and normalizing
    FRS_SIC = pd.merge(FRS_SIC, SIC_to_NAICS,
                       on='SIC_CODE', how='inner')
    del SIC_to_NAICS
    FRS_SIC['REPORTING YEAR'] = 2002
    FRS_SIC = normalizing_naics(FRS_SIC,
                                naics_column='NAICS_CODE',
                                column_year='REPORTING YEAR')
    FRS_SIC.drop(columns=['REPORTING YEAR', 'SIC_CODE'], inplace=True)
    FRS_SIC.drop_duplicates(keep='first', inplace=True)

    # Concatenating both datasets
    FRS_NAICS = pd.concat([FRS_NAICS, FRS_SIC], sort=False, axis=0,
                          ignore_index=True)
    FRS_NAICS.drop_duplicates(keep='first', inplace=True)
    del FRS_SIC

    FRS = pd.merge(FRS_NAMES, FRS_NAICS,
                   on='REGISTRY_ID',
                   how='left')
    del FRS_NAICS, FRS_NAMES

    # Organizing national production volume
    organizing_national_production_volume(cdr_path, saving_path, SRS)
    for year in years:
        cdr_path_year = f'{cdr_path}/cdr_{year}'
        # Organizing substances
        Substances = searching_substances(cdr_path_year, SRS)
        # Organizing facilities
        Facilities = searching_facilities(cdr_path_year, FRS)
        # Substance by facility
        organizing_information(cdr_path_year, Facilities,
                               Substances, year, saving_path)
    del Substances, Facilities, FRS, SRS
    searching_naics_codes_for_downstream(cdr_path, saving_path)


if __name__ == '__main__':

    years = [2012, 2016]
    main_function(years)
