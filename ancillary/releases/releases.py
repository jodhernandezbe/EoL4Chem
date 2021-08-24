# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import numpy as np
import pandas as pd
import os


def source_reduction(sr_reduction_code):
    '''
    Function to convert the source reduction
    code to numeric values
    '''

    if sr_reduction_code == 'R1':
        return 1
    elif sr_reduction_code == 'R2':
        return np.random.uniform(0.50, 0.999999)
    elif sr_reduction_code == 'R3':
        return np.random.uniform(0.25, 0.499999)
    elif sr_reduction_code == 'R4':
        return np.random.uniform(0.15, 0.249999)
    elif sr_reduction_code == 'R5':
        return np.random.uniform(0.05, 0.159999)
    elif sr_reduction_code == 'R6':
        return np.random.uniform(0, 0.049999)
    else:
        return 0.0


def calling_source_reduction_db(year, CAS, TRIFID):
    '''
    Calling dataset with source reduction and production
    rete information
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = dir_path + '/../../transform/source_reduction/csv/'
    df = pd.read_csv(path + f'source_reduction_{year}.csv',
                     low_memory=False)
    df_chem_facility = df.loc[(df['TRIFID'] == TRIFID) &
                              (df['TRI_CHEM_ID'] == CAS)]
    df_chem_facility.drop(columns=['TRIFID', 'TRI_CHEM_ID',
                                   'REPORTING YEAR'],
                          inplace=True)
    if df_chem_facility.empty:
        Pr = 1
        Sr = 0
    else:
        cols_sr = [col for col in df_chem_facility.columns if 'SR ACTIVITY' in col]
        df_chem_facility[cols_sr] = df_chem_facility[cols_sr]\
            .applymap(lambda x: source_reduction(x))
        df_chem_facility['SR ACTIVITY'] = df_chem_facility[cols_sr].sum(axis=1)
        df_chem_facility.drop(columns=cols_sr, inplace=True)
        Pr = df_chem_facility['PROD RATIO/ACTIVITY INDEX'].mean()
        Sr = df_chem_facility['SR ACTIVITY'].mean()
        if Sr > 1:
            Sr = 1
    return Pr, Sr


def sigmoid_function(x):
    '''
    Sigmoid function
    '''
    return 1/(1 + np.exp(-x))


def mode_location(Pr, Sr, min_value, max_value):
    '''
    Function to assign the location of the mode for the
    annual change distribution
    '''

    production_contribution = sigmoid_function(Pr)
    source_reduction_contribution = sigmoid_function(Sr + 1)
    value_mode_location = production_contribution - source_reduction_contribution
    if value_mode_location == 0.0:
        return 0.0
    elif value_mode_location > 0.0:
        return max_value*value_mode_location
    else:
        return abs(min_value)*value_mode_location


def maximum_on_site(maximum_code):
    '''
    Function for assigning the value of the highest value for the
    maximum amount of chemical present at a facility
    '''

    if maximum_code == 1:
        return 0.453592*99
    elif maximum_code == 2:
        return 0.453592*999
    elif maximum_code == 3:
        return 0.453592*9999
    elif maximum_code == 4:
        return 0.453592*99999
    elif maximum_code == 5:
        return 0.453592*999999
    elif maximum_code == 6:
        return 0.453592*9999999
    elif maximum_code == 7:
        return 0.453592*49999999
    elif maximum_code == 8:
        return 0.453592*99999999
    elif maximum_code == 9:
        return 0.453592*499999999
    elif maximum_code == 10:
        return 0.453592*999999999
    elif maximum_code == 11:
        return 0.453592*10000000000
    elif maximum_code == 12:
        return 0.001*0.099
    elif maximum_code == 13:
        return 0.001*0.99
    elif maximum_code == 14:
        return 0.001*9.99
    elif maximum_code == 15:
        return 0.001*99
    elif maximum_code == 16:
        return 0.001*999
    elif maximum_code == 17:
        return 0.001*9999
    elif maximum_code == 18:
        return 0.001*99999
    elif maximum_code == 19:
        return 0.001*999999
    elif maximum_code == 20:
        return 0.001*100000000


def annual_change(Max_onsite, Total_releases_from_facility,
                  Total_waste_at_facility, Pr, Sr):
    '''
    Function to calculate the annual change of the chemical present at
    the RETDF
    '''

    min_value = max([-Max_onsite, Total_releases_from_facility
                    - Total_waste_at_facility])
    max_value = Max_onsite
    mode = mode_location(Pr, Sr, min_value, max_value)
    return np.random.triangular(min_value, mode, max_value)


def emission_factor(Release_to_compartments, Max_onsite_code, Tota_waste,
                    Total_release, CAS, TRIFID, year):
    '''
    Function to calculate the emission factors for the chemicals
    '''

    # Source reduction and production/activity rate
    Pr, Sr = calling_source_reduction_db(year, CAS, TRIFID)
    # The highest value for the maximum amount of chemical on-site
    Max_onsite = maximum_on_site(Max_onsite_code)
    # Calculating a random value for the annual change
    Denominator = annual_change(Max_onsite,
                                Total_release,
                                Tota_waste,
                                Pr, Sr)

    Denominator = Denominator + Tota_waste
    Emission_factors = dict()
    for compartment, Release_to_compartment in Release_to_compartments.items():
        Emission_factor = Release_to_compartment/Denominator
        Emission_factors.update({compartment: Emission_factor})
    return Emission_factors
