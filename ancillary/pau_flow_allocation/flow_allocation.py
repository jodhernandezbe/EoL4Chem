# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(
                os.path.realpath(__file__)) + '/..')
from releases.releases import emission_factor

def pau_efficiency(code, mode_location=0.75):
    '''
    Function to calculate a value for the efficiency
    using a triangular distribution function
    '''

    if code == 'E6':
        min = 0.0
        max = 50.0
    elif code == 'E5':
        min = 50.0
        max = 95.0
    elif code == 'E4':
        min = 95.0
        max = 99.0
    elif code == 'E3':
        min = 99.0
        max = 99.99
    elif code == 'E2':
        min = 99.99
        max = 99.9999
    elif code == 'E1':
        min = 99.9999
        max = 100

    mode = min + (max - min)*mode_location
    efficiency = np.random.triangular(min, mode, max)
    return efficiency


def calling_properties(CAS):
    '''
    This function calls the properties from CompTox for supporting
    the flow allocation in the PAUs
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))
    CompTox = pd.read_csv(dir_path + '/../../extract/properties/comptox/CompTox.csv',
                          usecols=['CAS NUMBER',
                                   'Melting Point',
                                   'Boiling Point',
                                   'Water Solubility',
                                   'Molecular Mass'],
                          dtype={'CAS NUMBER': 'str'})
    CompTox = CompTox.loc[CompTox['CAS NUMBER'] == CAS]
    CompTox['Water Solubility'] = 1000*CompTox['Water Solubility']*CompTox['Molecular Mass']
    CompTox.drop(columns=['CAS NUMBER',
                          'Molecular Mass'],
                 inplace=True)
    Properties = CompTox.to_dict('records')[0]
    return Properties


def calling_dataset_for_individuals_or_others(CAS,
                                    efficiency_code,
                                    waste_stream,
                                    dataset,
                                    desired_column,
                                    Off_transfer=None,
                                    list_PAU=None):
    '''
    Function to analyze the statistics behind the individual PAUs in a set
    or
    other treatments and define whether the treatment is physical or chemical
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_file = f'{dir_path}/../../transform/pau4chem/statistics/{dataset}.csv'
    df = pd.read_csv(path_file,
                     low_memory=False)

    if list_PAU:
        #path_off_on =
        df = df.loc[df[desired_column].isin(list_PAU)]
    df_chem = df.loc[df['CAS NUMBER'] == CAS]
    if df_chem.empty:
        df_chem = df
    df_chem.drop(columns=['CAS NUMBER'], inplace=True)
    df_stream = df_chem.loc[df_chem['WASTE STREAM CODE'] == waste_stream]
    if df_stream.empty:
        df_stream = df_chem
    df_stream.drop(columns=['WASTE STREAM CODE'],
                   inplace=True)
    df_efficiency = df_stream.loc[df_stream['EFFICIENCY RANGE CODE'] == efficiency_code]
    if df_efficiency.empty:
        df_efficiency = df_stream
    df_efficiency.drop(columns=['EFFICIENCY RANGE CODE'],
                       inplace=True)
    df_efficiency = df_efficiency.groupby(desired_column,
                                          as_index=False).sum()
    result = df_efficiency.loc[df_efficiency['NUMBER'].idxmax(), desired_column]
    return result


def calling_tri_on_site_methods(PAU_name, Type_of_WM, CAS,
                                efficiency_code,
                                waste_stream,
                                Off_transfer):
    '''
    Function for calling file with information for allocation
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))
    Allocation = pd.read_csv(dir_path + '/../others/Methods_TRI.csv',
                             low_memory=False,
                             usecols=['Method 2005 and after',
                                      'Output carrier',
                                      'Objective',
                                      'If it is treatment, what kind of?'])
    Allocation.drop_duplicates(keep='first', inplace=True)
    Allocation = Allocation.loc[Allocation['Objective'] != 'Controlling flow rate and composition']
    Allocation = Allocation.where(pd.notnull(Allocation), None)
    if ' + ' in PAU_name:
        off_on = pd.read_csv(dir_path + '/../others/TRI_Off_On_Site_Management_Match.csv',
                             usecols=['TRI off-site',
                                      'TRI on-site'])
        list_off_on = set(off_on.loc[off_on['TRI off-site'] == Off_transfer, 'TRI on-site'].tolist())
        list_PAU = list(set(PAU_name.split(' + ')).intersection(list_off_on))
        PAU_name = calling_dataset_for_individuals_or_others(CAS,
                                                             efficiency_code,
                                                             waste_stream,
                                                             'individual_pau_statistics',
                                                             'METHOD NAME - 2005 AND AFTER',
                                                             Off_transfer=Off_transfer,
                                                             list_PAU=list_PAU,)
    Allocation = Allocation.loc[Allocation['Method 2005 and after'] == PAU_name]
    if PAU_name == 'Other treatment':
        Type = calling_dataset_for_individuals_or_others(CAS,
                                                         efficiency_code,
                                                         waste_stream,
                                                         'other_treatments',
                                                         'CHEMICAL/PHYSICAL')
        Allocation = Allocation.loc[Allocation['If it is treatment, what kind of?'] == Type]
    Objective = Allocation.loc[Allocation['Method 2005 and after'] == PAU_name, 'Objective'].iloc[0]
    Carrier_for_PAU = Allocation.loc[Allocation['Method 2005 and after'] == PAU_name, 'Output carrier'].iloc[0]
    Type_of_treatment = Allocation.loc[Allocation['Method 2005 and after'] == PAU_name, 'If it is treatment, what kind of?'].iloc[0]
    return Objective, Carrier_for_PAU, Type_of_treatment


def building_pau_black_box(CAS, Flow_input, waste_stream,
                           PAU_name, efficiency_code,
                           Releases_to_compartments,
                           Maximum_amount, Total_waste,
                           TRIFID, year, Total_release,
                           Type_of_WM, Off_transfer, Metal_indicator='NO'):
    '''
    This is the function to build the black boxes for the chemicals
    based on the information submitted by the RETDF to the TRI Program
    '''

    # Calling efficiency based on efficiency code
    Efficiency = pau_efficiency(efficiency_code)
    # Estimating emission factors
    Emission_factors = emission_factor(Releases_to_compartments,
                                       Maximum_amount,
                                       Total_waste,
                                       Total_release,
                                       CAS, TRIFID, year)
    # Calling the needed chemical properties
    Properties = calling_properties(CAS)
    # Calling allocation information
    Objective, Carrier_for_PAU, Type_of_treatment =\
        calling_tri_on_site_methods(PAU_name, Type_of_WM, CAS,
                                    efficiency_code,
                                    waste_stream,
                                    Off_transfer)

    # Carrier analysis
    if Objective == 'Removal':
        Carrier_flow = Flow_input*Efficiency/100
        Destroyed_converted_degraded_flow = 0.0
        if '/' in Carrier_for_PAU:
            if Carrier_for_PAU == 'L/W':
                if Properties['Water Solubility'] >= 1000: # Soluble in water if 1,000-10,000 and very solubla if >= 10,000mg/L
                    Carrier = 'W'
                else:
                    Carrier = 'L'
            else:
                if waste_stream == 'W':
                    if Properties['Melting Point'] < 25:
                        Carrier = 'L'
                    else:
                        Carrier = 'S'
                elif waste_stream == 'L':
                    if Properties['Melting Point'] < 25:
                        Carrier = 'W'
                    else:
                        Carrier = 'S'
                else:
                    if Properties['Water Solubility'] >= 1000: # Soluble in water if 1,000-10,000 and very solubla if >= 10,000mg/L
                        Carrier = 'W'
                    else:
                        Carrier = 'L'
        else:
            Carrier = Carrier_for_PAU
    elif Objective == 'Reclamation':
        Destroyed_converted_degraded_flow = 0.0
        Carrier_flow = Flow_input*Efficiency/100
        if PAU_name == 'Solvent recovery (including distillation, evaporation, fractionation or extraction)':
            Carrier = 'L'
        elif PAU_name == 'Metal recovery (by retorting, smelting, or chemical or physical extraction)':
            Carrier = 'S'
        else:
            if Metal_indicator == 'YES':
                Carrier = 'S'
            else:
                Carrier = 'W'
    elif (Objective == 'Safer disposal (RCRA Subtitle C)') | (Objective == 'Controlling flow rate and composition'):
        Carrier_flow = 0.0
        Carrier = None
        Destroyed_converted_degraded_flow = 0.0
    else:
        Carrier_flow = 0.0
        Carrier = None
        Destroyed_converted_degraded_flow = Flow_input*Efficiency/100

    # Correction in case of zero efficiency
    if Efficiency == 0.0:
        Carrier = None

    # Hidden flow
    F_out_hidden = (1 - Efficiency/100)*Flow_input

    # Fugitive air release
    Fugitive_flow = Emission_factors['Fugitive air release']*F_out_hidden

    # By product and Effluent
    if (Type_of_WM == 'Energy recovery') | (Type_of_treatment == 'Incineration'):
        if PAU_name == 'A01':
            if Properties['Boiling Point'] <= 60: # deg C chemical is assessed as a gas or liquid (Knock-out Drum)
                Effluent_flow = F_out_hidden
                Effluent = 'A'
                Waste_release_flow = 0.0
                Waste_release = None
            else:
                Effluent_flow = Emission_factors['Stack air release']*F_out_hidden
                Effluent = 'A'
                Waste_release_flow = F_out_hidden - (Effluent_flow + Fugitive_flow)
                if Properties['Melting Point'] <= 850: # Assesing normal operating temperature of flares
                    Waste_release = 'L'
                else:
                    Waste_release = 'S'
        else:
            # Assesing normal operating temperature of these devices
            if (Properties['Boiling Point'] <= 980) | (Properties['Melting Point'] <= 980): # deg C chemical is assessed as a gas or liquid
                Effluent_flow = F_out_hidden - Fugitive_flow
                Effluent = 'A'
                Waste_release_flow = 0.0
                Waste_release = None
            else: # deg C Chemical is assessed as a solid
                Effluent_flow = Emission_factors['Stack air release']*F_out_hidden
                Effluent = 'A'
                Waste_release_flow = F_out_hidden - (Effluent_flow + Fugitive_flow)
                Waste_release = 'S'
    else:
        if PAU_name in ['Solvent recovery (including distillation, evaporation, fractionation or extraction)',
                        'Other recovery or reclamation for reuse (including acid regeneration or other chemical reaction process)']:
            if waste_stream == 'A':
                Effluent_flow = Emission_factors['Stack air release']*F_out_hidden
                Effluent = 'A'
            else:
                Effluent_flow = Emission_factors['On-site surface water release']*F_out_hidden
                Effluent = waste_stream
            Waste_release_flow = F_out_hidden - (Effluent_flow + Fugitive_flow)
            Waste_release = 'W'
        elif PAU_name in ['Metal recovery (by retorting, smelting, or chemical or physical extraction)']:
            if Properties['Water Solubility'] >= 1000:
                Waste_release = 'W'
                Waste_release_flow = Emission_factors['On-site surface water release']*F_out_hidden
            else:
                Waste_release = 'S'
                Waste_release_flow = Emission_factors['On-site soil release']*F_out_hidden
            Effluent_flow =  F_out_hidden - (Waste_release_flow + Fugitive_flow)
            Effluent = waste_stream
        else:
            Effluent_flow =  F_out_hidden - Fugitive_flow
            Effluent = waste_stream
            Waste_release_flow = 0.0
            Waste_release = None

    # Correction in case of zero flow
    if Waste_release_flow == 0.0:
        Waste_release = None

    if Effluent_flow == 0.0:
        Effluent = None

    if Objective == 'Reclamation':
        Recycled_flow = Carrier_flow
        Recycled = Carrier
        Carrier_flow = 0.0
        Carrier = None
    else:
        Recycled_flow = 0.0
        Recycled = None


    Result = {'Destroyed/converted/degraded flow': [Destroyed_converted_degraded_flow],
              'Carrier flow': [Carrier_flow],
              'Carrier phase': [Carrier],
              'Recycled flow': [Recycled_flow],
              'Recycled phase': [Recycled],
              'Effluent flow': [Effluent_flow],
              'Effluent phase': [Effluent],
              'Waste/release flow': [Waste_release_flow],
              'Waste/release phase': [Waste_release],
              'Fugitive air release': [Fugitive_flow]}

    return Result
