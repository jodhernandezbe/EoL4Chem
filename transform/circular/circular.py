# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import os
import pandas as pd
pd.options.mode.chained_assignment = None
import argparse
import time
import numpy as np
import bisect
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(os.path.dirname(
                os.path.realpath(__file__)) + '/../..')
from ancillary.releases.releases import emission_factor
from ancillary.pau_flow_allocation.flow_allocation import building_pau_black_box
from ancillary.plots.sankey_diagram import sankey_diagram
from ancillary.plots.chord_diagram import chord_diagram
from ancillary.plots.heatmap_diagram import heatmap_diagram
from ancillary.plots.stacked_barplot import stacked_barplot
from ancillary.normalizing_naics.normalizing import normalizing_naics


class Network:

    def __init__(self, n_cycles, chem):
        self.n_cycles = n_cycles
        self.chem = chem
        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        self._supporting_table = pd.read_csv(f'{self._dir_path}/../../ancillary/others/TRI_Off_On_Site_Management_Match.csv')

    def _callin_cdr(self):
        '''
        Method for calling CDR
        '''

        path_cdr = f'{self._dir_path}/../cdr/csv/Uses_information.csv'
        CDR = pd.read_csv(path_cdr, low_memory=False,
                          usecols=['NAICS_CODE', 'NAICS',
                                   'REGISTRY_ID',
                                   'STRIPPED_CHEMICAL_ID_NUMBER',
                                   'OPTION',
                                   'TYPE_PROCESS_USE',
                                   'FUNCTION_CATEGORY',
                                   'PRODUCT_CATEGORY',
                                   'PCT_PROD_VOLUME'],
                          dtype={'STRIPPED_CHEMICAL_ID_NUMBER': 'str'})
        CDR = CDR.loc[CDR['STRIPPED_CHEMICAL_ID_NUMBER'] == self.chem]
        CDR = CDR.where(pd.notnull(CDR), None)
        CDR = CDR.where(CDR != 'Not known or reasonably ascertainable', None)
        CDR = CDR.where(CDR != 'Not Known or Reasonably Ascertainable', None)
        CDR.drop(columns=['STRIPPED_CHEMICAL_ID_NUMBER'],
                 inplace=True)
        CDR = CDR.loc[CDR['OPTION'] != 'Manufacturing']
        groups = CDR.groupby('REGISTRY_ID',
                             as_index=False)
        for _, group in groups:
            FRS_ID = group['REGISTRY_ID'].unique()[0]
            NAICS_CODE = group['NAICS_CODE'].unique()[0]
            if not NAICS_CODE:
                CDR_aux = CDR.drop_duplicates(keep='first', subset=['REGISTRY_ID'])
                NAICS_selected = np.random.choice(CDR_aux['NAICS_CODE'].dropna().values)
                CDR.loc[CDR['REGISTRY_ID'] == FRS_ID, 'NAICS_CODE'] = NAICS_selected
        Industrial = CDR.loc[CDR['OPTION'] == 'Industrial']
        # Removing industrial function category with hight purity
        Industrial = Industrial.loc[Industrial['FUNCTION_CATEGORY'].str.capitalize() != 'Laboratory chemicals']
        Commercial_consumer = CDR.loc[CDR['OPTION'] != 'Industrial']
        del CDR
        # Filling null values in industrial
        Industrial.drop(columns=['PRODUCT_CATEGORY'],
                        inplace=True)
        relation = {'NAICS': 'NAICS_CODE',
                    'FUNCTION_CATEGORY': 'NAICS',
                    'TYPE_PROCESS_USE': 'FUNCTION_CATEGORY',
                    'PCT_PROD_VOLUME': 'FUNCTION_CATEGORY'}
        for col in ['NAICS', 'FUNCTION_CATEGORY', 'TYPE_PROCESS_USE', 'PCT_PROD_VOLUME']:
            if col in ['TYPE_PROCESS_USE', 'FUNCTION_CATEGORY']:
                Industrial[col] = Industrial[col].str.capitalize()
            for idx, row in Industrial.iterrows():
                if pd.isnull(row[col]):
                    try:
                        Industrial.loc[idx, col] =\
                            np.random.choice(Industrial.loc[Industrial[relation[col]]
                                                            == row[relation[col]], col]\
                               .dropna().values)
                    except ValueError:
                        Industrial.loc[idx, col] = np.random.choice(Industrial[col].dropna().values)
                else:
                    Industrial.loc[idx, col] = row[col]
        Industrial['NAICS_CODE'] =\
            Industrial['NAICS_CODE'].astype('int')\
            .astype('str')
        Industrial['NAICS'] =\
            Industrial['NAICS'].astype('int')\
            .astype('str')
        Industrial['PCT_PROD_VOLUME'] = Industrial['PCT_PROD_VOLUME'].astype('float')
        grouping = list(set(Industrial.columns.tolist()) - set(['PCT_PROD_VOLUME']))
        Industrial = Industrial.groupby(grouping,
                                        as_index=False).sum()
        # Filling null values in commercial and consumer
        Commercial_consumer.drop(columns=['TYPE_PROCESS_USE',
                                          'FUNCTION_CATEGORY',
                                          'NAICS'],
                                 inplace=True)
        Commercial_consumer = Commercial_consumer.loc[pd.notnull(Commercial_consumer['OPTION'])]
        Commercial_consumer['OPTION'] = Commercial_consumer['OPTION'].str.capitalize()
        Commercial_consumer['PRODUCT_CATEGORY'] = Commercial_consumer['PRODUCT_CATEGORY'].str.capitalize()
        relation = {'PRODUCT_CATEGORY': 'OPTION',
                    'PCT_PROD_VOLUME': 'PRODUCT_CATEGORY'}
        for col in ['PRODUCT_CATEGORY', 'PCT_PROD_VOLUME']:
            for idx, row in Commercial_consumer.iterrows():
                if pd.isnull(row[col]):
                    try:
                        Commercial_consumer.loc[idx, col] =\
                            np.random.choice(Commercial_consumer.loc[Commercial_consumer[relation[col]]
                                                            == row[relation[col]], col]\
                               .dropna().values)
                    except ValueError:
                        Commercial_consumer.loc[idx, col] =\
                            np.random.choice(Commercial_consumer[col].dropna().values)
                else:
                    Commercial_consumer.loc[idx, col] = row[col]
        Commercial_consumer['NAICS_CODE'] =\
            Commercial_consumer['NAICS_CODE'].astype('int')\
            .astype('str')
        Commercial_consumer['PCT_PROD_VOLUME'] = Commercial_consumer['PCT_PROD_VOLUME'].astype('float')
        grouping = list(set(Commercial_consumer.columns.tolist()) - set(['PCT_PROD_VOLUME']))
        Commercial_consumer = Commercial_consumer.groupby(grouping,
                                                          as_index=False).sum()

        return Industrial, Commercial_consumer

    def _comparing_naics(self, NAICS_1, NAICS_2):
        '''
        Method for comparing NAICS codes
        '''

        sum = 0
        for i in range(len(NAICS_2)):
            if NAICS_1.startswith(NAICS_2[0:i+1]):
                sum += 1
            else:
                break

        return sum

    def _possible_pathway_for_recycled_flows(self, Industrial,
                                             Commercial_consumer,
                                             NAICS_code,
                                             FRS_ID,
                                             RETDF_selected,
                                             RETDF_reporting_year):
        '''
        Method to determine the possible pathway for a recycled chemical:
        consumer, commercial, and industrial use
        '''

        cou_path = f'{self._dir_path}/../waste_tracking/csv/on_site_tracking/Conditions_of_use_by_facility_and_chemical_{RETDF_reporting_year}.csv'
        df_cou = pd.read_csv(cou_path, low_memory=False,
                             usecols=['TRIFID', 'TRI_CHEM_ID',
                                      'SALE OR DISTRIBUTION OF THE CHEMICAL',
                                      'USED AS A REACTANT',
                                      'ADDED AS A FORMULATION COMPONENT',
                                      'USED AS AN ARTICLE COMPONENT',
                                      'REPACKAGING',
                                      'USED AS A CHEMICAL PROCESSING AID',
                                      'USED AS A MANUFACTURING AID',
                                      'ANCILLARY OR OTHER USE'])
        df_cou['TRI_CHEM_ID'] = df_cou['TRI_CHEM_ID'].str.lstrip('0')
        df_cou = df_cou.loc[(df_cou['TRI_CHEM_ID'] == self.chem) &
                            (df_cou['TRIFID'] == RETDF_selected)]
        df_cou.drop(columns=['TRIFID', 'TRI_CHEM_ID'],
                    inplace=True)
        sale_distribution = df_cou['SALE OR DISTRIBUTION OF THE CHEMICAL'].values[0]
        distribution_list = ['SALE OR DISTRIBUTION OF THE CHEMICAL',
                             'ADDED AS A FORMULATION COMPONENT',
                             'USED AS AN ARTICLE COMPONENT',
                             'REPACKAGING']
        on_site_use = ['USED AS A REACTANT',
                       'USED AS A CHEMICAL PROCESSING AID',
                       'USED AS A MANUFACTURING AID',
                       'ANCILLARY OR OTHER USE']

        if sale_distribution == 'Yes':
            cou_yes = [key for key, val in df_cou.to_dict('record')[0].items() if val == 'Yes']
            cou_selected = cou_yes[np.random.randint(0,len(cou_yes))]
            if cou_selected in distribution_list:
                Input = {'Industry': Industrial,
                         'Commercial_consumer': Commercial_consumer}
                df = pd.DataFrame()
                for key, value in Input.items():
                    # Facility
                    df_f = value[value['REGISTRY_ID'] == FRS_ID]
                    How = 'Facility'
                    if df_f.empty:
                        df_f = value
                        df_f['VAL'] = df_f['NAICS_CODE']\
                            .apply(lambda x: self._comparing_naics(x, str(NAICS_code)))
                        max = df_f['VAL'].max()
                        # Industry
                        if max >= 2:
                            df_f = df_f.loc[df_f['VAL'] == max]
                            df_f.drop(columns=['VAL'], inplace=True)
                            How = 'Industry'
                        else:
                            # General
                            df_f = value
                            How = 'General'
                    df_f = df_f[['OPTION', 'PCT_PROD_VOLUME']]
                    df_f['HOW'] = How
                    df = pd.concat([df, df_f], ignore_index=True,
                                   sort=True, axis=0)

                if (df['HOW'] == 'Facility').any():
                    df = df.loc[df['HOW'] == 'Facility']
                elif (df['HOW'] == 'Industry').any():
                    df = df.loc[df['HOW'] == 'Industry']

                df['ACCUMULATED'] = df['PCT_PROD_VOLUME'].transform(pd.Series.cumsum)
                df.sort_values(by=['ACCUMULATED'], inplace=True)
                df.reset_index(inplace=True, drop=True)

                max_accumulated = df['ACCUMULATED'].max()
                rnd_pathway = np.random.uniform(0, max_accumulated)
                idx = bisect.bisect_left(df['ACCUMULATED'].tolist(), rnd_pathway)
                Option = df['OPTION'].iloc[idx]
                How = df['HOW'].iloc[idx]
                Sale = 'Yes'
                What_processing = None
            else:
                Option = 'Industrial'
                How = None
                Sale = 'No'
                What_processing = cou_selected
        else:
            cou_yes = [key for key, val in df_cou.to_dict('record')[0].items() if val == 'Yes']
            cou_yes = [col for col in cou_yes if col not in distribution_list]
            try:
                cou_selected = cou_yes[np.random.randint(0,len(cou_yes))]
            except ValueError:
                cou_selected = 'ANCILLARY OR OTHER USE'
            What_processing = cou_selected
            Option = 'Industrial'
            How = None
            Sale = 'No'

        return Option, How, Sale, What_processing

    def _searching_6_digit_naics(self, IS):
        '''
        Method that uses the U.S. Bureau Census' SUSB
        to search for the 6-digit NAICS code
        '''

        path_susb = f'{self._dir_path}/../../ancillary/others/SUSB_2017.csv'
        susb = pd.read_csv(path_susb,
                           low_memory=True,
                           usecols=['NAICS CODE',
                                    'ENTERPRISE EMPLOYMENT SIZE',
                                    'NUMBER OF ESTABLISHMENTS'])
        susb = susb.loc[susb['ENTERPRISE EMPLOYMENT SIZE']
                        .str.contains(r'01:')]
        susb = susb.loc[pd.notnull(susb['NAICS CODE'])]
        susb = susb.loc[susb['NAICS CODE'].str.len() == 6]
        susb['VAL'] = susb['NAICS CODE']\
            .apply(lambda x: self._comparing_naics(x, IS))
        susb = susb.loc[susb['VAL'] == len(IS)]

        susb['ACCUMULATED'] = susb['NUMBER OF ESTABLISHMENTS']\
            .transform(pd.Series.cumsum)
        susb.sort_values(by=['ACCUMULATED'], inplace=True)
        susb.reset_index(inplace=True, drop=True)

        max_accumulated = susb['ACCUMULATED'].max()
        rnd_pathway = np.random.uniform(0, max_accumulated)
        idx = bisect.bisect_left(susb['ACCUMULATED'].tolist(), rnd_pathway)
        IS = int(susb['NAICS CODE'].iloc[idx])
        return IS

    def _checking_industrial_use_no_sale(self, Industrial,
                                         NAICS_code,
                                         What_processing):
        '''
        Method for checking industrial use when the RETDF does not
        sell the recycled chemical
        '''

        cdr_tri = f'{self._dir_path}/../../ancillary/others/CDR_TRI_parallel_TPU.csv'
        cdr_tri = pd.read_csv(cdr_tri)
        cdr_pross = cdr_tri.loc[cdr_tri['TRI'] == What_processing, 'CDR'].values[0]

        Industrial_pross = Industrial.loc[Industrial['TYPE_PROCESS_USE'] == cdr_pross]
        if Industrial_pross.empty: # only in case
            Industrial_pross = Industrial
        Industrial_pross['VAL'] = Industrial_pross['NAICS']\
                                 .apply(lambda x: self._comparing_naics(x, str(NAICS_code)))
        max = Industrial_pross['VAL'].max()
        if max >= 2:
            Industrial_pross = Industrial_pross.loc[Industrial_pross['VAL'] == max]
        Industrial_pross.drop(columns=['OPTION', 'REGISTRY_ID',
                                       'NAICS_CODE', 'NAICS',
                                       'VAL'],
                              inplace=True)

        Industrial_pross = Industrial_pross.groupby(['TYPE_PROCESS_USE',
                                                     'FUNCTION_CATEGORY'],
                                                    as_index=False).sum()

        Industrial_pross['ACCUMULATED'] = Industrial_pross['PCT_PROD_VOLUME']\
            .transform(pd.Series.cumsum)
        Industrial_pross.sort_values(by=['ACCUMULATED'], inplace=True)
        Industrial_pross.reset_index(inplace=True, drop=True)

        max_accumulated = Industrial_pross['ACCUMULATED'].max()
        rnd_pathway = np.random.uniform(0, max_accumulated)
        idx = bisect.bisect_left(Industrial_pross['ACCUMULATED'].tolist(),
                                 rnd_pathway)

        # Industry Sector
        IS = NAICS_code
        # Industrial Processing or Use Operations
        TPU = Industrial_pross['TYPE_PROCESS_USE'].iloc[idx].capitalize().strip()
        # Industrial Function Category
        IFC = Industrial_pross['FUNCTION_CATEGORY'].iloc[idx].capitalize().strip()

        Result = {'Industry sector': [int(IS)],
                  'Industrial processing or use operation': [TPU],
                  'Industry function category': [IFC]}

        return Result

    def _checking_industrial_use_and_sector_sale(self, Industrial,
                                                 NAICS_code,
                                                 FRS_ID, How):
        '''
        Method for checking industry sector and industrial use
        for sale cases
        '''

        if How == 'Industry':
            Industrial['VAL'] = Industrial['NAICS_CODE']\
                .apply(lambda x: self._comparing_naics(x, str(NAICS_code)))
            max = Industrial['VAL'].max()
            if max >= 2:
                Industrial = Industrial.loc[Industrial['VAL'] == max]
            Industrial.drop(columns=['VAL'], inplace=True)
        elif How == 'Facility':
            Industrial = Industrial[Industrial['REGISTRY_ID'] == FRS_ID]

        Industrial.drop(columns=['NAICS_CODE',
                                 'REGISTRY_ID',
                                 'OPTION'],
                        inplace=True)

        Industrial = Industrial.groupby(['TYPE_PROCESS_USE',
                                         'FUNCTION_CATEGORY',
                                         'NAICS'],
                                        as_index=False).sum()

        Industrial['ACCUMULATED'] = Industrial['PCT_PROD_VOLUME']\
            .transform(pd.Series.cumsum)
        Industrial.sort_values(by=['ACCUMULATED'], inplace=True)
        Industrial.reset_index(inplace=True, drop=True)

        max_accumulated = Industrial['ACCUMULATED'].max()
        rnd_pathway = np.random.uniform(0, max_accumulated)
        idx = bisect.bisect_left(Industrial['ACCUMULATED'].tolist(),
                                 rnd_pathway)

        # Industry Sector
        IS = Industrial['NAICS'].iloc[idx]
        # Industrial Processing or Use Operations
        TPU = Industrial['TYPE_PROCESS_USE'].iloc[idx].capitalize().strip()
        # Industrial Function Category
        IFC = Industrial['FUNCTION_CATEGORY'].iloc[idx].capitalize().strip()

        # Searching NAICS at levels lowers than 6
        if len(IS) < 6:
            IS = self._searching_6_digit_naics(IS)

        Result = {'Industry sector': [int(IS)],
                  'Industrial processing or use operation': [TPU],
                  'Industry function category': [IFC]}

        return Result

    def _checking_commercial_and_consumer_product(self,
                                                  Commercial_consumer,
                                                  NAICS_code,
                                                  FRS_ID,
                                                  How):
        '''
        Method to select the product category
        '''

        if How == 'Industry':
            Commercial_consumer['VAL'] = Commercial_consumer['NAICS_CODE']\
                .apply(lambda x: self._comparing_naics(x, str(NAICS_code)))
            max = Commercial_consumer['VAL'].max()
            Commercial_consumer =\
                Commercial_consumer.loc[Commercial_consumer['VAL'] == max]
            Commercial_consumer.drop(columns=['VAL'], inplace=True)
        elif How == 'Facility':
            Commercial_consumer =\
                Commercial_consumer[Commercial_consumer['REGISTRY_ID'] == FRS_ID]

        Commercial_consumer['ACCUMULATED'] =\
            Commercial_consumer['PCT_PROD_VOLUME']\
            .transform(pd.Series.cumsum)
        Commercial_consumer.sort_values(by=['ACCUMULATED'], inplace=True)
        Commercial_consumer.reset_index(inplace=True, drop=True)

        max_accumulated = Commercial_consumer['ACCUMULATED'].max()
        rnd_pathway = np.random.uniform(0, max_accumulated)
        idx = bisect.bisect_left(Commercial_consumer['ACCUMULATED'].tolist(),
                                 rnd_pathway)

        # Product Use and Category
        PUC = Commercial_consumer['PRODUCT_CATEGORY'].iloc[idx].capitalize().strip()
        Result = {'Product category': [PUC]}

        return Result

    def _ratio(self, x):
        '''
        Method for calculating the ratio between the flow of chemical
        of insterest and others that went into the same PAU at the
        same facility
        '''

        Denominator = x.loc[x['TRI_CHEM_ID'] == self.chem,
                            'ON-SITE - RECYCLED'].values[0]
        x['RATIO'] = x['ON-SITE - RECYCLED']/Denominator
        return x

    def _selecting_highest_degree(self, df, n_loop):
        '''
        method for selecting the highest evidence degree through the loop
        '''

        max_degree = df['Evidence-degree'].max()
        df = df.loc[df['Evidence-degree'] == max_degree]
        if n_loop == 0:
            df['Sample'] = 1
        else:
            df.loc[pd.isnull(df['Sample']), 'Sample'] = 1
        df['Evidence-degree'] = df['Evidence-degree'].astype('int')
        df = df.groupby(['Category',
                         'Chemical',
                         'Option',
                         'Evidence-degree'],
                        as_index=False)\
            .agg({'Sample': 'sum', 'Ratio': 'sum'})
        return df

    def _checking_possible_chemicals(self,
                                     On_recycling,
                                     RETDF_selected,
                                     RETDF_reporting_year,
                                     NAICS_code):
        '''
        Method for checking chemicals that can flow with the
        recycling chemical of insterest
        '''

        Codes_change = {'R11': 'H20', 'R12': 'H20',
                        'R13': 'H20', 'R14': 'H20',
                        'R19': 'H20', 'R21': 'H10',
                        'R22': 'H10', 'R23': 'H10',
                        'R24': 'H10', 'R26': 'H10',
                        'R27': 'H10', 'R28': 'H10',
                        'R29': 'H10', 'R30': 'H10',
                        'R40': 'H39', 'R99': 'H39',
                        'H20': 'H20', 'H10': 'H10',
                        'H39': 'H39'}
        Names = {'H20': 'Solvent recovery (including distillation, evaporation, fractionation or extraction)',
                 'H10': 'Metal recovery (by retorting, smelting, or chemical or physical extraction)',
                 'H39': 'Other recovery or reclamation for reuse (including acid regeneration or other chemical reaction process)'}

        path_pau = f'{self._dir_path}/../pau4chem/statistics/db_for_solvents/DB_for_Solvents_{RETDF_reporting_year}.csv'
        pau = pd.read_csv(path_pau,
                          low_memory=True,
                          usecols=['TRIFID',
                                   'PRIMARY NAICS CODE',
                                   'TRI_CHEM_ID',
                                   'ON-SITE - RECYCLED',
                                   'UNIT OF MEASURE',
                                   'METHOD'])

        # Organizing methods
        pau['TRI_CHEM_ID'] = pau['TRI_CHEM_ID'].str.lstrip('0')
        Grouping = ['TRIFID', 'METHOD',
                    'PRIMARY NAICS CODE']
        pau = pau.loc[pau.groupby(Grouping)['TRI_CHEM_ID']
                      .transform(lambda x: (x == self.chem).any())]
        pau['METHOD'] = pau['METHOD'].map(Codes_change)
        pau.drop_duplicates(keep='first', inplace=True)
        pau['METHOD'] = pau['METHOD'].map(Names)
        method = pau.loc[pau['METHOD'] == On_recycling]
        del pau
        method.loc[method['UNIT OF MEASURE'] == 'Pounds',
                   'ON-SITE - RECYCLED'] *= 0.453592
        method.loc[method['UNIT OF MEASURE'] == 'Grams',
                   'ON-SITE - RECYCLED'] *= 10**-3
        method.drop(columns=['UNIT OF MEASURE',
                             'METHOD'],
                    inplace=True)

        # Ratio by facility
        Grouping = ['TRIFID',
                    'PRIMARY NAICS CODE']
        method = method.groupby(Grouping, as_index=False)\
            .apply(lambda x:  self._ratio(x))
        aux = method.loc[method['TRI_CHEM_ID'] == self.chem,
                         Grouping + ['ON-SITE - RECYCLED']]
        aux.rename(columns={'ON-SITE - RECYCLED': 'FLOW'},
                   inplace=True)
        method = pd.merge(method, aux, on=Grouping, how='inner')
        del aux
        method = method.loc[method['TRI_CHEM_ID'] != self.chem]
        method.drop(columns=['ON-SITE - RECYCLED'], inplace=True)

        # Searching the ratio
        facility = method.loc[method['TRIFID'] == RETDF_selected]
        if facility.empty:
            facility = method
            facility['VAL'] = facility['PRIMARY NAICS CODE']\
                .apply(lambda x: self._comparing_naics(str(x), str(NAICS_code)))
            max = method['VAL'].max()
            if max >= 2:
                facility = facility.loc[facility['VAL'] == max]
                facility.drop(columns=['PRIMARY NAICS CODE', 'VAL'],
                              inplace=True)

                Flows = facility[['TRIFID', 'FLOW']]\
                    .drop_duplicates(keep='first')
                Flows['FLOW'] = Flows['FLOW'].transform(pd.Series.cumsum)
                Flows.sort_values(by=['FLOW'], inplace=True)
                Flows.reset_index(inplace=True)

                max_accumulated = Flows['FLOW'].max()
                rnd_facility = np.random.uniform(0, max_accumulated)
                idx = bisect.bisect_left(Flows['FLOW'].tolist(), rnd_facility)
                ID = Flows['TRIFID'].iloc[idx]
                facility = facility.loc[facility['TRIFID'] == ID]

                Evidence = 'Yes'
                Evidence_degree = max - 1
            else:
                Evidence = 'No'
        else:
            Evidence = 'Yes'
            Evidence_degree = 6

        if Evidence == 'Yes':
            n_rows = facility.shape[0]
            Result = {'Evidence': ['Yes']*n_rows,
                      'Evidence-degree': [Evidence_degree]*n_rows,
                      'Chemical': facility['TRI_CHEM_ID'].tolist(),
                      'Ratio': facility['RATIO'].tolist()}
        else:
            Result = {'Evidence': ['No'], 'Evidence-degree': [None],
                      'Chemical': [None], 'Ratio': [None]}

        return Result

    def _Disposal_activities_checking(self, year, TRIFID, WM_TRI,
                                      Fugitive_release,
                                      Maximum_amount, Total_waste,
                                      Total_release, Flow_transferred,
                                      NAICS_code, RCRA_ID):
        '''
        This method is for checking the disposal activities at the RETDF.
        In addition, this method supports the tracking of the chemical
        flows for disposal
        '''

        Output = {'On-site soil release': [None],
                  'Fugitive air release': [None],
                  'Net disposal': [None],
                  'Waste management under TRI': [None]}

        # Fugitive air emission
        Emission_factor =\
            emission_factor({'Fugitive air release': Fugitive_release},
                            Maximum_amount,
                            Total_waste,
                            Total_release,
                            self.chem, TRIFID, year)
        Emission_factor = Emission_factor['Fugitive air release']
        Output['Fugitive air release'] = [Flow_transferred*Emission_factor]

        on_site = self._supporting_table.loc[
            self._supporting_table['TRI off-site'] == WM_TRI]
        if on_site.shape[0] == 1:
            if pd.notnull(on_site['TRI on-site']).all():
                Output['Waste management under TRI'] = [WM_TRI]
                if (on_site['TRI on-site'] == 'On-site soil release').all():
                    Output['Net disposal'] = [0.0]
                    Output['On-site soil release'] = [Flow_transferred*(1 - Emission_factor)]
                else:
                    Output['On-site soil release'] = [0.0]
                    Output['Net disposal'] = [Flow_transferred*(1 - Emission_factor)]
            else:
                if WM_TRI in ['Storage only', 'Other off-site management']:
                    Output['Waste management under TRI'] = [WM_TRI]
                    Output['On-site soil release'] = [0.0]
                    Output['Net disposal'] = [Flow_transferred*(1 - Emission_factor)]
                else:
                    path_disposal = self._dir_path + '/../waste_tracking/csv/disposal_activities/'
                    df_disposal = pd.read_csv(path_disposal + f'Disposal_{year}.csv')
                    df_disposal_chem = df_disposal.loc[df_disposal['TRI_CHEM_ID'] == self.chem]
                    if not df_disposal_chem.empty:
                        df_disposal_chem_RETDF = df_disposal_chem.loc[df_disposal_chem['TRIFID'] == TRIFID]
                        if not df_disposal_chem_RETDF.empty:
                            RETDF = df_disposal_chem_RETDF[['DISPOSAL ACTIVITY',
                                                            'FLOW']]
                            RANDOM = RETDF
                        else:
                            NAICS = df_disposal_chem.loc[df_disposal_chem['PRIMARY NAICS CODE'] == int(NAICS_code),
                                                         ['DISPOSAL ACTIVITY', 'FLOW']]
                            if not NAICS.empty:
                                RANDOM = NAICS
                            else:
                                RANDOM = df_disposal_chem[['DISPOSAL ACTIVITY', 'FLOW']]
                        RANDOM = RANDOM.groupby('DISPOSAL ACTIVITY', as_index=False).sum()
                        RANDOM['PROPORTION'] = RANDOM['FLOW']/RANDOM['FLOW'].sum()
                        RANDOM['PROPORTION'] = RANDOM['PROPORTION'].transform(pd.Series.cumsum)
                        RANDOM.sort_values(by=['PROPORTION'], inplace=True)
                        RANDOM.reset_index(inplace=True, drop=True)
                        max_proportion = RANDOM['PROPORTION'].max()
                        rnd_disposal = np.random.uniform(0, max_proportion)
                        idx = bisect.bisect_left(RANDOM['PROPORTION'].tolist(), rnd_disposal)
                        DISPOSAL_selected = RANDOM['DISPOSAL ACTIVITY'].iloc[idx]
                        On_off_dict = {'Other disposal': 'Other off-site management',
                                       'Land treatment/application farming': 'Land treatment'}
                        List_for_metals = ['Solidification/stabilization (metals)',
                                           'Wastewater treatment (excluding POTWS) - metals and metal compounds only',
                                           'Wastewater treatment (metals)']
                        if DISPOSAL_selected in On_off_dict.keys():
                            if WM_TRI not in List_for_metals:
                                Output['Waste management under TRI'] = [On_off_dict[DISPOSAL_selected]]
                            else:
                                Output['Waste management under TRI'] = [WM_TRI]
                            Output['Net disposal'] = [0.0]
                            Output['On-site soil release'] = [Flow_transferred*(1 - Emission_factor)]
                        else:
                            Output['On-site soil release'] = [0.0]
                            Output['Net disposal'] = [Flow_transferred*(1 - Emission_factor)]
                            if WM_TRI not in List_for_metals:
                                off_site = self._supporting_table.loc[self._supporting_table['TRI on-site'] == DISPOSAL_selected]
                                if off_site.shape[0] == 1:
                                    Output['Waste management under TRI'] = [off_site['TRI off-site'].values[0]]
                                else:
                                    if RCRA_ID:
                                        Output['Waste management under TRI'] =\
                                            [off_site.loc[off_site['TRI off-site'].str.contains('RCRA'), 'TRI off-site'].values[0]]
                                    else:
                                        Output['Waste management under TRI'] =\
                                            [off_site.loc[~off_site['TRI off-site'].str.contains('RCRA'), 'TRI off-site'].values[0]]
                            else:
                                Output['Waste management under TRI'] = [WM_TRI]
        else:
            Output['Waste management under TRI'] = [WM_TRI]
            Output['On-site soil release'] = [0.0]
            Output['Net disposal'] = [Flow_transferred*(1 - Emission_factor)]
        Output.update({'Type of waste management': ['Disposal']})
        return Output

    def _pattern_searcher(self, method, search_list):
        '''
        Method for search if the methods include one of the codes of interest
        '''

        method = set(method.split(' + '))
        search_obj = list(set(search_list) - method)
        if len(search_obj) < len(search_list):
            return True
        else:
            return False

    def _Non_disposal_activities_checking(self, year, TRIFID, WM_TRI,
                                          Releases_to_compartments,
                                          Maximum_amount, Total_waste,
                                          Total_release, Flow_transferred,
                                          Metal_indicator, NAICS_code,
                                          WMH):
        '''
        This method is for checking the recycling, energy recovery, and treatment activities at the RETDF
        In addition, this method supports the tracking of the chemical flows for these three activities
        '''

        Output = dict()

        off_on =\
            self._supporting_table.loc[self._supporting_table['TRI off-site'] == WM_TRI]
        Methods = off_on['TRI on-site'].tolist()

        path_pau = self._dir_path + '/../pau4chem/datasets/final_pau_datasets/'
        df_pau = pd.read_csv(f'{path_pau}PAUs_DB_filled_{year}.csv',
                             usecols=['TRIFID', 'PRIMARY NAICS CODE',
                                      'TRI_CHEM_ID', 'WASTE STREAM CODE',
                                      'METHOD NAME - 2005 AND AFTER',
                                      'TYPE OF MANAGEMENT',
                                      'EFFICIENCY RANGE CODE'])
        df_pau_chem = df_pau.loc[(df_pau['TRI_CHEM_ID'] == self.chem) &
                                 (df_pau['WASTE STREAM CODE'] != 'A')]

        # Searching PAU
        if not df_pau_chem.empty:
            try:
                df_pau_WM = df_pau_chem.loc[df_pau_chem['METHOD NAME - 2005 AND AFTER']\
                                            .apply(lambda x: self._pattern_searcher(x, Methods))]
                df_pau_NAICS =  df_pau_WM.loc[df_pau_WM['PRIMARY NAICS CODE'] == NAICS_code]
                if not df_pau_NAICS.empty:
                    df_pau_RETDF = df_pau_NAICS.loc[df_pau_NAICS['TRIFID'] == TRIFID]
                    if not df_pau_RETDF.empty:
                        RANDOM = df_pau_RETDF[['METHOD NAME - 2005 AND AFTER',
                                               'EFFICIENCY RANGE CODE',
                                               'WASTE STREAM CODE']]
                    else:
                        RANDOM = df_pau_NAICS[['METHOD NAME - 2005 AND AFTER',
                                               'EFFICIENCY RANGE CODE',
                                               'WASTE STREAM CODE']]
                else:
                    RANDOM = df_pau_WM[['METHOD NAME - 2005 AND AFTER',
                                        'EFFICIENCY RANGE CODE',
                                        'WASTE STREAM CODE']]
                # Selecting PAU and efficiency
                RANDOM['PROPORTION'] = 1
                groupings = [['METHOD NAME - 2005 AND AFTER'],
                             ['WASTE STREAM CODE',
                              'METHOD NAME - 2005 AND AFTER'],
                             ['EFFICIENCY RANGE CODE',
                              'WASTE STREAM CODE',
                              'METHOD NAME - 2005 AND AFTER']]
                PAU_selected = list()
                for grouping in groupings:
                    RAND_AUX = \
                        RANDOM[['PROPORTION'] + grouping]\
                            .groupby(grouping, as_index=False).sum()
                    if len(grouping) == 2:
                        RAND_AUX = RAND_AUX.loc[RAND_AUX[grouping[1]] == PAU_selected[0]]
                    elif len(grouping) == 3:
                        RAND_AUX = RAND_AUX.loc[(RAND_AUX[grouping[2]] == PAU_selected[0]) &
                                                (RAND_AUX[grouping[1]] == PAU_selected[1])]
                    RAND_AUX['PROPORTION'] = RAND_AUX['PROPORTION'].transform(pd.Series.cumsum)
                    RAND_AUX.sort_values(by=['PROPORTION'], inplace=True)
                    RAND_AUX.reset_index(inplace=True, drop=True)
                    RAND_AUX['PROPORTION'] = RAND_AUX['PROPORTION']/RAND_AUX['PROPORTION'].sum()
                    max_proportion = RAND_AUX['PROPORTION'].max()
                    rnd_pau = np.random.uniform(0, max_proportion)
                    idx = bisect.bisect_left(RAND_AUX['PROPORTION'].tolist(), rnd_pau)
                    PAU_selected.append(RAND_AUX[grouping[0]].iloc[idx])

                # On-site to off-site
                if 'broker' in WM_TRI:
                    on_off = self._supporting_table.loc[self._supporting_table['Type of waste management'] == WMH]
                    on_off = on_off.loc[~on_off['TRI off-site'].str.contains('broker')]
                    if 'treatment' in WM_TRI:
                        on_off = on_off.loc[on_off['TRI on-site']\
                                            .apply(lambda x: self._pattern_searcher(PAU_selected[0],
                                                                                    [x]))]
                        offs = on_off['TRI off-site'].unique().tolist()
                        path_transfers = self._dir_path + '/../waste_tracking/csv/Receiver_TRI_input_streams.csv'
                        df_transfers = pd.read_csv(path_transfers,
                                                   usecols=['QUANTITY TRANSFERRED',
                                                            'FOR WHAT IS TRANSFERRED',
                                                            'CAS', 'RECEIVER TRIFID'])
                        df_transfers['CAS'] = df_transfers['CAS'].str.replace('-', '')
                        df_transfers = df_transfers.loc[df_transfers['CAS'] == self.chem]
                        df_transfers = df_transfers.loc[df_transfers['FOR WHAT IS TRANSFERRED'].isin(offs)]
                        df_transfers['PROPORTION'] = df_transfers['QUANTITY TRANSFERRED']/df_transfers['QUANTITY TRANSFERRED'].sum()
                        df_receiver = df_transfers.loc[df_transfers['RECEIVER TRIFID'] == TRIFID]
                        if not df_receiver.empty:
                            RANDOM = df_receiver[['PROPORTION', 'FOR WHAT IS TRANSFERRED']]
                        else:
                            RANDOM = df_transfers[['PROPORTION', 'FOR WHAT IS TRANSFERRED']]
                            RANDOM = RANDOM.groupby('FOR WHAT IS TRANSFERRED', as_index=False).sum()
                        RANDOM['PROPORTION'] = RANDOM['PROPORTION'].transform(pd.Series.cumsum)
                        RANDOM.sort_values(by=['PROPORTION'], inplace=True)
                        RANDOM.reset_index(inplace=True, drop=True)
                        max_proportion = RANDOM['PROPORTION'].max()
                        rnd_transfer = np.random.uniform(0, max_proportion)
                        idx = bisect.bisect_left(RANDOM['PROPORTION'].tolist(), rnd_transfer)
                        Output['Waste management under TRI'] = [RANDOM['FOR WHAT IS TRANSFERRED'].iloc[idx]]
                    else:
                        Output['Waste management under TRI'] = [on_off.loc[on_off['TRI on-site'] == PAU_selected[0], 'TRI off-site'].values[0]]
                else:
                    Output['Waste management under TRI'] = [WM_TRI]

                # Black box
                Result = building_pau_black_box(self.chem, Flow_transferred,
                                                PAU_selected[1], PAU_selected[0],
                                                PAU_selected[2], Releases_to_compartments,
                                                Maximum_amount, Total_waste,
                                                TRIFID, year, Total_release, WMH,
                                                Output['Waste management under TRI'][0],
                                                Metal_indicator=Metal_indicator)

                if WMH == 'Recycling':
                    On_recycling = PAU_selected[0]
                else:
                    On_recycling = None

                Output.update(Result)
                Output.update({'Type of waste management': [WMH]})
                return Output, On_recycling
            except OverflowError:
                Output, On_recycling = self._Non_disposal_activities_checking(2018, TRIFID, WM_TRI,
                                                       Releases_to_compartments,
                                                       Maximum_amount, Total_waste,
                                                       Total_release, Flow_transferred,
                                                       Metal_indicator, NAICS_code,
                                                       WMH)
                return Output, On_recycling


    def _EoL_activities_checking(self, year, TRIFID, WM_TRI, WMH,
                                 Releases_to_compartments,
                                 Maximum_amount, Total_waste,
                                 Total_release, Flow_transferred,
                                 NAICS_code, RCRA_ID, Metal_indicator):
        '''
        This method is for checking the EoL activities at the RETDF
        '''

        if WMH == 'Disposal':
            Output = self._Disposal_activities_checking(year, TRIFID, WM_TRI,
                                                        Releases_to_compartments['Fugitive air release'],
                                                        Maximum_amount, Total_waste,
                                                        Total_release, Flow_transferred,
                                                        NAICS_code, RCRA_ID)
            On_recycling = None
        else:
            Output, On_recycling =\
                self._Non_disposal_activities_checking(year, TRIFID, WM_TRI,
                                                       Releases_to_compartments,
                                                       Maximum_amount, Total_waste,
                                                       Total_release, Flow_transferred,
                                                       Metal_indicator, NAICS_code,
                                                       WMH)

        return Output, On_recycling

    def loop(self):
        '''
        Following the loop
        '''

        # Calling EoL dataset
        path_columns_from_EoL = self._dir_path + '/../../ancillary/others/columns_for_using_in_network.txt'
        columns_from_EoL = pd.read_csv(path_columns_from_EoL,
                                       header=None,
                                       sep='\t').iloc[:, 0].tolist()
        path_EoL_dataset = self._dir_path + '/../tri/2018/TRI_SRS_FRS_CompTox_RETDF_2018_EoL.csv'
        EoL_dataset = pd.read_csv(path_EoL_dataset,
                                  usecols=columns_from_EoL,
                                  header=0, sep=',',
                                  low_memory=False)

        # Selecting the information of the chemical under study
        EoL_dataset = EoL_dataset.loc[EoL_dataset['TRI CHEMICAL ID NUMBER'] == self.chem]
        Metal_indicator = EoL_dataset['METAL INDICATOR'].unique()[0]
        if pd.notnull(EoL_dataset['RCRAInfo CHEMICAL ID NUMBER']).any():
            RCRA_ID = True
        else:
            RCRA_ID = False
        EoL_dataset.drop(columns=['TRI CHEMICAL ID NUMBER',
                                  'RCRAInfo CHEMICAL ID NUMBER',
                                  'METAL INDICATOR'],
                         inplace=True)
        RETDF = EoL_dataset.iloc[:, 8:]
        RETDF.drop_duplicates(keep='first', inplace=True)

        # This is temporal
        grouping = ['RETDF TRIFID',
                    'RETDF FRS ID',
                    'COMPARTMENT']
        RETDF = RETDF.loc[RETDF.groupby(grouping)['RETDF REPORTING YEAR'].idxmax()]
        del grouping

        EoL_dataset = EoL_dataset.iloc[:, 0:10]
        EoL_dataset.drop(columns=['RETDF FRS ID'], inplace=True)
        EoL_dataset.drop_duplicates(keep='first', inplace=True)
        grouping_columns = list(set(EoL_dataset.columns) - set(['PATHWAY RELATIVE IMPORTANCE']))
        EoL_dataset['ACCUMULATED'] = EoL_dataset.groupby(grouping_columns)['PATHWAY RELATIVE IMPORTANCE']\
                                                .transform(pd.Series.cumsum)

        # Running the loop
        Network_nodes = ['Generator Industry Sector',
                         'Type of waste management',
                         'Waste management under TRI',
                         'RETDF Industry Sector',
                         'Option', 'Industry sector',
                         'Industrial processing or use operation',
                         'Industry function category',
                         'Product category']
        df_ratio_loop = pd.DataFrame()
        df_markov_network = pd.DataFrame()
        df_flow_MN = pd.DataFrame()
        n_loop = 0
        n_sale = 0
        n_no_sale = 0
        n_non_industrial = 0
        n_industrial = 0
        n_recycling = 0
        n_total = 0
        m_total = 0
        m_total_2 = 0
        m_transfer_for_recycling = 0
        m_transfer_for_recycling_2 = 0
        m_recycled_for_sale = 0
        m_recycled_for_sale_2 = 0
        m_recycled_for_no_sale = 0
        m_recycled_for_no_sale_2 = 0
        m_recycled_for_industrial = 0
        m_recycled_for_industrial_2 = 0
        m_recycled_for_no_industrial = 0
        m_recycled_for_no_industrial_2 = 0
        for i in range(self.n_cycles):
            print('Cycle {}'.format(i + 1))
            groups = EoL_dataset.groupby(grouping_columns,
                                         as_index=False)
            df_output = pd.DataFrame()
            m_total_cycle = 0
            m_transfer_for_recycling_cycle = 0
            m_recycled_for_sale_cycle = 0
            m_recycled_for_no_sale_cycle = 0
            m_recycled_for_industrial_cycle = 0
            m_recycled_for_no_industrial_cycle = 0
            for _, group in groups:
                n_total = n_total + 1
                Flow_transferred = group['QUANTITY TRANSFER OFF-SITE'].unique()[0]
                m_total_cycle = m_total_cycle + Flow_transferred
                WMH = group['WASTE MANAGEMENT UNDER EPA WMH'].unique()[0].strip()
                WM_TRI = group['WASTE MANAGEMENT UNDER TRI'].unique()[0].strip()
                GiS = group['GENERATOR TRI PRIMARY NAICS TITLE'].unique()[0].strip()
                group.drop(columns='PATHWAY RELATIVE IMPORTANCE',
                           inplace=True)
                group = group.groupby(grouping_columns,
                                      as_index=False).sum()
                group.sort_values(by=['ACCUMULATED'], inplace=True)
                group.reset_index(inplace=True)

                # Selecting the RETDF
                max_accumulated = group['ACCUMULATED'].max()
                rnd_pathway = np.random.uniform(0, max_accumulated)
                idx = bisect.bisect_left(group['ACCUMULATED'].tolist(), rnd_pathway)
                RETDF_selected = group['RETDF TRIFID'].iloc[idx]

                # Going into the RETDF
                RETDF_Information = RETDF.loc[RETDF['RETDF TRIFID'] == RETDF_selected]
                Maximum_amount = int(RETDF_Information['MAXIMUM AMOUNT PRESENT AT RETDF'].unique()[0])
                Total_waste = RETDF_Information['TOTAL WASTE GENERATED BY RETDF'].unique()[0]
                Total_release = RETDF_Information['TOTAL RELEASE FROM RETDF'].unique()[0]
                RETDFiS = RETDF_Information['RETDF PRIMARY NAICS TITLE'].unique()[0].strip()
                NAICS_code = RETDF_Information['RETDF PRIMARY NAICS CODE'].unique()[0]
                FRS_ID = RETDF_Information['RETDF FRS ID'].unique()[0]
                Releases_to_compartments = {row['COMPARTMENT']:\
                                            row['FLOW TO COMPARTMENT FROM RETDF']\
                                            for idx, row in\
                                            RETDF_Information[['COMPARTMENT',\
                                                               'FLOW TO COMPARTMENT FROM RETDF']].iterrows()}
                RETDF_reporting_year = RETDF_Information['RETDF REPORTING YEAR'].unique()[0]
                Output, On_recycling =\
                    self._EoL_activities_checking(RETDF_reporting_year,
                                                  RETDF_selected,
                                                  WM_TRI, WMH,
                                                  Releases_to_compartments,
                                                  Maximum_amount,
                                                  Total_waste,
                                                  Total_release,
                                                  Flow_transferred,
                                                  NAICS_code,
                                                  RCRA_ID, Metal_indicator)
                Output.update({'Generator Industry Sector': [GiS.capitalize()],
                               'RETDF Industry Sector': [RETDFiS.capitalize()],
                               'Flow transferred': [Flow_transferred]})

                # Analyzing recycling flows
                if WMH == 'Recycling':

                    # Calling CDR
                    Industrial, Commercial_consumer = self._callin_cdr()

                    # Selecting between industrial and non-industrial activities
                    Option, How , Sale, What_processing =\
                        self._possible_pathway_for_recycled_flows(Industrial.copy(),
                                                                  Commercial_consumer.copy(),
                                                                  NAICS_code,
                                                                  FRS_ID,
                                                                  RETDF_selected,
                                                                  RETDF_reporting_year)

                    n_recycling = n_recycling + 1
                    m_transfer_for_recycling_cycle = m_transfer_for_recycling_cycle + Flow_transferred
                    if Sale == 'Yes':
                        n_sale = n_sale + 1
                        m_recycled_for_sale_cycle = m_recycled_for_sale_cycle + Flow_transferred
                    else:
                        n_no_sale =  n_no_sale + 1
                        m_recycled_for_no_sale_cycle = m_recycled_for_no_sale_cycle + Flow_transferred

                    if Option == 'Industrial':

                        m_recycled_for_industrial_cycle = m_recycled_for_industrial_cycle + Flow_transferred
                        n_industrial = n_industrial + 1

                        # Analyzing industrial activities
                        if Sale == 'Yes':
                            Result =\
                                self._checking_industrial_use_and_sector_sale(Industrial.copy(),
                                                                              NAICS_code,
                                                                              FRS_ID, How)
                        else:
                            Result =\
                                self._checking_industrial_use_no_sale(Industrial.copy(),
                                                                      NAICS_code,
                                                                      What_processing)
                        del Industrial

                        Result = pd.DataFrame(Result)
                        # NAICS
                        NAICS_structure = pd.read_csv('/home/jose-d-hernandez/Documents/EoL4Chem/ancillary/others/NAICS_Structure.csv',
                                                      dtype={'NAICS Code': 'str'})
                        NAICS_structure = NAICS_structure[NAICS_structure['NAICS Code'].str.len() == 6]
                        NAICS_structure['NAICS Code'] = NAICS_structure['NAICS Code'].astype('int')
                        NAICS_structure.rename(columns={'NAICS Code': 'Industry sector',
                                                        'NAICS Title': 'Industry title'},
                                               inplace=True)
                        Result = pd.merge(Result, NAICS_structure,
                                          on='Industry sector',
                                          how='left')
                        # Normalizing NAICS without known year
                        Exploring_years = [1987, 2002, 2007, 2012]
                        for year in Exploring_years:
                            Result['Year'] = year
                            Result = normalizing_naics(Result,
                                                       naics_column='Industry sector',
                                                       column_year='Year',
                                                       title=True,
                                                       circular=True)
                        Result['Industry sector'] =\
                            Result['Industry title'].str.capitalize().str.strip()
                        Result.drop(columns=['Year', 'Industry title'],
                                    inplace=True)
                        Result = Result.to_dict('records')[0]

                        Category = Result['Industry function category']

                    else:

                        m_recycled_for_no_industrial_cycle = m_recycled_for_no_industrial_cycle + Flow_transferred
                        n_non_industrial = n_non_industrial + 1

                        # Analyzing non-industrial activities
                        Result =\
                        self._checking_commercial_and_consumer_product(Commercial_consumer.copy(),
                                                                       NAICS_code,
                                                                       FRS_ID,
                                                                       How)
                        del Commercial_consumer

                        Category = Result['Product category'][0]

                    Output.update({'Option': [Option.capitalize()]})
                    Output.update(Result)
                    del Result

                    # Analyzing other chemicals that may flow with the
                    # chemical of interest
                    if Sale == 'Yes':
                        Evidence =\
                            self._checking_possible_chemicals(On_recycling,
                                                              RETDF_selected,
                                                              RETDF_reporting_year,
                                                              NAICS_code)
                        df_ratio = pd.DataFrame(Evidence)
                        df_ratio['Option'] = Option
                        df_ratio['Category'] = Category
                        df_ratio = df_ratio.loc[df_ratio['Evidence'] != 'No']
                        if not df_ratio.empty:
                            df_ratio.drop(columns=['Evidence'], inplace=True)
                            df_ratio_loop = pd.concat([df_ratio_loop, df_ratio],
                                                      ignore_index=True,
                                                      sort=True, axis=0)
                            del df_ratio
                            df_ratio_loop =\
                                df_ratio_loop.groupby(['Category',
                                                       'Chemical',
                                                       'Option'],
                                                      as_index=False)\
                                .apply(lambda x:
                                       self._selecting_highest_degree(x,
                                                                      n_loop))
                            n_loop = n_loop + 1

                df_output = pd.concat([df_output,
                                       pd.DataFrame(Output)],
                                      ignore_index=True,
                                      sort=True, axis=0)

            # Loop total
            m_total = m_total + m_total_cycle
            m_total_2 = m_total_2 + m_total_cycle**2
            m_transfer_for_recycling = m_transfer_for_recycling + m_transfer_for_recycling_cycle
            m_transfer_for_recycling_2 = m_transfer_for_recycling_2 + m_transfer_for_recycling_cycle**2
            m_recycled_for_sale = m_recycled_for_sale + m_recycled_for_sale_cycle
            m_recycled_for_sale_2 = m_recycled_for_sale_2 + m_recycled_for_sale_cycle**2
            m_recycled_for_no_sale = m_recycled_for_no_sale + m_recycled_for_no_sale_cycle
            m_recycled_for_no_sale_2 = m_recycled_for_no_sale_2 + m_recycled_for_no_sale_cycle**2
            m_recycled_for_industrial = m_recycled_for_industrial + m_recycled_for_industrial_cycle
            m_recycled_for_industrial_2 = m_recycled_for_industrial_2 + m_recycled_for_industrial_cycle**2
            m_recycled_for_no_industrial = m_recycled_for_no_industrial + m_recycled_for_no_industrial_cycle
            m_recycled_for_no_industrial_2 = m_recycled_for_no_industrial_2 + m_recycled_for_no_industrial_cycle**2
            # Filling null values in the flows
            num_cols = df_output._get_numeric_data().columns
            cat_cols = [col for col in df_output.columns
                        if col not in num_cols]
            df_output[num_cols] =\
                df_output[num_cols]\
                .where(pd.notnull(df_output[num_cols]),
                       0.0)
            # Filling null values in categorical
            df_output[cat_cols] =\
                df_output[cat_cols]\
                .where(pd.notnull(df_output[cat_cols]),
                       'N/N')
            # Dataframe for Markov Network joint probability
            Network_nodes = list(set(Network_nodes)
                                 .intersection(set(list(df_output.columns))))
            df_markov_network_loop = df_output[Network_nodes + ['Flow transferred']]
            df_markov_network_loop.rename(columns={'Flow transferred': 'Times'},
                                          inplace=True)
            df_markov_network = pd.concat([df_markov_network,
                                           df_markov_network_loop],
                                          ignore_index=True,
                                          sort=True, axis=0)
            df_markov_network =\
                df_markov_network.groupby(Network_nodes,
                                          as_index=False).sum()
            # Dataframe for total flow and std for Markov Network
            df_output.loc[df_output['Recycled phase']
                          != 'N/N', 'Recycled phase'] = 'P' # For product (P)
            df_output =\
                df_output.groupby(cat_cols,
                                  as_index=False).sum()
            df_output['Cycle'] = i + 1
            df_flow_MN = pd.concat([df_flow_MN,
                                    df_output],
                                   ignore_index=True,
                                   sort=True, axis=0)

        # Queries for Markov Network
        Queries = {1: 'General',
                   2: {'Type_of_waste_management': 'Recycling'}}
        # Sankey diagrams for queries
        sankey_diagram(df_flow_MN.copy(),
                       df_markov_network.copy(),
                       Queries, self._dir_path, True,
                       self.n_cycles)
        # Chord diagrams for inter industry transfers
        chord_diagram(df_flow_MN.copy(),
                      self.n_cycles,
                      self._dir_path)
        # Ratio of chemicals
        heatmap_diagram(df_ratio_loop.copy(),
                        self.chem,
                        self._dir_path)
        # Stacked bar plot
        stacked_barplot(df_markov_network.copy(),
                        self._dir_path)
        # Printing some results
        ## Total transfer
        m_total_average = m_total/self.n_cycles
        m_total_std = (m_total_2/self.n_cycles - m_total_average**2)**0.5
        try:
            m_total_cv = m_total_std/m_total_average
        except ZeroDivisionError:
            m_total_cv = 0
        ## Recycling transfer
        m_transfer_for_recycling_avg = m_transfer_for_recycling/self.n_cycles
        m_transfer_for_recycling_std = (m_transfer_for_recycling_2/self.n_cycles - m_transfer_for_recycling_avg**2)**0.5
        try:
            m_transfer_for_recycling_cv = m_transfer_for_recycling_std/m_transfer_for_recycling_avg
        except ZeroDivisionError:
            m_transfer_for_recycling_cv = 0
        ## Sale
        m_recycled_for_sale_avg = m_recycled_for_sale/self.n_cycles
        m_recycled_for_sale_std = (m_recycled_for_sale_2/self.n_cycles - m_recycled_for_sale_avg**2)**0.5
        try:
            m_recycled_for_sale_cv = m_recycled_for_sale_std/m_recycled_for_sale_avg
        except ZeroDivisionError:
            m_recycled_for_sale_cv = 0
        ## No sale
        m_recycled_for_no_sale_avg = m_recycled_for_no_sale/self.n_cycles
        m_recycled_for_no_sale_std = (m_recycled_for_no_sale_2/self.n_cycles - m_recycled_for_no_sale_avg**2)**0.5
        try:
            m_recycled_for_no_sale_cv = m_recycled_for_no_sale_std/m_recycled_for_no_sale_avg
        except ZeroDivisionError:
            m_recycled_for_no_sale_cv = 0
        ## For industrial
        m_recycled_for_industrial_avg = m_recycled_for_industrial/self.n_cycles
        m_recycled_for_industrial_std = (m_recycled_for_industrial_2/self.n_cycles - m_recycled_for_industrial_avg**2)**0.5
        try:
            m_recycled_for_industrial_cv = m_recycled_for_industrial_std/m_recycled_for_industrial_avg
        except ZeroDivisionError:
            m_recycled_for_industrial_cv = 0
        ## For no industrial
        m_recycled_for_no_industrial_avg = m_recycled_for_no_industrial/self.n_cycles
        m_recycled_for_no_industrial_std = (m_recycled_for_no_industrial_2/self.n_cycles - m_recycled_for_no_industrial_avg**2)**0.5
        try:
            m_recycled_for_no_industrial_cv = m_recycled_for_no_industrial_std/m_recycled_for_no_industrial_avg
        except ZeroDivisionError:
            m_recycled_for_no_industrial_cv = 0

        print('*'*45)
        print(f'Average total mass is {round(m_total_average)} kg/yr')
        print(f'CV for total mass is {round(m_total_cv, 2)}')
        print('*'*45)
        print(f'% of times that recycling ocurred {round(n_recycling*100/n_total, 2)}')
        print(f'Average recycling mass transfer is {round(m_transfer_for_recycling_avg)} kg/yr')
        print(f'CV for recycling mass transfer is {round(m_transfer_for_recycling_cv, 2)}')
        print('*'*45)
        print(f'% of times that the recycled flow was sold by the RETDF: {round(n_sale/n_recycling*100, 2)}')
        print(f'Average recycling mass sold by RETDF is {round(m_recycled_for_sale_avg)} kg/yr')
        print(f'CV for recycling mass sold by RETDF is {round(m_recycled_for_sale_cv, 2)}')
        print('*'*45)
        print(f'% of times that the recycled flow was not sold by the RETDF: {round(n_no_sale/n_recycling*100, 2)}')
        print(f'Average recycling mass not sold by RETDF is {round(m_recycled_for_no_sale_avg)} kg/yr')
        print(f'CV for recycling mass not sold by RETDF is {round(m_recycled_for_no_sale_cv, 2)}')
        print('*'*4)
        print(f'% of times that the recycled flow was used for industrial: {round(n_industrial/n_recycling*100, 2)}')
        print(f'Average recycling mass for industrial is {round(m_recycled_for_industrial_avg)} kg/yr')
        print(f'CV for recycling mass for industrial  is {round(m_recycled_for_no_sale_cv, 2)}')
        print('*'*4)
        print(f'% of times that the recycled flow was used for no industrial: {round(n_non_industrial/n_recycling*100, 2)}')
        print(f'Average recycling mass for no industrial is {round(m_recycled_for_no_industrial_avg)} kg/yr')
        print(f'CV for recycling mass for no industrial  is {round(m_recycled_for_no_industrial_cv, 2)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('-CAS',
                        help='Enter the TRI ID of the chemical(s) you want to analyze.',
                        type=str,
                        required=False,
                        default='110543')

    parser.add_argument('-N_cycles',
                        help='Enter the number of cycles you want to run.',
                        type=int,
                        required=False,
                        default=1)


    args = parser.parse_args()
    n_cycles = args.N_cycles
    chem = args.CAS

    start_time = time.time()
    Network = Network(n_cycles, chem)
    Network.loop()
    print('*'*45)
    print('Execution time: %s sec' % (time.time() - start_time))
