# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Note:
# 1. Range of Influent Concentration was reported from 1987 through 2004
# 2. Treatment Efficiency Estimation was reported from 1987 through 2004

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import argparse
import numpy as np
import re
import time
import unicodedata
from itertools import combinations
from ancillary.normalizing_naics.normalizing import normalizing_naics


class PAU_DB:

    def __init__(self, Year):
        self._dir_path = os.path.dirname(os.path.realpath(__file__)) # Working Directory
        self.Year = Year
        #self._dir_path = os.getcwd() # if you are working on Jupyter Notebook

    def calling_tri_files(self):
        TRI_Files = dict()
        for file in ['1a', '1b', '2b']:
            columns = pd.read_csv(self._dir_path + '/../../ancillary/tri/TRI_File_' + file + '_needed_columns_PAU4Chem.txt',
                                header = None)
            columns =  list(columns.iloc[:,0])
            df = pd.read_csv(self._dir_path + '/../extract/datasets/US_' + file + '_' + str(self.Year) + '.csv',
                            usecols = columns,
                            low_memory = False)
            df = df.where(pd.notnull(df), None)
            TRI_Files.update({file: df})
        return TRI_Files

    def is_number(self, s):
        try:
            float(s)
            return True
        except (TypeError, ValueError):
            pass
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def _efficiency_estimation_to_range(self, x):
        if x != np.nan:
            x = np.abs(x)
            if (x >= 0.0) & (x <= 50.0):
                return 'E6'
            elif (x > 50.0) & (x <= 95.0):
                return 'E5'
            elif (x > 95.0) & (x <= 99.0):
                return 'E4'
            elif (x > 99.0) & (x <= 99.99):
                return 'E3'
            elif (x > 99.99) & (x <= 99.9999):
                return 'E2'
            elif (x > 99.9999):
                return 'E1'
        else:
            return None

    def _efficiency_estimation_empties_based_on_EPA_regulation(self, classification, HAP, RCRA):
        if RCRA == 'YES':
            if classification == 'DIOXIN':
                result = np.random.uniform(99.9999, 100)
                if self.Year >= 2005:
                    result = self._efficiency_estimation_to_range(result)
            else:
                result = np.random.uniform(99.99, 100)
                if self.Year >= 2005:
                    result = self._efficiency_estimation_to_range(result)
            return result
        elif HAP == 'YES':
            result = np.random.uniform(95, 100)
            if self.Year >= 2005:
                result = self._efficiency_estimation_to_range(result)
            return result
        else:
            return None

    def _calling_SRS(self):
        Acronyms = ['TRI_Chem', 'CAA', 'RCRA_F', 'RCRA_K', 'RCRA_P', 'RCRA_T', 'RCRA_U']
        srs_path = self._dir_path + '/../../ancillary/others/'
        Files = {Acronym: File for File in os.listdir(srs_path) for Acronym in Acronyms if Acronym in File}
        columns = ['ID', 'Internal Tracking Number']
        df = pd.read_csv(srs_path + Files['TRI_Chem'], low_memory=False,
                         usecols=['ID', 'Internal Tracking Number'],
                         converters={'ID': lambda x: str(int(x)) if re.search('^\d', x) else x},
                         dtype={'Internal Tracking Number': 'int64'})
        df = df.assign(HAP=['NO']*df.shape[0], RCRA=['NO']*df.shape[0])
        del Files['TRI_Chem']
        for Acronym, File in Files.items():
            col = 'HAP'
            if Acronym in Acronyms[2:]:
                col = 'RCRA'
            ITN = pd.read_csv(srs_path + File,
                              low_memory=False,
                              usecols=['Internal Tracking Number'],
                              dtype={'Internal Tracking Number': 'int64'})
            df.loc[df['Internal Tracking Number'].isin( ITN['Internal Tracking Number'].tolist()), col] = 'YES'
        df.drop(columns='Internal Tracking Number', inplace=True)
        df.rename(columns={'ID': 'TRI_CHEM_ID'}, inplace=True)
        return df

    def _changin_management_code_for_2004_and_prior(self, x, m_n):
        Change = pd.read_csv(self._dir_path + '/../../ancillary/others/Methods_TRI.csv',
                        usecols = ['Code 2004 and prior', 'Code 2005 and after'],
                        low_memory = False)
        if list(x.values).count(None) != m_n:
            y = {v:'T' if v in Change['Code 2004 and prior'].unique().tolist() else 'F' for v in x.values}
            result = [Change.loc[Change['Code 2004 and prior'] == v, 'Code 2005 and after'].tolist()[0] \
                    if s == 'T' else None for v, s in y.items()]
            L = len(result)
            result = result + [None]*(m_n - L)
            return result
        else:
            return [None]*m_n

    def organizing(self):
        dfs = self.calling_tri_files()
        df = dfs['2b'].where(pd.notnull(dfs['2b']), None)
        if self.Year >= 2005:
            df.drop(columns = df.iloc[:, list(range(18, 71, 13))].columns.tolist(), inplace = True)
        else:
            df.drop(columns = df.iloc[:, list(range(20, 73, 13))].columns.tolist(), inplace = True)
        df_PAUs = pd.DataFrame()
        Columns_0 = list(df.iloc[:, 0:8].columns)
        for i in range(5):
            Starting = 8 + 12*i
            Ending = Starting + 11
            Columns_1 = list(df.iloc[:, Starting:Ending + 1].columns)
            Columns = Columns_0 + Columns_1
            df_aux = df[Columns]
            Columns_to_change = {col: re.sub(r'STREAM [1-5] - ', '', col) for col in Columns_1}
            df_aux.rename(columns = Columns_to_change, inplace =  True)
            df_PAUs = pd.concat([df_PAUs, df_aux], ignore_index = True,
                                       sort = True, axis = 0)
            del Columns
        del df, df_aux
        cols =  list(df_PAUs.iloc[:, 9:17].columns)
        df_PAUs.dropna(subset = cols, how = 'all', axis = 0, inplace = True)
        if self.Year <= 2004:
            df_PAUs.dropna(subset = ['WASTE STREAM CODE', 'RANGE INFLUENT CONCENTRATION', \
                        'TREATMENT EFFICIENCY ESTIMATION'], how = 'any', axis = 0, inplace = True)
            df_PAUs.reset_index(inplace = True, drop = True)
            df_PAUs['METHOD CODE - 2004 AND PRIOR'] = df_PAUs[cols].apply(lambda x: None if  list(x).count(None) == len(cols) else ' + '.join(xx for xx in x if xx), axis = 1)
            df_PAUs[cols] = df_PAUs.apply(lambda row: pd.Series(self._changin_management_code_for_2004_and_prior(row[cols], len(cols))),
                                    axis =  1)
            df_PAUs = df_PAUs.loc[pd.notnull(df_PAUs[cols]).any(axis = 1)]
            df_PAUs['EFFICIENCY RANGE CODE'] = df_PAUs['TREATMENT EFFICIENCY ESTIMATION']\
                                      .apply(lambda x: self._efficiency_estimation_to_range(float(x)))
            df_PAUs.rename(columns = {'TREATMENT EFFICIENCY ESTIMATION': 'EFFICIENCY ESTIMATION'}, inplace = True)
            mask = pd.to_numeric(df_PAUs['RANGE INFLUENT CONCENTRATION'], errors='coerce').notnull()
            df_PAUs = df_PAUs[mask]
            df_PAUs['RANGE INFLUENT CONCENTRATION'] = df_PAUs['RANGE INFLUENT CONCENTRATION'].apply(lambda x: abs(int(x)))
        else:
            df_PAUs.rename(columns = {'TREATMENT EFFICIENCY RANGE CODE': 'EFFICIENCY RANGE CODE'}, inplace = True)
            df_PAUs.dropna(subset = ['WASTE STREAM CODE', 'EFFICIENCY RANGE CODE'],
                            how = 'any', axis = 0, inplace = True)
        df_PAUs['METHOD CODE - 2005 AND AFTER'] = df_PAUs[cols].apply(lambda x: None if  list(x).count(None) == len(cols) else ' + '.join(xx for xx in x if xx), axis = 1)
        df_PAUs = df_PAUs.loc[pd.notnull(df_PAUs['METHOD CODE - 2005 AND AFTER'])]
        df_PAUs['TYPE OF MANAGEMENT'] = 'Treatment'
        df_PAUs.drop(columns = cols, inplace = True)
        df_PAUs.reset_index(inplace =  True, drop =  True)
        df_PAUs.loc[pd.isnull(df_PAUs['BASED ON OPERATING DATA?']), 'BASED ON OPERATING DATA?'] = 'NO'
        try:
            # On-site energy recovery
            df = dfs['1a'].iloc[:, list(range(12))]
            cols = [c for c in df.columns if 'METHOD' in c]
            df.dropna(subset = cols, how = 'all', axis = 0, inplace = True)
            Columns_0 = list(df.iloc[:, 0:8].columns)
            Columns_1 = list(df.iloc[:, 8:].columns)
            dfs_energy = pd.DataFrame()
            for col in Columns_1:
                Columns = Columns_0 + [col]
                df_aux = df[Columns]
                df_aux.rename(columns = {col: re.sub(r' [1-4]', '', col)},
                                inplace =  True)
                dfs_energy = pd.concat([dfs_energy, df_aux], ignore_index = True,
                                           sort = True, axis = 0)
                del Columns
            del df, df_aux
            dfs_energy = dfs_energy.loc[pd.notnull(dfs_energy['ON-SITE ENERGY RECOVERY METHOD'])]
            dfs_energy['TYPE OF MANAGEMENT'] = 'Energy recovery'
            if self.Year <= 2004:
                dfs_energy['METHOD CODE - 2004 AND PRIOR'] = dfs_energy['ON-SITE ENERGY RECOVERY METHOD']
                dfs_energy['ON-SITE ENERGY RECOVERY METHOD'] = dfs_energy.apply(lambda row: \
                            pd.Series(self._changin_management_code_for_2004_and_prior(pd.Series(row['ON-SITE ENERGY RECOVERY METHOD']), 1)),
                                            axis =  1)
                dfs_energy = dfs_energy.loc[pd.notnull(dfs_energy['ON-SITE ENERGY RECOVERY METHOD'])]
            dfs_energy.rename(columns = {'ON-SITE ENERGY RECOVERY METHOD': 'METHOD CODE - 2005 AND AFTER'},
                            inplace =  True)
            dfs_energy = dfs_energy.loc[pd.notnull(dfs_energy['METHOD CODE - 2005 AND AFTER'])]
            df_PAUs = pd.concat([df_PAUs, dfs_energy], ignore_index = True,
                                   sort = True, axis = 0)
            del dfs_energy
        except ValueError as e:
            print('{}:\nThere is not information about energy recovery activities'.format(e))
        try:
            # On-site recycling
            df = dfs['1a'].iloc[:, list(range(8)) + list(range(12,19))]
            cols = [c for c in df.columns if 'METHOD' in c]
            df.dropna(subset = cols, how = 'all', axis = 0, inplace = True)
            Columns_0 = list(df.iloc[:, 0:8].columns)
            Columns_1 = list(df.iloc[:, 8:].columns)
            dfs_recycling = pd.DataFrame()
            for col in Columns_1:
                Columns = Columns_0 + [col]
                df_aux = df[Columns]
                df_aux.rename(columns = {col: re.sub(r' [1-7]', '', col)},
                                inplace =  True)
                dfs_recycling = pd.concat([dfs_recycling, df_aux], ignore_index = True,
                                           sort = True, axis = 0)
                del Columns
            del df, df_aux
            dfs_recycling = dfs_recycling.loc[pd.notnull(dfs_recycling['ON-SITE RECYCLING PROCESSES METHOD'])]
            dfs_recycling['TYPE OF MANAGEMENT'] = 'Recycling'
            if self.Year <= 2004:
                dfs_recycling['METHOD CODE - 2004 AND PRIOR'] = dfs_recycling['ON-SITE RECYCLING PROCESSES METHOD']
                dfs_recycling['ON-SITE RECYCLING PROCESSES METHOD'] = dfs_recycling.apply(lambda row: \
                            pd.Series(self._changin_management_code_for_2004_and_prior(pd.Series(row['ON-SITE RECYCLING PROCESSES METHOD']), 1)),
                                            axis =  1)
                dfs_recycling = dfs_recycling.loc[pd.notnull(dfs_recycling['ON-SITE RECYCLING PROCESSES METHOD'])]
            dfs_recycling.rename(columns = {'ON-SITE RECYCLING PROCESSES METHOD': 'METHOD CODE - 2005 AND AFTER'},
                            inplace =  True)
            dfs_recycling = dfs_recycling.loc[pd.notnull(dfs_recycling['METHOD CODE - 2005 AND AFTER'])]
            df_PAUs = pd.concat([df_PAUs, dfs_recycling], ignore_index = True,
                                   sort = True, axis = 0)
            del dfs_recycling
        except ValueError as e:
            print('{}:\nThere is not information about recycling activities'.format(e))
        # Changing units
        df_PAUs = df_PAUs.loc[(df_PAUs.iloc[:,0:] != 'INV').all(axis = 1)]
        df_PAUs.dropna(how = 'all', axis = 0, inplace = True)
        if self.Year >= 2005:
            Change = pd.read_csv(self._dir_path + '/../../ancillary/others/Methods_TRI.csv',
                            usecols = ['Code 2004 and prior', 'Code 2005 and after'],
                            low_memory = False)
            Codes_2004 = Change.loc[(pd.notnull(Change['Code 2004 and prior'])) \
                        & (Change['Code 2005 and after'] != Change['Code 2004 and prior']),\
                        'Code 2004 and prior'].unique().tolist()
            idx = df_PAUs.loc[df_PAUs['METHOD CODE - 2005 AND AFTER'].isin(Codes_2004)].index.tolist()
            del Change, Codes_2004
            if len(idx) != 0:
                df_PAUs.loc[idx, 'METHOD CODE - 2005 AND AFTER'] = \
                df_PAUs.loc[idx]\
                            .apply(lambda row: self._changin_management_code_for_2004_and_prior(\
                            pd.Series(row['METHOD CODE - 2005 AND AFTER']),\
                            1),
                            axis = 1)
        # Adding methods name
        Methods = pd.read_csv(self._dir_path + '/../../ancillary/others/Methods_TRI.csv',
                            usecols = ['Code 2004 and prior',
                                    'Method 2004 and prior',
                                    'Code 2005 and after',
                                    'Method 2005 and after'])
        Methods.drop_duplicates(keep =  'first', inplace = True)
        # Adding chemical activities and uses
        df_PAUs['DOCUMENT CONTROL NUMBER'] = df_PAUs['DOCUMENT CONTROL NUMBER'].apply(lambda x: str(int(float(x))) if self.is_number(x) else x)
        dfs['1b'].drop_duplicates(keep = 'first', inplace = True)
        dfs['1b']['DOCUMENT CONTROL NUMBER'] = dfs['1b']['DOCUMENT CONTROL NUMBER'].apply(lambda x: str(int(float(x))) if self.is_number(x) else x)
        df_PAUs = pd.merge(df_PAUs, dfs['1b'], on = ['TRIFID', 'DOCUMENT CONTROL NUMBER', 'TRI_CHEM_ID'],
                                               how = 'inner')
        columns_DB_F = ['REPORTING YEAR', 'TRIFID', 'PRIMARY NAICS CODE', 'TRI_CHEM_ID',
                         'CHEMICAL NAME', 'METAL INDICATOR', 'CLASSIFICATION',
                         'PRODUCE THE CHEMICAL', 'IMPORT THE CHEMICAL',
                         'ON-SITE USE OF THE CHEMICAL','SALE OR DISTRIBUTION OF THE CHEMICAL',
                         'AS A BYPRODUCT', 'AS A MANUFACTURED IMPURITY', 'USED AS A REACTANT',
                         'ADDED AS A FORMULATION COMPONENT', 'USED AS AN ARTICLE COMPONENT',
                         'REPACKAGING', 'AS A PROCESS IMPURITY', 'RECYCLING',
                         'USED AS A CHEMICAL PROCESSING AID', 'USED AS A MANUFACTURING AID',
                         'ANCILLARY OR OTHER USE',
                         'WASTE STREAM CODE', 'METHOD CODE - 2005 AND AFTER',
                         'METHOD NAME - 2005 AND AFTER', 'TYPE OF MANAGEMENT',
                         'EFFICIENCY RANGE CODE', 'BASED ON OPERATING DATA?']
        if self.Year <= 2004:
            Method = {row.iloc[0]: row.iloc[1] for index, row in Methods[['Code 2004 and prior', 'Method 2004 and prior']].iterrows()}
            def _checking(x, M):
                if x:
                    return ' + '.join(M[xx] for xx in x.split(' + ') if xx and xx and xx in M.keys())
                else:
                    return None
            df_PAUs = df_PAUs.loc[df_PAUs['METHOD CODE - 2004 AND PRIOR'].str.contains(r'[A-Z]').where(df_PAUs['METHOD CODE - 2004 AND PRIOR'].str.contains(r'[A-Z]'), False)]
            df_PAUs['METHOD NAME - 2004 AND PRIOR'] = df_PAUs['METHOD CODE - 2004 AND PRIOR'].apply(lambda x: _checking(x, Method))
            df_PAUs = df_PAUs.loc[(df_PAUs['METHOD CODE - 2004 AND PRIOR'] != '') | (pd.notnull(df_PAUs['METHOD CODE - 2004 AND PRIOR']))]
            columns_DB_F = ['REPORTING YEAR', 'TRIFID', 'PRIMARY NAICS CODE', 'TRI_CHEM_ID',
                             'CHEMICAL NAME', 'METAL INDICATOR', 'CLASSIFICATION',
                             'PRODUCE THE CHEMICAL', 'IMPORT THE CHEMICAL', 'ON-SITE USE OF THE CHEMICAL',
                             'SALE OR DISTRIBUTION OF THE CHEMICAL', 'AS A BYPRODUCT',
                             'AS A MANUFACTURED IMPURITY', 'USED AS A REACTANT',
                             'ADDED AS A FORMULATION COMPONENT', 'USED AS AN ARTICLE COMPONENT',
                             'REPACKAGING', 'AS A PROCESS IMPURITY', 'RECYCLING',
                             'USED AS A CHEMICAL PROCESSING AID', 'USED AS A MANUFACTURING AID',
                             'ANCILLARY OR OTHER USE',
                             'WASTE STREAM CODE', 'RANGE INFLUENT CONCENTRATION',
                             'METHOD CODE - 2004 AND PRIOR', 'METHOD NAME - 2004 AND PRIOR',
                             'METHOD CODE - 2005 AND AFTER', 'METHOD NAME - 2005 AND AFTER',
                             'TYPE OF MANAGEMENT', 'EFFICIENCY RANGE CODE', 'EFFICIENCY ESTIMATION',
                             'BASED ON OPERATING DATA?']
        Method = {row.iloc[0]: row.iloc[1] for index, row in Methods[['Code 2005 and after', 'Method 2005 and after']].iterrows()}
        df_PAUs = df_PAUs.loc[df_PAUs['METHOD CODE - 2005 AND AFTER'].str.contains(r'[A-Z]').where(df_PAUs['METHOD CODE - 2005 AND AFTER'].str.contains(r'[A-Z]'), False)]
        df_PAUs['METHOD NAME - 2005 AND AFTER'] = df_PAUs['METHOD CODE - 2005 AND AFTER'].apply(lambda x: ' + '.join(Method[xx] for xx in x.split(' + ') if xx and xx in Method.keys()))
        # Saving information
        df_PAUs['REPORTING YEAR'] = self.Year
        df_PAUs = df_PAUs[columns_DB_F]
        df_PAUs.to_csv(self._dir_path + '/datasets/intermediate_pau_datasets/PAUs_DB_' + str(self.Year) + '.csv',
                     sep = ',', index = False)

    def Building_database_for_statistics(self):
        columns = pd.read_csv(self._dir_path + '/../../ancillary/tri/TRI_File_2b_needed_columns_for_statistics.txt',
                             header = None)
        columns =  list(columns.iloc[:,0])
        df = pd.read_csv(self._dir_path + '/../extract/datasets/US_2b_' + str(self.Year) + '.csv',
                         usecols=columns,
                         low_memory=False)
        df_statistics = pd.DataFrame()
        if self.Year >= 2005:
            df.drop(columns=df.iloc[:, list(range(12, 61, 12))].columns.tolist(), inplace = True)
            codes_incineration = ['A01', 'H040', 'H076', 'H122']
        else:
            df.drop(columns=df.iloc[:, list(range(13, 62, 12))].columns.tolist(), inplace = True)
            codes_incineration = ['A01', 'F01', 'F11', 'F19', 'F31',
                                'F41', 'F42', 'F51', 'F61',
                                'F71', 'F81', 'F82', 'F83',
                                'F99']
        Columns_0 = list(df.iloc[:, 0:2].columns)
        for i in range(5):
            Columns_1 = list(df.iloc[:, [2 + 11*i, 11 + 11*i, 12 + 11*i]].columns)
            Treatmes = list(df.iloc[:, 3 + 11*i: 11 + 11*i].columns)
            Columns = Columns_0 + Columns_1
            df_aux = df[Columns]
            df_aux['INCINERATION'] = 'NO'
            df_aux.loc[df[Treatmes].isin(codes_incineration).any(axis = 1), 'INCINERATION'] = 'YES'
            df_aux['IDEAL'] = df[Treatmes].apply(lambda x: 'YES' if  \
                                            len(list(np.where(pd.notnull(x))[0])) == 1  \
                                            else 'NO',
                                            axis = 1)
            Columns_to_change = {col: re.sub(r'STREAM [1-5] - ', '', col) for col in Columns_1}
            df_aux.rename(columns = Columns_to_change, inplace =  True)
            df_statistics = pd.concat([df_statistics, df_aux], ignore_index = True,
                                       sort = True, axis = 0)
            del Columns
        del df, df_aux
        if self.Year <= 2004:
            df_statistics.dropna(how = 'any', axis = 0, inplace = True)
            mask = pd.to_numeric(df_statistics['TREATMENT EFFICIENCY ESTIMATION'], errors='coerce').notnull()
            df_statistics = df_statistics[mask]
            df_statistics['TREATMENT EFFICIENCY ESTIMATION'] = df_statistics['TREATMENT EFFICIENCY ESTIMATION'].astype(float)
            df_statistics['EFFICIENCY RANGE'] = df_statistics['TREATMENT EFFICIENCY ESTIMATION']\
                                  .apply(lambda x: self._efficiency_estimation_to_range(float(x)))
            mask = pd.to_numeric(df_statistics['RANGE INFLUENT CONCENTRATION'], errors='coerce').notnull()
            df_statistics = df_statistics[mask]
            df_statistics['RANGE INFLUENT CONCENTRATION'] = df_statistics['RANGE INFLUENT CONCENTRATION'].astype(int)
            df_statistics.rename(columns = {'TREATMENT EFFICIENCY ESTIMATION': 'EFFICIENCY ESTIMATION'},
                                inplace = True)
        else:
            df_statistics.rename(columns = {'TREATMENT EFFICIENCY RANGE CODE': 'EFFICIENCY RANGE'},
                                inplace = True)
            df_statistics.dropna(subset = ['EFFICIENCY RANGE', 'WASTE STREAM CODE'], how = 'any', axis = 0, inplace = True)
        df_statistics.rename(columns = {'PRIMARY NAICS CODE': 'NAICS',
                                    'TRI_CHEM_ID': 'CAS',
                                    'WASTE STREAM CODE': 'WASTE',
                                    'RANGE INFLUENT CONCENTRATION': 'CONCENTRATION'},
                            inplace = True)
        df_statistics.loc[df_statistics['INCINERATION'] == 'NO', 'IDEAL'] = None
        df_statistics.to_csv(self._dir_path + '/statistics/db_for_general/DB_for_Statistics_' + str(self.Year) + '.csv',
                     sep = ',', index = False)


    def Building_database_for_recycling_efficiency(self):
        def _division(row, elements_total):
            if row['ON-SITE - RECYCLED'] == 0.0:
                if row['CLASSIFICATION'] == 'TRI':
                    row['ON-SITE - RECYCLED'] = 0.5
                elif row['CLASSIFICATION'] == 'PBT':
                    row['ON-SITE - RECYCLED'] = 0.1
                else:
                    row['ON-SITE - RECYCLED'] = 0.0001
            values = [abs(v) for v in row[elements_total] if v != 0.0]
            cases = list()
            for n_elements_sum in range(1, len(values) + 1):
                comb = combinations(values, n_elements_sum)
                for comb_values in comb:
                    sumatory = sum(comb_values)
                    cases.append(row['ON-SITE - RECYCLED']/(row['ON-SITE - RECYCLED'] + sumatory)*100)
            try:
                if len(list(set(cases))) == 1 and cases[0] == 100:
                    return [100]*6 + [0] + [row['ON-SITE - RECYCLED']]
                else:
                    return [np.min(cases)] + np.quantile(cases, [0.25, 0.5, 0.75]).tolist() + \
                           [np.max(cases), np.mean(cases), np.std(cases)/np.mean(cases),\
                            row['ON-SITE - RECYCLED']]
            except ValueError:
                a = np.empty((8))
                a[:] = np.nan
                return a.tolist()
        columns = pd.read_csv(self._dir_path + '/../../ancillary/tri/TRI_File_1a_needed_columns_for_statistics.txt',
                             header = None)
        columns =  list(columns.iloc[:,0])
        df = pd.read_csv(self._dir_path + '/../extract/datasets/US_1a_' + str(self.Year) + '.csv',
                        usecols = columns,
                        low_memory = False)
        elements_total = list(set(df.iloc[:, 5:64].columns.tolist()) - set(['ON-SITE - RECYCLED']))
        df.iloc[:, 5:64] = df.iloc[:, 5:64].where(pd.notnull(df.iloc[:, 5:64]), 0.0)
        df.iloc[:, 5:64] = df.iloc[:, 5:64].apply(pd.to_numeric, errors='coerce')
        cols = [c for c in df.columns if 'METHOD' in c]
        df['IDEAL'] = df[cols].apply(lambda x: 'YES' if  \
                                         len(list(np.where(pd.notnull(x))[0])) == 1  \
                                         else 'NO',
                                         axis = 1)
        df = df.loc[df['IDEAL'] == 'YES']
        df['METHOD'] = df[cols].apply(lambda x: x.values[np.where(pd.notnull(x))[0]][0], axis = 1)
        df.drop(columns = ['IDEAL'] + cols, inplace = True)
        df = df.loc[df['METHOD'] != 'INV']
        df[['LOWER EFFICIENCY', 'Q1', 'Q2','Q3', 'UPPER EFFICIENCY',
            'MEAN OF EFFICIENCY', 'CV', 'ON-SITE - RECYCLED']]\
             = df.apply(lambda x: pd.Series(_division(x, elements_total)), axis = 1)
        df = df.loc[pd.notnull(df['UPPER EFFICIENCY'])]
        df['IQR'] = df.apply(lambda x: x['Q3'] - x['Q1'], axis = 1)
        df['Q1 - 1.5xIQR'] = df.apply(lambda x: 0 if x['Q1'] - 1.5*x['IQR'] < 0 \
                                    else x['Q1'] - 1.5*x['IQR'], axis = 1)
        df['Q3 + 1.5xIQR'] = df.apply(lambda x: 100 if x['Q3'] + 1.5*x['IQR'] > 100 \
                                    else x['Q3'] + 1.5*x['IQR'], axis = 1)
        df['UPPER EFFICIENCY OUTLIER?'] = df.apply(lambda x: 'YES' if x['UPPER EFFICIENCY'] > \
                                x['Q3 + 1.5xIQR'] else 'NO', axis = 1)
        df['LOWER EFFICIENCY OUTLIER?'] = df.apply(lambda x: 'YES' if x['LOWER EFFICIENCY'] < \
                                x['Q1 - 1.5xIQR'] else 'NO', axis = 1)
        df['HIGH VARIANCE?'] = df.apply(lambda x: 'YES' if x['CV'] > 1 else 'NO', axis = 1)
        df = df[['TRIFID', 'PRIMARY NAICS CODE', 'TRI_CHEM_ID', 'ON-SITE - RECYCLED', 'UNIT OF MEASURE', \
                'LOWER EFFICIENCY', 'LOWER EFFICIENCY OUTLIER?', 'Q1 - 1.5xIQR', 'Q1', \
                'Q2', 'Q3', 'Q3 + 1.5xIQR', 'UPPER EFFICIENCY', 'UPPER EFFICIENCY OUTLIER?', \
                'IQR', 'MEAN OF EFFICIENCY', 'CV', 'HIGH VARIANCE?', 'METHOD']]
        df.iloc[:, [5, 7, 8, 9, 10, 11, 12, 14, 15, 16]] = \
                df.iloc[:, [5, 7, 8, 9, 10, 11, 12, 14, 15, 16]].round(4)
        df.to_csv(self._dir_path + '/statistics/db_for_solvents/DB_for_Solvents_' + str(self.Year) + '.csv',
                      sep = ',', index = False)



    def _searching_naics(self, x, naics):
        # https://www.census.gov/programs-surveys/economic-census/guidance/understanding-naics.html
        values = {0:'Nothing',
                 1:'Nothing',
                 2:'Sector',
                 3:'Subsector',
                 4:'Industry Group',
                 5:'NAICS Industry',
                 6:'National Industry'}
        naics = str(naics)
        x = str(x)
        equal = 0
        for idx, char in enumerate(naics):
            try:
                if char == x[idx]:
                    equal = equal + 1
                else:
                    break
            except IndexError:
                break
        return values[equal]


    def _phase_estimation_recycling(self, df_s, row):
        if row['METHOD CODE - 2005 AND AFTER'] == 'H20': # Solvent recovery
            phases = ['L']
        elif row['METHOD CODE - 2005 AND AFTER'] == 'H39': # Acid regeneration and other reactions
            phases = ['W']
        elif row['METHOD CODE - 2005 AND AFTER'] == 'H10': # Metal recovery
            phases = ['W', 'S']
            if self.Year <= 2004:
                Pyrometallurgy = ['R27', 'R28', 'R29'] # They work with scrap
                if row['METHOD CODE - 2004 AND PRIOR'] in Pyrometallurgy:
                    Phases = ['S']
                else:
                    Phases = ['W', 'S']
        naics_structure = ['National Industry', 'NAICS Industry', 'Industry Group',
                    'Subsector', 'Sector', 'Nothing']
        df_cas = df_s.loc[df_s['CAS'] == row['TRI_CHEM_ID'], ['NAICS', 'WASTE', 'VALUE']]
        df_cas = df_cas.groupby(['NAICS', 'WASTE'], as_index = False).sum()
        df_cas.reset_index(inplace = True)
        if (not df_cas.empty):
            df_cas['NAICS STRUCTURE'] = df_cas.apply(lambda x: \
                            self._searching_naics(x['NAICS'], \
                            row['PRIMARY NAICS CODE']), \
                            axis = 1)
            i = 0
            phase = None
            while i <= 5 and (not phase in phases):
                structure = naics_structure[i]
                i = i + 1
            #for structure in naics_structure:
                df_naics = df_cas.loc[df_cas['NAICS STRUCTURE'] == structure]
                if (df_naics.empty):
                    phase = None
                    #continue
                else:
                    if (df_naics['WASTE'].isin(phases).any()):
                        df_phase = df_naics.loc[df_naics['WASTE'].isin(phases)]
                        row['NAICS STRUCTURE'] = structure
                        row['WASTE STREAM CODE'] = df_phase.loc[df_phase['VALUE'].idxmax(), 'WASTE']
                    else:
                        row['NAICS STRUCTURE'] = structure
                        row['WASTE STREAM CODE'] = df_naics.loc[df_naics['VALUE'].idxmax(), 'WASTE']
                    phase =  row['WASTE STREAM CODE']
            return row
        else:
            row['NAICS STRUCTURE'] = None
            row['WASTE STREAM CODE'] = None
            return row


    def _concentration_estimation_recycling(self, df_s, cas, naics, phase, structure):
        df_s = df_s[['NAICS', 'CAS', 'WASTE', 'CONCENTRATION', 'VALUE']]
        df_s = df_s.loc[(df_s['CAS'] == cas) & \
                    (df_s['WASTE'] == phase)]
        df_s = df_s.groupby(['NAICS', 'CAS', 'WASTE', 'CONCENTRATION'], as_index = False).sum()
        df_s['NAICS STRUCTURE'] = df_s.apply(lambda x: \
                        self._searching_naics(x['NAICS'], \
                        naics), \
                        axis = 1)
        df = df_s.loc[(df_s['NAICS STRUCTURE'] == structure)]
        return df.loc[df['VALUE'].idxmax(), 'CONCENTRATION']


    def _recycling_efficiency(self, row, df_s):
        naics_structure = ['National Industry', 'NAICS Industry', 'Industry Group',
                    'Subsector', 'Sector', 'Nothing']
        if self.Year <= 2004:
            code = row['METHOD CODE - 2004 AND PRIOR']
        else:
            code = row['METHOD CODE - 2005 AND AFTER']
        df_cas = df_s.loc[(df_s['TRI_CHEM_ID'] == row['TRI_CHEM_ID']) & (df_s['METHOD'] == code)]
        if (not df_cas.empty):
            df_fid = df_cas.loc[df_cas['TRIFID'] == row['TRIFID']]
            if (not df_fid.empty):
                return df_fid['UPPER EFFICIENCY'].iloc[0]
            else:
                df_cas['NAICS STRUCTURE'] = df_cas.apply(lambda x: \
                                self._searching_naics(x['PRIMARY NAICS CODE'], \
                                                    row['PRIMARY NAICS CODE']), \
                                axis = 1)
                i = 0
                efficiency = None
                while (i <= 5) and (not efficiency):
                    structure = naics_structure[i]
                    i = i + 1
                    df_naics = df_cas.loc[df_cas['NAICS STRUCTURE'] == structure]
                    if df_naics.empty:
                        efficiency = None
                    else:
                        df_naics['WEIGHT'] = df_naics['WEIGHT']/df_naics['WEIGHT'].sum()
                        df_naics.sort_values('UPPER EFFICIENCY', inplace=True)
                        df_naics['CUMULATIVE'] = df_naics['WEIGHT'].transform(pd.Series.cumsum)
                        cutoff = df_naics['WEIGHT'].sum() / 2.0
                        df_naics.reset_index(inplace=True, drop=True)
                        idx = df_naics.loc[df_naics['CUMULATIVE'] >= cutoff].index.tolist()[0]
                        efficiency = df_naics['UPPER EFFICIENCY'].iloc[idx]
                return efficiency
        else:
            return None


    def _phase_estimation_energy(self, df_s, row):
        phases = ['S', 'L', 'A']
        if row['METHOD CODE - 2005 AND AFTER'] == 'U01':
            phases = ['S', 'L'] # Industrial Kilns (specially rotatory kilns) are used to burn hazardous liquid and solid wastes
        naics_structure = ['National Industry', 'NAICS Industry', 'Industry Group',
                    'Subsector', 'Sector', 'Nothing']
        df_cas = df_s.loc[df_s['CAS'] == row['TRI_CHEM_ID'], ['NAICS', 'WASTE', 'VALUE', 'INCINERATION']]
        df_cas = df_cas.groupby(['NAICS', 'WASTE', 'INCINERATION'], as_index = False).sum()
        df_cas.reset_index(inplace = True)
        if (not df_cas.empty):
            df_cas['NAICS STRUCTURE'] = df_cas.apply(lambda x: \
                            self._searching_naics(x['NAICS'], \
                            row['PRIMARY NAICS CODE']), \
                            axis = 1)
            i = 0
            phase = None
            #for structure in naics_structure:
            while i <= 5 and (not phase in phases):
                structure = naics_structure[i]
                i = i + 1
                df_naics = df_cas.loc[df_cas['NAICS STRUCTURE'] == structure]
                if df_naics.empty:
                    phase = None
                else:
                    df_incineration = df_naics.loc[df_cas['INCINERATION'] == 'YES']
                    if df_incineration.empty:
                        if (df_naics['WASTE'].isin(phases).any()):
                            df_phase = df_naics.loc[df_naics['WASTE'].isin(phases)]
                            row['NAICS STRUCTURE'] = structure
                            row['WASTE STREAM CODE'] = df_phase.loc[df_phase['VALUE'].idxmax(), 'WASTE']
                        else:
                            row['NAICS STRUCTURE'] = structure
                            row['WASTE STREAM CODE'] = df_naics.loc[df_naics['VALUE'].idxmax(), 'WASTE']
                        row['BY MEANS OF INCINERATION'] = 'NO'
                    else:
                        if (df_incineration['WASTE'].isin(phases).any()):
                            df_phase = df_incineration.loc[df_incineration['WASTE'].isin(phases)]
                            row['NAICS STRUCTURE'] = structure
                            row['WASTE STREAM CODE'] = df_phase.loc[df_phase['VALUE'].idxmax(), 'WASTE']
                        else:
                            row['NAICS STRUCTURE'] = structure
                            row['WASTE STREAM CODE'] = df_incineration.loc[df_incineration['VALUE'].idxmax(), 'WASTE']
                        row['BY MEANS OF INCINERATION'] = 'YES'
                    phase =  row['WASTE STREAM CODE']
            return row
        else:
            row['NAICS STRUCTURE'] = None
            row['WASTE STREAM CODE'] = None
            row['BY MEANS OF INCINERATION'] = None
            return row


    def _concentration_estimation_energy(self, df_s, cas, naics, phase, structure, incineration):
        df_s = df_s[['NAICS', 'CAS', 'WASTE', 'CONCENTRATION', \
                    'VALUE', 'INCINERATION']]
        df_s = df_s.loc[(df_s['CAS'] == cas) & \
                    (df_s['WASTE'] == phase) & \
                    (df_s['INCINERATION'] == incineration)]
        df_s = df_s.groupby(['NAICS', 'CAS', 'WASTE', 'CONCENTRATION', 'INCINERATION'],
                    as_index = False).sum()
        df_s['NAICS STRUCTURE'] = df_s.apply(lambda x: \
                        self._searching_naics(x['NAICS'], \
                        naics), \
                        axis = 1)
        df = df_s.loc[(df_s['NAICS STRUCTURE'] == structure)]
        return df.loc[df['VALUE'].idxmax(), 'CONCENTRATION']


    def _energy_efficiency(self, df_s, row):
        if self.Year <= 2004:
            df_s = df_s[['NAICS', 'CAS', 'WASTE', 'INCINERATION', 'IDEAL', 'EFFICIENCY ESTIMATION']]
        else:
            df_s = df_s[['NAICS', 'CAS', 'WASTE', 'INCINERATION', 'IDEAL', 'EFFICIENCY RANGE', 'VALUE']]
            df_s['WASTE'] = df_s.groupby(['NAICS', 'CAS', 'WASTE', 'IDEAL', 'INCINERATION', 'EFFICIENCY RANGE'],
                    as_index = False).sum()
        df_s = df_s.loc[(df_s['CAS'] == row['TRI_CHEM_ID']) & \
                        (df_s['INCINERATION'] == 'YES') & \
                        (df_s['IDEAL'] == 'YES')]
        if (not df_s.empty):
            df_s['NAICS STRUCTURE'] = df_s.apply(lambda x: \
                        self._searching_naics(x['NAICS'], \
                        row['PRIMARY NAICS CODE']), \
                        axis = 1)
            df_structure = df_s.loc[df_s['NAICS STRUCTURE'] == row['NAICS STRUCTURE']]
            if (not df_structure.empty):
                df_phase = df_structure.loc[df_structure['WASTE'] \
                        == row['WASTE STREAM CODE']]
                if (not df_phase.empty):
                    if self.Year <= 2004:
                        result =  df_phase['EFFICIENCY ESTIMATION'].median()
                    else:
                        result = df_phase.loc[df_phase['VALUE'].idxmax(), 'EFFICIENCY RANGE']
                else:
                    if self.Year <= 2004:
                        result =  df_structure['EFFICIENCY ESTIMATION'].median()
                    else:
                        result = df_structure.loc[df_structure['VALUE'].idxmax(), 'EFFICIENCY RANGE']
            else:
                df_phase = df_s.loc[df_s['WASTE'] \
                        == row['WASTE STREAM CODE']]
                if (not df_phase.empty):
                    if self.Year <= 2004:
                        result =  df_phase['EFFICIENCY ESTIMATION'].median()
                    else:
                        result = df_phase.loc[df_phase['VALUE'].idxmax(), 'EFFICIENCY RANGE']
                else:
                    if self.Year <= 2004:
                        result =  df_s['EFFICIENCY ESTIMATION'].median()
                    else:
                        result = df_s.loc[df_s['VALUE'].idxmax(), 'EFFICIENCY RANGE']
        else:
            return None
        return result


    def cleaning_database(self):
        # Calling TRI restriction for metals
        Restrictions = pd.read_csv(self._dir_path + '/../../ancillary/others/Metals_divided_into_4_groups_can_be_reported.csv',
                                    low_memory = False,
                                    usecols = ['ID',
                                               "U01, U02, U03 (Energy recovery)",
                                               'H20 (Solvent recovey)'])
        Energy_recovery = Restrictions.loc[Restrictions["U01, U02, U03 (Energy recovery)"] == 'NO', 'ID'].tolist()
        Solvent_recovery = Restrictions.loc[Restrictions['H20 (Solvent recovey)'] == 'NO', 'ID'].tolist()
        # Calling PAU
        PAU = pd.read_csv(self._dir_path + '/datasets/intermediate_pau_datasets/PAUs_DB_' + str(self.Year) + '.csv',
                            low_memory = False,
                            converters = {'TRI_CHEM_ID': lambda x: x if re.search(r'^[A-Z]', x) else str(int(x))})
        columns_DB_F = PAU.columns.tolist()
        PAU['PRIMARY NAICS CODE'] = PAU['PRIMARY NAICS CODE'].astype('int')
        if self.Year <= 2004:
            grouping = ['TRIFID', 'METHOD CODE - 2004 AND PRIOR']
            PAU.sort_values(by = ['PRIMARY NAICS CODE', 'TRIFID',
                            'METHOD CODE - 2004 AND PRIOR', 'TRI_CHEM_ID'],
                            inplace = True)
        else:
            grouping = ['TRIFID', 'METHOD CODE - 2005 AND AFTER']
            PAU.sort_values(by = ['PRIMARY NAICS CODE', 'TRIFID',
                            'METHOD CODE - 2005 AND AFTER', 'TRI_CHEM_ID'],
                            inplace = True)
        # Calling database for statistics
        Statistics = pd.read_csv(self._dir_path + '/statistics/db_for_general/DB_for_Statistics_' + str(self.Year) + '.csv',
                                low_memory = False,
                                converters = {'CAS': lambda x: x if re.search(r'^[A-Z]', x) else str(int(x))})
        Statistics['NAICS'] = Statistics['NAICS'].astype('int')
        Statistics['VALUE'] = 1
        Statistics.sort_values(by = ['NAICS', 'CAS'], inplace = True)
        # Treatment
        Efficiency_codes = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']
        df_N_PAU = PAU.loc[PAU['TYPE OF MANAGEMENT'] == 'Treatment']
        df_N_PAU = df_N_PAU.loc[df_N_PAU['EFFICIENCY RANGE CODE'].isin(Efficiency_codes)]
        # Recycling
        PAU_recycling = PAU.loc[PAU['TYPE OF MANAGEMENT'] == 'Recycling']
        if not PAU_recycling.empty:
            PAU_recycling =  PAU_recycling.loc[~ ((PAU_recycling['METHOD CODE - 2005 AND AFTER'] == 'H20') & (PAU_recycling['TRI_CHEM_ID'].isin(Solvent_recovery)))]
            PAU_recycling.reset_index(inplace = True, drop = True)
            PAU_recycling['BASED ON OPERATING DATA?'] = 'NO'
            # Calling database for recycling efficiency
            Recycling_statistics = pd.read_csv(self._dir_path + '/statistics/db_for_solvents/DB_for_Solvents_' + str(self.Year) +  '.csv',
                                    low_memory = False,
                                    usecols = ['TRIFID', 'PRIMARY NAICS CODE', 'TRI_CHEM_ID', \
                                               'UPPER EFFICIENCY', 'UPPER EFFICIENCY OUTLIER?',
                                               'METHOD', 'HIGH VARIANCE?', 'CV'],
                                    converters = {'TRI_CHEM_ID': lambda x: x if re.search(r'^[A-Z]', x) else str(int(x))})
            Recycling_statistics['PRIMARY NAICS CODE'] = Recycling_statistics['PRIMARY NAICS CODE'].astype('int')
            Recycling_statistics = Recycling_statistics\
                                    .loc[(Recycling_statistics['UPPER EFFICIENCY OUTLIER?'] == 'NO') &
                                         (Recycling_statistics['HIGH VARIANCE?'] == 'NO')]
            Recycling_statistics['WEIGHT'] = Recycling_statistics['CV'].apply(lambda x: 1/x if x != 0 else 10)
            Max_value = Recycling_statistics['WEIGHT'].max()
            Recycling_statistics.loc[(Recycling_statistics['CV'] == 0) &
                                           (Recycling_statistics['WEIGHT'] != Max_value),
                                           'WEIGHT'] = Max_value + 1
            Recycling_statistics.drop(columns=['UPPER EFFICIENCY OUTLIER?',
                                               'HIGH VARIANCE?', 'CV'],
                                      axis=1)
            efficiency_estimation = \
                    PAU_recycling.apply(lambda x: self._recycling_efficiency(x, Recycling_statistics), axis = 1).round(4)
            PAU_recycling['EFFICIENCY RANGE CODE'] = \
                            efficiency_estimation.apply(lambda x: self._efficiency_estimation_to_range(x))
            PAU_recycling = PAU_recycling.loc[pd.notnull(PAU_recycling['EFFICIENCY RANGE CODE'])]
            PAU_recycling = \
                     PAU_recycling.apply(lambda x: \
                     self._phase_estimation_recycling(Statistics, x), axis = 1)
            PAU_recycling = PAU_recycling.loc[pd.notnull(PAU_recycling['WASTE STREAM CODE'])]
            if self.Year <= 2004:
                PAU_recycling['EFFICIENCY ESTIMATION'] = efficiency_estimation
                PAU_recycling['RANGE INFLUENT CONCENTRATION'] = \
                          PAU_recycling.apply(lambda x: \
                         self._concentration_estimation_recycling(Statistics, \
                                         x['TRI_CHEM_ID'], \
                                         x['PRIMARY NAICS CODE'],\
                                         x['WASTE STREAM CODE'], \
                                         x['NAICS STRUCTURE']), \
                                         axis = 1)
            PAU_recycling.drop(columns = ['NAICS STRUCTURE'], inplace = True)
            df_N_PAU = pd.concat([df_N_PAU, PAU_recycling],
                                     ignore_index = True,
                                     sort = True, axis = 0)
        else:
            pass
        # Energy recovery
        PAU_energy = PAU.loc[PAU['TYPE OF MANAGEMENT'] == 'Energy recovery']
        if not PAU_energy.empty:
            PAU_energy =  PAU_energy.loc[~ ((PAU_energy['METHOD CODE - 2005 AND AFTER'].isin(['U01', 'U02', 'U03'])) & (PAU_energy['TRI_CHEM_ID'].isin(Energy_recovery)))]
            PAU_energy.reset_index(inplace = True, drop = True)
            PAU_energy['BASED ON OPERATING DATA?'] = 'NO'
            PAU_energy = \
                     PAU_energy.apply(lambda x: \
                     self._phase_estimation_energy(Statistics, x), axis = 1)
            PAU_energy = PAU_energy.loc[pd.notnull(PAU_energy['WASTE STREAM CODE'])]
            SRS = self._calling_SRS()
            if self.Year <= 2004:
                PAU_energy['RANGE INFLUENT CONCENTRATION'] = \
                         PAU_energy.apply(lambda x: \
                         self._concentration_estimation_energy(Statistics, \
                                         x['TRI_CHEM_ID'], \
                                         x['PRIMARY NAICS CODE'],\
                                         x['WASTE STREAM CODE'], \
                                         x['NAICS STRUCTURE'], \
                                         x['BY MEANS OF INCINERATION']), \
                         axis = 1)
                PAU_energy.drop(columns = ['BY MEANS OF INCINERATION'], inplace = True)
                PAU_energy['EFFICIENCY ESTIMATION'] = \
                        PAU_energy.apply(lambda x: \
                        self._energy_efficiency(Statistics, x), axis = 1).round(4)
                PAU_energy = pd.merge(PAU_energy, SRS, on = 'TRI_CHEM_ID', how = 'left')
                PAU_energy['EFFICIENCY ESTIMATION'] = PAU_energy.apply(lambda x: \
                                    self._efficiency_estimation_empties_based_on_EPA_regulation(\
                                    x['CLASSIFICATION'], x['HAP'], x['RCRA']) \
                                    if not x['EFFICIENCY ESTIMATION'] else
                                    x['EFFICIENCY ESTIMATION'],
                                    axis =  1)
                PAU_energy = PAU_energy.loc[pd.notnull(PAU_energy['EFFICIENCY ESTIMATION'])]
                PAU_energy['EFFICIENCY RANGE CODE'] = PAU_energy['EFFICIENCY ESTIMATION']\
                                          .apply(lambda x: self._efficiency_estimation_to_range(float(x)))
            else:
                PAU_energy.drop(columns = ['BY MEANS OF INCINERATION'], inplace = True)
                PAU_energy['EFFICIENCY RANGE CODE'] = \
                        PAU_energy.apply(lambda x: \
                        self._energy_efficiency(Statistics, x), axis = 1)
                PAU_energy = pd.merge(PAU_energy, SRS, on = 'TRI_CHEM_ID', how = 'left')
                PAU_energy['EFFICIENCY RANGE CODE'] = PAU_energy.apply(lambda x: \
                                    self._efficiency_estimation_empties_based_on_EPA_regulation(\
                                    x['CLASSIFICATION'], x['HAP'], x['RCRA']) \
                                    if not x['EFFICIENCY RANGE CODE'] else
                                    x['EFFICIENCY RANGE CODE'],
                                    axis =  1)
                PAU_energy = PAU_energy.loc[pd.notnull(PAU_energy['EFFICIENCY RANGE CODE'])]
            PAU_energy.drop(columns = ['NAICS STRUCTURE', 'HAP', 'RCRA'], inplace = True)
            PAU_energy.loc[(PAU_energy['WASTE STREAM CODE'] == 'W') & \
                           (PAU_energy['TYPE OF MANAGEMENT'] == 'Energy recovery'),\
                            'WASTE STREAM CODE'] = 'L'
            df_N_PAU = pd.concat([df_N_PAU, PAU_energy],
                                     ignore_index = True,
                                     sort = True, axis = 0)
        else:
            pass
        Chemicals_to_remove = ['MIXTURE', 'TRD SECRT']
        df_N_PAU = df_N_PAU.loc[~df_N_PAU['TRI_CHEM_ID'].isin(Chemicals_to_remove)]
        df_N_PAU['TRI_CHEM_ID'] = df_N_PAU['TRI_CHEM_ID'].apply(lambda x: str(int(x)) if not 'N' in x else x)
        df_N_PAU = normalizing_naics(df_N_PAU,
                                    naics_column='PRIMARY NAICS CODE',
                                    column_year='REPORTING YEAR')
        df_N_PAU = df_N_PAU[columns_DB_F]
        df_N_PAU.to_csv(self._dir_path + '/datasets/final_pau_datasets/PAUs_DB_filled_' + str(self.Year) + '.csv',
                     sep = ',', index = False)
        # Chemicals and groups
        Chemicals = df_N_PAU[['TRI_CHEM_ID', 'CHEMICAL NAME']].drop_duplicates(keep = 'first')
        Chemicals['TYPE OF CHEMICAL'] = None
        Path_c = self._dir_path + '/chemicals/Chemicals.csv'
        if os.path.exists(Path_c):
            df_c = pd.read_csv(Path_c)
            for index, row in Chemicals.iterrows():
                if (df_c['TRI_CHEM_ID'] != row['TRI_CHEM_ID']).all():
                    df_c = df_c.append(pd.Series(row, index = row.index.tolist()), \
                                        ignore_index = True)
            df_c.to_csv(Path_c, sep = ',', index  = False)
        else:
            Chemicals.to_csv(Path_c, sep = ',', index = False)


    def Searching_information_for_years_after_2004(self):
        df_tri_older = pd.DataFrame()
        for year in range(1987, 2005):
            df_tri_older_aux = pd.read_csv(f'{self._dir_path }/datasets/final_pau_datasets/PAUs_DB_filled_{year}.csv',
                                           usecols=['TRIFID',
                                                    'TRI_CHEM_ID',
                                                    'WASTE STREAM CODE',
                                                    'METHOD CODE - 2004 AND PRIOR',
                                                    'METHOD NAME - 2004 AND PRIOR',
                                                    'METHOD CODE - 2005 AND AFTER',
                                                    'RANGE INFLUENT CONCENTRATION',
                                                    'EFFICIENCY ESTIMATION'],
                                           low_memory=False)
            df_tri_older = pd.concat([df_tri_older, df_tri_older_aux], ignore_index=True,
                                       sort=True, axis=0)
            del df_tri_older_aux
        df_tri_older.drop_duplicates(keep='first', inplace=True)
        df_PAU = pd.read_csv(f'{self._dir_path }/datasets/final_pau_datasets/PAUs_DB_filled_{self.Year}.csv',
                             low_memory=False)
        columns = df_PAU.columns.tolist()
        df_PAU.drop_duplicates(keep='first', inplace=True)
        df_PAU = pd.merge(df_PAU, df_tri_older, how='left',
                          on=['TRIFID',
                              'TRI_CHEM_ID',
                              'WASTE STREAM CODE',
                              'METHOD CODE - 2005 AND AFTER'])
        del df_tri_older
        df_PAU.drop_duplicates(keep='first', subset=columns, inplace=True)
        df_PAU['EQUAL RANGE'] = df_PAU[['EFFICIENCY RANGE CODE',
                                        'EFFICIENCY ESTIMATION']]\
                                        .apply(lambda x: x.values[0] == self._efficiency_estimation_to_range(x.values[1])
                                                        if x.values[1]
                                                        else True,
                                               axis=1)
        idx = df_PAU.loc[((df_PAU['EQUAL RANGE'] == False) & (pd.notnull(df_PAU['EFFICIENCY ESTIMATION'])))].index.tolist()
        df_PAU.drop(columns=['EQUAL RANGE'], inplace=True)
        df_PAU_aux = df_PAU.loc[idx]
        df_PAU_aux.drop(columns=['METHOD CODE - 2004 AND PRIOR',
                                 'METHOD NAME - 2004 AND PRIOR',
                                 'RANGE INFLUENT CONCENTRATION',
                                 'EFFICIENCY ESTIMATION'],
                        inplace=True)
        df_PAU.drop(idx, inplace=True)
        df_PAU = pd.concat([df_PAU, df_PAU_aux], ignore_index=True,
                                   sort=True, axis=0)
        del df_PAU_aux
        df_PAU.to_csv(f'{self._dir_path}/datasets/final_pau_datasets/PAUs_DB_filled_{self.Year}.csv',
                      sep=',', index=False)

    def dataset_for_individual_statistics(self):
        cols = ['TRIFID', 'TRI_CHEM_ID',
                'TYPE OF MANAGEMENT',
                'EFFICIENCY RANGE CODE',
                'METHOD NAME - 2005 AND AFTER',
                'WASTE STREAM CODE']
        if int(self.Year) <= 2004:
            cols.append('METHOD NAME - 2004 AND PRIOR')
            col_method = 'METHOD NAME - 2004 AND PRIOR'
        else:
            col_method = 'METHOD NAME - 2005 AND AFTER'

        path = f'{self._dir_path}/datasets/final_pau_datasets'
        df = pd.DataFrame()

        for year in range(1987, int(self.Year) + 1):
            df_aux = pd.read_csv(f'{path}/PAUs_DB_filled_{year}.csv',
                                 usecols=cols)
            df_aux = df_aux.loc[pd.notnull(df_aux).all(axis=1)]
            df_aux = df_aux.loc[df_aux['TYPE OF MANAGEMENT'] == 'Treatment']
            df_aux = df_aux.loc[~df_aux[col_method].str.contains(r' \+ ')]
            df = pd.concat([df, df_aux], ignore_index=True,
                           sort=True, axis=0)
        del df_aux
        df.drop_duplicates(keep='first', inplace=True)
        df.drop(columns=['TYPE OF MANAGEMENT',
                         'TRIFID'],
                inplace=True)
        df['NUMBER'] = 1

        # For other treatments
        if int(self.Year) <= 2004:
            tri_method = pd.read_csv(f'{self._dir_path}/../../ancillary/others/Methods_TRI.csv',
                                     usecols=['Method 2004 and prior',
                                              'Method 2005 and after',
                                              'If it is treatment, what kind of?'])
            tri_method = tri_method.loc[tri_method['Method 2005 and after'] == 'Other treatment']
            tri_method.drop(columns=['Method 2005 and after'], inplace=True)
            tri_method.rename(columns={'Method 2004 and prior': 'METHOD NAME - 2004 AND PRIOR',
                                       'If it is treatment, what kind of?': 'CHEMICAL/PHYSICAL'},
                              inplace=True)
            df = df.loc[df['METHOD NAME - 2005 AND AFTER'] == 'Other treatment']
            df = pd.merge(df, tri_method,
                         on='METHOD NAME - 2004 AND PRIOR',
                         how='inner')
            df.drop(columns=['METHOD NAME - 2005 AND AFTER',
                             'METHOD NAME - 2004 AND PRIOR'],
                    inplace=True)
            grouping = ['TRI_CHEM_ID',
                        'EFFICIENCY RANGE CODE',
                        'WASTE STREAM CODE',
                        'CHEMICAL/PHYSICAL']
            df = df.groupby(grouping, as_index=False).sum()
            df.to_csv(f'{self._dir_path}/statistics/other_treatments.csv',
                      sep=',', index=False)
        # For treatment in general
        else:
            grouping = ['TRI_CHEM_ID',
                        'EFFICIENCY RANGE CODE',
                        'METHOD NAME - 2005 AND AFTER',
                        'WASTE STREAM CODE']
            df = df.groupby(grouping, as_index=False).sum()
            df.to_csv(f'{self._dir_path}/statistics/individual_pau_statistics.csv',
                      sep=',', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)

    parser.add_argument('Option',
                        help = 'What do you want to do:\
                        [A]: Recover information from TRI.\
                        [B]: File for statistics. \
                        [C]: File for recycling. \
                        [D]: Further cleaning of database. \
                        [E]: Searching information for years after 2004. \
                        [F]: Dataset for individual statistics and support', \
                        type = str)

    parser.add_argument('-Y', '--Year', nargs='+',
                        help='Records with up to how many PAUs you want to include?.',
                        type=str,
                        required=False,
                        default=[2018])

    args = parser.parse_args()
    start_time = time.time()

    for Year in args.Year:
        Building = PAU_DB(int(Year))
        if args.Option == 'A':
            Building.organizing()
        elif args.Option == 'B':
            Building.Building_database_for_statistics()
        elif args.Option == 'C':
            Building.Building_database_for_recycling_efficiency()
        elif args.Option == 'D':
            Building.cleaning_database()
        elif args.Option == 'E':
            Building.Searching_information_for_years_after_2004()
        elif args.Option == 'F':
            Building.dataset_for_individual_statistics()

    print('Execution time: %s sec' % (time.time() - start_time))
