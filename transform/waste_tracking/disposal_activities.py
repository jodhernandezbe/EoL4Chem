# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import pandas as pd
import argparse
import os
import sys
sys.path.append(os.path.dirname(
                os.path.realpath(__file__)) + '/../..')
from ancillary.normalizing_naics.normalizing import normalizing_naics
pd.options.mode.chained_assignment = None


class Diposal_tracker:
    '''
    This class is for buiding the dataset for disposal
    '''

    def __init__(self, year=None):
        self.year = year
        # Working Directory
        self._dir_path = os.path.dirname(os.path.realpath(__file__))


    def _checking_rounding(self, flow, dq, classification):
        '''
        Method for checking if a facility rounded a flow to 0
        '''

        if flow == 0.0:
            if dq:
                if classification == 'TRI':
                    return 0.5
                elif classification == 'DIOXIN':
                    return 0.00005
                elif classification == 'PBT':
                    return 0.1
            else:
                return 0.0
        else:
            return flow


    def building_disposal_by_year(self):
        '''
        Method to build the dataset with disposal information by facility, chemical, and year
        '''

        path_needed_columns = self._dir_path + '/../../ancillary/tri/TRI_File_1a_needed_columns_disposal.txt'
        Path_csv = self._dir_path + f'/../../extract/tri/csv/US_1a_{self.year}.csv'

        # Calling needed columns
        needed_columns = pd.read_csv(path_needed_columns,
                                     header=None,
                                     sep='\t').iloc[:, 0].tolist()

        # Calling TRI File 1a
        columns_converting = {'TRI_CHEM_ID': lambda x: x.lstrip('0')}
        tri = pd.read_csv(Path_csv, usecols=needed_columns,
                         header=0, sep=',',
                         low_memory=False,
                         converters=columns_converting)
        tri['REPORTING YEAR'] = int(self.year)
        # Organizing the columns
        fix_columns = ['REPORTING YEAR',
                       'PRIMARY NAICS CODE',
                       'TRIFID',
                       'TRI_CHEM_ID',
                       'CLASSIFICATION',
                       'UNIT OF MEASURE']
        flow_columns = [col for col in tri.columns if ('ON-SITE' in col) and ('BASIS OF ESTIMATE' not in col)]
        df_disposal = pd.DataFrame()
        for col in flow_columns:
            df_aux = tri[fix_columns +\
                         [col,
                          f'{col} - BASIS OF ESTIMATE']]
            df_aux['DISPOSAL ACTIVITY'] = col.replace('ON-SITE - ', '')\
                                             .capitalize()\
                                             .replace('Rcra', 'RCRA')
            df_aux.rename(columns={col: 'FLOW',
                                   f'{col} - BASIS OF ESTIMATE': 'RELIABILITY'},
                          inplace=True)
            df_disposal = pd.concat([df_disposal, df_aux],
                                    ignore_index=True,
                                    sort=True, axis=0)
            del df_aux
        del tri

        # Organizing the roundins flow values
        df_disposal = df_disposal.where(pd.notnull(df_disposal), None)
        df_disposal['FLOW'] = df_disposal.apply(lambda x:\
                                                self._checking_rounding(x['FLOW'],
                                                                        x['RELIABILITY'],
                                                                        x['CLASSIFICATION']),
                                                axis=1)
        df_disposal.loc[df_disposal['UNIT OF MEASURE'] == 'Pounds', 'FLOW'] *= 0.453592
        df_disposal.loc[df_disposal['UNIT OF MEASURE'] == 'Grams', 'FLOW'] *= 10**-3
        df_disposal['UNIT OF MEASURE'] = 'kg'
        df_disposal = df_disposal.loc[(df_disposal['FLOW'] != 0.0) &\
                                      (pd.notnull(df_disposal['FLOW']))]
        df_disposal.drop(columns=['RELIABILITY',
                                  'CLASSIFICATION'],
                         inplace=True)
        grouping_columns = list(set(df_disposal.columns.tolist()) - set(['FLOW']))
        df_disposal = df_disposal.groupby(grouping_columns,
                                          as_index=False).sum()
        df_disposal = normalizing_naics(df_disposal,
                                        naics_column='PRIMARY NAICS CODE',
                                        column_year='REPORTING YEAR')
        # Saving the information
        df_disposal.to_csv(self._dir_path + f'/csv/disposal_activities/Disposal_{self.year}.csv',
                           sep=',',
                           index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('-Y', '--Year', nargs='+',
                        help='What TRI year do you want to organize?.',
                        type=str,
                        required=False)

    args = parser.parse_args()

    for Y in args.Year:
        tracker = Diposal_tracker(Y)
        tracker.building_disposal_by_year()
