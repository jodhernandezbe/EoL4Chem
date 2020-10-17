# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import pandas as pd
import argparse
import os
pd.options.mode.chained_assignment = None

class SR_tracker:
    '''
    This class is for builing the dataset for source reduction and
    production/activity rate
    '''

    def __init__(self, year):
        self.year = year
        # Working Directory
        self._dir_path = os.path.dirname(os.path.realpath(__file__))


    def building_source_reduction_by_year(self):
        '''
        Method to build the dataset with source reduction activities and
        production/activity rate
        '''

        columns_converting = {'CAS NUMBER': lambda x: x.lstrip('0')}
        df = pd.read_csv(self._dir_path + f'/../../extract/tri/csv/US_2a_{self.year}.csv',
                         header=0, sep=',',
                         converters=columns_converting,
                         low_memory=False,
                         usecols=['TRIFID', 'CAS NUMBER', 'PROD RATIO/ACTIVITY INDEX',
                                  'EST ANNUAL REDUCTION - FIRST SOURCE REDUCTION ACTIVITY - CODE',
                                  'EST ANNUAL REDUCTION - SECOND SOURCE REDUCTION ACTIVITY - CODE',
                                  'EST ANNUAL REDUCTION - THIRD SOURCE REDUCTION ACTIVITY - CODE',
                                  'EST ANNUAL REDUCTION - FOURTH SOURCE REDUCTION ACTIVITY - CODE'])
        df['REPORTING YEAR'] = year
        df.rename(columns={col: f'SR ACTIVITY {idx + 1}' for idx, col in enumerate(_col for _col in df.columns if 'EST' in _col)},
                  inplace=True)
        df = df.loc[~(pd.isnull(df).sum(axis=1) == 5)]
        df.to_csv(self._dir_path + f'/csv/source_reduction_{self.year}.csv',
                  sep=',', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('-Y', '--Year', nargs='+',
                        help='What is the TRI reporting year?.',
                        type=str,
                        required=False)

    args = parser.parse_args()

    for year in args.Year:
        tracker = SR_tracker(year)
        tracker.building_source_reduction_by_year()
