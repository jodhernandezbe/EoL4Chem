# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import pandas as pd
import os
import re
pd.options.mode.chained_assignment = None


def searching_equivalent_naics(df, naics, naics_column,
                               column_year, title, circular):
    '''
    Function to search for the NAICS equivalent for years before 2017
    '''
    year = df[column_year].iloc[0]
    if year >= 2017:
        return df
    else:
        if (year < 2017) and (year >= 2012):
            df = pd.merge(df, naics[['2012 NAICS Code',
                                     '2017 NAICS Code',
                                     '2017 NAICS Title']],
                          how='left', left_on=naics_column,
                          right_on='2012 NAICS Code')
            df.drop(columns='2012 NAICS Code', inplace=True)
        elif (year < 2012) and (year >= 2007):
            df = pd.merge(df, naics[['2007 NAICS Code',
                                     '2017 NAICS Code',
                                     '2017 NAICS Title']],
                          how='left', left_on=naics_column,
                          right_on='2007 NAICS Code')
            df.drop(columns='2007 NAICS Code', inplace=True)
        elif (year < 2007) and (year >= 2002):
            df = pd.merge(df, naics[['2002 NAICS Code',
                                     '2017 NAICS Code',
                                     '2017 NAICS Title']],
                          how='left', left_on=naics_column,
                          right_on='2002 NAICS Code')
            df.drop(columns='2002 NAICS Code', inplace=True)
        elif (year < 2002) and (year >= 1987):
            df = pd.merge(df, naics[['1997 NAICS Code',
                                     '2017 NAICS Code',
                                     '2017 NAICS Title']],
                          how='left', left_on=naics_column,
                          right_on='1997 NAICS Code')
            df.drop(columns='1997 NAICS Code', inplace=True)
        df.drop_duplicates(keep='first', inplace=True)
        idx = df.loc[pd.notnull(df['2017 NAICS Title'])].index.tolist()
        df.loc[idx, naics_column] =\
            df.loc[idx, '2017 NAICS Code']
        if title:
            if circular:
                df.loc[idx, 'Industry title'] = df.loc[idx, '2017 NAICS Title']
            else:
                df.loc[idx, 'RETDF PRIMARY NAICS TITLE'] =\
                    df.loc[idx, '2017 NAICS Title']
        df.drop(columns=['2017 NAICS Code', '2017 NAICS Title'],
                inplace=True)
        return df


def normalizing_naics(TRI, naics_column='RETDF PRIMARY NAICS CODE',
                      column_year='RETDF REPORTING YEAR',
                      title=False,
                      circular=False):
    '''
    Function to normalize NAICS codes
    '''
    def _organizing(name_file):
        return int(re.search(r'to_(\d{4})', name_file).group(1))
    # Calling NAICS changes
    Path_naics = os.path.dirname(os.path.realpath(__file__)) + '/../others'
    naics_files = [file for file in os.listdir(Path_naics)
                   if re.search(r'\d{4}_to_\d{4}_NAICS.csv', file)]
    naics_files.sort(key=_organizing, reverse=True)
    for i, file in enumerate(naics_files):
        df_aux = pd.read_csv(Path_naics + '/' + file,
                             low_memory=False,
                             sep=',', header=0)
        if i == 0:
            df = df_aux
            df.drop(columns=['2012 NAICS Title'], inplace=True)
        else:
            cols = df_aux.iloc[:, :].columns.tolist()
            cols_for_merge = cols[2]
            cols = cols[0:3:2]
            df = pd.merge(df, df_aux[cols], how='outer', on=cols_for_merge)
            df.drop_duplicates(keep='first', inplace=True)
    TRI.sort_values(by=[column_year], inplace=True)
    TRI.reset_index(inplace=True, drop=True)
    TRI = TRI.groupby(column_year, as_index=False)\
             .apply(lambda x: searching_equivalent_naics(x, df, naics_column,
                                                         column_year, title,
                                                         circular))
    TRI[naics_column] =\
        TRI[naics_column].astype(pd.Int32Dtype())
    return TRI
