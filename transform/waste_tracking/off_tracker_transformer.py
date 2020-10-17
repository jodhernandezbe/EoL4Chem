# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import pandas as pd
import numpy as np
import os
import sys
import re
import warnings
import argparse
sys.path.append(os.path.dirname(
                os.path.realpath(__file__))+'/../../extract/gps')
from project_nominatim import NOMINATIM_API
from project_osrm import OSRM_API
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None


class Off_tracker:

    def __init__(self, year=None, database=None):
        self.year = year
        self.database = database
        # Working Directory
        self._dir_path = os.path.dirname(os.path.realpath(__file__))

    # Function for calculating weighted average and avoiding ZeroDivisionError,
    # which ocurres "when all weights along axis are zero".
    def _weight_mean(self, v, w):
        try:
            return round(np.average(v, weights=w))
        except ZeroDivisionError:
            return round(v.mean())

    def _searching_lat_long(self, df, col_id, col_lat, col_long):
        non_lat_log = df.loc[pd.isnull(df[col_lat]), [col_id,
                                                      'LOCATION_ADDRESS',
                                                      'CITY_NAME',
                                                      'STATE_CODE',
                                                      'POSTAL_CODE']]
        non_lat_log.drop_duplicates(keep='first',
                                    inplace=True,
                                    subset=[col_id])
        non_lat_log.rename(columns={'LOCATION_ADDRESS': 'ADDRESS',
                                    'CITY_NAME': 'CITY',
                                    'STATE_CODE': 'STATE',
                                    'POSTAL_CODE': 'ZIP'},
                           inplace=True)
        Path_lat_long = self._dir_path + '/Latitude_&_Longitude.csv'
        if os.path.exists(Path_lat_long):
            df_lat_long_saved = pd.read_csv(Path_lat_long,
                                            dtype={'FRS ID': 'int',
                                                   'LATITUDE': 'float',
                                                   'LATITUDE': 'float'})
            df_lat_long_saved.rename(columns={'FRS ID': col_id},
                                     inplace=True)
            To_search = pd.merge(non_lat_log, df_lat_long_saved,
                                 how='left', on=col_id)
            non_lat_log = To_search.loc[pd.notnull(To_search['LATITUDE'])]
            To_search = To_search.loc[pd.isnull(To_search['LATITUDE'])]
            non_lat_log.drop(columns=['ADDRESS', 'CITY',
                                      'STATE', 'ZIP'],
                             inplace=True)
        else:
            To_search = non_lat_log
            non_lat_log = pd.DataFrame()
        if not To_search.empty:
            Nominatim = NOMINATIM_API()
            To_search = Nominatim.request_coordinates(To_search)
        To_search.drop(columns=['ADDRESS', 'CITY',
                                'STATE', 'ZIP'],
                       inplace=True)
        non_lat_log = pd.concat([non_lat_log, To_search], ignore_index=True,
                                axis=0)
        To_search.rename(columns={col_id: 'FRS ID'}, inplace=True)
        To_search = To_search[['FRS ID', 'LATITUDE', 'LONGITUDE']]
        if os.path.exists(Path_lat_long):
            To_search.to_csv(Path_lat_long, index=False, mode='a', sep=',',
                             header=False)
        else:
            To_search.to_csv(Path_lat_long, index=False, sep=',')
        del To_search
        df = pd.merge(df, non_lat_log,
                      on=col_id,
                      how='left')
        df.drop_duplicates(keep='first', inplace=True)
        del non_lat_log
        df.reset_index(inplace=True)
        idx = df.loc[pd.isnull(df[col_lat])].index.tolist()
        df.loc[idx, col_lat] = df.loc[idx, 'LATITUDE']
        df.loc[idx, col_long] = df.loc[idx, 'LONGITUDE']
        df.drop(columns=['LONGITUDE', 'LATITUDE'], inplace=True)
        return df

    def _generating_srs_database(self, Database_name=['TRI']):
        Dictionary_databases = {'TRI': 'TRI_Chemical_List',
                                'RCRA_T': 'RCRA_T_Char_Characteristics_of_Hazardous_Waste_Toxicity_Characteristic',
                                'RCRA_F': 'RCRA_F_Waste_Hazardous_Wastes_From_Non-Specific_Sources',
                                'RCRA_K': 'RCRA_K_Waste_Hazardous_Wastes_From_Specific_Sources',
                                'RCRA_P': 'RCRA_P_Waste_Acutely_Hazardous_Discarded_Commercial_Chemical_Products',
                                'RCRA_U': 'RCRA_U_Waste_Hazardous_Discarded_Commercial_Chemical_Products'}
        path = self._dir_path + '/../../ancillary/others'
        df_SRS = pd.DataFrame()
        for Schema in Database_name:
            df_db = pd.read_csv(path + f'/{Dictionary_databases[Schema]}.csv',
                                usecols=['ID', 'Internal Tracking Number',
                                         'CAS'])
            df_db['Internal Tracking Number'] =\
                df_db['Internal Tracking Number'].astype(pd.Int32Dtype())
            df_SRS = pd.concat([df_SRS, df_db], ignore_index=True,
                               sort=True, axis=0)
        return df_SRS

    def _generating_frs_database(self, program):
        FSR_FACILITY = pd.read_csv(self._dir_path + '/../../extract/frs/csv/NATIONAL_FACILITY_FILE.CSV',
                                   low_memory=False,
                                   dtype={'POSTAL_CODE': 'object',
                                          'REGISTRY_ID': 'int'},
                                   usecols=['REGISTRY_ID', 'LOCATION_ADDRESS',
                                            'CITY_NAME', 'STATE_CODE',
                                            'POSTAL_CODE',
                                            'LATITUDE83', 'LONGITUDE83'])
        FSR_FACILITY = FSR_FACILITY.drop_duplicates(subset=['REGISTRY_ID'],
                                                    keep='first')
        ENVIRONMENTAL_INTEREST = pd.read_csv(self._dir_path + '/../../extract/frs/csv/NATIONAL_ENVIRONMENTAL_INTEREST_FILE.CSV',
                                             low_memory=False,
                                             dtype={'REGISTRY_ID': 'int'},
                                             usecols=['REGISTRY_ID',
                                                      'PGM_SYS_ACRNM',
                                                      'PGM_SYS_ID'])
        ENVIRONMENTAL_INTEREST =\
            ENVIRONMENTAL_INTEREST.drop_duplicates(keep='first')
        ENVIRONMENTAL_INTEREST =\
            ENVIRONMENTAL_INTEREST.loc[ENVIRONMENTAL_INTEREST[
                'PGM_SYS_ACRNM'].isin(program)]
        df_FRS = pd.merge(ENVIRONMENTAL_INTEREST, FSR_FACILITY, how='inner',
                          on='REGISTRY_ID')
        return df_FRS

    def retrieving_needed_information(self):
        columns_converting = {'REPORTING YEAR': lambda x: str(int(x)),
                              'CAS NUMBER': lambda x: x.lstrip('0')}
        if self.database == 'TRI':
            mapping = {'M': 1, 'M1': 1, 'M2': 1, 'E': 2,
                       'E1': 2, 'E2': 2, 'C': 3, 'O': 4,
                       'X': 5, 'N': 5, 'NA': 5}
            cols = ['REPORTING YEAR', 'SENDER FRS ID', 'SENDER LATITUDE',
                    'SENDER LONGITUDE', 'SENDER STATE',
                    'SRS INTERNAL TRACKING NUMBER', 'CAS',
                    'QUANTITY TRANSFERRED', 'RELIABILITY',
                    'FOR WHAT IS TRANSFERRED', 'UNIT OF MEASURE',
                    'RECEIVER FRS ID', 'RECEIVER TRIFID',
                    'RECEIVER LATITUDE', 'RECEIVER LONGITUDE',
                    'RECEIVER STATE']
            Path_txt = self._dir_path + '/../../ancillary/tri/TRI_File_3a_needed_columns_tracking.txt'
            columns_needed = pd.read_csv(Path_txt,
                                         header=None,
                                         sep='\t').iloc[:, 0].tolist()
            Path_csv = self._dir_path + f'/../../extract/tri/csv/US_3a_{self.year}.csv'
            df = pd.read_csv(Path_csv, header=0, sep=',', low_memory=False,
                             converters=columns_converting,
                             usecols=columns_needed)
            df = df.loc[~pd.notnull(df['OFF-SITE COUNTRY ID'])]
            df.drop(columns=['OFF-SITE COUNTRY ID'], inplace=True)
            column_flows = [col.replace(' - BASIS OF ESTIMATE', '')
                            for col in columns_needed
                            if 'BASIS OF ESTIMATE' in col]
            df[column_flows].fillna(value=0, inplace=True)
            df = df.loc[df['OFF-SITE RCRA ID NR'].str.contains(
                                    r'\s?[A-Z]{2,3}[0-9]{8,9}\s?',
                                    na=False)]
            df = df.loc[(df[column_flows] != 0).any(axis=1)]
            columns = ['TRIFID', 'CAS NUMBER', 'UNIT OF MEASURE',
                       'REPORTING YEAR',
                       'LATITUDE', 'LONGITUDE', 'OFF-SITE RCRA ID NR',
                       'QUANTITY TRANSFERRED',
                       'RELIABILITY', 'FOR WHAT IS TRANSFERRED']
            _df = pd.DataFrame(columns=columns)
            for col in column_flows:
                df_aux = df[['TRIFID', 'CAS NUMBER', 'UNIT OF MEASURE',
                             'REPORTING YEAR',
                             'LATITUDE', 'LONGITUDE',
                             'OFF-SITE RCRA ID NR',
                             col, col + ' - BASIS OF ESTIMATE']]
                df_aux.rename(columns={col: 'QUANTITY TRANSFERRED',
                                       f'{col} - BASIS OF ESTIMATE':
                                       'RELIABILITY'},
                              inplace=True)
                df_aux['FOR WHAT IS TRANSFERRED'] =\
                    re.sub(r'potws',
                           'POTWS',
                           re.sub(r'Rcra|rcra',
                                  'RCRA',
                                  col.replace('OFF-SITE - ',
                                              '').strip().capitalize()))
                _df = pd.concat([_df, df_aux], ignore_index=True,
                                sort=True, axis=0)
            del df, df_aux
            _df = _df.loc[_df['QUANTITY TRANSFERRED'] != 0.0]
            _df.loc[_df['UNIT OF MEASURE'] == 'Pounds',
                    'QUANTITY TRANSFERRED'] *= 0.453592
            _df.loc[_df['UNIT OF MEASURE'] == 'Grams',
                    'QUANTITY TRANSFERRED'] *= 10**-3
            _df['UNIT OF MEASURE'] = 'kg'
            _df['RELIABILITY'] = _df['RELIABILITY'].str.strip().map(mapping)
            _df['RELIABILITY'].fillna(value=5, inplace=True)
            func = {'QUANTITY TRANSFERRED': 'sum',
                    'RELIABILITY': lambda x:
                    self._weight_mean(x,
                                      _df.loc[x.index, 'QUANTITY TRANSFERRED'])
                    }
            _df = _df.groupby(['TRIFID', 'CAS NUMBER', 'UNIT OF MEASURE',
                               'REPORTING YEAR', 'LATITUDE',
                               'LONGITUDE', 'OFF-SITE RCRA ID NR',
                               'FOR WHAT IS TRANSFERRED'],
                              as_index=False).agg(func)
            # Searching EPA Internal Tracking Number of a Substance
            SRS = self._generating_srs_database()
            _df = pd.merge(SRS, _df, how='inner', left_on='ID',
                           right_on='CAS NUMBER')
            _df['SRS INTERNAL TRACKING NUMBER'] =\
                _df['Internal Tracking Number']
            _df.drop(columns=['ID', 'CAS NUMBER', 'Internal Tracking Number'],
                     inplace=True)
            # Searching info for sender
            FRS = self._generating_frs_database(['TRIS', 'RCRAINFO'])
            RCRA = FRS.loc[FRS['PGM_SYS_ACRNM'] == 'RCRAINFO']
            TRI = FRS.loc[FRS['PGM_SYS_ACRNM'] == 'TRIS']
            del FRS
            RCRA.drop('PGM_SYS_ACRNM', axis=1, inplace=True)
            TRI.drop(['PGM_SYS_ACRNM', 'LOCATION_ADDRESS', 'CITY_NAME',
                      'POSTAL_CODE', 'LATITUDE83', 'LONGITUDE83'],
                     axis=1, inplace=True)
            TRI.rename(columns={'PGM_SYS_ID': 'TRIFID'}, inplace=True)
            _df = pd.merge(_df, TRI, how='inner', on='TRIFID')
            _df.rename(columns={'REGISTRY_ID': 'SENDER FRS ID',
                                'LATITUDE': 'SENDER LATITUDE',
                                'LONGITUDE': 'SENDER LONGITUDE',
                                'STATE_CODE': 'SENDER STATE'},
                       inplace=True)
            _df.drop(['TRIFID'], axis=1, inplace=True)
            # Searching info for receiver
            RCRA.rename(columns={'PGM_SYS_ID': 'OFF-SITE RCRA ID NR'},
                        inplace=True)
            _df = pd.merge(_df, RCRA, how='inner', on='OFF-SITE RCRA ID NR')
            _df.rename(columns={'REGISTRY_ID': 'RECEIVER FRS ID',
                                'LATITUDE83': 'RECEIVER LATITUDE',
                                'LONGITUDE83': 'RECEIVER LONGITUDE'},
                       inplace=True)
            _df.drop(['OFF-SITE RCRA ID NR'], axis=1, inplace=True)
            # Searching latitude and longitude for receivers
            _df = self._searching_lat_long(_df, 'RECEIVER FRS ID',
                                           'RECEIVER LATITUDE',
                                           'RECEIVER LONGITUDE')
            _df.rename(columns={'STATE_CODE': 'RECEIVER STATE'},
                       inplace=True)
            # TRIFID for receivers
            _df = pd.merge(_df, TRI, how='left', left_on='RECEIVER FRS ID',
                           right_on='REGISTRY_ID')
            _df.rename(columns={'TRIFID': 'RECEIVER TRIFID'}, inplace=True)
            _df.drop(['REGISTRY_ID'], axis=1, inplace=True)
            # Saving
            _df = _df[cols]
            _df.drop_duplicates(keep='first', inplace=True)
            _df.to_csv(self._dir_path + f'/csv/off_site_tracking/TRI_{self.year}_Off-site_Tracking.csv',
                       sep=',',  index=False)
        else:
            cols = ['REPORTING YEAR', 'SENDER FRS ID', 'SENDER LATITUDE',
                    'SENDER LONGITUDE', 'SENDER STATE',
                    'SRS INTERNAL TRACKING NUMBER', 'CAS',
                    'WASTE SOURCE CODE', 'QUANTITY RECEIVED',
                    'QUANTITY TRANSFERRED', 'RELIABILITY',
                    'FOR WHAT IS TRANSFERRED', 'UNIT OF MEASURE',
                    'RECEIVER FRS ID', 'RECEIVER TRIFID', 'RECEIVER LATITUDE',
                    'RECEIVER LONGITUDE', 'RECEIVER STATE']
            Path_txt = self._dir_path + '/../../ancillary/rcrainfo/RCRAInfo_needed_columns.txt'
            columns_needed = pd.read_csv(Path_txt, header=None,
                                         sep='\t').iloc[:, 0].tolist()
            Path_csv = self._dir_path + f'/../../extract/rcrainfo/csv/BR_REPORTING_{self.year}.csv'
            df = pd.read_csv(Path_csv, header=0, sep=',', low_memory=False,
                             usecols=columns_needed)
            df['QUANTITY TRANSFERRED'] =\
                df['Total Quantity Shipped Off-site (in tons)']*907.18
            df['QUANTITY RECEIVED'] = df['Quantity Received (in tons)']*907.18
            df['UNIT OF MEASURE'] = 'kg'
            df = df.loc[(df['QUANTITY TRANSFERRED'] != 0)
                        | (df['QUANTITY RECEIVED'] != 0)]
            # Searching EPA Internal Tracking Number of a Substance
            SRS = self._generating_srs_database(['RCRA_T', 'RCRA_F', 'RCRA_K',
                                                 'RCRA_P', 'RCRA_U'])
            df = pd.merge(SRS, df, how='inner', left_on='ID',
                          right_on='Waste Code Group')
            df['SRS INTERNAL TRACKING NUMBER'] = df['Internal Tracking Number']
            df.drop(['ID', 'Waste Code Group', 'Internal Tracking Number',
                     'Total Quantity Shipped Off-site (in tons)',
                     'Quantity Received (in tons)'],
                    axis=1, inplace=True)
            Received = df.loc[df['QUANTITY RECEIVED'] != 0]
            Received.drop(columns=['Waste Source Code',
                                   'QUANTITY TRANSFERRED'],
                          inplace=True)
            group = ['CAS', 'EPA Handler ID', 'Reporting Cycle Year',
                     'Management Method Code',
                     'EPA ID Number of Facility to Which Waste was Shipped',
                     'UNIT OF MEASURE', 'SRS INTERNAL TRACKING NUMBER']
            Received = Received.groupby(group, as_index=False)\
                               .agg({'QUANTITY RECEIVED': lambda x: x.sum()})
            Transferred = df.loc[df['QUANTITY TRANSFERRED'] != 0]
            Transferred.drop(columns=['QUANTITY RECEIVED'], inplace=True)
            group = ['CAS', 'EPA Handler ID', 'Reporting Cycle Year',
                     'Waste Source Code', 'Management Method Code',
                     'EPA ID Number of Facility to Which Waste was Shipped',
                     'UNIT OF MEASURE', 'SRS INTERNAL TRACKING NUMBER']
            Transferred = Transferred.groupby(group, as_index=False)\
                                     .agg({'QUANTITY TRANSFERRED':
                                           lambda x: x.sum()})
            Transferred['RELIABILITY'] = 1
            df = pd.concat([Received, Transferred], ignore_index=True,
                           sort=True, axis=0)
            del Received, Transferred
            # Searching info for sender
            FRS = self._generating_frs_database(['TRIS', 'RCRAINFO'])
            RCRA = FRS.loc[FRS['PGM_SYS_ACRNM'] == 'RCRAINFO']
            TRI = FRS.loc[FRS['PGM_SYS_ACRNM'] == 'TRIS']
            del FRS
            RCRA.drop('PGM_SYS_ACRNM', axis=1, inplace=True)
            TRI.drop('PGM_SYS_ACRNM', axis=1, inplace=True)
            df = pd.merge(df, RCRA, how='inner', left_on='EPA Handler ID',
                          right_on='PGM_SYS_ID')
            df.rename(columns={'REGISTRY_ID': 'SENDER FRS ID',
                               'LATITUDE83': 'SENDER LATITUDE',
                               'LONGITUDE83': 'SENDER LONGITUDE'},
                      inplace=True)
            df.drop(['PGM_SYS_ID', 'EPA Handler ID'],
                    axis=1, inplace=True)
            df = self._searching_lat_long(df, 'SENDER FRS ID',
                                          'SENDER LATITUDE',
                                          'SENDER LONGITUDE')
            df.rename(columns={'STATE_CODE': 'SENDER STATE'},
                      inplace=True)
            # Searching info for receiver
            df = pd.merge(df, RCRA, how='inner',
                          left_on='EPA ID Number of Facility to Which Waste was Shipped',
                          right_on='PGM_SYS_ID')
            df.rename(columns={'REGISTRY_ID': 'RECEIVER FRS ID',
                               'LATITUDE83': 'RECEIVER LATITUDE',
                               'LONGITUDE83': 'RECEIVER LONGITUDE',
                               'Reporting Cycle Year': 'REPORTING YEAR'},
                      inplace=True)
            df = self._searching_lat_long(df, 'RECEIVER FRS ID',
                                          'RECEIVER LATITUDE',
                                          'RECEIVER LONGITUDE')
            df.rename(columns={'STATE_CODE': 'RECEIVER STATE'},
                      inplace=True)
            df.drop(['PGM_SYS_ID',
                     'EPA ID Number of Facility to Which Waste was Shipped'],
                    axis=1, inplace=True)
            df = pd.merge(df, TRI, how='left', left_on='RECEIVER FRS ID',
                          right_on='REGISTRY_ID')
            df['RECEIVER TRIFID'] = df['PGM_SYS_ID']
            # Translate management codes
            Path_WM = self._dir_path + '/../../ancillary/rcrainfo/RCRA_Management_Methods.csv'
            Management = pd.read_csv(Path_WM, header=0, sep=',',
                                     usecols=['Management Method Code',
                                              'Management Method'])
            df = pd.merge(df, Management, how='left',
                          on=['Management Method Code'])
            df.rename(columns={'Management Method': 'FOR WHAT IS TRANSFERRED',
                               'Waste Source Code': 'WASTE SOURCE CODE'},
                      inplace=True)
            df.drop(['REGISTRY_ID', 'PGM_SYS_ID', 'Management Method Code'],
                    axis=1, inplace=True)
            df = df[cols]
            df.drop_duplicates(keep='first', inplace=True)
            df.to_csv(self._dir_path + f'/csv/off_site_tracking/RCRAInfo_{self.year}_Off-site_Tracking.csv',
                      sep=',', index=False)

    def joining_databases(self):
        Path_csv = self._dir_path + '/csv/off_site_tracking/'
        Files = os.listdir(Path_csv)
        Tracking = pd.DataFrame()
        for File in Files:
            Tracking_year = pd.read_csv(Path_csv + File, header=0)
            Tracking = pd.concat([Tracking, Tracking_year], ignore_index=True,
                                 axis=0)
        Tracking['Year_difference'] = Tracking.apply(lambda row:
                                                     abs(int(
                                                         row['REPORTING YEAR'])
                                                         - int(self.year[0])),
                                                     axis=1)
        grouping = ['SENDER FRS ID', 'SRS INTERNAL TRACKING NUMBER',
                    'CAS', 'RECEIVER FRS ID']
        Tracking = Tracking.loc[Tracking.groupby(grouping,
                                                 as_index=False)
                                        .Year_difference.idxmin()]
        Tracking.drop(['Year_difference', 'REPORTING YEAR'], axis=1,
                      inplace=True)
        Tracking.to_csv(self._dir_path+f'/csv/Tracking_{self.year[0]}.csv',
                        sep=',', index=False)

    def searching_shortest_distance_from_maps(self):
        Path_csv = self._dir_path + '/csv/off_site_tracking/'
        Files = os.listdir(Path_csv)
        Tracking = pd.DataFrame()
        for File in Files:
            Tracking_year = pd.read_csv(Path_csv + File, header=0,
                                        usecols=[
                                                'SENDER FRS ID',
                                                'SENDER LATITUDE',
                                                'SENDER LONGITUDE',
                                                'RECEIVER FRS ID',
                                                'RECEIVER LATITUDE',
                                                'RECEIVER LONGITUDE'
                                                ],
                                        dtype={'SENDER FRS ID': 'int',
                                               'SENDER LATITUDE': 'float',
                                               'SENDER LONGITUDE': 'float',
                                               'RECEIVER FRS ID': 'int',
                                               'RECEIVER LATITUDE': 'float',
                                               'RECEIVER LONGITUDE': 'float'})
            Tracking = pd.concat([Tracking, Tracking_year], ignore_index=True,
                                 axis=0)
        Maps = OSRM_API()
        Tracking.drop_duplicates(keep='first', inplace=True)
        # Searching distance
        Path_distances = self._dir_path + '/csv/Tracking_distances.csv'
        if os.path.exists(Path_distances):
            df_distances = pd.read_csv(Path_distances,
                                       usecols=['SENDER FRS ID',
                                                'RECEIVER FRS ID',
                                                'DISTANCE'],
                                       dtype={'SENDER FRS ID': 'int',
                                              'RECEIVER FRS ID': 'int',
                                              'DISTANCE': 'float'})
            Tracking = pd.merge(Tracking, df_distances,
                                on=['SENDER FRS ID', 'RECEIVER FRS ID'],
                                how='left')
            Tracking = Tracking.loc[pd.isnull(Tracking['DISTANCE'])]
        if not Tracking.empty:
            Tracking[['DISTANCE', 'MARITIME FRACTION']] =\
                            Tracking.apply(lambda x:
                                           pd.Series(
                                                  Maps.request_directions(
                                                    x['SENDER LATITUDE'],
                                                    x['SENDER LONGITUDE'],
                                                    x['RECEIVER LATITUDE'],
                                                    x['RECEIVER LONGITUDE'])
                                                    ),
                                           axis=1)
            Tracking['UNIT'] = 'km'
            columns = ['SENDER FRS ID', 'SENDER LATITUDE', 'SENDER LONGITUDE',
                       'RECEIVER FRS ID', 'RECEIVER LATITUDE',
                       'RECEIVER LONGITUDE',
                       'DISTANCE', 'UNIT', 'MARITIME FRACTION']
            Tracking = Tracking[columns]
            if os.path.exists(Path_distances):
                Tracking.to_csv(Path_distances, sep=',',
                                index=False, mode='a',
                                header=False)
            else:
                Tracking.to_csv(Path_distances, sep=',',
                                index=False)

    def creating_dataset_for_statistics(self):
        Path_WM = self._dir_path + '/../../ancillary/others/TRI_RCRA_Management_Match.csv'
        Management = pd.read_csv(Path_WM, header=0, sep=',',
                                 usecols=['TRI Waste Management',
                                          'RCRA Waste Management'])
        Management.drop_duplicates(keep='first', inplace=True,
                                   subset='TRI Waste Management')
        Management.loc[Management['TRI Waste Management']
                       .str.contains('broker', na=False),
                       'RCRA Waste Management'] =\
            'Storage and Transfer -The site receiving this waste stored/bulked and transferred the waste with no reclamation, recovery, destruction, treatment, or disposal at that site'
        Management.rename(columns={'TRI Waste Management':
                                   'FOR WHAT IS TRANSFERRED'},
                          inplace=True)
        Path_csv = self._dir_path + '/csv/off_site_tracking/'
        Files = os.listdir(Path_csv)
        Tracking = pd.DataFrame()
        for File in Files:
            Tracking_year = pd.read_csv(Path_csv + File, header=0,
                                        usecols=[
                                                'SENDER FRS ID',
                                                'RECEIVER FRS ID',
                                                'FOR WHAT IS TRANSFERRED',
                                                'SRS INTERNAL TRACKING NUMBER',
                                                'QUANTITY TRANSFERRED'],
                                        dtype={'SENDER FRS ID': 'int',
                                               'RECEIVER FRS ID': 'int',
                                               'SRS INTERNAL TRACKING NUMBER':
                                               'int'})
            if 'TRI' in File:
                Tracking_year = pd.merge(Tracking_year, Management,
                                         on='FOR WHAT IS TRANSFERRED',
                                         how='left')
                Tracking_year.drop(columns='FOR WHAT IS TRANSFERRED',
                                   inplace=True)
                Tracking_year.rename(columns={'RCRA Waste Management':
                                              'FOR WHAT IS TRANSFERRED'},
                                     inplace=True)
            Tracking = pd.concat([Tracking, Tracking_year],
                                 ignore_index=True,
                                 axis=0)
            del Tracking_year
        # Organizing information for relation sender-receiver
        Sender_Receiver = Tracking[['SENDER FRS ID', 'RECEIVER FRS ID',
                                    'QUANTITY TRANSFERRED']]
        Sender_Receiver = Sender_Receiver.loc[
                        pd.notnull(Sender_Receiver['QUANTITY TRANSFERRED'])
                                             ]
        Sender_Receiver.drop(columns=['QUANTITY TRANSFERRED'],
                             inplace=True)
        Sender_Receiver = Sender_Receiver.loc[Sender_Receiver['SENDER FRS ID']
                                              != Sender_Receiver[
                                              'RECEIVER FRS ID']]
        Sender_Receiver = Sender_Receiver.groupby(['SENDER FRS ID',
                                                   'RECEIVER FRS ID']).size()\
                                         .reset_index(name='TIME')
        Sender_Receiver.to_csv(self._dir_path + '/csv/Relation_Sender_Receiver.csv',
                               index=False, sep=',')
        # Organizing information for relation chemical-management
        Chemical_management =\
            Tracking[['SRS INTERNAL TRACKING NUMBER',
                      'FOR WHAT IS TRANSFERRED']]
        Chemical_management = Chemical_management.loc[
                        Chemical_management['FOR WHAT IS TRANSFERRED']
                        != 'Storage and Transfer -The site receiving this waste stored/bulked and transferred the waste with no reclamation, recovery, destruction, treatment, or disposal at that site']
        Chemical_management =\
            Chemical_management.groupby(
                                        ['SRS INTERNAL TRACKING NUMBER',
                                         'FOR WHAT IS TRANSFERRED'])\
            .size().reset_index(name='TIME')
        Chemical_management.to_csv(self._dir_path + '/csv/Relation_Chemical_Management.csv',
                                   index=False, sep=',')


    def creating_dataset_for_receivers(self):
        '''
        Method to find the streams received by the receivers
        in TRI
        '''
        df = pd.DataFrame()
        path = self._dir_path + '/csv/off_site_tracking/'
        for year in range(2003, 2018):
            df_aux = pd.read_csv(f'{path}TRI_{year}_Off-site_Tracking.csv',
                                 usecols=['SENDER FRS ID',
                                          'RECEIVER TRIFID',
                                          'QUANTITY TRANSFERRED',
                                          'FOR WHAT IS TRANSFERRED',
                                          'SRS INTERNAL TRACKING NUMBER', 'CAS'])
            df_aux['REPORTING YEAR'] = year
            df = pd.concat([df, df_aux], ignore_index=True,
                            axis=0)
        grouping = ['SENDER FRS ID',
                    'RECEIVER TRIFID',
                    'FOR WHAT IS TRANSFERRED',
                    'SRS INTERNAL TRACKING NUMBER']
        df = df.loc[df.groupby(grouping)['REPORTING YEAR'].idxmax()]
        chem = df[['SRS INTERNAL TRACKING NUMBER', 'CAS']]
        chem.drop_duplicates(keep='first', inplace=True)
        df.drop(columns=['REPORTING YEAR',
                         'SENDER FRS ID',
                         'CAS'],
                inplace=True)
        grouping = ['RECEIVER TRIFID',
                    'FOR WHAT IS TRANSFERRED',
                    'SRS INTERNAL TRACKING NUMBER']
        df = df.groupby(grouping, as_index=False).sum()
        df = pd.merge(df, chem,
                      on=['SRS INTERNAL TRACKING NUMBER'],
                      how='inner')
        df.to_csv(self._dir_path + '/csv/Receiver_TRI_input_streams.csv',
                  index=False, sep=',')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('Option',
                        help='What do you want to do?:\
                        [A]: Organize only one database\
                        [B]: Join all the databases\
                        [C]: Search distances\
                        [D]: Organizing files with statistics\
                        [E]: Searching for receiver input flows',
                        type=str)

    parser.add_argument('-db', '--database', nargs='?',
                        help='What database want to use (TRI or RCRAInfo)?.',
                        type=str,
                        default=None,
                        required=False)

    parser.add_argument('-Y', '--Year', nargs='+',
                        help='What TRI or RCRAInfo \
                        year do you want to organize?.',
                        type=str,
                        required=False)

    args = parser.parse_args()

    if args.Option == 'A':

        for Y in args.Year:
            T = Off_tracker(Y, args.database)
            T.retrieving_needed_information()

    elif args.Option == 'B':

        T = Off_tracker(args.Year)
        T.joining_databases()

    elif args.Option == 'C':

        T = Off_tracker()
        T.searching_shortest_distance_from_maps()

    elif args.Option == 'D':

        T = Off_tracker()
        T.creating_dataset_for_statistics()

    elif args.Option == 'E':

        T = Off_tracker()
        T.creating_dataset_for_receivers()
