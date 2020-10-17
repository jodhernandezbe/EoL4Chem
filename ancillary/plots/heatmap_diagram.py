# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def heatmap_diagram(df_ratio_loop,
                    chemical,
                    dir_path):
    '''
    Function to plot chemicals that may be in the recycled flow with
    the chemical of interest
    '''

    this_path = os.path.dirname(os.path.realpath(__file__))
    chem_path = f'{this_path}/../others/TRI_CompTox_AIM.csv'
    df_chem = pd.read_csv(chem_path,
                          usecols=['PREFERRED_NAME',
                                   'ID'])
    chemical = df_chem.loc[df_chem['ID'] == chemical,
                           'PREFERRED_NAME'].values[0]

    df_ratio_loop = pd.merge(df_ratio_loop,
                             df_chem,
                             left_on='Chemical',
                             right_on='ID',
                             how='left')
    df_ratio_loop.drop(columns=['Chemical', 'ID'],
                       inplace=True)
    df_ratio_loop.rename(columns={'PREFERRED_NAME': 'Chemical'},
                         inplace=True)

    Options = {'industrial': ['Industrial'],
               'non_industrial': ['Consumer and commercial',
                                  'Commercial', 'Consumer']}
    X_label = {'industrial': 'Industrial function category',
               'non_industrial': 'Production use category'}
    for Key, Option in Options.items():
        try:
            df_diagram = df_ratio_loop[df_ratio_loop['Option'].isin(Option)]
            df_diagram.drop(columns=['Option'], inplace=True)

            df_diagram['Ratio'] = np.log(df_diagram['Ratio']/df_diagram['Sample'])
            df_diagram.drop(columns=['Sample'], inplace=True)
            df_diagram.sort_values('Ratio', inplace=True)
            df_diagram_low = df_diagram.groupby('Category').head(7)
            df_diagram_high = df_diagram.groupby('Category').tail(7)
            df_diagram = pd.concat([df_diagram_high,
                                    df_diagram_low],
                                   ignore_index=True,
                                   sort=True, axis=0)
            del df_diagram_low, df_diagram_high
            df_diagram.drop_duplicates(keep='first', inplace=True)
            df_diagram.rename(columns={'Category': X_label[Key]},
                              inplace=True)

            fig = plt.figure(num=None,
                             figsize=(10, 10),
                             dpi=80,
                             facecolor='w',
                             edgecolor='k')

            degree = pd.pivot_table(df_diagram,
                                    values='Evidence-degree',
                                    index='Chemical',
                                    columns=X_label[Key])
            table = pd.pivot_table(df_diagram, values='Ratio',
                                   index='Chemical',
                                   columns=X_label[Key])


            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            ax = sns.heatmap(table,
                             cbar_kws={'label': f'ln(kg chemical/kg {chemical})'},
                             cmap=cmap,
                             annot=degree)
            ax.set_xlabel(X_label[Key],
                          fontsize = 12,
                          weight='bold')
            ax.set_ylabel('Chemical',
                          fontsize = 12,
                          weight='bold')
            for _, spine in ax.spines.items():
                spine.set_visible(True)

            plt.savefig(f'{dir_path}/seaborn_heatmap_{Key}.pdf',
                        bbox_inches='tight')

        except ValueError:

            Key = Key.replace('_', ' ')
            print(f'There are not records for {Key} activities')
