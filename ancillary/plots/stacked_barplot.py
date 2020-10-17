# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def stacked_barplot(df_markov_network, dir_path):
    '''
    Function to plot a stacked barplot with the
    probability to have and specific activity
    downstream the rycling process
    '''

    df_markov_network = df_markov_network.loc[df_markov_network['Type of waste management'] == 'Recycling']
    df_markov_network = df_markov_network[['RETDF Industry Sector',
                                           'Option', 'Times']]
    ##############################################################
    df_markov_network['RETDF Industry Sector'] = df_markov_network['RETDF Industry Sector'].str.capitalize()
    df_markov_network['Option'] = df_markov_network['Option'].str.capitalize()
    ##############################################################
    df_markov_network = df_markov_network.groupby(['RETDF Industry Sector',
                                                   'Option'],
                                                  as_index=False).sum()
    df_markov_network['Total'] = df_markov_network.groupby('RETDF Industry Sector',
                                                           as_index=False)\
                                                           ['Times'].transform('sum')
    df_markov_network['percentage'] = df_markov_network['Times']*100/df_markov_network['Total']
    df_markov_network['percentage'] = df_markov_network['percentage'].round(0)
    df_markov_network['percentage'] = df_markov_network['percentage'].astype('int')
    df_markov_network.drop(columns=['Times', 'Total'], inplace=True)
    df_markov_network.rename(columns={'Option': 'Activity'},
                             inplace=True)
    sns.set()
    df_markov_network = pd.pivot_table(df_markov_network,
                                       values='percentage',
                                       index='RETDF Industry Sector',
                                       columns='Activity')
    ax = df_markov_network.plot(kind='bar', stacked=True,
                                legend=False)
    ax.legend(loc='center left', bbox_to_anchor=(0.45, 0.9), ncol=1)
    ax.set_xlabel('RETDF industry sector',
                  fontsize = 12,
                  weight='bold')
    ax.set_ylabel('Probability [%]',
                  fontsize = 12,
                  weight='bold')
    plt.savefig(f'{dir_path}/stacked_barplot.pdf',
                bbox_inches='tight')
