# -*- coding: utf-8 -*-
# !/usr/bin/env python

# Importing libraries
import holoviews as hv
from holoviews import opts, dim
import pandas as pd
import numpy as np

hv.extension("matplotlib")
hv.output(fig='pdf', size=250)


def chord_diagram(df_flow_MN, n_cycles, dir_path):
    '''
    Function to plot chord diagram for flows across industry sectors
    '''
    df_flow_MN = df_flow_MN.loc[df_flow_MN['Option'] == 'Industrial']
    df_flow_MN = df_flow_MN[['Cycle', 'Generator Industry Sector',
                             'Flow transferred', 'RETDF Industry Sector',
                             'Recycled flow', 'Industry sector']]

    Flows = {'waste': {'Generator Industry Sector': 'source',
                       'RETDF Industry Sector': 'target',
                       'Flow transferred': 'value'},
             'recyled': {'RETDF Industry Sector': 'source',
                         'Industry sector': 'target',
                         'Recycled flow': 'value'}}

    df_links = pd.DataFrame()
    for Flow, Link in Flows.items():

        cols = list(Link.keys())
        df_links_aux = df_flow_MN[['Cycle'] + cols]
        df_links_aux = df_links_aux.groupby(['Cycle'] + cols[0:2],
                                            as_index=False).sum()
        df_links_aux.drop(columns='Cycle', inplace=True)
        df_links_aux = df_links_aux.groupby(cols[0:2], as_index=False).sum()
        df_links_aux[cols[2]] = df_links_aux[cols[2]]/n_cycles
        df_links_aux['flow'] = Flow
        df_links_aux.rename(columns=Link, inplace=True)
        if Flow == 'waste':
            # 1 metric ton/yr
            df_links_aux = df_links_aux[df_links_aux['value'] >= 1000]
        df_links = pd.concat([df_links, df_links_aux],
                             ignore_index=True,
                             sort=True, axis=0)
    df_links = df_links.loc[df_links['source'] != df_links['target']]
    Nodes = set(df_links['source'].unique().tolist()
                + df_links['target'].unique().tolist())
    Nodes = {node: i for i, node in enumerate(Nodes)}
    df_links = df_links.replace({'source': Nodes,
                                 'target': Nodes})

    df_nodes = pd.DataFrame({'index': [idx for idx in Nodes.values()],
                             'name sector': [name for name in Nodes.keys()]})
    df_nodes['name'] = df_nodes['index'].apply(lambda x: f'Sector {x+1}')

    for Flow in ['waste', 'recyled']:
        try:
            df_links_plot = df_links.loc[df_links['flow'] == Flow,
                                         ['source', 'target', 'value']]
            sources = df_links_plot['source'].unique().tolist()
            search = df_links_plot.loc[~df_links_plot['target']
                                       .isin(sources), 'target'].unique().tolist()
            for s in search:
                df_links_plot = pd.concat([df_links_plot,
                                           pd.DataFrame({'source': [s],
                                                         'target': [s],
                                                         'value': [10**-50]})],
                                          ignore_index=True,
                                          sort=True, axis=0)
            hv.Chord(df_links_plot)
            nodes = hv.Dataset(df_nodes, 'index')

            chord = hv.Chord((df_links_plot, nodes)).select(value=(5, None))
            chord.opts(
                       opts.Chord(cmap='Category20', edge_cmap='Category20',
                                  edge_color=dim('source').str(),
                                  labels='name', node_color=dim('index').str()))

            df_nodes.to_csv(f'{dir_path}/chord_{Flow}.csv', sep=',', index=False)
            hv.save(chord, f'{dir_path}/chord_{Flow}.pdf', fmt='pdf')
        except ValueError:

            print(f'There are not records for {Flow} activities')
