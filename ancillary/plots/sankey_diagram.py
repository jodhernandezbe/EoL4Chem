import pandas as pd
import warnings
import plotly.graph_objects as go

warnings.simplefilter(action='ignore', category=FutureWarning)


def sankey_diagram(df_flow_MN, df_markov_network,
                   Queries, dir_path, offline,
                   n_cycles):
    '''
    Function for sankey diagram
    '''
    # Queries for Markov Network
    df_flow_MN.rename(columns={col: '_'.join(col.split()) for
                           col in df_flow_MN.columns},
                  inplace=True)
    df_markov_network.rename(columns={col: '_'.join(col.split()) for
                                  col in df_markov_network.columns},
                         inplace=True)

    query_n = 0
    for Query in Queries.values():
        query_n = query_n + 1
        if Query == 'General':
            Max_times = df_markov_network['Times'].max()
            Result = df_markov_network.loc[df_markov_network['Times']
                                           == Max_times]
        else:
            query_string_1 = ' & '.join([f'{col}=="{val}"' for
                                         col, val in Query.items()])
            Result = df_markov_network.query(query_string_1)
            Max_times = Result['Times'].max()
            Result = Result.loc[Result['Times'] == Max_times]

        # Checking values by row due to the possibility of
        # many combinations with the same joint probablity
        Result.drop(columns='Times',
                    inplace=True)
        Result = Result.to_dict('records')
        df_flow_result = pd.DataFrame()
        for Res in Result:
            query_string_2 = ' & '.join([f'{col}=="{val}"' for
                                         col, val in Res.items()])
            Resul_f = df_flow_MN.query(query_string_2)
            df_flow_result = pd.concat([df_flow_result,
                                        Resul_f],
                                       ignore_index=True,
                                       sort=True, axis=0)
        del Resul_f, Result
        df_flow_result.drop(columns=df_flow_result
                            .columns[df_flow_result.apply(lambda col: (col=='N/N').all() or (col==0.0).all())],
                            inplace=True)
        num_cols = df_flow_result._get_numeric_data().columns
        cat_cols = list(set(df_flow_result.columns) - set(num_cols))
        df_flow_result = df_flow_result.groupby(cat_cols + ['Cycle'],
                                                as_index=False).sum()
        df_flow_result.drop(columns='Cycle', inplace=True)
        num_cols = list(set(num_cols) - set(['Cycle']))
        for col in num_cols:
            df_flow_result[f'{col}^2'] = df_flow_result[col]**2
        df_flow_result = df_flow_result.groupby(cat_cols,
                                                as_index=False).sum()
        for col in num_cols:
            df_flow_result[col] = df_flow_result[col]/n_cycles
            df_flow_result[f'{col}^2'] = (df_flow_result[f'{col}^2']/n_cycles - df_flow_result[col]**2)**0.5/df_flow_result[col]
            df_flow_result.rename(columns={col: f'Mean_{col.lower()}',
                                           f'{col}^2': f'CV_-_{col.lower()}'},
                                  inplace=True)


        colors_links = {'Generator_Industry_Sector': '#fbb4ae',
                        'RETDF_Industry_Sector': '#b3cde3',
                        'Waste_management_under_TRI': '#abdea0',
                        'Industry_sector': '#decbe4',
                        'Industrial_processing_or_use_operation': '#fed9a6'}
        Wastes = {'W': 'Wastewater', 'L': 'Liquid waste',
                  'S': 'Solid waste', 'A': 'Gaseous'}
        Dictionary_source_target = {'Generator_Industry_Sector': {'RETDF_Industry_Sector': ['Mean_flow_transferred',
                                                                                            'CV_-_flow_transferred']},
                                    'RETDF_Industry_Sector': {'Waste_management_under_TRI': ['Mean_flow_transferred',
                                                                                             'CV_-_flow_transferred']},
                                    'Waste_management_under_TRI': {'Carrier_phase': ['Mean_carrier_flow',
                                                                                     'CV_-_carrier_flow'],
                                                                   'Industry_sector': ['Mean_recycled_flow',
                                                                                       'CV_-_recycled_flow'],
                                                                   'Product_category': ['Mean_recycled_flow',
                                                                                        'CV_-_recycled_flow'],
                                                                   'Effluent_phase': ['Mean_effluent_flow',
                                                                                      'CV_-_effluent_flow'],
                                                                   'Mean_fugitive_air_release': ['Mean_fugitive_air_release',
                                                                                                 'CV_-_fugitive_air_release'],
                                                                   'Mean_destroyed/converted/degraded_flow': ['Mean_destroyed/converted/degraded_flow',
                                                                                                              'CV_-_destroyed/converted/degraded_flow'],
                                                                   'Waste/release_phase': ['Mean_waste/release_flow',
                                                                                           'CV_-_waste/release_flow'],
                                                                   'Mean_on-site_soil_release': ['Mean_on-site_soil_release',
                                                                                                 'CV_-_on-site_soil_release']},
                                    'Industry_sector': {'Industrial_processing_or_use_operation': ['Mean_recycled_flow',
                                                                                                   'CV_-_recycled_flow']},
                                    'Industrial_processing_or_use_operation': {'Industry_function_category': ['Mean_recycled_flow',
                                                                                                              'CV_-_recycled_flow']}}

        Sources = list()
        Targets = list()
        Values = list()
        CVs = list()
        Labels = list()
        Colors_source_dict = dict()

        # Labels
        for key_1, val_1 in Dictionary_source_target.items():
            if key_1 in df_flow_result.columns:
                Val = df_flow_result[key_1].values[0]
                Colors_source_dict.update({Val: colors_links[key_1]})
                if Val not in Labels:
                    Labels.append(Val)
                for key_2, val_2 in val_1.items():
                    if key_2 in df_flow_result.columns:
                        if key_2 == 'Mean_fugitive_air_release':
                            Val = 'Fugitive air release'
                        elif key_2 == 'Mean_destroyed/converted/degraded_flow':
                            Val = 'Destruction'
                        elif key_2 == 'Mean_on-site_soil_release':
                            Val = 'On-site soil release'
                        else:
                            Val = df_flow_result[key_2].values[0]
                        if Val not in Labels:
                            Labels.append(Val)

        # Source, Target, and Value
        Colors_source_list = list()
        for key_1, val_1 in Dictionary_source_target.items():
            if key_1 in df_flow_result.columns:
                Val_s = df_flow_result[key_1].values[0]
                for key_2, val_2 in val_1.items():
                    if key_2 in df_flow_result.columns:
                        if key_2 == 'Mean_fugitive_air_release':
                            Val_t = 'Fugitive air release'
                        elif key_2 == 'Mean_destroyed/converted/degraded_flow':
                            Val_t = 'Destruction'
                        elif key_2 == 'Mean_on-site_soil_release':
                            Val_t = 'On-site soil release'
                        else:
                            Val_t = df_flow_result[key_2].values[0]
                        Colors_source_list.append(Colors_source_dict[Val_s])
                        Sources.append(Labels.index(Val_s))
                        Targets.append(Labels.index(Val_t))
                        Values.append(df_flow_result[val_2[0]].values[0])
                        CVs.append(df_flow_result[val_2[1]].values[0])

        Labels = [Wastes[l] if l in Wastes.keys()
                  else l for l in Labels]
        # Sankey diagram
        fig1 = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=35,
                    thickness=5,
                    line=dict(
                        color="black",
                        width=0),
                    label=Labels,
                    color='#400080'
                    ),
                link=dict(
                    source=Sources,
                    target=Targets,
                    value=Values,
                    color=Colors_source_list)
                    )])
        fig1.update_layout(plot_bgcolor='#ffffff',
                           paper_bgcolor='#ffffff',
                           width=1200,
                           height=350
                           )

        if offline:
            fig1.write_image(f'{dir_path}/Sankey_{query_n}.pdf')

            df = pd.DataFrame({'Source': [Labels[i] for i in Sources],
                               'Target': [Labels[i] for i in Targets],
                               'Value': Values, 'CV': CVs})
            df.to_csv(f'{dir_path}/Sankey_flows_{query_n}.csv',
                      sep=',', index=False)
        else:
            fig1.show()
