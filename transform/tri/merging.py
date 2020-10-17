# -*- coding: utf-8 -*-
#!/usr/bin/env python

# Importing libraries

import pandas as pd

def searching(df):
    if any(df.str.upper() == 'YES'):
        return 'YES'
    else:
        return 'NO'


def fuctions_rows_grouping(x):
    f = {'REPORTING YEAR': lambda x: x.drop_duplicates(keep = 'first'),
        'FACILITY NAME': lambda x: x.drop_duplicates(keep = 'first'),
        'FACILITY STREET': lambda x: x.drop_duplicates(keep = 'first'),
        'FACILITY CITY': lambda x: x.drop_duplicates(keep = 'first'),
        'FACILITY COUNTY': lambda x: x.drop_duplicates(keep = 'first'),
        'FACILITY STATE': lambda x: x.drop_duplicates(keep = 'first'),
        'FACILITY ZIP CODE': lambda x: x.drop_duplicates(keep = 'first'),
        'CHEMICAL NAME': lambda x: x.drop_duplicates(keep = 'first'),
        'CLASSIFICATION': lambda x: x.drop_duplicates(keep = 'first'),
        'UNIT OF MEASURE': lambda x: x.drop_duplicates(keep = 'first'),
        'METAL INDICATOR': lambda x: x.drop_duplicates(keep = 'first'),
        'PRODUCE THE CHEMICAL': lambda x: searching(x),
        'IMPORT THE CHEMICAL': lambda x: searching(x),
        'USED AS A REACTANT': lambda x: searching(x),
        'ADDED AS A FORMULATION COMPONENT': lambda x: searching(x),
        'USED AS AN ARTICLE COMPONENT': lambda x: searching(x),
        'REPACKAGING': lambda x: searching(x),
        'AS A PROCESS IMPURITY': lambda x: searching(x),
        'RECYCLING': lambda x: searching(x),
        'USED AS A CHEMICAL PROCESSING AID': lambda x: searching(x),
        'USED AS A MANUFACTURING AID': lambda x: searching(x),
        'ANCILLARY OR OTHER USE': lambda x: searching(x)}
    return f
