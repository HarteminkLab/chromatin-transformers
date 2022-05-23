

import pandas as pd
import numpy as np


def read_orfs_data(filename=None, index_name='orf_name'):
    """
    Load ORFs data set with time columns converting time columns from string
    to integers if necessary. Assumes file is a csv with columns. Assumes there
    is an orf_name column that can be set as the index.
    """

    data = pd.read_csv(filename).set_index(index_name)
    times = [0, 7.5, 15, 30, 60, 120]
    
    non_time_cols = list(data.columns)
    int_time_cols = []
    for time in times:

        # check if time column exists
        str_time_col = "%.1f" % time
        if not str_time_col in data.columns:
            str_time_col = str(int(time))
            if not str_time_col in data.columns:
                continue

        # rename column to numeric
        data = data.rename(columns={str_time_col:time})

    return data