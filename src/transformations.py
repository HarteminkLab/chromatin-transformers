import itertools
import numpy as np
import pandas as pd


def exhaustive_counts(data, x_span, y_span, x_key='mid', y_key='length'):
    """
    Create exhaustive dataframe of all xs and y values for a dataframe, counting
    existing of x, y combination. Returns narrow list or pivoted
    """

    exhaustive_values = [np.arange(x_span[0], x_span[1]), 
                         np.arange(y_span[0], y_span[1])]
    
    # create dataframe of full range of values to join in case fragment doesnt exist
    # at every position
    full_range_list = list(itertools.product(*exhaustive_values))
    xs = [e[0] for e in full_range_list]
    ys = [e[1] for e in full_range_list]

    full_range = pd.DataFrame()
    full_range[y_key] = ys
    full_range[x_key] = xs

    index = [x_key, y_key]

    # pivot MNase-seq data into a count histogram
    if data is None:
        full_range['count'] = 0
        narrow_counts = full_range.reset_index()
    else:
        narrow_counts = data[index].copy()
        narrow_counts['count'] = 1
        narrow_counts = narrow_counts.groupby(index).count()

        # join with full range of values
        narrow_counts = narrow_counts.reset_index().merge(full_range.reset_index(), 
            how='outer').fillna(0)
        narrow_counts = narrow_counts.set_index(['length'])

    pivot_idx = 'length'

    wide_counts = narrow_counts.pivot_table(index=pivot_idx, columns=x_key, values='count')
    wide_counts = wide_counts.fillna(0).astype(int)
    wide_counts = wide_counts.loc[exhaustive_values[1]]

    return wide_counts


def collect_peaks(data, window, minimum, pos_key, value_key):

    win_2 = (window-1)//2

    collected = data.copy()
    collected = collected.sort_values(value_key, ascending=False)

    peaks = pd.DataFrame()

    def get_highest(collected):

        high_ind = collected.index.values[0]
        highest = collected.loc[high_ind]
        
        return high_ind, highest

        
    high_ind, highest = get_highest(collected)

    while highest[value_key] > minimum:
        remove_select = ((collected[pos_key] >= highest.position - win_2) & 
                         (collected[pos_key] <= highest.position + win_2))

        remove_ind = collected[remove_select].index


        peaks = peaks.append(highest)

        collected = collected.drop(remove_ind)
        high_ind, highest = get_highest(collected)

    return peaks


def multi_df_to_arr(hist):
    dim = len(hist.index.get_level_values(0).unique())
    hist_array = hist.values.reshape((dim, -1, hist.shape[1]))
    feature_array = hist_array.reshape((hist_array.shape[0], -1))
    return hist_array, feature_array


def scale_down(A, scale):
    n, m = A.shape
    B = np.zeros((n//scale, m//scale))
    for i in range(0, n, scale):
        for j in range(0, m, scale):
            B[i//scale, j//scale] = A[i:i+scale, j:j+scale].mean()
    return B
