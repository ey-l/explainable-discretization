import pandas as pd
import numpy as np
import sys
import os
from typing import List, Union, Any, Tuple, Dict
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer

# Project path
ppath = sys.path[0] + '/../'

def discretize(df, n_bins:int=10, method:str='equal-width', cols:List[str]=None, min_val=None) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Discretize the continuous variables in the dataframe df.
    The method can be 'equal-width' or 'equal-frequency'.
    Return the dataframe and the intervals for each column.
    """
    intervals = {}
    if cols is None:
        cols = df.columns
    for col in cols:
        # minimum value
        if min_val is None: min_val = df[col].min()
        if method == 'equal-width':
            try: col_data = pd.cut(df[col], n_bins)
            except: continue
        elif method == 'equal-frequency':
            try: col_data = pd.qcut(df[col], n_bins)
            except: continue
        else:
            raise ValueError('Method must be equal-width or equal-frequency.')
        intervals[col] = col_data.unique()
    # Convert intervals to a numeric array
    for col in cols:
        intervals[col] = list(np.insert(np.sort(np.array([x.right for x in intervals[col]])), 0, min_val))
    return intervals

def equal_width(df, n_bins:int=10, cols:List[str]=None, min_val=None):
    """
    Discretize the continuous variables in the dataframe df using equal-width method.
    """
    return discretize(df, n_bins, 'equal-width', cols, min_val)

def equal_frequency(df, n_bins:int=10, cols:List[str]=None, min_val=None):
    """
    Discretize the continuous variables in the dataframe df using equal-frequency method.
    """
    return discretize(df, n_bins, 'equal-frequency', cols, min_val)

def chimerge(data, attr, label, max_intervals):
    """
    Original code from: https://gist.github.com/alanzchen/17d0c4a45d59b79052b1cd07f531689e
    ChiMerge discretization algorithm.
    Example: 
        for attr in ['sepal_l', 'sepal_w', 'petal_l', 'petal_w']:
            print('Interval for', attr)
            chimerge(data=iris, attr=attr, label='type', max_intervals=6)
    """
    distinct_vals = sorted(set(data[attr])) # Sort the distinct values
    labels = sorted(set(data[label])) # Get all possible labels
    empty_count = {l: 0 for l in labels} # A helper function for padding the Counter()
    intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))] # Initialize the intervals for each attribute
    while len(intervals) > max_intervals: # While loop
        chi = []
        for i in range(len(intervals)-1):
            # Calculate the Chi2 value
            obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
            obs1 = data[data[attr].between(intervals[i+1][0], intervals[i+1][1])]
            total = len(obs0) + len(obs1)
            count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
            count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
            count_total = count_0 + count_1
            expected_0 = count_total*sum(count_0)/total
            expected_1 = count_total*sum(count_1)/total
            chi_ = (count_0 - expected_0)**2/expected_0 + (count_1 - expected_1)**2/expected_1
            chi_ = np.nan_to_num(chi_) # Deal with the zero counts
            chi.append(sum(chi_)) # Finally do the summation for Chi2
        min_chi = min(chi) # Find the minimal Chi2 for current iteration
        for i, v in enumerate(chi):
            if v == min_chi:
                min_chi_index = i # Find the index of the interval to be merged
                break
        new_intervals = [] # Prepare for the merged new data array
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done: # Merge the intervals
                t = intervals[i] + intervals[i+1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        intervals = new_intervals
    #for i in intervals:
    #    print('[', i[0], ',', i[1], ']', sep='')
    return intervals

def chimerge_wrap(df, cols, target:str, max_intervals:int=6, min_val=None):
    """
    Wrap the chimerge function.
    Return the dataframe and the intervals for each column.
    """
    intervals = {}
    for col in cols:
        if min_val is None: min_val = df[col].min()
        intervals[col] = chimerge(df, col, target, max_intervals)
        intervals[col] = np.array([x[1] for x in intervals[col]]).astype(np.float32)
        intervals[col] = np.insert(np.sort(intervals[col]), 0, min_val)
        intervals[col] = list(np.unique(intervals[col], axis=0))
    return intervals

def KBinsDiscretizer_wrap(df, cols, n_bins:int=10, min_val=None):
    """
    Wrap the sklearn.preprocessing.KBinsDiscretizer.
    Return the dataframe and the intervals for each column.
    """
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    try: kbd.fit(df[cols])
    except: return {}
    intervals = {}
    for i in range(len(cols)):
        if min_val is None: min_val = df[col].min()
        intervals[cols[i]] = list(np.insert(kbd.bin_edges_[i], 0, min_val))
    return intervals

if __name__ == "__main__":
    # Test the discretizers
    attrs = ['Glucose', 'BMI']
    target = 'Outcome'
    n_bins = 3
    df = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    # Sort on Glucose
    df = df.sort_values(by=['Glucose'])
    intervals = equal_width(df, n_bins, attrs)

    print(intervals)

    # Test the chimerge
    intervals = chimerge_wrap(df, attrs, target, 6)
    print(intervals)

    for col in attrs:
        print('Interval for', col)
        bins = intervals[col]
        df[col + '.binned'] = pd.cut(df[col], bins=bins, labels=bins[1:])
        df[col + '.binned'] = df[col + '.binned'].astype('float64')
    print(df.head(10))

    # Test the KBinsDiscretizer
    intervals = KBinsDiscretizer_wrap(df, attrs, n_bins)
    print(intervals)