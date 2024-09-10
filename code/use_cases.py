import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *

ID_COUNT = 0

def data_imputation_one_attr(data, attr:str, bins:List):
    """
    Wrapper function to impute missing values in a dataset
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:])
    #data_i[attr + '.binned'] = data_i[attr + '.binned'].astype('float64')

    imputer = KNNImputer(n_neighbors=len(bins)-1)
    data_imputed = imputer.fit_transform(data[attr + '.binned'].values.reshape(-1, 1))
    data_imputed = np.round(data_imputed)
    data[attr+'.imputed'] = data_imputed
    data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=bins, labels=bins[1:])
    data[attr + '.final'] = data[attr + '.final'].astype('float64')

    #if len(data[data[attr + '.final'].isnull()]) > 200:
    #    print(f"Skipping {bins}")
    #    return None
    #data[attr + '.final'] = data[attr + '.final'].fillna(-1)
    value_final = np.array(data[attr + '.final'].values)
    value_final[np.isnan(value_final)] = -1
    value_final = np.round(value_final)

    # Evaluate data imputation
    data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=bins, labels=bins[1:])
    data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
    value_gt = np.array(data[attr + '.gt'].values)
    value_gt[np.isnan(value_gt)] = -1
    value_gt = np.round(value_gt)
    #data_i[attr + '.gt'] = data_i[attr + '.gt'].fillna(-1)
    #data_i = data_i.dropna(subset=[attr + '.final', attr + '.gt'])
    impute_accuracy = accuracy_score(value_gt, value_final)

    return impute_accuracy

if __name__ == '__main__':
    # Load the diabetes dataset
    raw_data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    # Load the age partitions
    attr = 'Glucose'
    values = list(raw_data[attr].values)
    target = 'Outcome'
    # Define gold standard bins
    gold_standard = Partition(bins=[-1, 140, 200], values=values, method='gold-standard', gold_standard=True)

    # Randomly sample 30% of the data and replace the age values with NaN
    data = raw_data.copy()
    data[attr + '.gt'] = data[attr]
    nans = raw_data.sample(frac=0.3, random_state=42)
    data.loc[raw_data.index.isin(nans.index),attr] = np.nan

    # Generate bins
    canditate_partitions = []
    for n_bins in range(2, 4):
        bins = equal_width(raw_data, n_bins, [attr])[attr]
        bl = Partition(bins=bins, values=values, method='equal-width', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)

        bins = equal_frequency(raw_data, n_bins, [attr])[attr]
        bl = Partition(bins=bins, values=values, method='equal-frequency', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)

        bins = chimerge_wrap(raw_data, [attr], target, n_bins)[attr]
        bl = Partition(bins=bins, values=values, method='chi-merge', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)

        bins = KBinsDiscretizer_wrap(raw_data, [attr], n_bins)[attr]
        bl = Partition(bins=bins, values=values, method='kbins', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)

        bins = KBinsDiscretizer_wrap(raw_data, [attr], n_bins, 'quantile')[attr]
        bl = Partition(bins=bins, values=values, method='kbins-quantile', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)

        bins = DecisionTreeDiscretizer_wrap(raw_data, [attr], target, n_bins)[attr]
        bl = Partition(bins=bins, values=values, method='decision-tree', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)

        bins = KMeansDiscretizer_wrap(raw_data, [attr], n_bins)[attr]
        bl = Partition(bins=bins, values=values, method='kmeans', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)

        bins = RandomForestDiscretizer_wrap(raw_data, [attr], target, n_bins)[attr]
        bl = Partition(bins=bins, values=values, method='random-forest', ref_bucket_list=gold_standard, ID=ID_COUNT)
        ID_COUNT += 1
        canditate_partitions.append(bl)
    
    bins = BayesianBlocksDiscretizer_wrap(raw_data, [attr])[attr]
    bl = Partition(bins=bins, values=values, method='bayesian-blocks', ref_bucket_list=gold_standard, ID=ID_COUNT)
    ID_COUNT += 1
    canditate_partitions.append(bl)

    bins = MDLPDiscretizer_wrap(raw_data, [attr], target)[attr]
    bl = Partition(bins=bins, values=values, method='mdlp', ref_bucket_list=gold_standard, ID=ID_COUNT)
    ID_COUNT += 1
    canditate_partitions.append(bl)

    print(f"Number of bucket lists: {len(canditate_partitions)}")

    # Exhausitve search for the pareto curve partitions
    for partition in canditate_partitions:
        data_i = data.copy()
        acc = data_imputation_one_attr(data_i, attr, partition.bins)
        partition.utility = acc
    
    semantics = [p.l2_norm for p in canditate_partitions]
    utility = [p.utility for p in canditate_partitions]
    IDs = [p.ID for p in canditate_partitions]
    datapoints = [np.array(semantics), np.array(utility)]
    print(f"Data points: {datapoints}")
    print("shape:", np.array(datapoints).shape)
    lst = compute_pareto_front(datapoints)
    
    # Plot the Pareto front
    pareto_df = pd.DataFrame({'ID': IDs, 'Semantic': semantics, 'Utility': utility})
    pareto_df['pareto'] = 0
    pareto_df.loc[lst, 'pareto'] = 1
    pareto_points = pareto_df[pareto_df['pareto'] == 1][['Semantic', 'Utility']]
    pareto_points = pareto_points.values.tolist()
    print(f"Pareto points: {pareto_points}")