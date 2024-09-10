import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from Bucket import *
from utils import *

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

    if len(data[data[attr + '.final'].isnull()]) > 200:
        print(f"Skipping {bins}")
        return None
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
    data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    # Load the age partitions
    attr = 'Glucose'
    values = list(data[attr].values)
    target = 'Outcome'
    # Define gold standard bins
    gold_standard = BucketList(bins=[-1, 140, 200], values=values, method='gold-standard', gold_standard=True)

    # Generate bins
    bl_list = []
    for n_bins in range(2, 4):
        bins = equal_width(data, n_bins, [attr])[attr]
        bl = BucketList(bins=bins, values=values, method='equal-width', ref_bucket_list=gold_standard)
        bl_list.append(bl)

        bins = equal_frequency(data, n_bins, [attr])[attr]
        bl = BucketList(bins=bins, values=values, method='equal-frequency', ref_bucket_list=gold_standard)
        bl_list.append(bl)

        bins = chimerge_wrap(data, [attr], target, n_bins)[attr]
        bl = BucketList(bins=bins, values=values, method='chi-merge', ref_bucket_list=gold_standard)
        bl_list.append(bl)

        bins = KBinsDiscretizer_wrap(data, [attr], n_bins)[attr]
        bl = BucketList(bins=bins, values=values, method='kbins', ref_bucket_list=gold_standard)
        bl_list.append(bl)

        bins = KBinsDiscretizer_wrap(data, [attr], n_bins, 'quantile')[attr]
        bl = BucketList(bins=bins, values=values, method='kbins-quantile', ref_bucket_list=gold_standard)
        bl_list.append(bl)

        bins = DecisionTreeDiscretizer_wrap(data, [attr], target, n_bins)[attr]
        bl = BucketList(bins=bins, values=values, method='decision-tree', ref_bucket_list=gold_standard)
        bl_list.append(bl)

        bins = KMeansDiscretizer_wrap(data, [attr], n_bins)[attr]
        bl = BucketList(bins=bins, values=values, method='kmeans', ref_bucket_list=gold_standard)
        bl_list.append(bl)

        bins = RandomForestDiscretizer_wrap(data, [attr], target, n_bins)[attr]
        bl = BucketList(bins=bins, values=values, method='random-forest', ref_bucket_list=gold_standard)
        bl_list.append(bl)
    
    bins = BayesianBlocksDiscretizer_wrap(data, [attr])[attr]
    bl = BucketList(bins=bins, values=values, method='bayesian-blocks', ref_bucket_list=gold_standard)
    bl_list.append(bl)

    bins = MDLPDiscretizer_wrap(data, [attr], target)[attr]
    bl = BucketList(bins=bins, values=values, method='mdlp', ref_bucket_list=gold_standard)
    bl_list.append(bl)

    print(f"Number of bucket lists: {len(bl_list)}")