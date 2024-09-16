import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *

ID_COUNT = 0

def explainable_modeling_one_attr(data, y_col, attr:str, partition:Partition) -> float:
    """
    Wrapper function to model the data using an explainable model
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    bins = partition.bins
    data[attr + '.binned'] = partition.binned_values
    data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
    data = data.dropna(subset=[attr + '.binned'])
    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col != attr]
    X = data[X_cols]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    partition.f_time.append((partition.ID, 'explainable_modeling_one_attr', time.time() - start_time))
    partition.utility = model_accuracy
    return model_accuracy

def data_imputation_one_attr(data, y_col, attr:str, partition:Partition) -> float:
    """
    Wrapper function to impute missing values in a dataset
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    bins = partition.bins
    # Bin attr column, with nan values
    data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:])
    
    # Impute the missing values using KNN
    X_cols = [col for col in data.columns if col != y_col and col != attr and col != attr + '.gt']
    X = data[X_cols]
    idx = X.columns.get_loc(attr + '.binned')
    imputer = KNNImputer(n_neighbors=len(bins)-1)
    X_imputed = imputer.fit_transform(X)
    
    # Bin imputed values
    data_imputed = np.round(X_imputed[:, idx])
    data[attr+'.imputed'] = data_imputed
    data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=bins, labels=bins[1:])
    data[attr + '.final'] = data[attr + '.final'].astype('float64')
    value_final = np.array(data[attr + '.final'].values)
    value_final[np.isnan(value_final)] = -1
    value_final = np.round(value_final)

    # Evaluate data imputation
    data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=bins, labels=bins[1:])
    data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
    value_gt = np.array(data[attr + '.gt'].values)
    value_gt[np.isnan(value_gt)] = -1
    value_gt = np.round(value_gt)
    impute_accuracy = accuracy_score(value_gt, value_final)

    partition.f_time.append((partition.ID, 'data_imputation_one_attr', time.time() - start_time))
    partition.utility = impute_accuracy
    return impute_accuracy

def cluster_with_sampling(search_space) -> List:
    # Cluster the binned data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    pca = PCA(n_components=20)
    X = np.array([p.binned_values for p in search_space.candidates])
    print("X:", X)
    X = pca.fit_transform(X)
    hdbscan_clusters = clusterer.fit_predict(X)

    # Get indices of -1 cluster
    outlier_indices = np.where(hdbscan_clusters == -1)[0]
    outlier_indices = np.random.choice(outlier_indices, int(len(outlier_indices) * 0.5), replace=False)

    sampled_indices = []
    for c in np.unique(hdbscan_clusters):
        if c == -1: sampled_indices.extend(outlier_indices)
        cluster_indices = np.where(hdbscan_clusters == c)[0]
        # Sample one partition from the cluster
        sample_index = np.random.choice(cluster_indices)
        sampled_indices.append(sample_index)
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points = get_pareto_front(sampled_partitions)
    return explored_points, pareto_points

def sampling(search_space, frac=0.5) -> List:
    # Sample frac of the partitions
    sampled_indices = np.random.choice(len(search_space.candidates), int(len(search_space.candidates) * frac), replace=False)
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points = get_pareto_front(sampled_partitions)
    return explored_points, pareto_points
    
def get_pareto_front(partitions) -> List:
    semantics = [p.l2_norm for p in partitions]
    utility = [p.utility for p in partitions]
    IDs = [p.ID for p in partitions]
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
    return datapoints, pareto_points

def get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins):
    # Randomly sample 30% of the data and replace the age values with NaN
    data = raw_data.copy()
    data[attr + '.gt'] = data[attr]
    nans = raw_data.sample(frac=0.3, random_state=42)
    data.loc[raw_data.index.isin(nans.index),attr] = np.nan

    # Define gold standard bins
    data_i = data.copy()
    data_i = data_i.dropna(subset=[attr, target])
    values = data_i[attr].values
    binned_values = pd.cut(values, bins=gold_standard_bins, labels=gold_standard_bins[1:])
    gold_standard = Partition(bins=gold_standard_bins, binned_values=binned_values, values=values, method='gold-standard', gold_standard=True)

    # Generate bins
    ss = PartitionSearchSpace()
    ss.prepare_candidates(data, attr, target, 2, 20, gold_standard)
    ss.standardize_semantics()

    # Exhausitve search for the pareto curve partitions
    for partition in ss.candidates:
        data_i = data.copy()
        data_imputation_one_attr(data_i, target, attr, partition)
    
    return ss

def get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins):
    data = raw_data.dropna(subset=[attr, target])
    data_i = data.copy()
    data_i = data_i.dropna(subset=[attr, target])
    values = data_i[attr].values
    binned_values = pd.cut(values, bins=gold_standard_bins, labels=gold_standard_bins[1:])
    gold_standard = Partition(bins=gold_standard_bins, binned_values=binned_values, values=values, method='gold-standard', gold_standard=True)

    # Generate bins
    ss = PartitionSearchSpace()
    ss.prepare_candidates(data, attr, target, 2, 20, gold_standard)
    ss.standardize_semantics()

    # Exhausitve search for the pareto curve partitions
    for partition in ss.candidates:
        data_i = data.copy()
        explainable_modeling_one_attr(data_i, target, attr, partition)
    
    return ss

if __name__ == '__main__':
    np.random.seed(42)

    # Load the diabetes dataset
    raw_data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    # Load the age partitions
    attr = 'Glucose'
    target = 'Outcome'
    

    ss = get_data_imputation_search_space(raw_data, attr, target, [-1, 140, 200])
    datapoints, gt_pareto_points = get_pareto_front(ss.candidates)

    # Get runtime statistics
    runtime_df = ss.get_runtime()
    print(runtime_df.head())

    explored_points, est_pareto_points = cluster_with_sampling(ss)
    average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
    #print(f"Time to cluster and estimate Pareto front: {time.time() - start_time}")
    f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
    f.savefig(os.path.join(ppath, 'figs', f'{attr}.pareto_curve.cluster_sampling.png'))

    explored_points, est_pareto_points = sampling(ss, frac=0.33)
    average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
    #print(f"Time to sample and estimate Pareto front: {time.time() - start_time}")
    f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
    f.savefig(os.path.join(ppath, 'figs', f'{attr}.pareto_curve.random_sampling.png'))