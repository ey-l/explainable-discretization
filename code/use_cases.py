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

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
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

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = impute_accuracy
    return impute_accuracy

def get_runtime_stats(search_space, semantic_metric='l2_norm', indices=None) -> List:
    # Get runtime statistics
    runtime_stats = []
    runtime_df = search_space.get_runtime()
    partition_gen = runtime_df[runtime_df['function'] == 'get_bins']['runtime'].sum()
    if indices is not None:
        runtime_df = runtime_df[runtime_df['ID'].isin(indices)]
        num_explored_points = len(indices)
    else:
        num_explored_points = len(search_space.candidates)
    
    runtime_stats.append(num_explored_points)
    runtime_stats.append(partition_gen)
    functions = [f'cal_{semantic_metric}', 'utility_comp']
    for f in functions:
        total_time = runtime_df[runtime_df['function'] == f]['runtime'].sum()
        runtime_stats.append(total_time)
    return runtime_stats

def cluster_sampling(search_space, num_samples:int=2, frac_outlier_samples=0.5, semantic_metric='l2_norm') -> List:
    runtime_stats = []

    # Cluster the binned data
    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    pca = PCA(n_components=5)
    X = np.array([p.binned_values for p in search_space.candidates])
    #print("X:", X)
    X = pca.fit_transform(X)
    hdbscan_clusters = clusterer.fit_predict(X)

    # Get indices of -1 cluster
    outlier_indices = np.where(hdbscan_clusters == -1)[0]
    outlier_indices = np.random.choice(outlier_indices, int(len(outlier_indices) * frac_outlier_samples), replace=False)

    sampled_indices = []
    for c in np.unique(hdbscan_clusters):
        if c == -1: sampled_indices.extend(outlier_indices)
        cluster_indices = np.where(hdbscan_clusters == c)[0]
        # Sample two partition from the cluster
        sampled_indices.extend(np.random.choice(cluster_indices, num_samples, replace=False))
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points = get_pareto_front(sampled_partitions, semantic_metric)
    method_comp = time.time() - start_time

    # Compute the runtime statistics
    runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
    runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats

def random_sampling(search_space, semantic_metric='l2_norm', frac=0.5) -> List:
    runtime_stats = []
    start_time = time.time()
    # Sample frac of the partitions
    sampled_indices = np.random.choice(len(search_space.candidates), int(len(search_space.candidates) * frac), replace=False)
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points = get_pareto_front(sampled_partitions, semantic_metric)

    method_comp = time.time() - start_time
    # Compute the runtime statistics
    runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
    runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats
    
def get_pareto_front(partitions, semantic_metric='l2_norm') -> List:
    if semantic_metric == 'l2_norm':
        semantics = [p.l2_norm for p in partitions]
    elif semantic_metric == 'gpt_distance':
        semantics = [p.gpt_distance for p in partitions]
    elif semantic_metric == 'KLDiv':
        semantics = [p.KLDiv for p in partitions]
    else: raise ValueError("Invalid semantic metric")
    utility = [p.utility for p in partitions]
    IDs = [p.ID for p in partitions]
    datapoints = [np.array(semantics), np.array(utility)]
    #print(f"Data points: {datapoints}")
    print("Datapoint shape to compute Pareto points:", np.array(datapoints).shape)
    lst = compute_pareto_front(datapoints)

    # Plot the Pareto front
    pareto_df = pd.DataFrame({'ID': IDs, 'Semantic': semantics, 'Utility': utility})
    pareto_df['pareto'] = 0
    pareto_df.loc[lst, 'pareto'] = 1
    pareto_points = pareto_df[pareto_df['pareto'] == 1][['Semantic', 'Utility']]
    pareto_points = pareto_points.values.tolist()
    print(f"Pareto points: {pareto_points}")
    return datapoints, pareto_points

def get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins=2, max_num_bins=20, gpt_measure=True):
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
    gold_standard = Partition(bins=gold_standard_bins, binned_values=binned_values, values=values, method='gold-standard', gold_standard=True, gpt_measure=gpt_measure)

    # Generate bins
    ss = PartitionSearchSpace(gpt_measure=gpt_measure)
    ss.prepare_candidates(data, attr, target, min_num_bins, max_num_bins, gold_standard)
    ss.standardize_semantics()

    # Exhausitve search for the pareto curve partitions
    for partition in ss.candidates:
        data_i = data.copy()
        data_imputation_one_attr(data_i, target, attr, partition)
    
    return ss

def get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins=2, max_num_bins=20, gpt_measure=True):
    data = raw_data.dropna(subset=[attr, target])
    data_i = data.copy()
    data_i = data_i.dropna(subset=[attr, target])
    values = data_i[attr].values
    binned_values = pd.cut(values, bins=gold_standard_bins, labels=gold_standard_bins[1:])
    gold_standard = Partition(bins=gold_standard_bins, binned_values=binned_values, values=values, method='gold-standard', gold_standard=True, gpt_measure=gpt_measure)

    # Generate bins
    ss = PartitionSearchSpace(gpt_measure=gpt_measure)
    ss.prepare_candidates(data, attr, target, min_num_bins, max_num_bins, gold_standard)
    ss.standardize_semantics()

    # Exhausitve search for the pareto curve partitions
    for partition in ss.candidates:
        data_i = data.copy()
        explainable_modeling_one_attr(data_i, target, attr, partition)
    
    return ss

if __name__ == '__main__':
    np.random.seed(0)
    f_runtime_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'num_explored_points', 'partition_gen', 'semantic_comp', 'utility_comp', 'method_comp']
    f_quality_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'avg_dist', 'min_num_bins', 'max_num_bins']

    # Load the diabetes dataset
    use_case = 'imputation'
    gpt_measure = True
    raw_data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    dataset = 'pima'
    min_num_bins = 2
    max_num_bins = 20
    target = 'Outcome'
    semantic_metrics = ['gpt_distance', 'l2_norm', 'KLDiv']
    attributes = {'Age': [-1, 18, 35, 50, 65, 100], 'Glucose': [-1, 140, 200], 'BMI': [-1, 18.5, 25, 30, 68], }
    
    for attr, gold_standard_bins in attributes.items():
        f_runtime = []
        f_quality = []
        
        if use_case == 'modelling':
            ss = get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
        elif use_case == 'imputation':
            ss = get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
        else:
            raise ValueError("Invalid use case")
        
        for semantic_metric in semantic_metrics:
            datapoints, gt_pareto_points = get_pareto_front(ss.candidates, semantic_metric)
            f_runtime.append([use_case, dataset, attr, 'exhaustive', semantic_metric] + get_runtime_stats(ss, semantic_metric) + [0])

            explored_points, est_pareto_points, runtime_stats = cluster_sampling(ss, 1, 1, semantic_metric)
            average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
            method_name = 'cluster_sampling_1_1'
            f_quality.append([use_case, dataset, attr, method_name, semantic_metric, average_distance, min_num_bins, max_num_bins])
            f_runtime.append([use_case, dataset, attr, method_name, semantic_metric] + runtime_stats)
            #print(f"Time to cluster and estimate Pareto front: {time.time() - start_time}")
            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
            f.savefig(os.path.join(ppath, 'figs', dataset, use_case, f'{attr}.{semantic_metric}.pareto_curve.{method_name}.png'))
            
            explored_points, est_pareto_points, runtime_stats = cluster_sampling(ss, 2, 1, semantic_metric)
            average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
            method_name = 'cluster_sampling_2_1'
            f_quality.append([use_case, dataset, attr, method_name, semantic_metric, average_distance, min_num_bins, max_num_bins])
            f_runtime.append([use_case, dataset, attr, method_name, semantic_metric] + runtime_stats)
            #print(f"Time to cluster and estimate Pareto front: {time.time() - start_time}")
            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
            f.savefig(os.path.join(ppath, 'figs', dataset, use_case, f'{attr}.{semantic_metric}.pareto_curve.{method_name}.png'))
            
            explored_points, est_pareto_points, runtime_stats = cluster_sampling(ss, 3, 1, semantic_metric)
            average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
            method_name = 'cluster_sampling_3_1'
            f_quality.append([use_case, dataset, attr, method_name, semantic_metric, average_distance, min_num_bins, max_num_bins])
            f_runtime.append([use_case, dataset, attr, method_name, semantic_metric] + runtime_stats)
            #print(f"Time to cluster and estimate Pareto front: {time.time() - start_time}")
            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
            f.savefig(os.path.join(ppath, 'figs', dataset, use_case, f'{attr}.{semantic_metric}.pareto_curve.{method_name}.png'))
            
            frac=0.2
            explored_points, est_pareto_points, runtime_stats = random_sampling(ss, semantic_metric, frac=frac)
            average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
            f_quality.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric, average_distance, min_num_bins, max_num_bins])
            f_runtime.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric] + runtime_stats)
            #print(f"Time to sample and estimate Pareto front: {time.time() - start_time}")
            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
            f.savefig(os.path.join(ppath, 'figs', dataset, use_case, f'{attr}.{semantic_metric}.pareto_curve.random_sampling_{frac}.png'))

            frac=0.3
            explored_points, est_pareto_points, runtime_stats = random_sampling(ss, semantic_metric, frac=frac)
            average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
            f_quality.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric, average_distance, min_num_bins, max_num_bins])
            f_runtime.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric] + runtime_stats)
            #print(f"Time to sample and estimate Pareto front: {time.time() - start_time}")
            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
            f.savefig(os.path.join(ppath, 'figs', dataset, use_case, f'{attr}.{semantic_metric}.pareto_curve.random_sampling_{frac}.png'))

            frac=0.5
            explored_points, est_pareto_points, runtime_stats = random_sampling(ss, semantic_metric, frac=frac)
            average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
            f_quality.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric, average_distance, min_num_bins, max_num_bins])
            f_runtime.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric] + runtime_stats)
            #print(f"Time to sample and estimate Pareto front: {time.time() - start_time}")
            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
            f.savefig(os.path.join(ppath, 'figs', dataset, use_case, f'{attr}.{semantic_metric}.pareto_curve.random_sampling_{frac}.png'))

            frac=0.7
            explored_points, est_pareto_points, runtime_stats = random_sampling(ss, semantic_metric, frac=frac)
            average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
            f_quality.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric, average_distance, min_num_bins, max_num_bins])
            f_runtime.append([use_case, dataset, attr, f'random_sampling_{frac}', semantic_metric] + runtime_stats)
            #print(f"Time to sample and estimate Pareto front: {time.time() - start_time}")
            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, datapoints, explored_points)
            f.savefig(os.path.join(ppath, 'figs', dataset, use_case, f'{attr}.{semantic_metric}.pareto_curve.random_sampling_{frac}.png'))

        f_runtime_df = pd.DataFrame(f_runtime, columns=f_runtime_cols)
        f_quality_df = pd.DataFrame(f_quality, columns=f_quality_cols)
        f_runtime_df.to_csv(os.path.join(ppath, 'exp', dataset, use_case, f'{attr}_runtime.csv'), index=False)
        f_quality_df.to_csv(os.path.join(ppath, 'exp', dataset, use_case, f'{attr}_quality.csv'), index=False)