import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *

ID_COUNT = 0

def visualization_one_attr(data, y_col, attr:str, partition:Partition) -> float:
    """
    Wrapper function to visualize the data using ANOVA
    """
    start_time = time.time()
    data = data[[attr, y_col]]
    data[attr] = pd.cut(data[attr], bins=partition.bins, labels=partition.bins[1:])
    data[attr] = data[attr].astype('float64')
    data = data.groupby(attr)[y_col]
    data = [group[1] for group in data]
    f, p = f_oneway(*data)
    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = f
    return f

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

def DBSCAN_distributions(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :return: List of clusters
    """
    eps = parameters['eps']
    min_samples = parameters['min_samples']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    dbscan_clusters = model.fit_predict(X)
    # For outliers, assign them to each of their separate cluster
    max_cluster = np.max(dbscan_clusters)
    for i, c in enumerate(dbscan_clusters):
        if c == -1:
            dbscan_clusters[i] = max_cluster + 1
            max_cluster += 1
    return dbscan_clusters

def HDBSCAN_distributions(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :return: List of clusters
    """
    min_cluster_size = parameters['min_cluster_size']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size)
    hdbscan_clusters = model.fit_predict(X)
    # For outliers, assign them to each of their separate cluster
    max_cluster = np.max(hdbscan_clusters)
    for i, c in enumerate(hdbscan_clusters):
        if c == -1:
            hdbscan_clusters[i] = max_cluster + 1
            max_cluster += 1
    return hdbscan_clusters

def HDBSCAN_binned_values(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :return: List of clusters
    """
    n_components = parameters['n_components']
    X = np.array([p.binned_values for p in search_space.candidates])
    model = hdbscan.HDBSCAN(min_cluster_size=2)
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    hdbscan_clusters = model.fit_predict(X)
    return hdbscan_clusters

def linkage_distributions(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :return: List of clusters
    """
    t = parameters['t']
    criterion = parameters['criterion']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    Z = linkage(X, method='ward')
    agg_clusters = fcluster(Z, t=t, criterion=criterion)
    #agg_clusters = fcluster(Z, t=0.5, criterion='distance')
    agg_clusters = [x-1 for x in agg_clusters] # 0-indexing
    return agg_clusters

def random_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    sampled_indices = []
    print("Budget start:", budget)
    print("Unique clusters:", np.unique(cluster_assignments))
    if budget >= len(np.unique(cluster_assignments)):
        # Only sample one partition from each cluster
        for c in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == c)[0]
            # Sample one partition from the cluster
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
        if budget > len(np.unique(cluster_assignments)):
            assignments = cluster_assignments.copy()
            # Remove sampled_indices from the cluster assignments by index
            new_list_to_sample = [item for i, item in enumerate(assignments) if i not in sampled_indices]
            sampled_indices.extend(np.random.choice(new_list_to_sample, budget - len(sampled_indices), replace=False))
        # Add gold standard partition to the sampled partitions
        if 0 not in sampled_indices:
            sampled_indices.append(0)
    
    return sampled_indices

def random_sampling_clusters_robust(cluster_assignments, parameters) -> List:
    """
    Sample partitions from clusters with a budget constraint.
    Similar to random_sampling_clusters, but more robust.
    In the sense that if the budget (n) is less than the number of clusters,
    this method will sample n clusters and sample one partition from each cluster.
    """
    p = parameters['p']
    budget = int(len(cluster_assignments) * p) - 1
    sampled_indices = []
    if budget >= len(np.unique(cluster_assignments)):
        sampled_indices = random_sampling_clusters(cluster_assignments, parameters)
    
    # Sample clusters when budget is less than the number of clusters
    else:
        sampled_clusters = np.random.choice(np.unique(cluster_assignments), budget, replace=False)
        for c in sampled_clusters:
            cluster_indices = np.where(cluster_assignments == c)[0]
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def random_with_inverse_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    sampled_indices = []
    print("Budget start:", budget)
    print("Unique clusters:", np.unique(cluster_assignments))
    if budget >= len(np.unique(cluster_assignments)):
        # Only sample one partition from each cluster
        for c in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == c)[0]
            # Sample one partition from the cluster
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
        
        if budget > len(np.unique(cluster_assignments)):
            assignments = cluster_assignments.copy()
            # Remove sampled_indices from the cluster assignments by index
            new_cluster_assignments = [item for i, item in enumerate(assignments) if i not in sampled_indices]
            #sampled_indices.extend(np.random.choice(new_list_to_sample, budget - len(sampled_indices), replace=False))
            budget = budget - len(sampled_indices)
            # Calculate cluster size from cluster assignment
            cluster_size = [len(np.where(new_cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
            cluster_probs = 1 / (cluster_size / np.sum(cluster_size))
            cluster_probs = cluster_probs / np.sum(cluster_probs)
            cluster_probs = np.nan_to_num(cluster_probs)
            # get number of samples per cluster, with at least one sample per cluster
            cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
            # sample from each cluster based on the number of samples
            for c in np.unique(new_cluster_assignments):
                cluster_indices = np.where(new_cluster_assignments == c)[0]
                sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))

        
        # Add gold standard partition to the sampled partitions
        if 0 not in sampled_indices:
            sampled_indices.append(0)
    
    return sampled_indices
    
def proportional_sampling_clusters(cluster_assignments, parameters) -> List:
    # Proportionally sample from each cluster
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    clusters = [i for i in range(len(np.unique(cluster_assignments)))]
    # get cluster probabilities
    cluster_probs = np.bincount(cluster_assignments) / len(cluster_assignments)
    cluster_size = [len(np.where(cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
    # get number of samples per cluster, with at least one sample per cluster
    cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
    # get number of samples per cluster, with at least one sample per cluster
    #cluster_samples = [0] * len(np.unique(cluster_assignments))
    #samples = np.random.choice(clusters, p=cluster_probs, size=budget)
    #for c in samples: cluster_samples[c] += 1
    # sample from each cluster based on the number of samples
    sampled_indices = []
    for c in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == c)[0]
        sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))

    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def find_actual_cluster_sample_size(total_budget, norm_inv_probs, cluster_sizes):

    # Step 3: Calculate the ideal number of samples for each cluster
    # Based on the inverse probabilities and total budget
    ideal_samples = [int(p * total_budget) for p in norm_inv_probs]
    ideal_excess = sum(ideal_samples) - total_budget
    #print("Inv probs:", norm_inv_probs)
    #print("Ideal samples:", ideal_samples)

    # Step 4: Initialize an array to track the actual samples drawn from each cluster
    actual_samples = [0] * len(norm_inv_probs)

    # Step 5: First pass: Assign as many samples as possible without exceeding cluster capacity
    excess_budget = 0  # Track how much of the budget is left after clusters with limited points
    for i in range(len(norm_inv_probs)):
        if ideal_samples[i] <= cluster_sizes[i]:
            # We can sample the ideal number from this cluster
            actual_samples[i] = ideal_samples[i]
        else:
            # Not enough points in this cluster, so sample all available points
            actual_samples[i] = cluster_sizes[i]
            # Add the remaining unused budget
            excess_budget += ideal_samples[i] - cluster_sizes[i]
    if ideal_excess < 0: excess_budget -= ideal_excess
    #print("Excess budget:", excess_budget)
    
    # Step 6: Redistribute the excess budget
    # Only distribute to clusters that still have points left to sample
    prev_excess_budget = 0
    remaining_inv_probs, remaining_inv_sum = [], 0
    clusters = [i for i in range(len(cluster_sizes))]
    while excess_budget > 0 and excess_budget != prev_excess_budget:
        #print("Excess budget:", excess_budget)
        remaining_inv_probs = [inv_p if actual_samples[i] < cluster_sizes[i] else 0 for i, inv_p in enumerate(norm_inv_probs)]
        remaining_inv_sum = sum(remaining_inv_probs)
        #print("Remaining inv probs:", np.array(remaining_inv_probs) / remaining_inv_sum)
        # remove clusters that have been fully sampled
        clusters = [i for i in range(len(cluster_sizes)) if remaining_inv_probs[i] > 0]
        remaining_inv_probs = [inv_p for i, inv_p in enumerate(remaining_inv_probs) if inv_p > 0]
        
        if remaining_inv_sum == 0:
            break  # No more clusters to redistribute to
        
        additionals = [0] * len(norm_inv_probs)
        samples = np.random.choice(clusters, p=np.array(remaining_inv_probs)/remaining_inv_sum, size=excess_budget)
        for c in samples: additionals[c] += 1

        for i in range(len(norm_inv_probs)):
            if actual_samples[i] < cluster_sizes[i]:
                # Compute additional samples to allocate
                additional_samples = additionals[i]
                #print("Additional samples:", additional_samples)
                # Ensure we don't exceed the cluster's capacity
                available_capacity = cluster_sizes[i] - actual_samples[i]
                
                if additional_samples <= available_capacity:
                    actual_samples[i] += additional_samples
                    prev_excess_budget = excess_budget
                    excess_budget -= additional_samples
                else:
                    # Take all remaining points from the cluster and update the excess budget
                    actual_samples[i] += available_capacity
                    prev_excess_budget = excess_budget
                    excess_budget -= available_capacity
    
    # Output: The final number of samples to draw from each cluster
    #print("Actual samples:", actual_samples)
    return actual_samples


def reverse_propotional_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    #print("Budget start:", budget)
    cluster_probs = 1 / (np.bincount(cluster_assignments) / len(cluster_assignments))
    cluster_probs = cluster_probs / np.sum(cluster_probs)
    # Calculate cluster size from cluster assignment
    cluster_size = [len(np.where(cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
    sampled_indices = []
    # get number of samples per cluster, with at least one sample per cluster
    cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
    # sample from each cluster based on the number of samples
    for c in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == c)[0]
        sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def reverse_propotional_sampling_clusters_(cluster_assignments, parameters) -> List:
    budget = int(len(cluster_assignments) * 0.2)
    # order the clusters by size
    cluster_sizes = np.bincount(cluster_assignments)
    sorted_clusters = np.argsort(cluster_sizes)
    sampled_indices = []
    for c in sorted_clusters:
        c_budget = budget - len(sampled_indices)
        cluster_indices = np.where(cluster_assignments == c)[0]
        if len(cluster_indices) > c_budget:
            sampled_indices.extend(np.random.choice(cluster_indices, c_budget, replace=False))
        else: sampled_indices.extend(cluster_indices)
        if len(sampled_indices) >= budget:
            break
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def cluster_sampling(search_space, clustering, sampling, semantic_metric='l2_norm', clustering_params:Dict={}, sampling_params:Dict={}, if_runtime_stats=True) -> List:
    runtime_stats = []

    # Cluster the binned data
    start_time = time.time()
    
    cluster_assignments = clustering(search_space, clustering_params)

    sampled_indices = sampling(cluster_assignments, sampling_params)
    if len(sampled_indices) == 0:
        return None, None, runtime_stats, cluster_assignments
    
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points, _ = get_pareto_front(sampled_partitions, semantic_metric)
    
    method_comp = time.time() - start_time
    #points_df['Cluster'] = cluster_assignments

    # Compute the runtime statistics
    if if_runtime_stats:
        runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
        runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats, cluster_assignments

def random_sampling(search_space, semantic_metric='l2_norm', frac=0.5, if_runtime_stats=True) -> List:
    runtime_stats = []
    start_time = time.time()
    # Sample frac of the partitions
    sampled_indices = np.random.choice(len(search_space.candidates), int(len(search_space.candidates) * frac), replace=False)
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points, _ = get_pareto_front(sampled_partitions, semantic_metric)

    method_comp = time.time() - start_time
    # Compute the runtime statistics
    if if_runtime_stats:
        runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
        runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats

def get_points(partitions, semantic_metric) -> List:
    if semantic_metric == 'l2_norm':
        semantics = [p.l2_norm for p in partitions]
    elif semantic_metric == 'gpt_distance':
        semantics = [p.gpt_distance for p in partitions]
    elif semantic_metric == 'KLDiv':
        semantics = [p.KLDiv for p in partitions]
    else: raise ValueError("Invalid semantic metric")
    utility = [p.utility for p in partitions]
    datapoints = [np.array(semantics), np.array(utility)]
    return datapoints

def get_pareto_front(partitions, semantic_metric='l2_norm') -> List:
    datapoints = get_points(partitions, semantic_metric)
    IDs = [p.ID for p in partitions]
    #print(f"Data points: {datapoints}")
    #print("Datapoint shape to compute Pareto points:", np.array(datapoints).shape)
    lst = compute_pareto_front(datapoints)

    # Plot the Pareto front
    pareto_df = pd.DataFrame({'ID': IDs, 'Semantic': datapoints[0], 'Utility': datapoints[1]})
    pareto_df['pareto'] = 0
    pareto_df.loc[lst, 'pareto'] = 1
    pareto_points = pareto_df[pareto_df['pareto'] == 1][['Semantic', 'Utility']]
    pareto_points = pareto_points.values.tolist()
    #print(f"Pareto points: {pareto_points}")
    return datapoints, pareto_points, pareto_df

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
    #######################
    # Drop missing values #
    #######################
    data = raw_data.dropna(subset=[attr, target]) # TODO: Consider handling missing values per dataset

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
        explainable_modeling_one_attr(data_i, target, attr, partition)
    
    return ss

def get_visualization_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins=2, max_num_bins=20, gpt_measure=True):
    #######################
    # Drop missing values #
    #######################
    data = raw_data.dropna(subset=[attr, target]) # TODO: Consider handling missing values per dataset

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
        visualization_one_attr(data_i, target, attr, partition)
    # ANOVA needs to be standardized
    ss.standardize_utility()
    return ss


if __name__ == '__main__':
    #np.random.seed(0)
    f_runtime_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'num_explored_points', 'partition_gen', 'semantic_comp', 'utility_comp', 'method_comp']
    f_quality_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'avg_dist', 'min_num_bins', 'max_num_bins']

    # Load the diabetes dataset
    use_case = 'imputation'
    N_components = [3]
    rounds = 50
    gpt_measure = True
    #raw_data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    raw_data = pd.read_csv(os.path.join(ppath, 'data', 'titanic', 'train.csv'))
    raw_data = raw_data[['Age', 'Fare', 'SibSp', 'Survived', 'Pclass', 'Parch', 'PassengerId']]
    dataset = 'titanic' #'pima'
    min_num_bins = 2
    max_num_bins = 20
    target = 'Survived'
    semantic_metrics = ['gpt_distance', 'l2_norm', 'KLDiv']
    #attributes = {'Age': [-1, 18, 35, 50, 65, 100], 'Glucose': [-1, 140, 200], 'BMI': [-1, 18.5, 25, 30, 68], }
    attributes = {'Age': [-1, 18, 35, 50, 65, 100], 'Fare': [-1, 10, 20, 30, 40, 50, 100, 600], 'SibSp': [-1, 1, 2, 3, 4, 5, 6, 7, 8]}

    # Make a new folder to save the results
    date_today = datetime.datetime.today().strftime('%Y_%m_%d')
    dst = os.path.join(ppath, 'exp', f"{dataset}.{date_today}.linkage_distributions.run2")
    dst_folder = os.path.join(dst, use_case)
    dst_fig_folder = os.path.join(dst, use_case, 'figs')
    if not os.path.exists(dst):
        os.mkdir(dst)
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
    elif not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)

    for attr, gold_standard_bins in attributes.items():
        f_runtime = []
        f_quality = []
        
        
        for j in range(1):
            if use_case == 'modeling':
                ss = get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
            elif use_case == 'imputation':
                ss = get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
            else:
                raise ValueError("Invalid use case")
            
            for semantic_metric in semantic_metrics:

                for i in range(rounds):
                    datapoints, gt_pareto_points, points_df = get_pareto_front(ss.candidates, semantic_metric)
                    f_runtime.append([use_case, dataset, attr, 'exhaustive', semantic_metric] + get_runtime_stats(ss, semantic_metric) + [0])

                    cluster_params = {'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                    sampling_params = {'num_samples': 1}
                    explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, random_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                    distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                    method_name = f'cs_linkage_rand'
                    f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                    f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                    points_df["Cluster"] = clusters
                    f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                    f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                    

                    for p in [0.2, 0.25, 0.3]:
                        cluster_params = {'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, proportional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_linkage_prop_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                        cluster_params = {'t': int(len(ss.candidates)/10), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_linkage_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'eps': 0.03, 'min_samples': 3}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, DBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_dbscan_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'min_cluster_size': 3}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, HDBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_hdbscan_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)
                        points_df["Cluster"] = clusters
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                    for frac in [0.2, 0.4, 0.8]:
                        method_name = f'random_sampling_{frac}'
                        explored_points, est_pareto_points, runtime_stats = random_sampling(ss, semantic_metric, frac=frac)
                        distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, distance, min_num_bins, max_num_bins])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i] + runtime_stats)


        f_runtime_df = pd.DataFrame(f_runtime, columns=f_runtime_cols)
        f_quality_df = pd.DataFrame(f_quality, columns=f_quality_cols)
        f_runtime_df.to_csv(os.path.join(dst_folder, f'{attr}_runtime.csv'), index=False)
        f_quality_df.to_csv(os.path.join(dst_folder, f'{attr}_quality.csv'), index=False)
        