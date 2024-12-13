import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *
from end_to_end_pipeline import *

class Cluster():
    def __init__(self, clusterID):
        self.count = 0
        self.value = 0.0
        self.clusterID = clusterID
        self.points = []
        self.unexplored_start_index = 0

class UCB():
    def __init__(self, alpha=2) -> None:
        """
        Initialize the UCB algorithm with the given alpha value.
        :param alpha: The exploration parameter.
        """
        self.alpha = alpha
        self.explored_nodes = []
        self.fully_explored_clusters = []
    
    def initialize(self, cluster_assigns:List, search_space, semantic_metric:str):
        self.cluster_assigns = cluster_assigns
        self.search_space = search_space
        self.num_clusters = len(set(cluster_assigns))
        self.clusters = [Cluster(i) for i in range(self.num_clusters)]
        self.semantic_metric = semantic_metric
        
        for i, c in enumerate(cluster_assigns):
            # List of Partitions
            self.clusters[c].points.append(self.search_space.candidates[i])
        
        # Shuffle the points in each cluster now 
        # So later we explore in order
        for c in self.clusters:
            random.shuffle(c.points)
            c.count += 1
            datapoint = get_points([c.points[c.unexplored_start_index]], self.semantic_metric)
            c.value += datapoint[0][0] + datapoint[1][0]
            self.explored_nodes.append(c.points[c.unexplored_start_index])
            c.unexplored_start_index += 1 # Advance the unexplored start index
    
    def _select_cluster(self):
        # Select the cluster with the highest value
        max_value = -1
        selected_cluster = None
        for c in self.clusters:
            # Scale the exploratory right hand term
            ucb = c.value / c.count + self.alpha * np.sqrt(2 * np.log(len(self.explored_nodes)) / c.count)
            if ucb > max_value:
                max_value = ucb
                selected_cluster = c
        return selected_cluster
    
    def explore(self):
        while True:
            selected_cluster = self._select_cluster()
            if selected_cluster.unexplored_start_index < len(selected_cluster.points): break
            self.fully_explored_clusters.append(selected_cluster)
            self.clusters.remove(selected_cluster)

        datapoint = get_points([selected_cluster.points[selected_cluster.unexplored_start_index]], self.semantic_metric)
        selected_cluster.value += datapoint[0][0] + datapoint[1][0]
        selected_cluster.count += 1
        self.explored_nodes.append(selected_cluster.points[selected_cluster.unexplored_start_index])
        selected_cluster.unexplored_start_index += 1
        return selected_cluster

def UCB_estimate(alpha, search_space, clustering, semantic_metric='l2_norm', clustering_params:Dict={}, sampling_params:Dict={}, if_runtime_stats=True) -> List:
    """
    Estimate the Pareto front using the UCB algorithm.
    When budget is less than the number of clusters, we use random sampling.
    """
    runtime_stats = []

    # Cluster the binned data
    start_time = time.time()
    
    cluster_assignments = clustering(search_space, clustering_params)

    p = sampling_params['p']
    budget = int(len(cluster_assignments) * p)
    if budget < len(np.unique(cluster_assignments)):
        sampled_indices = random_sampling_clusters_robust(cluster_assignments, sampling_params)
        sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    else:
        ucb = UCB(alpha=alpha)
        ucb.initialize(cluster_assignments, search_space, semantic_metric)
        budget = budget - len(np.unique(cluster_assignments))
        for _ in range(budget):
            ucb.explore()
        sampled_partitions = ucb.explored_nodes
        sampled_indices = [p.ID for p in sampled_partitions]

    if search_space.candidates[0] not in sampled_partitions:
        sampled_partitions.append(search_space.candidates[0])
    explored_points, pareto_points, _ = get_pareto_front(sampled_partitions, semantic_metric)

    method_comp = time.time() - start_time
    #points_df['Cluster'] = cluster_assignments

    # Compute the runtime statistics
    if if_runtime_stats:
        runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
        runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats, cluster_assignments

if __name__ == '__main__':
    dataset = 'pima'
    use_case = 'modeling'
    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()

    for attr in attributes:
        f_quality = []
        f_runtime = []
        # load experiment data
        if attr == "Glucose":
            data = pd.read_csv(os.path.join(ppath, 'experiment_data', dataset, use_case, f'{attr}.csv'))
            ss = TestSearchSpace(data)
            break
    
    semantic_metric = 'l2_norm'
    search_space = ss
    datapoints, gt_pareto_points, points_df = get_pareto_front(ss.candidates, semantic_metric)

    parameters = {'t': 0.5, 'criterion': 'distance'}
    t = parameters['t']
    criterion = parameters['criterion']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    Z = linkage(X, method='ward')
    #fig = plt.figure(figsize=(25, 10))
    #dn = dendrogram(Z, color_threshold=t)
    agg_clusters = fcluster(Z, t=t, criterion=criterion)
    agg_clusters = [x-1 for x in agg_clusters] # 0-indexing
    #print(Z)

    avg_distance_results = []
    for round in range(1):
        # Create dendrogram
        ucb = UCB()
        ucb.initialize(agg_clusters, search_space, semantic_metric)
        for i in range(10):
            print(f"Round {i}")
            print(ucb.explore().clusterID)

        explored_nodes = ucb.explored_nodes
        explored_points, est_pareto_points, _ = get_pareto_front(explored_nodes, semantic_metric)
        print(est_pareto_points)
        average_distance = average_distance(gt_pareto_points, est_pareto_points, debug=True)
        avg_distance_results.append(average_distance)

        # Sort the points for plotting
        gt_pareto_points = sorted(gt_pareto_points, key=lambda x: x[0])
        est_pareto_points = sorted(est_pareto_points, key=lambda x: x[0])
        # Plot the Pareto front
        gt_pareto_points = np.array(gt_pareto_points)
        est_pareto_points = np.array(est_pareto_points)
        #datapoints = np.array(datapoints)
        # Set size of the plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(explored_points[0], explored_points[1], c='gray', label='Explored Points', marker='x',)
        ax.plot(gt_pareto_points[:, 0], gt_pareto_points[:, 1], '+-', c='red', label='Ground Truth')
        ax.plot(est_pareto_points[:, 0], est_pareto_points[:, 1], 'x-', c='green', label='Estimated')
        ax.legend(bbox_to_anchor=(1, 1),ncol=3)
        ax.set_xlabel('Semantic Distance', fontsize=14)
        ax.set_ylabel('Utility', fontsize=14)
        ax.set_title('Pareto Curve Estimated vs. Ground-Truth', fontsize=14)

        fig.savefig(os.path.join(ppath, 'code', 'plots', f'UCB_{attr}_{round}.png'), bbox_inches='tight')
    
    # plot the average distance as a boxplot
    fig, ax = plt.subplots()
    ax.boxplot(avg_distance_results)
    ax.set_xlabel('UCB')
    ax.set_ylabel('Average Distance')
    fig.savefig(os.path.join(ppath, 'code', 'plots', f'UCB_{attr}_boxplot.png'), bbox_inches='tight')