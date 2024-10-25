import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *
from end_to_end_pipeline import *
SEMANTICS = ['l2_norm', 'KLDiv', 'gpt_distance']
f_quality_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'avg_dist']
f_runtime_cols = ['use_case', 'dataset', 'attr', 'method', 'semantic_metric', 'round', 'num_explored_points']

if __name__ == '__main__':
    dataset = 'pima'
    use_case = 'modeling'
    rounds = 20

    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    # Make a new folder to save the results
    date_today = datetime.datetime.today().strftime('%Y_%m_%d')
    dst = os.path.join(ppath, 'exp', f"{dataset}.{date_today}.test1")
    dst_folder = os.path.join(dst, use_case)
    dst_fig_folder = os.path.join(dst, use_case, 'figs')
    if not os.path.exists(dst):
        os.mkdir(dst)
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)
    elif not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
        os.mkdir(dst_fig_folder)


    for attr in attributes:
        f_quality = []
        f_runtime = []
        # load experiment data
        data = pd.read_csv(os.path.join(ppath, 'experiment_data', dataset, use_case, f'{attr}.csv'))
        ss = TestSearchSpace(data)
        
        for semantic_metric in SEMANTICS:

            for i in range(rounds):
                datapoints, gt_pareto_points, points_df = get_pareto_front(ss.candidates, semantic_metric)

                cluster_params = {'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                sampling_params = {'num_samples': 1}
                explored_points, est_pareto_points, _, clusters = cluster_sampling(ss, linkage_distributions, random_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                method_name = f'cs_linkage_rand'
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                if i < 5:
                    points_df["Cluster"] = clusters
                    f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                    f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                
                

                for p in [0.2, 0.25, 0.3]:
                        cluster_params = {'t': int(len(ss.candidates)/5), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, proportional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_linkage_prop_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                        if i < 5:
                            points_df["Cluster"] = clusters
                            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                            f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                        cluster_params = {'t': int(len(ss.candidates)/10), 'criterion': 'maxclust'}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, linkage_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_linkage_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                        if i < 5:
                            points_df["Cluster"] = clusters
                            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                            f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'eps': 0.02, 'min_samples': 2}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, DBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_dbscan_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                        if i < 5:
                            points_df["Cluster"] = clusters
                            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                            f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                        
                        cluster_params = {'min_cluster_size': 3}
                        sampling_params = {'p': p}
                        explored_points, est_pareto_points, runtime_stats, clusters = cluster_sampling(ss, HDBSCAN_distributions, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                        average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                        method_name = f'cs_hdbscan_reverse_{p}'
                        f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                        f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                        if i < 5:
                            points_df["Cluster"] = clusters
                            f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                            f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                for frac in [0.2, 0.4, 0.5, 0.8]:   
                    method_name = f'random_sampling_{frac}'
                    explored_points, est_pareto_points, _ = random_sampling(ss, semantic_metric, frac=frac, if_runtime_stats=False)
                    average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                    f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                    f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                    if i < 5:
                        f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, None, method_name)
                        f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')


        f_quality_df = pd.DataFrame(f_quality, columns=f_quality_cols)
        f_quality_df.to_csv(os.path.join(dst_folder, f'{attr}_quality.csv'), index=False)
        f_runtime_df = pd.DataFrame(f_runtime, columns=f_runtime_cols)
        f_runtime_df.to_csv(os.path.join(dst_folder, f'{attr}_runtime.csv'), index=False)
        #break
