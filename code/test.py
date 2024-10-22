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
    dataset = 'satimage'
    use_case = 'imputation'
    rounds = 50

    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    attributes = exp_config['attributes'].keys()
    # Make a new folder to save the results
    date_today = datetime.datetime.today().strftime('%Y_%m_%d')
    dst = os.path.join(ppath, 'exp', f"{dataset}.{date_today}.linkage_distributions.test1")
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

                cluster_params = {'t': int(len(ss.candidates)/5)}
                sampling_params = {'num_samples': 1}
                explored_points, est_pareto_points, _, clusters = cluster_sampling(ss, proportional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                method_name = f'cluster_sampling_random'
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                points_df["Cluster"] = clusters
                f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                
                explored_points, est_pareto_points, _, clusters = cluster_sampling(ss, proportional_sampling_clusters, semantic_metric, cluster_params, {'num_samples': 2}, False)
                comparable_frac = np.round(len(explored_points[0]) / len(datapoints[0]), 1)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                method_name = f'cluster_sampling_proportional'
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                points_df["Cluster"] = clusters
                f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')

                cluster_params = {'t': int(len(ss.candidates)/10)}
                sampling_params = {'num_samples': 1}
                explored_points, est_pareto_points, _, clusters = cluster_sampling(ss, reverse_propotional_sampling_clusters, semantic_metric, cluster_params, sampling_params, False)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                method_name = f'cluster_sampling_reverse'
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])
                points_df["Cluster"] = clusters
                f, ax = plot_pareto_points(gt_pareto_points, est_pareto_points, explored_points, points_df, method_name)
                f.savefig(os.path.join(dst_fig_folder, f'{attr}.{semantic_metric}.{method_name}.{i}.png'), bbox_inches='tight')
                
                frac=0.2
                method_name = f'random_sampling_{frac}'
                explored_points, est_pareto_points, _ = random_sampling(ss, semantic_metric, frac=frac, if_runtime_stats=False)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])

                frac=comparable_frac
                method_name = f'random_sampling_{0.4}'
                explored_points, est_pareto_points, _ = random_sampling(ss, semantic_metric, frac=frac, if_runtime_stats=False)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])

                frac=0.5
                method_name = f'random_sampling_{frac}'
                explored_points, est_pareto_points, _ = random_sampling(ss, semantic_metric, frac=frac, if_runtime_stats=False)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])

                frac=0.8
                method_name = f'random_sampling_{frac}'
                explored_points, est_pareto_points, _ = random_sampling(ss, semantic_metric, frac=frac, if_runtime_stats=False)
                average_distance = eval_pareto_points(gt_pareto_points, est_pareto_points, debug=True)
                f_quality.append([use_case, dataset, attr, method_name, semantic_metric, i, average_distance])
                f_runtime.append([use_case, dataset, attr, method_name, semantic_metric, i, len(explored_points[0])])

        f_quality_df = pd.DataFrame(f_quality, columns=f_quality_cols)
        f_quality_df.to_csv(os.path.join(dst_folder, f'{attr}_quality.csv'), index=False)
        f_runtime_df = pd.DataFrame(f_runtime, columns=f_runtime_cols)
        f_runtime_df.to_csv(os.path.join(dst_folder, f'{attr}_runtime.csv'), index=False)
        #break
