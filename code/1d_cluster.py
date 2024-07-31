import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *

if __name__ == '__main__':
    # Load the diabetes dataset
    df = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))

    # Load the age partitions
    f = open(os.path.join(ppath, 'scratch/age_partitions.json'), "r")
    data = json.load(f)
    f.close()
    binnings = list(eval(data))
    col = 'Age'
    outs = []
    age = list(df['Age'])
    N = len(age)

    # Randomly sample 30% of the data and replace the age values with NaN
    data = df.copy()
    data['Age.gt'] = data['Age']
    df_nan = df.sample(frac=0.3, random_state=42)
    data.loc[df.index.isin(df_nan.index),'Age'] = np.nan

    # Impute the missing values using KNN
    for i in binnings:
        #bins = i
        #semantic = 0
        bins = i['bins']
        semantic = i['gpt']
        data_i = data.copy()
        data_i[col + '.binned'] = pd.cut(data_i[col], bins=bins, labels=bins[1:])
        #data_i[col + '.binned'] = data_i[col + '.binned'].astype('float64')

        imputer = KNNImputer(n_neighbors=len(bins)-1)
        data_imputed = imputer.fit_transform(data_i[col + '.binned'].values.reshape(-1, 1))
        data_imputed = np.round(data_imputed)
        data_i['Age.imputed'] = data_imputed
        data_i[col + '.final'] = pd.cut(data_i[col+'.imputed'], bins=bins, labels=bins[1:])
        data_i[col + '.final'] = data_i[col + '.final'].astype('float64')

        if len(data_i[data_i[col + '.final'].isnull()]) > 200:
            print(f"Skipping {bins}")
            continue
        #data_i['Age.final'] = data_i['Age.final'].fillna(-1)
        value_final = np.array(data_i['Age.final'].values)
        value_final[np.isnan(value_final)] = -1

        # Evaluate data imputation
        data_i['Age.gt'] = pd.cut(data_i['Age.gt'], bins=bins, labels=bins[1:])
        data_i['Age.gt'] = data_i['Age.gt'].astype('float64')
        value_gt = np.array(data_i['Age.gt'].values)
        value_gt[np.isnan(value_gt)] = -1
        #data_i['Age.gt'] = data_i['Age.gt'].fillna(-1)
        #data_i = data_i.dropna(subset=['Age.final', 'Age.gt'])
        impute_accuracy = accuracy_score(value_gt, value_final)

        #print(f"{bins}:", 1-i['gpt'], impute_accuracy)

        hist, bin_edges = np.histogram(age, bins=bins)
        distribution = hist / N

        outs.append({'bins': bins, 'distribution': distribution, 'partitioned': np.array(data_i[col + '.final']), 'gpt': 1-semantic, 'impute_accuracy': impute_accuracy})

    # Compute ground truth Pareto front
    # dictionary to dataframe
    df_out = pd.DataFrame(outs)
    # get partitioned data as a two dimensional array
    partitioned = np.array(df_out['partitioned'].values.tolist())
    datapoints = np.array([np.array(df_out['gpt']), np.array(df_out['impute_accuracy'])])

    pareto=oapackage.ParetoDoubleLong()
    for ii in range(0, datapoints.shape[1]):
        w=oapackage.doubleVector( (datapoints[0,ii], datapoints[1,ii]))
        pareto.addvalue(w, ii)
    lst=pareto.allindices() # the indices of the Pareto optimal designs
    
    # label the Pareto optimal points in the dataframe as 1; otherwise 0
    df_out['pareto'] = 0
    df_out.loc[lst, 'pareto'] = 1

    # Cluster the data
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    pca = PCA(n_components=20)
    X = list(df_out['partitioned'].values)
    processed = pca.fit_transform(X)
    hdbscan_clusters = clusterer.fit_predict(processed)
    # Add the cluster labels to the dataframe
    df_out['cluster'] = hdbscan_clusters

    # Separate the data into outliers and non-outliers
    df_no_outliers = df_out[df_out['cluster'] != -1]
    df_outliers = df_out[df_out['cluster'] == -1]

    # For each cluster in the non-outliers, rank the points based on 'gpt'
    # and get the top 1 point for each cluster
    top_points = []
    for i in range(df_no_outliers['cluster'].nunique()):
        cluster = df_no_outliers[df_no_outliers['cluster'] == i]
        cluster = cluster.sort_values(by='gpt', ascending=False)
        top_points.append(cluster.iloc[0])
    
    # Combine the top points with the outliers, and reset index
    df_top_points = pd.DataFrame(top_points)
    df_combined = pd.concat([df_top_points, df_outliers])
    df_combined = df_combined.reset_index(drop=True)

    # Compute pareto front for the combined data
    datapoints = np.array([np.array(df_combined['gpt']), np.array(df_combined['impute_accuracy'])])

    pareto=oapackage.ParetoDoubleLong()
    for ii in range(0, datapoints.shape[1]):
        w=oapackage.doubleVector( (datapoints[0,ii], datapoints[1,ii]))
        pareto.addvalue(w, ii)
    lst=pareto.allindices() # the indices of the Pareto optimal designs

    # label the Pareto optimal points in the dataframe as 1; otherwise 0
    df_combined['est_pareto'] = 0
    df_combined.loc[lst, 'est_pareto'] = 1
    print(df_combined)
    # get the 'gpt' and 'impute_accuracy' values of the Pareto optimal points
    # as a list of lists
    est_pareto_points = df_combined[df_combined['est_pareto'] == 1][['gpt', 'impute_accuracy']]
    est_pareto_points = est_pareto_points.values.tolist()
    # get the ground truth Pareto optimal points
    pareto_points = df_out[df_out['pareto'] == 1][['gpt', 'impute_accuracy']]
    pareto_points = pareto_points.values.tolist()

    print("Estimated Pareto front:", est_pareto_points)
    print("Ground truth Pareto front:", pareto_points)

    # Compute the Wasserstein distance between the estimated and ground truth Pareto fronts
    #wasserstein = wasserstein_distance(est_pareto_points, pareto_points)
    #print("Wasserstein distance:", wasserstein)

    est_pareto_points = np.array(est_pareto_points)
    pareto_points = np.array(pareto_points)
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(est_pareto_points)
    # Find nearest neighbor in estimated curve for each point in ground truth curve
    distances, indices = tree.query(pareto_points)
    # Average distance
    average_distance = np.mean(distances)
    print(f"Average Distance: {average_distance}")

    plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'o-', label='Ground Truth')
    plt.plot(est_pareto_points[:, 0], est_pareto_points[:, 1], 'x-', label='Estimated')
    plt.legend()
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Curve Comparison')
    plt.show()
