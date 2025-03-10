import sys
import os

ppath = sys.path[0] + '/../../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *

openai.api_key = "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"
MODEL_ID = "gpt-3.5-turbo"

def pairwise_distance(X, metric):
    distances = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            distances[i, j] = metric(X[i], X[j])
    return distances

def zero_pad_vectors(v1, v2):
    # Identify the length of the longer vector
    max_len = max(len(v1), len(v2))

    # Pad each vector with zeros at the end to match the length of the longer vector
    v1_padded = np.pad(v1, (0, max_len - len(v1)), 'constant')
    v2_padded = np.pad(v2, (0, max_len - len(v2)), 'constant')

    return v1_padded, v2_padded

def call_gpt(prompt, model="gpt-3.5-turbo") -> List[str]:
    try:
        result = openai.ChatCompletion.create(
            model=model, messages=prompt
        )
        return result.choices[0].message.content
    except Exception as e:
        print("GPT Error:", e)
        return ""
    
def get_message_memory(newquestion, lastmessage, model_id="gpt-3.5-turbo"):
    # Append the new question to the last message
    #if len(str(lastmessage)) > 16385
    # Make a copy of the last message
    newmessage = lastmessage.copy()
    newmessage.append({"role": "user", "content": newquestion})
    # We limit the length of the message to 16385 tokens
    if len(str(newmessage)) > 16385:
        newmessage = newmessage[-2:]
    lastmessage = newmessage

    # Print the new answer
    msgresponse = call_gpt(lastmessage, model=model_id)
    #print(msgresponse)

    # We return the new answer
    return msgresponse

def get_gpt_score(ref_bins, cand_bins, context:str='human age', model_id:str="gpt-3.5-turbo"):
    messages = []
    prompt1 = f"You are given a set of bins for {context}. Can you describe the semantic meaning of the bins?\nBins: " + str(ref_bins)
    f = open(os.path.join(ppath, 'prompts/surprising.txt'), "r")
    prompt2 = f.read()
    f.close()
    prompt2 = prompt2 + f"#### INPUT:\nData context: {context}\nSemantic gold-standard binning: " + str(ref_bins) + "\nCandidate binning: " + str(cand_bins) + "\n\n" + "#### OUTPUT:"
    messages.append({"role": "user", "content": prompt1})
    msgresponse = call_gpt(messages, model=model_id)
    messages.append({"role": "assistant", "content": msgresponse})
    msgresponse = get_message_memory(prompt2, messages, model_id)

    # add to binning
    try:
        return float(msgresponse)
    except: 
        return 0.5

def average_distance(ground_truth:List, estimated:List, debug=False) -> float:
    """
    Evaluate the Pareto front using nearest neighbor search.
    Args:
        ground_truth (List): Ground truth Pareto front
        estimated (List): Estimated Pareto front
    Returns:
        float: Average distance between the estimated and ground truth Pareto fronts
    """
    estimated = np.array(estimated)
    ground_truth = np.array(ground_truth)
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(estimated)
    # Find nearest neighbor in estimated curve for each point in ground truth curve
    distances, _ = tree.query(ground_truth)
    # Average distance
    average_distance = np.mean(distances)
    if debug: print(f"Average Distance: {average_distance}")
    return average_distance

def euclidean_distance(point_a, point_b):
    """Compute the Euclidean distance between two points."""
    return np.linalg.norm(point_a - point_b)

def generational_distance(ground_truth, estimated):
    """
    Calculate Generational Distance (GD).
    Args:
        estimated (ndarray): Estimated Pareto front (N x d).
        ground_truth (ndarray): Ground truth Pareto front (M x d).
    Returns:
        float: GD score.
    """
    estimated = np.array(estimated)
    ground_truth = np.array(ground_truth)
    distances = [min(euclidean_distance(e, g) for g in ground_truth) for e in estimated]
    return np.sqrt(np.mean(np.square(distances)))

def inverted_generational_distance(ground_truth, estimated):
    """
    Calculate Inverted Generational Distance (IGD).
    Args:
        estimated (ndarray): Estimated Pareto front (N x d).
        ground_truth (ndarray): Ground truth Pareto front (M x d).
    Returns:
        float: IGD score.
    """
    estimated = np.array(estimated)
    ground_truth = np.array(ground_truth)
    distances = [min(euclidean_distance(g, e) for e in estimated) for g in ground_truth]
    return np.sqrt(np.mean(np.square(distances)))

def hausdorff_distance(ground_truth, estimated):
    """
    Calculate Hausdorff Distance (HD) as max(GD, IGD).
    Args:
        estimated (ndarray): Estimated Pareto front (N x d).
        ground_truth (ndarray): Ground truth Pareto front (M x d).
    Returns:
        float: HD score.
    """
    gd = generational_distance(ground_truth, estimated)
    igd = inverted_generational_distance(ground_truth, estimated)
    return max(gd, igd)

def plot_pareto_points(pareto_points:List, est_pareto_points:List, explored_points:List=None, points_df=None, title:str='') -> Tuple:
    """
    Plot the estimated and ground truth Pareto fronts.
    Args:
        pareto_points (List): Ground truth Pareto front
        est_pareto_points (List): Estimated Pareto front
    """
    # Sort the points for plotting
    pareto_points = sorted(pareto_points, key=lambda x: x[0])
    est_pareto_points = sorted(est_pareto_points, key=lambda x: x[0])
    # Plot the Pareto front
    pareto_points = np.array(pareto_points)
    est_pareto_points = np.array(est_pareto_points)
    #datapoints = np.array(datapoints)
    # Set size of the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    #f, ax = plt.subplots()
    #ax.scatter(datapoints[0], datapoints[1], c='gray', label='Data Points', alpha=0.3)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
    if points_df is not None:
        # Plot clusters
        colors = cm.rainbow(np.linspace(0, 1, len(points_df['Cluster'].unique())))
        for cluster in points_df['Cluster'].unique():
            cluster_points = points_df[points_df['Cluster'] == cluster]
            marker_index = int(cluster % len(markers))
            ax.scatter(cluster_points['Semantic'], cluster_points['Utility'], label=cluster, color=colors[cluster], alpha=0.5, marker=markers[marker_index])

    ax.scatter(explored_points[0], explored_points[1], c='gray', label='Explored Points', marker='x',)
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], '+-', c='red', label='Ground Truth')
    ax.plot(est_pareto_points[:, 0], est_pareto_points[:, 1], 'x-', c='green', label='Estimated')
    ax.legend(bbox_to_anchor=(1, 1),ncol=3)
    ax.set_xlabel('Semantic Distance', fontsize=14)
    ax.set_ylabel('Utility', fontsize=14)
    if title != '':
        ax.set_title(title, fontsize=14)
    else: ax.set_title('Pareto Curve Comparison', fontsize=14)
    return fig, ax

def plot_clusters(points_df=None, title:str='') -> Tuple:
    """
    Plot the estimated and ground truth Pareto fronts.
    Args:
        pareto_points (List): Ground truth Pareto front
        est_pareto_points (List): Estimated Pareto front
    """
    #datapoints = np.array(datapoints)
    # Set size of the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    #f, ax = plt.subplots()
    #ax.scatter(datapoints[0], datapoints[1], c='gray', label='Data Points', alpha=0.3)
    markers = ["." , "," , "o" , "v" , "^" , "<", ">"]
    if points_df is not None:
        # Plot clusters
        colors = cm.rainbow(np.linspace(0, 1, len(points_df['Cluster'].unique())))
        for cluster in points_df['Cluster'].unique():
            cluster_points = points_df[points_df['Cluster'] == cluster]
            marker_index = int(cluster % len(markers))
            ax.scatter(cluster_points['Semantic'], cluster_points['Utility'], label=cluster, color=colors[cluster], alpha=0.5, marker=markers[marker_index])

    ax.legend(bbox_to_anchor=(1, 1),ncol=3)
    ax.set_xlabel('Semantic Distance', fontsize=14)
    ax.set_ylabel('Utility', fontsize=14)
    if title != '':
        ax.set_title(title, fontsize=14)
    else: ax.set_title('Pareto Curve Comparison', fontsize=14)
    return fig, ax

def compute_pareto_front(datapoints:List) -> List:
    """
    Compute the Pareto front for a set of data points.
    Args:
        datapoints (List): Data points, 
            where first dimension is semantic similarity and second dimension is utility
    Returns:
        List: Pareto front, a list of indices of the Pareto optimal designs
    """
    datapoints = np.array(datapoints)
    pareto=oapackage.ParetoDoubleLong()
    for ii in range(0, datapoints.shape[1]):
        w=oapackage.doubleVector((datapoints[0,ii], datapoints[1,ii]))
        pareto.addvalue(w, ii)
    lst=pareto.allindices() # the indices of the Pareto optimal designs
    return lst

def get_pareto_points(data, semantic_col:str='gpt', utility_col:str='impute_accuracy') -> List:
    """
    Get the Pareto front from the data.
    Args:
        data (DataFrame): DataFrame containing the data
        semantic_col (str): Column name for semantic similarity
        utility_col (str): Column name for utility
    Returns:
        List: Pareto front
    """
    # Compute Pareto front for the data
    datapoints = [np.array(data[semantic_col]), np.array(data[utility_col])]
    lst = compute_pareto_front(datapoints)
    # label the Pareto optimal points in the dataframe as 1; otherwise 0
    data['pareto'] = 0
    data.loc[lst, 'pareto'] = 1
    # get the Pareto optimal points
    pareto_points = data[data['pareto'] == 1][[semantic_col, utility_col]]
    pareto_points = pareto_points.values.tolist()
    return pareto_points