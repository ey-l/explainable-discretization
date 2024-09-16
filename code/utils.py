import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *

openai.api_key = "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"
MODEL_ID = "gpt-3.5-turbo"

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

def eval_pareto_points(pareto_points:List, est_pareto_points:List, debug=False) -> float:
    """
    Evaluate the Pareto front using nearest neighbor search.
    Args:
        pareto_points (List): Ground truth Pareto front
        est_pareto_points (List): Estimated Pareto front
    Returns:
        float: Average distance between the estimated and ground truth Pareto fronts
    """
    est_pareto_points = np.array(est_pareto_points)
    pareto_points = np.array(pareto_points)
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(est_pareto_points)
    # Find nearest neighbor in estimated curve for each point in ground truth curve
    distances, _ = tree.query(pareto_points)
    # Average distance
    average_distance = np.mean(distances)
    if debug: print(f"Average Distance: {average_distance}")
    return average_distance

def plot_pareto_points(pareto_points:List, est_pareto_points:List, datapoints:List, explored_points:List=None) -> Tuple:
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
    datapoints = np.array(datapoints)
    # Set size of the plot
    f, ax = plt.subplots(figsize=(8, 6))
    #f, ax = plt.subplots()
    ax.scatter(datapoints[0], datapoints[1], c='gray', label='Data Points', alpha=0.3)
    ax.scatter(explored_points[0], explored_points[1], c='blue', label='Explored Points', alpha=0.3)
    ax.plot(pareto_points[:, 0], pareto_points[:, 1], 's-', c='red', label='Ground Truth')
    ax.plot(est_pareto_points[:, 0], est_pareto_points[:, 1], 'x-', c='green', label='Estimated')
    ax.legend()
    ax.set_xlabel('Semantic Similarity')
    ax.set_ylabel('Utility')
    ax.set_title('Pareto Curve Comparison')
    return f, ax

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