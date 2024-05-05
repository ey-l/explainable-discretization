import sys
import os
import openai
import json
from scipy.stats import spearmanr, f_oneway
# Project path
ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))
from discretizers import *
from Bucket import *

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

def get_gpt_score(ref_bins, cand_bins, model_id="gpt-3.5-turbo"):
    prompt1 = "You are given a set of bins for human age. Can you describe the semantic meaning of the bins?\nBins: " + str(ref_bins)
    f = open(os.path.join(ppath, 'prompts/surprising.txt'), "r")
    prompt2 = f.read()
    f.close()
    prompt2 = prompt2 + "#### INPUT:\nData context: human age\nSemantic gold-standard binning: " + str(ref_bins) + "\nCandidate binning: " + str(cand_bins) + "\n\n" + "#### OUTPUT:"
    messages.append({"role": "user", "content": prompt1})
    msgresponse = call_gpt(messages, model=model_id)
    messages.append({"role": "assistant", "content": msgresponse})
    msgresponse = get_message_memory(prompt2, messages, model_id)

    # add to binning
    try:
        return float(msgresponse)
    except: 
        return 0.5

if __name__ == "__main__":
    openai.api_key = "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"
    model_id = "gpt-4"
    
    data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    data = data[(data['Age'] > 0) & (data['Age'] < 60)]
    
    gold_binning = {'bins':[0, 19, 45, 65, 85, 100],
                    'labels': ['0-18', '19-44', '45-64', '65-84', '85-100'],
                    'method': 'gold-standard'}
    binnings = [
        gold_binning,
        {'bins': list(range(0, 101, 2)), 'labels': list(range(1, 51)), 'method': 'equal-width'},
        {'bins': list(range(0, 101, 5)), 'labels': list(range(1, 21)), 'method': 'equal-width'},
        {'bins': list(range(0, 101, 10)), 'labels': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100'], 'method': 'equal-width'},
        {'bins': list(range(0, 101, 20)), 'labels': ['0-19', '20-39', '40-59', '60-79', '80-100'], 'method': 'equal-width'},
        {'bins': list(range(0, 101, 25)), 'labels': ['0-24', '25-49', '50-74', '75-100'], 'method': 'equal-width'},
        {'bins': list(range(0, 101, 30)), 'labels': ['0-29', '30-59', '60-89'], 'method': 'equal-width'},
        {'bins': [0, 15, 25, 65, 101], 'labels': ['0-14', '15-24', '25-64', '65-100'], 'method': 'expert'},
        {'bins': [0, 3, 40, 60, 101], 'labels': ['0-2', '3-39', '40-59', '60-100'], 'method': 'expert'},
        {'bins': [0, 10, 30, 60, 80, 101], 'labels': ['0-9', '10-29', '30-59', '60-79', '80-100'], 'method': 'expert'},
    ]

    attr = 'Age'
    target = 'Outcome'
    for i in range(2, 11):
        # Add equal frequency binning
        intervals = equal_frequency(data, i, [attr], 0)
        binnings.append({'bins': intervals[attr], 'labels': list(range(1, i+1)), 'method': 'equal-frequency'})
        # Add chi-merge binning
        intervals = chimerge_wrap(data, [attr], target, i)
        binnings.append({'bins': intervals[attr], 'labels': list(range(1, i+1)), 'method': 'chi-merge'})
        # Add kmeans binning
        #intervals = KBinsDiscretizer_wrap(data, [attr], i)
        #binnings.append({'bins': intervals[attr], 'labels': list(range(1, i+1)), 'method': 'kmeans'})

    # Populate the dictionary with stats
    for i in range(len(binnings)):
        print(f"Processing binning {binnings[i]['bins']}")
        messages = []
        df = data.copy()
        df = df[df[attr] > 0]
        # Get the GPT score
        if binnings[i]['method'] == 'gold-standard':
            binnings[i]['gpt'] = 0.0
        binnings[i]['gpt'] = get_gpt_score(gold_binning['bins'], binnings[i]['bins'], model_id=model_id)
        print(f'GPT distance: {binnings[i]["gpt"]}')

        # Calculate L2 norm between each binning and gpt binning
        x = np.array(binnings[i]['bins'])
        y = np.array(gold_binning['bins'])
        x, y = zero_pad_vectors(x, y)
        l2_norm = np.linalg.norm(x - y)
        binnings[i]['l2_norm'] = l2_norm
        print(f'L2 norm between binning {i} and gpt binning: {l2_norm}')

        # Calculate spearman correlation
        df[attr] = pd.cut(df[attr], bins=binnings[i]['bins'], labels=False)
        df[attr] = df[attr].astype('float64')
        df = df[[attr, target]]
        df1 = df.groupby(attr).mean().reset_index()
        rho, p = spearman = spearmanr(df1[attr], df1[target])
        binnings[i]['spearman'] = rho
        print(f'Spearman correlation: {rho}')

        # Calculate ANOVA
        df1 = df.groupby(attr)[target]
        df1 = [group[1] for group in df1]
        f, p = f_oneway(*df1)
        binnings[i]['anova'] = f
        print(f'ANOVA: {f}')
    
    # Save the results to json
    with open(os.path.join(ppath, 'scratch', 'binnings.json'), 'w') as f:
        json.dump(str(binnings), f)