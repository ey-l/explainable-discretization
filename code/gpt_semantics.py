import sys
import os

# Project path
ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))
sys.path.append(os.path.join(ppath, 'code', 'framework'))
from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *


if __name__ == "__main__":
    
    openai.api_key = "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"
    model_id = MODEL_ID
    
    dataset = 'diabetes'
    data = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', f'{dataset}.csv'))
    attr = 'BMI'
    target = 'Outcome'

    # Gold standard binning from literature (NOT GPT)
    gold_binning = {'bins':[0, 18.5, 25, 30, 68],
                    'labels': ['0-18.5', '18.5-25', '25-30', '30-68'],
                    'method': 'gold-standard'}
    binnings = [
        gold_binning,
    ]

    # Generate strategies
    start_time = time.time()
    for i in range(2, 11):
        # Equal width
        intervals = equal_width(data, i, [attr], 0)
        binnings.append({'bins': intervals[attr], 'labels': list(range(1, i+1)), 'method': 'equal-width'})
        #rounded = [round(x) for x in intervals[attr]]
        #binnings.append({'bins': rounded, 'labels': list(range(1, i+1)), 'method': 'equal-width-rounded'})
        # Add equal frequency binning
        intervals = equal_frequency(data, i, [attr], 0)
        binnings.append({'bins': intervals[attr], 'labels': list(range(1, i+1)), 'method': 'equal-frequency'})
        #rounded = [round(x) for x in intervals[attr]]
        #binnings.append({'bins': rounded, 'labels': list(range(1, i+1)), 'method': 'equal-frequency-rounded'})
        # Add chi-merge binning
        intervals = chimerge_wrap(data, [attr], target, i)
        binnings.append({'bins': intervals[attr], 'labels': list(range(1, i+1)), 'method': 'chi-merge'})
        #rounded = [round(x) for x in intervals[attr]]
        #binnings.append({'bins': rounded, 'labels': list(range(1, i+1)), 'method': 'chi-merge-rounded'})
        # Add kmeans binning
        #intervals = KBinsDiscretizer_wrap(data, [attr], i)
        #binnings.append({'bins': intervals[attr], 'labels': list(range(1, i+1)), 'method': 'kmeans'})
    print(f"Number of binning strategies: {len(binnings)}")
    print(f"Time to generate binning strategies: {time.time() - start_time}")

    
    # Populate the dictionary with stats
    start_time = time.time()
    for i in range(len(binnings)):
        print(f"Processing binning {binnings[i]['bins']} from method {binnings[i]['method']}")
        messages = []
        # Get the GPT score
        if binnings[i]['method'] == 'gold-standard':
            binnings[i]['gpt'] = 0.0
        binnings[i]['gpt'] = get_gpt_score(gold_binning['bins'], binnings[i]['bins'], model_id=model_id)
        print(f'GPT distance: {binnings[i]["gpt"]}')
    print(f"Time to get semantic similarity of binning strategies using GPT-4: {time.time() - start_time}")

    # Save the results to json
    with open(os.path.join(ppath, 'scratch', f'{dataset}_{attr}_{model_id}.json'), 'w') as f:
        json.dump(str(binnings), f)
    