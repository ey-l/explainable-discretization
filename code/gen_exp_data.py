import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from SearchSpace import *
from utils import *
np.set_printoptions(threshold=sys.maxsize)

def load_raw_data(dataset:str) -> pd.DataFrame:
    """
    Load the data for a given dataset, use case, and attribute
    :param dataset: str
    :return: pd.DataFrame
    """
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    
    if dataset in ['pima', 'titanic']:
        raw_data = pd.read_csv(os.path.join(ppath, exp_config['data_path']))
        raw_data = raw_data[exp_config['features'] + [exp_config['target']]]
    elif dataset == 'satimage':
        train = pd.read_csv(os.path.join(ppath, exp_config['data_path_train']), header=None, delim_whitespace=True)
        test = pd.read_csv(os.path.join(ppath, exp_config['data_path_test']), header=None, delim_whitespace=True)
        raw_data = pd.concat([train, test])
        raw_data.columns = ['pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9', 'pixel10', 'pixel11', 'pixel12', 'pixel13', 'pixel14', 'pixel15', 'pixel16', 'pixel17', 'pixel18', 'pixel19', 'pixel20', 'pixel21', 'pixel22', 'pixel23', 'pixel24', 'pixel25', 'pixel26', 'pixel27', 'pixel28', 'pixel29', 'pixel30', 'pixel31', 'pixel32', 'pixel33', 'pixel34', 'pixel35', 'target']
    return raw_data

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
    value_final[np.isnan(value_final)] = -1 # Replace NaN with -1
    value_final = np.round(value_final)

    # Evaluate data imputation
    data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=bins, labels=bins[1:])
    data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
    value_gt = np.array(data[attr + '.gt'].values)
    value_gt[np.isnan(value_gt)] = -1 # Replace NaN with -1
    value_gt = np.round(value_gt)
    impute_accuracy = accuracy_score(value_gt, value_final)

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = impute_accuracy
    return impute_accuracy

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

def get_visualization_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins=2, max_num_bins=20, gpt_measure=True):
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
        visualization_one_attr(data_i, target, attr, partition)
    
    ss.standardize_utility()
    
    return ss


if __name__ == '__main__':
    #np.random.seed(0)
    f_data_cols = ['ID', 'method', 'bins', 'binned_values', 'distribution', 'utility', 'kl_d', 'l2_norm', 'gpt_prompt']
    semantic_metrics = ['gpt_distance', 'l2_norm', 'KLDiv']
    
    # Load the diabetes dataset
    use_case = 'visualization' #'imputation'
    gpt_measure = False
    dataset = 'pima' #'pima'

    # read json file
    exp_config = json.load(open(os.path.join(ppath, 'code', 'configs', f'{dataset}.json')))
    raw_data = load_raw_data(dataset)
    min_num_bins = exp_config['min_num_bins']
    max_num_bins = exp_config['max_num_bins']
    max_num_bins = 3
    target = exp_config['target']
    attributes = exp_config['attributes']

    dst_folder = os.path.join(ppath, 'experiment_data', dataset, use_case)
    if not os.path.exists(dst_folder): 
        os.makedirs(dst_folder)
    else: print(f"Folder {dst_folder} already exists")
    
    for attr, gold_standard_bins in attributes.items():
        f_data = []
            
        if use_case == 'modeling':
            ss = get_explainable_modeling_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
        elif use_case == 'imputation':
            ss = get_data_imputation_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
        elif use_case == 'visualization':
            ss = get_visualization_search_space(raw_data, attr, target, gold_standard_bins, min_num_bins, max_num_bins, gpt_measure)
        else:
            raise ValueError("Invalid use case")
            
        
        for partition in ss.candidates:
            binned_values = partition.binned_values.to_numpy()
            #print(a.tolist())
            #print(a)
            f_data.append([partition.ID, partition.method, np.array(partition.bins), binned_values, partition.distribution, partition.utility, partition.KLDiv, partition.l2_norm, partition.gpt_distance])
            #print(partition.binned_values.values)
        f_data_df = pd.DataFrame(f_data, columns=f_data_cols)
        f_data_df.to_csv(os.path.join(dst_folder, f'{attr}.csv'), index=False)
        break
        