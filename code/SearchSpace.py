import sys
import os

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))

from import_packages import *
from discretizers import *
from utils import *

class Bucket:
    """
    Class for bucket.
    """
    def __init__(self, startpoint:float, endpoint:float, count:int, label:Union[str, int]):
        """
        Initialize the bucket with startpoint, endpoint, count, and label.
        Value included in the bucket is in the range [startpoint, endpoint].
        """
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.count = count
        self.label = label
    
    def __repr__(self):
        return f'Bucket({self.startpoint}, {self.endpoint}, {self.count}, {self.label})'
    
    def __eq__(self, __value: object) -> bool:
        return self.startpoint == __value.startpoint and self.endpoint == __value.endpoint and self.count == __value.count and self.label == __value.label
    
    def __lt__(self, __value: object) -> bool:
        return self.endpoint < __value.startpoint
    
    def __le__(self, __value: object) -> bool:
        return self.endpoint <= __value.startpoint
    
    def __gt__(self, __value: object) -> bool:
        return self.startpoint > __value.endpoint
    
    def __ge__(self, __value: object) -> bool:
        return self.startpoint >= __value.endpoint
    
    def value_in_bucket(self, value:float) -> bool:
        return self.startpoint <= value and value <= self.endpoint
    

class Partition:
    """
    Class for a list of buckets
    A Partition is a binning strategy.
    """
    def __init__(self, bins:List, binned_values:List, values:List, buckets:List[Bucket]=None, method:str=None, attribute:str=None, ref_bucket_list=None, gold_standard:bool=False, ID:int=None, gpt_measure=False):
        #print(method,bins)
        if ID is not None: self.ID = ID
        # Bins to apply discretization to data
        self.bins = bins
        self.binned_values = binned_values
        self.gpt_measure = gpt_measure

        # check if binned_values has any nan values
        if binned_values.isnull().any():
            # print the index of the nan values
            print("Nan value at:", binned_values[binned_values.isnull()])
        
        # only count non-none values
        values = [round(x) if isinstance(x, (int, float, complex, np.int64, np.float64)) and not isinstance(x, bool) else None for x in values]
        self.total_count = len([x for x in values if x is not None])
        
        # Buckets to calculate KL-divergence
        if buckets is not None: self.buckets = buckets
        elif values is not None: 
            self.buckets = self._create_buckets(bins, values)
            #self.value_odict = self._create_value_odict(values)
        else: raise ValueError('Either buckets or values must be provided.')
        
        self.start_value = sorted(values)[0]
        self.end_value = sorted(values)[-1]
        self.method = method # method used to create the buckets
        self.attribute = attribute # attribute used to create the buckets

        self.f_time = []
        self.utility = None
        self._set_distribution(values)
        self._init_semantics(ref_bucket_list, gold_standard)
        
    
    def _init_semantics(self, ref_bucket_list, gold_standard) -> None:
        if ref_bucket_list is not None:
            self.KLDiv = self.cal_KLDiv(ref_bucket_list) # Kullback-Leibler divergence
            self.l2_norm = self.cal_l2_norm(ref_bucket_list) # L2 norm
            self.gpt_distance = self.cal_gpt_distance(model_id=MODEL_ID, ref_bucket_list=ref_bucket_list) # GPT distance
        else: 
            self.KLDiv = None
            self.l2_norm = None
            self.gpt_distance = None

        # If gold standard, set all semantic distances to 0
        if gold_standard:
            self.KLDiv = 0
            self.l2_norm = 0
            self.gpt_distance = 0

    def _set_distribution(self, values:List) -> None:
        hist, _ = np.histogram(values, bins=self.bins)
        self.distribution = hist / len(values)

    def __repr__(self) -> str:
        return f'Partition({self.bins}, {self.buckets}, {self.method}, {self.KLDiv}, {self.l2_norm})'

    def set_KLDiv(self, score:float) -> None:
        """
        Set the Kullback-Leibler divergence (KL-divergence) score.
        """
        self.KLDiv = score

    def _create_buckets(self, bins, values:List) -> List[Bucket]:
        """
        Create buckets from the bins and unique values.
        """
        buckets = []
        unique_values = np.array(sorted(list(set(values))))
        counter = Counter(values)
        for i in range(len(bins)-1):
            if len(buckets) == 0: startpoint = min(unique_values)
            # Use the next unique value as the startpoint
            else: 
                idx = np.searchsorted(unique_values,[bins[i],],side='right')[0]
                #print("index",idx)
                startpoint = unique_values[idx]
                #print("unique_values",unique_values)
                #print("startpoint",startpoint)
            endpoint = bins[i+1]
            count = 0
            for value in unique_values:
                if startpoint <= value and value <= endpoint: count += counter[value]
            buckets.append(Bucket(startpoint, endpoint, count, None))
        return buckets

    def _create_value_odict(self, values:List) -> odict:
        """
        Create an ordered dictionary of values and their counts.
        """
        value_dict = {}
        for bucket in self.buckets:
            value_dict[bucket.startpoint] = bucket.count
        return odict(sorted(value_dict.items()))
    
    def get_bucket_containing_q(self, q:float) -> Bucket:
        """
        Get the bucket that contains the value q.
        """
        for bucket in self.buckets:
            if bucket.value_in_bucket(q): return bucket
        return None
    
    def cal_sse(self, values:odict) -> float:
        """
        Calculate the sum of squared errors (SSE) for the given values.
        :param values: An ordered dictionary of values and their counts.
        """
        sse = 0
        dict_keys = list(values.keys())
        for q in range(self.start_value, self.end_value+1):
            # Get fp_q
            fp_q = 0
            if q in dict_keys: fp_q = values[q]
            # Get the bucket that contains q
            bucket = self.get_bucket_containing_q(q)
            # If the bucket is None, continue
            if not bucket: continue
            eb_q = bucket.count
            # Calculate the squared error
            sse += (fp_q - (eb_q / (bucket.endpoint - bucket.startpoint + 1)))**2
        return sse
    
    def cal_KLDiv(self, ref_bucket_list) -> float:
        """
        Calculate the Kullback-Leibler divergence (KL-divergence) for the given reference bucket list.
        :param ref_bucket_list: The reference bucket list.
        """
        start_time = time.time()
        kl_div = 0
        for q in range(self.start_value, self.end_value+1):
            # Get the bucket that contains q
            bucket = self.get_bucket_containing_q(q)
            ref_bucket = ref_bucket_list.get_bucket_containing_q(q)
            # If the bucket is None, continue
            if not bucket or not ref_bucket: continue
            p = ref_bucket.count / ref_bucket_list.total_count
            q = bucket.count / self.total_count
            kl_div += p * np.log(p / q)
        self.f_time.append((self.ID, 'cal_KLDiv', time.time() - start_time))
        return abs(kl_div)
    
    def cal_l2_norm(self, ref_bucket_list) -> float:
        """
        Calculate the L2 norm for the given reference bucket list.
        :param ref_bucket_list: The reference bucket list.
        """
        start_time = time.time()
        l2_norm = 0
        x, y = zero_pad_vectors(self.bins, ref_bucket_list.bins)
        l2_norm = np.linalg.norm(x - y)
        self.f_time.append((self.ID, 'cal_l2_norm', time.time() - start_time))
        return l2_norm
    
    def cal_gpt_distance(self, model_id:str, ref_bucket_list) -> float:
        """
        Calculate the GPT distance for the given model ID.
        :param model_id: The model ID.
        """
        start_time = time.time()
        gpt_distance = 0
        if self.gpt_measure:
            gpt_distance = get_gpt_score(ref_bucket_list.bins, self.bins, model_id=model_id, context=self.attribute)
        self.f_time.append((self.ID, 'cal_gpt_distance', time.time() - start_time))
        return gpt_distance


class Strategy:
    """
    Class for a strategy.
    """
    def __init__(self, buckets_list:List[List[Bucket]], statistical_score:float, alpha:float=0.5):
        self.buckets_list = buckets_list
        self.semantic_score = self._cal_semantic_score()
        self.statistical_score = statistical_score
        self.alpha = alpha
        #self.score = self._cal_score()
    
    def _cal_semantic_score(self) -> float:
        """
        Calculate the semantic score for the given list of bucket lists.
        """
        score = 0
        for i in range(len(self.buckets_list)):
            score += self.buckets_list[i].KLDiv
        return score
    
    def _cal_score(self) -> float:
        """
        Calculate the score for the given list of bucket lists.
        """
        return self.alpha * self.statistical_score + (1 - self.alpha) * self.semantic_score
    
    def __lt__(self, __value: object) -> bool:
        return self.score < __value.score
    
    def __le__(self, __value: object) -> bool:
        return self.score <= __value.score
    
    def __gt__(self, __value: object) -> bool:
        return self.score > __value.score
    
    def __ge__(self, __value: object) -> bool:
        return self.score >= __value.score


class PartitionSearchSpace:
    """
    Class for search space.
    """
    def __init__(self, candidates:List[Partition]=None, gpt_measure=False):
        self.candidates = candidates
        self.ID_count = 0
        self.gpt_measure = gpt_measure
        self.f_time = []

    def _generate_bins(self, candidates, raw_data, n_bins, attr, target, gold_standard, method) -> List[Partition]:
        """
        Generate bins for the given raw data.
        """
        if method == 'equal-width':
            start_time = time.time()
            bins = equal_width(raw_data, n_bins, [attr])[attr]
        elif method == 'equal-frequency':
            start_time = time.time()
            bins = equal_frequency(raw_data, n_bins, [attr])[attr]
        elif method == 'chi-merge':
            start_time = time.time()
            bins = chimerge_wrap(raw_data, [attr], target, n_bins)[attr]
        elif method == 'kbins':
            start_time = time.time()
            bins = KBinsDiscretizer_wrap(raw_data, [attr], n_bins)[attr]
        elif method == 'kbins-quantile':
            start_time = time.time()
            bins = KBinsDiscretizer_wrap(raw_data, [attr], n_bins, 'quantile')[attr]
        elif method == 'decision-tree':
            start_time = time.time()
            bins = DecisionTreeDiscretizer_wrap(raw_data, [attr], target, n_bins)[attr]
        elif method == 'kmeans':
            start_time = time.time()
            bins = KMeansDiscretizer_wrap(raw_data, [attr], n_bins)[attr]
        elif method == 'random-forest':
            start_time = time.time()
            bins = RandomForestDiscretizer_wrap(raw_data, [attr], target, n_bins)[attr]
        elif method == 'bayesian-blocks':
            start_time = time.time()
            bins = BayesianBlocksDiscretizer_wrap(raw_data, [attr])[attr]
        elif method == 'mdlp':
            start_time = time.time()
            bins = MDLPDiscretizer_wrap(raw_data, [attr], target)[attr]
        else: raise ValueError('Invalid method.')

        if bins is None: return candidates
        
        bins = sorted(list(set(bins)))
        binned_values = pd.cut(raw_data[attr], bins=bins, labels=bins[1:])
        partition = Partition(bins=bins, binned_values=binned_values, values=list(raw_data[attr]), method=method, ref_bucket_list=gold_standard, ID=self.ID_count, gpt_measure=self.gpt_measure)
        print(partition)
        partition.f_time.append((partition.ID, 'get_bins', time.time() - start_time))
        self.ID_count += 1
        candidates.append(partition)
        return candidates

    
    def prepare_candidates(self, raw_data, attr, target, min_bins:int, max_bins:int, gold_standard) -> None:
        if self.candidates is not None: 
            print("Candidates already exist.")
            return
        # Remove rows with missing values in the attribute and target columns
        raw_data = raw_data.copy()
        raw_data = raw_data.dropna(subset=[attr, target])
        
        canditate_partitions = []
        # Add gold standard
        gold_standard.ID = self.ID_count
        self.ID_count += 1
        canditate_partitions.append(gold_standard)

        for n_bins in range(min_bins, max_bins+1):
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'equal-width')
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'equal-frequency')
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'chi-merge')
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'kbins')
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'kbins-quantile')
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'decision-tree')
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'kmeans')
            canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'random-forest')
        canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'bayesian-blocks')
        canditate_partitions = self._generate_bins(canditate_partitions, raw_data, n_bins, attr, target, gold_standard, 'mdlp')

        print(f"Number of partitions: {len(canditate_partitions)}")
        self.candidates = canditate_partitions
    
    def standardize_semantics(self) -> None:
        if self.candidates is None: 
            print("Candidates do not exist.")
            return
        # Standardize the candidates
        self._standardize_KLDiv()
        self._standardize_l2_norm()
    
    def gold_standard_init_runtime(self) -> None:
        """
        Initialize the runtime for the gold standard.
        """
        for candidate in self.candidates:
            if candidate.method == 'gold-standard':
                candidate.f_time.append((candidate.ID, 'cal_KLDiv', 0))
                candidate.f_time.append((candidate.ID, 'cal_l2_norm', 0))
                candidate.f_time.append((candidate.ID, 'cal_gpt_distance', 0))
                candidate.f_time.append((candidate.ID, 'get_bins', 0))

    def get_runtime(self):
        """
        Get the runtime for each partition.
        """
        self.f_time = [] #refresh
        self.gold_standard_init_runtime()
        for candidate in self.candidates:
            self.f_time.extend(candidate.f_time)
        runtime_df = pd.DataFrame(self.f_time, columns=['ID', 'function', 'runtime'])
        return runtime_df
    
    def _standardize_KLDiv(self) -> None:
        """
        Standardize the candidates.
        """
        # Get the maximum and minimum KL-divergence scores
        max_KLDiv = max([candidate.KLDiv for candidate in self.candidates])
        min_KLDiv = min([candidate.KLDiv for candidate in self.candidates])
        # Standardize the KL-divergence scores
        for candidate in self.candidates:
            candidate.KLDiv = (candidate.KLDiv - min_KLDiv) / (max_KLDiv - min_KLDiv)
            candidate.KLDiv = 1-candidate.KLDiv

    def _standardize_l2_norm(self) -> None:
        """
        Standardize the candidates.
        """
        # Get the maximum and minimum L2 norm scores
        max_l2_norm = max([candidate.l2_norm for candidate in self.candidates])
        min_l2_norm = min([candidate.l2_norm for candidate in self.candidates])
        # Standardize the L2 norm scores
        for candidate in self.candidates:
            candidate.l2_norm = (candidate.l2_norm - min_l2_norm) / (max_l2_norm - min_l2_norm)
            candidate.l2_norm = 1-candidate.l2_norm
    
    def _standardize_gpt_distance(self) -> None:
        """
        Standardize the candidates.
        """
        # Get the maximum and minimum GPT distance scores
        max_gpt_distance = max([candidate.gpt_distance for candidate in self.candidates])
        min_gpt_distance = min([candidate.gpt_distance for candidate in self.candidates])
        # Standardize the GPT distance scores
        for candidate in self.candidates:
            candidate.gpt_distance = (candidate.gpt_distance - min_gpt_distance) / (max_gpt_distance - min_gpt_distance)
            candidate.gpt_distance = 1-candidate.gpt_distance

if __name__ == '__main__':
    # Test Bucket class
    b1 = Bucket(0, 0, 4, 'A')
    b2 = Bucket(10, 50, 9, 'B')
    b3 = Bucket(0, 50, 13, 'C')
    print(b1)
    print(b2)
    print(b3)
    print(b1 == b2)
    print(b1 < b2)
    print(b1 < b3)
    print(b1 > b2)
    print(b1 > b3)
    print(b1.value_in_bucket(2))
    print(b1.value_in_bucket(3))
    print(b1.value_in_bucket(4))
    print(b1.value_in_bucket(5))

    d = {0:4,10:2,20:2,30:2,40:2,50:1}
    values = [0,20,30,20,30,40,10,40,50,0,0,0]
    print(sorted(d.items()))
    od = odict(sorted(d.items()))
    print(list(od.keys())[0])

    bls = Partition(bins=[0, 10, 50], values=values, buckets=[b1, b2])
    print(bls.cal_sse(od))

    bls = Partition(bins=[0,50], values=values, buckets=[b3])
    print(bls.cal_sse(od))

    values = [0, 0, 0, 102, 102, 102, 102, 102, 140, 141, 151, 200]
    glucose_gpt = Partition(bins=[-1, 140, 200], values=values)
    print(glucose_gpt.buckets)
    #print(glucose_gpt.value_odict)

    glucose0 = Partition(bins=[-1, 100, 200], values=values)
    print(glucose0.buckets)
    print("KL-divergence:",glucose0.cal_KLDiv(glucose_gpt))

    glucose1 = Partition(bins=[-1, 150, 200], values=values)
    print(glucose1.buckets)
    print("KL-divergence:",glucose1.cal_KLDiv(glucose_gpt))

    bls = Partition(bins=[0, 10, 50], values=values, buckets=[b1, b2])
    print(bls.cal_KLDiv(bls))

    gold_standard = Partition(bins=[0, 50, 100, 200], values=values, method='gold-standard', gold_standard=True)
    bl1 = Partition(bins=[0, 50, 100, 150, 200], values=values, method='equal-width', ref_bucket_list=gold_standard)
    print(bl1.KLDiv)
    print(bl1.l2_norm)
    print(bl1.gpt_distance)