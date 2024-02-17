import numpy as np
import sys
import os
from typing import List, Union, Any, Tuple, Dict
from collections import OrderedDict as odict
from collections import Counter

# Project path
ppath = sys.path[0] + '/../'

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
    

class BucketList:
    """
    Class for a list of buckets
    """
    def __init__(self, bins, values:List, buckets:List[Bucket]=None):
        # Bins to apply discretization to data
        self.bins = bins
        # Buckets to calculate errors
        if buckets is not None: self.buckets = buckets
        elif values is not None: 
            self.buckets = self._create_buckets(bins, values)
            self.value_odict = self._create_value_odict(values)
        else: raise ValueError('Either buckets or values must be provided.')
        
        self.start_value = sorted(values)[0]
        self.end_value = sorted(values)[-1]

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
        kl_div = 0
        # Get the shorter bucket list
        shorter_length = 0
        if len(self.buckets) < len(ref_bucket_list.buckets): shorter_length = len(self.buckets)
        else: shorter_length = len(ref_bucket_list.buckets)
        # Calculate the KL-divergence
        for i in range(shorter_length):
            p = self.buckets[i].count
            q = ref_bucket_list.buckets[i].count
            if p == 0 or q == 0: continue
            kl_div += p * np.log(p / q)
        return kl_div


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

    bls = BucketList(bins=[0, 10, 50], values=values, buckets=[b1, b2])
    print(bls.cal_sse(od))

    bls = BucketList(bins=[0,50], values=values, buckets=[b3])
    print(bls.cal_sse(od))

    values = [0, 0, 0, 102, 102, 102, 102, 102, 140, 141, 151, 200]
    glucose_gpt = BucketList(bins=[-1, 140, 200], values=values)
    print(glucose_gpt.buckets)
    print(glucose_gpt.value_odict)

    glucose0 = BucketList(bins=[-1, 100, 200], values=values)
    print(glucose0.buckets)
    print(glucose0.cal_sse(glucose_gpt.value_odict))
    print("KL-divergence:",glucose0.cal_KLDiv(glucose_gpt))

    glucose1 = BucketList(bins=[-1, 150, 200], values=values)
    print(glucose1.buckets)
    print(glucose1.cal_sse(glucose_gpt.value_odict))
    print("KL-divergence:",glucose1.cal_KLDiv(glucose_gpt))

    bls = BucketList(bins=[0, 10, 50], values=values, buckets=[b1, b2])
    print(bls.cal_KLDiv(bls))