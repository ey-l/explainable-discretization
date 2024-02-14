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
    def __init__(self, bins, values:List=None, buckets:List[Bucket]=None):
        # Bins to apply discretization to data
        self.bins = bins
        # Buckets to calculate errors
        if buckets is not None: self.buckets = buckets
        elif values is not None: self.buckets = self._create_buckets(bins, values)
        else: raise ValueError('Either buckets or values must be provided.')

    def _create_buckets(self, bins, values:List) -> List[Bucket]:
        """
        Create buckets from the bins and unique values.
        """
        buckets = []
        unique_values = np.array(list(set(values)))
        counter = Counter(values)
        for i in range(len(bins)-1):
            if len(buckets) == 0: startpoint = bins[i]
            # Use the next unique value as the startpoint
            else: 
                idx = np.searchsorted(unique_values,[bins[i],],side='right')[0]
                startpoint = unique_values[idx]
            endpoint = bins[i+1]
            count = 0
            for value in unique_values:
                if startpoint <= value and value <= endpoint: count += counter[value]
            buckets.append(Bucket(startpoint, endpoint, count, None))
        return buckets

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
        start_value = dict_keys[0]
        end_value = dict_keys[-1]
        for q in range(start_value, end_value+1):
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
    print(sorted(d.items()))
    od = odict(sorted(d.items()))
    print(list(od.keys())[0])

    bls = BucketList(bins=[0, 10, 50], buckets=[b1, b2])
    print(bls.cal_sse(od))

    bls = BucketList(bins=[0,50], buckets=[b3])
    print(bls.cal_sse(od))

    bls = BucketList(bins=[-1, 140, 200], values=[0, 0, 0, 140, 141, 200])
    print(bls.buckets)