import numpy as np
import sys
import os
from typing import List, Union, Any, Tuple, Dict

# Project path
ppath = sys.path[0] + '/../'

class Bucket:
    """
    Class for discret
    """
    def __init__(self, startpoint:float, endpoint:float, count:int, label:Union[str, int]):
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
    

class BucketList:
    """
    Class for a list of buckets
    """
    def __init__(self, buckets:List[Bucket]):
        self.buckets = buckets
    
    def cal_sse(self, other:List[Bucket]) -> float:
        pass