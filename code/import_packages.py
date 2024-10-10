import pandas as pd
import numpy as np
import sys
import os
import json
import openai
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import graphviz
import itertools
import datetime
from collections import OrderedDict as odict
from collections import Counter

#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
# import Kmeans from sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error
from typing import List, Union, Any, Tuple, Dict
import time
from permetrics.regression import RegressionMetric
import oapackage
from scipy.stats import wasserstein_distance, binned_statistic, spearmanr, f_oneway
from scipy.spatial import cKDTree
import hdbscan
from astropy.stats import bayesian_blocks
from optbinning import MDLP


import warnings
warnings.filterwarnings('ignore')

# Project path
ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'code'))