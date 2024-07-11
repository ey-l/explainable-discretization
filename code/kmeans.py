import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math
from scipy.stats import wasserstein_distance, binned_statistic
import sys
import os
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score

ppath = sys.path[0] + '/../'
sys.path.append(os.path.join(ppath, 'scratch'))

class K_Means:
    
    def __init__(self, k=2, tolerance = 0.001, max_iter = 500):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tolerance
    
    def euclidean_distance(self, point1, point2):
        #return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 + (point1[2]-point2[2])**2)   #sqrt((x1-x2)^2 + (y1-y2)^2)
        return np.linalg.norm(point1-point2, axis=0)

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
    
    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
        
        
        for i in range(self.max_iterations):
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []
                
            for point in data:
                distances = []
                for index in self.centroids:
                    #distances.append(self.euclidean_distance(point,self.centroids[index]))
                    distances.append(wasserstein_distance(point, self.centroids[index]))
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)
            
            previous = dict(self.centroids)
            for cluster_index in self.classes:
                print(self.classes[cluster_index])
                self.centroids[cluster_index] = np.average(self.classes[cluster_index], axis = 0)
            

                
            isOptimal = True
            
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid)/original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break


                
def main():
    K=3
    # read csv file
    df = pd.read_csv(os.path.join(ppath, 'data', 'uciml_pima-indians-diabetes-database', 'diabetes.csv'))
    import json
    f = open(os.path.join(ppath, 'scratch/age_partitions.json'), "r")
    data = json.load(f)
    f.close()
    binnings = list(eval(data))
    col = 'Age'
    outs = []
    age = list(df['Age'])
    N = len(age)

    data = df.copy()
    data['Age.gt'] = data['Age']
    df_nan = df.sample(frac=0.3, random_state=42)
    data.loc[df.index.isin(df_nan.index),'Age'] = np.nan

    for i in binnings:
        bins = i['bins']
        data_i = data.copy()
        data_i[col + '.binned'] = pd.cut(data_i[col], bins=bins, labels=bins[1:])
        #data_i[col + '.binned'] = data_i[col + '.binned'].astype('float64')

        imputer = KNNImputer(n_neighbors=len(bins)-1)
        data_imputed = imputer.fit_transform(data_i[col + '.binned'].values.reshape(-1, 1))
        data_imputed = np.round(data_imputed)
        data_i['Age.imputed'] = data_imputed
        data_i[col + '.final'] = pd.cut(data_i[col+'.imputed'], bins=bins, labels=bins[1:])
        data_i[col + '.final'] = data_i[col + '.final'].astype('float64')

        if len(data_i[data_i[col + '.final'].isnull()]) > 200:
            print(f"Skipping {bins}")
            continue
        #data_i['Age.final'] = data_i['Age.final'].fillna(-1)
        value_final = np.array(data_i['Age.final'].values)
        value_final[np.isnan(value_final)] = -1

        # Evaluate data imputation
        data_i['Age.gt'] = pd.cut(data_i['Age.gt'], bins=bins, labels=bins[1:])
        data_i['Age.gt'] = data_i['Age.gt'].astype('float64')
        value_gt = np.array(data_i['Age.gt'].values)
        value_gt[np.isnan(value_gt)] = -1
        #data_i['Age.gt'] = data_i['Age.gt'].fillna(-1)
        #data_i = data_i.dropna(subset=['Age.final', 'Age.gt'])
        impute_accuracy = accuracy_score(value_gt, value_final)

        #print(f"{bins}:", 1-i['gpt'], impute_accuracy)

        hist, bin_edges = np.histogram(age, bins=bins)
        distribution = hist / N

        outs.append({'bins': bins, 'distribution': distribution, 'gpt': 1-i['gpt'], 'impute_accuracy': impute_accuracy})

    df = pd.DataFrame(outs)
    # order by impute accuracy
    df = df.sort_values(by='impute_accuracy', ascending=False)
    df['cluster'] = pd.cut(df['impute_accuracy'], bins=[0, 0.7, 0.75, 0.8, 0.85, 0.9], labels=[0, 1, 2, 3, 4])
    df = df[['bins', 'distribution', 'cluster']]

    alist = list(df['distribution'].values)
    data = np.empty(len(alist), dtype=object)
    data[:] = alist

    k_means = K_Means(K)
    k_means.fit(data)
    
    
    # Plotting starts here
    colors = 10*["r", "g", "c", "b", "k"]

    for centroid in k_means.centroids:
        plt.scatter(k_means.centroids[centroid][0], k_means.centroids[centroid][1], s = 130, marker = "x")

    for cluster_index in k_means.classes:
        color = colors[cluster_index]
        for features in k_means.classes[cluster_index]:
            plt.scatter(features[0], features[1], color = color,s = 30)
    #plt.show()
    
if __name__ == "__main__":
    main()