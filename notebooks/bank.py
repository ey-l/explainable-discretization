import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import graphviz
import itertools
#from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Project path
ppath = os.path.join(sys.path[0], '..//')

df = pd.read_csv(os.path.join(ppath, 'data', 'bank', 'train.csv'))
df.drop(['Surname'], axis=1, inplace=True)
df.drop(['Geography'], axis=1, inplace=True)
df.drop(['Gender'], axis=1, inplace=True)
df.drop(['id'], axis=1, inplace=True)
df.drop(['CustomerId'], axis=1, inplace=True)

# Select all rows and all columns except the last one
X = df.iloc[:, :-1]
# Select all rows and only the last column
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

cols = ['Balance', 'Age', 'NumOfProducts']
binning_strategies = {
    'Balance': [
        [-1, 0, 100000, 200000, 1000000],
        [-1, 0, 1000000],
        ],
    'NumOfProducts': [
        [-1, 1, 1000],
        [-1, 0, 1, 2, 3, 1000],
        [-1, 2, 1000],
        [-1, 0, 1000],
        ],
    'Age': [
        [-1, 0, 1000],
        [-1, 25, 40, 55, 75, 100],
        [-1, 20, 30, 40, 50, 100],
        [-1, 25, 50, 75, 1000]
        ],
}
strategy_combos = list(itertools.product(*binning_strategies.values()))

results = []
for strategy in strategy_combos:
    # load data
    df = pd.read_csv(os.path.join(ppath, 'data', 'bank', 'train.csv'))
    # bin fixed columns

    # bin variable columns
    for i in range(len(cols)):
        col = cols[i]
        bins = strategy[i]
        df[col + '.binned'] = pd.cut(df[col], bins=bins, labels=bins[1:])
        df[col + '.binned'] = df[col + '.binned'].astype('float64')
    # split data
    X = df[['CreditScore', 'Age.binned', 'Tenure', 'Balance.binned',
       'NumOfProducts.binned', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    clf = DecisionTreeClassifier(random_state=0,max_depth=3).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print("Strategy:", strategy)
    print("Accuracy:", accuracy)
    #print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    #print("Classification report:\n", classification_report(y_test, y_pred))
    results.append(accuracy)