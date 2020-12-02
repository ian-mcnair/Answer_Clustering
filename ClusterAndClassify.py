import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score
class Cluster_and_Classify:
    def __init__(self, df, doc, answer_row = -1):
        self.doc = doc.copy()
        self.df = df.copy()
        self.clf_df = df.copy()
        self.answer_row = answer_row
        self.model = KMeans(n_clusters = 2)
        self.clf = LogisticRegression()
        self.closest = []
        self.furthest = []
        self.X = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = np.array([])
        self.X_test = pd.DataFrame()
        self.y_pred = []
        self.y_true = []
#         self.accuracy = -1
        self.cm = []
    
    def cluster(self):
        self.model = self.model.fit(self.df)
        self.df['clusters'] = self.model.labels_
        self.correct_cluster_labels(self.answer_row)
        
    def find_train_set_idxs(self):
        i = 0
        self.df['distances'] = 0
        for value in self.df.iloc[self.answer_row, :].values[:-1]:
            self.df['distances'] = self.df['distances'] + ((self.df.iloc[:, i] - value) ** 2)
            i+=1
        self.df['distances'] = self.df['distances'] ** 0.5
        self.df.loc[int(self.answer_row),'distances'] = -1
        self.closest = self.df[(self.df.clusters == 1) & (self.df.distances > 0)].nsmallest(2, 'distances').index.values.tolist()
        self.closest.append(self.answer_row)
        self.furthest = self.df[(self.df.clusters == 0) & (self.df.distances > 0)].nlargest(2, 'distances').index.values.tolist()
    
    def correct_cluster_labels(self, answer_row):
        """
        Assumes last row is teachers answer
        """
        cluster_label = self.df.iloc[answer_row, -1]
        if 1 != cluster_label:
            self.df['clusters'] = (self.df['clusters'] - 1)**2
            
    def create_train_test_sets(self):
        X_correct = self.clf_df[self.clf_df.index.isin(self.closest)].copy()
        X_correct['label'] = 1
        X_incorrect = self.clf_df[self.clf_df.index.isin(self.furthest)].copy()
        X_incorrect['label'] = 0
        self.X = pd.concat([X_correct, X_incorrect])
        self.X_train = self.X.iloc[:,:-1]
        self.y_train = self.X.iloc[:,-1]
        self.X_test = self.clf_df[~self.clf_df.index.isin(self.closest + self.furthest)].copy()
    
    def classify(self):
        self.clf = self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.clf_df)
        self.y_true = self.doc['label'].values
        return self.y_pred, self.y_true
    
    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)
    
    def f1_scorer(self):
        return f1_score(self.y_true, self.y_pred)
    
    def balanced_accuracy(self):
        return balanced_accuracy_score(self.y_true, self.y_pred, adjusted = True)
    
    def precision(self):
        return precision_score(self.y_true, self.y_pred)
    
    def recall(self):
        return recall_score(self.y_true, self.y_pred)
    
    def kappa(self, weighting):
        return cohen_kappa_score(self.y_true, self.y_pred, weights = weighting)
    
    def confusion_mtx(self):
        return confusion_matrix(self.y_true, self.y_pred)
    
    def run(self):
        self.cluster()
        self.find_train_set_idxs()
        self.create_train_test_sets()
        self.classify()
#         return self.accuracy()
#         self.confusion_mtx()
