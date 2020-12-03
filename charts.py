import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tools
import streamlit as st 
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


sns.set_style("darkgrid")
sns.set_context("talk")


def plot_confusion_matrix(true, pred):
    fig, ax = plt.subplots()
    ax = sns.heatmap(
        confusion_matrix(true, pred ),
        annot = True, 
        cbar = False,
        cmap = 'Blues'
    )
    ax.set_xlabel('Predicted', fontsize = 15)
    ax.set_ylabel('Actual', fontsize = 15)
    ax.set_title('Confusion Matrix', fontsize = 20)
    ax.set_yticklabels( labels = ['Wrong', 'Right'], rotation = 0, va = 'center')
    ax.set_xticklabels(labels = ['Wrong', 'Right'], ha = 'center')
    return fig


def plot_pca_chart(data, labels, cluster_centers):
    pca = PCA(n_components = 2)
    comps = pca.fit_transform(data)
    cluster_centers = pca.transform(cluster_centers)

    fig, ax = plt.subplots()
    ax = sns.scatterplot(
        x = cluster_centers[:,0],
        y = cluster_centers[:,1],
        color  = 'black',
        s= 250,
        marker = '8'
    )
    ax = sns.scatterplot(
        x = comps[:,0],
        y = comps[:,1],
        hue = labels,
    )
   
    ax.set_title("PCA Cluster Plot")
    
    return fig, ax

def plot_tsne_chart(data, labels, cluster_centers, num_clusters = 2):
    tsne = TSNE(n_components = 2,perplexity = 5)
    data = tsne.fit_transform(data.append(pd.DataFrame(cluster_centers, columns = data.columns)))
    cluster_centers = data[-(num_clusters):,:].copy()
    data = data[:-(num_clusters),:].copy()
    fig, ax = plt.subplots()
    ax = sns.scatterplot(
        x = cluster_centers[:,0],
        y = cluster_centers[:,1],
        color  = 'black',
        s= 250,
        marker = '8'
    )

    ax = sns.scatterplot(
        x = data[:,0],
        y = data[:,1],
        hue = labels,
    )
    
    ax.set_title("t-SNE Visualization")
    
    return fig, ax
    
def plot_heatmap(df):
    fig, ax = plt.subplots(figsize = (6, 10))
    ax = sns.heatmap(df, yticklabels = True, linewidths = 1)
    return fig

def plot_logistic_function(nlp, test_size):
    pca = PCA(n_components = 1)
    comps = pca.fit_transform(nlp.data)
    X_train, X_test, y_train, y_test = train_test_split(comps, nlp.doc.label, test_size = test_size, stratify = nlp.doc.label, random_state = 42)
    clf = LogisticRegression().fit(X_train, y_train)
    pred = clf.predict(comps)
    clf.score(X_test, y_test)
    min_val = int(min(comps)) - 15
    max_val = int(max(comps)) + 15
    line = np.linspace(min_val, max_val, 1000)
    line_pred = clf.predict_proba(line.reshape(-1,1))

    
    fig, ax = plt.subplots()
    ax = sns.lineplot(
        x = line,
        y = line_pred[:,1],
    )
    ax = sns.scatterplot(
        x = comps.ravel(),
        y = pred,
        hue = nlp.doc.label, 
        palette = "deep"
    )

    ax.set_xlabel('PCA Reduced Feature')
    ax.set_ylabel('Prediction')
    ax.set_title('Logistic Function')
    return fig
   
    
    
    