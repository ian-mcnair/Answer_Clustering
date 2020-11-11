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
sns.set_style("darkgrid")
sns.set_context("talk")


def plot_confusion_matrix(true, pred):
    fig, cm = plt.subplots()
    cm = sns.heatmap(
        confusion_matrix(true, pred ),
        annot = True, 
        cbar = False,
        cmap = 'Blues'
    )
    cm.set_xlabel('Predicted', fontsize = 15)
    cm.set_ylabel('Actual', fontsize = 15)
    cm.set_title('Confusion Matrix', fontsize = 20)
    cm.set_yticklabels( labels = ['Wrong', 'Right'], rotation = 0, va = 'center')
    cm.set_xticklabels(labels = ['Wrong', 'Right'], ha = 'center')
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
        hue= labels,
    )
   
    ax.set_title("PCA Cluster Plot")
    
    return fig, ax

def plot_tsne_chart(data, labels, cluster_centers):
    tsne = TSNE(n_components = 2,perplexity = 5)
    data = tsne.fit_transform(data.append(pd.DataFrame(cluster_centers, columns = data.columns)))
    cluster_centers = data[-2:,:].copy()
    data = data[:-2,:].copy()
                              
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
        hue= labels,
    )
    
    ax.set_title("t-SNE Visualization")
    
    return fig, ax
    
    
    
    
    
    
    