import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tools
import streamlit as st 
from sklearn.metrics import accuracy_score, confusion_matrix


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

