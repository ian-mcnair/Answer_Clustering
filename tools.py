import pandas as pd 
import numpy as np 
import streamlit as st
import os

def get_files():
    directory = "data/feature_sets/"
    datasets = {'name': [], 'data': [], 'doc': []}
    for filename in os.listdir(directory):
        if filename[-4:] == '.csv':
            if filename[-8:-5] == 'dat':
                datasets['data'].append(pd.read_csv(directory + filename))
                
            else:
                df = pd.read_csv(directory + filename)
                datasets['doc'].append(df)
                datasets['name'].append(df.teacher_answer.values[0][0:45] + "...")
    return datasets