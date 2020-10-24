import pandas as pd 
import numpy as np 
import streamlit as st

@st.cache
def read_data(filename, dtype = False):
    if dtype:
        return pd.read_csv(filename, dtype = object)
    else:        
        return pd.read_csv(filename)