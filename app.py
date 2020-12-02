import pandas as pd 
import numpy as np 
import streamlit as st 
import text
# import charts
import views

def run():
    st.sidebar.markdown("# Navigation")
    pageview = st.sidebar.radio('Topic Select',(
        'Project Introduction',
        'Data Exploration',
        'Clustering',
        'Classification',
        'Combination'
        ),
        index = 0
    )

    if pageview == 'Project Introduction':
        views.landing()
    elif pageview == 'Data Exploration':
        views.data_exploration()
    elif pageview == 'Clustering':
        views.clustering()
    elif pageview == 'Classification':
        views.classification()
    elif pageview == 'Combination':
        views.combo()


if __name__ == "__main__":
     run()