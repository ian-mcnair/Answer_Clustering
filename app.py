import pandas as pd 
import numpy as np 
import streamlit as st # pylint: disable=import-error
import text
# import charts
import views

def run():
    st.sidebar.markdown("# Navigation")
    pageview = st.sidebar.radio('Topic Select',(
        'Project Introduction',
        'Timeline',
        'Feature Engineering',
        'Clustering',
        'Classification',
        'Conclusion',
        'Appendix',
        ),
        index = 0
    )

    if pageview == 'Project Introduction':
        views.landing()
    elif pageview == 'Timeline':
        views.timeline()
    elif pageview == 'Feature Engineering':
        views.feature_engineering()
    elif pageview == 'Clustering':
        views.clustering()
    elif pageview == 'Classification':
        views.classification()
    elif pageview == 'Conclusion':
        views.conclusion()
    elif pageview == "Appendix":
        views.appendix()
    


if __name__ == "__main__":
     run()