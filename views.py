import pandas as pd 
import numpy as np 
import streamlit as st
import text
import base64
from ClusteringNLP import Clustering_NLP
from ClassificationNLP import Classification_NLP
from sklearn.metrics import accuracy_score
import charts
import tools

def landing():
    file_ = open("./resources/comp_teach.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    
    # Heading
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Automated Short Response Grading**")
    
    st.markdown(f'## Purpose')
    st.markdown(
        """ Grading has to be one of the most tedious, unfun aspects of being a teacher. Having talked to former collegues, most would be willing to even pay an external party to do grading for them. Personally as a teacher, I always felt I had a duty to my students to complete grading on time, but with all of the other preparation work, it can feel overwhelming.   
        
Recognizing that most problems of repatition are perfect for programs, I set out to see if I could create an application which would, to some extent, reduce the amount of grading a teacher had to do. Specifically, attempting to use unsupervised and supervised machine learning methods to address short answer response questions. """
    )
    
    button = st.button('Result of Teacher using Application')
    if button:
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
        st.markdown(" *Example of a teacher using my application*")
        st.balloons()
        

def data_exploration():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Data Exploration**")
    
    st.markdown("### Choose a Dataset")
    files = tools.get_files()
    option = st.selectbox(
        'Select a Teacher Answer',
        files['name']
    )
    
    index = files['name'].index(option)
    st.write('You selected file:', index)
    
    doc= files['doc'][index] 
    data= files['data'][index]
    
    button_doc = st.checkbox('Display Question Data')
    button_data = st.checkbox('Display Features')
    
    
    if button_doc:
        start, end = st.slider(
            label = 'Data View Select',
            min_value = 0,
            max_value = len(doc)-1,
            value = (0,2)
        )

        st.table(doc.iloc[start:end+1, :])
    
    if button_data:
        st.write(data)
        button_stat = st.checkbox('Display Descriptive Statistics')
        if button_stat:
            st.markdown('### Topdown of Feature Set')
            st.pyplot(fig = charts.plot_heatmap(data.transpose()))
            st.markdown('### Descriptive Statistics')
            st.dataframe(data.describe())
            info = data.dtypes.to_frame().reset_index()
            info.columns = ['Feature Name', 'Data Type']
            st.markdown('### Datatype Info')
            st.write(info)
