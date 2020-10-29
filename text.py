import base64
import streamlit as st

def timeline_start():
    file_ = open("./resources/grading_sleep.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown("""
    Although generally stated in the introduction, this project came from my immense hatred of grading.  

    **Grading is:**  
    - Tedious
    - Takes forever
    - Usually done at home because there isn't enough time during the day
    - Sloppy (Handwriting)
    - Difficult to do well
    - Not completely accurate
    - Not always consistent
    """)

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">  ',
        unsafe_allow_html=True,
    )

    st.markdown("""
    
    ### My question simply was "Can I make grading easier, even if just a bit? " 
    
    ## Not the one to over achieve, I decided to set out on trying to autograde the next easiest thing to multiple choice, **short answers.**
    """)

def timeline_data():
    st.markdown("""
    Data was generally easy to find. Specifically, this paper/github repo...  
    
    https://github.com/patilpranjal/Auto-Grader-for-short-answer-Question-using-Siamese-Neural-Network/blob/master/Dataset.csv  
    
    ... ended up having many different types of questions (most science related), which were also labeled. Since all of the data needed was contained in the student's raw answer, not much data cleaning was needed. The goal is to be able to simply put in the teacher's answer, the students answer, and get a result.
    
    *This paper also details using a Bi-directional LSTM, which is not the direction I decided to follow.*
    
    """)
    
def timeline_feature_engineering():
    st.markdown(""" 
    Probably more important than even the model used to autograde, feature engineering is the process of using domain knowledge to extract features from raw data (wikipedia). Coming up with a particluarly good feature can mean all the difference in the end performance of the model. Bad features however can add noise, making predictions less accurate.  
      
    **This process was the most time consuming of the entire project. Even now, there are features I didn't develop which could greatly improve performace simply because of time.**
    """)
    
    
def timeline_modelling():
    st.markdown("""
    The models used for this study were some of the most simple,** K-Means Clustering** for unsupervised grouping and **Logistic Regression** for supervised classification.  
    
    
    ## K-Means Clustering
    Although more information on K-Means is provided in the index, the summary is that each observation is grouped to its closest centroid. Centroids represent a "center of mass" of data points. Although the rules for what metric to use when finding the centroid vary, the overall goal for K-Means is to minimize the average distances of points to their centroid. In this context, the teacher's answer is also provided in finding the ideal centroid points. The clsuter assigned to teh teachers answer becomes the "correct" cluste/groupr while answers belonging to the otehr cluster are labeled incorrect.
    
    ## Logistic Regression
    Logistic Regression is a tried and true method for supervised classification which utlizes the sigmoid function. During the training process, predictions are squished to either 0 or 1, which in this usecase is whether or not an answer is correct. The training process tunes the coeffcients of the model to the specific question.
    
    """)
    
def timeline_tuning():
    pass

def timeline_application():
    pass