import pandas as pd 
import numpy as np 
import streamlit as st
import text
import base64
from ClusteringNLP import Clustering_NLP
from ClassificationNLP import Classification_NLP
from ClusterAndClassify import Cluster_and_Classify
from sklearn.metrics import accuracy_score
import charts
import tools

files = tools.get_files()

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
            
def clustering():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Unsupervised Learning | Clustering**")
    
    st.markdown("### Choose a Dataset")
    option = st.selectbox(
        'Select a Teacher Answer',
        files['name']
    )
    
    index = files['name'].index(option)
    st.write('You selected:', index)

    doc= files['doc'][index] 
    data= files['data'][index]
    
    st.markdown("## Modeling Example")
    st.markdown('**Select the below based on what you want to see:**')
    chart_flag = st.checkbox('Display Reduced-Dimensionality Chart')
    model_flag = st.checkbox('Display Model Info and Performance')    
        
    nlp = Clustering_NLP(data, doc)
    nlp.correct_cluster_labels()
    
    if chart_flag:
        col1, col2 = st.beta_columns(2)
        fig1, ax1 = charts.plot_pca_chart(data, doc['label'], nlp.model.cluster_centers_)
        fig2, ax2 = charts.plot_tsne_chart(data, doc['label'], nlp.model.cluster_centers_)
        with col1:
            st.pyplot(fig = fig1)
            
        with col2:
            st.pyplot(fig = fig2)

    if model_flag:
        st.markdown('## Model Data Metrics') 
        scores = {}
        scores['Accuracy'] = round(nlp.accuracy(),3)
        scores['Balanced Accuract'] = round(nlp.balanced_accuracy(), 3)
        scores['F1 Score'] = round(nlp.f1_scorer(), 3)
        scores['Cohens Kappa'] = round(nlp.kappa(), 3)
        scores = pd.DataFrame.from_dict(scores, orient = 'index').transpose()
        scores.rename(index = {0:"Scores"}, inplace = True)
        st.table(scores)
        
        
        st.pyplot(fig = charts.plot_confusion_matrix(nlp.doc['label'], nlp.doc.cluster))
        
        
        results = doc[['student_answer', 'label', 'cluster']]
        try_it = st.checkbox('Try it Yourself!')
        explore_flag = st.checkbox('Explore Data')
        if try_it:
            tryit(nlp, 'cluster')
                
            
        if explore_flag:
            st.markdown(f"""**Dataset Length: {len(results)}** """)
            start, end = st.slider(
                label = 'Data View Select',
                min_value = 0,
                max_value = len(nlp.doc)-1,
                value = (0,5)
            )
            st.markdown(f"""**Teacher Answer: {nlp.doc['teacher_answer'].values[0]}**""")
            for i in range(int(start),int(end)+1):
                st.markdown(f"""{i}. {'Label:':>10} {str(nlp.doc.loc[i,'label'])}  Pred: {str(nlp.doc.loc[i,'cluster'])}    {str(nlp.doc.loc[i,'student_answer'])}""")
                

def classification():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Supervised Learning | Classification**")
    
    st.markdown("### Choose a Dataset")
#     files = tools.get_files()
    option = st.selectbox(
        'Select a Teacher Answer',
        files['name']
    )
    
    index = files['name'].index(option)
    st.write('You selected:', index)

    doc= files['doc'][index] 
    data= files['data'][index]

   
    st.markdown("## Modeling Example")
    st.markdown('**Select the below based on what you want to see:**')
    chart_flag = st.checkbox('Display Logistic Function Chart')
    model_flag = st.checkbox('Display Model Info and Performance')
    
    
    test_size = st.number_input(
            'Test Size',
            min_value = 0.01,
            max_value = 0.91,
            value = .75,
            step = 0.05
            )
    nlp = Classification_NLP(data, doc, test_size)
    st.markdown(f'Training Set Size: {len(nlp.X_train)}')
    st.markdown(f'Test Set Size: {len(nlp.X_test)}')
    
    if chart_flag:
        st.pyplot(fig = charts.plot_logistic_function(nlp, test_size))
    if model_flag:
        st.markdown('## Model Data Metrics')
        scores = {}
        _, accuracy =  nlp.accuracy()
        scores['Accuracy'] = round(accuracy,3)
        scores['Balanced Accuract'] = round(nlp.balanced_accuracy(), 3)
        scores['F1 Score'] = round(nlp.f1_scorer(), 3)
        scores['Cohens Kappa'] = round(nlp.kappa(), 3)
        scores = pd.DataFrame.from_dict(scores, orient = 'index').transpose()
        scores.rename(index = {0:"Scores"}, inplace = True)
        st.table(scores)
        
        st.pyplot(fig = charts.plot_confusion_matrix(nlp.doc.label, nlp.doc.prediction))
        
        
        results = doc[['student_answer', 'label', 'prediction']]
        try_it = st.checkbox('Try it Yourself!')
        explore_flag = st.checkbox('Explore Data')
        
        if try_it:
            tryit(nlp, 'prediction')
        if explore_flag:
            st.markdown(f"""**Dataset Length: {len(results)}** """)
            start, end = st.slider(
                label = 'Data View Select',
                min_value = 0,
                max_value = len(nlp.doc)-1,
                value = (0,5)
            )
            st.markdown(f"""**Teacher Answer: {nlp.doc['teacher_answer'].values[0]}**""")
            for i in range(int(start),int(end)+1):
                if nlp.doc.loc[i,'label'] != nlp.doc.loc[i,'prediction']:
                    st.markdown(f"""{i}. {'Label:':>10} {str(nlp.doc.loc[i,'label'])}  Pred: {str(nlp.doc.loc[i,'prediction'])}    {str(nlp.doc.loc[i,'student_answer'])}""")
                else:
                    st.markdown(f"""{i}. {'Label:':>10} {str(nlp.doc.loc[i,'label'])}  Pred: {str(nlp.doc.loc[i,'prediction'])}    {str(nlp.doc.loc[i,'student_answer'])}""")     
                    
                    
def combo():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Unsupervised Learning | Combination**")
    
    st.markdown("### Choose a Dataset")
#     files = tools.get_files()
    option = st.selectbox(
        'Select a Teacher Answer',
        files['name']
    )
    
    index = files['name'].index(option)
    st.write('You selected:', index)

    doc= files['doc'][index] 
    data= files['data'][index]
    
    st.markdown("## Modeling Example")
    st.markdown('**Select the below based on what you want to see:**')
    chart_flag = st.checkbox('Display Reduced-Dimensionality Chart')
    model_flag = st.checkbox('Display Model Info and Performance')    
    
    nlp = 0
    
    nlp = Cluster_and_Classify(data, doc, len(data)-1)
    nlp.run()
    
    if chart_flag:
        col1, col2 = st.beta_columns(2)
        fig1, ax1 = charts.plot_pca_chart(nlp.X_train, nlp.y_train, data.iloc[-1,:].values.reshape(1,-1))
        fig2, ax2 = charts.plot_tsne_chart(nlp.X_train, nlp.y_train, data.iloc[-1,:].values.reshape(1,-1), num_clusters = 1)
        with col1:
            st.pyplot(fig = fig1)
            
        with col2:
            st.pyplot(fig = fig2)

    if model_flag:
        st.markdown('## Model Data Metrics')

        scores = {}
        scores['Accuracy'] = round(nlp.accuracy(),3)
        scores['Balanced Accuracy'] = round(nlp.balanced_accuracy(), 3)
        scores['F1 Score'] = round(nlp.f1_scorer(), 3)
        scores['Cohens Kappa'] = round(nlp.kappa(), 3)
        scores = pd.DataFrame.from_dict(scores, orient = 'index').transpose()
        scores.rename(index = {0:"Scores"}, inplace = True)
        st.table(scores)

        
        st.pyplot(fig = charts.plot_confusion_matrix(nlp.y_true, nlp.y_pred))
        
        doc['prediction'] = nlp.y_pred
        results = doc[['student_answer', 'label', 'prediction']]
        try_it = st.checkbox('Try it Yourself!')
        explore_flag = st.checkbox('Explore Data')
        if try_it:
            tryit(nlp, 'prediction')
                
        if explore_flag:
            st.markdown(f"""**Dataset Length: {len(results)}** """)
            start, end = st.slider(
                label = 'Data View Select',
                min_value = 0,
                max_value = len(nlp.doc)-1,
                value = (0,5)
            )
            st.markdown(f"""**Teacher Answer: {nlp.doc['teacher_answer'].values[0]}**""")
            for i in range(int(start),int(end)+1):
                st.markdown(f"""{i}. {'Label:':>10} {str(doc.loc[i,'label'])}  Pred: {str(doc.loc[i,'prediction'])}    {str(doc.loc[i,'student_answer'])}""")
                
def tryit(nlp, pred_type):
    st.markdown(f"""**Teacher Answer: {nlp.doc['teacher_answer'].values[0]}**""")
    answer = st.text_input("Submit Your Own Answer")
    if answer != '':
        nlp.create_features(answer)
        """
        Weird issue where a random dataset gets injected into it
        maybe an object or caching issue?
        """
        nlp.score_new_sentences(nlp.new_answers.loc[:, 'wordcount':])
        prediction = nlp.new_answers.loc[0,pred_type]
        pred = ''
        if prediction == 1:
            pred = 'correct'
        else:
            pred = "incorrect"
        
        if pred != '':
            st.markdown(f'## **Your answer is predicted {pred}**')
        
        st.markdown('----')
        st.markdown('Breakdown of your answer:')
        st.write(nlp.new_answers.rename({0:"Your Answer"}))

        
def application():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Autograders**")
   
    st.markdown('## Choose your model')
    
    
    
    option = st.selectbox(
        'Model Type',
        ['Clustering', 'Classification', 'Combination']
    )
#     answer_row = st.number_input('Which row is your answer on?', value = 0, step = 1, format= '%i')
#     test_size = 0    
#     if option == 'Classification':
#         test_size = st.number_input('Number of Labeled Questions', value = 0, step = 1, format = '%i')

    st.markdown('### **Please fill out all the above first before selecting your file**')
    file = st.file_uploader("Upload your csv file", type='csv')
        
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a csv file")
        return
    else:
        file.seek(0)
        doc, data = tools.generate_features(file)
        predictions = []
        if option == 'Clustering':
            nlp = Clustering_NLP(data, doc)
            nlp.correct_cluster_labels()
            st.write(nlp.doc)
            predictions = nlp.doc['cluster']

        if option == 'Classification':
            nlp = Classification_NLP(data, doc)
            predictions = nlp.model.predict(data)
        if option == 'Combination':
            nlp = Cluster_and_Classify(data, doc, len(data)-1)
            nlp.run()
            predictions = nlp.y_pred

        doc['prediction'] = predictions
        doc = doc[['student_answer', 'prediction']]
        st.markdown('## **Model Results**')
        st.table(doc)
        csv = doc.to_csv(index = False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
        st.markdown(href, unsafe_allow_html=True)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        