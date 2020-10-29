import pandas as pd 
import numpy as np 
import streamlit as st
import text
import base64
from ClusteringNLP import Clustering_NLP
from sklearn.metrics import accuracy_score, confusion_matrix
# import charts
# import views
# import tools
# from geopy.geocoders import Nominatim

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
        
def timeline():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Project Timeline**")
    
    step = st.select_slider(
        '',
        options = ['Start', 'Data', 'Feature Engineering', 'Modeling', 'Tuning', 'Application', 'Finish'],
        value = ('Start')
    )
    
    if step == 'Start':
        text.timeline_start()
    elif step == 'Data':
        text.timeline_data()
    elif step == 'Feature Engineering':
        text.timeline_feature_engineering()
    elif step == 'Modeling':
        text.timeline_modelling()
    elif step == 'Tuning':
        text.timeline_tuning()
    elif step == 'Application':
        text.timeline_application()

def feature_engineering():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Feature Engineering**")
    
    st.markdown("### Select a Dataset to View")
    dataset = st.radio("# Test", ['one', 'two'])
    dataset
    choice = {
        'one': 'question112_len12_size36.csv',
        'two' : 'question123_len9_size36.csv'
    }
    
    df = ''
    
    if dataset == 'one':
        df = pd.read_csv(
#             'data/feature_sets/saltwaterdata.csv'
            'data/one_sentence/' + choice['one']
        )
        
    st.write(df)
    

def clustering():
    st.markdown("## **Master's Project**")
    st.markdown('---')
    st.markdown("# **Unsupervised Learning | Clustering**")
    
    st.markdown("### Choose a Dataset")
    options = [0,1,2, 3]
    display = (
        'How do you separate the water and salt from saltwater?', 
        'Which temperature system was Bill using if he considered 28 degree warm?',
        'How can Lee check if this object is conductive?',
        'After being heated, a Maillard reaction occurs. What does this indicate the presence of?'
    )
    num = st.radio(
        "The question below indicates the type of answers contained in the dataset", 
        options, 
        format_func = lambda x: display[x]
    )
    path = 'data/feature_sets/'
    files = {
        0: ['saltwaterdoc.csv','saltwaterdata.csv'],
        1: ['celciusdoc.csv', 'celciusdata.csv'],
        2: ['circuitdoc.csv', 'circuitdata.csv'],
        3: ['sugardoc.csv', 'sugardata.csv'],
    }


    doc= pd.read_csv(path + files[num][0])
    data= pd.read_csv(path + files[num][1])

    
    st.markdown("## Modeling Example")
    st.markdown('**Select the below based on what you want to see:**')
    doc_flag = st.checkbox('Display Question Info')
    data_flag = st.checkbox('Display Prediction Data')
    model_flag = st.checkbox('Display Model Info and Performance')
    chart_flag = st.checkbox('Display Reduced-Dimensionality Chart')
    
    
    if doc_flag:
        st.markdown('### Question Info')
        st.write(doc)
    if data_flag:
        st.markdown('### Prediction Data')
        st.write(data)
    if model_flag:
        st.markdown('### Model Data')
        
        nlp = Clustering_NLP(data, doc)
        st.write(nlp)
        nlp.correct_cluster_labels()
        st.write(nlp.accuracy())
        st.write(confusion_matrix(nlp.doc['label'], nlp.doc.cluster))
    if chart_flag:
        st.markdown('place holder')
        
def classification():
    pass

def conclusion():
    pass

def appendix():
    pass
# def plotlymap():
#     st.title("Market Intelligence Maps")
#     #Transactions per Capita
#     st.markdown(" ## **Ploty ScatterGeo**")
#     st.write(charts.create_scattergeo())

#     # County Trends over the month 
#     df = tools.read_data('data/linear_trends_nov.csv')
#     options = df.county_id.value_counts().index.to_list()
#     st.markdown('\n')
#     st.markdown(" ### **Trends Across the Month**")

#     county_id = st.multiselect('Type or Select a County', options)
#     if st.button('Run Search'):
#         st.write(charts.county_trends(county_id))

#     source_map = st.checkbox("View Map Source Code")
#     if source_map:
#         with st.echo():
#             """
#             **********Plotly Express**********
#             agg_usa = px.scatter_geo(
#                 data_frame= df, 
#                 lon = 'lon',
#                 lat = 'lat',
#                 color='Transactions <br>per Capita <br>Log Scale',
#                 size ='population',
#                 color_continuous_scale='Portland',
#                 locationmode = 'USA-states',
#                 scope = 'usa',
#                 title = '<b>Zelle Purple App <br>Per Capita Transactions <br>'+
#                 'October 2019 </b> <br> (Diameter Proportional to Population)',
#                 size_max = 25,
#                 opacity = 0.8,
#                 hover_data = ['text'],
#                 hover_name = 'title_text',
#                 text = 'text',
#             )

#             agg_usa.update_layout(
#                 showlegend = True,
#                 geo = dict(
#                     scope = 'usa',
#                     landcolor = 'rgb(50,50,50)',
#                 ),
#                 title_x=0.5,
#                 margin={"r":0,"t":200,"l":0,"b":0}

#             )

#             agg_usa

#             **********Low Level Plotly API**********
#             sgeo = go.Figure(data = go.Scattergeo(  
#             locationmode = 'USA-states',
#             lon = df.lon,
#             lat = df.lat,
#             mode = 'markers',
#             hoverinfo = 'none',
#             hovertemplate = df.text,
#             marker = dict(
#                 sizeref = 100000,                  
#                 colorscale= [
#                     [0, 'rgb(0, 51, 102)'],        
#                     [0.37, 'rgb(0, 51, 153)'],
#                     [0.50, 'rgb(0, 102, 0)'],
#                     [0.55, 'rgb(102, 153, 0)'],
#                     [0.66, 'rgb(255, 102, 0)'],
#                     [.75, 'rgb(204, 0, 0)'],
#                     [1, 'rgb(204, 0, 0)'], 
#                 ],
#                 size = df.population ,
#                 opacity = 0.8,
#                 color = df['Transactions <br>per Capita <br>Log Scale'],
#                 cmin = 0,
#                 colorbar = dict(
#                     title = 'Transactions per Capita <br>Log Scale',
#                     titleside = 'right',
#                     ticks = 'outside',
#                     tick0= 0,
#                     tickmode= 'array',
#                     tickvals= [-5, -4, -3, -2, -1, 0],
#                     dtick = "log"
#                 ),
#                 line = dict(
#                     width = 1,
#                 )
#             ),
#         ))

#         sgeo.update_layout(
#             title= \
#                 '<b>Zelle Purple App <br>' +
#                 'Transactions Per Capita </b><br>' +
#                 'November 2019 <br>' + 
#                 '(Diameter Proportional to Population)',
#             showlegend = False,
#             geo = dict(
#                 scope = 'usa',
#                 landcolor = 'rgb(75,75,75)',
#             ),
#             title_x=0.5,
#             margin={
#                 "r":50,
#                 "t":200,
#                 "l":0,
#                 "b":0
#             },
#             title_font_size = 14
#         )
#             """

# def mapboxmap():
#     st.title("Market Intelligence Maps")
#     st.markdown(" ## **Plotly and Mapbox**")

#     # Trial Transactions
#     st.markdown("\n")
#     datarange = st.slider(
#         "November 2019 Date Range",
#         1,
#         30,
#         (1,30)
#     )
#     st.write(charts.create_density_map(datarange))
    
#     if st.checkbox("View Map Source Code"):
#         with st.echo():
#             """
#             # Creates Figure
#             dollar = go.Figure()

#             # Adds First Trance Layer
#             dollar.add_trace(go.Scattermapbox(
#                     lat=df.latitude,
#                     lon=df.longitude,
#                     mode='markers',
#                     marker=go.scattermapbox.Marker(
#                         size=df.payment*5,
#                         color='rgb(138, 23, 252)',
#                         opacity=0.25
#                     ),
#                     text=df.text,
#                     hoverinfo='text'
#             ))

#             # Changes the Layout
#             dollar.update_layout(
#                 title='\
#                     <b>Zelle Purple App</b> <br>' +
#                     'November 2019 <br>' +
#                     'Trial Transactions < or = to $1',
#                 title_x = 0.5,
#                 autosize=True,
#                 hovermode='closest',
#                 showlegend=False,
#                 mapbox=go.layout.Mapbox(
#                     accesstoken=mapbox_token,           # Some mapbox maps require tokens
#                     bearing=0,                          # Create a username on their site to obtain
#                     center=go.layout.mapbox.Center(
#                         lat=38,
#                         lon=-97
#                     ),
#                     pitch=0,
#                     zoom= 3.25,
#                     style='light'
#                 ),
#                 height = 800,
#                 width = 1100,
#                 margin={
#                     "r":0,
#                     "t":100,
#                     "l":0,
#                     "b":0
#                 },
#             )
#             dollar.show()
#             """

# def deckgl_viz():
#     st.title("DeckGL Example")
#     with st.spinner("Loading in data..."):
#         charts.deckgl_example()

# def geocode():
#     st.title("Geocoding")
#     address_string = st.text_input("Input Address Here")
#     geolocator = Nominatim(user_agent="st_app")
#     location = geolocator.geocode(address_string)
#     # if st.button("Search for Address"):
#     if location:
#         pos = "Latitude: " + str(round(location.latitude,4)) + ", Longitude: " + str(round(location.longitude,4))
#         st.write(pos)
#         charts.geocode_map([location.latitude, location.longitude])

#     else:
#         st.write("No address input or found")

#     if st.checkbox("View Geocoding Example & Map Source Code"):
#         with st.echo():
#             """
#             **********Geocode Example**********
#             from geopy.geocoders import Nominatim

#             # Geocoding
#             geolocator = Nominatim(user_agent="generic_app_name")
#             location = geolocator.geocode(address_string)
#             print(location.latitude, location.longitude)
#             #Location object has much more information than just coordinates

#             # Reverse Geocoding:
#             location = geolocator.reverse("52.509669, 13.376294")

#             Geopy Documentation
#             https://pypi.org/project/geopy/

#             **********Plotly Express**********
#             fig = px.scatter_geo(
#                 temp,                       
#                 lon = 'lon',
#                 lat = 'lat',
#                 color = 'color',
#                 size = 'size',
#                 projection = 'orthographic'
#             )
#             fig.update_layout(
#                 geo = dict(
#                     landcolor = 'rgb(50,50,50)',
#                 )
#             )          
#             fig.show           
#             """
    
# def multiprocessing():
#     st.title("Multiprocessing with Pandas and Dask")

#     if st.checkbox("Show Dask Info", 1):
#         st.markdown("## What is Dask?")
#         st.markdown('- "Dask provides advanced parallelism for analytics, enabling performance at scale for the tools you love"')
#         st.markdown("## Why use Dask?")
#         st.markdown('- "Dasks schedulers scale to thousand-node clusters and its algorithms have been tested on some of the largest supercomputers in the world. But you do not need a massive cluster to get started. Dask ships with schedulers designed for use on personal machines. Many people use Dask today to scale computations on their laptop, using multiple cores for computation and their disk for excess storage."')
#     if st.checkbox("Show Problem Info"):
#         st.markdown("## Specific Problem")
#         st.markdown("### 1.7 million user locations need to be aggregated to the county level")
#         st.markdown("## Solution")
#         st.markdown("### Utilize multiprocessing and vectorized operations to do just that")
#         st.markdown('- Get rectangular coordinates of US, Alaska, and Hawaii')
#         st.markdown('- Drop all user locations not inside the these coordinates')
#         st.markdown('- Get single user location, assign to closet country (1.7M Users x 3k Counties)')
#         st.markdown('- Repeat')

#     if st.checkbox('Show Definitions'):
#         st.markdown('### **Multiprocessing:** the running of two or more programs or sequences of instructions simultaneously by a computer with more than one central processor. Dask handles multiprocessing.')
#         st.markdown('### **Vectorized Operations:** Rather than operating on a single value at a time, operate on a set of values(vector) at a time. Pandas supports vectorized operations.')


#     if st.checkbox("Show Dask Implementation"):
#         with st.echo():
#             """
#             ********Dask Implementation of Mutliprocessing********
#             import dask.dataframe as dd
#             import multiprocessing
#             import pandas as pd

#             new_data = dd.from_pandas(dataframe, npartitions=4*multiprocessing.cpu_count())\
#                 .map_partitions(lambda df: df.apply((lambda row: generic_function(parameter1, parameter2)), axis = 1), meta = dtype)
#                 .compute(scheduler = 'processes')
#             """

# def workflow():
#     st.title("Microteam Workflow Git/Lab")
#     st.markdown("## **General Overview**")
#     st.markdown("- Each Developer branches off Master ")
#     st.markdown("- When a feature is complete, push code, create merge request")
#     st.markdown("- Code maintainer approves request, deletes branch")
#     st.markdown("- All other branches rebase")
#     st.markdown("- Work continues")

#     st.markdown("## **General Workflow**")
#     st.graphviz_chart('''
#         digraph {
#             Code_Repo_Created -> Create_Branch -> Finish_Feature -> Update_Master -> Create_Branch
#             Update_Master -> Branches_Rebase -> Finish_Feature

#         }
#     ''', width = 1000, height = 400)

#     if st.checkbox("Show Cheat Sheet"):
#         st.markdown("## **Mini Workflow Cheatsheet**")
#         st.markdown("### Adding a Repo")
#         st.markdown("- git clone <repo clone url or ssh here>")
#         st.markdown("- cd <folder name of repo>")
#         st.markdown("### Creating and developing on a Branch")
#         st.markdown("- git branch <insert branch name here>")
#         st.markdown("- git checkout <same branch name from above>")
#         st.markdown("### Committing")
#         st.markdown("- git status")
#         st.markdown("- git add .")
#         st.markdown('- git commit -m <"52 Character message that is meaningful">')
#         st.markdown("*These are all local changes. To make remote changes, you need to Push*")
#         st.markdown("### Pushing")
#         st.markdown("- git push origin <Branch name goes here>")
#         st.markdown('*Navigate to GitLab and create a merge request if the feature is ready.\n If not, feel free to continue to develop your branch. In general, strive for making small incremental changes*')
#         st.markdown("### Rebasing")
#         st.markdown("- git pull --rebase origin master")
#         st.markdown("*If you forget the rebase part, you will be asked to create a commit message which will open up in Vim.*")
#         st.markdown("*Pulling: Fetches latest changes of current branch from remote, merges changes into local copy of branch.* ")
#         st.markdown("*Rebasing: Moves the commits of one of the branches on top of the other, i.e. takes all prior commits and then puts yours ontop of them.* ")

# def experience():
#     st.title("Internship Program Takeaways")
#     st.markdown("- How to make maps")
#     st.markdown("- How to manage environments for coding")
#     st.markdown("- Utilizing git effciently")
#     st.markdown("- Difference between Industry and School")



