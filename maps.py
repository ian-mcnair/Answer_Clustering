from plotly.offline import plot
import plotly.graph_objects as go
import pandas as pd
import plotly.figure_factory as ff
import numpy as np

def create_density_map(daterange = [1,30]):
    # Load data
    df = pd.read_csv('data/201911_Zelle_AllCounties_AllInfo_03.csv')

    #Filter Data
    df = df[
        (df.payment <= 1) &
        (df.day >= daterange[0]) &
        (df.day <= daterange[1])
    ].copy()

    #Create Text
    df['text'] = \
    '<b>' + df.county + " County, " + df.state_abv + '</b> <br>' + \
    'Payment Amount: $' + df.payment.map('{:.2f}'.format)
    mapbox_token = 'pk.eyJ1IjoiaW1jbnJld3MiLCJhIjoiY2szaGZ6NTViMGNhajNibmpuN3lucjh1dSJ9.JIE-BxaKnB66HACq5j5tmg'

    # Create base figure
    dollar = go.Figure()

    # Create map trace
    dollar.add_trace(go.Scattermapbox(
            lat=df.latitude,
            lon=df.longitude,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=df.payment*5,
                color='rgb(138, 23, 252)',
                opacity=0.25
            ),
            text=df.text,
            hoverinfo='text'
        ))

    # Update layout
    dollar.update_layout(
        title='<b>Zelle Purple App</b> <br>November 2019 <br>Trial Transactions < or = to $1',
        title_x = 0.5,
        autosize=True,
        hovermode='closest',
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=38,
                lon=-97
            ),
            pitch=0,
            zoom= 2.75,
            style='light'
        ),
        margin={"r":0,"t":100,"l":0,"b":0},
    )

    return dollar


def create_scattergeo():
    df = pd.read_csv('data/county_positions.csv', dtype = object)
    records = pd.read_csv('data/201911_Zelle_Monthly_Capita_AllCounties.csv')
    records['fips'] = records.fips.astype(str).apply(lambda x: '{0:0>5}'.format(x))
    users = pd.read_csv('data/Unique_User_Counts.csv')
    users['fips'] = users.fips.astype(str).apply(lambda x: '{0:0>5}'.format(x))
    users = users[['fips', 'id']]
    records['Transactions <br>per Capita <br>Log Scale'] = round(np.log10(records.per_capita_trans),3)
    rec2 = pd.merge(
        left = records,
        on = 'fips',
        right = users,
        how = 'left'
    )
    rec2['saturation'] = rec2.id / rec2.population *100
    rec2['Saturation <br>Log Scale'] = round(np.log10(rec2.saturation),3)
    rec2['per_1000'] = rec2.per_capita_trans*1000
    rec2['text'] ='<b>'+\
    rec2.county + ' County, ' + rec2.state_abv +'</b> <br>'\
    'Population: ' + rec2.population.map('{:,}'.format) + '<br>'\
    'Number of Transactions: ' + rec2.attp_trans.map('{:,}'.format) + ' <br>' +\
    'Transaction Total: ' + (rec2.total_payments).map('${:,.2f}'.format) + ' <br>' +\
    'Transactions per Capita: ' + rec2.per_capita_trans.map('{:,.4f}'.format)+ ' <br>'\
    'Transactions per Capita (Log): ' + rec2['Transactions <br>per Capita <br>Log Scale'].map('{:.2f}'.format) + ' <br>' +\
    'Zelle CMA Usage: ' + rec2.saturation.map('{:.3f}'.format) + "%" \
    '<extra></extra>'

    sgeo = go.Figure(data = go.Scattergeo(
        locationmode = 'USA-states',
        lon = rec2.clon,
        lat = rec2.clat,
        mode = 'markers',
        hoverinfo = 'none',
        hovertemplate = rec2.text,

        marker = dict(
            sizeref = 100000,
            colorscale= [
                [0, 'rgb(0, 51, 102)'],        
                [0.37, 'rgb(0, 51, 153)'],
                [0.50, 'rgb(0, 102, 0)'],
                [0.55, 'rgb(102, 153, 0)'],
                [0.66, 'rgb(255, 102, 0)'],
                [.75, 'rgb(204, 0, 0)'],
                [1, 'rgb(204, 0, 0)'], 
            ],
            size = rec2.population ,
            opacity = 0.8,
            color = rec2['Transactions <br>per Capita <br>Log Scale'],
            cmin = 0,
            colorbar = dict(
                title = 'Transactions per Capita <br>Log Scale',
                titleside = 'right',
                ticks = 'outside',
                tick0= 0,
                tickmode= 'array',
                tickvals= [-5, -4, -3, -2, -1, 0],
                dtick = "log"
            ),
            line = dict(
                width = 1,
            )
        ),
    ))

    sgeo.update_layout(
        title= '<b>Zelle Purple App <br>Transactions Per Capita </b><br>November 2019 <br>(Diameter Proportional to Population)',
        showlegend = False,
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(75,75,75)',
        ),
        title_x=0.5,
        margin={"r":50,"t":200,"l":0,"b":0},
        title_font_size = 14

    )

    return sgeo