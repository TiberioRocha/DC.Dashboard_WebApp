# ------------------------------
# IMPORTS
# ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import ipywidgets as widgets
from ipywidgets import fixed
import plotly.express as px
import datetime
import _imp
import pickle
st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True)
def get_data(path):
    data = pd.read_excel(path)
    return data

def get_data(path_label):
    data_label = pd.read_excel(path_label)
    return data_label

def overview_data(data):
    f_action = st.sidebar.multiselect('Enter Action', data['Action'].unique())
    f_atributes = st.sidebar.multiselect('Enter Columns', data.columns)
    st.title('------------------Data Overview------------------')

    if (f_action != []) & (f_atributes != []):
        data = data.loc[data['Action'].isin(f_action), f_atributes]
    elif (f_action != []) & (f_atributes == []):
        data = data.loc[data['Action'].isin(f_action), :]
    elif (f_action == []) & (f_atributes != []):
        data = data.loc[:, f_atributes]
    else:
        data = data.copy()
    st.dataframe(data)
    st.write(f_atributes)
    # -----------Descriptive Stats
    num_atributes = data[['Num_Likes', 'Report_Age']]
    media = pd.DataFrame(num_atributes.apply(np.mean))
    mediana = pd.DataFrame(num_atributes.apply(np.median))
    std = pd.DataFrame(num_atributes.apply(np.std))
    variancia = pd.DataFrame(num_atributes.apply(np.var))

    max_ = pd.DataFrame(num_atributes.apply(np.max))
    min_ = pd.DataFrame(num_atributes.apply(np.min))

    df1 = pd.concat([max_, min_, media, mediana, std, variancia], axis=1).reset_index()
    df1.columns = ['atributes', 'max', 'min', 'media', 'mediana', 'std', 'variancia']
    # data_label = pd.concat([ID,CPF,Total_Likes_User,Citizen_Label], axis = 1).reset.index()
    # data_label = ['ID','CPF','Total_Likes_User','Citizen_Label']

    st.title('--------------Descriptive Statistics--------------')
    st.dataframe(df1, height=600)
    st.title('--------------Citizen Label--------------')
    st.dataframe(data_label, height=600)

    return None


def portfolio_density(data):
    st.title('----------------Report Density Map--------------')
    c1, c2 = st.columns((1, 1))
    df = data.sample(100)
    density_map = folium.Map(location=[data['Lat'].mean(),
                                       data['Lon'].mean()], default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in df.iterrows():
        folium.Marker([row['Lat'], row['Lon']],
                      popup='ID{0} on: {1}, Number of Likes {2}'.format(row['ID'],
                                                                        row['Date_day'],
                                                                        row['Num_Likes'])).add_to(marker_cluster)
    with c1:
        folium_static(density_map)
    return None


def data_map(data):
    st.title('-----------------Relevance Map-----------------')
    data_map = data[['ID', 'CPF', 'Date_day', 'Action', 'Lat', 'Lon', 'Num_Likes']]

    map_relevance = px.scatter_mapbox(data_map, lat='Lat', lon='Lon', color='Num_Likes',
                                      color_continuous_scale=px.colors.sequential.Bluered,
                                      size='Num_Likes', hover_name='Action',
                                      hover_data=['ID', 'CPF'],
                                      color_discrete_sequence=['fuchsia'],
                                      zoom=8, height=300)

    map_relevance.update_layout(mapbox_style='open-street-map')
    map_relevance.update_layout(height=600, margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

    st.plotly_chart(map_relevance)

    st.title('-----------------Urgency Map-----------------')
    data_map = data[['ID', 'CPF', 'Date_day', 'Action', 'Lat', 'Lon', 'Num_Likes', 'Report_Age']]

    map_urgency = px.scatter_mapbox(data_map, lat='Lat', lon='Lon', color='Report_Age',
                                    color_continuous_scale=px.colors.sequential.Jet,
                                    size='Report_Age', hover_name='Action',
                                    hover_data=['ID', 'CPF'],
                                    color_discrete_sequence=['fuchsia'],
                                    zoom=8, height=300)

    map_urgency.update_layout(mapbox_style='open-street-map')
    map_urgency.update_layout(height=600, margin={'r': 0, 't': 0, 'l': 0, 'b': 0})

    st.plotly_chart(map_urgency)
    st.title('-----------Interactivity through LIKES----------')

    df = data[['Date_day', 'Num_Likes']].groupby('Date_day').sum().reset_index()

    fig = px.line(df, x='Date_day', y='Num_Likes')

    st.plotly_chart(fig, use_container_width=True)

    return None


if __name__ == '__main__':
    path = r'TiberioRocha/DC.Dashboard_WebApp/Report_Total3.xls'
    path_label = r'TiberioRocha/DC.Dashboard_WebApp/Ranking_Label.xlsx'
    data = get_data(path)
    data_label = get_data(path_label)
    overview_data(data)
    portfolio_density(data)
    data_map(data)
st.header('EXEMPLO')
st.sidebar.subheader('Num')

f_dar = st.sidebar.slider('Date_day', min_value=0, max_value=100)

pickle.dump('__main__', open('DC_Dashboard_clf.pkl', 'wb'))
