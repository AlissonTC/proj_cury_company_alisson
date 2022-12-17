
#Import package

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px 
import datetime
from datetime import datetime
from streamlit_folium import folium_static
import folium
from haversine import haversine
from PIL import Image 

st.set_page_config(page_title='Visão Empresa',page_icon = "📈",layout='wide')                  
                   
#====================================================================
# Funções
#====================================================================
def  country_maps(df): 
    #6. A localização central de cada cidade por tipo de tráfego
    data_plot = (df.loc[:,['City','Road_traffic_density','Delivery_location_latitude','Delivery_location_longitude']]
                 .groupby(['City','Road_traffic_density'])
                 .median()
                 .reset_index())
    map =folium.Map()
    df_aux = data_plot
    

    for index , location_info in df_aux.iterrows():
        folium.Marker([location_info['Delivery_location_latitude'],
                       location_info['Delivery_location_longitude']],
                      popup=location_info[['City','Road_traffic_density']]).add_to(map)


    folium_static(map, width = 1024, height = 600 ) 
    
    return None
        


def  order_share_by_week(df):
    """ Esta função tem a responsabilidade de limpar o dataframe
    Tipos de limpeza:
    1.Remoção dos dados NaN
    2. Mudança do tipo coluna de dados
    3. Remoção dos espaços das variáveis de texto
    4. Formatação da coluna de datas
    5.Limpeza da coluan de tempo ( remoção do texo da varia´vel numérica)
    Input: Dataframe
    Output: Dataframe  

    """    
    #Quantidade de padidos por semana / Número único de entregadores por semana
    df_aux01 = df.loc[:,['ID','week_of_year']].groupby('week_of_year').count().reset_index()
    df_aux02 = df.loc[:,['Delivery_person_ID','week_of_year']].groupby('week_of_year').nunique().reset_index()

    df_aux = pd.merge(df_aux01,df_aux02,how='inner',on='week_of_year')
    df_aux['order_delivery'] = df_aux['ID'] / df_aux['Delivery_person_ID']

    #Desenha o grafico de linhas
    fig = px.line(df_aux, x='week_of_year', y = 'order_delivery')

    return fig


def order_by_week(df):    
    #2. Quantidade de pedidos por semana
        df['week_of_year'] = df['Order_Date'].dt.strftime('%U')
        df_aux = (df.loc[:,['ID','week_of_year']]
                  .groupby(['week_of_year'])
                  .count()
                  .reset_index())

        #Desenha gráfico de piza
        fig = px.line(df_aux,x='week_of_year',y='ID') 
        
        return fig


def traffic_order_city(df):

    #4. Comparação do volume de pedidos por cidade e tipo de tráfego.
    df_aux= (df.loc[:,['ID','City','Road_traffic_density']]
             .groupby(['City','Road_traffic_density'])
             .count()
             .reset_index())

    #Desenhar gráfico de bolhas            
    fig =px.scatter(df_aux,x='City',y='Road_traffic_density',size ='ID',color='City')
    
    return fig



def traffic_order_share(df):
    #3. Distribuição dos pedidos por tipo de tráfego.
    df_aux1= (df.loc[:,['ID','Road_traffic_density']]
              .groupby('Road_traffic_density')
              .count()
              .reset_index())
    df_aux1['frac_ID']= (df_aux1['ID'])/(df_aux1['ID'].sum())
    #Desenhar o gráfico de linhas
    fig = px.pie(df_aux1,values='frac_ID',names='Road_traffic_density')
    
    return fig
                

def order_metric(df):
    #Order metric
    #1. Quantidade de pedidos por dia.
    df_aux = (df.loc[:,['ID','Order_Date']]
              .groupby(['Order_Date'])
              .count()
              .reset_index())

    #Desenhar o gráfico de linhas
    fig = px.bar( df_aux,x ='Order_Date',y='ID' )

    return fig


def clean_code(df):
    """ Esta função tem a responsabilidade de limpar o dataframe
    Tipos de limpeza:
    1.Remoção dos dados NaN
    2. Mudança do tipo coluna de dados
    3. Remoção dos espaços das variáveis de texto
    4. Formatação da coluna de datas
    5.Limpeza da coluan de tempo ( remoção do texo da varia´vel numérica)
    Input: Dataframe
    Output: Dataframe  

    """    
    # Remover espaços nas strings/texto/objetos
    df = df.replace({' ':'','\(min\)':''},regex= True)
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(int)
    linhas_vazias = df['Delivery_person_Age'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    linhas_nan = df['Road_traffic_density'] != 'NaN'
    df = df.loc[linhas_nan,:]

    # Conversao de texto/categoria/string para numeros inteiros
    df['Delivery_person_Age'] = df['Delivery_person_Age'].astype( int )

    # Conversao de texto/categoria/strings para numeros decimais
    df['Delivery_person_Ratings'] = df['Delivery_person_Ratings'].astype( float )

    # Conversao de texto para data
    df['Order_Date'] = pd.to_datetime( df['Order_Date'], format='%d-%m-%Y' )

    # Remove as linhas da coluna multiple_deliveries que tenham o 
    # conteudo igual a 'NaN '
    linhas_vazias = df['multiple_deliveries'] != 'NaN'
    df = df.loc[linhas_vazias, :]
    df['multiple_deliveries'] = df['multiple_deliveries'].astype( int )
    # Remove as linhas da coluna  city que tenham o 
    # conteudo igual a 'NaN '
    linhas_v = df['City'] != 'NaN'
    df = df.loc[linhas_v, :]
    # Comando para remover o texto de números
    #for i in range( len( df ) ):
    #df.loc[:, 'Time_taken(min)'] = re.findall( r'\d+', df.loc[i, 'Time_taken(min)'] )
    #Remover os NAs do dataframe
    df = df.dropna(axis=0,how='any')
    df =df.reset_index(drop=True)
    
    return df            

#---------------------------- Inicio da Estrutura lógica do código ------------
#====================
# Import dataset
#====================
df_raw = pd.read_csv('dataset/train.csv')

#====================
#Copy Dataset
#====================
df = df_raw.copy()

#====================
# Cleaning Dataset
#====================
df = clean_code(df)


#==========================================================================
# Barra Lateral
#==========================================================================
st.header('Markplace - Visão Cliente')

image = Image.open('logo.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown('# Cury Company')
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.sidebar.markdown('## Select a Limit Date ') 

order_date_range = st.sidebar.slider(
    "Choose the start date?",
    value = pd.datetime(2022,4,13),
    min_value = pd.datetime(2022,2,11),
    max_value = pd.datetime(2022,4,6),
    format="DD-MM-YYYY")
st.sidebar.markdown("""---""")


traffic_level = st.sidebar.multiselect(
    "How would you like to choose the Traffic Condition ?",
    ["Low", "Medium", "High","Jam"],
    default=["Low", "Medium", "High","Jam"])


st.sidebar.markdown("""___""")


wheather_type = st.sidebar.multiselect(
    "How would you like to choose the Wheather Condition ?",
    ['conditionsSunny', 'conditionsStormy',
     'conditionsSandstorms','conditionsCloudy',
     'conditionsFog', 'conditionsWindy'],
    default=['conditionsSunny', 'conditionsStormy',
             'conditionsSandstorms','conditionsCloudy',
             'conditionsFog', 'conditionsWindy'])


st.sidebar.markdown("""---""")
st.sidebar.markdown("### Powered by Comunidade4 DS")

#Filtro de data
df = df.loc[(df['Order_Date'] < order_date_range),:]

#Filtro de transito
df = df.loc[df['Road_traffic_density'].isin(traffic_level),:]



#Filtro de condição climatica
df = df.loc[df['Weatherconditions'].isin(wheather_type),:]
st.dataframe(df)


#=========================================================================
# Layout no Streamlit
#=========================================================================
tab1, tab2, tab3 = st.tabs(["Visão Gerencial", "Visão Tática", "Visão Geográfica"])

with tab1: 
    with st.container():
        st.markdown('##### Order By Day')
        fig = order_metric(df)
        st.plotly_chart(fig, use_container_width=True )

    with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('##### Traffic Order Share')
                fig = traffic_order_share(df)        
                st.plotly_chart(fig, use_container_width=True)           


            with col2:              
                st.markdown('##### Traffic Order City')
                fig = traffic_order_city(df)
                st.plotly_chart(fig,use_container_width = True)
        
with tab2:
    with st.container():
        st.markdown('##### Order By Week') 
        fig = order_by_week(df)        
        st.plotly_chart(fig,use_container_width=True) 
        
    with st.container():        
        st.markdown('##### Order Share By Week')  
        fig = order_share_by_week(df)
        st.plotly_chart(fig,use_container_width = True)                                  
        
      
with tab3:
    st.markdown('##### Country Maps')
    country_maps(df)
    

      

  
    


                               
