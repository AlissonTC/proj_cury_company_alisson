
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
 
st.set_page_config(page_title='Visão Entregadores',page_icon="🚚",layout='wide')

#====================================================================
# Funções
#====================================================================
def top_delivers(df , top_asc= False):                     
                df_aux = (df.loc[:,['Time_taken(min)','Delivery_person_ID','City']]
                          .groupby(['City','Delivery_person_ID'])
                          .mean()
                          .sort_values(['City','Time_taken(min)'],ascending= top_asc )
                          .reset_index())

                df_aux01 = df_aux.loc[df_aux['City'] == 'Metropolitian',:].head(10)
                df_aux02 = df_aux.loc[df_aux['City'] == 'Urban' ,:].head(10)
                df_aux03 = df_aux.loc[df_aux['City'] == 'Semi-Urban' ,:].head(10) 

                df3 = pd.concat([df_aux01,df_aux02,df_aux03]).reset_index(drop=True) 
                
                return df3
            
            
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
st.header('Markplace - Visão Entregadores')

image_path ='logo.png'
image = Image.open(image_path)
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

tab1, tab2, tab3 = st.tabs(["Visão Gerencial", "_", "_"])

with tab1:
    with st.container():
        st.title('Overhall Metrics')
        
        col1, col2, col3, col4 = st.columns(4, gap='Large')
        
      
        with col1: 
            #1. A maior idade dos entregadores.
                
            maior_idade = df['Delivery_person_Age'].max()
            col1.metric("Maior  Idade",maior_idade)

            
        with col2:  
            #1. A menor idade dos entregadores.
            
            menor_idade = df['Delivery_person_Age'].min()
            col2.metric("Menor  idade",menor_idade)
            
         
        
        with col3:    
            #2. A melhor condição de veículos.
            
            melhor_condicao = df['Vehicle_condition'].max()
            col3.metric('Melhor Condição',melhor_condicao)
            
        with col4:   
            #2. A pior condição de veículos.
            
            pior_condicao = df['Vehicle_condition'].min() 
            col4.metric('Pior condição',pior_condicao)
                        
    with st.container():
        st.markdown("""___""")
        st.title('Avaliações')
        
        col1, col2 = st.columns(2, gap= 'Large')
    
        with col1:                
            #3. A avaliação média por entregador.
            st.markdown('##### Avalição medias por Entregador')
            df_aux = (df.loc[:,['Delivery_person_Ratings','Delivery_person_ID']]   
            .groupby(['Delivery_person_ID'])
            .mean()
            .reset_index())
            st.dataframe(df_aux)
            
          
        with col2:
            #4. A avaliação média e o desvio padrão por tipo de tráfego.
            
            st.markdown('##### Avaliações medias por transito')
            df_aux = df.loc[:,['Delivery_person_Ratings','Road_traffic_density']].groupby('Road_traffic_density').agg({'Delivery_person_Ratings':['mean','std']})
            df_aux.columns=['Ratings_mean','Ratings_std']
            df_aux = df_aux.reset_index()
            st.dataframe(df_aux)           
            
            
            #5. A avaliação média e o desvio padrão por condições climáticas.
            
            st.markdown('##### Avaliações medias por clima')
            df2 = df.loc[:,['Delivery_person_Ratings','Weatherconditions']].groupby('Weatherconditions').agg({'Delivery_person_Ratings':['mean','std']})
            df2.columns =['Rantins_mean','Ranting_std']
            df2=df2.reset_index()
            st.dataframe(df_aux)
            
            
    with st.container():
        st.markdown("""___""")
        st.title('Velocidade de entrega')
        
        col1, col2 = st.columns(2, gap ='Large')
        
       
        with col1:
            #6.Os 10 entregadores mais rápidos por cidade.            
            st.markdown('##### Top entregadores mais rapidos')
            df3 = top_delivers(df, top_asc=True )                                  
            st.dataframe(df3)
            
        with col2:
            st.markdown('##### Top entregadores mais lentos') 
            #7.Os 10 entregadores mais lentos por cidade.
            df3 = top_delivers(df, top_asc=False )            
            st.dataframe(df3)