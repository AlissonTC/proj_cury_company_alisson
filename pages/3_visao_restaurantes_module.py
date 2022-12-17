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
import plotly.graph_objects as go

st.set_page_config(page_title='Visão Restaurantes',page_icon="🍽",layout='wide')

#====================================================================
# Funções
#====================================================================
def avg_std_time_on_traffic(df):
    df_aux = df.loc[:,['City','Time_taken(min)','Road_traffic_density']].groupby(['City','Road_traffic_density']).agg({'Time_taken(min)' :[ 'mean','std']})
    df_aux.columns = ['avg_time','std_time']
    df_aux = df_aux.reset_index()

    fig = px.sunburst( df_aux, path =[ 'City','Road_traffic_density'],
                      values = 'avg_time',
                      color='std_time',
                      color_continuous_scale = 'RdBu',
                      color_continuous_midpoint = np.average(df_aux['std_time']))
    return fig

            
def avg_std_time_graph(df):
    df_aux = (df.loc[:,['City','Time_taken(min)']]
              .groupby('City')
              .agg({'Time_taken(min)':['mean','std']}))
    df_aux.columns=['avg_time','std_time']
    df_aux =  df_aux.reset_index()
    
    fig= go.Figure()
    fig.add_trace( go.Bar( name='Control',
                          x=df_aux['City'],
                          y=df_aux['avg_time'],
                          error_y=dict(type='data', array=df_aux['std_time'])))
    
    fig.update_layout(barmode='group')

    return fig

    
def avg_std_time_delivery(df,Festival, op):    
    """
    Esta função calcula o tempo médio eo desvio padrão de tempo de netrga.
    Parâmetros:
    Input:
    - df : dataframe com os dados necessarios para o calculo
    - op : tipo de operação que precisa ser calculado
    'avg_time': Calcula o tempo medio
    'std_time': calcula o desvio padrão do tempo

    Output:
    - df: dataframe com duas colunas e uma linha                   


    """
    df_aux =( df.loc[:, ['Time_taken(min)','Festival']]
             .groupby(['Festival'])
             .agg({'Time_taken(min)': ['mean','std']}))

    df_aux.columns = ['avg_time' , 'std_time']
    df_aux = df_aux.reset_index()
    df_aux = np.round(df_aux.loc[df_aux['Festival'] == Festival, op] ,2)

    return df_aux


def distance(df, fig):
    if fig == False:
        cols = ['Delivery_location_latitude','Delivery_location_longitude','Restaurant_latitude','Restaurant_longitude']
        df['distance'] = df.loc[:,cols].apply(lambda x: haversine((x['Delivery_location_latitude'],x['Delivery_location_longitude']),
                                                                  ( x['Restaurant_latitude'],x['Restaurant_longitude'])), axis=1)
        avg_distance =np.round( df['distance'].mean(),2)         
        return avg_distance
        
    else:
        cols = ['Delivery_location_latitude','Delivery_location_longitude','Restaurant_latitude','Restaurant_longitude']
        df['distance'] = df.loc[:,cols].apply(lambda x: haversine((x['Delivery_location_latitude'],x['Delivery_location_longitude']),
                                                                  ( x['Restaurant_latitude'],x['Restaurant_longitude'])), axis=1)
        
        avg_distance =( df.loc[:,['City', 'distance']]
                           .groupby('City')
                           .mean()
                           .reset_index())
        fig = go.Figure( data=[go.Pie( labels=avg_distance['City'],
                                      values=avg_distance['distance'],
                                      pull=[0,0.1,0])])  
        
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
st.header('Markplace - Visão Restaurantes')

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
st.sidebar.markdown("### Powered by Comunidade DS")

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
        st.title('Overall  Metrics')
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)        
        with col1:         
            delivery_unique = len(df.loc[:,'Delivery_person_ID'].unique())
            col1.metric('Entregadores Únicos',delivery_unique)                            
                                  
        with col2: 
            avg_distance = distance(df, fig=False)
            col2.metric('Distancia Media de entregas', avg_distance)
                                  
            
        with col3:        
            df_aux = avg_std_time_delivery(df,'Yes','avg_time')
            col3.metric('Tempo Médio de Entrega c/ Festival', df_aux)


        with col4:
            df_aux = avg_std_time_delivery(df,'Yes','std_time')
            col4.metric('Desvio Médio de Entrega c/ Festival', df_aux)
               
            
        with col5:
            df_aux = avg_std_time_delivery(df,'No','avg_time')            
            col5.metric('Tempo Médio de Entrega s/ Festival', df_aux)
            
                  
        with col6:
            df_aux = avg_std_time_delivery(df,'No','std_time') 
            col6.metric('Desvio Médio de Entrega s/ Festival', df_aux)                     
                                        
            
    with st.container(): 
            st.markdown("""---""")
            col1,col2 = st.columns(2,gap='Large')
            with col1:                       
                st.markdown('### Dsitribuição de Tempo')  
                st.markdown("""---""")
                st.markdown('##### Tempo medio por desvio padrao de entrega por cidade')
                fig = avg_std_time_graph(df)
                st.plotly_chart(fig, use_container_width=True) 
         
            with col2:
                    st.markdown('### Distribuição de Distância') 
                    st.markdown("""---""")
                    st.markdown('##### Tempo medio e o desvio padrao de entrega por cidade e tipo de pedido') 
                                   
                    cols = ['City','Time_taken(min)', 'Type_of_order']
                    df_aux =( df.loc[:, ['City','Time_taken(min)', 'Type_of_order']]
                             .groupby(['City','Type_of_order'])
                             .agg({'Time_taken(min)': ['mean','std']}))

                    df_aux.columns=['avg_time','std_time']
                    df_aux= df_aux.reset_index()
                    st.dataframe(df_aux)                   

                                   

            
                        
    with st.container():
        st.markdown("""---""")        
        col1,col2 = st.columns(2, gap='Large')
        
        with col1:
            st.markdown('### Distribuição da Distância')
            st.markdown("""---""") 
            st.markdown('##### Distância media e o desvio padrao de entrega entre cidades')
            fig = distance(df, fig=True) 
            st.plotly_chart(fig,use_container_width=True) 
            
            
        with col2: 
            st.markdown('### Distribuição do Tempo')
            st.markdown("""---""")            
            st.markdown('##### Tempo medio e o desvio padrao de entrega por cidade e trafego')
            fig = avg_std_time_on_traffic(df)
            st.plotly_chart(fig,use_container_width=True)     

          

    








     
        
    
    
#1 Quantidade de entregadores unicos
#delivery_unique = len(df.loc[:,'Delivery_person_ID'].unique()
    
#2. dsitãncia media dos restautantes e dos locais de entrega
#cols = ['Delivery_location_latetude','Delivery_location_longitude','Restaurant_latitude','Restaurant_longitude']
   #df['distance'] = df.loc[:,cols].apply(lambda x: haversine((x['Delivery_location_latetude'].x['Delivery_location_longitude']) , ( x['Restaurant_latitude'],x['Restaurant_longitude']), axis=1)
# avg_distance = df['distance'].mean()  
    
#3.#cols = ['Delivery_location_latetude','Delivery_location_longitude','Restaurant_latitude','Restaurant_longitude']
#df['distance'] = df.loc[:,cols].apply(lambda x: haversine((x['Delivery_location_latetude'].x['Delivery_location_longitude']) , ( x['Restaurant_latitude'],x['Restaurant_longitude']), axis=1)
# avg_distance = df['distance'].mean().reset_index()

#Avg_distance
#pull is given as a fraction of the pie radius

#fig= go.Figure( data =(labels = avg_distance['City'], value = avg_distance['distance'], pull=[0,0.1,0])) 

#4. Tempo medio por desvio padrao de entrega por cidade

#cols = ['City','Time_taken(min)']
#df_aux = df.loc[:,cols].groupby('City').agg({'Time_taken(min)': ['mean' ,'std']})

#df_aux,columns = [ 'avg_time','std_time']
#df_aux = df_aux.reset_index()

#fig = go.Figure()
#fig.add_trace( go.Bar( name = 'Control',
                     # x = df_aux['City'],
                     # y = df_aux['afg_time'],
                     # error_y = dict( type = 'data' ,  array= df_aux['std_time'])))

#fig.updata_layout( barmode = 'group')
#fig.show()

#5 O tempo medio e o desvio padrao de entrega por cidade e tipo de pedido

#import numpy as np

#cols = ['City','Time_taken(min)', 'Type_of_order']
#df_aux =  df.loc[; cols].groupby(['Type_of_order']).agg({'Time_taken(min)': ['mean','std']})

#df_aux,columns=['avg_time','std_time']

#df_aux= df_aux.reeset_index()

#6.    
#cols = ['City','Time_taken(min)','Road_traffic_density']
#df_aux = df.loc{:,cols].groupby(['City','Raod_traffic_density']).agg({'Time_taken(min)' :[ 'mean','std']})

#df_aux.columns = ['avg_time','std_time']
    
#df_aux = df_aux.reset_index()
                
#fig = px.sunburst( df_aux, path =[ 'City','Road_traffic_density'],
                  #values = 'avg_time',
                  #color='std_time',
                  #color_continuous_scale = 'RdBu',
                  #color_continuous_midpoint = np.average(df_aux['std_time']))                  
                
#fig.show()                
                
#6. o tempo medio de entrega durante os Festivais

#cols = ['Time_taken(min)','Festival']
#df_aux = df.loc[:, cols].groupby(['Ferstival']).agg({'Time_taken(min)': ['mean','std']})

#df_aux.columns = ['avg_time' , 'std_time']
#df_aux = df_aux.reset_index()

#df_aux




















