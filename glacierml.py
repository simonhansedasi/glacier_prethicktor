# import sys
# !{sys.executable} -m pip install 
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import geopy.distance
import matplotlib.patches as mpatches
import plotly.express as px
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.ticker as ticker
tf.random.set_seed(42)


pd.set_option('mode.chained_assignment',None)

def module_selection_tool():
    print('please input module code:')

    module = input()

    if module == 'sm1':
        df1 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'n'
        )
        dataset = df1
        dataset.name = 'df1'
        res = 'sr1'
        layer_1_list = ['10','16', '24']
        layer_2_list = ['5', '8',  '12']

    if module == 'sm2':
        df2 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
        )
        df2 = df2.drop(['RGIId', 'region', 'Centroid Distance'], axis = 1)
        dataset = df2
        dataset.name = 'df2'
        res = 'sr2'
        layer_1_list = ['10','50', '64']
        layer_2_list = ['5', '28', '48']

    if module == 'sm3':
        df3 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 25
        )
        df3 = df3.drop(['RGIId', 'region', 'Centroid Distance'], axis = 1)
        dataset = df3
        dataset.name = 'df3'
        res = 'sr3'
        layer_1_list = ['10', '32', '45']
        layer_2_list = ['5',  '17', '28']

    if module == 'sm4':
        df4 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 75
        )
        df4 = df4.drop(['RGIId', 'region', 'Centroid Distance'], axis = 1)
        dataset = df4
        dataset.name = 'df4'
        res = 'sr4'
        layer_1_list = ['10', '47', '64']
        layer_2_list = ['5',  '21', '36']

    if module == 'sm5':
        df5 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
        )
        df5 = df5.drop(['RGIId', 'region', 'Centroid Distance'], axis = 1)
        df5['Zdelta'] = df5['Zmax'] - df5['Zmin']
        res = 'sr5'
        dataset = df5
        dataset.name = 'df5'
        layer_1_list = ['10','48', '64']
        layer_2_list = ['5', '32', '52']


    if module == 'sm6':
        df6 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 25
        )
        df6 = df6.drop(['RGIId', 'region', 'Centroid Distance'], axis = 1)
        df6['Zdelta'] = df6['Zmax'] - df6['Zmin']
        dataset = df6
        dataset.name = 'df6'
        res = 'sr6'
        layer_1_list = ['10', '32', '48']
        layer_2_list = ['5',  '18', '28']

    if module == 'sm7':
        df7 = data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 75
        )
        df7 = df7.drop(['RGIId', 'region', 'Centroid Distance'], axis = 1)
        df7['Zdelta'] = df7['Zmax'] - df7['Zmin']
        dataset = df7
        dataset.name = 'df7'
        res = 'sr7'
        layer_1_list = ['10', '42', '64']
        layer_2_list = ['5',  '26', '40']

        
    if module == 'sm8':
        df8 = data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 25,
            data_version = 'v2'
        )
        df8 = df8.drop(['RGIId', 'region', 'Centroid Distance'], axis = 1)
        df8['Zdelta'] = df8['Zmax'] - df8['Zmin']
        dataset = df8
        dataset.name = 'df8'
        res = 'sr8'
        

        
        
    return module, dataset, dataset.name, res



'''
RGI_loader


'''
def RGI_loader(
    pth = '/data/fast1/glacierml/data/RGI/rgi60-attribs/', 
    region_selection = 'all'
):
    if len(str(region_selection)) == 1:
        N = 1
        region_selection = str(region_selection).zfill(N + len(str(region_selection)))
    else:
        region_selection = region_selection
        
    RGI_extra = pd.DataFrame()
    for file in (os.listdir(pth)):
        
        region_number = file[:2]
        if str(region_selection) == 'all':
            file_reader = pd.read_csv(pth + file, encoding_errors = 'replace', on_bad_lines = 'skip')
            RGI_extra = pd.concat([RGI_extra,file_reader], ignore_index = True)
            
        elif str(region_selection) != str(region_number):
            pass
        
        elif str(region_selection) == str(region_number):
            file_reader = pd.read_csv(pth + file, encoding_errors = 'replace', on_bad_lines = 'skip')
            RGI_extra = pd.concat([RGI_extra,file_reader], ignore_index = True)
            
    RGI = RGI_extra[[
        'RGIId',
        'CenLat',
        'CenLon',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Area',
        'Aspect',
        'Lmax',
#         'Name',
#         'GLIMSId',
    ]]
    RGI['region'] = RGI['RGIId'].str[6:8]
    
    for region_number in (range(1,20,1)):
        if len(str(region_number)) == 1:
            N = 1
            region_number = str(region_number).zfill(N + len(str(region_number)))
        else:
            region_number == str(region_number)


        if region_number != 19:
            drops = RGI[
                ((RGI['region'] == str(region_number)) & (RGI['Zmin'] < 0)) |
                ((RGI['region'] == str(region_number)) & (RGI['Zmed'] < 0)) |
                ((RGI['region'] == str(region_number)) & (RGI['Zmax'] < 0)) |
                ((RGI['region'] == str(region_number)) & (RGI['Slope'] < 0)) |
                ((RGI['region'] == str(region_number)) & (RGI['Aspect'] < 0))
            ].index

            if not drops.empty:
                RGI = RGI.drop(drops)
    return RGI

def data_loader(
    root_dir = '/data/fast1/glacierml/data/',
    RGI_input = 'y',
    scale = 'g',
    region_selection = 1,
    area_scrubber = 'off',
    anomaly_input = 0.5,
    data_version = 'v1'
):        
    
    pth_1 = root_dir + 'T_data/'
    pth_2 = root_dir + 'RGI/rgi60-attribs/'
    pth_3 = root_dir + 'matched_indexes/' + data_version + '/'
    pth_4 = root_dir + 'regional_data/training_data/' + data_version + '/'
    
    
    # load glacier GlaThiDa data v1
    if data_version == 'v1':
        glacier = pd.read_csv(pth_1 + 'glacier.csv', low_memory = False)    
        glacier = glacier.rename(columns = {
            'lat':'Lat',
            'lon':'Lon',
            'area':'area_g',
            'mean_slope':'Mean Slope',
            'mean_thickness':'Thickness'
        })   
        glacier = glacier.dropna(subset = ['Thickness'])
        print('# of raw thicknesses: ' + str(len(glacier)))
                                 
                                 
    # load glacier GlaThiDa data v2
    if data_version == 'v2':
        glacier = pd.read_csv(pth_1 + 'T.csv', low_memory = False)    
        glacier = glacier.rename(columns = {
            'LAT':'Lat',
            'LON':'Lon',
            'AREA':'area_g',
            'MEAN_SLOPE':'Mean Slope',
            'MEAN_THICKNESS':'Thickness'
        })   
        glacier = glacier.dropna(subset = ['Thickness'])

        print('# of raw thicknesses: ' + str(len(glacier)))
        
        
    # keep it just GlaThiDa
    if RGI_input == 'n':
        df = glacier.rename(columns = {
            'Mean Slope':'Slope'
        }, inplace = True)
        df = glacier[[
            'Lat',
            'Lon',
            'area_g',
            'Slope',
            'Thickness'
        ]]
        df = df.rename(columns = {
            'area_g':'Area'
        })
#         df = df.dropna()        
        return df

    
    # add in RGI attributes
    elif RGI_input == 'y':
        RGI_extra = pd.DataFrame()
        for file in os.listdir(pth_2):
            file_reader = pd.read_csv(
                pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip'
            )            
            RGI_extra = pd.concat([RGI_extra, file_reader], ignore_index=True)
            RGI = RGI_extra
        
        
        # read csv of combined GlaThiDa and RGI indexes, matched glacier for glacier
        comb = pd.read_csv(
                pth_3 + 'GlaThiDa_RGI_matched_indexes_' + data_version + '.csv'
        )
        
        idx = comb[comb['GlaThiDa_index'] == 203].index
        comb = comb.drop(idx)
        # force indexes to be integers rather than floats, and drop duplicates
        comb['GlaThiDa_index'] = comb['GlaThiDa_index'].astype(int)
        comb['RGI_index'] = comb['RGI_index'].astype(int)
        
        rgi_matches = len(comb['RGI_index'])
        rgi_matches_unique = len(comb['RGI_index'].unique())
        
        print(f'# of raw thickness matched to RGI = {rgi_matches}, {rgi_matches_unique} unique')

        comb = comb.drop_duplicates(subset = 'RGI_index', keep = 'last')
        # locate data in both datasets and line them up
        glacier = glacier.loc[comb['GlaThiDa_index']]
        RGI = RGI.loc[comb['RGI_index']]
        # reset indexes for merge
        glacier = glacier.reset_index()
        RGI = RGI.reset_index()
        RGI['region'] = RGI['RGIId'].str[6:8]
        # rename RGI area to differentiate from glathida area.
        # important for area scrubbing
        # don't forget to change the name back from area_r to Area when exiting function with df
        RGI = RGI.rename(columns = {
            'Area':'area_r'
        })
        
        # GlaThiDa and RGI are lined up, just stick them together and keep both left and right idx
        df = pd.merge(
            RGI, 
            glacier,
            left_index = True,
            right_index = True
        )
        df['Centroid Distance'] = np.nan
        for i in df.index:
            df['Centroid Distance'].loc[i] = geopy.distance.geodesic(
                (RGI['CenLat'].loc[i], RGI['CenLon'].loc[i]),
                (glacier['Lat'].loc[i], glacier['Lon'].loc[i])
            ).km
        
        
#         for i in tqdm(glathida.index):
#             #obtain lat and lon from glathida 
#             glathida_ll = (glathida.loc[i].LAT,glathida.loc[i].LON)

#             # find distance between selected glathida glacier and all RGI
#             distances = RGI.apply(
#                 lambda row: geopy.distance.geodesic((row.CenLat,row.CenLon),glathida_ll),
#                 axis = 1
#             )
        
        
#         df = df.rename(columns = {
#             'name':'name_g',
#             'Name':'name_r',

#             'BgnDate':'date_r',
#             'date':'date_g'
#         })

        # make a temp df for the duplicated entries
        
        # calculate the difference in size as a percentage
        df['size difference'] = abs(
            ( (df['area_g'] - df['area_r']) / df['area_g'] ) * 100
        )                
        
        # go by unique glacier. If more than one shows up, keep the min size difference.
        while len(df) > len(df['RGIId'].unique()):
            for glacier in df['RGIId'].unique():
                if len(df[df['RGIId'] == glacier]) > 1:
                    idx = df[df['RGIId'] == glacier]['size difference'].idxmax()
                    df = df.drop(idx)
                elif len(df[df['RGIId'] == glacier]) == 1:
                    pass
        
               
        
        df = df[[
            'RGIId',
            'CenLat',
            'CenLon',
#             'Lat',
#             'Lon',
            'area_r',
            'Zmin',
            'Zmed',
            'Zmax',
            'Slope',
            'Aspect',
            'Lmax',
            'Thickness',
            'area_g',
            'region',
            'index_x',
            'index_y',
            'Centroid Distance'
        ]]
        
        # archive error right here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # somehow drops non null df entries. Unknown myster bug located here
#         df = df.dropna()
        
        # global scale 
        if scale == 'g':
            df = df
            
            # finds anomalies between RGI and GlaThiDa areas. 
            # if anomaly > 1, drop data
            if area_scrubber == 'on':
                df = df.rename(columns = {
                    'name':'name_g',
                    'Name':'name_r',
                    
                    'BgnDate':'date_r',
                    'date':'date_g'
                })
                df['size_anomaly'] = abs(
                    ( (df['area_g'] - df['area_r']) / df['area_g'] ) * 100
                )                
                df = df[df['size_anomaly'] <= anomaly_input]
                df = df.drop([
                    'size_anomaly',
                    'area_g'
                ], axis = 1)
                df = df.rename(columns = {
                    'area_r':'Area'
                })
                df = df.drop(df.loc[df['Zmed']<0].index)
                df = df.drop(df.loc[df['Lmax']<0].index)
                df = df.drop(df.loc[df['Slope']<0].index)
#                 df = df.drop(df.loc[df['Aspect']<0].index)
                df = df.reset_index()
                df = df.drop('index', axis=1)
                df = df[[
                    'RGIId',
#                     'Lat',
#                     'Lon',
                    'CenLat',
                    'CenLon',
                    'Slope',
                    'Zmin',
                    'Zmed',
                    'Zmax',
                    'Area',
                    'Aspect',
                    'Lmax',
                    'Thickness',
                    'region',
                    'Centroid Distance'
                ]]
                
            elif area_scrubber == 'off':
                df = df.drop([
                    'area_g', 
#                     'RGIId'
                             ], axis = 1)
                df = df.rename(columns = {
                    'area_r':'Area'
                })
                return df
            
            
        # regional scale
        elif scale == 'r':
            # create temp df to hold regional data
            r_df = pd.DataFrame()
            
            # sort through regional data previously sorted and cleaned
            for file in os.listdir(pth_4):
                f = pd.read_csv(pth_4 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
                r_df = pd.concat([r_df, f], ignore_index = True)
                r_df = r_df.drop_duplicates(subset = ['CenLon','CenLat'], keep = 'last')
                r_df = r_df[[
                #     'GlaThiDa_index',
                #     'RGI_index',
                    'RGIId',
                    'region',
                #     'geographic region',
                    'CenLat',
                    'CenLon',
                    'Area',
                    'Zmin',
                    'Zmed',
                    'Zmax',
                    'Slope',
                    'Aspect',
                    'Lmax'
                ]]
                
            
            r_df = r_df.rename(columns = {
                'Area':'area_r'
            })
            # select only data for specific region
            r_df = r_df[r_df['region'] == region_selection]   
            df = df[[
#                 'Lat',
#                 'Lon',
                'area_g',
                'Thickness',
                'RGIId'
            ]]
            
            # merge df and temp df on RGIId to get regional data
            df = pd.merge(
                df, 
                r_df, 
#                 left_index = True,
#                 right_index = True,
                how = 'inner',
                on = 'RGIId'
            )
            
            if area_scrubber == 'on':
                df = df.rename(columns = {
                    'name':'name_g',
                    'Name':'name_r',
                    'Area':'area_r',
                    'BgnDate':'date_r',
                    'date':'date_g'
                })
                df['size_anomaly'] = abs(
                    ( (df['area_g'] - df['area_r']) / df['area_g'] ) * 100
                )
                df = df[df['size_anomaly'] <= anomaly_input]
                df = df.drop([
                    'size_anomaly',
                    'area_g'
                ], axis = 1)
                df = df.rename(columns = {
                    'area_r':'Area'
                })
                df = df[[
#                     'RGIId',
#                     'Lat',
#                     'Lon',
                    'CenLat',
                    'CenLon',
                    'Slope',
                    'Zmin',
                    'Zmed',
                    'Zmax',
                    'Area',
                    'Aspect',
                    'Lmax',
                    'Thickness',
                    'region',
#                     'Centroid Distance'
                ]]
                return df
                
            elif area_scrubber == 'off':
                df = df[[
                    'RGIId'
#                     'Lat',
#                     'Lon',
                    'CenLat',
                    'CenLon',
                    'Slope',
                    'Zmin',
                    'Zmed',
                    'Zmax',
                    'area_r',
                    'Aspect',
                    'Lmax',
                    'Thickness',
                    'region',
                    'Centroid Distance'
                ]]
                df = df.rename(columns = {'area_r':'Area'})
                return df



            
    for region_number in (range(1,20,1)):
        if len(str(region_number)) == 1:
            N = 1
            region_number = str(region_number).zfill(N + len(str(region_number)))
        else:
            region_number == str(region_number)


        if region_number != 19:
            drops = df[
                ((df['region'] == str(region_number)) & (df['Zmin'] < 0)) |
                ((df['region'] == str(region_number)) & (df['Zmed'] < 0)) |
                ((df['region'] == str(region_number)) & (df['Zmax'] < 0)) |
                ((df['region'] == str(region_number)) & (df['Slope'] < 0)) |
                ((df['region'] == str(region_number)) & (df['Aspect'] < 0))
            ].index

            if not drops.empty:
                df = df.drop(drops)
    return df



'''
GlaThiDa_RGI_index_matcher:
'''
def GlaThiDa_RGI_index_matcher(
#     pth_1 = '/data/fast1/glacierml/data/T_data/',
#     pth_2 = '/data/fast1/glacierml/data/RGI/rgi60-attribs/',
#     pth_3 = '/data/fast1/glacierml/data/matched_indexes/'

    pth_1 = '/home/prethicktor/data/T_data/',
    pth_2 = '/home/prethicktor/data/RGI/rgi60-attribs/',
    pth_3 = '/home/prethicktor/data/matched_indexes/v2/'
):
    glathida = pd.read_csv(pth_1 + 'T.csv')
    glathida = glathida.dropna(subset = ['MEAN_THICKNESS'])

    RGI = pd.DataFrame()
    for file in os.listdir(pth_2):
        print(file)
        file_reader = pd.read_csv(pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI = pd.concat([RGI, file_reader], ignore_index = True)
    RGI = RGI.reset_index()
    df = pd.DataFrame(columns = ['GlaThiDa_index', 'RGI_index', 'Centroid Distance'])
    #iterate over each glathida index
    for i in tqdm(glathida.index):
        #obtain lat and lon from glathida 
        glathida_ll = (glathida.loc[i].LAT,glathida.loc[i].LON)
        
        # find distance between selected glathida glacier and all RGI
        distances = RGI.apply(
            lambda row: geopy.distance.geodesic((row.CenLat,row.CenLon),glathida_ll),
            axis = 1
        )
        
        # find index of minimum distance between glathida and RGI glacier
        RGI_index = np.argmin(distances)
        RGI_match = RGI.loc[RGI_index]
        
        # concatonate two rows and append to dataframe with indexes for both glathida and RGI
        temp_df = pd.concat([RGI_match, glathida.loc[i]], axis = 0)
        df = df.append(temp_df, ignore_index = True)
    #     df = df.append(GlaThiDa_and_RGI, ignore_index = True)
        df['GlaThiDa_index'].iloc[-1] = i
        df['RGI_index'].iloc[-1] = RGI_index
        df['Centroid Distance'].iloc[-1] = np.min(distances)


    df.to_csv(pth_3 + 'GlaThiDa_RGI_matched_indexes_v2.csv')
        
        
'''
data_splitter
input = name of dataframe and selected random state.
output = dataframe and series randomly selected and populated as either training or test features or labels
'''
# Randomly selects data from a df for a given random state (usually iterated over a range of 25)
# Necessary variables for training and predictions
def data_splitter(df, random_state = 0):
    train_dataset = df.sample(frac=0.8, random_state=random_state)
    test_dataset = df.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    #define label - attribute training to be picked
    train_labels = train_features.pop('Thickness')
    test_labels = test_features.pop('Thickness')
    
    return train_features, test_features, train_labels, test_labels


'''
prethicktor_inputs
input = none
output = hyperparameters and layer architecture for DNN model
'''
# designed to provide a CLI to the model for each run rather modifying code
def prethicktor_inputs():
    print('')
    print('Please set neurons for first layer')
    layer_1_input = input()
    
    print('')
    print('Please set neurons for second layer')
    layer_2_input = input()
    
    print('')
    print('Please set learning rate: 0.1, 0.01, 0.001')
    lr_list = ('0.1, 0.01, 0.001')
    lr_input = input()
    while lr_input not in lr_list:
        print('Please set valid learning rate: 0.1, 0.01, 0.001')
        lr_input = input()
        
    print('')
    print('Please set epochs')
    ep_input = int(input())
    while type(ep_input) != int:
        print('Please input an integer for epochs')
        ep_input = input()
    

    
    return layer_1_input, layer_2_input, lr_input, ep_input




'''
build_linear_model
input = normalized data and desired learning rate
output = linear regression model
'''
# No longer used
def build_linear_model(normalizer,learning_rate=0.1):
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_absolute_error')
    
    return model



'''
build_dnn_model
input = normalized data and selected learning rate
output = dnn model with desired layer architecture, ready to be trained.
'''
def build_dnn_model(
    norm, learning_rate=0.1, layer_1 = 10, layer_2 = 5, dropout = True
):
    
    if dropout == True:
        model = keras.Sequential(
            [
                  norm,
                  layers.Dense(layer_1, activation='relu'),
                  layers.Dropout(rate = 0.1, seed = 0),
                  layers.Dense(layer_2, activation='relu'),

                  layers.Dense(1) 
            ]
        )

        model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate))
        
    else:
        model = keras.Sequential(
            [
                  norm,
                  layers.Dense(layer_1, activation='relu'),
                  layers.Dense(layer_2, activation='relu'),

                  layers.Dense(1) 
            ]
        )

        model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate))
        
    
    return model



'''
plot_loss
input = desired test results
output = loss plots for desired model
'''
def plot_loss(history):
#     plt.subplots(figsize=(10,5))
    plt.plot(
        history['loss'], 
         label='loss',
        color = 'blue'
    )
    plt.plot(
        history['val_loss'], 
        label='val_loss',
        color = 'orange'
    )
    #   plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    

     
'''
build_and_train_model
input = dataset, desired: learning rate, validation split, epochs, random state. module and res are defined as inputs when run and determine where data is saved.
output = saved weights for trained model and model results saved as a csv
'''

def build_and_train_model(dataset,
                          learning_rate = 0.001,
                          validation_split = 0.2,
                          epochs = 100,
                          random_state = 0,
                          module = 'sm',
                          res = 'sr',
                          layer_1 = 10,
                          layer_2 = 5,
                          dropout = True,
                          verbose = False
                         ):
    # define paths
    arch = str(layer_1) + '-' + str(layer_2)
    svd_mod_pth = 'saved_models/' + module + '/sm_' + arch + '/'
    svd_res_pth = 'saved_results/' + res + '/sr_' + arch + '/'

    # code snippet to make folders for saved models and results if they do not already exist
    isdir = os.path.isdir(svd_mod_pth)

    if isdir == False:
        os.makedirs(svd_mod_pth)

    isdir = os.path.isdir(svd_res_pth)
    if isdir == False:
        os.makedirs(svd_res_pth)


    if dropout == True:
        dropout = '1'
    elif dropout == False:
        dropout = '0'


#     split data
    (
        train_features, test_features, train_labels, test_labels
    ) = data_splitter(dataset)
#         print(dataset.name)

#     normalize data
#         print('Normalizing ' + str(dataset.name) + ' data')
    normalizer = {}
    variable_list = list(train_features)
    for variable_name in variable_list:
        normalizer[variable_name] = preprocessing.Normalization(input_shape=[1,], axis=None)
        normalizer[variable_name].adapt(np.array(train_features[variable_name]))

    normalizer['ALL'] = preprocessing.Normalization(axis=-1)
    normalizer['ALL'].adapt(np.array(train_features))

#      DNN model
    dnn_model = {}
    dnn_history = {}
    dnn_results = {}

    # set up model with  normalized data and defined layer architecture
    dnn_model = build_dnn_model(normalizer['ALL'], learning_rate, layer_1, layer_2, dropout)
    
    # set up callback function to cut off training when performance reaches peak
    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.001,
        patience = 10,
        verbose = 0,
        mode = 'auto',
        baseline = None,
        restore_best_weights = True
    )
    
    # train model on previously selected and splitdata
    dnn_history['MULTI'] = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=validation_split,
        callbacks = [callback],
        verbose=0, 
        epochs=epochs
    )

    #save model, results, and history



    df = pd.DataFrame(dnn_history['MULTI'].history)

    
    history_filename = (
        svd_res_pth +
       str(dataset.name) +
       '_' +
       dropout +
       '_dnn_history_MULTI_' +
       str(learning_rate) +
       '_' +
       str(validation_split) +
       '_' +
       str(epochs) +
       '_' +
       str(random_state)
    )

    df.to_csv(  history_filename  )

    model_filename =  (
        svd_mod_pth + 
        str(dataset.name) + 
        '_' +
        dropout +
        '_dnn_MULTI_' + 
        str(learning_rate) + 
        '_' + 
        str(validation_split) + 
        '_' + 
        str(epochs) + 
        '_' + 
        str(random_state)
    )
    
    dnn_model.save(  model_filename  )
    
    return history_filename, model_filename
    

    
'''

'''
def predictions_maker(
    rs,
    dropout,
    arch,
    dataset,
    folder,
    model_loc,
    model_name,
):
    df = pd.DataFrame()
    dnn_model = {}
    
    (
        train_features, test_features, train_labels, test_labels
    ) = data_splitter(
        dataset, random_state = rs
    )
    dnn_model[model_name] = tf.keras.models.load_model(model_loc)
    
    mae_test = dnn_model[model_name].evaluate(
                    test_features, test_labels, verbose=0
                )
    mae_train = dnn_model[model_name].evaluate(
        train_features, train_labels, verbose=0
    )

    pred_train = dnn_model[model_name].predict(
        train_features, verbose=0
    )

    pred_test = dnn_model[model_name].predict(
        test_features, verbose=0
    )
    
    train_thickness = pd.Series(pred_train.flatten(), name = 'thickness')
    train_features = train_features.reset_index()
    train_features = train_features.drop('index', axis = 1)
    dft = pd.concat([train_features, train_thickness], axis = 1)
    dft['vol'] = dft['thickness'] * (dft['Area'] * 1e6)
    avg_train_thickness = sum(dft['vol']) / sum(dft['Area'] * 1e6)
    
    avg_thickness = pd.Series(
        np.mean(pred_train), name = 'avg train thickness'
    )

    test_thickness = pd.Series(pred_test.flatten(), name = 'thickness')
    test_features = test_features.reset_index()
    test_features = test_features.drop('index', axis = 1)
    dft = pd.concat([test_features, test_thickness], axis = 1)
    dft['vol'] = dft['thickness'] * (dft['Area'] * 1e6)
    avg_test_thickness = sum(dft['vol']) / sum(dft['Area'] * 1e6)
    
    
    avg_test_thickness = pd.Series(
        np.mean(pred_test),  name = 'avg test thickness'
    )

    temp_df = pd.merge(
        avg_thickness, avg_test_thickness, right_index=True, left_index=True
    )

    df = pd.concat(
        [df, temp_df], ignore_index = True
    )

    df.loc[df.index[-1], 'model'] = folder
    df.loc[df.index[-1], 'test mae'] = mae_test
    df.loc[df.index[-1], 'train mae'] = mae_train
    df.loc[df.index[-1], 'architecture'] = arch[3:]
    df.loc[df.index[-1], 'validation split'] = '0.2'
    df.loc[df.index[-1], 'coregistration'] = dataset.name
    df.loc[df.index[-1], 'dropout'] = dropout
    df.loc[df.index[-1], 'total parameters'] = dnn_model[model_name].count_params() 
    if '0.1' in folder:
        df.loc[df.index[-1], 'learning rate'] = '0.1'
    if '0.01' in folder:
        df.loc[df.index[-1], 'learning rate'] = '0.01'
    if '0.001' in folder:
        df.loc[df.index[-1], 'learning rate']= '0.001'
    if '100' in folder:
        df.loc[df.index[-1], 'epochs']= '100'
    if '999' in folder:
        df.loc[df.index[-1], 'epochs']= '999'
    if '2000' in folder:
        df.loc[df.index[-1], 'epochs']= '2000'

    return df
    

def deviations_calculator(
    model_loc,
    model_name,
    ep,
    arch, 
    lr,
    dropout,
    dataframe,
    dataset,
    dfsrq
):
    dnn_model = {}
    df = pd.DataFrame()
    test_mae_mean = np.mean(dfsrq['test mae'])
    test_mae_std_dev = np.std(dfsrq['test mae'])

    # find mean and std dev of train mae
    train_mae_mean = np.mean(dfsrq['train mae'])
    train_mae_std_dev = np.std(dfsrq['train mae'])

    # find mean and std dev of predictions made based on training data
    train_thickness_mean = np.mean(dfsrq['avg train thickness']) 
    train_thickness_std_dev = np.std(dfsrq['avg train thickness'])

    # find mean and std dev of predictions made based on test data
    test_thickness_mean = np.mean(dfsrq['avg test thickness']) 
    test_thickness_std_dev = np.std(dfsrq['avg test thickness'])

    # put something in a series that can be appended to a df
    s = pd.Series(train_thickness_mean)

    df = pd.concat(
        [df, s], ignore_index=True
    )

    # begin populating deviations table
    df.loc[
        df.index[-1], 'layer architecture'
    ] = arch  

    dnn_model[model_name] = tf.keras.models.load_model(model_loc)
    
    df.loc[
        df.index[-1], 'total parameters'
    ] = dnn_model[model_name].count_params() 

    df.loc[
        df.index[-1], 'trained parameters'
    ] = df.loc[
        df.index[-1], 'total parameters'
    ] - (len(dataset.columns) + (len(dataset.columns) - 1))

    df.loc[
        df.index[-1], 'total inputs'
    ] = (len(dataset) * (len(dataset.columns) -1))

    df.loc[
        df.index[-1], 'df'
    ] = dataframe

    df.loc[
        df.index[-1], 'dropout'
    ] = dropout

    df.loc[
        df.index[-1], 'learning rate'
    ] = lr

    df.loc[
        df.index[-1], 'validation split'
    ]= 0.2

    df.loc[
        df.index[-1], 'epochs'
    ] = ep

    df.loc[
        df.index[-1], 'test mae avg'
    ] = test_mae_mean

    df.loc[df.index[-1], 'train mae avg'] = train_mae_mean

    df.loc[df.index[-1], 'test mae std dev'] = test_mae_std_dev

    df.loc[df.index[-1], 'train mae std dev'] = train_mae_std_dev

    df.loc[
        df.index[-1], 'test predicted thickness std dev'
    ] = test_thickness_std_dev

    df.loc[
        df.index[-1], 'train predicted thickness std dev'
    ] = train_thickness_std_dev



    df.drop(columns = {0},inplace = True)    
    df = df.dropna()


    df = df.sort_values('test mae avg')
    df['epochs'] = df['epochs'].astype(int)
    
    return df
    
    
def random_state_finder(
    folder
):
    if folder.endswith('_0'):
        rs = 0
    if folder.endswith('_1'):
        rs = 1
    if folder.endswith('_2'):
        rs = 2
    if folder.endswith('_3'):
        rs = 3
    if folder.endswith('_4'):
        rs = 4
    if folder.endswith('_5'):
        rs = 5
    if folder.endswith('_6'):
        rs = 6
    if folder.endswith('_7'):
        rs = 7
    if folder.endswith('_8'):
        rs = 8
    if folder.endswith('_9'):
        rs = 9
    if folder.endswith('10'):
        rs = 10
    if folder.endswith('11'):
        rs = 11
    if folder.endswith('12'):
        rs = 12
    if folder.endswith('13'):
        rs = 13
    if folder.endswith('14'):
        rs = 14
    if folder.endswith('15'):
        rs = 15
    if folder.endswith('16'):
        rs = 16
    if folder.endswith('17'):
        rs = 17
    if folder.endswith('18'):
        rs = 18
    if folder.endswith('19'):
        rs = 19
    if folder.endswith('20'):
        rs = 20
    if folder.endswith('21'):
        rs = 21
    if folder.endswith('22'):
        rs = 22
    if folder.endswith('23'):
        rs = 23
    if folder.endswith('24'):
        rs = 24
        
    return rs
    
    
'''
predictions_loader
'''
def regional_predictions_loader(
    training_module,
    architecture,
    learning_rate,
    epochs,
):
    root_dir = 'zults/'
    RGI_predicted = pd.DataFrame()

    for file in tqdm(os.listdir(root_dir)):
        # print(file)
        if 'RGI_predicted' in file and training_module + '_' in file and architecture in file and learning_rate in file and epochs in file:
            file_reader = pd.read_csv(root_dir + file)
            file_reader['volume km3'] = (
                file_reader['avg predicted thickness'] / 1e3
            ) * file_reader['Area']
            file_reader = file_reader.dropna()
            print(file_reader)
            sum_volume = sum(file_reader['volume km3'])
            total_volume = pd.Series(sum_volume, name = 'total volume')
            RGI_predicted = pd.concat([RGI_predicted, total_volume], ignore_index = True)    
#             print(RGI_predicted)
            file_reader['variance'] = file_reader['predicted thickness std dev'] **2 
            variance = sum(file_reader['variance'])
            
            RGI_predicted.loc[
                RGI_predicted.index[-1], 'total variance'
            ] = np.sqrt(variance)/1e3

            area = sum(file_reader['Area'])

            RGI_predicted.loc[
                RGI_predicted.index[-1], 'area'
            ] = area
            
            for i in range(1,10,1):
                if ('df' + str(i) + '_1_') in file or ('df' + str(i) + '_0_'):
                    RGI_predicted.loc[
                            RGI_predicted.index[-1], 'volf'
                    ] = 158.17

                    RGI_predicted.loc[
                        RGI_predicted.index[-1], 'tolerance'
                    ] = 41.0

                    RGI_predicted.loc[
                        RGI_predicted.index[-1], 'h mean f'
                    ] = 224
            if '_01_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '01'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 18.98
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 4.92
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 218    
                
                
            if '_02_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '02'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 1.06
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.27
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 72
                
                
            if '_03_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '03'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 28.33
            
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 7.35
            
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 270
                
            if '_04_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '04'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 8.61
            
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 2.23
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 210
                
            if '_05_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '05'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 15.69
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 4.07
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 175
                
            if '_06_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '06'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] =  3.77
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] =  0.98 
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] =  341 
                
            if '_07_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '07'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] =  7.47 
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] =  1.94 
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] =  220 
                
            if '_08_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '08'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] =  0.30 
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] =  0.08 
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] =  101 
                
            if '_09_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '09'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 14.64
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 3.80
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 283
                
            if '_10_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '10'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 0.14
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.04
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 56
                
            if '_11_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '11'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 0.13
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.03
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 61   
                
            
            if '_12_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '12'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 0.06
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.02
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 48
                
            if '_13_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '13'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 3.27
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.85
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 66
                
            if '_14_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '14'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 2.87
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.74
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 85
            
            if '_15_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '15'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 0.88
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.23
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 59
                
            if '_16_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '16'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 0.10
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.03
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 42
            
            
            if '_17_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '17'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 5.34
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 1.39
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 181
                
                
            if '_18_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '18'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 0.07
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 0.02
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 63
            
            if '_19_' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'region'
                ] = '19'
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'volf'
                ] = 46.47
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'tolerance'
                ] = 12.06
                
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'h mean f'
                ] = 349
            
            
            
            if 'df1_' not in file:
                att_list = [
                    'CenLat',
                    'CenLon',
                    'Area',
                    'Aspect',
                    'Lmax',
                    'Slope',
                    'Zmin',
                    'Zmax',
                ]
                for att in att_list:
                    mean = file_reader[att].mean()
                    median = file_reader[att].median()
                    std = file_reader[att].std()
                    q3 = np.quantile(file_reader[att], 0.75)
                    q1 = np.quantile(file_reader[att], 0.25)
                    iqr = q3 - q1  

                    RGI_predicted.loc[
                        RGI_predicted.index[-1], att + '_RGI_mean'
                    ] = mean

                    RGI_predicted.loc[
                        RGI_predicted.index[-1], att + '_RGI_median'
                    ] = median

                    RGI_predicted.loc[
                        RGI_predicted.index[-1], att + '_RGI_std'
                    ] = std

                    RGI_predicted.loc[
                        RGI_predicted.index[-1], att + '_RGI_iqr'
                    ] = iqr
            str_1 = '_1_'
            str_2 = '-'
            str_3 = '_0_'
            str_4 = '_0.'
            str_7 = '.csv'
            str_8 = 'df'
            str_8_idx = file.index(str_8)
            str_7_idx = file.index(str_7)
            str_2_idx = file.index(str_2)
            str_4_idx = file.index(str_4)
            if str_1 in file:

                str_1_idx = file.index(str_1)
                layer_1_start = (str_1_idx + 3)
                layer_2_start = str_2_idx + 1
                layer_1_length = str_2_idx - layer_1_start
                layer_2_length = str_4_idx - (str_2_idx + 1)


            if str_3 in file :
                str_3_idx = file.index(str_3)

                layer_1_start = (str_3_idx + 3)
                layer_2_start = str_2_idx + 1
                layer_1_length = str_2_idx - layer_1_start
                layer_2_length = str_4_idx - (str_2_idx + 1)


            layer_1 = file[layer_1_start:(layer_1_start + layer_1_length)]
            layer_2 = file[layer_2_start:(layer_2_start + layer_2_length)]

            arch = (str(layer_1) + '-' + str(layer_2))
            RGI_predicted.loc[
                RGI_predicted.index[-1], 'architecture'
            ] = arch

            for i in range(1,11,1):
                 if ('df' + str(i) + '_') in file:
                    RGI_predicted.loc[
                        RGI_predicted.index[-1], 'dataframe'
                    ] = 'df' + str(i) + '_' + str(
                        RGI_predicted['region'].loc[RGI_predicted.index[-1]])
        


                
            if '0.1' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'learning rate']= '0.100'
            if '0.01' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'learning rate']= '0.010'
            if '0.001' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'learning rate']= '0.001'
            if '_100' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '100'           
            if '_999' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '999'
            if '_2000' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '2000'
    print(RGI_predicted)
    RGI_predicted = RGI_predicted.rename(columns = {
        0:'vol'
    })

#     RGI_predicted['vol'] = RGI_predicted['vol'] / 1e3

    RGI_predicted['mean thickness'] = (
        (RGI_predicted['vol'] * 1e3) / RGI_predicted['area']
    ) * 1e3
    RGI_predicted['voldiff'] = (RGI_predicted['vol']) - RGI_predicted['volf'] 
    RGI_predicted['vol_ratio'] = RGI_predicted['vol'] / RGI_predicted['volf']
    RGI_predicted = RGI_predicted.reset_index()
    RGI_predicted = RGI_predicted.drop('index', axis = 1)
#     RGI_predicted = RGI_predicted.sort_values([
# #         'mean thickness (km)',
#         'architecture',
#         'learning rate',
#         'dataframe'
#     ], ascending = True)
                        
    return RGI_predicted



def global_predictions_loader(
    coregistration,
    architecture,
    learning_rate,
    epochs,
):
    root_dir = 'zults/'
    RGI_predicted = pd.DataFrame()
    for file in (os.listdir(root_dir)):
            # print(file)
        if ('RGI_predicted' in file and 
            coregistration in file and 
            architecture in file and 
            learning_rate in file and 
            epochs in file):
#             print(file)
            file_reader = pd.read_csv(root_dir + file)
#             print(file_reader)
            file_reader['volume km3'] = (
                file_reader['avg predicted thickness'] / 1e3
            ) * file_reader['Area']
            file_reader = file_reader.dropna()
            RGI_predicted = pd.concat([RGI_predicted, file_reader], ignore_index = True)  

    RGI_predicted = RGI_predicted.drop('Unnamed: 0', axis = 1)
    RGI_predicted['dataframe' ] = 'df'+ coregistration

    return RGI_predicted


'''
'''
def glathida_stats_adder(
    df,
    pth_1 = '/data/fast1/glacierml/data/regional_data/raw/',
    pth_2 = '/data/fast1/glacierml/data/RGI/rgi60-attribs/',
    pth_3 = '/data/fast1/glacierml/data/regional_data/training_data/',
    
):
    # finish building df

    dfa = pd.DataFrame()
    for file in tqdm(os.listdir(pth_1)):
        dfb = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
        region_and_number = file[:-4]
        region_number = region_and_number[:2]
        region = region_and_number[3:]

        dfb['geographic region'] = region
        dfb['region'] = region_number
        dfa = dfa.append(dfb, ignore_index=True)

    dfa = dfa.reset_index()

    dfa = dfa[[
        'GlaThiDa_index',
        'RGI_index',
        'RGIId',
        'region',
        'geographic region'
    ]]
    RGI_extra = pd.DataFrame()
    for file in os.listdir(pth_2):
        f = pd.read_csv(pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI_extra = pd.concat([RGI_extra, f], ignore_index = True)

        region_and_number = file[:-4]
        region_number = region_and_number[:2]
        region = region_and_number[9:]
        dfc = dfa[dfa['region'] == region_number]

        for file in os.listdir(pth_3):
            print(file)
            if file[:2] == region_number:
                glathida_regional = pd.read_csv(pth_3 + file)

        GlaThiDa_mean_area = glathida_regional['Area'].mean()
        GlaThiDa_mean_aspect = glathida_regional['Aspect'].mean()
        GlaThiDa_mean_lmax = glathida_regional['Lmax'].mean()
        GlaThiDa_mean_slope = glathida_regional['Slope'].mean()
        GlaThiDa_mean_zmin = glathida_regional['Zmin'].mean()
        GlaThiDa_mean_zmax = glathida_regional['Zmax'].mean()

        GlaThiDa_median_area = glathida_regional['Area'].median()
        GlaThiDa_median_aspect = glathida_regional['Aspect'].median()
        GlaThiDa_median_lmax = glathida_regional['Lmax'].median()
        GlaThiDa_median_slope = glathida_regional['Slope'].median()
        GlaThiDa_median_zmin = glathida_regional['Zmin'].median()
        GlaThiDa_median_zmax = glathida_regional['Zmax'].median()

        GlaThiDa_std_area = glathida_regional['Area'].std(ddof=0)
        GlaThiDa_std_aspect = glathida_regional['Aspect'].std(ddof=0)
        GlaThiDa_std_lmax = glathida_regional['Lmax'].std(ddof=0)
        GlaThiDa_std_slope = glathida_regional['Slope'].std(ddof=0)
        GlaThiDa_std_zmin = glathida_regional['Zmin'].std(ddof=0)
        GlaThiDa_std_zmax = glathida_regional['Zmax'].std(ddof=0)

        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Area_GlaThiDa_mean'
        ] = GlaThiDa_mean_area
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Aspect_GlaThiDa_mean'
        ] = GlaThiDa_mean_aspect
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Lmax_GlaThiDa_mean'
        ] = GlaThiDa_mean_lmax
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Slope_GlaThiDa_mean'
        ] = GlaThiDa_mean_slope
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Zmin_GlaThiDa_mean'
        ] = GlaThiDa_mean_zmin
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Zmax_GlaThiDa_mean'
        ] = GlaThiDa_mean_zmax


        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Area_GlaThiDa_median'
        ] = GlaThiDa_median_area
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Aspect_GlaThiDa_median'
        ] = GlaThiDa_median_aspect
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Lmax_GlaThiDa_median'
        ] = GlaThiDa_median_lmax
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Slope_GlaThiDa_median'
        ] = GlaThiDa_median_slope
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Zmin_GlaThiDa_median'
        ] = GlaThiDa_median_zmin
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Zmax_GlaThiDa_median'
        ] = GlaThiDa_median_zmax




        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Area_GlaThiDa_std'
        ] = GlaThiDa_std_area
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Aspect_GlaThiDa_std'
        ] = GlaThiDa_std_aspect
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Lmax_GlaThiDa_std'
        ] = GlaThiDa_std_lmax
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Slope_GlaThiDa_std'
        ] = GlaThiDa_std_slope
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Zmin_GlaThiDa_std'
        ] = GlaThiDa_std_zmin
        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'Zmax_GlaThiDa_std'
        ] = GlaThiDa_std_zmax






        trainable_ratio = (len(dfc) / len(f))
        percent_trainable = trainable_ratio * 100

        df.loc[
            df[df['dataframe'].str[4:] == region_number].index, 'ratio trainable'
        ] = trainable_ratio

#     df['vol_ratio'] = df['vol'] / df['volf']
#     df['vol_from_zero'] = abs(1 - df['vol_ratio'])

    return df






'''
'''
def predictions_finder(
    coregistration = 'df8'
):
    root_dir = 'zults/'
    prethicked = pd.DataFrame()
    for file in tqdm(os.listdir(root_dir)):
    # print(file)
        if 'RGI_predicted' in file and coregistration in file:
            file_reader = pd.read_csv(root_dir + file)
            file_reader = file_reader.rename(columns = {
                0:'vol'
            })

            file_reader['volume km3'] = (
                file_reader['avg predicted thickness'] / 1e3
            ) * file_reader['Area']
            
            file_reader['pred std dev'] = (
                (file_reader['predicted thickness std dev'] / 1e3) * file_reader['Area']
            )
#             print(file)
            str_1 = '_1_'
            str_2 = '-'
            str_3 = '_0_'
            str_4 = '_0.'
            str_7 = '.csv'
            str_8 = 'df'
            str_8_idx = file.index(str_8)
            str_7_idx = file.index(str_7)
            str_2_idx = file.index(str_2)
            str_4_idx = file.index(str_4)
            if str_1 in file:

                str_1_idx = file.index(str_1)
                layer_1_start = (str_1_idx + 3)
                layer_2_start = str_2_idx + 1
                layer_1_length = str_2_idx - layer_1_start
                layer_2_length = str_4_idx - (str_2_idx + 1)


            if str_3 in file :
                str_3_idx = file.index(str_3)

                layer_1_start = (str_3_idx + 3)
                layer_2_start = str_2_idx + 1
                layer_1_length = str_2_idx - layer_1_start
                layer_2_length = str_4_idx - (str_2_idx + 1)


            layer_1 = file[layer_1_start:(layer_1_start + layer_1_length)]
            layer_2 = file[layer_2_start:(layer_2_start + layer_2_length)]

            arch = pd.Series(str(layer_1) + '-' + str(layer_2), name = 'architecture')
            prethicked = pd.concat([prethicked, arch])

            # epochs = 100
            if file[str_7_idx - 3] == str(1) or file[str_7_idx - 3] == str(9):

                learning_rate = file[
                    layer_2_start + layer_2_length + 1 : str_7_idx - 5
                ]
                epochs = file[
                    str_7_idx - 4 : str_7_idx
                ]
                
            # epochs = 2000
            if file[str_7_idx - 4] == str(2) or file[str_7_idx - 3] == str(9):

                learning_rate = file[
                    layer_2_start + layer_2_length + 1 : str_7_idx - 5
                ]
                epochs = file[
                    str_7_idx - 4 : str_7_idx
                ]

                    # epochs < 100
            elif file[str_7_idx - 3] == '_':

                learning_rate = file[
                    layer_2_start + layer_2_length + 1 : str_7_idx - 3
                ]

                epochs = file[
                    str_7_idx - 2 : str_7_idx
                ]
            prethicked = prethicked.reset_index()
            prethicked = prethicked.drop('index', axis = 1)
            prethicked.loc[prethicked.index[-1], 'learning rate'] = learning_rate
            prethicked.loc[prethicked.index[-1], 'epochs'] = epochs
            prethicked.loc[prethicked.index[-1], 'volume'] = sum(file_reader['volume km3'])
            prethicked.loc[prethicked.index[-1], 'std dev'] = sum(
                file_reader['pred std dev']
            )
            if file[str_8_idx + 3] == '_':
                prethicked.loc[prethicked.index[-1], 'coregistration'] = file[
                    str_8_idx : str_8_idx + 3]                
            elif file[str_8_idx + 3] !='_':
                prethicked.loc[prethicked.index[-1], 'coregistration'] = (
                    file[str_8_idx + 2] + file[str_8_idx + 3]
                )
            predicted = pd.DataFrame()
            
#             break
    prethicked = prethicked.rename(columns = {
        0:'architecture'
    })
    for arch in prethicked['architecture'].unique():
        for lr in prethicked['learning rate'].unique():
            dft = prethicked[
                (prethicked['architecture'] == arch) & 
                (prethicked['learning rate'] == lr)
            ]
            dft['predicted volume'] = sum(dft['volume']) / 1e3
            dft['std dev'] = sum(dft['std dev']) / 1e3
    #         print(dft.iloc[-1])
            predicted = pd.concat([predicted,dft],ignore_index = True)
    predicted = predicted[[
        'architecture',
#         'epochs',
        'learning rate',
        'coregistration',
        'predicted volume',
        'std dev'
    ]]
    predicted = predicted.drop_duplicates()
    prethicked = prethicked.drop_duplicates()
    return predicted


    

'''
cluster functions
'''
# define functions

def silhouette_plot(X, model, ax, colors):
    y_lower = 10
    y_tick_pos_ = []
    sh_samples = silhouette_samples(X, model.labels_)
    sh_score = silhouette_score(X, model.labels_)
    
    for idx in range(model.n_clusters):
        values = sh_samples[model.labels_ == idx]
        values.sort()
        size = values.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper),0,values,
                         facecolor=colors[idx],edgecolor=colors[idx]
        )
        y_tick_pos_.append(y_lower + 0.5 * size)
        y_lower = y_upper + 10

    ax.axvline(x=sh_score, color="red", linestyle="--", label="Avg Silhouette Score")
    ax.set_title("Silhouette Plot for {} clusters".format(model.n_clusters))
    l_xlim = max(-1, min(-0.1, round(min(sh_samples) - 0.1, 1)))
    u_xlim = min(1, round(max(sh_samples) + 0.1, 1))
    ax.set_xlim([l_xlim, u_xlim])
    ax.set_ylim([0, X.shape[0] + (model.n_clusters + 1) * 10])
    ax.set_xlabel("silhouette coefficient values")
    ax.set_ylabel("cluster label")
    ax.set_yticks(y_tick_pos_)
    ax.set_yticklabels(str(idx) for idx in range(model.n_clusters))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.legend(loc="best")
    return ax



def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb




def cluster_comparison_bar(df_std, RGI_comparison, colors, deviation=True ,title="Cluster results"):
    
    features = RGI_comparison.index
    ncols = 3
    # calculate number of rows
    nrows = len(features) // ncols + (len(features) % ncols > 0)
    # set figure size
    fig = plt.figure(figsize=(15,15), dpi=200)
    #interate through every feature
    for n, feature in enumerate(features):
        # create chart

#     plt.show()    
        ax = plt.subplot(nrows, ncols, n + 1)
        RGI_comparison[RGI_comparison.index==feature].plot(
            kind='bar', 
            ax=ax, 
            title=feature,
            color=colors[0:df_std['cluster'].nunique()],
            legend=False
                                                            )
        plt.axhline(y=0)
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)

    c_labels = RGI_comparison.columns.to_list()
    c_colors = colors[0:3]
    mpats = [mpatches.Patch(color=c, label=l) for c,l in list(zip(
        colors[0:df_std['cluster'].nunique()],
        RGI_comparison.columns.to_list()
    ))]

    fig.legend(handles=mpats,
               ncol=ncols,
               loc="upper center",
               fancybox=True,
               bbox_to_anchor=(0.5, 0.98)
              )
    axes = fig.get_axes()
    
    fig.suptitle(title, fontsize=18, y=1)
    fig.supylabel('Deviation from overall mean in %')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    
    
    
class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]

        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0/self.n)
        
        self.axes = [
            figure.add_axes(
                rect, projection='polar', label='axes%d' % i
            ) for i in range(self.n)
        ]
        
        self.ax = self.axes[0]
        self.ax.set_thetagrids(
            self.angles, 
            labels=title, 
            fontsize=14, 
            backgroundcolor="white",
            zorder=999
        ) 
        # Feature names
        self.ax.set_yticklabels([])
#         self.ax.set_xscale('log')
#         self.ax.set_yscale('log')
        for ax in self.axes[1:]:
            ax.xaxis.set_visible(False)
            ax.set_yticklabels([])
            ax.set_zorder(-99)
            
        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.spines['polar'].set_color('black')
            ax.spines['polar'].set_zorder(-99)
                     
            ax.set_rscale('symlog')
#             ax.set_yscale('log')
    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)
        kw['label'] = '_noLabel'
        self.ax.fill(angle, values,*args,**kw)


def color_grabber(
    n_colors = 6,
    color_map = 'viridis'
):
    

    cluster_colors = px.colors.sample_colorscale(
        color_map, 
        [n/(n_colors -1) for n in range(n_colors)]
    )
    colors = pd.Series()
    for i in cluster_colors:
        color_1 = i[4:]
        color_2 = color_1[:-1]
    #     print(i[4:])
    #     print(i[:-1])
        numbers = color_2.split(',')
        rgb_1 = int(numbers[0])
        rgb_2 = int(numbers[1])
        rgb_3 = int(numbers[2])
        colors = colors.append(pd.Series('#' + rgb_to_hex((rgb_1, rgb_2, rgb_3))))
    #     print(color)
    #     colors = pd.concat([colors, color], ignore_index = True)
    # cluster_colors

    colors = colors.T
    colors = colors.reset_index()
    colors = colors.drop('index', axis = 1)
    colors = colors.squeeze()
    return colors




def notebook_data_loader():
    df = pd.read_csv(
            'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_coregistration_df8.csv'
        )
    df['region'] = df['RGIId'].str[6:8]


    RGI = RGI_loader()
    RGI = RGI[[
        'RGIId',
        'CenLat',
        'CenLon',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Area',
        'Aspect',
        'Lmax'
    ]]

    RGI['Zdelta'] = RGI['Zmax'] - RGI['Zmin']

    df = pd.merge(df, RGI, on = 'RGIId')

    df['Upper Bound'] = df['Upper Bound'] - df['Mean Thickness']
    df['Lower Bound'] = df['Mean Thickness'] - df['Lower Bound']
    df['UB'] = (df['Upper Bound'] / 1e3) * df['Area']
    df['LB'] = (df['Lower Bound'] / 1e3) * df['Area']

    upper_bound = np.round(
        sum(df['UB']) / 1e3, 2)

    lower_bound = np.round(
        sum(df['LB']) / 1e3 , 2) 

    volume = np.round(
        sum(df['Mean Thickness'] / 1e3 * df['Area']) / 1e3, 2)

    std = np.round(
        sum(df['Thickness Std Dev'] / 1e3 * df['Area']) / 1e3, 2)


    print(f'Global Volume: {volume}, UB: {upper_bound}, LB: {lower_bound}, STD: {std}')
    df['Edasi Volume'] = df['Mean Thickness'] / 1e3 * df['Area']
    df['Volume Std Dev'] = df['Thickness Std Dev'] / 1e3 * df['Area']
    
    ref = pd.read_csv('reference_thicknesses/farinotti_mean_thickness_rgi_id.csv')
    ref = ref[[
        'RGIId',
        'Farinotti Mean Thickness'
    ]]
    ref['region'] = ref['RGIId'].str[6:8]
    ref = ref.sort_values('RGIId')
    ref = ref.dropna()

    ref = pd.merge(ref, df, 
    #                left_index = True, right_index = True)
    on = [
        'RGIId'
    ])
    ref = ref.rename(columns = {
        'Mean Thickness':'Edasi Mean Thickness'
    })

    ref['Farinotti Volume'] = (ref['Farinotti Mean Thickness'] / 1e3 )* ref['Area']

    ref['region'] = ref['RGIId'].str[6:8]
    ref['Edasi Volume'] = (ref['Edasi Mean Thickness'] / 1e3) * ref['Area']
    ref['Volume Std Dev'] = (ref['Thickness Std Dev'] / 1e3 )* ref['Area']
    ref = ref.reset_index()
    ref = ref.drop('index', axis = 1)
    ref = ref.dropna()
    ref['VE / VF'] = ref['Edasi Mean Thickness'] / ref['Farinotti Mean Thickness']
    ref = ref.drop_duplicates()
    # sum(ref['volume km3'])

    ref['Upper Bound'] = ref['Upper Bound'] - ref['Edasi Mean Thickness']
    ref['Lower Bound'] = ref['Edasi Mean Thickness'] - ref['Lower Bound']
    ref

    ref['UB'] = (ref['Upper Bound'] / 1e3) * ref['Area']
    ref['LB'] = (ref['Lower Bound'] / 1e3) * ref['Area']

    return df, ref
