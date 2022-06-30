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
pd.set_option('mode.chained_assignment',None)

'''
RGI_loader


'''
def RGI_loader(pth = '/data/fast1/glacierml/T_models/RGI/rgi60-attribs/'):
    RGI_extra = pd.DataFrame()
    for file in tqdm(os.listdir(pth)):
        file_reader = pd.read_csv(pth+file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI_extra = RGI_extra.append(file_reader)
    RGI = RGI_extra[[
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
    return RGI

def data_loader(
    pth_1 = '/data/fast1/glacierml/T_models/T_data/',
    pth_2 = '/data/fast1/glacierml/T_models/RGI/rgi60-attribs/',
    pth_3 = '/data/fast1/glacierml/T_models/matched_indexes/',
    pth_4 = '/data/fast1/glacierml/T_models/regional_data/training_data/',
    RGI_input = 'y',
    scale = 'g',
    region_selection = 1,
    area_scrubber = 'off',
    anomaly_input = 5
):        
    # load glacier GlaThiDa data
    glacier = pd.read_csv(pth_1 + 'glacier.csv', low_memory = False)    
    glacier = glacier.rename(columns = {
        'lat':'Lat',
        'lon':'Lon',
        'area':'area_g',
        'mean_slope':'Mean Slope',
        'mean_thickness':'Thickness'
    })   
    
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
        df = df.dropna()        
        return df

    
    # add in RGI attributes
    elif RGI_input == 'y':
        RGI_extra = pd.DataFrame()
        for file in os.listdir(pth_2):
            file_reader = pd.read_csv(
                pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip'
            )            
            RGI_extra = RGI_extra.append(file_reader, ignore_index=True)
            RGI = RGI_extra
        
        
        # read csv of combined GlaThiDa and RGI indexes, matched glacier for glacier
        comb = pd.read_csv(
                pth_3 + 'GlaThiDa_RGI_matched_indexes.csv'
        )
        # force indexes to be integers rather than floats, and drop duplicates
        comb['GlaThiDa_index'] = comb['GlaThiDa_index'].astype(int)
        comb['RGI_index'] = comb['RGI_index'].astype(int)
        comb = comb.drop_duplicates(subset = 'RGI_index', keep = 'last')
        
        # locate data in both datasets and line them up
        glacier = glacier.loc[comb['GlaThiDa_index']]
        RGI = RGI.loc[comb['RGI_index']]
        
        # reset indexes for merge
        glacier = glacier.reset_index()
        RGI = RGI.reset_index()
        
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
        
        # drop bad data
        df = df.drop(df.loc[df['Zmed']<0].index)
        df = df.drop(df.loc[df['Lmax']<0].index)
        df = df.drop(df.loc[df['Slope']<0].index)
        df = df.drop(df.loc[df['Aspect']<0].index)
        df = df.dropna(subset = ['Thickness'])
        
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
            'area_g'
        ]]
        df = df.dropna()
        
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
                df['size_anomaly'] = abs(df['area_g'] - df['area_r'])
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
                df = df.drop(df.loc[df['Aspect']<0].index)
                df = df.reset_index()
                df = df.drop('index', axis=1)
                df = df[[
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
                ]]
                
            elif area_scrubber == 'off':
                df = df.drop(['area_g', 'RGIId'], axis = 1)
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
                r_df = r_df.append(f, ignore_index = True)
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
                df['size_anomaly'] = abs(df['area_g'] - df['area_r'])
                df = df[df['size_anomaly'] <= anomaly_input]
                df = df.drop([
                    'size_anomaly',
                    'area_g'
                ], axis = 1)
                df = df.rename(columns = {
                    'area_r':'Area'
                })
                df = df[[
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
                ]]
                return df
                
            elif area_scrubber == 'off':
                df = df[[
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
                ]]
                return df



    return df


        
        
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
    print('Please set neurons for first layer')
    layer_1_input = input()
    
    print('Please set neurons second layer')
    layer_2_input = input()
    
    print('Please set learning rate: 0.1, 0.01, 0.001')
    lr_list = ('0.1, 0.01, 0.001')
    lr_input = input()
    while lr_input not in lr_list:
        print('Please set valid learning rate: 0.1, 0.01, 0.001')
        lr_input = input()
        
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
def build_dnn_model(norm, learning_rate=0.1, layer_1 = 10, layer_2 = 5, dropout = True):
    
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
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
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
                          module = 'sm2',
                          res = 'sr2',
                          layer_1 = 10,
                          layer_2 = 5,
                          dropout = True
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
    (train_features,test_features,
     train_labels,test_labels) = data_splitter(dataset)
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
#         print(dataset.name + ' data normalized')

#      DNN model
    dnn_model = {}
    dnn_history = {}
    dnn_results = {}

#         print(
#             'Running multi-variable DNN regression on ' + 
#             str(dataset.name) + 
#             ' dataset with parameters: Learning Rate = ' + 
#             str(learning_rate) + 
#             ', Layer Architechture = ' +
#             arch +
#             ', dropout = ' + 
#             dropout +
#             ', Validation split = ' + 
#             str(validation_split) + 
#             ', Epochs = ' + 
#             str(epochs) + 
#             ', Random state = ' + 
#             str(random_state) 
#         )

    # set up model with  normalized data and defined layer architecture
    dnn_model = build_dnn_model(normalizer['ALL'], learning_rate, layer_1, layer_2, dropout)

    # train model on previously selected and splitdata
    dnn_history['MULTI'] = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=validation_split,
        verbose=0, 
        epochs=epochs
    )

    #save model, results, and history

#         print('Saving results')


    df = pd.DataFrame(dnn_history['MULTI'].history)


    df.to_csv(            
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

    dnn_model.save(
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
#         print('model training complete')
#         print('')
        
        



