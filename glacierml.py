# import sys
# !{sys.executable} -m pip install pyjanitor
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
# test for push
# import janitor

# /home/sa42/data/
# /data/fast1/glacierml/T_models/

def data_loader_2(pth = '/home/prethicktor/data/'):
    
    pth2 = '/data/fast1/glacierml/T_models/'
#     TTT = pd.read_csv(pth2 + 'TTT.csv', low_memory = False)
    T = pd.read_csv(pth + 'T.csv', low_memory = False)
    rootdir = pth + 'attribs/rgi60-attribs/'
    RGI_extra = pd.DataFrame()
    for file in os.listdir(rootdir):
    #     print(file)
        f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI_extra = RGI_extra.append(f, ignore_index = True)

    comb = pd.read_csv('GlaThiDa_RGI_matched_indexes.csv')
    drops = comb.index[comb['0']!=0]
    comb = comb.drop(drops)
    comb = comb.drop_duplicates(subset = 'RGI_index', keep = 'last')
    T = T.loc[comb['GlaThiDa_index']]
    RGI = RGI_extra.loc[comb['RGI_index']]


    RGI = RGI.reset_index()

    T = T.reset_index()



    RGI = RGI[[
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
    T = T[[
        'LAT',
        'LON',
        'AREA',
        'MEAN_SLOPE',
        'MEAN_THICKNESS'
    ]]

    Glam = pd.merge(T,RGI, left_index=True, right_index=True)


    Glam = Glam[[
#         'LAT',
#         'LON',
        'CenLon',
        'CenLat',
        'Area',
        'MEAN_THICKNESS',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Aspect'
    ]]
    Glam = Glam.dropna(subset = ['MEAN_THICKNESS'])

#     Glam['distance'] = 1
#     for G_idx in Glam.index:
#         GlaThiDa_coords = (Glam['LAT'].loc[G_idx],
#                            Glam['LON'].loc[G_idx])
#         RGI_coords = (Glam['CenLat'].loc[G_idx],
#                            Glam['CenLon'].loc[G_idx])

#         distance = geopy.distance.geodesic(GlaThiDa_coords,RGI_coords).km
#         Glam['distance'].loc[G_idx] = distance

    return Glam
    
def data_loader(pth = '/data/fast1/glacierml/T_models/'):
    print('Importing data...')
    print('Importing T database')
    glacier = pd.read_csv(pth + 'glacier.csv', low_memory = False)
    glacier = glacier[[
        'id',
        'lat',
        'lon',
        'area',
        'mean_slope',
        'mean_thickness'
    ]]
        
    glacier = glacier.dropna()
#     print('Importing TT database')
#     TT = pd.read_csv(pth + 'TT.csv', low_memory = False)
#     TT = TT[[
#         'glacier_id',
#         'from_elevation',
#         'to_elevation',
#         'area',
#         'mean_slope',
#         'mean_thickness',
#     ]]
#     TT = TT.dropna()
    
#     print('Importing TTT database')
#     TTT = pd.read_csv(pth + 'TTT.csv', low_memory = False)
#     TTT = TTT[[
#         'glacier_id',
#         'lat',
#         'lon',
#         'elevation',
#         'thickness'
#     ]]
#     TTT = TTT.dropna()
    
#     print('Building TTTx')
#     TTTx = pd.merge(T,TTT, how = 'inner', on = 'GlaThiDa_ID')
#     TTTx.rename(columns = {
#         'LAT':'CENT_LAT',
#         'LON':'CENT_LON'
#     },inplace = True)
    
#     TTTx = TTTx.drop([
#         'GlaThiDa_ID',
#         'MEAN_THICKNESS'
#     ],axis = 1)
#     TTTx = TTTx.dropna()
    
#     print('Building TTT_full')
#     df1 = pd.merge(T,TT, how = 'inner', on = 'GlaThiDa_ID')
#     df1 = df1.rename(columns = {
#         'AREA_x':'T_AREA',
#         'MEAN_SLOPE_x':'T_MEAN_SLOPE',
#         'AREA_y':'TT_AREA',
#         'MEAN_SLOPE_y':'TT_MEAN_SLOPE'
#     })
    
#     df1 = df1.drop([
#         'MEAN_THICKNESS_x',
#         'MEAN_THICKNESS_y',
#     ],axis=1)

#     df1['UPPER_BOUND'] = df1['UPPER_BOUND'].astype('float')
#     df1['LOWER_BOUND'] = df1['LOWER_BOUND'].astype('float')

#     TTT_full = (df1.conditional_join(
#         TTT,
#         ('UPPER_BOUND', 'ELEVATION', '>='), 
#         ('LOWER_BOUND', 'ELEVATION', '<='),
#         how = 'inner'))

#     TTT_full.columns = [
#         'GlaThiDa_ID',
#         'CENT_LAT',
#         'CENT_LON',
#         'T_AREA',
#         'T_MEAN_SLOPE',
#         'LOWER_BOUND',
#         'UPPER_BOUND',
#         'TT_AREA',
#         'TT_MEAN_SLOPE',
#         'GlaThiDa_ID_2',
#         'POINT_LAT',
#         'POINT_LON',
#         'ELEVATION',
#         'THICKNESS'
#     ]

#     TTT_full = TTT_full.drop([
#         'GlaThiDa_ID',
#         'GlaThiDa_ID_2'
#     ],axis=1)
    
    glacier = glacier.drop('id',axis = 1)
#     TT = TT.drop('GlaThiDa_ID',axis = 1)
#     TTT = TTT.drop('GlaThiDa_ID',axis =1)
    print('Import complete')
    return glacier


def thickness_renamer(T):
    T = T.rename(columns = {
        'mean_thickness':'thickness'
    },inplace = True)
    
    
def data_splitter(T, random_state = 0):
    train_dataset = T.sample(frac=0.8, random_state=random_state)
    test_dataset = T.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    #define label - attribute training to be picked
    train_labels = train_features.pop('thickness')
    test_labels = test_features.pop('thickness')
    
    return train_features, test_features, train_labels, test_labels



def build_linear_model(normalizer,learning_rate=0.1):
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_absolute_error')
    
    return model

def build_dnn_model(norm,learning_rate=0.1):
    model = keras.Sequential([
              norm,
              layers.Dense(32, activation='relu'),
              layers.Dense(16, activation='relu'),
              layers.Dense(8, activation='relu'),

              layers.Dense(1) ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate))
    
    return model


def plot_loss(history):
#     plt.subplots(figsize=(10,5))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    #   plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    

     

    
def build_and_train_model(dataset,
                          learning_rate = 0.001,
                          validation_split = 0.2,
                          epochs = 300,
                          random_state = 0):
        # define paths
        arch = '32-16-8'
        svd_mod_pth = 'saved_models/sm5/sm_' + arch + '/'
        svd_res_pth = 'saved_results/sr5/sr_' + arch + '/'
    #     split data
        (train_features,test_features,
         train_labels,test_labels) = data_splitter(dataset)
#         print(dataset.name)
        
    #     normalize data
        print('Normalizing ' + str(dataset.name) + ' data')
        normalizer = {}
        variable_list = list(train_features)
        for variable_name in tqdm(variable_list):
            normalizer[variable_name] = preprocessing.Normalization(input_shape=[1,], axis=None)
            normalizer[variable_name].adapt(np.array(train_features[variable_name]))

        normalizer['ALL'] = preprocessing.Normalization(axis=-1)
        normalizer['ALL'].adapt(np.array(train_features))
        print(dataset.name + ' data normalized')
        
#          # linear model
#         print('Running single-variable linear regression on ' 
#               + str(dataset.name) 
#               + ' dataset with parameters: Learning Rate = ' 
#               + str(learning_rate) 
#               + ', Validation split = ' 
#               + str(validation_split) 
#               + ', Epochs = ' 
#               + str(epochs)
#               + ', Random state = '
#               + str(random_state)
#               + ', Layer Architechture = '
#               + arch)
#         linear_model = {}
#         linear_history = {}
#         linear_results = {}
#         variable_list = list(train_features)

#         for variable_name in tqdm(variable_list):
#             linear_model[variable_name] = build_linear_model(normalizer[variable_name],learning_rate)
#             linear_history[variable_name] = linear_model[variable_name].fit(
#                                                 train_features[variable_name], train_labels,        
#                                                 epochs=epochs,
#                                                 verbose=0,
#                                                 validation_split=validation_split)
            
#             linear_model[variable_name].save(svd_mod_pth 
#                                              + str(dataset.name) 
#                                              + '_linear_' 
#                                              + str(variable_name) 
#                                              + '_' 
#                                              + str(learning_rate) 
#                                              + '_' 
#                                              + str(validation_split) 
#                                              + '_' 
#                                              + str(epochs)
#                                              + '_'
#                                              + str(random_state))
                                         
            

#         print('Running multi-variable linear regression on ' 
#               + str(dataset.name) 
#               + ' dataset with parameters: Learning Rate = ' 
#               + str(learning_rate) 
#               + ', Validation split = ' 
#               + str(validation_split) 
#               + ', Epochs = ' 
#               + str(epochs)
#               + ', Random state = '
#               + str(random_state)
#               + ', Layer Architechture = '
#               + arch)
        
#         linear_model = build_linear_model(normalizer['ALL'],learning_rate)
#         linear_history['MULTI'] = linear_model.fit(
#            train_features, train_labels,        
#            epochs=epochs,
#            verbose=0,
#            validation_split=validation_split)

#         print('Saving results')
#         for variable_name in tqdm(list(linear_history)):
#             df = pd.DataFrame(linear_history[variable_name].history)
#             df.to_csv(svd_res_pth 
#                       + str(dataset.name) 
#                       + '_linear_history_' 
#                       + str(variable_name) 
#                       + '_' 
#                       + str(learning_rate)  
#                       + '_' 
#                       + str(validation_split) 
#                       + '_' 
#                       + str(epochs)
#                       + '_'
#                       + str(random_state))

#         df = pd.DataFrame(linear_history['MULTI'].history)
#         df.to_csv(svd_res_pth 
#                   + str(dataset.name) 
#                   + '_linear_history_MULTI_' 
#                   + str(learning_rate) 
#                   + '_' 
#                   + str(validation_split) 
#                   + '_' 
#                   + str(epochs)
#                   + '_'
#                   + str(random_state))
        
#         linear_model.save(svd_mod_pth 
#                           + str(dataset.name) 
#                           + '_linear_MULTI_' 
#                           + str(learning_rate) 
#                           + '_' 
#                           + str(validation_split) 
#                           + '_' 
#                           + str(epochs)
#                           + '_'
#                           + str(random_state))

    #      DNN model
        dnn_model = {}
        dnn_history = {}
        dnn_results = {}

#         print('Running single-variable DNN regression on '
#               + str(dataset.name) 
#               + ' dataset with parameters: Learning Rate = ' 
#               + str(learning_rate) 
#               + ', Validation split = ' 
#               + str(validation_split) 
#               + ', Epochs = ' 
#               + str(epochs)
#               + ', Random state = '
#               + str(random_state)
#               + ', Layer Architechture = '
#               + arch)
#         variable_list = tqdm(list(train_features))
#         for variable_name in variable_list:
#             dnn_model[variable_name] = build_dnn_model(normalizer[variable_name],learning_rate)
#             dnn_history[variable_name] = dnn_model[variable_name].fit(
#                                                 train_features[variable_name], train_labels,        
#                                                 epochs=epochs,
#                                                 verbose=0,
#                                                 validation_split=validation_split)    
#             dnn_model[variable_name].save(svd_mod_pth 
#                                           + str(dataset.name) 
#                                           + '_dnn_' 
#                                           + str(variable_name) 
#                                           + '_' 
#                                           + str(learning_rate) 
#                                           + '_' 
#                                           + str(validation_split) 
#                                           + '_' 
#                                           + str(epochs)
#                                           + '_'
#                                           + str(random_state)
#                                          )

        print('Running multi-variable DNN regression on ' 
              + str(dataset.name) 
              + ' dataset with parameters: Learning Rate = ' 
              + str(learning_rate) 
              + ', Validation split = ' 
              + str(validation_split) 
              + ', Epochs = ' 
                  + str(epochs)
              + ', Random state = '
              + str(random_state)
              + ', Layer Architechture = '
              + arch)
        dnn_model = build_dnn_model(normalizer['ALL'],learning_rate)
        dnn_history['MULTI'] = dnn_model.fit(
            train_features, train_labels,
            validation_split=validation_split,
            verbose=0, epochs=epochs)
        
        dnn_model.save(svd_mod_pth 
                       + str(dataset.name) 
                       + '_dnn_MULTI' 
                       + '_' 
                       + str(learning_rate)  
                       + '_' 
                       + str(validation_split) 
                       + '_' 
                       + str(epochs)
                       + '_'
                       + str(random_state))

        print('Saving results')
        for variable_name in tqdm(list(dnn_history)):
            df = pd.DataFrame(dnn_history[variable_name].history)
            df.to_csv(svd_res_pth 
                      + str(dataset.name) 
                      + '_dnn_history_'
                      +str(variable_name) 
                      + '_' 
                      + str(learning_rate) 
                      + '_' 
                      + str(validation_split) 
                      + '_' 
                      + str(epochs)
                      + '_'
                      + str(random_state))

        df = pd.DataFrame(dnn_history['MULTI'].history)
        df.to_csv(svd_res_pth 
                  + str(dataset.name) 
                  + '_dnn_history_MULTI_' 
                  + str(learning_rate) 
                  + '_' 
                  + str(validation_split) 
                  + '_' 
                  + str(epochs)
                  + '_'
                  + str(random_state))
        
        dnn_model.save(svd_mod_pth
                       + str(dataset.name) 
                       + '_dnn_MULTI_' 
                       + str(learning_rate)  
                       + '_' 
                       + str(validation_split) 
                       + '_' 
                       + str(epochs)
                       + '_'
                       + str(random_state))
        
        
        
        


    