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
        'Lmax'
    ]]
    return RGI

def data_loader(
    root_dir = '/data/fast1/glacierml/data/',
    RGI_input = 'y',
    scale = 'g',
    region_selection = 1,
    area_scrubber = 'off',
    anomaly_input = 5
):        
    
    pth_1 = root_dir + 'T_data/'
    pth_2 = root_dir + 'RGI/rgi60-attribs/'
    pth_3 = root_dir + 'matched_indexes/'
    pth_4 = root_dir + 'regional_data/training_data/'
    
    
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
            RGI_extra = pd.concat([RGI_extra, file_reader], ignore_index=True)
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
        if region_selection == 19:
            df = df
#             df = df.drop(df.loc[df['Zmed']<0].index)
#             df = df.drop(df.loc[df['Lmax']<0].index)
#             df = df.drop(df.loc[df['Slope']<0].index)
#             df = df.drop(df.loc[df['Aspect']<0].index)
#             df = df.dropna(subset = ['Thickness'])

        elif region_selection != 19:
            df = df.drop(df.loc[df['Zmed']<0].index)
            df = df.drop(df.loc[df['Lmax']<0].index)
            df = df.drop(df.loc[df['Slope']<0].index)
#             df = df.drop(df.loc[df['Aspect']<0].index)
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
                ]]
                
            elif area_scrubber == 'off':
                df = df.drop([
                    'area_g', 
                    'RGIId'
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
#                     'RGIId'
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
                    'region'
                ]]
                return df
                
            elif area_scrubber == 'off':
                df = df[[
#                     'RGIId'
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
                    'region'
                ]]
                df = df.rename(columns = {'area_r':'Area'})
                return df



    return df



'''
GlaThiDa_RGI_index_matcher:
'''
def GlaThiDa_RGI_index_matcher(
    pth_1 = '/data/fast1/glacierml/data/T_data/',
    pth_2 = '/data/fast1/glacierml/data/RGI/rgi60-attribs/',
    pth_3 = '/data/fast1/glacierml/data/matched_indexes/'
):
    glathida = pd.read_csv(pth_1 + 'glacier.csv')
    glathida = glathida.dropna(subset = ['mean_thickness'])

    RGI = pd.DataFrame()
    for file in os.listdir(pth_2):
        print(file)
        file_reader = pd.read_csv(pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI = pd.concat([RGI, file_reader], ignore_index = True)
    RGI = RGI.reset_index()
    df = pd.DataFrame(columns = ['GlaThiDa_index', 'RGI_index'])
    #iterate over each glathida index
    for i in tqdm(glathida.index):
        #obtain lat and lon from glathida 
        glathida_ll = (glathida.loc[i].lat,glathida.loc[i].lon)
        
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


        df.to_csv(pth_3 + 'GlaThiDa_RGI_matched_indexes_live.csv')
        
        
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

    # train model on previously selected and splitdata
    dnn_history['MULTI'] = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=validation_split,
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
    dft = pd.concat([train_features, thickness], axis = 1)
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
    df.loc[df.index[-1], 'dataset'] = dataset.name
    df.loc[df.index[-1], 'dropout'] = dropout

#                 if chosen_dir in global_list:
#                     predictions.loc[predictions.index[-1], 'region'] = 'g'
#                 if chosen_dir in region_list:
#                     predictions.loc[predictions.index[-1], 'region'] = int(reg)

    if '0.1' in folder:
        df.loc[df.index[-1], 'learning rate'] = '0.1'
    if '0.01' in folder:
        df.loc[df.index[-1], 'learning rate'] = '0.01'
    if '0.001' in folder:
        df.loc[df.index[-1], 'learning rate']= '0.001'

    if '10' in folder:
        df.loc[df.index[-1], 'epochs']= '10'
    if '15' in folder:
        df.loc[df.index[-1], 'epochs']= '15'               
    if '20' in folder:
        df.loc[df.index[-1], 'epochs']= '20' 
    if '25' in folder:
        df.loc[df.index[-1], 'epochs']= '25'
    if '30' in folder:
        df.loc[df.index[-1], 'epochs']= '30'
    if '35' in folder:
        df.loc[df.index[-1], 'epochs']= '35'
    if '40' in folder:
        df.loc[df.index[-1], 'epochs']= '40'
    if '45' in folder:
        df.loc[df.index[-1], 'epochs']= '45'
    if '50' in folder:
        df.loc[df.index[-1], 'epochs']= '50'
    if '55' in folder:
        df.loc[df.index[-1], 'epochs']= '55'
    if '60' in folder:
        df.loc[df.index[-1], 'epochs']= '60'
    if '65' in folder:
        df.loc[df.index[-1], 'epochs']= '65'
    if '70' in folder:
        df.loc[df.index[-1], 'epochs']= '70'
    if '75' in folder:
        df.loc[df.index[-1], 'epochs']= '75'
    if '80' in folder:
        df.loc[df.index[-1], 'epochs']= '80'
    if '85' in folder:
        df.loc[df.index[-1], 'epochs']= '85'
    if '90' in folder:
        df.loc[df.index[-1], 'epochs']= '90'
    if '95' in folder:
        df.loc[df.index[-1], 'epochs']= '95'
    if '100' in folder:
        df.loc[df.index[-1], 'epochs']= '100'
    if '150' in folder:
        df.loc[df.index[-1], 'epochs']= '150'
    if '200' in folder:
        df.loc[df.index[-1], 'epochs']= '200'       
    if '300' in folder:
        df.loc[df.index[-1], 'epochs']= '300'
    if '400' in folder:
        df.loc[df.index[-1], 'epochs']= '400'

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
def predictions_loader(
    training_module,
    architecture,
    learning_rate,
    epochs,
):
    root_dir = 'zults/'
    RGI_predicted = pd.DataFrame()

    for file in tqdm(os.listdir(root_dir)):
        # print(file)
        if 'RGI_predicted' in file and 'df' + training_module + '_' in file and architecture in file and learning_rate in file and epochs in file:
            # print(file)
            file_reader = pd.read_csv(root_dir + file)
            # print(file_reader)
            file_reader['volume km3'] = (
                file_reader['avg predicted thickness'] / 1e3
            ) * file_reader['Area']
    #         print(file_reader)
            file_reader = file_reader.dropna()
        
            sum_volume = sum(file_reader['volume km3'])
            total_volume = pd.Series(sum_volume, name = 'total volume')
            RGI_predicted = pd.concat([RGI_predicted, total_volume], ignore_index = True)    
            
            file_reader['variance'] = file_reader['predicted thickness std dev'] **2 
            variance = sum(file_reader['variance'])
            
            RGI_predicted.loc[
                RGI_predicted.index[-1], 'total variance'
            ] = np.sqrt(variance)/1e3

            area = sum(file_reader['Area'])

            RGI_predicted.loc[
                RGI_predicted.index[-1], 'area'
            ] = area
            
            for i in range(1,9,1):
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
            
            
            
            if 'df1' not in file:
                att_list = [
                    'Area',
                    'Aspect',
                    'Lmax',
                    'Slope',
                    'Zmin',
                    'Zmax'
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


            if '10-5' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '10-5'
                
            if '16-8' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '16-8'
                
            if '24-12' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '24-12'
                
            if '37-20' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '37-20'
                
            if '47-21' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '47-21'
                
            if '50-28' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '50-28'
                
            if '50-25' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '50-25'
                
            if '59-28' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '59-28'
                                
            if '60-46' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '60-46'
                
            if '64-36' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '64-36'
                
            if '64-42' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '64-42'
                
            if '64-48' in file:
                RGI_predicted.loc[
                    RGI_predicted.index[-1], 'architecture'
                ] = '64-48'
                
                
            for i in range(1,9,1):
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
            if '_20' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '20'
            if '_25' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '25'
            if '_50' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '50'
            if '_60' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '60'
            if '_15' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '15'
            if '_30' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '30'
            if '_40' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '40'
            if '_100' in file:
                RGI_predicted.loc[RGI_predicted.index[-1], 'epochs']= '100'
                
    RGI_predicted = RGI_predicted.rename(columns = {
        0:'vol'
    })

    RGI_predicted['vol'] = RGI_predicted['vol'] / 1e3

    RGI_predicted['mean thickness'] = (
        (RGI_predicted['vol'] * 1e3) / RGI_predicted['area']
    ) * 1e3
    RGI_predicted['voldiff'] = (RGI_predicted['vol']) - RGI_predicted['volf'] 
    RGI_predicted = RGI_predicted.reset_index()
    RGI_predicted = RGI_predicted.drop('index', axis = 1)
    RGI_predicted = RGI_predicted.sort_values([
    #     'mean thickness (km)',
    #     'architecture',
    #     'learning rate',
        'dataframe'
    ], ascending = True)
                        
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

    df['vol_ratio'] = df['vol'] / df['volf']
    df['vol_from_zero'] = abs(1 - df['vol_ratio'])

    return df