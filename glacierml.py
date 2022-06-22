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



'''
data_loader
input = path to GlaThiDa data. Default coded in.
output = dataframe containing glacier-scale GlaThiDa information with null entries dropped.
'''
def data_loader(pth = '/data/fast1/glacierml/T_models/'):
    print('Importing glacier data')
    glacier = pd.read_csv(pth + 'glacier.csv', low_memory = False)
    glacier = glacier[[
        'id',
        'lat',
        'lon',
        'area',
        'mean_slope',
        'mean_thickness'
    ]]
        
    df1 = glacier.dropna()
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
    
    df1 = df1.drop('id',axis = 1)
#     TT = TT.drop('GlaThiDa_ID',axis = 1)
#     TTT = TTT.drop('GlaThiDa_ID',axis =1)
    return df1






'''
data_loader_2
input = path to GlaThiDa data. Default coded in.
output = dataframe containing glacier-scale GlaThiDa information with null entries dropped paired with RGI attributes.
'''
def data_loader_2(pth = '/data/fast1/glacierml/T_models/'):
    print('matching GlaThiDa and RGI data method 1...')
    
    T = pd.read_csv(pth + 'T.csv', low_memory = False)
    rootdir = pth + 'attribs/rgi60-attribs/'
    RGI_extra = pd.DataFrame()
    for file in os.listdir(rootdir):
    #     print(file)
        f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI_extra = RGI_extra.append(f, ignore_index = True)

    comb = pd.read_csv(pth + 'GlaThiDa_RGI_matched_indexes.csv')
#     drops = comb.index[comb['0']!=0]
#     comb = comb.drop(drops)
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

    df2 = pd.merge(T, RGI, left_index=True, right_index=True)


    df2 = df2[[
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
        'Aspect',
        'Lmax'
    ]]
    df2 = df2.dropna(subset = ['MEAN_THICKNESS'])
    
    return df2



# data_loader_3 was skipped in favor of df3 = df2 without lat and lon


'''
data_loader_4
input = path to GlaThiDa data. Default coded in.
output = dataframe containing glacier-scale GlaThiDa information with null entries dropped paired with RGI attributes. GlaThiDa and RGI are matched using a different, more rigorous technique than data_loader_2()
'''
def data_loader_4(pth = '/data/fast1/glacierml/T_models/'):
    print('matching GlaThiDa and RGI data method 2...')
    comb = pd.read_csv(pth + 'GlaThiDa_RGI_live.csv')
    comb = comb.rename(columns = {'0':'distance'})

    glacier = pd.read_csv(pth + 'glacier.csv')
    glacier = glacier.dropna(subset = ['mean_thickness'])

    comb = comb[[
        'GlaThiDa_index',
        'RGI_index',
        'distance'
    ]]

    combined_indexes = pd.DataFrame()
    
    for GlaThiDa_index in comb['GlaThiDa_index'].index:
        df = comb[comb['GlaThiDa_index'] == GlaThiDa_index]
        f = df.loc[df[df['distance'] == df['distance'].min()].index]
        combined_indexes = combined_indexes.append(f)
    combined_indexes
    
    combined_indexes = combined_indexes.drop_duplicates(subset = ['GlaThiDa_index'])
    combined_indexes = combined_indexes.reset_index()
    combined_indexes = combined_indexes[[
        'GlaThiDa_index',
        'RGI_index',
        'distance'
    ]]

    RGI_extra = pd.DataFrame()
    rootdir = pth + 'attribs/rgi60-attribs/'
    for file in os.listdir(rootdir):
#     print(file)
        f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI_extra = RGI_extra.append(f, ignore_index = True)
    
    data = pd.DataFrame(columns = ['GlaThiDa_index', 'thickness'])
    for GlaThiDa in combined_indexes['GlaThiDa_index'].index:
        glathida_thickness = glacier['mean_thickness'].iloc[GlaThiDa] 
        rgi_index = combined_indexes['RGI_index'].loc[GlaThiDa]  
        rgi = RGI_extra.iloc[[rgi_index]]

        data = data.append(rgi)
        
        data['GlaThiDa_index'].iloc[-1] = combined_indexes['GlaThiDa_index'].loc[GlaThiDa]
        data['thickness'].iloc[-1] = glathida_thickness

    data = data.drop_duplicates(subset = ['RGIId'])
    data = data.reset_index()
    
    df4 = data[[
    #     'RGIId',
#         'GlaThiDa_index',
        'CenLon',
        'CenLat',
        'Area',
        'thickness',
        'Zmin',
        'Zmed',
        'Zmax',
        'Slope',
        'Aspect',
        'Lmax'
    ]]
    df4['thickness'] = pd.to_numeric(df4['thickness'])
    
    return df4




'''
data_loader_5
input = path to GlaThiDa data. Default coded in. will also request regional data when run
output = dataframe containing glacier-scale GlaThiDa information with null entries dropped paired with RGI attributes and divided up by selected region. Uses the same matched index csv as data_loader_2(). 
'''
def data_loader_5(pth = '/data/fast1/glacierml/T_models/regional_data_1/training_data/'):
    print('matching GlaThiDa and RGI data...')
    df = pd.DataFrame()
    for file in os.listdir(pth):
        f = pd.read_csv(pth+file, encoding_errors = 'replace', on_bad_lines = 'skip')
        df = df.append(f, ignore_index = True)
        df = df.drop_duplicates(subset = ['CenLat','CenLon'], keep = 'last')
        df = df[[
        #     'GlaThiDa_index',
        #     'RGI_index',
        #     'RGIId',
            'region',
        #     'geographic region',
            'CenLon',
            'CenLat',
            'Area',
            'Zmin',
            'Zmed',
            'Zmax',
            'Slope',
            'Aspect',
            'Lmax',
            'thickness'
        ]]
    print(
        'please select region: ' + str(list(
            df['region'].unique()
        ) )
    )
    df5 = df[df['region'] == float(input())]    
    return df5




'''
data_loader_6
input = path to GlaThiDa data. Default coded in. will also request regional data when run
output = dataframe containing glacier-scale GlaThiDa information with null entries dropped paired with RGI attributes and divided up by selected region. Uses the same matched index csv as data_loader_4(). 
'''
def data_loader_6(pth = '/data/fast1/glacierml/T_models/regional_data_2/training_data/'):
    print('matching GlaThiDa and RGI data...')
    df = pd.DataFrame()
    for file in tqdm(os.listdir(pth)):
        f = pd.read_csv(pth+file, encoding_errors = 'replace', on_bad_lines = 'skip')
        df = df.append(f, ignore_index = True)

#         df = df.drop_duplicates(subset = ['CenLon','CenLat'], keep = 'last')
        df = df[[
        #     'GlaThiDa_index',
        #     'RGI_index',
        #     'RGIId',
            'region',
        #     'geographic region',
            'CenLon',
            'CenLat',
            'Area',
            'Zmin',
            'Zmed',
            'Zmax',
            'Slope',
            'Aspect',
            'Lmax',
            'thickness'
        ]]
    print(
        'please select region: ' + str(list(
            df['region'].unique()
        ) )
    )
    df6 = df[df['region'] == float(input())]    
    return df6




'''
thickness_renamer
input = name of dataframe containing column named either 'MEAN_THICKNESS' or 'mean_thickness'
output = dataframe returned withe name changed to 'thickness'
'''
def thickness_renamer(df):
    if 'MEAN_THICKNESS' in df.columns:
        
        df = df.rename(columns = {
            'MEAN_THICKNESS':'thickness'
        },inplace = True)
        
    else:
        df = df.rename(columns = {
            'mean_thickness':'thickness'
        },inplace = True)
        
'''
data_splitter
input = name of dataframe and selected random state.
output = dataframe and series randomly selected and populated as either training or test features or labels
'''
def data_splitter(df, random_state = 0):
    train_dataset = df.sample(frac=0.8, random_state=random_state)
    test_dataset = df.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    #define label - attribute training to be picked
    train_labels = train_features.pop('thickness')
    test_labels = test_features.pop('thickness')
    
    return train_features, test_features, train_labels, test_labels


'''
prethicktor_inputs
input = none
output = hyperparameters and layer architecture for DNN model
This function is designed and integrated into prethicktor to allow for hyperparameters to be input via CLI rather than hard coding for each run
'''
def prethicktor_inputs():
    print('This model currently supports two layer architecture. Please define first layer')
    layer_1_input = input()
    print('Please define second layer')
    layer_2_input = input()
    print('Please define learning rate: 0.1, 0.01, 0.001')
    lr_input = input()
    print('Please define epochs')
    ep_input = input()
    
    return layer_1_input, layer_2_input, lr_input, ep_input




'''
build_linear_model
input = normalized data and desired learning rate
output = linear regression model
No longer really used.
'''

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
def build_dnn_model(norm, learning_rate=0.1, layer_1 = 10, layer_2 = 5):
    model = keras.Sequential([
              norm,
#               layers.Dense(32, activation='relu'),
              layers.Dense(layer_1, activation='relu'),
              layers.Dense(layer_2, activation='relu'),

              layers.Dense(1) ])

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
                          epochs = 300,
                          random_state = 0,
                          module = 'sm2',
                          res = 'sr2',
                          layer_1 = 10,
                          layer_2 = 5
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
        
    #      DNN model
        dnn_model = {}
        dnn_history = {}
        dnn_results = {}

        print(
            'Running multi-variable DNN regression on ' + 
            str(dataset.name) + 
            ' dataset with parameters: Learning Rate = ' + 
            str(learning_rate) + 
            ', Validation split = ' + 
            str(validation_split) + 
            ', Epochs = ' + 
            str(epochs) + 
            ', Random state = ' + 
            str(random_state) + 
            ', Layer Architechture = ' + 
            arch
        )
        
        # set up model with  normalized data and defined layer architecture
        dnn_model = build_dnn_model(normalizer['ALL'],learning_rate, layer_1, layer_2)
        
        # train model on previously selected and splitdata
        dnn_history['MULTI'] = dnn_model.fit(
            train_features,
            train_labels,
            validation_split=validation_split,
            verbose=0, 
            epochs=epochs
        )
        
        #save model, results, and history
        
        print('Saving results')


        df = pd.DataFrame(dnn_history['MULTI'].history)
        df.to_csv(            
           svd_res_pth +
           str(dataset.name) +
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
            '_dnn_MULTI_' + 
            str(learning_rate) + 
            '_' + 
            str(validation_split) + 
            '_' + 
            str(epochs) + 
            '_' + 
            str(random_state)
        )
        print('model training complete')
        print('')
        
        
        
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



#         for variable_name in tqdm(list(dnn_history)):
#             df = pd.DataFrame(dnn_history[variable_name].history)
#             df.to_csv(
#                 svd_res_pth + 
#                 str(dataset.name) + 
#                 '_dnn_history_' +
#                 str(variable_name) + 
#                 '_' +
#                 str(learning_rate) +
#                 '_' +
#                 str(validation_split) +
#                 '_' +
#                 str(epochs) +
#                 '_' +
#                 str(random_state) 
#             )