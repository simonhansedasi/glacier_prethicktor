# import sys
# !{sys.executable} -m pip install 
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
# from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import geopy.distance
# import matplotlib.patches as mpatches
# import plotly.express as px
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.ticker as ticker
import warnings
from tensorflow.python.util import deprecation
import logging
from scipy.stats import shapiro
# import pickle5 as pickle
import pickle
  

tf.random.set_seed(42)
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)

pd.set_option('mode.chained_assignment',None)




'''

'''
def load_RGI(
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
#         print(file)
        region_number = file[:2]
        if str(region_selection) == 'all':
            file_reader = pd.read_csv(pth + file, encoding_errors = 'replace', on_bad_lines = 'skip')
            RGI_extra = pd.concat([RGI_extra,file_reader], ignore_index = True)
            
        elif str(region_selection) != str(region_number):
            pass
        
        elif str(region_selection) == str(region_number):
            file_reader = pd.read_csv(pth + file, encoding_errors = 'replace', on_bad_lines = 'skip')
            RGI_extra = pd.concat([RGI_extra,file_reader], ignore_index = True)
            
    RGI = RGI_extra
#     [[
#         'RGIId',
#         'CenLat',
#         'CenLon',
#         'Slope',
#         'Zmin',
#         'Zmed',
#         'Zmax',
#         'Area',
#         'Aspect',
#         'Lmax',
#         'Name',
#         'GLIMSId',
#     ]]
    RGI['region'] = RGI['RGIId'].str[6:8]
#     RGI['Area'] = np.log10(RGI['Area']
    return RGI


def parameterize_data(parameterization = '1', pth = '/data/fast1/glacierml/data/'):
    import configparser
    config = configparser.ConfigParser()
    config.read('model_parameterization.txt')

    data = load_training_data(
        pth = pth,
        area_scrubber = config[parameterization]['area scrubber'],
        anomaly_input = float(config[parameterization]['size threshold'])
    )


    data = data.drop(
        data[data['distance test'] >= float(config[parameterization]['distance threshold'])].index
    )
    data = data.drop([
#         'RGIId',
        'region', 
        'RGI Centroid Distance', 
        'AVG Radius', 
        'Roundness', 
        'distance test', 
        'size difference'
    ], axis = 1)
    
#     if parameterization == '5':
#         data['Area'] = np.log(data['Area'])
            
#     if parameterization == '6':
#         data['Area'] = np.log(data['Area'])
#         data = data.drop(['CenLat', 'CenLon'], axis = 1)
        
#     if parameterization == '7':
#         data['Area'] = np.log(data['Area'])
#         data = data.drop(
#             ['Zmin', 'Zmed', 'Zmax', 'Lmax','Aspect'], axis = 1
#         )
    
#     if parameterization == '8':
#         data['Area'] = np.log(data['Area'])
#         data = data.drop(
#             ['CenLat', 'CenLon', 'Zmin', 'Zmed', 'Zmax', 'Aspect','Lmax' ], axis = 1
#         )
    
    return data

def load_training_data(
    pth = '/data/fast1/glacierml/data/',
#     alt_pth = '/home/prethicktor/data/',
    RGI_input = 'y',
    scale = 'g',
    region_selection = 1,
    area_scrubber = 'off',
    anomaly_input = 0.5,
#     data_version = 'v1'
):        
    import os
    pth_1 = os.path.join(pth, 'T_data/')
    pth_2 = os.path.join(pth, 'RGI/rgi60-attribs/')
    pth_3 = os.path.join(pth, 'matched_indexes/', 'v2')
    pth_4 = os.path.join(pth, 'regional_data/training_data/', 'v2')
    
    
    pth_5 = pth_3 + '/GlaThiDa_with_RGIId_' + 'v2' + '.csv'                        
                                 
    # load glacier GlaThiDa data v2
    glacier = pd.read_csv(pth_1 + 'T.csv', low_memory = False)    
    glacier = glacier.rename(columns = {
        'LAT':'Lat',
        'LON':'Lon',
        'AREA':'area_g',
        'MEAN_SLOPE':'Mean Slope',
        'MEAN_THICKNESS':'Thickness'
    })   
    glacier = glacier.dropna(subset = ['Thickness'])

#         print('# of raw thicknesses: ' + str(len(glacier)))
        
        
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
        RGI = load_RGI(pth = os.path.join(pth, 'RGI/rgi60-attribs/'))
#         print(RGI)
        RGI['region'] = RGI['RGIId'].str[6:8]

        # load glacier GlaThiDa data v2
        glacier = pd.read_csv(pth_5)    
        glacier = glacier.rename(columns = {
            'LAT':'Lat',
            'LON':'Lon',
            'AREA':'Area',
            'MEAN_SLOPE':'Mean Slope',
            'MEAN_THICKNESS':'Thickness'
        })   
        glacier = glacier.dropna(subset = ['Thickness'])
#         glacier = pd.read_csv(pth_5)
        df = pd.merge(RGI, glacier, on = 'RGIId', how = 'inner')
        
        
        
        
        glacier = glacier.dropna(subset = ['RGIId'])
        rgi_matches = len(glacier)
        rgi_matches_unique = len(glacier['RGIId'].unique())
        
        
#         df = df.rename(columns = {
#             'name':'name_g',
#             'Name':'name_r',

#             'BgnDate':'date_r',
#             'date':'date_g'
#         })

        # make a temp df for the duplicated entries
        
        # calculate the difference in size as a percentage
        df['size difference'] = abs(
            ( (df['Area_x'] - df['Area_y']) )/ df['Area_y'] )
                       
        df = df.rename(columns = {'Area_x':'Area',})
#         df = df.rename(columns = {'Area_y':'Area_GlaThiDa',})
        df = df[[
            'RGIId',
            'CenLat',
            'CenLon',
#             'Lat',
#             'Lon',
            'Area',
#             'Area_RGI',
#             'Area_GlaThiDa',
            'Zmin',
            'Zmed',
            'Zmax',
            'Slope',
            'Aspect',
            'Lmax',
            'Thickness',
#             'area_g',
            'region',
            'size difference',
#             'index_x',
#             'index_y',
            'RGI Centroid Distance'
        ]]
        
        if area_scrubber == 'on':          
            df = df[df['size difference'] <= anomaly_input]
#             df = df.drop([
#                 'size difference',
# #                 'Area_y'
#             ], axis = 1)
#             df = df.rename(columns = {
#                 'Area_x':'Area'
#             })

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
#                 'Area_RGI',
                'Area_GlaThiDa',
                'Aspect',
                'Lmax',
                'Thickness',
                'region',
                'RGI Centroid Distance',
                'size difference'
            ]]
    
    # convert everything to common units (m)
    df['RGI Centroid Distance'] = df['RGI Centroid Distance'].str[:-2].astype(float)
    df['RGI Centroid Distance'] = df['RGI Centroid Distance'] * 1e3

    df['Area'] = df['Area'] * 1e6     # Put area to meters for radius and roundness calc

    # make a guess of an average radius and "roundness" -- ratio of avg radius / width
    df['AVG Radius'] = np.sqrt(df['Area'] / np.pi)
    df['Roundness'] = (df['AVG Radius']) / (df['Lmax'])
    df['distance test'] = df['RGI Centroid Distance'] / df['AVG Radius']
    
    
    df['Area'] = df['Area'] / 1e6     # Put area back to sq km
#     df['Area'] = np.log10(df['Area'])
#     df['Lmax'] = np.log10(df['Lmax'])
        
    return df



'''
GlaThiDa_RGI_index_matcher:
'''
def match_GlaThiDa_RGI_index(
    pth = '/data/fast1/glacierml/data/',
    verbose = False,
    useMP = False
):
    
    import os
    pth_1 = os.path.join(pth, 'T_data/')
    pth_2 = os.path.join(pth, 'RGI/rgi60-attribs/')
    pth_3 = os.path.join(pth, 'matched_indexes/', version)
    
    glathida = pd.read_csv(pth_1 + 'T.csv')
    glathida = glathida.dropna(subset = ['MEAN_THICKNESS'])
    glathida['RGIId'] = np.nan
    glathida['RGI Centroid Distance'] = np.nan
    glathida = glathida.reset_index()
    glathida = glathida.drop('index', axis = 1)
    if verbose: print(glathida)
    RGI = pd.DataFrame()
    for file in os.listdir(pth_2):
#         print(file)
        file_reader = pd.read_csv(pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI = pd.concat([RGI, file_reader], ignore_index = True)
    RGI = RGI.reset_index()
    df = pd.DataFrame()
    #iterate over each glathida index
    
    if useMP == False:
        centroid_distances = []
        RGI_ids = []
        for i in tqdm(glathida.index):
            RGI_id_match, centroid_distance = get_id(RGI,glathida,version,verbose,i)
            centroid_distances.append(centroid_distance)
            RGI_ids.append(RGI_id_match)
    else:
        from functools import partial
        import multiprocessing
        pool = multiprocessing.pool.Pool(processes=48)         # create a process pool with 4 workers
        newfunc = partial(get_id,RGI,glathida,version,verbose) #now we can call newfunc(i)
        output = pool.map(newfunc, glathida.index)
#         print(output)
#         print(output)
    for i in tqdm(glathida.index):      
#         print(output)

        glathida.loc[glathida.index[i], 'RGIId'] = output[i][0]
        glathida.loc[glathida.index[i], 'RGI Centroid Distance'] = output[i][1]
#         print(output)
    isdir = os.path.isdir(pth_3)
    if isdir == False:
        os.makedirs(pth_3)
    glathida.to_csv(pth_3 + 'GlaThiDa_with_RGIId_' + version + '.csv')

    
def get_id(RGI,glathida,version,verbose,i):
    if verbose: print(f'Working on Glathida ID {i}')
    #obtain lat and lon from glathida 
    glathida_ll = (glathida.loc[i].LAT,glathida.loc[i].LON)

    # find distance between selected glathida glacier and all RGI
    distances = RGI.apply(
        lambda row: geopy.distance.geodesic((row.CenLat,row.CenLon),glathida_ll),
            axis = 1
    )
    
    RGI_index = pd.Series(np.argmin(distances), name = 'RGI_indexes')
    centroid_distance = distances.min()
    number_glaciers_matched = len(RGI_index)

    if len(RGI_index) == 1:
        RGI_id_match = (RGI['RGIId'].iloc[RGI_index.loc[0]])
    else:
        RGI_id_match = -1
        centroid_distance = -1
        
    return RGI_id_match, centroid_distance

        
        
'''
split_data
input = name of dataframe and selected random state.
output = dataframe and series randomly selected and populated as either training or test features or labels
'''
# Randomly selects data from a df for a given random state (usually iterated over a range of 25)
# Necessary variables for training and predictions
def split_data(df, random_state = 0):
    train_dataset = df.sample(frac=0.8, random_state=random_state)
    test_dataset = df.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    #define label - attribute training to be picked
    train_labels = train_features.pop('Thickness')
    test_labels = test_features.pop('Thickness')
    
    return train_features, test_features, train_labels, test_labels



'''
build_linear_model
input = normalized data and desired learning rate
output = linear regression model
'''
# No longer used
def build_linear_model(normalizer,learning_rate=0.01):
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
    norm, learning_rate=0.01, layer_1 = 10, layer_2 = 5, dropout = True, 
    loss = 'mean_absolute_error'
):
#     def coeff_determination(y_true, y_pred):
#         SS_res =  K.sum(K.square( y_true-y_pred )) 
#         SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
#     return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    
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
#         if loss == 'mean_squared_error':
#             def coeff_determination(y_true, y_pred):
#                 SS_res =  K.sum(K.square( y_true-y_pred )) 
#                 SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
#             return ( 1 - SS_res/(SS_tot + K.epsilon()) )
#             model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])
#         if loss == 'mean_absolute_error':
#             model.compile(loss='mean_absolute_error',
#                     optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate))
        
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
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True)
    

     
'''
build_and_train_model
input = dataset, desired: learning rate, validation split, epochs, random state. module and res are defined as inputs when run and determine where data is saved.
output = saved weights for trained model and model results saved as a csv
'''

def build_and_train_model(dataset,
#                           learning_rate = 0.01,
#                           validation_split = 0.2,
#                           epochs = 100,
                          
                          parameterization = 'sm',
#                           res = 'sr',
                          layer_1 = 10,
                          layer_2 = 5,
                          random_state = 0,
                          dropout = True,
                          verbose = False,
                          writeToFile = True,
                          loss = 'mean_absolute_error'
                         ):
    # define paths
    arch = str(layer_1) + '-' + str(layer_2)
    svd_mod_pth = 'saved_models/' + parameterization + '/' + arch + '/'
    svd_res_pth = 'saved_results/' + parameterization + '/' + arch + '/'

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
    ) = split_data(dataset, random_state)
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
    dnn_model = build_dnn_model(normalizer['ALL'], 0.01, layer_1, layer_2, dropout, loss)
    
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
        validation_split=0.2,
        callbacks = [callback],
        verbose=0, 
        epochs=2000
    )
    df = pd.DataFrame(dnn_history['MULTI'].history)
    #save model, results, and history

    if writeToFile:

#         df = pd.DataFrame(dnn_history['MULTI'].history)


        history_filename = (
            svd_res_pth +
#             str(layer_1) + '_' + str(layer_2) + '_' + 
            str(random_state)
        )

        df.to_pickle(  history_filename + '.pkl' )

        model_filename =  (
            svd_mod_pth + 
            str(random_state)
        )

        dnn_model.save(  model_filename  )

        return history_filename, model_filename
    
    else:
        return dnn_model, df, normalizer
    

    

def load_dnn_model(
#     model_name,
    model_loc
):
    
#     dnn_model = {}
    dnn_model = tf.keras.models.load_model(model_loc)
    
    return dnn_model
    
     
        
        
'''
Workflow functions
'''



def build_model_ensemble(
    data, parameterization, useMP = False, verbose = True
):
    # build models
    RS = range(0,25,1)
    print(data)
    layer_1_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    layer_2_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    if useMP == False:
#         print('Building model ensemble')
        for layer_2_input in (layer_2_list):
            for layer_1_input in (layer_1_list):
                if layer_1_input <= layer_2_input:
                    pass
                elif layer_1_input > layer_2_input:

                    arch = str(layer_1_input) + '-' + str(layer_2_input)
                    dropout = True
                    print('Running multi-variable DNN regression with parameterization ' + 
                        str(parameterization) + 
                        ', layer architecture = ' +
                        arch)

                    for rs in tqdm(RS):

                        build_and_train_model(
                            data, 
                            parameterization = parameterization, 
#                             res = parameterization,
                            layer_1 = layer_1_input,
                            layer_2 = layer_2_input,
                            random_state = rs, 
                        )   
    else:
        for layer_2_input in (layer_2_list):
            for layer_1_input in (layer_1_list):
                if layer_1_input <= layer_2_input:
                    pass
                elif layer_1_input > layer_2_input:
                    arch = str(layer_1_input) + '-' + str(layer_2_input)
                    if verbose: print(
                        'Running multi-variable DNN regression with parameterization ' + 
                        str(parameterization) + 
                        ', layer architecture = ' +
                        arch)
#                     arch = model_statistics['layer architecture']
                    from functools import partial
                    import multiprocessing
                    pool = multiprocessing.pool.Pool(processes=25) 

                    newfunc = partial(
                        build_and_train_model,
                        data,
                        parameterization,
                        layer_1_input,
                        layer_2_input
            #             verbose
            #             arch
                    )
                    output = pool.map(newfunc, RS)

                    
def assess_model_performance(data, parameterization = '1'):
    # evaluate model loss and then calculate model statistics
    model_predictions = pd.DataFrame()
    model_statistics = pd.DataFrame()
    rootdir = 'saved_models/' + parameterization + '/'

    print('loading and evaluating models...')
    for arch in tqdm(os.listdir(rootdir)):       
        pth = os.path.join(rootdir, arch)
        for folder in (os.listdir(pth)):
            architecture = arch
            model_loc = (
                rootdir + 
                arch + 
                '/' + 
                folder
            )

            model_name = folder
            dnn_model = load_dnn_model(model_loc)
            df = evaluate_model(architecture, model_name, data, dnn_model, parameterization)

            model_predictions = pd.concat([model_predictions, df], ignore_index = True)
    model_predictions.rename(columns = {0:'avg train thickness'},inplace = True)
    model_predictions.to_pickle('zults/model_predictions_' + parameterization + '.pkl')
    # calculate statistics
    print('calculating statistics...')
    dnn_model = {}

    for arch in tqdm(list(model_predictions['layer architecture'].unique())):
        model_thicknesses = model_predictions[model_predictions['layer architecture'] == arch]


        model_name = ('0')

        model_loc = (
            rootdir + 
            arch + 
            '/' +
            '0'
        )
        isdir = os.path.isdir(model_loc)
        if isdir == False:
            print('model not here, calculating next model')
        elif isdir == True:


            dnn_model = load_dnn_model(model_loc)
            df = calculate_model_avg_statistics(
                dnn_model,
                arch,
                data,
                model_thicknesses
            )

            model_statistics = pd.concat(
                [model_statistics, df], ignore_index = True
            )


        model_statistics['architecture weight 1'] = (
            sum(model_statistics['test mae avg']) / model_statistics['test mae avg']
        )
        model_statistics['architecture weight 2'] = (
            model_statistics['test mae avg'] / sum(model_statistics['test mae avg'])
        )
        model_statistics.to_pickle(
            'zults/model_statistics_' + 
            parameterization + 
            '.pkl'
        )

def evaluate_model(
    arch,
    rs,
    dataset,
    dnn_model,
    parameterization
):

    (
        train_features, test_features, train_labels, test_labels
    ) = split_data(
        dataset, random_state = int(rs)
    )
    
    features = pd.concat([train_features,test_features], ignore_index = True)
    labels = pd.concat([train_labels, test_labels], ignore_index = True)
    
    mae_test = dnn_model.evaluate(
                    test_features, test_labels, verbose=0
                )
    mae_train = dnn_model.evaluate(
        train_features, train_labels, verbose=0
    )
    df = features
    thicknesses = (dnn_model.predict(features).flatten())
    df['model'] = rs
    df['test mae'] = mae_test
    df['train mae'] = mae_train
    df['layer architecture'] = arch
    df['parameterization'] = parameterization
    df['total parameters'] = dnn_model.count_params() 
    df['Thickness'] = labels
    df['Estimated Thickness'] = thicknesses
    df['Residual'] = df['Estimated Thickness'] - df['Thickness']

    return df


'''

'''
def calculate_model_avg_statistics(
    dnn_model,
    arch, 
    dataset,
    model_thicknesses
):
    
    df = pd.DataFrame({
                'Line1':[1]
    })
    
    test_mae_mean = np.mean(model_thicknesses['test mae'])
    test_mae_std_dev = np.std(model_thicknesses['test mae'])

    train_mae_mean = np.mean(model_thicknesses['train mae'])
    train_mae_std_dev = np.std(model_thicknesses['train mae'])

    df.loc[
        df.index[-1], 'layer architecture'
    ] = arch  

    
    df.loc[
        df.index[-1], 'total parameters'
    ] = dnn_model.count_params() 

    df.loc[
        df.index[-1], 'trained parameters'
    ] = df.loc[
        df.index[-1], 'total parameters'
    ] - (len(dataset.columns) + (len(dataset.columns) - 1))

    df.loc[
        df.index[-1], 'total inputs'
    ] = (len(dataset) * (len(dataset.columns) -1))

    df.loc[
        df.index[-1], 'test mae avg'
    ] = test_mae_mean

    df.loc[df.index[-1], 'train mae avg'] = train_mae_mean

    df.loc[df.index[-1], 'test mae std dev'] = test_mae_std_dev

    df.loc[df.index[-1], 'train mae std dev'] = train_mae_std_dev
    
    df = df.dropna()


    df = df.sort_values('test mae avg')
    df = df.drop('Line1', axis = 1)

    return df



def estimate_thickness(
        model_statistics,
#         arch,
        parameterization = '1',
        verbose = True,
        useMP = False,
        
    ):
    RGI = load_RGI(pth = '/home/prethicktor/data/RGI/rgi60-attribs/')
#     RGI = load_RGI(pth = '/data/fast1/glacierml/data/RGI/rgi60-attribs/')

    RGI['region'] = RGI['RGIId'].str[6:8]
    RGI = RGI.reset_index()
    RGI = RGI.drop('index', axis=1)
    
    if useMP == False:
        print('Estimating thicknesses')
        for arch in tqdm(model_statistics['layer architecture'].unique()):
            make_estimates(
                RGI,
                parameterization, 
                verbose,
                arch,
            )


    else:
        arch = model_statistics['layer architecture']
        from functools import partial
        import multiprocessing
        pool = multiprocessing.pool.Pool(processes=5) 

        newfunc = partial(
            make_estimates,
            RGI,
            parameterization, 
            verbose
#             arch
        )
        output = pool.map(newfunc, arch.unique())
#     print(output[1])
#     for i in arch:
#         print(output[i])


        
def make_estimates(
    RGI,
    parameterization,
    verbose,
    arch,
    
):
    if verbose: print(f'Estimating RGI with layer architecture {arch}, parameterization {parameterization}')
    dfs = pd.DataFrame()
    RGI_for_predictions = RGI[[
        'CenLon', 'CenLat', 'Slope', 'Zmin', 'Zmed', 'Zmax', 'Area', 'Aspect', 'Lmax'
    ]]
    
#     .drop(['region', 'RGIId'], axis = 1)
    for rs in tqdm(range(0,25,1)):
        rs = str(rs)
        results_path = 'saved_results/' + parameterization + '/' + arch + '/'
        history_name = rs
        dnn_history = {}
        dnn_history[rs] = pd.read_csv(results_path + rs)
#         if exclude == True:
#         if abs((
#             dnn_history[history_name]['loss'].iloc[-1]
#         ) - dnn_history[history_name]['val_loss'].iloc[-1]) >= 3:
#             pass
#         else:

        model_path = (
            'saved_models/' + parameterization + '/' + arch + '/' + rs
        )

        dnn_model = tf.keras.models.load_model(model_path)

        s = pd.Series(
            dnn_model.predict(RGI_for_predictions, verbose=0).flatten(), 
            name = rs
        )
        dfs[rs] = s

    RGI_prethicked = RGI.copy() 
    RGI_prethicked = pd.concat([RGI_prethicked, dfs], axis = 1)
    RGI_prethicked['avg predicted thickness'] = dfs.mean(axis = 1)
    RGI_prethicked['predicted thickness std dev'] = dfs.std(axis = 1)

    RGI_prethicked.to_pickle(
        'zults/RGI_predicted_' +
        parameterization + '_' + arch + '.pkl'          
    )    

    return RGI_prethicked


def compile_model_weighting_data():

    for j in tqdm(range(1,5,1)):

        parameterization = str(j)

        # glac = gl.load_training_data(RGI_input = 'y')
        glac = parameterize_data(parameterization)
        arch = list_architectures(parameterization)

        dft = pd.DataFrame()
        for architecture in (arch['layer architecture'].unique()):
        #     print(architecture)
            df_glob = load_global_predictions(parameterization, architecture = architecture)
            dft = pd.concat([dft, df_glob])

        df = dft[[
                'layer architecture','RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24',
        ]]

        glathida_estimates = pd.merge(glac, df, how = 'inner', on = 'RGIId')

        est = glathida_estimates

        for i in range(0,25,1):
            est['pr_'+str(i)] = ((est[str(i)] - est['Thickness'])) / est['Thickness']

        for i in range(0,25,1):
            est['r_'+str(i)] = ((est[str(i)] - est['Thickness']))

        est.to_pickle('model_weights/param' + str(j) + '_weighting_data.pkl')




def compute_model_weights(model_statistics, parameterization, pth = '/home/prethicktor/data/'):
    path = 'model_weights/'
    file = path + 'architecture_weights_' + parameterization + '.pkl'   
    if os.path.isfile(file) == True:
        architecture_weights = pd.read_pickle(file)
        residual_model = np.load('model_weights/residual_model_' + parameterization + '.npy',)
    if os.path.isfile(file) == False:
    

        est = pd.read_pickle('model_weights/param' + parameterization + '_weighting_data.pkl')
        model_list = [
             '0', '1', '2', '3', '4', '5', '6', '7', '8',
             '9', '10', '11', '12', '13', '14', '15', '16',
             '17', '18', '19', '20', '21', '22', '23', '24',
        ]
        pool_list = [
             'pr_0', 'pr_1', 'pr_2', 'pr_3', 'pr_4', 'pr_5', 'pr_6', 'pr_7', 'pr_8',
             'pr_9', 'pr_10', 'pr_11', 'pr_12', 'pr_13', 'pr_14', 'pr_15', 'pr_16',
             'pr_17', 'pr_18', 'pr_19', 'pr_20', 'pr_21', 'pr_22', 'pr_23', 'pr_24',
        ]
        weight_list = [
             'w_0', 'w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6', 'w_7', 'w_8',
             'w_9', 'w_10', 'w_11', 'w_12', 'w_13', 'w_14', 'w_15', 'w_16',
             'w_17', 'w_18', 'w_19', 'w_20', 'w_21', 'w_22', 'w_23', 'w_24',
        ]

        res_list = [
             'r_0', 'r_1', 'r_2', 'r_3', 'r_4', 'r_5', 'r_6', 'r_7', 'r_8',
             'r_9', 'r_10', 'r_11', 'r_12', 'r_13', 'r_14', 'r_15', 'r_16',
             'r_17', 'r_18', 'r_19', 'r_20', 'r_21', 'r_22', 'r_23', 'r_24',
        ]

        weights = pd.DataFrame()
        architecture_weights = pd.DataFrame()
        print('Calculating weights')
        for i in tqdm(est['layer architecture'].unique()):
            dft = est[est['layer architecture'] == str(i)]
            
            simple_var = np.var(dft[res_list].to_numpy().flatten())
            q75, q25 = np.nanpercentile(dft[res_list], [75,25])
            sigma_simple = ((q75 - q25) ) / 1.34896
            
            bias_1 = np.mean(dft[pool_list].to_numpy()) * np.mean(dft[model_list].to_numpy())
            q75, q25 = np.nanpercentile(dft[pool_list], [75,25])
            IQR_1 = q75 - q25
            sigma_1 = (IQR_1 * np.mean(dft[model_list].to_numpy())) / 1.34896

            w_1 = pd.Series(abs(bias_1) + sigma_1**2, name = 'weight')

            bias_2 = np.mean(dft[pool_list].to_numpy() * dft[model_list].to_numpy())
            q75, q25 = np.nanpercentile(est[pool_list], [75,25])
            IQR_2 = q75 - q25
            sigma_2 = (IQR_2 * np.mean(dft[model_list].to_numpy())) / 1.34896

            w_2 = (abs(bias_2) + sigma_2**2)
            
            
            bias_3 = np.mean(est[pool_list].to_numpy()) * np.mean(dft[model_list].to_numpy())
            q75, q25 = np.nanpercentile(est[pool_list], [75,25])
            IQR_3 = q75 - q25
            sigma_3 = (IQR_3 * np.mean(dft[model_list].to_numpy())) / 1.34896

            w_3 = (abs(bias_3) + sigma_3**2)
            w_4 = simple_var


            architecture_weights = pd.concat([architecture_weights, w_1])
            architecture_weights = architecture_weights.reset_index()
            architecture_weights = architecture_weights.drop('index', axis = 1)
            architecture_weights.loc[architecture_weights.index[-1], 'layer architecture'] = i
            architecture_weights.loc[architecture_weights.index[-1], 'simple var'] = simple_var

            architecture_weights.loc[architecture_weights.index[-1], 'std_1'] = sigma_1
            architecture_weights.loc[architecture_weights.index[-1], 'IQR_1'] = IQR_1
            architecture_weights.loc[architecture_weights.index[-1], 'bias_1'] = bias_1
            architecture_weights.loc[architecture_weights.index[-1], 'std_2'] = sigma_2
            architecture_weights.loc[architecture_weights.index[-1], 'IQR_2'] = IQR_2
            architecture_weights.loc[architecture_weights.index[-1], 'bias_2'] = bias_2
            architecture_weights.loc[architecture_weights.index[-1], 'std_3'] = sigma_3
            architecture_weights.loc[architecture_weights.index[-1], 'IQR_3'] = IQR_3
            architecture_weights.loc[architecture_weights.index[-1], 'bias_3'] = bias_3
            architecture_weights.loc[architecture_weights.index[-1], 'IQR_4'] = sigma_simple

            architecture_weights.loc[architecture_weights.index[-1], 'aw_2'] = w_2
            architecture_weights.loc[architecture_weights.index[-1], 'aw_3'] = w_3
            architecture_weights.loc[architecture_weights.index[-1], 'aw_4'] = w_4

        print('calculating residual curve...')
        df = pd.DataFrame()
        for i in range(0,25,1):
            x = pd.DataFrame(
                    pd.Series(
                        est[str(i)] - est['Thickness'],
                        name = 'Residual'
                )
            )
            y = pd.DataFrame(
                pd.Series(
                    est['Thickness'],
                    name = 'GlaThiDa Survey Thickness'
                )
            )
            dft = x.join(y)
            df = pd.concat([df, dft])
        x = df['GlaThiDa Survey Thickness']
        y = df['Residual']


        residual_model = np.polyfit(x,y,2)
        print(residual_model)

        architecture_weights = architecture_weights.rename(columns = {0:'aw_1'})
        architecture_weights['var_1'] = architecture_weights['std_1']**2
        architecture_weights['var_2'] = architecture_weights['std_2']**2
        architecture_weights['var_3'] = architecture_weights['std_3']**2

        architecture_weights.to_pickle('model_weights/architecture_weights_' + 
                                       parameterization + '.pkl')
#         residual_model.to_pickle('model_weights/residual_model_' + parameterization + '.pkl')
        
        np.save(
            'model_weights/residual_model_' + parameterization,
            residual_model, 
            allow_pickle=True, 
            fix_imports=True)

        
        
        
        
        
        
    return architecture_weights, residual_model





def calculate_RGI_thickness_statistics(architecture_weights, residual_model, model_statistics, parameterization):
    # aggregate model thicknesses
#     print('Gathering architectures...')
    arch_list = model_statistics.sort_values('layer architecture')
#     print(arch_list)
#     arch_list = list_architectures(parameterization = parameterization)
#     arch_list = arch_list.sort_values('layer architecture')
#     arch_list = arch_list.reset_index()
#     arch_list = arch_list.drop('index', axis = 1)

    aggregate_statistics(architecture_weights, residual_model, arch_list, parameterization)


    
    

def GB_D_common_estimator(n, S, X):
    mu = sum((n / S)*X) / sum(n / S)
    
    return mu

def unbiased_variance_estimator(n_m, n_x, sigma_m, sigma_x):
    
    q_1 = 4 / (n_m - 1)
    q_2 = (n_m / sigma_m) / sum(n_x/sigma_x)
    q_3 = (n_m / sigma_m**2) / sum(n_x/sigma_x)**2
    q_4 = sum(n_m / sigma_m)
    
    var = (
        (1 + sum(q_1 * (q_2 - q_3))) / q_4
    )
    return var
    
    
    
    
    
    
def aggregate_statistics(
    architecture_weights, 
    residual_model, 
    arch_list, 
    parameterization, 
    verbose = True
):
    
    
    pth = 'predicted_thicknesses/compiled_raw_' + parameterization + '.h5'   
    if os.path.isfile(pth) == True:
        df = pd.read_hdf(
            'predicted_thicknesses/compiled_raw_' + parameterization + '.h5',
            key = 'compiled_raw', mode = 'a'
        )

    if os.path.isfile(pth) == False:
        df = pd.DataFrame(columns = [
                'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24',
        ])
    #     print('Architectures listed')


        print('Compiling predictions...')
    #     print(arch_list)
        for arch in tqdm(arch_list['layer architecture'].unique()):
            df_glob = load_global_predictions(
                parameterization = parameterization,
                architecture = arch
            )

            df = pd.concat([df,df_glob])
        statistics = pd.DataFrame()
        for file in (os.listdir('zults/')):
            if 'statistics_' + parameterization in file:
                file_reader = pd.read_pickle('zults/' + file)
                statistics = pd.concat([statistics, file_reader], ignore_index = True)

        df = pd.merge(df, statistics, on = 'layer architecture')
        df = df[[
                'layer architecture','RGIId','0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24'
        ]]


        print('Grouping predictions')
        compiled_raw = df.groupby('RGIId')[[
                'layer architecture','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24'
        ]]
        
        df.to_hdf('predicted_thicknesses/compiled_raw_' + parameterization + '.h5', 
                  key = 'compiled_raw', mode = 'a')
        
        
    print('Predictions compiled')
    print('Applying weights...')
    
    dft = pd.DataFrame()
    
    
    compiled_raw = df.groupby('RGIId')[[
                'layer architecture','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24'
        ]]
    
    for this_rgi_id, obj in tqdm(compiled_raw):
        
        rgi_id = pd.Series(this_rgi_id, name = 'RGIId')
        dft = pd.concat([dft, rgi_id])
        dft = dft.reset_index()
        dft = dft.drop('index', axis = 1)
        obj = obj[[
            'layer architecture','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
        ]]

        obj = pd.merge(obj, architecture_weights, how = 'inner', on = 'layer architecture')

        predictions = obj[[
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
        ]]

        arch_weight_1 = obj[['aw_1']]
        arch_weight_2 = obj[['aw_2']]
        arch_weight_3 = obj[['aw_3']]
        arch_weight_4 = obj[['aw_4']]

        aw_1 = arch_weight_1.values.flatten()
        aw_2 = arch_weight_2.values.flatten()
        aw_3 = arch_weight_3.values.flatten()
        aw_4 = arch_weight_4.values.flatten()

        pr = np.array(predictions.values)
        

        ### WEIGHTED MEAN ###
#         hat_h = GB_D_common_estimator(
#             n = 25, 
#             S = predictions.var(axis = 0), 
#             X = predictions.mean(axis = 0)
#         )
        bar_H = predictions.mean(axis = 1)
        
        hat_mu_1 = sum( (bar_H) / (aw_1) ) / sum(1/aw_1)
        dft.loc[dft.index[-1], 'Weighted Mean Thickness 1'] = hat_mu_1
        hat_mu_2 = sum( (bar_H) / (aw_2) ) / sum(1/aw_2)
        dft.loc[dft.index[-1], 'Weighted Mean Thickness 2'] = hat_mu_2       
        hat_mu_3 = sum( (bar_H) / (aw_3) ) / sum(1/aw_3)
        dft.loc[dft.index[-1], 'Weighted Mean Thickness 3'] = hat_mu_3 
        hat_mu_4 = sum( (bar_H) / (aw_4) ) / sum(1/aw_4)
        dft.loc[dft.index[-1], 'Weighted Mean Thickness 4'] = hat_mu_4 
#         weighted_mean = 0
#         for p, w in zip(pr, aw):
#             weighted_mean = weighted_mean + np.nanmean(p/w)
#         weighted_mean = weighted_mean / sum(1/aw)
#         dft.loc[dft.index[-1], 'Weighted Mean Thickness'] = weighted_mean
        
        
        
        
        
        ### UNCERTAINTY CALCULATIONS ###
         # deviation modeled uncertainty (Farinotti)
        gamma_1 = obj['IQR_1'] / 1.34896
        sigma_d_1 = gamma_1 * bar_H
        gamma_2 = obj['IQR_2'] / 1.34896
        sigma_d_2 = gamma_2 * bar_H
        gamma_3 = obj['IQR_3'] / 1.34896
        sigma_d_3 = gamma_3 * bar_H
        gamma_4 = obj['IQR_4'] / 1.34896
        sigma_d_4 = gamma_4 * bar_H
        
        sigma_sq_mu_1 = 1 / sum(1/sigma_d_1**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty_1'] = sigma_sq_mu_1
        sigma_sq_mu_2 = 1 / sum(1/sigma_d_2**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty_2'] = sigma_sq_mu_2
        sigma_sq_mu_3 = 1 / sum(1/sigma_d_3**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty_3'] = sigma_sq_mu_3
        sigma_sq_mu_4 = 1 / sum(1/sigma_d_4**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty_4'] = sigma_sq_mu_4
        
        
        
        
        
        
        sigma_d_31 = gamma_1[0:3] * bar_H[0:3]
        sigma_sq_mu_31 = 1 / sum(1/sigma_d_31**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty 3'] = sigma_sq_mu_31
        
        sigma_d_20 = gamma_1[0:32] * bar_H[0:32]
        sigma_sq_mu_20 = 1 / sum(1/sigma_d_20**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty 20'] = sigma_sq_mu_20
                
        sigma_d_40 = gamma_1[0:64] * bar_H[0:64]
        sigma_sq_mu_40 = 1 / sum(1/sigma_d_40**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty 40'] = sigma_sq_mu_40
        
        sigma_d_60 = gamma_1[0:96] * bar_H[0:96]
        sigma_sq_mu_60 = 1 / sum(1/sigma_d_60**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty 60'] = sigma_sq_mu_60
        
        sigma_d_80 = gamma_1[0:128] * bar_H[0:128]
        sigma_sq_mu_80 = 1 / sum(1/sigma_d_80**2)
        dft.loc[dft.index[-1], 'Composite Deviation Uncertainty 80'] = sigma_sq_mu_80
        
        
        
        weighted_variance_1 = sum(sigma_d_1**2 / aw_1) / sum(1 / aw_1)
        dft.loc[dft.index[-1], 'Weighted Deviation Uncertainty_1'] = weighted_variance_1
        weighted_variance_2 = sum(sigma_d_2**2 / aw_2) / sum(1 / aw_2)
        dft.loc[dft.index[-1], 'Weighted Deviation Uncertainty_2'] = weighted_variance_2
        
        weighted_variance_3 = sum(sigma_d_3**2 / aw_3) / sum(1 / aw_3)
        dft.loc[dft.index[-1], 'Weighted Deviation Uncertainty_3'] = weighted_variance_3
        
        weighted_variance_4 = sum(sigma_d_1**2 / aw_4) / sum(1 / aw_4)
        dft.loc[dft.index[-1], 'Weighted Deviation Uncertainty_4'] = weighted_variance_4
        
        weighted_variance_4 = sum(sigma_d_4**2 / aw_4) / sum(1 / aw_4)
        dft.loc[dft.index[-1], 'Weighted Deviation Uncertainty_4_1'] = weighted_variance_4
        
        sigma_d_simple = np.mean(predictions * 0.290)
        weighted_variance_4 = sum(sigma_d_simple**2 / aw_4) / sum(1 / aw_4)
        dft.loc[dft.index[-1], 'Simple Deviation Uncertainty_4'] = weighted_variance_4
        
        
#         total_uncertainty = residual_variance + MAE_GD + var_mu
#         dft.loc[dft.index[-1], 'Total Uncertainty'] = total_uncertainty       
        # model uncertainty
        var_mu = unbiased_variance_estimator(
            n_m = 161, 
            n_x = 25, 
            sigma_m = predictions.var(axis = 1), 
            sigma_x = predictions.var(axis = 0)
        )
        dft.loc[dft.index[-1], 'Bootstrap Uncertainty'] = var_mu
        
        
        
        
        boot = predictions.var(axis = 1)
        dft.loc[dft.index[-1], 'Weighted Deviation Uncertainty_4_2'] = 1 / sum(1/boot)

        weighted_boot = sum(boot / aw_1) / sum(1/aw_1)
        
        dft.loc[dft.index[-1], 'Weighted Bootstrap Uncertainty_1'] = weighted_boot
        
        weighted_boot = sum(boot / aw_2) / sum(1/aw_2)
        
        dft.loc[dft.index[-1], 'Weighted Bootstrap Uncertainty_2'] = weighted_boot
        
        weighted_boot = sum(boot / aw_3) / sum(1/aw_3)
        
        dft.loc[dft.index[-1], 'Weighted Bootstrap Uncertainty_3'] = weighted_boot
        
        weighted_boot = sum(boot / aw_4) / sum(1/aw_4)
        
        dft.loc[dft.index[-1], 'Weighted Bootstrap Uncertainty_4'] = weighted_boot
        
        # Residual Correction Factor
        
        gamma = (obj['IQR_1'][0] / 1.5)
        p_mean = predictions.mean(axis = 1)
        rc = residual_model[0]*p_mean**2 + residual_model[1]*p_mean + residual_model[2]
        
        weighted_residual = sum(rc / aw_1) / sum(1/aw_1)
        
        
        
        dft.loc[dft.index[-1], 'Residual Correction'] = weighted_residual
                                  
        sigma_rc = gamma * rc
        
        weighted_residual_uncertainty = sum(sigma_rc**2 / aw_1) / sum(1/aw_1)
        dft.loc[dft.index[-1], 'Residual Correction Uncertainty'] = weighted_residual_uncertainty
        
        
        if weighted_residual <= 0:
            corrected_thickness = hat_mu_1 - weighted_residual
            dft.loc[dft.index[-1], 'Corrected Thickness'] = corrected_thickness
            dft.loc[dft.index[-1], 'Corrected Thickness Uncertainty'] = weighted_variance_1 + weighted_residual_uncertainty
        if weighted_residual > 0:
            dft.loc[dft.index[-1], 'Corrected Thickness'] = hat_mu_1
            dft.loc[dft.index[-1], 'Corrected Thickness Uncertainty'] = weighted_variance_1 
        
        # MAE base uncertainty
        
        MAE_GD = 16.321**2
        dft.loc[dft.index[-1], 'MAE Uncertainty'] = MAE_GD

        
        

        
        
        
        
        ### UN-WEIGHTED MEAN & UNCERTAINTY ###
        stacked_object = obj[[
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
        ]].stack()
        dft.loc[dft.index[-1], 'Mean Thickness'] = stacked_object.mean()
        
        glacier_count = len(stacked_object)
        dft.loc[dft.index[-1], 'Median Thickness'] = stacked_object.median()
        dft.loc[dft.index[-1],'Thickness Std Dev'] = stacked_object.std()

        statistic, p_value = shapiro(stacked_object)    
        dft.loc[dft.index[-1],'Shapiro-Wilk statistic'] = statistic
        dft.loc[dft.index[-1],'Shapiro-Wilk p_value'] = p_value


        q75, q25 = np.percentile(stacked_object, [75, 25])    
        dft.loc[dft.index[-1],'IQR'] = q75 - q25 

        lower_bound = np.percentile(stacked_object, 50 - 34.1)
        median = np.percentile(stacked_object, 50)
        upper_bound = np.percentile(stacked_object, 50 + 34.1)

        dft.loc[dft.index[-1],'Lower Bound'] = lower_bound
        dft.loc[dft.index[-1],'Upper Bound'] = upper_bound
        dft.loc[dft.index[-1],'Median Value'] = median
        dft.loc[dft.index[-1],'Total estimates'] = glacier_count
#         break
    dft = dft.rename(columns = {
        0:'RGIId'
    })
    dft = dft.drop_duplicates()
    dft.to_pickle(
        'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_' + 
        parameterization + '.pkl'
    ) 


'''
'''
def list_architectures(
    parameterization = '1'
):
    root_dir = 'zults/'
    arch_list = pd.DataFrame()
    for file in tqdm(os.listdir(root_dir)):
        
        if 'RGI_predicted_' + parameterization in file :
            file_reader = pd.read_pickle(root_dir + file)
            arch = pd.Series(file[16:-4])
            arch_list = pd.concat([arch_list, arch], ignore_index = True)
            arch_list = arch_list.reset_index()
            arch_list = arch_list.drop('index', axis = 1)
#             arch_list.loc[arch_list.index[-1], 'parameterization'] = parameterization
#             arch_list.loc[arch_list.index[-1], 'arch'] = arch
    
    arch_list = arch_list.rename(columns = {
        0:'layer architecture'
    })
    arch_list = arch_list.drop_duplicates()
    return arch_list




def load_global_predictions(
    parameterization,
    architecture,
):
    
#     print(architecture)
    root_dir = 'zults/'
    for file in (os.listdir(root_dir)):
            # print(file)
        if ('RGI_predicted_' + parameterization + '_' + architecture in file):
            RGI_predicted = pd.read_pickle(root_dir + file)

            RGI_predicted['layer architecture'] = architecture

            RGI_predicted['parameterization'] =  parameterization


    return RGI_predicted



def load_notebook_data(
    parameterization = '1', pth = ''
):
    df = pd.read_pickle(
            pth + 'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_'+
            parameterization + '.pkl'
        )
    df['region'] = df['RGIId'].str[6:8]

    RGI = load_RGI()
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


    df = pd.merge(df, RGI, on = 'RGIId')
    
    
    df['Slope'][df['Slope'] == -9] = np.nan

    df['Lmax'][df['Lmax'] == -9] = np.nan
    df['Zmin'][df['Zmin'] == -999] = np.nan
    df['Zmax'][df['Zmax'] == -999] = np.nan
    df['Zmed'][df['Zmed'] == -999] = np.nan
#     df['Upper Bound'] = df['Upper Bound'] - df['Weighted Mean Thickness']
#     df['Lower Bound'] = df['Weighted Mean Thickness'] - df['Lower Bound']

#     volume = np.round(
#         sum(df['Weighted Mean Thickness'] / 1e3 * df['Area']) / 1e3, 2)

#     std = np.round(
#         sum(df['Thickness Std Dev'] / 1e3 * df['Area']) / 1e3, 2)


#     df['Weighted Volume (km3)'] = df['Weighted Mean Thickness'] / 1e3 * df['Area']
#     df['Weighted Volume Std Dev (km3)'] = df['Weighted Thickness Uncertainty'] / 1e3 * df['Area']
    
    reference_path = 'reference_thicknesses/'
    ref = pd.DataFrame()
    for file in os.listdir(reference_path):
        if 'Farinotti' in file:
            file_reader = pd.read_csv('reference_thicknesses/' + file)
            ref = pd.concat([ref, file_reader], ignore_index = True) 
    ref['region'] = ref['RGIId'].str[6:8]
    ref = ref.sort_values('RGIId')

    ref['Farinotti Volume (km3)'] = (ref['Farinotti Mean Thickness'] / 1e3 )* ref['Area']

    ref['region'] = ref['RGIId'].str[6:8]
#     ref = ref.dropna(subset = ['Farinotti Mean Thickness'])
#     print(ref)
    ref = ref.rename(columns = {
         'Mean Thickness':'Farinotti Mean Thickness',
         'Shapiro-Wilk statistic':'Farinotti Shapiro-Wilk statistic',
         'Shapiro-Wilk p_value':'Farinotti Shapiro-Wilk p_value',
         'Median Thickness':'Farinotti Median Thickness',
         'Thickness STD':'Farinotti Thickness STD',
         'Skew':'Farinotti Skew',
#          'Farinotti Volume (km3)'
    })
#     print(ref)
    ref = ref[[
         'Farinotti Mean Thickness',
         'Farinotti Shapiro-Wilk statistic',
         'Farinotti Shapiro-Wilk p_value',
         'Farinotti Median Thickness',
         'Farinotti Thickness STD',
         'Farinotti Skew',
         'RGIId',
         'Farinotti Volume (km3)'
    ]]
    df = df[[
         'RGIId',
#          'Weighted Mean Thickness',
         'Mean Thickness',
         'Median Thickness',
         'Thickness Std Dev',
        
         'Weighted Mean Thickness 1',
         'Weighted Mean Thickness 2',
         'Weighted Mean Thickness 3',
         'Weighted Mean Thickness 4',
         'Corrected Thickness',
         'Corrected Thickness Uncertainty',
         'Residual Correction',
         'Residual Correction Uncertainty',
         'Bootstrap Uncertainty',
         'Weighted Bootstrap Uncertainty_1',
         'Weighted Bootstrap Uncertainty_2',
         'Weighted Bootstrap Uncertainty_3',
         'Weighted Bootstrap Uncertainty_4',
         'Composite Deviation Uncertainty_1',
         'Composite Deviation Uncertainty_2',
         'Composite Deviation Uncertainty_3',
         'Composite Deviation Uncertainty_4',
         'Composite Deviation Uncertainty 3',
         'Composite Deviation Uncertainty 20',
         'Composite Deviation Uncertainty 40',
         'Composite Deviation Uncertainty 60',
         'Composite Deviation Uncertainty 80',
         'Weighted Deviation Uncertainty_1',
         'Weighted Deviation Uncertainty_2',
         'Weighted Deviation Uncertainty_3',
         'Weighted Deviation Uncertainty_4',
         'Weighted Deviation Uncertainty_4_1',
         'Weighted Deviation Uncertainty_4_2',
         'Simple Deviation Uncertainty_4',
#          'Weighted Thickness Uncertainty',
#          'Unc1',
#          'Unc2',
#          'Unc3',
#          'Unc4',
#          'Unc5', 
#          'Unc6',
#          'Unc7',
#          'Unc8',
         
        'MAE Uncertainty',
#         'Total Uncertainty',
#         'Weighted Volume (km3)',

#          'Total Uncertainty 2',
#          'Weighted Volume Std Dev (km3)',
        
         'Lower Bound',
         'Upper Bound',
         'Shapiro-Wilk statistic',
         'Shapiro-Wilk p_value',
         'IQR',
    #      'Median Value',
         'Total estimates',
         'region',
         'CenLat',
         'CenLon',
         'Slope',
         'Zmin',
         'Zmed',
         'Zmax',
         'Area',
         'Aspect',
         'Lmax',
#          'Weighted Mean Thickness',
#           'Architecture Weighted Mean Thickness'

    ]]

#     df = df.rename(columns = {
# #         'Weighted Mean Thickness':'Edasi Weighted Mean Thickness',
# #         'Mean Thickness':'Edasi Mean Thickness',
# #         'Median Thickness':'Edasi Median Thickness',
# #         'Thickness Std Dev':'Edasi Thickness Std Dev',
# #         'Shapiro-Wilk statistic':'Edasi Shapiro-Wilk statistic',
# #         'Shapiro-Wilk p_value':'Edasi Shapiro-Wilk p_value',
# #         'IQR':'Edasi IQR',
# #         'Lower Bound':'Edasi Lower Bound',
# #         'Upper Bound':'Edasi Upper Bound',
# #         'Volume Std Dev (km3)':'Edasi Volume Std Dev (km3)',
# #         'Weighted Mean Thickness':'Edasi Weighted Mean Thickness'

#     #     'Median Value':,

#     })
    
    df = pd.merge(df, ref, on = 'RGIId', how = 'inner')

    return df



def assign_arrays(
    parameterization = '4',method = '1',
    size_thresh_1 = 1e-5, size_thresh_2 = 1e4,
    new_data = True
):
    data = load_notebook_data(parameterization)
    data = data.dropna(subset = 'Farinotti Mean Thickness')


    thick_est_unc = (
        data['Weighted Mean Thickness ' + method].to_numpy() + 
        data['Weighted Deviation Uncertainty_' + method].to_numpy() + 
        data['MAE Uncertainty'].to_numpy()
    )

    thick_est = data['Weighted Mean Thickness '+ method].to_numpy()

    thick_far = data['Farinotti Mean Thickness'].to_numpy()

    area = data['Area'].to_numpy()

    x = thick_far / 1e3 * area
    y = thick_est / 1e3 * area
    unc = np.sqrt(thick_est_unc) / 1e3 * area


#         print(x.max())
#         print(y.max())
        
        
        
    index = np.where(
        (x<size_thresh_2)&(x>size_thresh_1)&(y<size_thresh_2)&(y>size_thresh_1)
    )
    x_new = x[index]
    y_new = y[index]
    unc_new = unc[index]
    
    pth = 'arrays/'+parameterization+method+'_vol_density.npy'
    if os.path.isfile(pth) == True:
#         print('density array found')
        z = np.load(pth)
    elif os.path.isfile(pth) == False:
#         print('calculating density array')
        from scipy.stats import gaussian_kde
        xy = np.vstack([np.log10(x),np.log10(y)])
        z = gaussian_kde(xy)(xy)
        np.save(pth, z)
    
    z_new_pth = (
        'arrays/'+parameterization+method+'_vol' + 
        str(size_thresh_1) + '-' + str(size_thresh_2)+'_density.npy'
    )
    if os.path.isfile(z_new_pth) == True:
#         print('threshold density array found')
        z_new = np.load(z_new_pth)
        
    elif os.path.isfile(z_new_pth) == False:
#         print('calculating density of desired threshold')
        from scipy.stats import gaussian_kde
        xy = np.vstack([np.log10(x_new),np.log10(y_new)])
#         print(xy)
        z_new = gaussian_kde(xy)(xy)
        np.save(z_new_pth, z_new)
        
    
    return x,y,z,unc,x_new,y_new,z_new,unc_new,data,index


def assign_sub_arrays(
    est_ind,i,j,
    parameterization = '4',method = '1', 
    feature = 'Area'
    
#     size_thresh_1 = 1e-5, size_thresh_2 = 1e4,
):
    data = load_notebook_data(parameterization)
    data = data.dropna(subset = 'Farinotti Mean Thickness')

    data = data.iloc[est_ind]
    data['Slope'][data['Slope'] == -9] = np.nan
    data['Lmax'][data['Lmax'] == -9] = np.nan
    data['Zmin'][data['Zmin'] == -999] = np.nan
    data['Zmax'][data['Zmax'] == -999] = np.nan
    data['Zmed'][data['Zmed'] == -999] = np.nan

#     data = data.drop(
#         data[
#             ( |
#             (data['Zmin'] == -999) |
#             (data['Zmax'] == -999) |
#             (data['Zmed'] == -999)
#         ].index, axis = 0
#     )
    thick_est_unc = (
        data['Weighted Mean Thickness ' + method].to_numpy() + 
        data['Weighted Deviation Uncertainty_' + method].to_numpy() + 
        data['MAE Uncertainty'].to_numpy()
    )

    thick_est = data['Weighted Mean Thickness '+ method].to_numpy()

    thick_far = data['Farinotti Mean Thickness'].to_numpy()

    feat = data[feature].to_numpy()


    x = thick_far
    y = thick_est
    unc = np.sqrt(thick_est_unc) 

#         print(x.max())
#         print(y.max())
        
        
    st1 = np.floor(x.min())
    st2 = np.ceil(x.max())


    
    
    pth = (
        'arrays/'+parameterization+method+'_thickness_z_'+
        str(
            min(np.floor(np.min(x)), np.floor(np.min(y)))
        ) + '-' + str(
            max(np.ceil(np.max(x)), np.ceil(np.max(y)))
        )+ 
        '-' + str(i) + '-' + str(j) + '_density.npy'
    )
    if os.path.isfile(pth) == True:
#         print('threshold density array found')
        z = np.load(pth)
        
    elif os.path.isfile(pth) == False:
#         print('calculating thickness density of desired threshold')
        from scipy.stats import gaussian_kde
        xy = np.vstack([np.log10(x),np.log10(y)])
        print(xy)
        z = gaussian_kde(xy)(xy)
        np.save(pth, z)
        
        
        
        
        
    pth_f = (
        'arrays/'+parameterization+method+'_' + feature + '-thickness_f_'+
        str(np.floor(np.min(x))) + '-' + str(np.ceil(np.max(x)))+ 
        '-' + str(i) + '-' + str(j) + '_density.npy'
    )
    pth_e = (
        'arrays/'+parameterization+method+'_' + feature + '-thickness_e_'+
        str(np.floor(np.min(y))) + '-' + str(np.ceil(np.max(y)))+ 
        '-' + str(i) + '-' + str(j) + '_density.npy'
    )
    if os.path.isfile(pth_f) == True:
#         print(feature + '-thickness density arrays found')
        zf = np.load(pth_f)
        ze = np.load(pth_e)
        
    elif os.path.isfile(pth_f) == False:
#         zf = 1
#         ze = 1
#         print('calculating '+feature + '-thickness density of desired threshold')
        from scipy.stats import gaussian_kde
        xy = np.vstack([np.log10(x),np.log10(feat)])
        print(xy)
        zf = gaussian_kde(xy)(xy)
        np.save(pth_f,zf)
        
        xy = np.vstack([np.log10(y),np.log10(feat)])
        ze = gaussian_kde(xy)(xy)
        np.save(pth_e,ze)
    data['Slope'] = data['Slope'] + .00001
    data['Zmin'] = data['Zmin'] + .00001
    return x,y,z,zf,ze,unc, data, feat


    

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
