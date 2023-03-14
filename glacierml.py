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
        'RGIId','region', 'RGI Centroid Distance', 
        'AVG Radius', 'Roundness', 
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
            ( (df['Area_x'] - df['Area_y']) / df['Area_y'] )
        )                
        df = df.rename(columns = {'Area_x':'Area',})
        df = df[[
            'RGIId',
            'CenLat',
            'CenLon',
#             'Lat',
#             'Lon',
            'Area',
#             'Area_y',
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
    version = 'v2',
    pth = '/home/prethicktor/data/',
    verbose = False,
    useMP = False
):
    
    import os
    pth_1 = os.path.join(pth, 'T_data/')
    pth_2 = os.path.join(pth, 'RGI/rgi60-attribs/')
    pth_3 = os.path.join(pth, 'matched_indexes/', version)
    
    if version == 'v2':
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
    ) = split_data(dataset)
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

        df.to_csv(  history_filename  )

        model_filename =  (
            svd_mod_pth + 
            str(random_state)
        )

        dnn_model.save(  model_filename  )

        return history_filename, model_filename
    
    else:
        return dnn_model, df
    

    

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
    model_predictions.to_csv('zults/model_predictions_' + parameterization + '.csv')
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
        model_statistics.to_csv(
            'zults/model_statistics_' + 
            parameterization + 
            '.csv'
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
    df['GlaThiDa Thickness'] = labels
    df['E&L Thickness'] = thicknesses
    df['Residual'] = df['GlaThiDa Thickness'] - df['E&L Thickness']

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
#     RGI['Area'] = np.log10(RGI['Area'])
#     print(list(RGI))
#     if parameterization == '5':
#         RGI['Area'] = np.log(RGI['Area'])
# #         RGI = RGI.drop(['CenLat', 'CenLon'], axis = 1)
    
#     if parameterization == '6':
#         RGI['Area'] = np.log(RGI['Area'])
#         RGI = RGI.drop(['CenLat', 'CenLon'], axis = 1)
    
#     if parameterization == '7':
#         RGI['Area'] = np.log(RGI['Area'])
#         RGI = RGI.drop(
#             ['Zmin', 'Zmed', 'Zmax', 'Lmax', 'Aspect'], axis = 1
#         )
    
#     if parameterization == '8':
#         RGI['Area'] = np.log(RGI['Area'])
#         RGI = RGI.drop(
#             ['CenLat', 'CenLon','Zmin', 'Zmed', 'Zmax', 'Aspect', 'Lmax'], axis = 1
#         )
#     print(RGI)
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
#     RGI_prethicked['avg predicted thickness'] = 'NaN'
#     RGI_prethicked['predicted thickness std dev'] = 'NaN'
    RGI_prethicked = pd.concat([RGI_prethicked, dfs], axis = 1)
    RGI_prethicked['avg predicted thickness'] = dfs.mean(axis = 1)
    RGI_prethicked['predicted thickness std dev'] = dfs.std(axis = 1)
#     if verbose: print(f'Averaging estimated thicknesses of layer architecture {arch}')
# #     print('Averaging estimated thicknesses')
#     for i in tqdm(dfs.index):
#         RGI_prethicked['avg predicted thickness'].loc[i] = np.mean(dfs.loc[i])
        
#     if verbose: print(f'Finding standard deviation of layer architecture {arch}')
# #     print('Finding standard deviation of estimated thicknesses')
#     for i in tqdm(dfs.index):
#         RGI_prethicked['predicted thickness std dev'].loc[i] = np.std(dfs.loc[i])

    RGI_prethicked.to_csv(
        'zults/RGI_predicted_' +
        parameterization + '_' + arch + '.csv'          
    )    

    return RGI_prethicked


def calculate_RGI_thickness_statistics(model_statistics, parameterization):
    # aggregate model thicknesses
#     print('Gathering architectures...')
    arch_list = model_statistics.sort_values('layer architecture')
#     arch_list = list_architectures(parameterization = parameterization)
#     arch_list = arch_list.sort_values('layer architecture')
#     arch_list = arch_list.reset_index()
#     arch_list = arch_list.drop('index', axis = 1)

    aggregate_statistics(arch_list, parameterization)


def aggregate_statistics(arch_list, parameterization, verbose = True):
    df = pd.DataFrame(columns = {
            'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
    })
#     print('Architectures listed')

    
    print('Compiling predictions...')
    for arch in tqdm(arch_list['layer architecture'].unique()):
        df_glob = load_global_predictions(
            parameterization = parameterization,
            architecture = arch
        )

        df = pd.concat([df,df_glob])
#         break
    statistics = pd.DataFrame()
    for file in (os.listdir('zults/')):
        if 'statistics_' + parameterization in file:
            file_reader = pd.read_csv('zults/' + file)
            statistics = pd.concat([statistics, file_reader], ignore_index = True)
            
    df = pd.merge(df, statistics, on = 'layer architecture')
    df = df[[
            'layer architecture','RGIId','0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24'
    ]]
    
    # insert model weights here with layer arch column
    
    
    compiled_raw = df.groupby('RGIId')[
            'layer architecture','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24'
    ]

    print('Predictions compiled')
    print('Applying weights...')
    
#     weights = pd.read_csv('model_weights.csv')
    architecture_weights = pd.read_csv('architecture_weights.csv')
    architecture_weights = architecture_weights.drop('Unnamed: 0', axis = 1)

#     print(list(architecture_weights))
#     print(architecture_weights)
#     print(weights)
#     print(df)
#     print('Aggregating statistics...')
    
    dft = pd.DataFrame()
    
    
    
    
    for this_rgi_id, obj in tqdm(compiled_raw):
        
        rgi_id = pd.Series(this_rgi_id, name = 'RGIId')
    #         print(f"Data associated with RGI_ID = {this_rgi_id}:")
        dft = pd.concat([dft, rgi_id])
        dft = dft.reset_index()
        dft = dft.drop('index', axis = 1)
        obj = obj[[
            'layer architecture','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
        ]]
    #         print(weights)
#         print(list(obj))
        obj = pd.merge(obj, architecture_weights, how = 'inner', on = 'layer architecture')
#         print(obj)
        
#         obj = pd.merge(obj, weights, how = 'inner', on = 'layer architecture',
#                        suffixes = ('','mw'))
#         print(list(obj))
        predictions = obj[[
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
        ]]
#         print('predictions')
#         print(predictions)
#         model_weights = obj[[
#             'w_0', 'w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6', 'w_7', 'w_8', 'w_9','w_10',
#             'w_11','w_12','w_13','w_14','w_15','w_16','w_17','w_18','w_19','w_20','w_21',
#             'w_22','w_23','w_24',
#         ]]

        arch_weight = obj[['architecture weight']]
        aw = arch_weight.values.flatten()
        pr = np.array(predictions.values)
        
        std = predictions.std(axis = 1)
        sd = np.array(std)
        
        weighted_std = 0
        for s, w in zip(sd, aw):
            weighted_std = weighted_std + np.nanmean(s/w)
        weighted_std = weighted_std / sum(1/aw)
        
        weighted_mean = 0
        for p, w in zip(pr, aw):
            weighted_mean = weighted_mean + np.nanmean(p/w)
        weighted_mean = weighted_mean / sum(1/aw)
        

        stacked_object = obj[[
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
        ]].stack()
        glacier_count = len(stacked_object)
        dft.loc[dft.index[-1], 'Mean Thickness'] = stacked_object.mean()
        
        dft.loc[dft.index[-1], 'Weighted Mean Thickness'] = weighted_mean
#         dft.loc[dft.index[-1], 'Composite Mean Thickness'] = arch_weighted_thickness
        dft.loc[dft.index[-1], 'Weighted Thickness Uncertainty'
               ] = weighted_std

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
    dft.to_csv(
        'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_' + 
        parameterization + '.csv'
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
            file_reader = pd.read_csv(root_dir + file)
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
            RGI_predicted = pd.read_csv(root_dir + file)

            RGI_predicted['layer architecture'] =  architecture

    RGI_predicted['parameterization'] =  parameterization


    return RGI_predicted



def load_notebook_data(
    parameterization = '1'
):
    df = pd.read_csv(
            'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_'+
            parameterization + '.csv'
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
    df['Upper Bound'] = df['Upper Bound'] - df['Mean Thickness']
    df['Lower Bound'] = df['Mean Thickness'] - df['Lower Bound']

#     volume = np.round(
#         sum(df['Weighted Mean Thickness'] / 1e3 * df['Area']) / 1e3, 2)

#     std = np.round(
#         sum(df['Thickness Std Dev'] / 1e3 * df['Area']) / 1e3, 2)


    df['Weighted Volume (km3)'] = df['Weighted Mean Thickness'] / 1e3 * df['Area']
    df['Weighted Volume Std Dev (km3)'] = df['Weighted Thickness Uncertainty'] / 1e3 * df['Area']
    
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
        
         'Weighted Mean Thickness',
         'Weighted Thickness Uncertainty',
         'Weighted Volume (km3)',
         'Weighted Volume Std Dev (km3)',
        
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
