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
import warnings
# from tensorflow.python.util import deprecation
import logging
from scipy.stats import shapiro
import pickle
import math


# tf.random.set_seed(42)
# tf.get_logger().setLevel(logging.ERROR)
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# # deprecation._PRINT_DEPRECATION_WARNINGS = False
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)

# pd.set_option('mode.chained_assignment',None)




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
            file_reader = pd.read_csv(pth + file, 
                                      encoding_errors = 'replace', 
                                      on_bad_lines = 'skip'
                                     )
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


def coregister_data(coregistration = '1', pth = '/data/fast1/glacierml/data/'):
    import configparser
    config = configparser.ConfigParser()
    config.read('model_coregistration.txt')

    data = load_training_data(
        pth = pth,
        area_scrubber = config[coregistration]['area scrubber'],
        anomaly_input = float(config[coregistration]['size threshold'])
    )


    data = data.drop(
        data[data['distance test'] >= float(config[coregistration]['distance threshold'])].index
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
#                 'Area_GlaThiDa',
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

input = name of dataframe and selected random state.
output = dataframe and series randomly selected and populated as either training or test features or labels
'''
# Randomly selects data from a df for a given random state (usually iterated over a range of 25)
# Necessary variables for training and predictions
def split_data(df, random_state = 0):
    train_dataset = df.sample(frac=0.7, random_state=random_state)
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
    norm, learning_rate=0.01, layer_1 = 10, layer_2 = 5, loss = 'mae'
):
#     def coeff_determination(y_true, y_pred):
#         SS_res =  K.sum(K.square( y_true-y_pred )) 
#         SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
#     return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    
    model = keras.Sequential(
        [
              norm,
              layers.Dense(layer_1, activation='relu'),
              layers.Dropout(rate = 0.1, seed = 0),
              layers.Dense(layer_2, activation='relu'),
              layers.Dense(1) 
        ]
    )
    
    if loss == 'mse':
        model.compile(optimizer='adam', loss='mean_squared_error')

#         model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
    if loss == 'mae':
        
        model.compile(
            loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate)
        )    
#     def coeff_determination(y_true, y_pred):
#         SS_res =  K.sum(K.square( y_true-y_pred )) 
#         SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
#         return ( 1 - SS_res/(SS_tot + keras.epsilon()) )
    
#     model_mse.compile(
#         optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination]
#     )


        
        
        
    
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
    plt.title('Model Training')
#     plt.show()
    

     
'''
build_and_train_model
input = dataset, desired: learning rate, validation split, epochs, random state. module and res are defined as inputs when run and determine where data is saved.
output = saved weights for trained model and model results saved as a csv
'''

def build_and_train_model(
    dataset,
    loss = 'mse',
    coregistration = '4',
    layer_1 = 10,
    layer_2 = 5,
    random_state = 0,
    dropout = True,
    verbose = False,
    writeToFile = True,
                         ):
    # define paths
    arch = str(layer_1) + '-' + str(layer_2)
    
    model_pth = 'saved_models/' + loss+ '_' + coregistration + '/'
    results_pth = 'saved_results/' + loss+ '_' + coregistration + '/'
    
    svd_mod_pth = 'saved_models/' + loss+ '_' + coregistration + '/' + arch + '/'
    svd_res_pth = 'saved_results/' + loss + '_' + coregistration + '/' + arch + '/'

    # code snippet to make folders for saved models and results if they do not already exist

    isdir = os.path.isdir(model_pth)
    if isdir == False:
        os.makedirs(model_pth)
    isdir = os.path.isdir(results_pth)
    if isdir == False:
        os.makedirs(results_pth)
    isdir = os.path.isdir(svd_mod_pth)
    if isdir == False:
        os.makedirs(svd_mod_pth)
    isdir = os.path.isdir(svd_res_pth)
    if isdir == False:
        os.makedirs(svd_res_pth)
        



#     split data
    (
        train_features, test_features, train_labels, test_labels
    ) = split_data(dataset, random_state)

    normalizer = {}
    variable_list = list(train_features)
    for variable_name in variable_list:
        normalizer[variable_name] = preprocessing.Normalization(input_shape=[1,], axis=None)
        normalizer[variable_name].adapt(np.array(train_features[variable_name]))

    normalizer['ALL'] = preprocessing.Normalization(axis=-1)
    normalizer['ALL'].adapt(np.array(train_features))

#      DNN model
    dnn_model = {}
    model = {}
    model_history = {}

    
    # set up model with  normalized data and defined layer architecture

    
    model = build_dnn_model(
        normalizer['ALL'], 0.01, layer_1, layer_2,loss = loss
    )

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
    model_history = model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        callbacks = [callback],
        verbose=0, 
        epochs=2000
    )
    df = pd.DataFrame(model_history.history)


    if writeToFile:



        history_filename = (
            svd_res_pth +
            str(random_state)
        )


        df.to_pickle(  history_filename + '.pkl' )


        model_filename =  (
            svd_mod_pth + 
            str(random_state)
        )


        model.save(  model_filename  )

#         return history_filename, model_filename
    
    else:
        return model, df, normalizer
    

    

def load_dnn_model(
    model_loc
):
    
    dnn_model = tf.keras.models.load_model(model_loc)
    
    return dnn_model
    
     
        
        
'''
Workflow functions
'''



def build_model_ensemble(
    data, coregistration, verbose = True
):
    
    
    
    # build models
    RS = range(0,25,1)
    print(data)
    layer_1_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    layer_2_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    loss_functions = ['mse','mae']
    for loss in loss_functions:
        for layer_2_input in (layer_2_list):
            for layer_1_input in (layer_1_list):
                if layer_1_input <= layer_2_input:
                    pass
                elif layer_1_input > layer_2_input:

                    arch = str(layer_1_input) + '-' + str(layer_2_input)
                    dropout = True
                    print(
                        'Running regression with coregistration ' + 
                        str(coregistration) + 
                        ', layer architecture = ' +
                        arch + ' compiled using ' + loss
                         )

                    for rs in tqdm(RS):
                        isdir = (
                            'saved_models/' + loss+ '_' + coregistration + 
                            '/' + arch + '/' + str(rs) + '/'
                        )
                        if os.path.isdir(isdir) == True:
                            pass
                        elif os.path.isdir(isdir) == False:

                            build_and_train_model(

                                data, 
                                loss,
                                coregistration = coregistration, 
                        #                             res = coregistration,
                                layer_1 = layer_1_input,
                                layer_2 = layer_2_input,
                                random_state = rs, 

                            )  

                    
def assess_model_performance(data, coregistration = '1'):
    rootdirs = [
        'saved_models/mse_' + coregistration + '/', 
        'saved_models/mae_' + coregistration + '/'
    ]

    print('loading and evaluating models...')

    for rootdir in rootdirs:
        
        if 'mse' in rootdir:
            loss = 'mse'
        if 'mae' in rootdir:
            loss = 'mae'
        
#         is
        
        mp = pd.DataFrame()
        ms = pd.DataFrame()

        mod_stats = 'temp/model_statistics_' + loss + '_' + coregistration + '.pkl'
        mod_preds = 'temp/model_predictions_' + loss + '_' + coregistration  +  '.pkl'


        if os.path.isfile(mod_stats) == True and os.path.isfile(mod_preds) == True:
            print('Models evaluated, moving on')
            pass

        elif os.path.isfile(mod_stats) == False or os.path.isfile(mod_preds) == False:
            for arch in tqdm(os.listdir(rootdir)):       
                pth = os.path.join(rootdir, arch)
                for folder in (os.listdir(pth)):
                    model_loc = (
                        rootdir + 
                        arch + 
                        '/' + 
                        folder
                    )
                    model_name = folder
                    dnn_model = load_dnn_model(model_loc)
                    df = evaluate_model(
                        arch, model_name, data, dnn_model, coregistration, loss
                    )

                    mp = pd.concat([mp, df], ignore_index = True)
            mp.rename(columns = {0:'avg train thickness'},inplace = True)
            mp.to_pickle(
                'temp/model_predictions_' + loss + '_' + coregistration  +  '.pkl'
            )
            ms = pd.DataFrame()
            for arch in mp['layer architecture'].unique():
                dft = mp[mp['layer architecture'] == arch]
                ms_temp = pd.DataFrame(
                    {
                        'layer architecture':[dft['layer architecture'].iloc[-1]],
                        'loss':[dft['loss'].iloc[-1]],
                        'train loss avg':[np.mean(dft['train loss'])],
                        'test loss avg':[np.mean(dft['test loss'])],
                        'parameters':[dft['total parameters'].iloc[-1]],
                        'inputs':[(len(data) * (len(data.columns) -1))],
                        'trained parameters':[dft['total parameters'].iloc[-1] - (
                            len(data.columns) + (len(data.columns) - 1)
                        )]

                    }
                )
                ms = pd.concat([ms, ms_temp])
            ms.to_pickle(
                'temp/model_statistics_' + loss + '_' +
                coregistration + 
                '.pkl'
            )


        

def evaluate_model(
    arch,
    rs,
    data,
    dnn_model,
    coregistration,
    loss,

):

    (
        train_features, test_features, train_labels, test_labels
    ) = split_data(
        data, random_state = int(rs)
    )
    
    features = pd.concat([train_features,test_features], ignore_index = True)
    labels = pd.concat([train_labels, test_labels], ignore_index = True)
    

    loss_train = dnn_model.evaluate(
        train_features, train_labels, verbose=0
    )
    loss_test = dnn_model.evaluate(
        test_features, test_labels, verbose=0
    )    


    df = features
    thicknesses = (dnn_model.predict(features, verbose = 0).flatten())
    df['model'] = rs
    df['loss'] = loss
    df['test loss'] = loss_test
    df['train loss'] = loss_train
    df['layer architecture'] = str(arch)
    df['coregistration'] = coregistration
    df['total parameters'] = dnn_model.count_params() 
    df['Thickness'] = labels
    df['Estimated Thickness'] = thicknesses
    df['Residual'] = df['Estimated Thickness'] - df['Thickness']
    
    

    df['trained parameters'] = df['total parameters'] - (
        len(data.columns) + (len(data.columns) - 1)
    )

#     df['total inputs'] = 

    return df


'''

'''
def calculate_model_avg_statistics(
    dnn_model,
    dataset,
    model_thicknesses,
    loss,
    architecture
):
    df = pd.DataFrame({
                'Line1':[1]
    })
    test_loss_avg = np.mean(model_thicknesses['test loss'])

    train_loss_avg = np.mean(model_thicknesses['train loss'])

    df.loc[
        df.index[-1], 'layer architecture'
    ] = architecture

                        

    
    df.loc[
        df.index[-1],  'loss' 
    ] = loss
    
    df.loc[
        df.index[-1], 'test loss avg'
    ] = test_loss_avg
    
    df.loc[
        df.index[-1], 'train loss avg'
    ] = train_loss_avg
    df['total parameters'] = dnn_model.count_params() 

    df['trained parameters'] = df['total parameters'] - (
        len(dataset.columns) + (len(dataset.columns) - 1)
    )
    df['total inputs'] = (len(dataset) * (len(dataset.columns) -1))
    
    df = df.dropna()

#     df = df.sort_values('test ' + loss + ' avg')
    df = df.drop('Line1', axis = 1)

    return df



def estimate_thickness(
        arch_list,
#         arch,
        coregistration = '1',
        verbose = True,
        useMP = False,
        
    ):
#     RGI = load_RGI(pth = '/home/prethicktor/data/RGI/rgi60-attribs/')
    RGI = load_RGI(pth = '/data/fast1/glacierml/data/RGI/rgi60-attribs/')

    RGI['region'] = RGI['RGIId'].str[6:8]
    RGI = RGI.reset_index()
    RGI = RGI.drop('index', axis=1)
    
    loss_functions = ['mse','mae']
    for loss in loss_functions:
    
        if useMP == False:
            print('Estimating thicknesses')
            for arch in tqdm(arch_list):
                make_estimates(
                    RGI,
                    coregistration, 
                    verbose,
                    arch,
                    loss
                )


        else:
            arch = model_statistics['layer architecture']
            from functools import partial
            import multiprocessing
            pool = multiprocessing.pool.Pool(processes=5) 

            newfunc = partial(
                make_estimates,
                RGI,
                coregistration, 
                verbose
    #             arch
            )
            output = pool.map(newfunc, arch.unique())
    #     print(output[1])
    #     for i in arch:
    #         print(output[i])


        
def make_estimates(
    RGI,
    coregistration,
    verbose,
    arch,
    loss
):
    
    file_name = 'temp/RGI_predicted_' + loss + '_' +  coregistration + '_' + arch + '.pkl' 

#     print(file_name)
    
#     mod_preds = 'temp/model_predictions_' + loss + '_' + coregistration  +  '.pkl'


    if os.path.isfile(file_name) == True:
#         print('Models evaluated, moving on')
        pass
    if os.path.isfile(file_name) == False:
#     if verbose: print(f'Estimating RGI with layer architecture {arch}, coregistration {coregistration}')
        dfs = pd.DataFrame()
        RGI_for_predictions = RGI[[
            'CenLon', 'CenLat', 'Slope', 'Zmin', 'Zmed', 'Zmax', 'Area', 'Aspect', 'Lmax'
        ]]

    #     .drop(['region', 'RGIId'], axis = 1)
#         print(f'Estimating RGI thicknesses with architecture {arch}, coregistration ' +
#               f'{coregistration}, compiled using {loss}')
        for rs in (range(0,25,1)):
            rs = str(rs)
    #         results_path = 'saved_results/' + loss + '_' + coregistration + '/' + arch + '/'
    #         history_name = rs
    #         dnn_history = {}
    #         dnn_history[rs] = pd.read_pickle(results_path + rs +'.pkl')
            model_path = (
                'saved_models/' + loss + '_' + coregistration + '/' + arch + '/' + rs
            )

            dnn_model = tf.keras.models.load_model(model_path)

            s = pd.Series(
                dnn_model.predict(RGI_for_predictions, verbose=0).flatten(), 
                name = rs
            )
            dfs[rs] = s

        RGI_prethicked = RGI.copy() 
        RGI_prethicked = pd.concat([RGI_prethicked, dfs], axis = 1)

        RGI_prethicked.to_pickle(
            'temp/RGI_predicted_' + loss + '_' + 
            coregistration + '_' + arch + '.pkl'          
        )    



def compile_model_weighting_data(coregistration, arch_list,loss):
    
    path = 'model_weights/'
#     for j in tqdm(reversed(range(1,5,1))):
    file = path + 'param' + coregistration + '_' + loss + '_weighting_data.pkl'  
    file_path = file
#     print(file_path)
    if os.path.isfile(file_path) == True:
        print('Weighting data compiled')
        pass
    if os.path.isfile(file_path) == False:
        glac = coregister_data(coregistration)
        dft = pd.DataFrame()
        print('Compiling model estimates...')
        for architecture in tqdm(arch_list):
        #     print(architecture)
            df_glob = load_global_predictions(coregistration,loss, architecture = architecture)
            dft = pd.concat([dft, df_glob])
        dft.to_hdf('predicted_thicknesses/compiled_raw_' + loss + '_' + coregistration + '.h5', 
          key = 'compiled_raw', mode = 'a')
        df = dft[[
                'layer architecture','RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24',
        ]]

        glathida_estimates = pd.merge(glac, df, how = 'inner', on = 'RGIId')

        est = glathida_estimates

        for i in range(0,25,1):
            est['pr_'+str(i)] = ((np.round(est[str(i)], 0) - est['Thickness'])) / est['Thickness']

        for i in range(0,25,1):
            est['r_'+str(i)] = ((np.round(est[str(i)], 0) - est['Thickness']))

        est.to_pickle('model_weights/param' + coregistration + '_' + loss + '_weighting_data.pkl')




def compute_model_weights(coregistration, loss, pth = '/home/prethicktor/data/'):
    path = 'model_weights/'
    file = path + 'architecture_weights_' + loss + '_' + coregistration  +  '.pkl'   
    if os.path.isfile(file) == True:
        architecture_weights = pd.read_pickle(file)
        residual_model = np.load('model_weights/residual_model_' + loss + '_' + 
                                 coregistration + '.npy',)
    if os.path.isfile(file) == False:
    

        est = pd.read_pickle('model_weights/param' + coregistration +
                             '_' + loss +  '_weighting_data.pkl')
        model_list = []
        res_list = []
        pool_list = []
        weight_list = []
        for i in range(0,25,1):
            model_list.append(str(i))
            res_list.append('r_'+str(i))
            pool_list.append('pr_'+str(i))
            weight_list.append('w_'+str(i))
        est[model_list] = np.round(est[model_list], 0)
        est[res_list] = np.round(est[model_list], 0)
        est[pool_list] = np.round(est[pool_list], 2)

        
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

        architecture_weights.to_pickle('model_weights/architecture_weights_' + loss + '_' + 
                                       coregistration + '.pkl')
#         residual_model.to_pickle('model_weights/residual_model_' + coregistration + '.pkl')
        
        np.save(
            'model_weights/residual_model_' + loss + '_' + coregistration,
            residual_model, 
            allow_pickle=True, 
            fix_imports=True)

        
        
        
        
        
        
    return architecture_weights, residual_model





def calculate_RGI_thickness_statistics(
    architecture_weights, residual_model, 
    coregistration,  loss,
    useMP = False
):
    
    aggregate_statistics(architecture_weights, residual_model, coregistration, loss,
                        useMP = useMP)


# def agg_stats():
#     df = pd.read_hdf(
#         'predicted_thicknesses/compiled_raw_' + coregistration + '.h5',
#         key = 'compiled_raw', mode = 'a'
#     )    
#     arch_list = df['layer architecture'].unique()
#     model_list = []
#     for i in range(0,25,1):
#         model_list.append(str(i))
    
#     df = df.to_numpy()
    
    
def aggregate_statistics(
    architecture_weights, 
    residual_model, 
    coregistration, 
    loss,
    verbose = True,
    useMP = False
):
    arch_list = architecture_weights['layer architecture']
    final_pth  = ('predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_' + 
                   str(loss) + '_' + str(coregistration) + '.pkl') 
    if os.path.isfile(final_pth) == True:
        print('Already done here')
        pass
    if os.path.isfile(final_pth) == False:       
        if useMP == False:
            print('Stacking estimates...')
            _,compiled_raw = stack_predictions(
                architecture_weights, coregistration, loss
            )
            print('Estimates stacked')
            stacked_stats = pd.DataFrame()
            print('Applying weights...')
            for this_rgi_id, obj in tqdm(compiled_raw): 
                rgiid = np.array([this_rgi_id])
                pr = np.array(obj[[
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                    '11','12','13','14','15','16','17','18','19','20','21',
                    '22','23','24',
                ]])

                aw = np.array(
                    [
                        obj['aw_1'], 
                        obj['aw_2'], 
                        obj['aw_3'], 
                        obj['aw_4']
                    ]
                )
                wt = weight_thicknesses(pr, aw)
                gamma = np.array([
                    obj['IQR_1'] / 1.34896, 
                    obj['IQR_2'] / 1.34896, 
                    obj['IQR_3'] / 1.34896, 
                    obj['IQR_4'] / 1.34896
                ])

                uwds, muwds, wds, wb, weu  = estimate_uncertainties(gamma, pr, aw)

                wrc, wrc_unc = compute_residual_correction(
                    wt,wds, pr, residual_model, gamma, aw
                )

                gc = np.array([
                    len(pr.flatten())
                ])
                stats_table = pd.DataFrame(
                    {'RGIId':[rgiid.flatten()], 'WT1':[wt[0]],'WT2':[wt[1]],'WT3':[wt[2]],
                     'WT4':[wt[3]],
                     'UWDS1':[uwds[0]],'UWDS2':[uwds[1]],'UWDS3':[uwds[2]],'UWDS4':[uwds[3]],
                     'MUWDS1':[muwds[0]],'MUWDS2':[muwds[1]],'MUWDS3':[muwds[2]],'MUWDS4':[muwds[3]],
                     'MUWDS5':[muwds[4]],
                     'WDS1':[wds[0]],'WDS2':[wds[1]],'WDS3':[wds[2]],'WDS4':[wds[3]],
                     'WB1':[wb[0]],'WB2':[wb[1]],'WB3':[wb[2]],'WB4':[wb[3]],
                     'WEU1':[weu[0]],'WEU2':[weu[1]],'WEU3':[weu[2]],'WEU4':[weu[3]],
                     'WRC1':[wrc[0]],'WRC2':[wrc[1]],'WRC3':[wrc[2]],'WRC4':[wrc[3]],
                     'WRC_UNC1':[wrc_unc[0]],'WRC_UNC2':[wrc_unc[1]],'WRC_UNC3':[wrc_unc[2]],
                     'WRC_UNC4':[wrc_unc[3]],'GC':[gc]
                    })

                stacked_stats = pd.concat((stacked_stats, stats_table), axis = 0)
            stacked_stats.to_pickle(
                'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_' + 
                loss + '_' + coregistration + '.pkl'
            ) 
        if useMP == True:
            crunch_numbers()
        
        
        
        
def crunch_numbers():
    import multiprocessing as mp

    print('loading df')
    df = pd.read_hdf(
        'predicted_thicknesses/compiled_raw_' + '4' + '.h5',
        key = 'compiled_raw', mode = 'a'
    )
    # df_index_list = []
    # for i in range(0, len(df), 3):
    #     df_index_list.append(i)

    # df = df.loc[
    #     df_index_list
    # ]

    print('df loaded')

    grp_lst_args = list(df.groupby('RGIId').groups.items())
    print('df grouped')



    model_list = []
    for i in range(0,25,1):
        model_list.append(str(i))



    pool = mp.Pool(processes = (32))
    for _ in tqdm(pool.imap_unordered(calc_dist2,grp_lst_args,df),total = len(grp_lst_args)):
        pass
    results = pool.map(calc_dist2, grp_lst_args)

    pool.close()
    pool.join()

    stacked_stats = np.concatenate(results)
    np.save(
        'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_' + 
        loss + '_' + coregistration + '.npy', stacked_stats
    ) 
    print(stacked_stats)

def calc_dist2(arg, df):
    stacked_stats = np.empty(shape = [1,31])
    grp, lst = arg
#     print('working on ' + grp)
    dft = df.loc[lst]
    aw = np.array(
            [
                dft['aw_1'], 
                dft['aw_2'], 
                dft['aw_3'], 
                dft['aw_4']
            ]
        )
    predictions = df[model_list].loc[lst].to_numpy()   
    wt = gl.weight_thicknesses(predictions, aw)

    gamma = np.array([
        dft['IQR_1'] / 1.34896, 
        dft['IQR_2'] / 1.34896, 
        dft['IQR_3'] / 1.34896, 
        dft['IQR_4'] / 1.34896
    ])

    uwds, muwds, wds, wb  = gl.estimate_uncertainties(gamma, predictions, aw)
    residual_model = np.load('model_weights/residual_model_' + coregistration + '.npy')

    wrc, wrc_unc = gl.compute_residual_correction(wt,wds, bar_H, residual_model, gamma, aw)

    gc = np.array([
        [len(predictions.flatten())]
    ])
    prg = np.array([[grp]])
    list_a = [prg, wt, uwds, muwds, wds, wb, wrc, wrc_unc, gc]



    stats_table = np.concatenate(
        (prg, wt, uwds, muwds, wds, wb, wrc, wrc_unc, gc), axis = 1
    )
    return stats_table
    
def stack_predictions(architecture_weights, coregistration = '4', loss = 'first'):
    pth = str(
        'predicted_thicknesses/compiled_raw_' + str(loss) + '_' + str(coregistration) + '.h5'
    )
    print(pth)
    statistics = pd.DataFrame()
#     for file in (os.listdir('zults/')):
#         if 'statistics_'  +  coregistration in file:
#             file_reader = pd.read_pickle('zults/' + file)
#             statistics = pd.concat([statistics, file_reader], ignore_index = True)
#     print(statistics)
    if os.path.isfile(pth) == True:
        print('Loading df')
        df = pd.read_hdf(
            'predicted_thicknesses/compiled_raw_' + loss + '_' + coregistration + '.h5',
            key = 'compiled_raw', mode = 'a'
        )
#         df = pd.merge(df, statistics, on = 'layer architecture')
        df = df[[
                'layer architecture','RGIId','0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24'
        ]]
        df = pd.merge(df, architecture_weights, how = 'inner', on = 'layer architecture')
        
        
    if os.path.isfile(pth) == False:
        print('Building df')
        df = pd.DataFrame(columns = [
                'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24',
        ])
        for arch in tqdm(architecture_weights['layer architecture'].unique()):
            df_glob = load_global_predictions(
                coregistration = coregistration,
                architecture = arch
            )

            df = pd.concat([df,df_glob])

#         df = pd.merge(df, statistics, on = 'layer architecture')
        df = df[[
                'layer architecture','RGIId','0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9','10',
                '11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24'
        ]]
        df = pd.merge(df, architecture_weights, how = 'inner', on = 'layer architecture')
        
#         print('Grouping predictions')

        
    compiled_raw = df.groupby('RGIId')[[
        'layer architecture','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24','aw_1','aw_2','aw_3','aw_4','IQR_1','IQR_2','IQR_3','IQR_4'
    ]]

    return df,compiled_raw



def weight_thicknesses(pr, aw):
    bar_H = pr.mean(axis = 1)
    weighted_thicknesses = np.array([
        sum( (bar_H) / (aw[:,:][0]) ) / sum(1/aw[:,:][0]),
        sum( (bar_H) / (aw[:,:][1]) ) / sum(1/aw[:,:][1]),
        sum( (bar_H) / (aw[:,:][2]) ) / sum(1/aw[:,:][2]),
        sum( (bar_H) / (aw[:,:][3]) ) / sum(1/aw[:,:][3]),

    ]).T
    return weighted_thicknesses


def estimate_uncertainties(gamma, pr, aw):
    
    bar_H = pr.mean(axis = 1)
    boot = pr.var(axis = 1)
    deviation_sigma = gamma * bar_H

    wds = np.array([
        sum( (deviation_sigma[0])**2 / (aw[:,:][0]) ) / sum(1/aw[:,:][0]),
        sum( (deviation_sigma[1])**2 / (aw[:,:][1]) ) / sum(1/aw[:,:][1]),
        sum( (deviation_sigma[2])**2 / (aw[:,:][2]) ) / sum(1/aw[:,:][2]),
        sum( (deviation_sigma[3])**2 / (aw[:,:][3]) ) / sum(1/aw[:,:][3]),

    ]).T
#     print(wds)
    uwds = np.array([
       1 / sum(deviation_sigma[0]**2),
       1 / sum(deviation_sigma[1]**2),
       1 / sum(deviation_sigma[2]**2),
       1 / sum(deviation_sigma[3]**2)
        
    ]).T
    muwds = np.array(
        [
            1 / sum(
                1/((gamma[0][0:3] * bar_H[0:3])**2)
            ),
            1 / sum(
                1/((gamma[0][0:32] * bar_H[0:32])**2)
            ),
            1 / sum(
                1/((gamma[0][0:64] * bar_H[0:64])**2)
            ),
            1 / sum(
                1/((gamma[0][0:96] * bar_H[0:96])**2)
            ),
            1 / sum(
                1/((gamma[0][0:128] * bar_H[0:128])**2)
            ),
        ]
    )  
    wb = np.array([
        sum( (boot) / (aw[:,:][0]) ) / sum(1/aw[:,:][0]),
        sum( (boot) / (aw[:,:][1]) ) / sum(1/aw[:,:][1]),
        sum( (boot) / (aw[:,:][2]) ) / sum(1/aw[:,:][2]),
        sum( (boot) / (aw[:,:][3]) ) / sum(1/aw[:,:][3]),

    ])
    
    lhs = np.array([
        pr.T * gamma[0],
        pr.T * gamma[1],
        pr.T * gamma[2],
        pr.T * gamma[3]
    ])**2    
    
    weu = np.array([
        sum( np.mean(lhs,axis = 1)[0] / (aw[:,:][0]) ) / sum(1/aw[:,:][0]),
        sum( np.mean(lhs,axis = 1)[1] / (aw[:,:][1]) ) / sum(1/aw[:,:][1]),
        sum( np.mean(lhs,axis = 1)[2] / (aw[:,:][2]) ) / sum(1/aw[:,:][2]),
        sum( np.mean(lhs,axis = 1)[3] / (aw[:,:][3]) ) / sum(1/aw[:,:][3]),

    ]).T
    
    return uwds, muwds, wds, wb, weu

def compute_residual_correction(wt,wds, predictions, residual_model, gamma, aw):
    bar_H = predictions.mean(axis = 1)
    rc = residual_model[0]*bar_H**2 + residual_model[1]*bar_H + residual_model[2]
    wrc = np.array(
        [
            sum(rc / aw[:,:][0]) / sum(1/aw[:,:][0]),
            sum(rc / aw[:,:][1]) / sum(1/aw[:,:][1]),
            sum(rc / aw[:,:][2]) / sum(1/aw[:,:][2]),
            sum(rc / aw[:,:][3]) / sum(1/aw[:,:][3]),

        ]
    )

    sigma_rc = np.array([
        gamma[0] * rc[0],
        gamma[1] * rc[1],
        gamma[2] * rc[2],
        gamma[3] * rc[3],
    ])
    wrc_unc = np.array([
        sum(sigma_rc[0]**2 / aw[:,:][0]) / sum(1/aw[:,:][0]),
        sum(sigma_rc[1]**2 / aw[:,:][1]) / sum(1/aw[:,:][1]),
        sum(sigma_rc[2]**2 / aw[:,:][2]) / sum(1/aw[:,:][2]),
        sum(sigma_rc[3]**2 / aw[:,:][3]) / sum(1/aw[:,:][3])
    ])
    


    wrc[np.where(wrc <= 0)] =  (-1 * wrc[np.where(wrc <= 0)]) + wt[np.where(wrc <= 0)]
    wrc_unc[np.where(wrc <= 0)]  = wrc_unc[np.where(wrc <= 0)] 

#     print(wds)
#     print(wrc)
    wrc[np.where(wrc > 0)] =  0 + wt[np.where(wrc > 0)]

    wrc_unc[np.where(wrc > 0)] = 0 + wds[np.where(wrc > 0)] 

    return wrc, wrc_unc

    

# def ci__weighter(mean_thickness, mean_ci, var, var_ci, coregistration = '4'):
#     weights = np.load(
#         'model_weights/architecture_weights_' + coregistration +'.pkl', allow_pickle = True
#     )
#     weight1 = weights['aw_1']
#     weight2 = weights['aw_2']
#     weight3 = weights['aw_3']
#     weight4 = weights['aw_4']
#     weights_1 = np.tile(weights['aw_1'], (2,1)).T
#     weights_2 = np.tile(weights['aw_2'], (2,1)).T
#     weights_3 = np.tile(weights['aw_3'], (2,1)).T
#     weights_4 = np.tile(weights['aw_4'], (2,1)).T
#     t = np.array(
#         [
#             (sum(mean_thickness/weight1) / sum(1/weight1)),
#             (sum(mean_thickness/weight2) / sum(1/weight2)),
#             (sum(mean_thickness/weight3) / sum(1/weight3)),
#             (sum(mean_thickness/weight4) / sum(1/weight4)),
#         ]
#     )
#     tu = np.array(
#         [
#             (sum(mean_ci/weights_1) / sum(1/weights_1)),
#             (sum(mean_ci/weights_2) / sum(1/weights_2)),
#             (sum(mean_ci/weights_3) / sum(1/weights_3)),
#             (sum(mean_ci/weights_4) / sum(1/weights_4))
#         ]
#     )
    
#     return t,tu



# def calculate_confidence_intervals(predictions):
#     mean = []
#     mean_ci = []
#     mean_ci_width = []

#     var = []
#     var_ci = []
#     var_ci_width = []

#     std = []
#     std_ci = []
#     std_ci_width = []
#     for i in range(0,161,1):
#     #     print(pr[i][:])
#     #     break
#         mean_i, var_i, std_i = st.bayes_mvs(np.array(predictions)[i][:], alpha=0.95)


#         mean_i_width = mean_i[1][1] - mean_i[1][0]
#         std_i_width = std_i[1][1] - std_i[1][0]
#         var_i_width = var_i[1][1] - var_i[1][0]

#         mean.append(mean_i[0])
#         mean_ci.append(mean_i[1])
#         mean_ci_width.append(mean_i_width)

#         std.append(std_i[0])
#         std_ci.append(std_i[1])
#         std_ci_width.append(std_i_width)

#         var.append(var_i[0])
#         var_ci.append(var_i[1])
#         var_ci_width.append(var_i_width)
        
#     return mean, mean_ci, var, var_ci

'''
'''
def list_architectures(
    coregistration = '1'
):
    root_dir = 'temp/'
    arch_list = pd.DataFrame()
    for file in (os.listdir(root_dir)):
        
        if 'RGI_predicted_' + coregistration in file :
            file_reader = pd.read_pickle(root_dir + file)
            arch = pd.Series(file[16:-4])
            arch_list = pd.concat([arch_list, arch], ignore_index = True)
            arch_list = arch_list.reset_index()
            arch_list = arch_list.drop('index', axis = 1)
#             arch_list.loc[arch_list.index[-1], 'coregistration'] = coregistration
#             arch_list.loc[arch_list.index[-1], 'arch'] = arch
    
    arch_list = arch_list.rename(columns = {
        0:'layer architecture'
    })
    arch_list = arch_list.drop_duplicates()
    return arch_list




def load_global_predictions(
    coregistration,
    loss,

    architecture,
    pth = 'temp/'
):
    
#     print(architecture)
    root_dir = pth
    for file in (os.listdir(root_dir)):
            # print(file)
        if ('RGI_predicted_' + loss + '_' + coregistration + '_' + architecture in file):
            RGI_predicted = pd.read_pickle(root_dir + file)

            RGI_predicted['layer architecture'] = architecture

            RGI_predicted['coregistration'] =  coregistration


    return RGI_predicted



def load_notebook_data(
    coregistration = '1',loss = 'first', pth = ''
):
    df = pd.read_pickle(
            pth + 'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_'+
            loss +  '_' + coregistration + '.pkl'
    )
    df['RGIId'] = df['RGIId'].str[0]
    df['GC'] = df['GC'].str[0]
#     print(df)
    df = df.rename(
        columns = {
            0:'RGIId', 1:'WT1', 2:'WT2', 3:'WT3', 4:'WT4', 5:'UWDS1', 6:'UWDS2', 7:'UWDS3',8:'UWDS4',
            9:'MUWDS1', 10:'MUWDS2', 11:'MUWDS3', 12:'MUWDS4', 13:'MUWDS5',
            14:'WDS1', 15:'WDS2', 16:'WDS3', 17:'WDS4', 18:'WB1', 19:'WB2', 20:'WB3', 21:'WB4',
            22:'WEU1', 23:'WEU2', 24:'WEU3', 25:'WEU4',
            26:'WRC1', 27:'WRC2', 28:'WRC3', 29:'WRC4',
            30:'WRC_UNC1', 31:'WRC_UNC2', 32:'WRC_UNC3', 33:'WRC_UNC4',
            34:'GC'
        }
    )
    df = df.reset_index()
    df = df.drop('index', axis = 1)

    
    
    
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
    
    reference_path = 'reference_thicknesses/'
    ref = pd.DataFrame()
    for file in os.listdir(reference_path):
        if 'Farinotti' in file:
            file_reader = pd.read_csv('reference_thicknesses/' + file)
            ref = pd.concat([ref, file_reader], ignore_index = True) 
    ref['region'] = ref['RGIId'].str[6:8]
    ref = ref.sort_values('RGIId')

#     ref['Farinotti Volume (km3)'] = (ref['Farinotti Mean Thickness'] / 1e3 )* ref['Area']

    ref['region'] = ref['RGIId'].str[6:8]
#     ref = ref.dropna(subset = ['Farinotti Mean Thickness'])
#     print(ref)
    ref = ref.rename(columns = {
         'Farinotti Mean Thickness':'FMT',
#          'Shapiro-Wilk statistic':'FSW stat',
#          'Shapiro-Wilk p_value':'FSW p',
#          'Farinotti Mean Thickness':'FMT',
#          'Median Thickness':'Farinotti Median Thickness',
#          'Thickness STD':'Farinotti Thickness STD',
#          'Skew':'Farinotti Skew',
#          'Farinotti Volume (km3)'
    })
#     print(ref)
    ref = ref[[
         'FMT',
#          'FSW stat',
#          'FSW p',
#          'FMT',
#          'Farinotti Median Thickness',
#          'Farinotti Thickness STD',
#          'Farinotti Skew',
         'RGIId',
#          'Farinotti Volume (km3)'
    ]]
#     print(ref)

    df = df[[
         'RGIId',
         'WT1',
         'WT2',
         'WT3',
         'WT4',
         'UWDS1',
         'UWDS2',
         'UWDS3',
         'UWDS4',
         'MUWDS1',
         'MUWDS2',
         'MUWDS3',
         'MUWDS4',
         'MUWDS5',
         'WDS1',
         'WDS2',
         'WDS3',
         'WDS4',
         'WB1',
         'WB2',
         'WB3',
         'WB4',
         'WEU1',
         'WEU2',
         'WEU3',
         'WEU4',
         'WRC1',
         'WRC2',
         'WRC3',
         'WRC4',
         'WRC_UNC1',
         'WRC_UNC2',
         'WRC_UNC3',
         'WRC_UNC4',
         'GC',
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
    ]]
    
    df = pd.merge(df, ref, on = 'RGIId', how = 'inner')
#     print(df)
    number_list = []
    item_list = ['WRC_UNC','WDS','WB','WEU','UWDS','MUWDS']
    column_list = []
    for k in range(1,5,1):
        number_list.append(k)
    for item in item_list:
        for k in number_list:
            column_string = item + str(k)
            column_list.append(column_string)

    col_string1 = item + str(5)
    column_list.append(col_string1)
    
    df[column_list] = np.sqrt(df[column_list])
    
    
    
    rounding_list = []
    item_list = ['WT','WRC','WRC_UNC','WDS','WB','WEU','UWDS','MUWDS']
    column_list = []
    for k in range(1,5,1):
        rounding_list.append(k)
    for item in item_list:
        for k in rounding_list:
            column_string = item + str(k)
            column_list.append(column_string)

    col_string1 = item + str(5)
    col_string2 = 'FMT'
    column_list.append(col_string1)
    column_list.append(col_string2)
    
    df[column_list] = np.round(df[column_list], 0)
    df['FMT'][df['FMT'] ==0] = 0.1
    
    return df



def assign_arrays(
    coregistration = '4',method = '1', loss = 'first',
    size_thresh_1 = 1e-5, size_thresh_2 = 1e4,
    new_data = True
):
    data = load_notebook_data(coregistration, loss)
    data = data.dropna(subset = 'FMT')
#     print(list(data))

#     thick_est_unc = (
#         data['Weighted Mean Thickness ' + method].to_numpy() + 
#         data['Weighted Deviation Uncertainty_' + method].to_numpy() + 
#         data['MAE Uncertainty'].to_numpy()
#     )

    thick_est = data['WT'+ method].to_numpy()

    thick_far = data['FMT'].to_numpy()

    area = data['Area'].to_numpy()
    
    pd_index = data[
        (data['WT' + method] * data['Area'] >= size_thresh_1) &
        (data['WT' + method] * data['Area'] <= size_thresh_2) &
        (data['FMT'] * data['Area'] >= size_thresh_1) &
        (data['FMT'] * data['Area'] <= size_thresh_2)
    ].index
    
    x = thick_far / 1e3 * area
    y = thick_est / 1e3 * area
#     unc = np.sqrt(thick_est_unc) / 1e3 * area


#         print(x.max())
#         print(y.max())
        
        
        
    index = np.where(
        (x<size_thresh_2)&(x>size_thresh_1)&(y<size_thresh_2)&(y>size_thresh_1)
    )
    x_new = x[index]
    y_new = y[index]
#     unc_new = unc[index]
    
    pth = 'arrays/'+coregistration+method+ '_' + loss +'_vol_density.npy'
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
        'arrays/'+coregistration+method+ '_' + loss + '_vol' + 
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
        
    
    return x,y,z,x_new,y_new,z_new,data,pd_index


def assign_sub_arrays(
    est_ind,i,j,
    coregistration = '4',method = '1', loss = 'first',
    feature = 'Area'
    
#     size_thresh_1 = 1e-5, size_thresh_2 = 1e4,
):
    data = load_notebook_data(coregistration, loss)
    data = data.dropna(subset = 'FMT')

    data = data.loc[est_ind]
#     data['Slope'][data['Slope'] == -9] = np.nan
#     data['Lmax'][data['Lmax'] == -9] = np.nan
#     data['Zmin'][data['Zmin'] == -999] = np.nan
#     data['Zmax'][data['Zmax'] == -999] = np.nan
#     data['Zmed'][data['Zmed'] == -999] = np.nan

    data = data.drop(
        data[
            (data['Lmax'] == -9) |
            (data['Slope'] == -9)|
            (data['Zmin'] == -999) |
            (data['Zmax'] == -999) |
            (data['Zmed'] == -999)
        ].index, axis = 0
    )
#     thick_est_unc = (
#         data['WMT ' + method].to_numpy() + 
#         data['WDS' + method].to_numpy() + 
#         data['MAE Uncertainty'].to_numpy()
#     )

    thick_est = data['WT'+ method].to_numpy()

    thick_far = data['FMT'].to_numpy()

    feat = data[feature].to_numpy()


    x = thick_far
    y = thick_est
#     unc = np.sqrt(thick_est_unc) 

#         print(x.max())
#         print(y.max())
        
        
    st1 = np.floor(x.min())
    st2 = np.ceil(x.max())


    
    
    pth = (
        'arrays/'+coregistration+method+ '_' + loss + '_thickness_z_'+
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
#         print(xy)
        z = gaussian_kde(xy)(xy)
        np.save(pth, z)
        
        
        
        
        
    pth_f = (
        'arrays/'+coregistration+method+'_' + loss + '_' + feature + '-thickness_f_'+
        str(np.floor(np.min(x))) + '-' + str(np.ceil(np.max(x)))+ 
        '-' + str(i) + '-' + str(j) + '_density.npy'
    )
    pth_e = (
        'arrays/'+coregistration+method+'_' + loss +'_'+ feature + '-thickness_e_'+
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
#         print(xy)
        zf = gaussian_kde(xy)(xy)
        np.save(pth_f,zf)
        
        xy = np.vstack([np.log10(y),np.log10(feat)])
        ze = gaussian_kde(xy)(xy)
        np.save(pth_e,ze)
    data['Slope'] = data['Slope'] + .00001
    data['Zmin'] = data['Zmin'] + .00001
    return x,y,z,zf,ze,unc, data, feat



'''
Residual Distrubtion Functions
'''
def find_glacier_resid(df):

    dfr = pd.DataFrame()
    for i in (range(1,4,1)):
        x = pd.DataFrame(
                pd.Series(
                    (df[str(i)] - df['Thickness']),
                    name = 'Residual'
            )
        )
        
        x_ = pd.DataFrame(
                pd.Series(
                    (df[str(i)] - df['Thickness']) / df['Thickness'],
                    name = 'Percent Residual'
            )
        )
#         x_ = x.divide(df['Thickness'])
        y = pd.DataFrame(
            pd.Series(
                df['Thickness'],
                name = 'Thickness'
            )
        )
        l = pd.DataFrame(
            pd.Series(
                df['Lmax'],
                name = 'Lmax'
            )
        )
        a = pd.DataFrame(
            pd.Series(
                df['Area'],
                name = 'Area'
            )
        )

        s = pd.DataFrame(
            pd.Series(
                df['Slope'],
                name = 'Slope'
            )
        )

        e = pd.DataFrame(
            pd.Series(
                df['Zmin'],
                name = 'Zmin'
            )
        )

        r = pd.DataFrame(
            pd.Series(
                df['RGIId'],
                name = 'RGIId'
            )
        )
        

        f = pd.DataFrame(
            pd.Series(
                df['CenLon'],
                name = 'Lon'
            )
        )
        

        g = pd.DataFrame(
            pd.Series(
                df['CenLat'],
                name = 'Lat'
            )
        )
        h = pd.DataFrame(
            pd.Series(
                df['Zmax'],
                name = 'Zmax'
            )
        )
        

        j = pd.DataFrame(
            pd.Series(
                df['Aspect'],
                name = 'Aspect'
            )
        )
        dft = x.join(y)
        dft = dft.join(x_)
        dft = dft.join(l)
        dft = dft.join(a)
        dft = dft.join(s)
        dft = dft.join(e)
        dft = dft.join(r)
        dft = dft.join(f)
        dft = dft.join(g)
        dft = dft.join(h)
        dft = dft.join(j)

        dfr = pd.concat([dfr, dft])
    return dfr



def findlog(x):
    if x > 0:
        log = math.log(x)
    elif x < 0:
        log = math.log(x*-1)*-1
    elif x == 0:
        log = 0
    return log



def find_variances(df, ml):
    variances = pd.DataFrame()
    for rgi_id in tqdm(df['RGIId'].unique()):
        temp_df = df[df['RGIId'] == rgi_id]
        x = pd.DataFrame(
                pd.Series(
                    temp_df[ml].var(axis = 1),
                    name = 'Variance'
            )
        )

    #     print(x)
        a = pd.DataFrame(
            pd.Series(
                temp_df['Area'],
                name = 'Area'
            )
        )
        b = pd.DataFrame(
            pd.Series(
                temp_df['Slope'],
                name = 'Slope'
            )
        )
        c = pd.DataFrame(
            pd.Series(
                temp_df['Lmax'],
                name = 'Lmax'
            )
        )
        d = pd.DataFrame(
            pd.Series(
                temp_df['Zmin'],
                name = 'Zmin'
            )
        )
        e = pd.DataFrame(
            pd.Series(
                temp_df['Thickness'],
                name = 'Thickness'
            )
        )
        f = pd.DataFrame(
            pd.Series(
                temp_df['RGIId'],
                name = 'RGIId'
            )
        )
        g = pd.DataFrame(
            pd.Series(
                temp_df['Thickness'],
                name = 'Thickness'
            )
        )

    #     print(y)
        another_temp_df = x.join(a)
    #     print(dft)
    #     another_temp_df = another_temp_df.join(a)
        another_temp_df = another_temp_df.join(b)
        another_temp_df = another_temp_df.join(c)
        another_temp_df = another_temp_df.join(d)
        another_temp_df = another_temp_df.join(e)
        another_temp_df = another_temp_df.join(f)
        variances = pd.concat([variances, another_temp_df])
    return variances


def variance_min_max(df, ml):
    minvar = pd.DataFrame()
    maxvar = pd.DataFrame()
    for i in tqdm(df['RGIId'].unique()):
        dft = df[df['RGIId'] == i]
        f = pd.Series(dft[ml].var(axis = 1),name = 'Variance')
        vmin = pd.DataFrame(
            pd.Series(
                    f.min(),
                    name = 'VarMin'
            )
        )
    #     print(var_rmin)
        vmax = pd.DataFrame(
            pd.Series(
                    f.max(),
                    name = 'VarMax'
            )
        )
        rgi = pd.DataFrame(
            pd.Series(
                    i,
                    name = 'RGIId'
            )
        )
        a = pd.DataFrame(
            pd.Series(
                    dft['Area'].min(),
                    name = 'Area'
            )
        )
        b = pd.DataFrame(
            pd.Series(
                    dft['Lmax'].min(),
                    name = 'Lmax'
            )
        )
        c = pd.DataFrame(
            pd.Series(
                    dft['Slope'].min(),
                    name = 'Slope'
            )
        )
        d = pd.DataFrame(
            pd.Series(
                    dft['Zmin'].min(),
                    name = 'Zmin'
            )
        )
        e = pd.DataFrame(
            pd.Series(
                    dft['Thickness'].min(),
                    name = 'Thickness'
            )
        )
    #     print(rgi)
        var_min = vmin.join(rgi)
        var_min = var_min.join(a)
        var_min = var_min.join(b)
        var_min = var_min.join(c)
        var_min = var_min.join(d)
        var_min = var_min.join(e)
        minvar = pd.concat([minvar,var_min])

        var_max = vmax.join(rgi)
        var_max = var_max.join(a)
        var_max = var_max.join(b)
        var_max = var_max.join(c)
        var_max = var_max.join(d)
        var_max = var_max.join(e)

        maxvar = pd.concat([maxvar,var_max])
    return minvar, maxvar

def sample_coregistration_data(c = '3'):

    tr = coregister_data(c)
    
#     if c == '4':
#         tr = tr.drop(tr[tr['Thickness'] >= 300].index)
#         tr = tr.drop(tr[tr['Thickness'] == 267].index)
#         tr = tr.drop(tr[tr['Thickness'] == tr['Thickness'].min()].index)
    tr = tr.drop('Thickness', axis = 1)
    rfp = load_RGI()[list(tr)]
    feat_list = ['Area','Lmax','Slope','Zmin']
    name = ['mean', 'median', 'min', 'max','IQR','STD']
    df1 = pd.DataFrame( columns = feat_list, index = name)
    for i in feat_list:
        df1t = tr[i]
        upp = np.nanquantile(df1t, 0.75)
        low = np.nanquantile(df1t, 0.25)
        functions = [
            np.round(np.nanmean(df1t), 3),
            np.round(np.nanmedian(df1t), 3), 
            np.round(np.nanmin(df1t), 3),
            np.round(np.nanmax(df1t), 3),
            np.round(upp - low, 3),
            np.round(np.nanstd(df1t),3),
    #         len(df1t)
        ]
        for n, fn in zip(name, functions):
            df1[i].loc[n] = fn
    df1 = df1.rename(columns = {
        'Area':'Area (km$^2$)',
        'Slope':'Slope (deg)',
        'Lmax':'Max Length (m)',
        'Zmin':'Min Elevation (m)',
    })
    df1 = df1.round(decimals = 3)
    df1

    name = ['mean', 'median', 'min', 'max','IQR','STD']
    df2 = pd.DataFrame( columns = feat_list, index = name)
    for i in feat_list:
        df2t = rfp[i]
        upp = np.nanquantile(df2t, 0.75)
        low = np.nanquantile(df2t, 0.25)
    #         print(np.quantile(df2t, 0.75))
    #         print(np.quantile(df2t, 0.25))
        functions = [
            np.round(np.nanmean(df2t), 3),
            np.round(np.nanmedian(df2t), 3), 
            np.round(np.nanmin(df2t), 3),
            np.round(np.nanmax(df2t), 3),
            np.round(upp - low, 3),
            np.round(np.nanstd(df2t),3),
    #         len(df2t)
        ]
        for n, fn in zip(name, functions):
            df2[i].loc[n] = fn
    df2 = df2.rename(columns = {
        'Area':'Area (km$^2$)',
        'Slope':'Slope (deg)',
        'Lmax':'Max Length (m)',
        'Zmin':'Min Elevation (m)',
    #         'WT1':'Est Thick (m)',
    #         'Vol Diff':'Vol Diff (km$^3$)'
    #         'Farinotti Mean Thickness':'Farinotti Thickness'
    })
    df2 = df2.round(decimals = 3)
    df2


    perc_samples = (df1 - df2) / df2
    return perc_samples



def sample_training_data(tr,rs = 0):
#     tr = coregister_data(c)
    trfeats, tefeats, trlabs, telabs = split_data(tr,rs)
    
    feat_list = ['Area','Lmax','Slope','Zmin']
    name = ['mean', 'median', 'min', 'max','IQR','STD']
    df1 = pd.DataFrame( columns = feat_list, index = name)
    for i in feat_list:
#         print(tefeats)
        df1t = tefeats[i]
        upp = np.nanquantile(df1t, 0.75)
        low = np.nanquantile(df1t, 0.25)
        functions = [
            np.round(np.nanmean(df1t), 3),
            np.round(np.nanmedian(df1t), 3), 
            np.round(np.nanmin(df1t), 3),
            np.round(np.nanmax(df1t), 3),
            np.round(upp - low, 3),
            np.round(np.nanstd(df1t),3),
    #         len(df1t)
        ]
        for n, fn in zip(name, functions):
            df1[i].loc[n] = fn
    df1 = df1.round(decimals = 3)
    df1

    name = ['mean', 'median', 'min', 'max','IQR','STD']
    df2 = pd.DataFrame( columns = feat_list, index = name)
    for i in feat_list:
        df2t = trfeats[i]
        upp = np.nanquantile(df2t, 0.75)
        low = np.nanquantile(df2t, 0.25)
        functions = [
            np.round(np.nanmean(df2t), 3),
            np.round(np.nanmedian(df2t), 3), 
            np.round(np.nanmin(df2t), 3),
            np.round(np.nanmax(df2t), 3),
            np.round(upp - low, 3),
            np.round(np.nanstd(df2t),3),
    #         len(df2t)
        ]
        for n, fn in zip(name, functions):
            df2[i].loc[n] = fn
    df2 = df2.round(decimals = 3)
    df2
    perc_samples = abs((df1 - df2) / df2)
    
    return perc_samples

def sub_sample_training_data(data1,data2,rs = 0):
#     trfeats, tefeats, trlabs, telabs = split_data(tr,rs)
    
    feat_list = ['Area','Lmax','Slope','Zmin']
    name = ['mean', 'median', 'min', 'max','IQR','STD']
    df1 = pd.DataFrame( columns = feat_list, index = name)
    for i in feat_list:
#         print(tefeats)
        df1t = data1[i]
        upp = np.nanquantile(df1t, 0.75)
        low = np.nanquantile(df1t, 0.25)
        functions = [
            np.round(np.nanmean(df1t), 3),
            np.round(np.nanmedian(df1t), 3), 
            np.round(np.nanmin(df1t), 3),
            np.round(np.nanmax(df1t), 3),
            np.round(upp - low, 3),
            np.round(np.nanstd(df1t),3),
    #         len(df1t)
        ]
        for n, fn in zip(name, functions):
            df1[i].loc[n] = fn
    df1 = df1.round(decimals = 3)
#     df1

    name = ['mean', 'median', 'min', 'max','IQR','STD']
    df2 = pd.DataFrame( columns = feat_list, index = name)
    for i in feat_list:
        df2t = data2[i]
        upp = np.nanquantile(df2t, 0.75)
        low = np.nanquantile(df2t, 0.25)
        functions = [
            np.round(np.nanmean(df2t), 3),
            np.round(np.nanmedian(df2t), 3), 
            np.round(np.nanmin(df2t), 3),
            np.round(np.nanmax(df2t), 3),
            np.round(upp - low, 3),
            np.round(np.nanstd(df2t),3),
    #         len(df2t)
        ]
        for n, fn in zip(name, functions):
            df2[i].loc[n] = fn
    df2 = df2.round(decimals = 3)
    df2
    perc_samples = (df1 - df2) / df2
    
    return perc_samples



    

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
