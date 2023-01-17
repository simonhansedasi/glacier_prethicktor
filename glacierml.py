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

def select_dataset_coregistration(
    pth='/home/prethicktor/data/',
    parameterization='sm'
):
    root_dir = pth

    if parameterization == 'sm':
        df = load_training_data(
            root_dir = pth,
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 1,
#             data_version = 'v2'
        )
        df = df.drop([
            'RGIId','region', 'RGI Centroid Distance', 
            'AVG Radius', 'Roundness', 'distance test', 'size difference'
                       ], axis = 1)
#         df9['Area'] = df9['Area'] * 1e6
#         df9['Area'] = np.log(df9['Area'])
#         df9['Lmax'] = np.log(df9['Lmax'])
        
        
        dataset = df
        dataset.name = 'df'
        res = 'sr'
        
    if parameterization == 'sm1':
        df1 = load_training_data(
            root_dir = pth,
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = .25,
#             data_version = 'v2'
        )
        df1 = df1.drop([
            'RGIId','region', 'RGI Centroid Distance', 
            'AVG Radius', 'Roundness', 'distance test', 'size difference'
                       ], axis = 1)
#         df9['Area'] = df9['Area'] * 1e6
#         df9['Area'] = np.log(df9['Area'])
#         df9['Lmax'] = np.log(df9['Lmax'])
        
        
        dataset = df1
        dataset.name = 'df1'
        res = 'sr1'
        
        
    return parameterization, dataset, dataset.name, res



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
        print(file)
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
    
    return RGI

def load_training_data(
    root_dir = '/data/fast1/glacierml/data/',
    RGI_input = 'y',
    scale = 'g',
    region_selection = 1,
    area_scrubber = 'off',
    anomaly_input = 0.5,
#     data_version = 'v1'
):        
    import os
    pth_1 = os.path.join(root_dir, 'T_data/')
    pth_2 = os.path.join(root_dir, 'RGI/rgi60-attribs/')
    pth_3 = os.path.join(root_dir, 'matched_indexes/', 'v2')
    pth_4 = os.path.join(root_dir, 'regional_data/training_data/', 'v2')
    
    
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
        RGI = load_RGI(pth = '/home/prethicktor/data/RGI/rgi60-attribs/')
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
            if anomaly_input == .25:
                indices_to_drop_25 = [114, 122, 140, 141, 142, 244, 245, 252, 253, 254,258,
                           259,276,277,278,290,291,293,294,295,307,308,321,322,323,
                           325,326,329,330,341,342,343,432,433]
                df = df.drop(indices_to_drop_25)

#             return df
        
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
    pth_1 = os.path.join(pth, '/T_data/')
    pth_2 = os.path.join(pth, '/RGI/rgi60-attribs/')
    pth_3 = os.path.join(pth, '/matched_indexes/', version)
    
    if version == 'v1':
        glathida = pd.read_csv(pth_1 + 'glacier.csv')
        glathida = glathida.dropna(subset = ['mean_thickness'])
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
        output = pool.map(newfunc,glathida.index )
        
    for i in tqdm(glathida.index):         
        glathida.loc[glathida.index[i], 'RGIId'] = output[i][0]
        glathida.loc[glathida.index[i], 'RGI Centroid Distance'] = output[i][1]

    isdir = os.path.isdir(pth_3)
    if isdir == False:
        os.makedirs(pth_3)
    glathida.to_csv(pth_3 + 'GlaThiDa_with_RGIId_' + version + '.csv')

    
def get_id(RGI,glathida,version,verbose,i):
    if verbose: print(f'Working on Glathida ID {i}')
    #obtain lat and lon from glathida 
    if version == 'v1':
        glathida_ll = (glathida.loc[i].lat,glathida.loc[i].lon)
    if version == 'v2':
        glathida_ll = (glathida.loc[i].LAT,glathida.loc[i].LON)

    # find distance between selected glathida glacier and all RGI
    distances = RGI.apply(
        lambda row: geopy.distance.geodesic((row.CenLat,row.CenLon),glathida_ll),
            axis = 1
    )
    
#     distances = RGI.apply(
#         lambda row: geopy.distance.great_circle((row.CenLat,row.CenLon),glathida_ll),
#             axis = 1
#     )
    
#         print(distances)

    # find index of minimum distance between glathida and RGI glacier
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
#                           learning_rate = 0.01,
#                           validation_split = 0.2,
#                           epochs = 100,
                          random_state = 0,
                          parameterization = 'sm',
                          res = 'sr',
                          layer_1 = 10,
                          layer_2 = 5,
                          dropout = True,
                          verbose = False,
                          writeToFile = True
                         ):
    # define paths
    arch = str(layer_1) + '-' + str(layer_2)
    svd_mod_pth = 'saved_models/' + parameterization + '/sm_' + arch + '/'
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
    dnn_model = build_dnn_model(normalizer['ALL'], 0.01, layer_1, layer_2, dropout)
    
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

    #save model, results, and history

    if writeToFile:

        df = pd.DataFrame(dnn_history['MULTI'].history)


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
        return dnn_model
    

def load_dnn_model(
    model_name,
    model_loc
):
    
    dnn_model = {}
    dnn_model[model_name] = tf.keras.models.load_model(model_loc)
    
    return dnn_model
    
     
'''
Workflow functions
'''
def evaluate_model(
    arch,
    rs,
    dataset,
    dnn_model
):
    df = pd.DataFrame({
                'Line1':[1]
    })
    print(df)
    (
        train_features, test_features, train_labels, test_labels
    ) = split_data(
        dataset, random_state = int(rs)
    )
    
    mae_test = dnn_model.evaluate(
                    test_features, test_labels, verbose=0
                )
    mae_train = dnn_model.evaluate(
        train_features, train_labels, verbose=0
    )

    df.loc[df.index[-1], 'model'] = rs
    df.loc[df.index[-1], 'test mae'] = mae_test
    df.loc[df.index[-1], 'train mae'] = mae_train
    df.loc[df.index[-1], 'architecture'] = arch
    df.loc[df.index[-1], 'coregistration'] = dataset.name
    df.loc[df.index[-1], 'total parameters'] = dnn_model.count_params() 
    df = df.drop('Line1', axis = 1)

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



'''
'''
def find_predictions(
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
            arch = pd.Series(file[14:-7], name = 'architecture')
#             print(arch)
            prethicked = pd.concat([prethicked, arch])

            prethicked = prethicked.reset_index()
            prethicked = prethicked.drop('index', axis = 1)
            prethicked.loc[prethicked.index[-1], 'volume'] = sum(file_reader['volume km3'])
            prethicked.loc[prethicked.index[-1], 'std dev'] = sum(
                file_reader['pred std dev']
            )
            prethicked.loc[prethicked.index[-1], 'coregistration'] = coregistration

    predicted = pd.DataFrame()
            
#             break
    prethicked = prethicked.rename(columns = {
        0:'architecture'
    })
    for arch in prethicked['architecture'].unique():
        dft = prethicked[
            (prethicked['architecture'] == arch)
        ]
        dft['predicted volume'] = sum(dft['volume']) / 1e3
        dft['std dev'] = sum(dft['std dev']) / 1e3
#         print(dft.iloc[-1])
        predicted = pd.concat([predicted,dft],ignore_index = True)
    predicted = predicted[[
        'architecture',
#         'epochs',
#         'learning rate',
        'coregistration',
        'predicted volume',
        'std dev'
    ]]
    predicted = predicted.drop_duplicates()
#     prethicked = prethicked.drop_duplicates()
    return predicted




def load_global_predictions(
    coregistration,
    architecture,
):
    root_dir = 'zults/'
    RGI_predicted = pd.DataFrame()
    for file in (os.listdir(root_dir)):
            # print(file)
        if ('RGI_predicted' in file and 
            coregistration in file and
            architecture in file
           ):
            file_reader = pd.read_csv(root_dir + file)
#             print(file_reader)
            file_reader['volume km3'] = (
                file_reader['avg predicted thickness'] / 1e3
            ) * file_reader['Area']
            file_reader = file_reader.dropna()
            RGI_predicted = pd.concat([RGI_predicted, file_reader], ignore_index = True)  
            RGI_predicted['layer architecture'] =  architecture[5:]
    RGI_predicted = RGI_predicted.drop('Unnamed: 0', axis = 1)
    RGI_predicted['dataframe'] =  coregistration
    print(RGI_predicted)

    return RGI_predicted


'''
'''
# def add_glathida_stats(
#     df,
#     pth_1 = '/data/fast1/glacierml/data/regional_data/raw/',
#     pth_2 = '/data/fast1/glacierml/data/RGI/rgi60-attribs/',
#     pth_3 = '/data/fast1/glacierml/data/regional_data/training_data/',
    
# ):
#     # finish building df

#     dfa = pd.DataFrame()
#     for file in tqdm(os.listdir(pth_1)):
#         dfb = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
#         region_and_number = file[:-4]
#         region_number = region_and_number[:2]
#         region = region_and_number[3:]

#         dfb['geographic region'] = region
#         dfb['region'] = region_number
#         dfa = dfa.append(dfb, ignore_index=True)

#     dfa = dfa.reset_index()

#     dfa = dfa[[
#         'GlaThiDa_index',
#         'RGI_index',
#         'RGIId',
#         'region',
#         'geographic region'
#     ]]
#     RGI_extra = pd.DataFrame()
#     for file in os.listdir(pth_2):
#         f = pd.read_csv(pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
#         RGI_extra = pd.concat([RGI_extra, f], ignore_index = True)

#         region_and_number = file[:-4]
#         region_number = region_and_number[:2]
#         region = region_and_number[9:]
#         dfc = dfa[dfa['region'] == region_number]

#         for file in os.listdir(pth_3):
#             print(file)
#             if file[:2] == region_number:
#                 glathida_regional = pd.read_csv(pth_3 + file)

#         GlaThiDa_mean_area = glathida_regional['Area'].mean()
#         GlaThiDa_mean_aspect = glathida_regional['Aspect'].mean()
#         GlaThiDa_mean_lmax = glathida_regional['Lmax'].mean()
#         GlaThiDa_mean_slope = glathida_regional['Slope'].mean()
#         GlaThiDa_mean_zmin = glathida_regional['Zmin'].mean()
#         GlaThiDa_mean_zmax = glathida_regional['Zmax'].mean()

#         GlaThiDa_median_area = glathida_regional['Area'].median()
#         GlaThiDa_median_aspect = glathida_regional['Aspect'].median()
#         GlaThiDa_median_lmax = glathida_regional['Lmax'].median()
#         GlaThiDa_median_slope = glathida_regional['Slope'].median()
#         GlaThiDa_median_zmin = glathida_regional['Zmin'].median()
#         GlaThiDa_median_zmax = glathida_regional['Zmax'].median()

#         GlaThiDa_std_area = glathida_regional['Area'].std(ddof=0)
#         GlaThiDa_std_aspect = glathida_regional['Aspect'].std(ddof=0)
#         GlaThiDa_std_lmax = glathida_regional['Lmax'].std(ddof=0)
#         GlaThiDa_std_slope = glathida_regional['Slope'].std(ddof=0)
#         GlaThiDa_std_zmin = glathida_regional['Zmin'].std(ddof=0)
#         GlaThiDa_std_zmax = glathida_regional['Zmax'].std(ddof=0)

#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Area_GlaThiDa_mean'
#         ] = GlaThiDa_mean_area
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Aspect_GlaThiDa_mean'
#         ] = GlaThiDa_mean_aspect
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Lmax_GlaThiDa_mean'
#         ] = GlaThiDa_mean_lmax
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Slope_GlaThiDa_mean'
#         ] = GlaThiDa_mean_slope
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Zmin_GlaThiDa_mean'
#         ] = GlaThiDa_mean_zmin
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Zmax_GlaThiDa_mean'
#         ] = GlaThiDa_mean_zmax


#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Area_GlaThiDa_median'
#         ] = GlaThiDa_median_area
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Aspect_GlaThiDa_median'
#         ] = GlaThiDa_median_aspect
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Lmax_GlaThiDa_median'
#         ] = GlaThiDa_median_lmax
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Slope_GlaThiDa_median'
#         ] = GlaThiDa_median_slope
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Zmin_GlaThiDa_median'
#         ] = GlaThiDa_median_zmin
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Zmax_GlaThiDa_median'
#         ] = GlaThiDa_median_zmax




#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Area_GlaThiDa_std'
#         ] = GlaThiDa_std_area
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Aspect_GlaThiDa_std'
#         ] = GlaThiDa_std_aspect
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Lmax_GlaThiDa_std'
#         ] = GlaThiDa_std_lmax
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Slope_GlaThiDa_std'
#         ] = GlaThiDa_std_slope
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Zmin_GlaThiDa_std'
#         ] = GlaThiDa_std_zmin
#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'Zmax_GlaThiDa_std'
#         ] = GlaThiDa_std_zmax






#         trainable_ratio = (len(dfc) / len(f))
#         percent_trainable = trainable_ratio * 100

#         df.loc[
#             df[df['dataframe'].str[4:] == region_number].index, 'ratio trainable'
#         ] = trainable_ratio


#     return df


def load_notebook_data(
    coregistration = 'df8'
):
    df = pd.read_csv(
            'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_coregistration_'+
            coregistration + '.csv'
        )
    print(df)
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

#     RGI['Zdelta'] = RGI['Zmax'] - RGI['Zmin']

    df = pd.merge(df, RGI, on = 'RGIId')
#     print(df)
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
