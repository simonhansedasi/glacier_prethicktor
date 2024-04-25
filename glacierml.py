# import sys
# !{sys.executable} -m pip install 
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from geopy.distance import geodesic
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
np.random.seed(42)



def set_paths(home_path):
    
    data_path = os.path.join(home_path,'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    RGI_path = os.path.join(data_path,'RGI')
    if not os.path.exists(RGI_path):
        os.makedirs(RGI_path)
        
    glathida_path = os.path.join(data_path,'glathida')
    if not os.path.exists(glathida_path):
        os.makedirs(glathida_path)
    
    
    model_path = os.path.join(home_path,'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    
    coregistration_testing_path = os.path.join(model_path,'coregistration_testing')
    if not os.path.exists(coregistration_testing_path):
        os.makedirs(coregistration_testing_path)
    
    arch_test_path = os.path.join(model_path,'arch_testing')
    if not os.path.exists(arch_test_path):
        os.makedirs(arch_test_path)
    
    LOO_path = os.path.join(model_path,'LOO')
    if not os.path.exists(LOO_path):
        os.makedirs(LOO_path)    
    return [
        data_path, RGI_path, glathida_path, 
        coregistration_testing_path, 
        arch_test_path, LOO_path
    ]



'''
Load RGI -- loops through the unzipped CSV attribute files of RGI glaciers and loads them into a pandas dataframe.
'''
def load_RGI(RGI_path):
    df = pd.DataFrame()
    for RGI_file in os.listdir(RGI_path):
        if RGI_file.endswith('.csv'):
            dft = pd.read_csv(
                os.path.join(RGI_path,RGI_file),
                encoding_errors = 'replace', 
                on_bad_lines = 'skip'
            )
            df = pd.concat([df,dft],axis = 0)
    df = df.sort_values('RGIId').reset_index()
    # df['region'] = df['RGIId'].str[6:8]
    df = pd.concat([df,pd.Series(df['RGIId'].str[6:8],name = 'region')],axis = 1)
    df = df.drop('index',axis = 1)
    return df


# def load_RGI(
#     pth = '/home/simonhans/glacierml/data/RGI/', 
#     region_selection = 'all'
# ):
#     if len(str(region_selection)) == 1:
#         N = 1
#         region_selection = str(region_selection).zfill(N + len(str(region_selection)))
#     else:
#         region_selection = region_selection
        
#     RGI_extra = pd.DataFrame()
#     for file in (os.listdir(pth)):
#         if file.endswith('.csv'):
# #         print(file)
#             region_number = file[:2]
#             if str(region_selection) == 'all':
#                 file_reader = pd.read_csv(pth + file, 
#                                           encoding_errors = 'replace', 
#                                           on_bad_lines = 'skip'
#                                          )
#                 RGI_extra = pd.concat([RGI_extra,file_reader], ignore_index = True)
                
#             elif str(region_selection) != str(region_number):
#                 pass
            
#             elif str(region_selection) == str(region_number):
#                 file_reader = pd.read_csv(pth + file, encoding_errors = 'replace', on_bad_lines = 'skip')
#                 RGI_extra = pd.concat([RGI_extra,file_reader], ignore_index = True)
            
#     RGI = RGI_extra
# #     [[
# #         'RGIId',
# #         'CenLat',
# #         'CenLon',
# #         'Slope',
# #         'Zmin',
# #         'Zmed',
# #         'Zmax',
# #         'Area',
# #         'Aspect',
# #         'Lmax',
# #         'Name',
# #         'GLIMSId',
# #     ]]
#     RGI['region'] = RGI['RGIId'].str[6:8]
# #     RGI['Area'] = np.log10(RGI['Area']
#     return RGI



'''
GlaThiDa_RGI_index_matcher: uses multiprocessing or not. Go through GlaThiDa,
and search for the nearest RGI glacier. Measure the distance and append.
'''
def match_GlaThiDa_RGI_index(
#     pth = '/data/fast1/glacierml/data/',
    RGI,glathida,
    verbose = False,
    useMP = False
):
    # 

    
    if useMP == False:
        centroid_distances = []
        RGI_ids = []
        for i in tqdm(glathida.index):
            RGI_id_match, centroid_distance = get_id(RGI,glathida,verbose,i)
            centroid_distances.append(centroid_distance)
            RGI_ids.append(RGI_id_match)
        df = pd.concat(
            [
                glathida,
                pd.Series(RGI_ids,name = 'RGIId'),
                pd.Series(centroid_distances,name = 'centroid distance')
            ],axis = 1
        )
        
            
        return df
            
    else:
        from functools import partial
        import multiprocessing
        pool = multiprocessing.Pool(processes=4)         # create a process pool with 4 workers
        newfunc = partial(get_id,RGI,glathida,verbose) #now we can call newfunc(i)
        output = pool.map(newfunc, glathida.index)
        for i in tqdm(glathida.index):      

            glathida.loc[glathida.index[i], 'RGIId'] = output[i][0]
            glathida.loc[glathida.index[i], 'RGI Centroid Distance'] = output[i][1]
        return glathida
    
    
''' 
Works with match function above.
'''
def get_id(RGI,glathida,verbose,i):
    if verbose: print(f'Working on Glathida index {i}')
    #obtain lat and lon from glathida 
    glathida_ll = (glathida.loc[i].LAT,glathida.loc[i].LON)

    # find distance between selected glathida glacier and all RGI
    distances = RGI.apply(
        lambda row: geodesic((row.CenLat,row.CenLon),glathida_ll),
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
Loads matched pickle file, merges with RGI attributes and keeps relevant information.
'''
def load_training_data(data_path):  
    # Read matched dataset and select data
    df = pd.read_pickle(os.path.join(data_path,'matched.pkl'))
    df = df[[
        'RGIId','RGI Centroid Distance','SURVEY_DATE',
        'AREA','MEAN_THICKNESS_UNCERTAINTY','GLACIER_NAME','MEAN_THICKNESS'
    ]]
    df = df.rename(columns = {
        'MEAN_THICKNESS':'Thickness',
    })
    
    # Load RGI to get RGI attributes
    RGI_path = os.path.join(data_path,'RGI')
    RGI = load_RGI(RGI_path)

    # Put the data together
    df = pd.merge(df,RGI,how = 'inner', on = 'RGIId')
    
    return df

# def load_training_data(
#     form,
#     pth = '/home/simonhans/glacierml/data/',
# #     alt_pth = '/home/prethicktor/data/',
#     RGI_input = 'y',
#     scale = 'g',
#     region_selection = 1,
#     area_scrubber = 'off',
#     anomaly_input = 0.5,
#     area = '',
    
# #     data_version = 'v1'
# ):        
#     import os
#     pth_1 = os.path.join(pth, 'T_data/')
#     pth_2 = os.path.join(pth, 'RGI/rgi60-attribs/')
#     pth_3 = os.path.join(pth, 'matched_indexes/', 'v2')
#     pth_4 = os.path.join(pth, 'regional_data/training_data/', 'v2')
    
    
#     pth_5 = pth_3 + '/GlaThiDa_with_RGIId_' + 'v2' + '.csv'                        
                                 
#     # load glacier GlaThiDa data v2
#     glacier = pd.read_csv(pth_1 + 'T.csv', low_memory = False)    
#     glacier = glacier.rename(columns = {
#         'LAT':'Lat',
#         'LON':'Lon',
#         'AREA':'area_g',
#         'MEAN_SLOPE':'Mean Slope',
#         'MEAN_THICKNESS':'Thickness'
#     })   
#     glacier = glacier.dropna(subset = ['Thickness'])

# #         print('# of raw thicknesses: ' + str(len(glacier)))
        
        
#     # keep it just GlaThiDa
#     if RGI_input == 'n':
#         df = glacier.rename(columns = {
#             'Mean Slope':'Slope'
#         }, inplace = True)
#         df = glacier[[
#             'Lat',
#             'Lon',
#             'area_g',
#             'Slope',
#             'Thickness',
#             'SURVEY_DATE'
#         ]]
#         df = df.rename(columns = {
#             'area_g':'Area'
#         })
# #         df = df.dropna()        
#         return df

#     # add in RGI attributes
#     elif RGI_input == 'y':
#         RGI = load_RGI(pth = os.path.join(pth, 'RGI/'))
# #         print(RGI)
#         RGI['region'] = RGI['RGIId'].str[6:8]
    
#         # drop glacier complexes
# #         RGI = RGI.drop(RGI[RGI['Status'] == 1].index)
        
#         # separate out glaciers
#         if form == 'glacier':
#             RGI = RGI[RGI['Form'] == 0]
            
#         # separate out ice caps
#         if form == 'cap':
#             RGI = RGI[RGI['Form'] == 1]
        
        
        
#         # load glacier GlaThiDa data v2
#         glacier = pd.read_csv(pth_5)    
#         glacier = glacier.rename(columns = {
#             'LAT':'Lat',
#             'LON':'Lon',
#             'AREA':'Area_GlaThiDa',
#             'MEAN_SLOPE':'Mean Slope',
#             'MEAN_THICKNESS':'Thickness'
#         })   
#         glacier = glacier.dropna(subset = ['Thickness'])
# #         glacier = pd.read_csv(pth_5)
#         df = pd.merge(RGI, glacier, on = 'RGIId', how = 'inner')
        
        
        
        
#         glacier = glacier.dropna(subset = ['RGIId'])
# #         print(glacier)
#         rgi_matches = len(glacier)
#         rgi_matches_unique = len(glacier['RGIId'].unique())
        
        
# #         df = df.rename(columns = {
# #             'name':'name_g',
# #             'Name':'name_r',

# #             'BgnDate':'date_r',
# #             'date':'date_g'
# #         })

#         # make a temp df for the duplicated entries
        
#         # calculate the difference in size as a percentage
#         df['perc smaller'] = (
#             ( (df['Area'] - df['Area_GlaThiDa']) )/ df['Area_GlaThiDa'] )
        
#         df['size difference'] = (
#             ( (df['Area'] - df['Area_GlaThiDa']) ))
                       
# #         df = df.rename(columns = {'Area':'Area_RGI',})
#         df = df[[
#             'RGIId',
#             'CenLat',
#             'CenLon',
# #             'Lat',
# #             'Lon',
#             'Area',
# #             'Area_RGI',
# #             'Area_GlaThiDa',
#             'Zmin',
#             'Zmed',
#             'Zmax',
#             'Slope',
#             'Aspect',
#             'Lmax',
#             'Thickness',
# #             'area_g',
#             'region',
#             'perc smaller',
#             'size difference',
# #             'index_x',
# #             'index_y',
#             'RGI Centroid Distance',
#             'SURVEY_DATE',
#             'BgnDate'
#         ]]
        
#         if area_scrubber == 'on':          
#             df = df[(df['perc smaller'] <= anomaly_input) & 
#                     (df['perc smaller'] >= -anomaly_input)]
# #             df = df.drop([
# #                 'size difference',
# # #                 'Area_y'
# #             ], axis = 1)
# #             df = df.rename(columns = {
# #                 'Area_x':'Area'
# #             })

#             df = df[[
#                 'RGIId',
# #                     'Lat',
# #                     'Lon',
#                 'CenLat',
#                 'CenLon',
#                 'Slope',
#                 'Zmin',
#                 'Zmed',
#                 'Zmax',
#                 'Area',
# #                 'Area_RGI',
#                 'Aspect',
#                 'Lmax',
#                 'Thickness',
#                 'region',
#                 'RGI Centroid Distance',
#                 'perc smaller',
#                 'size difference',
#                 'SURVEY_DATE',
# #                 'Area_GlaThiDa',
#                 'BgnDate'
#             ]]
    

# #     if area == 'R':
# #         df = df.drop('Area_GlaThiDa', axis = 1)
# #     if area == 'G':
# #         df = df.drop('Area',axis = 1)
# #         df = df.rename(columns = {'Area_GlaThiDa':'Area'})
# #         df = df.rename(columns = {'Area_GlaThiDa':'Area'})
# #     print(len(df))
#     # convert everything to common units (m)
#     df['RGI Centroid Distance'] = df['RGI Centroid Distance'].str[:-2].astype(float)
#     df['RGI Centroid Distance'] = df['RGI Centroid Distance'] * 1e3
# #     df = df.rename(columns = {
# #         'Area_GlaThiDa':'Area'
# #     })
#     df['Area'] = df['Area'] * 1e6     # Put area to meters for radius and roundness calc

#     # make a guess of an average radius and "roundness" -- ratio of avg radius / width
#     df['AVG Radius'] = np.sqrt(df['Area'] / np.pi)
#     df['Roundness'] = (df['AVG Radius']) / (df['Lmax'])
#     df['distance test'] = df['RGI Centroid Distance'] / df['AVG Radius']
    
    
#     df['Area'] = df['Area'] / 1e6     # Put area back to sq km
# #     df['Area'] = df['Area'] - df['size difference']
# #     df['Area'] = np.log10(df['Area'])
# #     df['Lmax'] = np.log10(df['Lmax'])
# #     df = df.rename(columns = {'Area_GlaThiDa':'Area'})
#     return df

'''
apply uncertainty thresholds to training data
'''
def coregister_data(
    data_path,
    coregistration = '1',
):
    
    df = load_training_data(data_path)
    
    
    import configparser
    config = configparser.ConfigParser()
    config.read('model_coregistration.txt')   
    
    
    df['RGI Centroid Distance'] = df['RGI Centroid Distance'].astype(str).str[:-2].astype(float)
    df['RGI Centroid Distance'] = df['RGI Centroid Distance'] * 1e3

    rad = np.sqrt((df['Area']*1e6) / np.pi)
    radius = pd.Series(rad,name = 'AVG Radius').astype(float)
    
    df = pd.concat([df,radius],axis = 1)
    distance_test = pd.Series(
        df['RGI Centroid Distance'] / df['AVG Radius'],
        name = 'distance test'
    )
    
    
    df = pd.concat([df,distance_test],axis = 1)
    
    df = df.drop(
        df[df['distance test'] >= float(config[coregistration]['distance threshold'])].index
    )
    
    # Look at size difference between data
    ps = pd.Series(
        (df['Area'] - df['AREA']) / df['AREA'],
        name = 'perc difference'
    )
    df = pd.concat([df,ps],axis = 1)
    area_scrubber = config[coregistration]['area scrubber']
    if area_scrubber == 'on':
        size_threshold = float(config[coregistration]['size threshold'])
        df = df[
            (df['perc difference'] <= size_threshold) & 
            (df['perc difference'] >= -size_threshold)
        ]
    
    df = df.drop(df[df['RGIId'].duplicated(keep = False)].index)

    df = df[[
        'RGIId','CenLat','CenLon','Slope','Zmin','Zmed','Zmax',
        'Area','Aspect','Lmax','Thickness'
    ]]
    df = df.sort_values('RGIId')
    return df




# def coregister_data(coregistration = '1',form = '', pth = '/data/fast1/glacierml/data/',area = 'R'):
#     import configparser
#     config = configparser.ConfigParser()
#     config.read('model_coregistration.txt')

#     data = load_training_data(
#         form,
#         pth = pth,
#         area_scrubber = config[coregistration]['area scrubber'],
#         anomaly_input = float(config[coregistration]['size threshold']),
#         area = area,
#     )


#     data = data.drop(
#         data[data['distance test'] >= float(config[coregistration]['distance threshold'])].index
#     )
#     data = data.drop([
# #         'RGIId',
#         'region', 
#         'RGI Centroid Distance', 
#         'AVG Radius', 
#         'Roundness', 
# #         'distance test', 
# #         'size difference'
#     ], axis = 1)
    
# #     sd = data['SURVEY_DATE'].astype(str)
#     data = data.reset_index().drop('index',axis = 1)
#     data['SURVEY_DATE'] = data['SURVEY_DATE'].astype(str)

#     data['SURVEY_DATE'] = data['SURVEY_DATE'].str[:-6]
#     data['SURVEY_DATE'][data['SURVEY_DATE'] == ''] = '1953'
#     rgi_date = data['BgnDate'].astype(str).str[:-4]
#     gla_date  = data['SURVEY_DATE']
# #     sd[sd == ''] = 1950
#     tp = pd.Series((rgi_date.values.astype(int) - gla_date.values.astype(int)),name = 'YearsBetween')
# #     data = data.reset_index().drop('index',axis = 1)
# #     print(tp)
#     data = pd.concat([data,tp],axis = 1)
#     data['YearsBetween'][data['YearsBetween'] == 0] = 0.1
#     data['CR'] = (data['perc smaller']) / data['YearsBetween']
# #     data = data.drop(['SURVEY_DATE','BgnDate','YearsBetween'],axis = 1)
# #     data['CR'][data['CR'] == np.inf] = 0.5
    
#     data = data.sort_values('RGIId')
#     return data

    

        
'''

input = name of dataframe and selected random state.
output = dataframe and series randomly selected and populated as either training or test features or labels
'''
# Randomly selects data from a df for a given random state (usually iterated over a range of 25)
# Necessary variables for training and predictions
def split_data(df, random_state = 0,var = 'Thickness'):
    train_dataset = df.sample(frac=0.7, random_state=random_state).drop('RGIId', axis = 1)
    test_dataset = df.drop(train_dataset.index).drop('RGIId', axis = 1)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    #define label - attribute training to be picked
    train_labels = train_features.pop(var)
    test_labels = test_features.pop(var)
    
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
        model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate)
        )   
    if loss == 'mae':
        
        model.compile(
            loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(learning_rate = learning_rate)
        )      
    return model



'''
plot_loss
input = desired test results
output = loss plots for desired model
'''
def plot_loss(history):
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
    
    

def findlog(x):
    if math.isnan(x):
        log = 0
    elif x > 0:
        log = math.log(x)
    elif x < 0:
        log = math.log(-x) * -1
    else:  # x == 0
        log = 0
    return log



def load_LOO_data(
    home_path,
    include_train = False,
):

    RGI = pd.read_pickle(os.path.join(home_path,'models','LOO','rgi_est_raw.pkl'))
    #### Add Farinotti mean thickness estimates ####
    ref_pth = 'reference_thicknesses/'
    ref = pd.DataFrame()
    for file in os.listdir(ref_pth):
        if 'Farinotti' in file:
            file_reader = pd.read_csv('reference_thicknesses/' + file)
            ref = pd.concat([ref, file_reader], ignore_index = True) 
    ref = ref.rename(columns = {
         'Farinotti Mean Thickness':'FMT',
    })
    ref = ref[[
         'FMT',
         'RGIId',
    ]]

    df = pd.merge(RGI, ref, how = 'inner', on = 'RGIId')

    # train = coregister_data_reform(home_path + '/data/', '4')
    train = coregister_data(home_path + '/data/', '4')
    train = train.sample(frac = 1,random_state = 0)
    train = train.reset_index().drop('index', axis = 1)

    cols = []
    for i in range(len(train)):
        cols.append(i)
    df[cols] = np.round(df[cols],0)

    if include_train == True:
        df = pd.merge(train, df, how = 'inner', on = list(train)[:-1])
        
    return cols, df








