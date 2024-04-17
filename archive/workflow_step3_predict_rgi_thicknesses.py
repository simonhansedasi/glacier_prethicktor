import glacierml as gl
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.util import deprecation
import logging
import warnings
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)
tf.random.set_seed(42)
# print('currently running tensorflow version: ' + tf.__version__)

parameterization = str(  4  )
config = configparser.ConfigParser()
config.read('model_parameterization.ini')

data = gl.load_training_data(
    RGI_input = config[parameterization]['RGI_input'],
    scale = config[parameterization]['scale'],
)

data.name = config[parameterization]['datasetname'] 
data = data.drop([
    'RGIId','region', 'RGI Centroid Distance', 
    'AVG Radius', 'Roundness', 'distance test', 'size difference'
], axis = 1)

    
    
model_statistics = pd.read_csv('zults/model_statistics_' + parameterization + '.csv')
# deviations_2 = pd.read_csv('zults/deviations_' + dataset.name + '_0.csv')
# deviations = pd.concat([deviations_1, deviations_2])
model_statistics = model_statistics.reset_index()
# print(list(model_statistics))

model_statistics = model_statistics[[
'layer architecture',
# 'model parameters',
# 'total inputs',
# 'test mae avg',
# 'train mae avg',
# 'test mae std dev',
# 'train mae std dev'
]]

print('Estimating thicknesses...')

for index in (model_statistics.index):
#     print(index)
#     print(model_statistics.iloc[index])
    arch = model_statistics['layer architecture'].iloc[index]

# arch = '3-2'
    print('Estimating thicknesses with parameterization ' + parameterization + ', layer architecture = ' + arch )
    for region_selection in tqdm(range(1,20,1)):
        RGI = gl.load_RGI(
            pth = '/home/prethicktor/data/RGI/rgi60-attribs/',
            region_selection = int(region_selection)
        )
        if len(str(region_selection)) == 1:
            N = 1
            region_selection = str(region_selection).zfill(N + len(str(region_selection)))
        else:
            region_selection = region_selection

        RGI['region'] = RGI['RGIId'].str[6:8]
        RGI = RGI.reset_index()
        RGI = RGI.drop(['RGIId', 'region', 'index'], axis=1)
#         print(RGI)
        
    #         print(region_selection)
    #             if region_selection != '19':
    #                 drops = RGI[
    #                     ((RGI['region'] == str(region_selection)) & (RGI['Zmin'] < 0)) |
    #                     ((RGI['region'] == str(region_selection)) & (RGI['Zmed'] < 0)) |
    #                     ((RGI['region'] == str(region_selection)) & (RGI['Zmax'] < 0)) |
    #                     ((RGI['region'] == str(region_selection)) & (RGI['Slope'] < 0)) |
    #                     ((RGI['region'] == str(region_selection)) & (RGI['Aspect'] < 0))
    #                 ].index
    #                 print(drops)
    #                 if not drops.empty:
    #                     print('dropping bad data')
    #                     RGI = RGI.drop(drops)
#         if 'Roundness' in dataset:
#     #         RGI['Area'] = RGI['Area'] * 1e6
#             RGI['AVG Radius'] = np.sqrt((RGI['Area'] * 1e6) / np.pi)
#             RGI['Roundness'] = (RGI['AVG Radius']) / (RGI['Lmax'])
#             RGI['Area'] = np.log(RGI['Area'] * 1e3)
#     #         RGI['Area'] = RGI['Area'] / 1e6
#             RGI_for_predictions = RGI.drop(['region', 'RGIId', 'AVG Radius'], axis = 1)
#         elif 'Roundness' not in dataset:
#             RGI_for_predictions = RGI.drop(['region', 'RGIId'], axis = 1)
# #         print(RGI_for_predictions)



        dnn_model = {}
        rootdir = 'saved_models/' + parameterization + '/'
        RS = range(0,0,1)
        dfs = pd.DataFrame()
#         for rs in (RS):
        rs = str(0)

    # each series is one random state of an ensemble of 25.
    # predictions are made on each random state and appended to a df as a column
        model = (
            rs
        )

        model_path = (
            rootdir + arch + '/' + str(rs)
        )
        results_path = 'saved_results/' + parameterization + '/' + arch + '/'

        history_name = (
            rs
        )

        dnn_history ={}
        dnn_history[rs] = pd.read_csv(results_path + rs)

        if abs((
            dnn_history[history_name]['loss'].iloc[-1]
        ) - dnn_history[history_name]['val_loss'].iloc[-1]) >= 3:

            pass
        else:

            dnn_model = tf.keras.models.load_model(model_path)

            s = pd.Series(
                dnn_model.predict(RGI, verbose=0).flatten(), 
                name = rs
            )

            dfs[rs] = s


        # make a copy of RGI to add predicted thickness and their statistics
        RGI_prethicked = RGI.copy() 
        RGI_prethicked['avg predicted thickness'] = 'NaN'
        RGI_prethicked['predicted thickness std dev'] = 'NaN'
        RGI_prethicked = pd.concat([RGI_prethicked, dfs], axis = 1)

    #         print('calculating average thickness across random state ensemble...')
        # loop through predictions df and find average across each ensemble of 25 random states
        for i in (dfs.index):
            RGI_prethicked['avg predicted thickness'].loc[i] = np.mean(dfs.loc[i])


    #         print('computing standard deviations and variances for RGI predicted thicknesses')
        # loop through predictions df and find std dev across each ensemble of 25 random states
        for i in (dfs.index):
            RGI_prethicked['predicted thickness std dev'].loc[i] = np.std(dfs.loc[i])
    #         print(' ')

        RGI_prethicked.to_csv(
            'zults/RGI_predicted_' +
            parameterization + '_' + arch + '_' + str(region_selection) + '.csv'          
        )    
