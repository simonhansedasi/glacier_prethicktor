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
print('currently running tensorflow version: ' + tf.__version__)
print('please select module: sm1, sm2, sm3, sm4, sm5, sm6, sm7')

dir_list = ('sm1', 'sm2', 'sm3', 'sm4', 'sm5', 'sm6', 'sm7')

chosen_dir = input()

while chosen_dir not in dir_list:
    print('Please enter valid module selection: sm1, sm2, sm3, sm4, sm5, sm6, sm7')
    chosen_dir = input()    


if chosen_dir == 'sm1':
    df1 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'n'
#                 scale = 'g',
#                 region_selection = 1,
#                 area_scrubber = 'off',
#                 anomaly_input = 5
    )
    dataset = df1
    dataset.name = 'df1'
    res = 'sr1'
#         print(module)
#         print(dataset)

if chosen_dir == 'sm2':
    df2 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'off'
#                 anomaly_input = 5
    )
    df2 = df2.drop(['RGIId', 'region'], axis = 1)
    dataset = df2
    dataset.name = 'df2'
    res = 'sr2'

if chosen_dir == 'sm3':
    df3 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 25
    )
    df3 = df3.drop(['RGIId', 'region'], axis = 1)
    dataset = df3
    dataset.name = 'df3'
    res = 'sr3'


if chosen_dir == 'sm4':
    df4 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 75
    )
    df4 = df4.drop(['RGIId', 'region'], axis = 1)
    dataset = df4
    dataset.name = 'df4'
    res = 'sr4'

# replicate df2 and change Area to sq m
if chosen_dir == 'sm5':
    df5 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        # region_selection = 1,
        area_scrubber = 'off',
        # anomaly_input = 5
    )
    df5 = df5.drop(['RGIId','region'], axis = 1)
    df5['Zdelta'] = df5['Zmax'] - df5['Zmin']
    dataset = df5
    dataset.name = 'df5'
    res = 'sr5'



if chosen_dir == 'sm6':
    df6 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 25
#                 anomaly_input = 5
    )
    df6 = df6.drop(['RGIId','region'], axis = 1)
    df6['Zdelta'] = df6['Zmax'] - df6['Zmin']
    dataset = df6
    dataset.name = 'df6'
    res = 'sr6'

if chosen_dir == 'sm7':
    df7 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 75
    )
    df7 = df7.drop(['RGIId', 'region'], axis = 1)
    df7['Zdelta'] = df7['Zmax'] - df7['Zmin']
    dataset = df7
    dataset.name = 'df7'
    res = 'sr7'
    

    
    
    
deviations_1 = pd.read_csv('zults/deviations_' + dataset.name + '_1.csv')
# deviations_2 = pd.read_csv('zults/deviations_' + dataset.name + '_0.csv')
# deviations = pd.concat([deviations_1, deviations_2])
deviations_1 = deviations_1.reset_index()

deviations_1 = deviations_1[deviations_1['learning rate'] == 0.01]

deviations = deviations_1[[
'layer architecture',
'dropout',
# 'model parameters',
# 'total inputs',
'learning rate',
'epochs',
# 'test mae avg',
# 'train mae avg',
# 'test mae std dev',
# 'train mae std dev'
]]



# print(deviations.to_string())
# here we can select an entry from the deviations table to make predictions. Default is top entry

# print('Please select model index to predict thicknesses for RGI')
# selected_model = int(input())
# while type(selected_model) != int:
#     print('Please select model index to predict thicknesses for RGI')
#     selected_model = int(input()) 


for index in deviations.index:
    print(index)
    print(deviations.iloc[index])
    selected_model = deviations.iloc[index]
    arch = deviations['layer architecture'].iloc[index]
    lr = deviations['learning rate'].iloc[index]
    ep = deviations['epochs'].iloc[index]
    dropout = deviations['dropout'].iloc[index]

    
    
    
    for region_selection in range(1,20,1):
        RGI = gl.RGI_loader(
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
        RGI = RGI.drop('index', axis=1)
        print(region_selection)
        if region_selection != 19:
            drops = RGI[
                ((RGI['region'] == str(region_selection)) & (RGI['Zmin'] < 0)) |
                ((RGI['region'] == str(region_selection)) & (RGI['Zmed'] < 0)) |
                ((RGI['region'] == str(region_selection)) & (RGI['Zmax'] < 0)) |
                ((RGI['region'] == str(region_selection)) & (RGI['Slope'] < 0)) |
                ((RGI['region'] == str(region_selection)) & (RGI['Aspect'] < 0))
            ].index
            print(drops)
            if not drops.empty:
                print('dropping bad data')
                RGI = RGI.drop(drops)
        RGI_for_predictions = RGI.drop(['region', 'RGIId'], axis = 1)
        print(RGI['Zmed'].min())
        if chosen_dir == 'sm1':

            RGI_for_predictions = RGI_for_predictions.rename(columns = {
                'CenLat':'Lat',
                'CenLon':'Lon',
                'Area':'Area',
                'Slope':'Mean Slope'
            })
            RGI_for_predictions = RGI_for_predictions[[
                'Lat',
                'Lon',
                'Area',
                'Mean Slope'
            ]]

        RGI_for_predictions['Zdelta'] = RGI_for_predictions['Zmax'] - RGI_for_predictions['Zmin']

        print(
            'layer architecture: ' + arch + 
            ', learning rate: ' + str(lr) + 
            ', epochs: ' + str(ep) +
            ', dataset: ' + dataset.name +
            ', region: ' + str(region_selection)
        )

        print('predicting thicknesses...')
        dnn_model = {}
        rootdir = 'saved_models/' + chosen_dir + '/'
        RS = range(0,25,1)
        dfs = pd.DataFrame()
        for rs in tqdm(RS):
        # each series is one random state of an ensemble of 25.
        # predictions are made on each random state and appended to a df as a column
            model = (
                str(arch) +
                '_' +
                dataset.name +
                '_' +
                str(dropout) + 
                '_dnn_MULTI_' +
                str(lr) +
                '_' +
                str(0.2) +
                '_' +
                str(ep) + 
                '_' + 
                str(rs)
            )

            path = (
                rootdir + 'sm_' + arch + '/' + 
                dataset.name + 
                '_' + 
                str(dropout) + 
                '_dnn_MULTI_' + 
                str(lr) + 
                '_' +
                str(0.2) +
                '_' +
                str(ep) + 
                '_' + 
                str(rs)
            )
            rootdir_1 = 'saved_results/' + res + '/sr_' + arch + '/'
            dnn_history_1 = {}
            history_name_1 = (
                arch + 
                '_' +
                dataset.name +
                '_' +
                str(dropout) +
                '_dnn_history_MULTI_' +
                str(lr) +
                '_0.2_' +
                str(ep) + 
                '_' + 
                str(rs)

            )
            history_name = (
                dataset.name +
                '_' +
                str(dropout) +
                '_dnn_history_MULTI_' +
                str(lr) +
                '_0.2_' +
                str(ep) + 
                '_' + 
                str(rs)
            )
            dnn_history ={}
            dnn_history[history_name] = pd.read_csv(rootdir_1 + history_name)
            
            if abs((
                dnn_history[history_name]['loss'].iloc[-1]
            ) - dnn_history[history_name]['val_loss'].iloc[-1]) >= 3:
                pass
            else:

                dnn_model[model] = tf.keras.models.load_model(path)

                s = pd.Series(
                    dnn_model[model].predict(RGI_for_predictions, verbose=0).flatten(), 
                    name = rs
                )

                dfs[rs] = s


        # make a copy of RGI to add predicted thickness and their statistics
        RGI_prethicked = RGI.copy() 
        RGI_prethicked['avg predicted thickness'] = 'NaN'
        RGI_prethicked['predicted thickness std dev'] = 'NaN'
        RGI_prethicked = pd.concat([RGI_prethicked, dfs], axis = 1)

        print('calculating average thickness across random state ensemble...')
        # loop through predictions df and find average across each ensemble of 25 random states
        for i in tqdm(dfs.index):
            RGI_prethicked['avg predicted thickness'].loc[i] = np.mean(dfs.loc[i])


        print('computing standard deviations and variances for RGI predicted thicknesses')
        # loop through predictions df and find std dev across each ensemble of 25 random states
        for i in tqdm(dfs.index):
            RGI_prethicked['predicted thickness std dev'].loc[i] = np.std(dfs.loc[i])
        print(' ')

        RGI_prethicked.to_csv(
            'zults/RGI_predicted_' +
            dataset.name + '_' + str(region_selection) +
            '_' + 
            str(dropout) + 
            '_' + 
            arch + 
            '_' + 
            str(lr) + 
            '_' + 
            str(ep) + 
            '.csv'
        )    