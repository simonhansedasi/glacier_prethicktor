import tensorflow as tf
import pandas as pd
import glacierml as gl
import numpy as np
import warnings
from tensorflow.python.util import deprecation
import os
import logging
import seaborn as sns
from tqdm import tqdm
tf.get_logger().setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)
tf.random.set_seed(42)

print('please select module: sm1, sm2, sm3, sm4, sm5, sm6, sm7, sm8, sm9')
# print(' ')
dir_list = ('sm1', 'sm2', 'sm3', 'sm4', 'sm5', 'sm6', 'sm7', 'sm8', 'sm9')
chosen_dir = input()



# while chosen_dir not in dir_list:
#     print('Please enter valid module selection: sm1, sm2, sm3, sm4, sm5, sm6, sm7', 'sm8')
#     chosen_dir = input()    

if chosen_dir == 'sm1':
    df1 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'n'
    )
    dataset = df1
    dataset.name = 'df1'
    res = 'sr1'

if chosen_dir == 'sm2':
    df2 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        area_scrubber = 'off'
    )
    df2 = df2.drop(['region'], axis = 1)
    dataset = df2
    dataset.name = 'df2'
    res = 'sr2'

if chosen_dir == 'sm3':
    df3 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
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
        area_scrubber = 'off',
    )
    df5 = df5.drop(['Zmed', 'region'], axis = 1)
    dataset = df5
    dataset.name = 'df5'
    res = 'sr5'

if chosen_dir == 'sm6':
    df6 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'off'
#                 anomaly_input = 5
    )
    df6 = df6.drop(['region'], axis = 1)
    dataset = df6
    dataset.name = 'df6'
    res = 'sr6'


if chosen_dir == 'sm7':
    df7 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        area_scrubber = 'on',
        anomaly_input = 75
    )
    df7 = df7.drop(['RGIId','Zmed', 'region'], axis = 1)
    dataset = df7
    dataset.name = 'df7'
    res = 'sr7'



if chosen_dir == 'sm8':
    df8 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        area_scrubber = 'on',
        anomaly_input = 25
    )
    df8 = df8.drop(['RGIId', 'region'], axis = 1)
    df8['Zdelta'] = df8['Zmax'] - df8['Zmin']
    dataset = df8
    dataset.name = 'df8'
    res = 'sr8'


if chosen_dir == 'sm9':
    df9 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        area_scrubber = 'on',
        anomaly_input = 50
    )
    df9= df9.drop(['RGIId', 'region'], axis = 1)
    df9['Zdelta'] = df9['Zmax'] - df9['Zmin']
    dataset = df9
    dataset.name = 'df9'
    res = 'sr9'

rootdir = 'saved_models/' + chosen_dir + '/'
(train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)
dnn_model = {}
print(' ')

dropout_input_list = ('y', 'n')
for dropout_input_iter in dropout_input_list:
    predictions = pd.DataFrame()
    deviations = pd.DataFrame()
    dropout_input = dropout_input_iter

    if dropout_input == 'y':
        dropout = '1'

    elif dropout_input == 'n':
        dropout = '0'
    print('loading and evaluating models...')
    for arch in os.listdir(rootdir):        
        if dropout == '1':
            print('layer architecture: ' + arch[3:] + ' dropout = True')

        elif dropout == '0':
            print('layer architecture: ' + arch[3:] + ' dropout = False')

        for folder in tqdm(os.listdir(rootdir + arch)):
            if '_' + dropout + '_' in folder and dataset.name + '_' in folder:
                model_loc = (
                    rootdir + 
                    arch + 
                    '/' + 
                    folder
                )

                model_name = arch[3:] + '_' + folder

                rs = gl.random_state_finder(folder)

                df = gl.predictions_maker(
                    rs = rs,
                    dropout = dropout,
                    arch = arch,
                    dataset = dataset,
                    folder = str(folder),
                    model_loc = model_loc,
                    model_name = model_name
                )

                predictions = pd.concat([predictions, df], ignore_index = True)

    predictions.rename(columns = {0:'avg train thickness'},inplace = True)
    predictions.to_csv('zults/predictions_' + dataset.name + '_' + dropout + '.csv')

    # calculate statistics

    print('calculating statistics...')
    print(' ')
    dnn_model = {}
    for epochs in list(predictions['epochs'].unique()):
        df = predictions[predictions['epochs'] == epochs]

        for dataframe in list(df['dataset'].unique()):
            dfs = df[df['dataset'] == dataframe]

            for arch in list(dfs['architecture'].unique()):
                dfsr = dfs[dfs['architecture'] == arch]



                for lr in list(dfsr['learning rate'].unique()):
                    dfsrq = dfsr[dfsr['learning rate'] == lr]


                    model_name = (
                            arch + 
                            '_' + 
                            dataset.name + 
                            '_' +
                            dropout +
                            '_dnn_MULTI_' +
                            str(lr) +
                            '_0.2_' +
                            str(100) +
                            '_0'
                    )

                    model_loc = (
                        rootdir + 
                        'sm_' +
                        arch + 
                        '/' + 
                        dataset.name + 
                        '_' +
                        dropout +
                        '_dnn_MULTI_' +
                        str(lr) +
                        '_0.2_' +
                        str(100) +
                        '_0'
                    )

                    isdir = os.path.isdir(model_loc)

                    if isdir == False:
                        print('model not here, calculating next model')
                    elif isdir == True:
                        df = gl.deviations_calculator(
                            model_loc = model_loc,
                            model_name = model_name,
                            ep = epochs,
                            arch = arch,
                            lr = lr,
                            dropout = dropout,
                            dataframe = dataframe,
                            dataset = dataset,
                            dfsrq = dfsrq
                        )

                        deviations = pd.concat(
                            [deviations, df], ignore_index = True
                        )

    deviations.to_csv(
        'zults/deviations_' + 
        dataset.name + 
        '_' + 
        dropout + 
        '.csv'
    )
