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
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)

print('please select module: sm1, sm2, sm3, sm4, sm5, sm6, sm7')
dir_list = ('sm01', 'sm02', 'sm1', 'sm2', 'sm031', 'sm3', 'sm4', 'sm5', 'sm6', 'sm7')
chosen_dir = input()

while chosen_dir not in dir_list:
    print('Please enter valid module selection: sm1, sm2, sm3, sm4, sm5, sm6, sm7')
    chosen_dir = input()    

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
    dataset = df2
    dataset.name = 'df2'
    res = 'sr2'

if chosen_dir == 'sm3':
    df3 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        area_scrubber = 'on',
        anomaly_input = 1
    )
    dataset = df3
    dataset.name = 'df3'
    res = 'sr3'

if chosen_dir == 'sm4':
    df4 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        area_scrubber = 'on',
        anomaly_input = 5
    )
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
    df5 = df5.drop('Zmed', axis = 1)
    dataset = df5
    dataset.name = 'df5'
    res = 'sr5'
    
if chosen_dir == 'sm6':
    for region_selection in range(1,20,1):
        if len(str(region_selection)) == 1:
            N = 1
            region_selection = str(region_selection).zfill(N + len(str(region_selection)))
        else:
            region_selection = region_selection
        print(region_selection)
        region_selection = region_selection
        
        df6 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'r',
            region_selection = int(region_selection),
            area_scrubber = 'off'
        )
        if len(df6) < 3:
            pass
        if len(df6) >= 3:
            df6 = df6.drop('region', axis=1)
            dataset = df6
            dataset.name = str('df6_' + str(region_selection))
            res = 'sr6'

            rootdir = 'saved_models/' + chosen_dir + '/'
            (train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)
            dnn_model = {}

            print('loading and evaluating models...')
            dropout_input_list = ('y', 'n')
            for dropout_input_iter in dropout_input_list:
                predictions = pd.DataFrame()
                deviations = pd.DataFrame()
                dropout_input = dropout_input_iter

                if dropout_input == 'y':
                    dropout = '1'

                elif dropout_input == 'n':
                    dropout = '0'

                for arch in os.listdir(rootdir):        
                    if dropout == '1':
                        print(
                            'layer architecture: ' + arch[3:] + 
                            ' dropout = True, ' + 
                            'dataset: ' + 
                            dataset.name
                        )

                    elif dropout == '0':
                        print(
                            'layer architecture: ' + arch[3:] + 
                            ' dropout = False, ' + 
                            'dataset: ' + dataset.name
                        )

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
                dnn_model = {}
                for epochs in list(predictions['epochs'].unique()):
                    df = predictions[predictions['epochs'] == epochs]

                    for dataframe in list(df['dataset'].unique()):
                        dfs = df[df['dataset'] == dataframe]

                        for arch in list(dfs['architecture'].unique()):
                            dfsr = dfs[dfs['architecture'] == arch]

                            if dfsr.empty:
                                pass

                            for lr in list(dfsr['learning rate'].unique()):
                                dfsrq = dfsr[dfsr['learning rate'] == lr]

                                if dfsrq.empty:
                                    pass

                                if not dfsrq.empty:
                                        # find mean and std dev of test mae
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
            
if chosen_dir == 'sm6' and region_selection == 19:
    raise SystemExit
    
if chosen_dir == 'sm7':
    for region_selection in range(1,20,1):
        df7 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
    #                 region_selection = 1,
            area_scrubber = 'off'
    #                 anomaly_input = 5
        )
        dataset = df7
        dataset.name = 'df7'
        res = 'sr7'
        

rootdir = 'saved_models/' + chosen_dir + '/'
(train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)
dnn_model = {}

print('loading and evaluating models...')
dropout_input_list = ('y', 'n')
for dropout_input_iter in dropout_input_list:
    predictions = pd.DataFrame()
    deviations = pd.DataFrame()
    dropout_input = dropout_input_iter
    
    if dropout_input == 'y':
        dropout = '1'
        
    elif dropout_input == 'n':
        dropout = '0'
        
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
    dnn_model = {}
    for epochs in list(predictions['epochs'].unique()):
        df = predictions[predictions['epochs'] == epochs]
        
        for dataframe in list(df['dataset'].unique()):
            dfs = df[df['dataset'] == dataframe]
            
            for arch in list(dfs['architecture'].unique()):
                dfsr = dfs[dfs['architecture'] == arch]
                
                if dfsr.empty:
                    pass
                
                for lr in list(dfsr['learning rate'].unique()):
                    dfsrq = dfsr[dfsr['learning rate'] == lr]
                    
                    if dfsrq.empty:
                        pass
                    
                    if not dfsrq.empty:
                            # find mean and std dev of test mae
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