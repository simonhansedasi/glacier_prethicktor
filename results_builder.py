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
tf.random.set_seed(42)

module, dataset, dataset.name, res = gl.module_selection_tool()


print(dataset)
rootdir = 'saved_models/' + module + '/'
(train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)
dnn_model = {}
print(' ')

dropout_input_list = (
    'y', 
#     'n'
)
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
    print(predictions)
    # calculate statistics

    print('calculating statistics...')
    print(' ')
    dnn_model = {}
    for epochs in list(predictions['epochs'].unique()):
        df = predictions[predictions['epochs'] == epochs]

        for dataframe in list(df['coregistration'].unique()):
            dfs = df[df['coregistration'] == dataframe]

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
                            str(epochs) +
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
                        str(epochs) +
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
