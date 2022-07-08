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
from IPython.display import display, HTML
# display(HTML("<style>.container { width:85% !important; }</style>"))
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)





print('please select module: sm1, sm2, sm3, sm4')
dir_list = ('sm01', 'sm02', 'sm1', 'sm2', 'sm031', 'sm3', 'sm4', 'sm5', 'all')
chosen_dir = input()

while chosen_dir not in dir_list:
    print('Please enter valid module selection: sm1, sm2, sm3, sm4, sm5')
    chosen_dir = input()    




if chosen_dir == 'sm1':
    df1 = gl.data_loader(
        root_dir = '/home/prethicktor/data/'
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
        root_dir = '/home/prethicktor/data/'
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'off'
#                 anomaly_input = 5
    )
    dataset = df2
    dataset.name = 'df2'
    res = 'sr2'

if chosen_dir == 'sm3':
    df3 = gl.data_loader(
        root_dir = '/home/prethicktor/data/'
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 1
    )
    dataset = df3
    dataset.name = 'df3'
    res = 'sr3'

if chosen_dir == 'sm4':
    df4 = gl.data_loader(
        root_dir = '/home/prethicktor/data/'
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'on',
        anomaly_input = 5
    )
    dataset = df4
    dataset.name = 'df4'
    res = 'sr4'
    
# replicate df2 and change Area to sq m
if chosen_dir == 'sm5':
    df5 = gl.data_loader(
        root_dir = '/home/prethicktor/data/'
        RGI_input = 'y',
        scale = 'g',
        # region_selection = 1,
        area_scrubber = 'off',
        # anomaly_input = 5
    )
    dataset = df5
    dataset.name = 'df5'
    res = 'sr5'
    df5 = df5.drop('Zmed', axis = 1)

# elif chosen_dir == 'sm5':
#     df5 = gl.data_loader()
#     reg = df5['region'].iloc[-1]
#     df5 = df5.drop('region', axis=1)
#     dataset = df5
#     dataset.name = str('df5_' + str(reg))

#     #code snippet to add a leading 0 to regional ID so it matches with RGI when built later
#     if len(str(reg)) ==1:
#         N = 1
#         reg = str(reg).zfill(N + len(str(reg)))
#     else:
#         reg = reg

# elif chosen_dir == 'sm6':
#     df6 = gl.data_loader()
#     reg = df6['region'].iloc[-1]
#     df6 = df6.drop('region', axis=1)
#     dataset = df6
#     dataset.name = str('df6_' + str(reg))

#     #code snippet to add a leading 0 to regional ID so it matches with RGI when built later
#     if len(str(reg)) ==1:
#         N = 1
#         reg = str(reg).zfill(N + len(str(reg)))
#     else:
#         reg = reg

global_list = ('sm1', 'sm2', 'sm3', 'sm4', 'sm5')
region_list = ('sm6')




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
            if '_' + dropout + '_' in folder:
                dnn_model[arch[3:] + '_' + folder] = tf.keras.models.load_model(
                    rootdir + 
                    arch + 
                    '/' + 
                    folder
                )
                mae_test = dnn_model[arch[3:] + '_' + folder].evaluate(
                    test_features, test_labels, verbose=0
                )
                mae_train = dnn_model[arch[3:] + '_' + folder].evaluate(
                    train_features, train_labels, verbose=0
                )

                pred_train = dnn_model[arch[3:] + '_' + folder].predict(
                    train_features, verbose=0
                )

                pred_test = dnn_model[arch[3:] + '_' + folder].predict(
                    test_features, verbose=0
                )

                avg_thickness = pd.Series(
                    np.mean(pred_train), name = 'avg train thickness'
                )

                avg_test_thickness = pd.Series(
                    np.mean(pred_test),  name = 'avg test thickness'
                )

                temp_df = pd.merge(
                    avg_thickness, avg_test_thickness, right_index=True, left_index=True
                )

                predictions = pd.concat(
                    [predictions, temp_df], ignore_index = True
                )
                
                predictions.loc[predictions.index[-1], 'model'] = folder
                predictions.loc[predictions.index[-1], 'test mae'] = mae_test
                predictions.loc[predictions.index[-1], 'train mae'] = mae_train
                predictions.loc[predictions.index[-1], 'architecture'] = arch[3:]
                predictions.loc[predictions.index[-1], 'validation split'] = '0.2'
                predictions.loc[predictions.index[-1], 'dataset'] = dataset.name
                predictions.loc[predictions.index[-1], 'dropout'] = dropout

    #                 if chosen_dir in global_list:
    #                     predictions.loc[predictions.index[-1], 'region'] = 'g'
    #                 if chosen_dir in region_list:
    #                     predictions.loc[predictions.index[-1], 'region'] = int(reg)

                if '0.1' in folder:
                    predictions.loc[predictions.index[-1], 'learning rate'] = '0.1'
                if '0.01' in folder:
                    predictions.loc[predictions.index[-1], 'learning rate'] = '0.01'
                if '0.001' in folder:
                    predictions.loc[predictions.index[-1], 'learning rate']= '0.001'
                    
                if '10' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '10'
                if '15' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '15'               
                if '20' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '20' 
                if '25' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '25'
                if '30' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '30'
                if '35' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '35'
                if '40' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '40'
                if '45' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '45'
                if '50' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '50'
                if '55' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '55'
                if '60' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '60'
                if '65' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '65'
                if '70' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '70'
                if '75' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '75'
                if '80' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '80'
                if '85' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '85'
                if '90' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '90'
                if '95' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '95'
                if '100' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '100'
                if '150' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '150'
                if '200' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '200'       

                if '300' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '300'
                if '400' in folder:
                    predictions.loc[predictions.index[-1], 'epochs']= '400'
        
    predictions.rename(columns = {0:'avg train thickness'},inplace = True)
    predictions.to_csv('zults/predictions_' + dataset.name + '_' + dropout + '.csv')
    
    # calculate statistics
    print('calculating statistics...')

    for ep in list(predictions['epochs'].unique()):
        df = predictions[predictions['epochs'] == ep]
        for dataframe in list(predictions['dataset'].unique()):
            dfs = df[df['dataset'] == dataframe]
            for arch in list(predictions['architecture'].unique()):
                dfsr = dfs[dfs['architecture'] == arch]
                if dfsr.empty:
                    pass
                for lr in list(predictions['learning rate'].unique()):
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
                                str(int(ep)) +
                                '_0'
                        )
                        if model_name not in dnn_model:
                            pass
                        else:
                            test_mae_mean = np.mean(dfsrq['test mae'])
                            test_mae_std_dev = np.std(dfsrq['test mae'])

                            # find mean and std dev of train mae
                            train_mae_mean = np.mean(dfsrq['train mae'])
                            train_mae_std_dev = np.std(dfsrq['train mae'])

                            # find mean and std dev of predictions made based on training data
                            train_thickness_mean = np.mean(dfsrq['avg train thickness']) 
                            train_thickness_std_dev = np.std(dfsrq['avg train thickness'])

                            # find mean and std dev of predictions made based on test data
                            test_thickness_mean = np.mean(dfsrq['avg test thickness']) 
                            test_thickness_std_dev = np.std(dfsrq['avg test thickness'])

                            # put something in a series that can be appended to a df
                            s = pd.Series(train_thickness_mean)
                            
                            deviations = pd.concat(
                                [deviations, s], ignore_index=True
                            )
                            

                            # begin populating deviations table
                            deviations.loc[
                                deviations.index[-1], 'layer architecture'
                            ] = arch  




                            deviations.loc[
                                deviations.index[-1], 'total parameters'
                            ] = dnn_model[
                                    arch + 
                                    '_' + 
                                    dataset.name + 
                                    '_' +
                                    dropout +
                                    '_dnn_MULTI_' +
                                    str(lr) +
                                    '_0.2_' +
                                    str(int(ep)) +
                                    '_0'
                            ].count_params() 

                            deviations.loc[
                                deviations.index[-1], 'trained parameters'
                            ] = deviations.loc[
                                deviations.index[-1], 'total parameters'
                            ] - (len(dataset.columns) + (len(dataset.columns) - 1))

                            deviations.loc[
                                deviations.index[-1], 'total inputs'
                            ] = (len(dataset) * (len(dataset.columns) -1))

                            deviations.loc[
                                deviations.index[-1], 'df'
                            ] = dataframe

                            deviations.loc[
                                deviations.index[-1], 'dropout'
                            ] = dropout

                            deviations.loc[
                                deviations.index[-1], 'learning rate'
                            ] = lr

                            deviations.loc[
                                deviations.index[-1], 'validation split'
                            ]= 0.2

                            deviations.loc[
                                deviations.index[-1], 'epochs'
                            ] = ep

                            deviations.loc[
                                deviations.index[-1], 'test mae avg'
                            ] = test_mae_mean

                            deviations.loc[
                                deviations.index[-1], 'train mae avg'] = train_mae_mean

                            deviations.loc[
                                deviations.index[-1], 'test mae std dev'
                            ] = test_mae_std_dev

                            deviations.loc[
                                deviations.index[-1], 'train mae std dev'
                            ] = train_mae_std_dev

                            deviations.loc[
                                deviations.index[-1], 'test predicted thickness std dev'
                            ] = test_thickness_std_dev

                            deviations.loc[
                                deviations.index[-1], 'train predicted thickness std dev'
                            ] = train_thickness_std_dev



                            deviations.drop(columns = {0},inplace = True)    
                            deviations = deviations.dropna()


                            deviations = deviations.sort_values('test mae avg')
                            deviations['epochs'] = deviations['epochs'].astype(int)
                            deviations.to_csv(
                                'zults/deviations_' + 
                                dataset.name + 
                                '_' + 
                                dropout + 
                                '.csv'
                            )