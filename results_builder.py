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

print('please select rootdir: sm1, sm2, sm3, sm4, sm5, sm6')

chosen_dir = input()
# chosen_dir = input()



rootdir = 'saved_models/' + chosen_dir + '/'
print(chosen_dir)
if chosen_dir == 'sm1':
    df1 = gl.data_loader()
    gl.thickness_renamer(df1)
    dataset = df1
    dataset.name = 'df1'

elif chosen_dir == 'sm2':
    df2 = gl.data_loader_2()
    gl.thickness_renamer(df2)
    dataset = df2
    dataset.name = 'df2'

elif chosen_dir == 'sm3':

    df2 = gl.data_loader_2()
    gl.thickness_renamer(df2)
    df3 = df2[[
        'Area',
        'thickness',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Aspect',
        'Lmax'
    ]]
    dataset = df3
    dataset.name = 'df3'

elif chosen_dir == 'sm4':
    df4 = gl.data_loader_4()
    gl.thickness_renamer(df4)
    dataset = df4
    dataset.name = 'df4'


elif chosen_dir == 'sm5':
    df5 = gl.data_loader_5()
    reg = df6['region'].iloc[-1]
    regional_data = regional_data.drop('region', axis=1)
    dataset = df5
    dataset.name = str('df5' + str(reg))


elif chosen_dir == 'sm6':
    df6 = gl.data_loader_6()
    reg = df6['region'].iloc[-1]
    regional_data = regional_data.drop('region', axis=1)
    dataset = df6
    dataset.name = str('df6' + str(reg))


(train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)


# define default model hyperparameters
RS = range(0,25,1)
ep = 300



# load and evaluate models
dnn_model = {}
predictions = pd.DataFrame()

print('loading and evaluating models...')
for arch in os.listdir(rootdir):
    print('layer architecture: ' + arch[3:])
    for folder in tqdm(os.listdir(rootdir + arch)):
#         print(folder)
        if 'MULTI' in folder and 'dnn' in folder:

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

            predictions = predictions.append(temp_df, ignore_index = True)
            predictions.loc[predictions.index[-1], 'model'] = folder
            predictions.loc[predictions.index[-1], 'test mae'] = mae_test
            predictions.loc[predictions.index[-1], 'train mae'] = mae_train
            predictions.loc[predictions.index[-1], 'architecture'] = arch[3:]
            predictions.loc[predictions.index[-1], 'validation split'] = '0.2'
            predictions.loc[predictions.index[-1], 'dataset'] = dataset

            if '0.1' in folder:
                predictions.loc[predictions.index[-1], 'learning rate'] = '0.1'
            if '0.01' in folder:
                predictions.loc[predictions.index[-1], 'learning rate'] = '0.01'
            if '0.001' in folder:
                predictions.loc[predictions.index[-1], 'learning rate']= '0.001'
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
predictions.to_csv('zults/predictions_' + dataset.name + '.csv')


# calculate statistics
print('calculating statistics...')
deviations = pd.DataFrame()
for architecture in tqdm(list(predictions['architecture'].unique())):
    for learning_rate in list(predictions['learning rate'].unique()):
        for epochs in list(predictions['epochs'].unique()):
            df = predictions[
                (predictions['architecture'] == architecture) & 
                (predictions['learning rate' ] == learning_rate) &
                (predictions['epochs' ] == epochs)
            ]
            test_mae_mean = np.mean(df['test mae'])
            test_mae_std_dev = np.std(df['test mae'])

            train_mae_mean = np.mean(df['train mae'])
            train_mae_std_dev = np.std(df['train mae'])

            train_thickness_mean = np.mean(df['avg train thickness']) 
            train_thickness_std_dev = np.std(df['avg train thickness'])

            test_thickness_mean = np.mean(df['avg test thickness']) 
            test_thickness_std_dev = np.std(df['avg test thickness'])

            s = pd.Series(train_thickness_mean)

            deviations = deviations.append(s, ignore_index=True)   
            deviations.loc[deviations.index[-1], 'layer architecture']= architecture  

            deviations.loc[
                deviations.index[-1], 'model parameters'
            ] = dnn_model[
                architecture + '_' + dataset.name + '_dnn_MULTI_0.1_0.2_300_0'
                ].count_params() 

            deviations.loc[deviations.index[-1], 'learning rate'] = learning_rate

            deviations.loc[deviations.index[-1], 'validation split']= 0.2

            deviations.loc[deviations.index[-1], 'epochs'] = epochs

            deviations.loc[deviations.index[-1], 'test mae avg'] = test_mae_mean

            deviations.loc[deviations.index[-1], 'train mae avg'] = train_mae_mean

            deviations.loc[deviations.index[-1], 'test mae std dev'] = test_mae_std_dev

            deviations.loc[deviations.index[-1], 'train mae std dev'] = train_mae_std_dev

            deviations.loc[
                deviations.index[-1], 'test predicted thickness std dev'
            ] = test_thickness_std_dev

            deviations.loc[
                deviations.index[-1], 'train predicted thickness std dev'
            ] = train_thickness_std_dev

            deviations.drop(columns = {0},inplace = True)    
            deviations = deviations.dropna()


            deviations = deviations.sort_values('test mae avg')

deviations.to_csv('zults/deviations_' + dataset.name + '.csv')



    #build RGI specific to modules chosen


#     if chosen_dir == 'sm1':
#         print('loading RGI...')
#         rootdir = '/data/fast0/datasets/rgi60-attribs/'
#         RGI_extra = pd.DataFrame()
#         for file in tqdm(os.listdir(rootdir)):
#             f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
#             RGI_extra = RGI_extra.append(f, ignore_index = True)


#         RGI = RGI_extra[[
#             'CenLat',
#             'CenLon',
#             'Slope',
#             'Area',
#         ]]

#         RGI = RGI.rename(columns = {
#         'CenLon':'lon',
#         'CenLat':'lat',
#         'Area':'area',
#         'Slope':'mean_slope'
#         })



#     elif chosen_dir == 'sm5' or chosen_dir == 'sm6':
#         print('loading RGI...')
#         rootdir = '/data/fast0/datasets/rgi60-attribs/'
#         RGI_extra = pd.DataFrame()
#         for file in tqdm(os.listdir(rootdir)):
#             f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
#             region_1 = f['RGIId'].iloc[-1][6:]
#             region = region_1[:2]
#             if str(region) == str(reg):
#                 RGI_extra = RGI_extra.append(f, ignore_index = True)
#         RGI = RGI_extra[[
#             'CenLat',
#             'CenLon',
#             'Slope',
#             'Zmin',
#             'Zmed',
#             'Zmax',
#             'Area',
#             'Aspect',
#             'Lmax'
#         ]]
#         RGI = RGI.drop(RGI.loc[RGI['Zmed']<0].index)
#         RGI = RGI.drop(RGI.loc[RGI['Lmax']<0].index)
#         RGI = RGI.drop(RGI.loc[RGI['Slope']<0].index)
#         RGI = RGI.drop(RGI.loc[RGI['Aspect']<0].index)
#         RGI = RGI.reset_index()
#         RGI = RGI.drop('index', axis=1)



#     else:
#         print('loading RGI...')
#         rootdir = '/data/fast0/datasets/rgi60-attribs/'
#         RGI_extra = pd.DataFrame()
#         for file in tqdm(os.listdir(rootdir)):
#             f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
#             RGI_extra = RGI_extra.append(f, ignore_index = True)


#         RGI = RGI_extra[[
#             'CenLat',
#             'CenLon',
#             'Slope',
#             'Zmin',
#             'Zmed',
#             'Zmax',
#             'Area',
#             'Aspect',
#             'Lmax'
#         ]]
#         RGI = RGI.drop(RGI.loc[RGI['Zmed']<0].index)
#         RGI = RGI.drop(RGI.loc[RGI['Lmax']<0].index)
#         RGI = RGI.drop(RGI.loc[RGI['Slope']<0].index)
#         RGI = RGI.drop(RGI.loc[RGI['Aspect']<0].index)
#         RGI = RGI.reset_index()
#         RGI = RGI.drop('index', axis=1)

#     if dataset.name == 'df3':
#         RGI = RGI[[
#     #         'CenLat',
#     #         'CenLon',
#             'Slope',
#             'Zmin',
#             'Zmed',
#             'Zmax',
#             'Area',
#             'Aspect',
#             'Lmax'
#         ]]






#     arch = deviations['layer architecture'].iloc[0]
#     lr = deviations['learning rate'].iloc[0]
#     vs = deviations['validation split'].iloc[0]
#     ep = deviations['epochs'].iloc[0]
#     print('layer architecture: ' + arch + ' learning rate: ' + str(lr))
#     print('predicting RGI thicknesses using model trained on RGI data matched with GlaThiDa thicknesses...')
#     dfs = pd.DataFrame()
#     for rs in tqdm(RS):
#         s = pd.Series(
#             dnn_model[
#             str(arch) +
#             '_' +
#             dataset.name +
#             '_dnn_MULTI_'+
#             str(lr)+
#             '_'+
#             str(vs)+
#             '_300_'+ 
#             str(rs)
#         ].predict(RGI, verbose=0).flatten(), name = rs
#         )
#         dfs[rs] = s



#     RGI_prethicked = RGI.copy() 
#     RGI_prethicked['avg predicted thickness'] = 'NaN'
#     RGI_prethicked['predicted thickness std dev'] = 'NaN'

#     print('calculating average thickness across random state ensemble...')
#     for i in tqdm(dfs.index):
#         avg_predicted_thickness = np.mean(dfs.loc[i])
#         RGI_prethicked['avg predicted thickness'].loc[i] = avg_predicted_thickness


#     print('computing standard deviations and variances for RGI predicted thicknesses')

#     for i in tqdm(dfs.index):
#         predicted_thickness_std_dev = np.std(dfs.loc[i])

#         RGI_prethicked['predicted thickness std dev'].loc[i] = predicted_thickness_std_dev

#     RGI_prethicked.to_csv(
#         'zults/RGI_predicted_' + dataset.name + '_' + arch + '_' + str(lr) + '_' + str(ep) + '.csv'
#     )


