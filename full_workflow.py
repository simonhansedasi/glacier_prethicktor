import pandas as pd
import numpy as np
import glacierml as gl
from tqdm import tqdm
import tensorflow as tf
import warnings
from tensorflow.python.util import deprecation
import os
import logging
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)
import configparser

tf.random.set_seed(42)
# chosen_parameterization = input()   
    
for i in range(1,4,1):
    
    # load training data information
    parameterization = str(i)
    config = configparser.ConfigParser()
    config.read('model_parameterization.txt')

    data = gl.load_training_data(
#         root_dir = '/home/prethicktor/data/',
        area_scrubber = config[parameterization]['area scrubber'],
        anomaly_input = float(   config[parameterization]['size threshold']   )
    )
    
    
    data = data.drop(
        data[data['distance test'] >= float(
                                                config[parameterization]['distance threshold']  
                                            )
            ].index
    )

    data = data.drop([
        'RGIId','region', 'RGI Centroid Distance', 
        'AVG Radius', 'Roundness', 
            'distance test', 
        'size difference'
    ], axis = 1)


#     # build models
#     RS = range(0,25,1)
#     print(data)
#     layer_1_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#     layer_2_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

#     for layer_2_input in (layer_2_list):
#         for layer_1_input in (layer_1_list):
#             if layer_1_input <= layer_2_input:
#                 pass
#             elif layer_1_input > layer_2_input:

#                 arch = str(layer_1_input) + '-' + str(layer_2_input)
#                 dropout = True
#                 print('')
#                 print('Running multi-variable DNN regression with parameterization ' + 
#                     str(parameterization) + 
#                     ', layer architecture = ' +
#                     arch)

#                 for rs in tqdm(RS):

#                     gl.build_and_train_model(
#                         data, 
#                         random_state = rs, 
#                         parameterization = parameterization, 
#                         res = parameterization,
#                         layer_1 = layer_1_input,
#                         layer_2 = layer_2_input,
#                     )   


                    
#     # evaluate model loss and then calculate model statistics
#     model_predictions = pd.DataFrame()
#     model_statistics = pd.DataFrame()
#     # dropout_input = dropout_input_iter
#     rootdir = 'saved_models/' + parameterization + '/'

#     print('loading and evaluating models...')
#     for arch in tqdm( os.listdir(rootdir)):       
#     #     print('layer architecture: ' + arch[3:])
#         pth = os.path.join(rootdir, arch)
#         for folder in (os.listdir(pth)):
#             architecture = arch
#     #         print(architecture)
#             model_loc = (
#                 rootdir + 
#                 arch + 
#                 '/' + 
#                 folder
#             )

#             model_name = folder
#             dnn_model = gl.load_dnn_model(model_loc)
#     #         print(dnn_model)
#             df = gl.evaluate_model(architecture, model_name, data, dnn_model, parameterization)

#             model_predictions = pd.concat([model_predictions, df], ignore_index = True)
#     #     break
#     # print(model_predictions['architecture'])
#     # print(list(model_predictions))
#     model_predictions.rename(columns = {0:'avg train thickness'},inplace = True)
#     model_predictions.to_csv('zults/model_predictions_' + parameterization + '.csv')
#     # calculate statistics
#     print('calculating statistics...')
#     dnn_model = {}

#     for arch in tqdm(list(model_predictions['layer architecture'].unique())):
#         model_thicknesses = model_predictions[model_predictions['layer architecture'] == arch]


#         model_name = ('0')

#         model_loc = (
#             rootdir + 
#             arch + 
#             '/' +
#             '0'
#         )
#     #     print(model_loc)
#         isdir = os.path.isdir(model_loc)
#     #     print(isdir)
#         if isdir == False:
#             print('model not here, calculating next model')
#         elif isdir == True:


#             dnn_model = gl.load_dnn_model(model_loc)
#             df = gl.calculate_model_avg_statistics(
#                 dnn_model,
#                 arch,
#                 data,
#                 model_thicknesses
#             )

#             model_statistics = pd.concat(
#                 [model_statistics, df], ignore_index = True
#             )
#             #         print(list(model_statistics))


#     model_statistics['architecture weight 1'] = (
#         sum(model_statistics['test mae avg']) / model_statistics['test mae avg']
#     )
#     model_statistics['architecture weight 2'] = (
#         model_statistics['test mae avg'] / sum(model_statistics['test mae avg'])
#     )
#     model_statistics.to_csv(
#         'zults/model_statistics_' + 
#         parameterization + 
#         '.csv'
#     )
    
    
#     # make glacier thickness estimates
#     model_statistics = pd.read_csv('zults/model_statistics_' + parameterization + '.csv')
#     model_statistics = model_statistics.reset_index()

#     model_statistics = model_statistics[[
#     'layer architecture',
#     ]]
    
#     gl.estimate_thickness(
#         model_statistics, parameterization, useMP = False, verbose = True
#     )
    
    
    # aggregate model thicknesses
    print('Gathering architectures...')
    arch_list = gl.list_architectures(parameterization = parameterization)
    arch_list = arch_list.reset_index()
    arch_list = arch_list.drop('index', axis = 1)

    df = pd.DataFrame(columns = {
            'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
    })
    arch_list = arch_list.sort_values('layer architecture')
    print('Architectures listed')
    print('Compiling predictions...')
    for arch in tqdm(arch_list['layer architecture'].unique()):
        df_glob = gl.load_global_predictions(
            parameterization = parameterization,
            architecture = arch,
        )


        df = pd.concat([df,df_glob])
        
    statistics = pd.DataFrame()
    for file in (os.listdir('zults/')):
        if 'statistics_' + parameterization in file:
            file_reader = pd.read_csv('zults/' + file)
            statistics = pd.concat([statistics, file_reader], ignore_index = True)
    df = pd.merge(df, statistics, on = 'layer architecture')

    df = df[[
            'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24','architecture weight 1'
    ]]

    compiled_raw = df.groupby('RGIId')[
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24','architecture weight 1'
    ]

    print('Predictions compiled')
    print('Aggregating statistics...')
    dft = pd.DataFrame()
    for this_rgi_id, obj in tqdm(compiled_raw):
        rgi_id = pd.Series(this_rgi_id, name = 'RGIId')
    #     print(f"Data associated with RGI_ID = {this_rgi_id}:")
        dft = pd.concat([dft, rgi_id])
        dft = dft.reset_index()
        dft = dft.drop('index', axis = 1)


        obj['weight'] = obj['architecture weight 1'] + 1 / (obj[['0', '1', '2', '3', '4',
                                                         '5', '6', '7', '8', '9',
                                                         '10','11','12','13','14',
                                                         '15','16','17','18','19',
                                                         '20','21','22','23','24']].var(axis = 1))


        obj['weighted mean'] = obj['weight'] * obj[['0', '1', '2', '3', '4',
                                                   '5', '6', '7', '8', '9',
                                                   '10','11','12','13','14',
                                                   '15','16','17','18','19',
                                                   '20','21','22','23','24']].mean(axis = 1)


        weighted_glacier_mean = sum(obj['weighted mean']) / sum(obj['weight'])


        stacked_object = obj[[
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
            '11','12','13','14','15','16','17','18','19','20','21',
            '22','23','24',
        ]].stack()

        glacier_count = len(stacked_object)
    #     dft.loc[dft.index[-1], 'Weighted Mean Thickness'] = weighted_glacier_mean
        dft.loc[dft.index[-1], 'Mean Thickness'] = stacked_object.mean()
        dft.loc[dft.index[-1], 'Median Thickness'] = stacked_object.median()
        dft.loc[dft.index[-1],'Thickness Std Dev'] = stacked_object.std()

        statistic, p_value = shapiro(stacked_object)    
        dft.loc[dft.index[-1],'Shapiro-Wilk statistic'] = statistic
        dft.loc[dft.index[-1],'Shapiro-Wilk p_value'] = p_value


        q75, q25 = np.percentile(stacked_object, [75, 25])    
        dft.loc[dft.index[-1],'IQR'] = q75 - q25 

        lower_bound = np.percentile(stacked_object, 50 - 34.1)
        median = np.percentile(stacked_object, 50)
        upper_bound = np.percentile(stacked_object, 50 + 34.1)

        dft.loc[dft.index[-1],'Lower Bound'] = lower_bound
        dft.loc[dft.index[-1],'Upper Bound'] = upper_bound
        dft.loc[dft.index[-1],'Median Value'] = median
        dft.loc[dft.index[-1],'Total estimates'] = glacier_count

    dft = dft.rename(columns = {
        0:'RGIId'
    })
    dft = dft.drop_duplicates()
    dft.to_csv(
        'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_parameterization_' + 
        parameterization + '.csv'
              )