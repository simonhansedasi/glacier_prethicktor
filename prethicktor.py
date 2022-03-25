import pandas as pd
import numpy as np
import glacierml as gl
from tqdm import tqdm
import tensorflow as tf
import glacierml as gl
from tqdm import tqdm
import warnings
from tensorflow.python.util import deprecation
import os
import logging
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('Loading data')
#load and organize data
T,TT,TTT,TTTx = gl.data_loader()
gl.thickness_renamer(T)
gl.thickness_renamer(TT)

glathida_list = T,TT,TTT,TTTx

T.name = 'T'
TT.name = 'TT'
TTT.name = 'TTT'
TTTx.name = 'TTTx'
for i in glathida_list:
#     split data
    (train_features,test_features,
     train_labels,test_labels) = gl.data_splitter(i.head())

#     normalize data
    print('Normalizing ' + str(i.name) + ' data')
    normalizer = {}
    variable_list = list(train_features)
    for variable_name in tqdm(variable_list):
        normalizer[variable_name] = gl.preprocessing.Normalization(input_shape=[1,], axis=None)
        normalizer[variable_name].adapt(np.array(train_features[variable_name]))
        
    normalizer['ALL'] = gl.preprocessing.Normalization(axis=-1)
    normalizer['ALL'].adapt(np.array(train_features))

#       linear model
    print('Running single-variable linear regression on ' + str(i.name) + ' dataset')
    linear_model = {}
    linear_history = {}
    linear_results = {}
    variable_list = list(train_features)
    
    for variable_name in tqdm(variable_list):
        linear_model[variable_name] = gl.build_linear_model(normalizer[variable_name])
        linear_history[variable_name] = linear_model[variable_name].fit(
                                            train_features[variable_name], train_labels,        
                                            epochs=100,
                                            verbose=0,
                                            validation_split = 0.2)
        linear_model[variable_name].save(
            'saved_models/' + str(i.name) + '_linear_' + str([variable_name]))

    print('Running multi-variable linear regression on ' + str(i.name) + ' dataset')
    linear_model = gl.build_linear_model(normalizer['ALL'])
    linear_history['MULTI'] = linear_model.fit(
    train_features, train_labels,        
       epochs=100,
       verbose=0,
       validation_split = 0.2)
    
    print('Saving results')
    for variable_name in tqdm(list(linear_history)):
        df = pd.DataFrame(linear_history[variable_name].history)
        df.to_csv(
            'saved_results/' + str(i.name) + '_linear_history' + str([variable_name]))

    df = pd.DataFrame(linear_history['MULTI'].history)
    df.to_csv('saved_results/' + str(i.name) + '_linear_history' + str(['MULTI']))
    linear_model.save('saved_models/' + str(i.name) + '_linear_' + str(['MULTI']))

#      DNN model
    dnn_model = {}
    dnn_history = {}
    dnn_results = {}

    print('Running single-variable DNN regression on ' + str(i.name) + ' dataset')
    variable_list = tqdm(list(train_features))
    for variable_name in variable_list:
        dnn_model[variable_name] = gl.build_dnn_model(normalizer[variable_name])
        dnn_history[variable_name] = dnn_model[variable_name].fit(
                                            train_features[variable_name], train_labels,        
                                            epochs=100,
                                            verbose=0,
                                            validation_split = 0.2)    
        dnn_model[variable_name].save('saved_models/' + str(i.name) + '_dnn_' + str([variable_name]))

    print('Running multi-variable DNN regression on ' + str(i.name) + ' dataset')
    dnn_model = gl.build_dnn_model(normalizer['ALL'])
    dnn_history['MULTI'] = dnn_model.fit(
        train_features, train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)

    dnn_model.save('saved_models/' + str(i.name) + '_dnn_' + str(['MULTI']))
    
    print('Saving results')
    for variable_name in tqdm(list(dnn_history)):
        df = pd.DataFrame(dnn_history[variable_name].history)
        df.to_csv('saved_results/' + str(i.name) + '_dnn_history'+str([variable_name]))

    df = pd.DataFrame(dnn_history['MULTI'].history)
    df.to_csv('saved_results/' + str(i.name) + '_dnn_' + str(['MULTI']))
    dnn_model.save('saved_models/' + str(i.name) + '_dnn_' + str(['MULTI']))