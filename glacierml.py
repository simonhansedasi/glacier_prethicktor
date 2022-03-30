import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm


def data_loader(pth = '/data/fast1/glacierml/T_models/'):
    T = pd.read_csv(pth + 'T.csv', low_memory = False)
    T = T[[
        'GlaThiDa_ID',
        'LAT',
        'LON',
        'AREA',
        'MEAN_SLOPE',
        'MEAN_THICKNESS'
    ]]
        
    T = T.dropna()
    
    TT = pd.read_csv(pth + 'TT.csv', low_memory = False)
    TT = TT[[
        'LOWER_BOUND',
        'UPPER_BOUND',
        'AREA',
        'MEAN_SLOPE',
        'MEAN_THICKNESS',
    ]]
    TT = TT.dropna()
    
    TTT = pd.read_csv(pth + 'TTT.csv', low_memory = False)
    TTT = TTT[[
        'GlaThiDa_ID',
        'POINT_LAT',
        'POINT_LON',
        'ELEVATION',
        'THICKNESS'
    ]]
    
    TTTx = pd.merge(T,TTT, how = 'inner', on = 'GlaThiDa_ID')
    TTTx.rename(columns = {
        'LAT':'CENT_LAT',
        'LON':'CENT_LON'
    },inplace = True)
    
    TTTx = TTTx.drop([
        'GlaThiDa_ID',
        'MEAN_THICKNESS'
    ],axis = 1)
    TTTx = TTTx.dropna()
    
    T = T.drop('GlaThiDa_ID',axis = 1)
    TTT = TTT.drop('GlaThiDa_ID',axis =1)
    TTT = TTT.dropna()
    
    return T,TT,TTT,TTTx

def thickness_renamer(T):
    T = T.rename(columns = {
        'MEAN_THICKNESS':'THICKNESS'
    },inplace = True)
    
    
def data_splitter(T):
    train_dataset = T.sample(frac=0.8, random_state=0)
    test_dataset = T.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    #define label - attribute training to be picked
    train_labels = train_features.pop('THICKNESS')
    test_labels = test_features.pop('THICKNESS')
    
    return train_features, test_features, train_labels, test_labels



def build_linear_model(normalizer):
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    
    return model

def build_dnn_model(norm):
    model = keras.Sequential([
              norm,
              layers.Dense(64, activation='relu'),
              layers.Dense(64, activation='relu'),
              layers.Dense(1) ])

    model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01))
    
    return model

def plot_single_model_variable(x, y,feature_name):
    plt.scatter(train_features[feature_name], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel(feature_name)
    plt.ylabel('Avg Thickness (m)')
#     plt.xlim((0,20))
    plt.legend()
      
def plot_loss(history):
#     plt.subplots(figsize=(10,5))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    #   plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
def build_and_train_model(i):
    #     split data
        (train_features,test_features,
         train_labels,test_labels) = data_splitter(i)

        print('epochs for ' + i.name + ' dataset?')
        epochs_input = 1000
#         int(input())
        print('validation split?')
        validation_split_input = 0.2
#         float(input())

        
        print(i.name)
        
    #     normalize data
        print('Normalizing ' + str(i.name) + ' data')
        normalizer = {}
        variable_list = list(train_features)
        for variable_name in tqdm(variable_list):
            normalizer[variable_name] = preprocessing.Normalization(input_shape=[1,], axis=None)
            normalizer[variable_name].adapt(np.array(train_features[variable_name]))

        normalizer['ALL'] = preprocessing.Normalization(axis=-1)
        normalizer['ALL'].adapt(np.array(train_features))
        print(i.name + ' data normalized')
        
    #       linear model
        print('Running single-variable linear regression on ' + str(i.name) + ' dataset')
        linear_model = {}
        linear_history = {}
        linear_results = {}
        variable_list = list(train_features)

        for variable_name in tqdm(variable_list):
            linear_model[variable_name] = build_linear_model(normalizer[variable_name])
            linear_history[variable_name] = linear_model[variable_name].fit(
                                                train_features[variable_name], train_labels,        
                                                epochs=epochs_input,
                                                verbose=0,
                                                validation_split = validation_split_input)
            linear_model[variable_name].save(
                'saved_models/' + str(i.name) + '_linear_' + str(variable_name))

        print('Running multi-variable linear regression on ' + str(i.name) + ' dataset')
        linear_model = build_linear_model(normalizer['ALL'])
        linear_history['MULTI'] = linear_model.fit(
        train_features, train_labels,        
           epochs=epochs_input,
           verbose=0,
           validation_split = validation_split_input)

        print('Saving results')
        for variable_name in tqdm(list(linear_history)):
            df = pd.DataFrame(linear_history[variable_name].history)
            df.to_csv(
                'saved_results/' + str(i.name) + '_linear_history_' + str(variable_name))

        df = pd.DataFrame(linear_history['MULTI'].history)
        df.to_csv('saved_results/' + str(i.name) + '_linear_history_MULTI')
        linear_model.save('saved_models/' + str(i.name) + '_linear_MULTI')

    #      DNN model
        dnn_model = {}
        dnn_history = {}
        dnn_results = {}

        print('Running single-variable DNN regression on ' + str(i.name) + ' dataset')
        variable_list = tqdm(list(train_features))
        for variable_name in variable_list:
            dnn_model[variable_name] = build_dnn_model(normalizer[variable_name])
            dnn_history[variable_name] = dnn_model[variable_name].fit(
                                                train_features[variable_name], train_labels,        
                                                epochs=epochs_input,
                                                verbose=0,
                                                validation_split = validation_split_input)    
            dnn_model[variable_name].save('saved_models/' + str(i.name) + '_dnn_' + str(variable_name))

        print('Running multi-variable DNN regression on ' + str(i.name) + ' dataset')
        dnn_model = build_dnn_model(normalizer['ALL'])
        dnn_history['MULTI'] = dnn_model.fit(
            train_features, train_labels,
            validation_split=validation_split_input,
            verbose=0, epochs=epochs_input)

        dnn_model.save('saved_models/' + str(i.name) + '_dnn_MULTI')

        print('Saving results')
        for variable_name in tqdm(list(dnn_history)):
            df = pd.DataFrame(dnn_history[variable_name].history)
            df.to_csv('saved_results/' + str(i.name) + '_dnn_history_'+str(variable_name))

        df = pd.DataFrame(dnn_history['MULTI'].history)
        df.to_csv('saved_results/' + str(i.name) + '_dnn_history_MULTI')
        dnn_model.save('saved_models/' + str(i.name) + '_dnn_MULTI')
    
