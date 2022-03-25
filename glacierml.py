import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

def data_loader():
    pth = '/home/sa42/data/glac/T_models/'
    T = pd.read_csv(pth + 'T.csv', low_memory = False)
    T = T[[
        'GlaThiDa_ID',
        'LAT',
        'LON',
        'AREA',
        'MEAN_SLOPE',
        'MAXIMUM_THICKNESS'
    ]]
        
    T = T.dropna()
    
    TT = pd.read_csv(pth + 'TT.csv', low_memory = False)
    TT = TT[[
        'LOWER_BOUND',
        'UPPER_BOUND',
        'AREA',
        'MEAN_SLOPE',
        'MAXIMUM_THICKNESS',
    ]]
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
    
    T = T.drop('GlaThiDa_ID',axis = 1)
    TTT = TTT.drop('GlaThiDa_ID',axis =1)
    return T,TT,TTT,TTTx

def thickness_renamer(T):
    T = T.rename(columns = {
        'MAXIMUM_THICKNESS':'THICKNESS'
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
                optimizer=tf.keras.optimizers.Adam(0.01))
    
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
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #   plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
    return plot_loss