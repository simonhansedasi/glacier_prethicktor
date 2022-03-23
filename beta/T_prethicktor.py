import numpy as np
import pandas as pd
import tensorflow as tf
import glacierml as gl
from tqdm import tqdm

# import the data
T = pd.read_csv('/home/sa42/data/glac/T_models/T.csv')
glathida = T
glathida = glathida[[
    'LAT',
    'LON',
    'AREA',
    'MEAN_SLOPE',
#     'MEAN_THICKNESS',
    'MAXIMUM_THICKNESS',
]]
# drop null data
glathida = glathida.dropna()

# split data set into training and testing
train_dataset = glathida.sample(frac=0.8, random_state=0)
test_dataset = glathida.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#define label - attribute training to be picked
train_labels = train_features.pop('MAXIMUM_THICKNESS')
test_labels = test_features.pop('MAXIMUM_THICKNESS')

# DATA NORMALIZER
normalizer = {}
variable_list = list(train_features)
for variable_name in tqdm(variable_list):
    normalizer[variable_name] = gl.preprocessing.Normalization(input_shape=[1,], axis=None)
    normalizer[variable_name].adapt(np.array(train_features[variable_name])) 
normalizer['ALL'] = gl.preprocessing.Normalization(axis=-1)
normalizer['ALL'].adapt(np.array(train_features))


# LINEAR REGRESSION MODELS

linear_model = {}
linear_history = {}
linear_results = {}

for variable_name in tqdm(variable_list):

    linear_model[variable_name] = gl.build_linear_model(normalizer[variable_name])
    linear_history[variable_name] = linear_model[variable_name].fit(
                                        train_features[variable_name], train_labels,        
                                        epochs=1000,
                                        verbose=0,
                                        validation_split = 0.2)    
    linear_results[variable_name] = linear_model[variable_name].evaluate(
                                        test_features[variable_name],
                                        test_labels, verbose=0)
    linear_model[variable_name].save('saved_models/T_linear_' + str([variable_name]))
    
    
# MULTIVARIABLE LINEAR REGRESSION 

linear_model = gl.build_linear_model(normalizer['ALL'])

linear_history['MULTI'] = linear_model.fit(
train_features, train_labels,        
   epochs=1000,
   verbose=0,
   validation_split = 0.2)

linear_results['MULTI'] = linear_model.evaluate(
    test_features,
    test_labels, verbose=0)

linear_model.save('saved_models/T_linear_multivariable')


# DNN MODELS
dnn_model = {}
dnn_history = {}
dnn_results = {}

for variable_name in tqdm(variable_list):
    dnn_model[variable_name] = gl.build_dnn_model(normalizer[variable_name])
    dnn_history[variable_name] = dnn_model[variable_name].fit(
                                        train_features[variable_name], train_labels,        
                                        epochs=1000,
                                        verbose=0,
                                        validation_split = 0.2)
    
    dnn_results[variable_name] = dnn_model[variable_name].evaluate(
                                        test_features[variable_name],
                                        test_labels, verbose=0)
    
    dnn_model[variable_name].save('saved_models/T_dnn_' + str([variable_name]))

    
# DNN MULTIVARIABLE MODEL     
dnn_model = gl.build_dnn_model(normalizer['ALL'])

dnn_history['MULTI'] = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=1000)

dnn_results['MULTI'] = dnn_model.evaluate(
    test_features,
    test_labels, verbose=0)   

dnn_model.save('saved_models/T_dnn_multivariable_model')

# results collector
dfs = pd.DataFrame()
for variable_name in list(dnn_history):    
    df1 = pd.DataFrame(dnn_history[variable_name].history)
    df2 = pd.DataFrame(linear_history[variable_name].history)
    df1 = df1.loc[[df1.last_valid_index()]]
    df2 = df2.loc[[df2.last_valid_index()]]
    df1['Architecture'] = 'DNN'
    df2['Architecture'] = 'Linear'
    df1.insert(0, 'Variable', [variable_name])
    df2.insert(0, 'Variable', [variable_name])
    df = pd.concat([df1,df2])
    dfs = dfs.append(df)
    
df = dfs[[
    'Architecture',
    'Variable',
    'loss',
    'val_loss'
]]
df.rename(columns = {
    'loss':'Training Loss',
    'val_loss':'Test loss'
},inplace=True)
df = df.sort_values(by=['Architecture','Variable'], ascending=[False,False])
# print(df.to_latex(index=False))
df.to_csv('saved_results/T_loss')