import numpy as np
import pandas as pd
import tensorflow as tf
import glacierml as gl
from tqdm import tqdm
import os
from tensorflow.python.util import deprecation
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    
print('Reading data')
# import the data
T = pd.read_csv('/home/sa42/data/glac/T_models/T.csv')
T = T.drop([
#     'GlaThiDa_ID',
    'POLITICAL_UNIT',
    'GLACIER_NAME',
    'GLACIER_DB',
    'GLACIER_ID',
#     'LAT',
#     'LON',
    'SURVEY_DATE',
    'ELEVATION_DATE',
#     'AREA',
#     'MEAN_SLOPE',
    'MEAN_THICKNESS',
    'MEAN_THICKNESS_UNCERTAINTY',
    'MAXIMUM_THICKNESS',
    'MAX_THICKNESS_UNCERTAINTY',
    'SURVEY_METHOD',
    'SURVEY_METHOD_DETAILS',
    'NUMBER_OF_SURVEY_POINTS',
    'NUMBER_OF_SURVEY_PROFILES',
    'TOTAL_LENGTH_OF_SURVEY_PROFILES',
    'INTERPOLATION_METHOD',
    'INVESTIGATOR',
    'SPONSORING_AGENCY',
    'REFERENCES',
    'DATA_FLAG',
    'REMARKS'
],axis=1)
T = T.dropna()

TTT = pd.read_csv('/home/sa42/data/glac/T_models/TTT.csv')
TTT = TTT.drop([
#     'GlaThiDa_ID',
    'POLITICAL_UNIT',
    'GLACIER_NAME',
    'SURVEY_DATE',
    'PROFILE_ID',
    'POINT_ID',
#     'POINT_LAT',
#     'POINT_LON',
#     'ELEVATION',
#     'THICKNESS',
    'THICKNESS_UNCERTAINTY',
    'DATA_FLAG',
    'REMARKS'
],axis=1)
TTT = TTT.dropna()

df = pd.merge(T,TTT, how='inner',on='GlaThiDa_ID')
df = df.drop([
    'GlaThiDa_ID'    
],axis=1)

# split data set into training and testing
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#define label - attribute training to be picked
train_labels = train_features.pop('THICKNESS')
test_labels = test_features.pop('THICKNESS')


print('Normalizing Data')
# DATA NORMALIZER
normalizer = {}
variable_list = list(train_features)
for variable_name in tqdm(variable_list):
    normalizer[variable_name] = gl.preprocessing.Normalization(input_shape=[1,], axis=None)
    normalizer[variable_name].adapt(np.array(train_features[variable_name])) 
normalizer['ALL'] = gl.preprocessing.Normalization(axis=-1)
normalizer['ALL'].adapt(np.array(train_features))


# LINEAR REGRESSION MODELS
print('Running single-variable linear regression')
linear_model = {}
linear_history = {}
linear_results = {}

for variable_name in tqdm(variable_list):

    linear_model[variable_name] = gl.build_linear_model(normalizer[variable_name])
    linear_history[variable_name] = linear_model[variable_name].fit(
                                        train_features[variable_name], train_labels,        
                                        epochs=100,
                                        verbose=0,
                                        validation_split = 0.2)    
    linear_results[variable_name] = linear_model[variable_name].evaluate(
                                        test_features[variable_name],
                                        test_labels, verbose=0)
    linear_model[variable_name].save('saved_models/TTTT_linear_' + str([variable_name]))
    
    
# MULTIVARIABLE LINEAR REGRESSION 
print('Running multi-variable linear regression')
linear_model = gl.build_linear_model(normalizer['ALL'])

linear_history['MULTI'] = linear_model.fit(
train_features, train_labels,        
   epochs=100,
   verbose=0,
   validation_split = 0.2)

linear_results['MULTI'] = linear_model.evaluate(
    test_features,
    test_labels, verbose=0)

linear_model.save('saved_models/TTTT_linear_multivariable')


# DNN MODELS
dnn_model = {}
dnn_history = {}
dnn_results = {}
print('Running single-variable dnn regression')
for variable_name in tqdm(variable_list):
    dnn_model[variable_name] = gl.build_dnn_model(normalizer[variable_name])
    dnn_history[variable_name] = dnn_model[variable_name].fit(
                                        train_features[variable_name], train_labels,        
                                        epochs=100,
                                        verbose=0,
                                        validation_split = 0.2)
    
    dnn_results[variable_name] = dnn_model[variable_name].evaluate(
                                        test_features[variable_name],
                                        test_labels, verbose=0)
    
    dnn_model[variable_name].save('saved_models/TTTT_dnn_' + str([variable_name]))

    
# DNN MULTIVARIABLE MODEL 
print('Running multi-variable dnn regression')
dnn_model = gl.build_dnn_model(normalizer['ALL'])

dnn_history['MULTI'] = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

dnn_results['MULTI'] = dnn_model.evaluate(
    test_features,
    test_labels, verbose=0)   

dnn_model.save('saved_models/TTTT_dnn_multivariable')

print('collecting results')
# results collector
dfs = pd.DataFrame()
for variable_name in list(dnn_history):    
    df1 = pd.DataFrame(dnn_history[variable_name].history)
    df2 = pd.DataFrame(linear_history[variable_name].history)
    df1.to_csv('saved_results/TTTT_dnn_history'+str([variable_name]))
    df2.to_csv('saved_results/TTTT_linear_history'+str([variable_name]))

#     df1 = df1.loc[[df1.last_valid_index()]]
#     df2 = df2.loc[[df2.last_valid_i ndex()]]
#     df1['Architecture'] = 'DNN'
#     df2['Architecture'] = 'Linear'
#     df1.insert(0, 'Variable', [variable_name])
#     df2.insert(0, 'Variable', [variable_name])
#     df = pd.concat([df1,df2])
#     dfs = dfs.append(df)
    
# df = dfs[[
#     'Architecture',
#     'Variable',
#     'loss',
#     'val_loss'
# ]]
# df.rename(columns = {
#     'loss':'Training Loss',
#     'val_loss':'Test loss'
# },inplace=True)
# df = df.sort_values(by=['Architecture','Variable'], ascending=[False,False])
# print(df.to_latex(index=False))
# df.to_csv('saved_results/T_loss')
print('prethicktor complete')