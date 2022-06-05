import matplotlib.pyplot as plt
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




print('load Glam  data (RGI data connected with GlaThiDa thicknesses)')

Glam = pd.read_csv('Glam.csv')
Glam = Glam[[
#         'LAT',
#         'LON',
    'CenLon',
    'CenLat',
    'Area',
    'thickness',
    'Slope',
    'Zmin',
    'Zmed',
    'Zmax',
    'Aspect',
    'Lmax'
]]
Glam_phys = Glam[[
#         'LAT',
#         'LON',
#     'CenLon',
#     'CenLat',
    'Area',
    'thickness',
    'Slope',
    'Zmin',
    'Zmed',
    'Zmax',
    'Aspect',
    'Lmax'
]]

rootdir = 'sm4/'
dataset = Glam_phys


# split data for training and validation
Glam.name = 'Glam'
Glam_phys.name = 'Glam_phys'

# (train_features, test_features, train_labels, test_labels) = gl.data_splitter(Glam_phys)
(train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)


# define model hyperparameters
RS = range(0,25,1)
ep = 300

# name databases



# print(rootdir)
dnn_model = {}
predictions = pd.DataFrame()

print('loading and evaluating models...')
for arch in tqdm(os.listdir(rootdir)):
    for folder in os.listdir(rootdir + arch):
        if 'MULTI' in folder and 'dnn' in folder:
            
            if '0.1' in folder:
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

                pred_train = dnn_model[arch[3:] + '_' + folder].predict(train_features, verbose=0)
                pred_test = dnn_model[arch[3:] + '_' + folder].predict(test_features, verbose=0)
                avg_thickness = pd.Series(
                    (np.sum(pred_train) / len(pred_train)), name = 'avg train thickness'
                )

                avg_test_thickness = pd.Series(
                    (np.sum(pred_test) / len(pred_test)),  name = 'avg test thickness'
                )
                
                temp_df = pd.merge(
                    avg_thickness, avg_test_thickness, right_index=True, left_index=True
                )
                
                predictions = predictions.append(temp_df, ignore_index = True)
                predictions.loc[predictions.index[-1], 'model'] = folder
                predictions.loc[predictions.index[-1], 'test mae'] = mae_test
                predictions.loc[predictions.index[-1], 'train mae'] = mae_train
                predictions.loc[predictions.index[-1], 'architecture'] = arch[3:]
                predictions.loc[predictions.index[-1], 'learning rate'] = '0.1'
                predictions.loc[predictions.index[-1], 'validation split'] = '0.2'
                
            if '0.01' in folder:
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

                pred_train = dnn_model[arch[3:] + '_' + folder].predict(train_features, verbose=0)

                pred_test = dnn_model[arch[3:] + '_' + folder].predict(test_features, verbose=0)
                avg_thickness = pd.Series(
                    (np.sum(pred_train) / len(pred_train)), name = 'avg train thickness'
                )

                avg_test_thickness = pd.Series(
                    (np.sum(pred_test) / len(pred_test)),  name = 'avg test thickness'
                )
                
                
                temp_df = pd.merge(
                    avg_thickness, avg_test_thickness, right_index=True, left_index=True
                )
                predictions = predictions.append(temp_df, ignore_index=True)
                predictions.loc[predictions.index[-1], 'model']= folder
                predictions.loc[predictions.index[-1], 'test mae']= mae_test
                predictions.loc[predictions.index[-1], 'train mae']= mae_train
                predictions.loc[predictions.index[-1], 'architecture']= arch[3:]
                predictions.loc[predictions.index[-1], 'learning rate']= '0.01'
                predictions.loc[predictions.index[-1], 'validation split']= '0.2'          
            
            if '0.001' in folder:
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
                    (np.sum(pred_train) / len(pred_train)), name = 'avg train thickness'
                )

                avg_test_thickness = pd.Series(
                    (np.sum(pred_test) / len(pred_test)),  name = 'avg test thickness'
                )
                
                temp_df = pd.merge(
                    avg_thickness, avg_test_thickness, right_index=True, left_index=True
                )
                
                
                predictions = predictions.append(temp_df, ignore_index=True)
                predictions.loc[predictions.index[-1], 'model']= folder
                predictions.loc[predictions.index[-1], 'test mae']= mae_test
                predictions.loc[predictions.index[-1], 'train mae']= mae_train
                predictions.loc[predictions.index[-1], 'architecture']= arch[3:]            
                predictions.loc[predictions.index[-1], 'learning rate']= '0.001'
                predictions.loc[predictions.index[-1], 'validation split']= '0.2'                
                
predictions.rename(columns = {0:'avg train thickness'},inplace = True)

# calculate statistics
print('calculating statistics...')
deviations = pd.DataFrame()
for architecture in list(predictions['architecture'].unique()):
    for learning_rate in list(predictions['learning rate'].unique()):
        df = predictions[
            (predictions['architecture'] == architecture) & 
            (predictions['learning rate' ]== learning_rate)
        ]
    
        # step 1: calculate mean of numbers
        test_mae_mean = np.sum(df['test mae']) / len(df) 

        diff_sq = pd.Series()

        for test_mae in df['test mae']:
            # step 2: subtract the mean from each, then square the result
            step_2 = pd.Series((test_mae - test_mae_mean)**2)
            diff_sq = diff_sq.append(step_2, ignore_index = True)

        # step 3: work out the mean of the squared differences    
        mean_diff_sq = (np.sum(diff_sq) / len(diff_sq))

        # step 4: take the square root
        test_mae_std_dev = np.sqrt(mean_diff_sq)


       # repeat for train mae 

        # step 1: calculate mean of numbers
        train_mae_mean = np.sum(df['train mae']) / len(df) 

        diff_sq = pd.Series()

        for train_mae in df['train mae']:
            # step 2: subtract the mean from each, then square the result
            step_2 = pd.Series((train_mae - train_mae_mean)**2)
            diff_sq = diff_sq.append(step_2, ignore_index=True)

        # step 3: work out the mean of the squared differences    
        mean_diff_sq = (np.sum(diff_sq) / len(diff_sq))

        # step 4: take the square root
        train_mae_std_dev = np.sqrt(mean_diff_sq)

        # repeat process for train thicknesses
        thickness_train_mean = np.sum(df['avg train thickness']) / len(df) 
        
        for thickness in df['avg train thickness']:
            step_2 = pd.Series((thickness - thickness_train_mean)**2)
            diff_sq = diff_sq.append(step_2, ignore_index=True)
        mean_diff_sq = (np.sum(diff_sq) / len(diff_sq))
        train_thickness_std_dev = np.sqrt(mean_diff_sq)


        # repeat process for test thicknesses
        thickness_test_mean = np.sum(df['avg test thickness']) / len(df)   
        for thickness in df['avg test thickness']:
            step_2 = pd.Series((thickness - thickness_test_mean)**2)
            diff_sq = diff_sq.append(step_2, ignore_index=True)
        mean_diff_sq = (np.sum(diff_sq) / len(diff_sq))
        test_thickness_std_dev = np.sqrt(mean_diff_sq)

        # turn the last number computed into a series so it may be appended to build the table.
        # it will be dropped later, no worries.
        test_thick_std_dev = pd.Series(test_thickness_std_dev)

        deviations = deviations.append(test_thick_std_dev, ignore_index=True)   
        
        deviations.loc[deviations.index[-1], 'layer architecture']= architecture    
        
        
        
        deviations.loc[
            deviations.index[-1], 'model parameters'
        ] = dnn_model[architecture + '_' + dataset.name + '_dnn_MULTI_0.1_0.2_300_0'].count_params() 
        
        deviations.loc[deviations.index[-1], 'learning rate'] = learning_rate
        
        deviations.loc[deviations.index[-1], 'validation split']= 0.2
        
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
# bootstrapped ensembles for predicted column    
#drop that appended line from earlier. Probably a better way to go about it

deviations.drop(columns = {0},inplace = True)    
deviations = deviations.dropna()

# deviations['training split'] = deviations['test mae avg'] - deviations['train mae avg']
# too_low = deviations.index[deviations['training split'] < 0]
# deviations = deviations.drop(too_low)

deviations = deviations.sort_values('test mae avg')
deviations.to_csv('deviations_' + dataset.name + '.csv')


print('loading RGI...')
rootdir = '/data/fast0/datasets/rgi60-attribs/'
RGI_extra = pd.DataFrame()
for file in os.listdir(rootdir):
    f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(f, ignore_index = True)

RGI = RGI_extra[[
    'CenLat',
    'CenLon',
    'Slope',
    'Zmin',
    'Zmed',
    'Zmax',
    'Area',
    'Aspect',
    'Lmax'
]]

bad_zmed = RGI.loc[RGI['Zmed']<0].index
RGI = RGI.drop(bad_zmed)

bad_lmax = RGI.loc[RGI['Lmax']<0].index
RGI = RGI.drop(bad_lmax)

bad_slope = RGI.loc[RGI['Slope']<0].index
RGI = RGI.drop(bad_slope)

bad_aspect = RGI.loc[RGI['Aspect']<0].index
RGI = RGI.drop(bad_aspect)

RGI = RGI.reset_index()
RGI = RGI.drop('index', axis=1)

arch = deviations['layer architecture'].iloc[0]
lr = deviations['learning rate'].iloc[0]
vs = deviations['validation split'].iloc[0]

print('prethicking RGI using model trained on RGI data matched with GlaThiDa thicknesses...')
dfs = pd.DataFrame()
for rs in tqdm(RS):
    s = pd.Series(
        dnn_model[
        str(arch) +
        '_' +
        dataset.name +
        '_dnn_MULTI_'+
        str(lr)+
        '_'+
        str(vs)+
        '_300_'+ 
        str(rs)
    ].predict(RGI, verbose=0).flatten(), name = rs
    )
    dfs[rs] = s

RGI_prethicked = RGI.copy() 
RGI_prethicked['avg predicted thickness'] = 'NaN'
RGI_prethicked.is_copy = False
for i in tqdm(dfs.index):
    avg_predicted_thickness = np.sum(dfs.loc[i]) / len(dfs.loc[i])
    RGI_prethicked['avg predicted thickness'].loc[i] = avg_predicted_thickness

RGI_prethicked['predicted thickness std dev'] = 'NaN'

print('computing standard deviations and variances for RGI predicted thicknesses')

for i in tqdm(dfs.index):
    # step 1: calculate mean of numbers
    avg_predicted_thickness = np.sum(dfs.loc[i]) / len(dfs.loc[i])
    
    # step 2: subtract the mean from each, then square the result
    
    diff_sq = pd.Series()

    for q in dfs:
        
        avg_predicted_thickness - dfs[q].loc[i]
        step_2 = pd.Series((avg_predicted_thickness - dfs[q].loc[i])**2)
        diff_sq = diff_sq.append(step_2, ignore_index=True)
    
    # step 3: work out the mean of the squared differences    
    mean_diff_sq = (np.sum(diff_sq) / len(diff_sq))
    # step 4: take the square root
    prethick_std_dev = np.sqrt(mean_diff_sq)
    RGI_prethicked['predicted thickness std dev'].loc[i] = prethick_std_dev


RGI_prethicked['variance'] = (RGI_prethicked['predicted thickness std dev'])**2

RGI_prethicked.to_csv('RGI_prethicked_' + dataset.name + '.csv')


