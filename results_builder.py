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


    
print('please select module: sm01, sm02, sm031, sm1, sm2, sm3, sm4, sm5, sm6')
dir_list = ('sm01', 'sm02', 'sm1', 'sm2', 'sm031', 'sm3', 'sm4', 'sm5', 'sm6', 'all')

chosen_dir = input()

while chosen_dir not in dir_list:
    print('Please enter valid module selection: sm1, sm2, sm3, sm4','sm5','sm6')
    chosen_dir = input()    

if chosen_dir == 'sm01':
    df01 = gl.data_loader_01()
    gl.thickness_renamer(df01)
    dataset = df01
    dataset.name = 'df01'

if chosen_dir == 'sm02':
    df02 = gl.data_loader_02()
    gl.thickness_renamer(df02)
    dataset = df02
    dataset.name = 'df02'
    
if chosen_dir == 'sm031':
    df031 = gl.data_loader_031()
    gl.thickness_renamer(df031)
    dataset = df031
    dataset.name = 'df031'
    
    
    
if chosen_dir == 'sm1':
    df1 = gl.data_loader_1()
    gl.thickness_renamer(df1)
    dataset = df1
    dataset.name = 'df1'

elif chosen_dir == 'sm2':
    df2 = gl.data_loader_2()
    dataset = df2
    dataset.name = 'df2'

elif chosen_dir == 'sm3':
    df3 = gl.data_loader_3()
    gl.thickness_renamer(df3)
    dataset = df3
    dataset.name = 'df3'

elif chosen_dir == 'sm4':
    df4 = gl.data_loader_4()
    gl.thickness_renamer(df4)
    dataset = df4
    dataset.name = 'df4'
    
elif chosen_dir == 'sm5':
    df5 = gl.data_loader_5()
    reg = df5['region'].iloc[-1]
    df5 = df5.drop('region', axis=1)
    dataset = df5
    dataset.name = str('df5_' + str(reg))
    
    #code snippet to add a leading 0 to regional ID so it matches with RGI when built later
    if len(str(reg)) ==1:
        N = 1
        reg = str(reg).zfill(N + len(str(reg)))
    else:
        reg = reg

elif chosen_dir == 'sm6':
    df6 = gl.data_loader_6()
    reg = df6['region'].iloc[-1]
    df6 = df6.drop('region', axis=1)
    dataset = df6
    dataset.name = str('df6_' + str(reg))
    
    #code snippet to add a leading 0 to regional ID so it matches with RGI when built later
    if len(str(reg)) ==1:
        N = 1
        reg = str(reg).zfill(N + len(str(reg)))
    else:
        reg = reg

global_list = ('sm01', 'sm02', 'sm1', 'sm2', 'sm3', 'sm4')
region_list = ('sm5', 'sm6')
        
print('Load dropout layers? y / n')

dropout_input = input()
dropout_input_list = ('y', 'n')
while dropout_input not in dropout_input_list:
    print('Please select valid input: y / n')
    dropout_input = input()
if dropout_input == 'y':
    dropout = '1'
elif dropout_input == 'n':
    dropout = '0'

(train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)



rootdir = 'saved_models/' + chosen_dir + '/'

dnn_model = {}
predictions = pd.DataFrame()
print('loading and evaluating models...')
for arch in os.listdir(rootdir):
    print('layer architecture: ' + arch[3:])
    for folder in tqdm(os.listdir(rootdir + arch)):
        if 'MULTI' in folder and 'dnn' in folder and '_' + dropout + '_' in folder:
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
            predictions.loc[predictions.index[-1], 'dataset'] = dataset.name
            predictions.loc[predictions.index[-1], 'dropout'] = dropout
            
            if chosen_dir in global_list:
                predictions.loc[predictions.index[-1], 'region'] = 'g'
            if chosen_dir in region_list:
                predictions.loc[predictions.index[-1], 'region'] = int(reg)

            if '0.1' in folder:
                predictions.loc[predictions.index[-1], 'learning rate'] = '0.1'
            if '0.01' in folder:
                predictions.loc[predictions.index[-1], 'learning rate'] = '0.01'
            if '0.001' in folder:
                predictions.loc[predictions.index[-1], 'learning rate']= '0.001'
            if '25' in folder:
                predictions.loc[predictions.index[-1], 'epochs']= '25'
            if '35' in folder:
                predictions.loc[predictions.index[-1], 'epochs']= '35'
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
# deviations df to hold statistics for each model architecture
deviations = pd.DataFrame()
for architecture in list(predictions['architecture'].unique()):
    for learning_rate in list(predictions['learning rate'].unique()):
        for epochs in list(predictions['epochs'].unique()):
            for dataframe in list(predictions['dataset'].unique()):

                # select section of predictions that matches particular arch, lr, and ep


                df = predictions[
                    (predictions['architecture'] == architecture) & 
                    (predictions['learning rate' ] == learning_rate) &
                    (predictions['epochs'] == epochs) &
                    (predictions['dataset'] == dataframe)
                ]
                if df.empty:
                    break
                if not df.empty:
                    # find mean and std dev of test mae
                    test_mae_mean = np.mean(df['test mae'])
                    test_mae_std_dev = np.std(df['test mae'])

                    # find mean and std dev of train mae
                    train_mae_mean = np.mean(df['train mae'])
                    train_mae_std_dev = np.std(df['train mae'])

                    # find mean and std dev of predictions made based on training data
                    train_thickness_mean = np.mean(df['avg train thickness']) 
                    train_thickness_std_dev = np.std(df['avg train thickness'])

                    # find mean and std dev of predictions made based on test data
                    test_thickness_mean = np.mean(df['avg test thickness']) 
                    test_thickness_std_dev = np.std(df['avg test thickness'])

                    # put something in a series that can be appended to a df
                    s = pd.Series(train_thickness_mean)
                    deviations = deviations.append(s, ignore_index=True)  

                    # begin populating deviations table
                    deviations.loc[
                        deviations.index[-1], 'layer architecture'
                    ] = architecture  

                    deviations.loc[
                        deviations.index[-1], 'model parameters'
                    ] = dnn_model[
                            architecture + 
                            '_' + 
                            dataset.name + 
                            '_' +
                            dropout +
                            '_dnn_MULTI_' +
                            str(learning_rate) +
                            '_0.2_' +
                            str(int(epochs)) +
                            '_0'
                        ].count_params() 
                    
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
                    ] = learning_rate

                    deviations.loc[
                        deviations.index[-1], 'validation split'
                    ]= 0.2

                    deviations.loc[
                        deviations.index[-1], 'epochs'
                    ] = epochs

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
                    deviations.to_csv('zults/deviations_' + dataset.name + '_' + dropout + '.csv')





# rootdir = 'zults/'
# predictions = pd.DataFrame()
# deviations = pd.DataFrame()
# for file in tqdm(os.listdir(rootdir)):
#     if 'predictions' in file:
#         file_reader = pd.read_csv(rootdir + file)
#         predictions = predictions.append(file_reader, ignore_index = True)
    
#     if 'deviations' in file:
#         file_reader = pd.read_csv(rootdir + file)
#         deviations = deviations.append(file_reader, ignore_index = True)
        
# dataset = deviations['df'].loc[0]
(train_features, test_features, train_labels, test_labels) = gl.data_splitter(dataset)

# load and evaluate models

#build RGI specific to modules chosen


print('loading RGI...')
rootdir = '/data/fast0/datasets/rgi60-attribs/'
RGI_extra = pd.DataFrame()
for file in os.listdir(rootdir):
    file_reader = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(file_reader, ignore_index = True)

# select only RGI data that was used to train the model   
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

RGI = RGI.drop(RGI.loc[RGI['Zmed']<0].index)
RGI = RGI.drop(RGI.loc[RGI['Lmax']<0].index)
RGI = RGI.drop(RGI.loc[RGI['Slope']<0].index)
RGI = RGI.drop(RGI.loc[RGI['Aspect']<0].index)
RGI = RGI.reset_index()
RGI = RGI.drop('index', axis=1)
# RGI = RGI.rename(columns = {
# 'CenLon':'lon',
# 'CenLat':'lat',
# 'Area':'area',
# 'Slope':'mean_slope'
# })



if chosen_dir == 'sm01':
    RGI = RGI.rename(columns = {
        'CenLat':'lat',
        'CenLon':'lon',
        'Area':'area',
        'Slope':'mean_slope'
    })
    RGI = RGI[[
        'lat',
        'lon',
        'area',
        'mean_slope'
    ]]
    
if chosen_dir == 'sm02':
    RGI = RGI.rename(columns = {
        'CenLat':'LAT',
        'CenLon':'LON',
        'Area':'AREA',
        'Slope':'MEAN_SLOPE'
    })
    RGI = RGI[[
        'LAT',
        'LON',
        'AREA',
        'MEAN_SLOPE'
    ]]
    
    
    
if chosen_dir == 'sm031':
    RGI = RGI[[
        'CenLat',
        'CenLon',
        'Lmax',
        'Zmed',
        'Area',
        'Slope'
    ]]





if chosen_dir == 'sm5' or chosen_dir == 'sm6':
    print('loading RGI...')
    rootdir = '/data/fast1/glacierml/T_models/RGI/rgi60-attribs/'
    RGI_extra = pd.DataFrame()
    for file in tqdm(os.listdir(rootdir)):
        file_reader = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')

        # trim the RGIId entry to locate 2 digit region number.
        # Loop will only load desired RGI region based on these region tags
        region_1 = file_reader['RGIId'].iloc[-1][6:]
        region = region_1[:2]
        if str(region) == str(reg):
            RGI_extra = RGI_extra.append(file_reader, ignore_index = True)

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

    # here we want to drop any bad RGI data that can throw off predictions
    RGI = RGI.drop(RGI.loc[RGI['Zmed']<0].index)
    RGI = RGI.drop(RGI.loc[RGI['Lmax']<0].index)
    RGI = RGI.drop(RGI.loc[RGI['Slope']<0].index)
    RGI = RGI.drop(RGI.loc[RGI['Aspect']<0].index)
    RGI = RGI.reset_index()
    RGI = RGI.drop('index', axis=1)


deviations = deviations [[
    'layer architecture',
    'model parameters',
    'total inputs',
    'learning rate',
    'epochs',
    'test mae avg',
    'train mae avg',
    'test mae std dev',
    'train mae std dev'
]]
    
    
    
    
print(deviations.to_string())
# here we can select an entry from the deviations table to make predictions. Default is top entry
print('Please select model index to predict thicknesses for RGI')
selected_model = int(input())
while type(selected_model) != int:
    print('Please select model index to predict thicknesses for RGI')
    selected_model = int(input()) 


arch = deviations['layer architecture'].loc[selected_model]
lr = deviations['learning rate'].loc[selected_model]
# vs = deviations['validation split'].iloc[selected_model]
ep = deviations['epochs'].loc[selected_model]
print('layer architecture: ' + arch + ' learning rate: ' + str(lr) + ' epochs: ' + str(ep))
print('predicting RGI thicknesses using model trained on RGI data matched with GlaThiDa thicknesses...')

RS = range(0,25,1)
dfs = pd.DataFrame()
for rs in tqdm(RS):
    # each series is one random state of an ensemble of 25.
    # predictions are made on each random state and appended to a df as a column
    s = pd.Series(
        dnn_model[
            str(arch) +
            '_' +
            dataset.name +
            '_' +
            dropout + 
            '_dnn_MULTI_' +
            str(lr) +
            '_' +
            str(0.2) +
            '_' +
            str(ep) + 
            '_' + 
            str(rs)
        ].predict(RGI, verbose=0).flatten(), 
        name = rs
    )

    dfs[rs] = s


# make a copy of RGI to add predicted thickness and their statistics
RGI_prethicked = RGI.copy() 
RGI_prethicked['avg predicted thickness'] = 'NaN'
RGI_prethicked['predicted thickness std dev'] = 'NaN'


print('calculating average thickness across random state ensemble...')
# loop through predictions df and find average across each ensemble of 25 random states
for i in tqdm(dfs.index):
    avg_predicted_thickness = np.mean(dfs.loc[i])
    RGI_prethicked['avg predicted thickness'].loc[i] = avg_predicted_thickness


print('computing standard deviations and variances for RGI predicted thicknesses')
# loop through predictions df and find std dev across each ensemble of 25 random states
for i in tqdm(dfs.index):


    predicted_thickness_std_dev = np.std(dfs.loc[i])
    RGI_prethicked['predicted thickness std dev'].loc[i] = predicted_thickness_std_dev

RGI_prethicked.to_csv(
    'zults/RGI_predicted_' + dataset.name + '_' + dropout + '_' + arch + '_' + str(lr) + '_' + str(ep) + '.csv'
)






        
        

