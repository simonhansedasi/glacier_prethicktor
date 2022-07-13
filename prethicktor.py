import glacierml as gl
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.util import deprecation
import logging
import warnings
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)

print('please select module: sm1, sm2, sm3, sm4, sm5, sm6, sm7')
dir_list = ('sm01', 'sm02', 'sm1', 'sm2', 'sm031', 'sm3', 'sm4', 'sm5', 'sm6', 'sm7')

chosen_dir = input()

while chosen_dir not in dir_list:
    print('Please enter valid module selection: sm1, sm2, sm3, sm4, sm5, sm6, sm7')
    chosen_dir = input()    

if chosen_dir == 'sm1':
    df1 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
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
        root_dir = '/home/prethicktor/data/',
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
        root_dir = '/home/prethicktor/data/',
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
        root_dir = '/home/prethicktor/data/',
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
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
        # region_selection = 1,
        area_scrubber = 'off',
        # anomaly_input = 5
    )
    df5 = df5.drop('Zmed', axis = 1)
    dataset = df5
    dataset.name = 'df5'
    res = 'sr5'

if chosen_dir == 'sm6':
    for region_selection in range(1,20,1):    
        RGI = gl.RGI_loader(
            pth = '/home/prethicktor/data/RGI/rgi60-attribs/',
            region_selection = int(region_selection)
        )
        
        
        if len(str(region_selection)) == 1:
            N = 1
            region_selection = str(region_selection).zfill(N + len(str(region_selection)))
        else:
            region_selection = str(region_selection)  
        
        print(region_selection)

        df6 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'r',
            region_selection = int(region_selection),
            area_scrubber = 'off'
        )
        
        df6 = df6.drop('region', axis=1)
        dataset = df6
        dataset.name = str('df6_' + str(region_selection))
        res = 'sr6'
        
        if len(df6) < 3:
            print('pass')
            pass
        
        elif len(df6) >= 3:
            deviations_1 = pd.read_csv('zults/deviations_' + dataset.name + '_1.csv')
            deviations_2 = pd.read_csv('zults/deviations_' + dataset.name + '_0.csv')
            deviations = pd.concat([deviations_1, deviations_2])
            deviations = deviations.reset_index()
            rootdir = '/home/prethicktor/data/RGI/rgi60-attribs/'

            selected_model = 0
            while type(selected_model) != int:
                print('Please select model index to predict thicknesses for RGI')
                selected_model = int(input()) 


            arch = deviations['layer architecture'].loc[selected_model]
            lr = deviations['learning rate'].loc[selected_model]
            # vs = deviations['validation split'].iloc[selected_model]
            ep = deviations['epochs'].loc[selected_model]
            dropout = deviations['dropout'].loc[selected_model]
            print(
                'layer architecture: ' + arch + 
                ' learning rate: ' + str(lr) + 
                ' epochs: ' + str(ep)
            )
            
            print('predicting thicknesses...')
            dnn_model = {}
            rootdir = 'saved_models/' + chosen_dir + '/'
            RS = range(0,25,1)
            dfs = pd.DataFrame()
            for rs in tqdm(RS):
            # each series is one random state of an ensemble of 25.
            # predictions are made on each random state and appended to a df as a column
                model = (
                    str(arch) +
                    '_' +
                    dataset.name +
                    '_' +
                    str(dropout) + 
                    '_dnn_MULTI_' +
                    str(lr) +
                    '_' +
                    str(0.2) +
                    '_' +
                    str(ep) + 
                    '_' + 
                    str(rs)
                )

                path = (
                    rootdir + 'sm_' + arch + '/' + 
                    dataset.name + 
                    '_' + 
                    str(dropout) + 
                    '_dnn_MULTI_' + 
                    str(lr) + 
                    '_' +
                    str(0.2) +
                    '_' +
                    str(ep) + 
                    '_' + 
                    str(rs)
                )

                dnn_model[model] = tf.keras.models.load_model(path)

                s = pd.Series(
                    dnn_model[model].predict(RGI, verbose=0).flatten(), 
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
                'zults/RGI_predicted_' +
                dataset.name + 
                '_' + 
                str(dropout) + 
                '_' + 
                arch + 
                '_' + 
                str(lr) + 
                '_' + 
                str(ep) + 
                '.csv'
            )
        
if chosen_dir == 'sm7':
    df7 = gl.data_loader(
        root_dir = '/home/prethicktor/data/',
        RGI_input = 'y',
        scale = 'g',
#                 region_selection = 1,
        area_scrubber = 'off'
#                 anomaly_input = 5
    )
    dataset = df7
    dataset.name = 'df7'
    res = 'sr7'
    deviations_1 = pd.read_csv('zults/deviations_' + dataset.name + '_1.csv')
    deviations_2 = pd.read_csv('zults/deviations_' + dataset.name + '_0.csv')
    deviations = pd.concat([deviations_1, deviations_2])
    deviations = deviations.reset_index()
    rootdir = '/home/prethicktor/data/RGI/rgi60-attribs/'
    for region_selection in range(1,20,1):
        RGI = gl.RGI_loader(
            pth = '/home/prethicktor/data/RGI/rgi60-attribs/',
            region_selection = int(region_selection)
        )
        if len(str(region_selection)) == 1:
            N = 1
            region_selection = str(region_selection).zfill(N + len(str(region_selection)))
        else:
            region_selection = region_selection
            
        deviations = deviations [[
            'layer architecture',
            'dropout',
            # 'model parameters',
            # 'total inputs',
            'learning rate',
            'epochs',
            # 'test mae avg',
            # 'train mae avg',
            # 'test mae std dev',
            # 'train mae std dev'
        ]]

#         print(deviations.to_string())
        # here we can select an entry from the deviations table to make predictions. Default is top entry
        print('Please select model index to predict thicknesses for RGI')
        selected_model = 0
        while type(selected_model) != int:
            print('Please select model index to predict thicknesses for RGI')
            selected_model = int(input()) 


        arch = deviations['layer architecture'].loc[selected_model]
        lr = deviations['learning rate'].loc[selected_model]
        # vs = deviations['validation split'].iloc[selected_model]
        ep = deviations['epochs'].loc[selected_model]
        dropout = deviations['dropout'].loc[selected_model]
        print(
            'layer architecture: ' + arch + 
            ', learning rate: ' + str(lr) + 
            ', epochs: ' + str(ep) +
            ', dataset: ' + dataset.name +
            ', region: ' + str(region_selection)
        )
        
        print('predicting thicknesses...')
        dnn_model = {}
        rootdir = 'saved_models/' + chosen_dir + '/'
        RS = range(0,25,1)
        dfs = pd.DataFrame()
        for rs in tqdm(RS):
        # each series is one random state of an ensemble of 25.
        # predictions are made on each random state and appended to a df as a column
            model = (
                str(arch) +
                '_' +
                dataset.name +
                '_' +
                str(dropout) + 
                '_dnn_MULTI_' +
                str(lr) +
                '_' +
                str(0.2) +
                '_' +
                str(ep) + 
                '_' + 
                str(rs)
            )

            path = (
                rootdir + 'sm_' + arch + '/' + 
                dataset.name + 
                '_' + 
                str(dropout) + 
                '_dnn_MULTI_' + 
                str(lr) + 
                '_' +
                str(0.2) +
                '_' +
                str(ep) + 
                '_' + 
                str(rs)
            )

            dnn_model[model] = tf.keras.models.load_model(path)

            s = pd.Series(
                dnn_model[model].predict(RGI, verbose=0).flatten(), 
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
            'zults/RGI_predicted_' +
            dataset.name + '_' + str(region_selection) +
            '_' + 
            str(dropout) + 
            '_' + 
            arch + 
            '_' + 
            str(lr) + 
            '_' + 
            str(ep) + 
            '.csv'
        )
if chosen_dir == 'sm7' and str(region_selection) == '19':
    raise SystemExit
    
    
if chosen_dir == 'sm6' and str(region_selection) == '19':
    raise SystemExit

    
deviations_1 = pd.read_csv('zults/deviations_' + dataset.name + '_1.csv')
deviations_2 = pd.read_csv('zults/deviations_' + dataset.name + '_0.csv')
deviations = pd.concat([deviations_1, deviations_2])
deviations = deviations.reset_index()
rootdir = '/home/prethicktor/data/RGI/rgi60-attribs/'



RGI_extra = pd.DataFrame()
for file in os.listdir(rootdir):
    file_reader = pd.read_csv(rootdir + file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = pd.concat([RGI_extra, file_reader], ignore_index = True)

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

if chosen_dir == 'sm1':
    RGI = RGI.rename(columns = {
        'CenLat':'Lat',
        'CenLon':'Lon',
        'Area':'Area',
        'Slope':'Mean Slope'
    })
    RGI = RGI[[
        'Lat',
        'Lon',
        'Area',
        'Mean Slope'
    ]]

if chosen_dir == 'sm5':
    RGI = RGI[[
        'CenLat',
        'CenLon',
        'Slope',
        'Zmin',
        # 'Zmed',
        'Zmax',
        'Area',
        'Aspect',
        'Lmax'
    ]]       
    
    
    

deviations = deviations [[
'layer architecture',
'dropout',
# 'model parameters',
# 'total inputs',
'learning rate',
'epochs',
# 'test mae avg',
# 'train mae avg',
# 'test mae std dev',
# 'train mae std dev'
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
dropout = deviations['dropout'].loc[selected_model]
print('layer architecture: ' + arch + ' learning rate: ' + str(lr) + ' epochs: ' + str(ep))
print('predicting thicknesses...')
dnn_model = {}
rootdir = 'saved_models/' + chosen_dir + '/'
RS = range(0,25,1)
dfs = pd.DataFrame()
for rs in tqdm(RS):
# each series is one random state of an ensemble of 25.
# predictions are made on each random state and appended to a df as a column
    model = (
        str(arch) +
        '_' +
        dataset.name +
        '_' +
        str(dropout) + 
        '_dnn_MULTI_' +
        str(lr) +
        '_' +
        str(0.2) +
        '_' +
        str(ep) + 
        '_' + 
        str(rs)
    )
    
    path = (
        rootdir + 'sm_' + arch + '/' + 
        dataset.name + 
        '_' + 
        str(dropout) + 
        '_dnn_MULTI_' + 
        str(lr) + 
        '_' +
        str(0.2) +
        '_' +
        str(ep) + 
        '_' + 
        str(rs)
    )
    
    dnn_model[model] = tf.keras.models.load_model(path)
    
    s = pd.Series(
        dnn_model[model].predict(RGI, verbose=0).flatten(), 
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
    'zults/RGI_predicted_' +
    dataset.name + 
    '_' + 
    str(dropout) + 
    '_' + 
    arch + 
    '_' + 
    str(lr) + 
    '_' + 
    str(ep) + 
    '.csv'
)