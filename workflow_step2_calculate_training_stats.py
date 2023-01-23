import pandas as pd
import glacierml as gl
import numpy as np
import os
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parameterization = str(  4  )
config = configparser.ConfigParser()
config.read('model_parameterization.ini')

data = gl.load_training_data(
    RGI_input = config[parameterization]['RGI_input'],
    scale = config[parameterization]['scale'],
)
data.name = config[parameterization]['datasetname'] 
data = data.drop([
    'RGIId','region', 'RGI Centroid Distance', 
    'AVG Radius', 'Roundness', 'distance test', 'size difference'
], axis = 1)


rootdir = 'saved_models/' + parameterization + '/'
(train_features, test_features, train_labels, test_labels) = gl.split_data(dataset)
# dnn_model = {}
# print(' ')

model_predictions = pd.DataFrame()
model_statistics = pd.DataFrame()
# dropout_input = dropout_input_iter

print('loading and evaluating models...')
for arch in tqdm( os.listdir(rootdir)):       
#     print('layer architecture: ' + arch[3:])
    pth = os.path.join(rootdir, arch)
    for folder in (os.listdir(pth)):
        architecture = arch[3:]
#         print(architecture)
        model_loc = (
            rootdir + 
            arch + 
            '/' + 
            folder
        )

        model_name = folder
        dnn_model = gl.load_dnn_model(model_loc)
#         print(dnn_model)
        df = gl.evaluate_model(architecture, model_name, data, dnn_model)

        model_predictions = pd.concat([model_predictions, df], ignore_index = True)
#     break
# print(model_predictions['architecture'])
# print(list(model_predictions))
model_predictions.rename(columns = {0:'avg train thickness'},inplace = True)
model_predictions.to_csv('zults/model_predictions_' + data.name  + '.csv')
# calculate statistics
print('calculating statistics...')
print(' ')
dnn_model = {}

for arch in tqdm(list(model_predictions['architecture'].unique())):
    model_thicknesses = model_predictions[model_predictions['architecture'] == arch]


    model_name = ('0')

    model_loc = (
        rootdir + 
        arch + 
        '/' +
        '0'
    )
#     print(model_loc)
    isdir = os.path.isdir(model_loc)
#     print(isdir)
    if isdir == False:
        print('model not here, calculating next model')
    elif isdir == True:
        
        
        dnn_model = gl.load_dnn_model(model_loc)
        df = gl.calculate_model_avg_statistics(
            dnn_model,
            arch,
            data,
            model_thicknesses
        )

        model_statistics = pd.concat(
            [model_statistics, df], ignore_index = True
        )
        #         print(list(model_statistics))
    
    
model_statistics['architecture weight 1'] = (
    sum(model_statistics['test mae avg']) / model_statistics['test mae avg']
)
model_statistics['architecture weight 2'] = (
    model_statistics['test mae avg'] / sum(model_statistics['test mae avg'])
)
model_statistics.to_csv(
    'zults/model_statistics_' + 
    data.name + 
    '.csv'
)
