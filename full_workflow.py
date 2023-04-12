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
    
# for i in range(7,9,1):
    
    # load training data information
    
# exclude_list = (False, True)

# for e in exclude_list:
# exclude = False
# for i in reversed(range(1,5,1)):
parameterization = str('4')

#     data = gl.parameterize_data(parameterization, 
#     #                             pth = '/data/fast1/glacierml/data')
#                                 pth = '/home/prethicktor/data/')

#     gl.build_model_ensemble(data, parameterization, useMP = False)

#     gl.assess_model_performance(data, parameterization)

# make glacier thickness estimates
model_statistics = pd.read_csv('zults/model_statistics_' + parameterization + '.csv')

model_statistics = model_statistics[[
'layer architecture',
]]

#     gl.estimate_thickness(
#         model_statistics, parameterization, useMP = False, verbose = True
#     )


        ####  Model Weighting

glac = gl.load_training_data(RGI_input = 'y', pth = '/home/prethicktor/data/')
arch = gl.list_architectures(parameterization = '3')
print('Compiling residuals')
dft = pd.DataFrame()
for architecture in tqdm(model_statistics['layer architecture'].unique()):
#     print(architecture)
    df_glob = gl.load_global_predictions(
        parameterization = parameterization, architecture = architecture
    )
    dft = pd.concat([dft, df_glob])

df = dft[[
        'layer architecture','RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
]]

glathida_estimates = pd.merge(glac, df, how = 'inner', on = 'RGIId')

est = glathida_estimates

for i in range(0,25,1):
    est['pr_'+str(i)] = (est[str(i)] - est['Thickness']) / est['Thickness']


model_list = [
     '0', '1', '2', '3', '4', '5', '6', '7', '8',
     '9', '10', '11', '12', '13', '14', '15', '16',
     '17', '18', '19', '20', '21', '22', '23', '24',
]
pool_list = [
     'pr_0', 'pr_1', 'pr_2', 'pr_3', 'pr_4', 'pr_5', 'pr_6', 'pr_7', 'pr_8',
     'pr_9', 'pr_10', 'pr_11', 'pr_12', 'pr_13', 'pr_14', 'pr_15', 'pr_16',
     'pr_17', 'pr_18', 'pr_19', 'pr_20', 'pr_21', 'pr_22', 'pr_23', 'pr_24',
]
weight_list = [
     'w_0', 'w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6', 'w_7', 'w_8',
     'w_9', 'w_10', 'w_11', 'w_12', 'w_13', 'w_14', 'w_15', 'w_16',
     'w_17', 'w_18', 'w_19', 'w_20', 'w_21', 'w_22', 'w_23', 'w_24',
]

weights = pd.DataFrame()
architecture_weights = pd.DataFrame()
print('Calculating weights')
for i in tqdm(est['layer architecture'].unique()):
#     print(i)
    dft = est[est['layer architecture'] == str(i)]
#     print(np.nanmean(dft[pool_list]))
#     print(
#         np.nanstd(dft[pool_list].to_numpy())
#     )
    bias = np.mean(dft[pool_list].to_numpy()) * np.mean(dft[model_list].to_numpy())


    q75, q25 = np.nanpercentile(dft[pool_list], [75,25])
    sigma = ((q75 - q25) * np.mean(dft[model_list].to_numpy()) / 1.5

    w = pd.Series(
        abs(bias) + sigma**2, 
        name = 'weight'
    )
#     print(w)
#     break
    architecture_weights = pd.concat([architecture_weights, w])
    architecture_weights = architecture_weights.reset_index()
    architecture_weights = architecture_weights.drop('index', axis = 1)
    architecture_weights.loc[architecture_weights.index[-1], 'layer architecture'] = i
    architecture_weights.loc[architecture_weights.index[-1], 'bias'] = bias
    architecture_weights.loc[architecture_weights.index[-1], 'std'] = sigma

architecture_weights = architecture_weights.rename(columns = {0:'architecture weight'})
architecture_weights['var'] = architecture_weights['std']**2
architecture_weights.to_csv('architecture_weights.csv')
    #### Model Weighting Close

gl.calculate_RGI_thickness_statistics(
    model_statistics, parameterization
)

