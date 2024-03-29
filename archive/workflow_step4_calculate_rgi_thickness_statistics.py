import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glacierml as gl
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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


print('Gathering architectures...')
arch_list = gl.list_architectures(parameterization = parameterization)
arch_list = arch_list.reset_index()
arch_list = arch_list.drop('index', axis = 1)

df = pd.DataFrame(columns = {
        'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
})
# df = pd.merge(df, arch_list, on = 'RGIId', how = 'inner')
print(arch_list)
arch_list = arch_list.sort_values('layer architecture')
# print(len(arch_list['architecture'].unique()))
# print(arch_list['architecture'].unique())
print('Architectures listed')
# print(list(predictions))
# print(predictions['architecture'].unique())
print('Compiling predictions...')
for arch in tqdm(arch_list['layer architecture'].unique()):
#     print(arch)
#     break
#     idx = index
#     print(idx)

#     coregistration =  arch_list['coregistration'].iloc[idx]
#     architecture = '_' + arch_list['architecture'].iloc[idx]
    df_glob = gl.load_global_predictions(
        parameterization = parameterization,
        architecture = arch,
    )
    

    df = pd.concat([df,df_glob])
#     print(df)
# print(df)
statistics = pd.DataFrame()
for file in (os.listdir('zults/')):
    if 'statistics_' + parameterization in file:
        file_reader = pd.read_csv('zults/' + file)
        statistics = pd.concat([statistics, file_reader], ignore_index = True)

#     print(file)
#     break
# deviations = deviations.dropna()
# print(list(statistics))
# statistics = statistics.rename(columns = {'layer architecture':'architecture'}, inplace = True)

df = pd.merge(df, statistics, on = 'layer architecture')
# df = pd.merge(df, statistics, on = 'architecture')


df = df[[
        'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24','architecture weight 1'
]]

compiled_raw = df.groupby('RGIId')[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24','architecture weight 1'
]

print('Predictions compiled')
print('Aggregating statistics...')
dft = pd.DataFrame()
for this_rgi_id, obj in tqdm(compiled_raw):
    rgi_id = pd.Series(this_rgi_id, name = 'RGIId')
#     print(f"Data associated with RGI_ID = {this_rgi_id}:")
    dft = pd.concat([dft, rgi_id])
    dft = dft.reset_index()
    dft = dft.drop('index', axis = 1)
    
    
    obj['weight'] = obj['architecture weight 1'] + 1 / (obj[['0', '1', '2', '3', '4',
                                                     '5', '6', '7', '8', '9',
                                                     '10','11','12','13','14',
                                                     '15','16','17','18','19',
                                                     '20','21','22','23','24']].var(axis = 1))
    
    
    obj['weighted mean'] = obj['weight'] * obj[['0', '1', '2', '3', '4',
                                               '5', '6', '7', '8', '9',
                                               '10','11','12','13','14',
                                               '15','16','17','18','19',
                                               '20','21','22','23','24']].mean(axis = 1)
    
    
    weighted_glacier_mean = sum(obj['weighted mean']) / sum(obj['weight'])

    
    stacked_object = obj[[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
    ]].stack()
    
    glacier_count = len(stacked_object)
#     dft.loc[dft.index[-1], 'Weighted Mean Thickness'] = weighted_glacier_mean
    dft.loc[dft.index[-1], 'Mean Thickness'] = stacked_object.mean()
    dft.loc[dft.index[-1], 'Median Thickness'] = stacked_object.median()
    dft.loc[dft.index[-1],'Thickness Std Dev'] = stacked_object.std()
    
    statistic, p_value = shapiro(stacked_object)    
    dft.loc[dft.index[-1],'Shapiro-Wilk statistic'] = statistic
    dft.loc[dft.index[-1],'Shapiro-Wilk p_value'] = p_value

    
    q75, q25 = np.percentile(stacked_object, [75, 25])    
    dft.loc[dft.index[-1],'IQR'] = q75 - q25 
    
    lower_bound = np.percentile(stacked_object, 50 - 34.1)
    median = np.percentile(stacked_object, 50)
    upper_bound = np.percentile(stacked_object, 50 + 34.1)
    
    dft.loc[dft.index[-1],'Lower Bound'] = lower_bound
    dft.loc[dft.index[-1],'Upper Bound'] = upper_bound
    dft.loc[dft.index[-1],'Median Value'] = median
    dft.loc[dft.index[-1],'Total estimates'] = glacier_count
    
dft = dft.rename(columns = {
    0:'RGIId'
})
dft = dft.drop_duplicates()
dft.to_csv(
    'predicted_thicknesses/sermeq_aggregated_bootstrap_predictions_coregistration_' + 
    parameterization + '.csv'
          )