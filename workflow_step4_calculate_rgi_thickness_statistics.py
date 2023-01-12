import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glacierml as gl
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

parameterization = input()

if parameterization == 'sm1':
    coregistration = 'df1'
if parameterization == 'sm2':
    coregistration = 'df2'
if parameterization == 'sm3':
    coregistration = 'df3'
if parameterization == 'sm4':
    coregistration = 'df4'
if parameterization == 'sm5':
    coregistration = 'df5'
if parameterization == 'sm6':
    coregistration = 'df6'
if parameterization == 'sm7':
    coregistration = 'df7'
if parameterization == 'sm8':
    coregistration = 'df8'
if parameterization == 'sm9':
    coregistration = 'df9'

print('Loading predictions...')
predictions = gl.find_predictions(coregistration = coregistration)
predictions = predictions.reset_index()
predictions = predictions.drop('index', axis = 1)

df = pd.DataFrame(columns = {
        'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
})

print('Predictions loaded')

print('Compiling predictions...')
for index in tqdm(predictions.index):
    idx = index
#     print(idx)

    coregistration =  predictions['coregistration'].iloc[idx]
    architecture = '_' + predictions['architecture'].iloc[idx]
    learning_rate = predictions['learning rate'].iloc[idx]
    epochs = '2000'
    df_glob = gl.load_global_predictions(
        coregistration = coregistration,
        architecture = architecture,
        learning_rate = learning_rate,
        epochs = epochs

    )
    

    df = pd.concat([df,df_glob])
    
    
    
deviations = pd.DataFrame()
for file in tqdm(os.listdir('zults/')):
    if 'deviations' in file and coregistration in file:
        file_reader = pd.read_csv('zults/' + file)
        deviations = pd.concat([deviations, file_reader], ignore_index = True)
    
# deviations = deviations.dropna()
df = pd.merge(df, deviations, on = 'layer architecture')
    
    

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
    
    
#     obj['weight'] = obj['architecture weight'] + 1 / (obj[['0', '1', '2', '3', '4',
#                                                      '5', '6', '7', '8', '9',
#                                                      '10','11','12','13','14',
#                                                      '15','16','17','18','19',
#                                                      '20','21','22','23','24']].var(axis = 1))
    
    
#     obj['weighted mean'] = obj['weight'] * obj[['0', '1', '2', '3', '4',
#                                                '5', '6', '7', '8', '9',
#                                                '10','11','12','13','14',
#                                                '15','16','17','18','19',
#                                                '20','21','22','23','24']].mean(axis = 1)
    
    
#     weighted_glacier_mean = sum(obj['weighted mean']) / sum(obj['weight'])

    
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
    coregistration + '.csv'
          )