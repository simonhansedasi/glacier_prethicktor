import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glacierml as gl
from scipy.stats import shapiro

predictions = gl.predictions_finder()
predictions = predictions.reset_index()
predictions = predictions.drop('index', axis = 1)

df = pd.DataFrame(columns = {
        'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
})

for index in tqdm(predictions.index):
    idx = index
#     print(idx)

    training_module =  predictions['coregistration'].iloc[idx]
    architecture = predictions['architecture'].iloc[idx]
    learning_rate = predictions['learning rate'].iloc[idx]
    epochs = '2000'
    df_glob = gl.global_predictions_loader(
        training_module = training_module,
        architecture = architecture,
        learning_rate = learning_rate,
        epochs = epochs

    )
    

    df = pd.concat([df,df_glob])

df = df[[
        'RGIId','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
]]

compiled_raw = df.groupby('RGIId')[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
]

dft = pd.DataFrame()
for this_rgi_id, obj in tqdm(compiled_raw):
    rgi_id = pd.Series(this_rgi_id, name = 'RGIId')
#     print(f"Data associated with RGI_ID = {this_rgi_id}:")
#     print(this_rgi_id)
#     break
    dft = pd.concat([dft, rgi_id])
    dft = dft.reset_index()
    dft = dft.drop('index', axis = 1)
    
    stacked_object = obj[[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
    ]].stack()
    
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
    
    
    
dft = dft.rename(columns = {
    0:'RGIId'
})
dft = dft.drop_duplicates()
dft.to_csv('aggregated/sermeq_aggregated_bootstrap_predictions.csv')