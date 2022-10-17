import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import glacierml as gl

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

agg = df.groupby(['RGIId'])[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24',
].agg([np.mean, np.std, np.var])

agg.to_csv('sermeq_agg.csv')
dft = pd.DataFrame()
for rgi in tqdm(agg.index):
    dft = pd.concat([dft, pd.Series(rgi, name = 'RGIId')])
    mean_thickness = agg[[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24']].loc[rgi].mean()
    thickness_std = agg[[
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
        '11','12','13','14','15','16','17','18','19','20','21',
        '22','23','24']].loc[rgi].std()
    dft.loc[dft.index[-1], 'RGIId'] = agg['RGIId'].loc[rgi]
    dft.loc[dft.index[-1], 'Mean Thickness'] = mean_thickness
    dft.loc[dft.index[-1], 'Thickness std dev'] = thickness_std
dft = dft.drop_duplicates()
dft.to_csv('sermeq_aggregated_bootstrap_predictions.csv')