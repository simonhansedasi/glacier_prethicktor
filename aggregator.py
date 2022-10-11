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
for index in predictions.index:
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
dfr = pd.DataFrame()
for rgi in tqdm(df['RGIId'].unique()):
    dft = df[df['RGIId'] == rgi]
    dfr = pd.concat([dfr, dft['RGIId'].index[-1]])
    dft = dft.drop([
        'RGIId',
        'CenLat',
        'CenLon',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Area',
        'Aspect',
        'Lmax',
        'region',
        'avg predicted thickness',
        'predicted thickness std dev',
        'volume km3', 
        'dataframe'
    ], axis = 1)
    dfr.loc[dfr.index[-1], 'Mean Thickness'] = dft.mean().mean()
    dfr.loc[dfr.index[-1], 'Thickness Std Dev'] = dft.stack().std()
dfr.to_csv('zults/aggregated_bootstrap_predictions.csv')