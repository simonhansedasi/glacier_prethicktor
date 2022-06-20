import pandas as pd
from tqdm import tqdm
import os as os
# import glacierml as gl
import numpy as np
import tensorflow as tf
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# load dataset containing thicknesses
glacier = pd.read_csv('/data/fast1/glacierml/T_models/T.csv')
glacier = glacier.rename(columns = {
    'MEAN_THICKNESS':'thickness'
})


glacier = glacier[[
    'LAT',
    'LON',
    'AREA',
    'thickness',
    'MEAN_SLOPE'
    
]]


# assemble dataframe from raw data to include regional references 
dfs = pd.DataFrame()
rootdir = '/data/fast1/glacierml/T_models/regional_data_1/raw/'
for file in tqdm(os.listdir(rootdir)):
    df = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    region_and_number = file[:-4]
    region_number = region_and_number[:2]
    region = region_and_number[3:]
    
    df['geographic region'] = region
    df['region'] = region_number
    dfs = dfs.append(df, ignore_index=True)
    

dfs = dfs.drop_duplicates(subset = ['GlaThiDa_index'])
dfs = dfs.drop_duplicates(subset = ['RGIId'])

dfs = dfs.reset_index()

dfs = dfs[[
    'GlaThiDa_index',
    'RGI_index',
    'RGIId',
    'region',
    'geographic region'
]]

# load RGI data and compare to what is available
RGI_extra = pd.DataFrame()
rootdir = '/data/fast1/glacierml/T_models/attribs/rgi60-attribs/'
for file in os.listdir(rootdir):
    f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(f, ignore_index = True)
    
    
    region_and_number = file[:-4]
    region_number = region_and_number[:2]
    region = region_and_number[9:]
    df = dfs[dfs['region'] == region_number]
    
    percent_trainable = (len(df) / len(f)) * 100
    
    print(
        'region ' + str(region_number) + ' has ' + str(len(f)) + ' lines of data, ' +
        str(len(df)) + ' or ' + str(percent_trainable) + '%'
        ' of which are trainable with GlaThiDa thicknesses'    
    )


# create temp df to house regional data, merge it with RGI attributes and save as training data
for region in dfs['region'].unique():
    df = dfs[dfs['region'] == region]
    
    df['thickness'] = np.nan
    for df_idx in df.index:
        g_idx = df['GlaThiDa_index'].loc[df_idx]
        thickness = glacier['thickness'].loc[g_idx]
        df['thickness'].loc[df_idx] = thickness
    df = pd.merge(df, RGI_extra, on = 'RGIId')
    df = df[[
        'GlaThiDa_index',
        'RGI_index',
        'RGIId',
        'region',
        'geographic region',
        'CenLon',
        'CenLat',
        'Area',
        'Zmin',
        'Zmed',
        'Zmax',
        'Slope',
        'Aspect',
        'Lmax',
        'thickness'
    ]]
    df.to_csv(
        '/data/fast1/glacierml/T_models/regional_data_1/training_data/' + 
        df['region'].loc[0] + '_' + df['geographic region'].loc[0] + '.csv')