import pandas as pd
import numpy as np
import os
import geopy.distance
import chardet
from chardet import detect
from tqdm import tqdm

comb = pd.read_csv('RGI_tools/GlaThiDa_RGI_live.csv')
comb = comb.rename(columns = {'0':'distance'})

glacier = pd.read_csv('/data/fast1/glacierml/T_models/glacier.csv')
glacier = glacier.dropna(subset = ['mean_thickness'])

comb = comb[[
    'GlaThiDa_index',
    'RGI_index',
    'distance'
]]

combined_indexes = pd.DataFrame()
for GlaThiDa_index in comb['GlaThiDa_index'].index:
    df = comb[comb['GlaThiDa_index'] == GlaThiDa_index]
    f = df.loc[df[df['distance'] == df['distance'].min()].index]
    combined_indexes = combined_indexes.append(f)
combined_indexes

combined_indexes = combined_indexes.drop_duplicates(subset = ['GlaThiDa_index'])
combined_indexes = combined_indexes.reset_index()
combined_indexes = combined_indexes[[
    'GlaThiDa_index',
    'RGI_index',
    'distance'
]]

data = pd.DataFrame(columns = ['GlaThiDa_index', 'thickness'])
for GlaThiDa in combined_indexes['GlaThiDa_index'].index:
    glathida_thickness = glacier['mean_thickness'].iloc[GlaThiDa] 
    rgi_index = combined_indexes['RGI_index'].loc[GlaThiDa]  
    rgi = RGI_extra.iloc[[rgi_index]]
    
    data = data.append(rgi)
    data['GlaThiDa_index'].iloc[-1] = combined_indexes['GlaThiDa_index'].loc[GlaThiDa]
    data['thickness'].iloc[-1] = glathida_thickness
#     print(combined_indexes['GlaThiDa_index'].iloc[[GlaThiDa]])

data = data.drop_duplicates(subset = ['RGIId'])
data = data.reset_index()
data = data[[
#     'RGIId',
    'GlaThiDa_index',
    'thickness',
    'CenLon',
    'CenLat',
    'Area',
    'Zmin',
    'Zmed',
    'Zmax',
    'Slope',
    'Aspect',
    'Lmax'
]]
data.to_csv('Glam_2.csv')