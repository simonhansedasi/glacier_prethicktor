# import sys
# !{sys.executable} -m pip install geopy

import pandas as pd
import numpy as np
import os
import geopy.distance
import chardet
from chardet import detect
from tqdm import tqdm
pth = '/data/fast1/glacierml/T_models/'
T = pd.read_csv(pth + 'glacier.csv', low_memory = False)
T = T.dropna(subset = ['mean_thickness'])

rootdir = '/data/fast0/datasets/rgi60-attribs/'
RGI_extra = pd.DataFrame()
for file in os.listdir(rootdir):
    print(file)
    f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(f, ignore_index = True)

RGI_coordinates = RGI_extra[[
    'CenLat',
    'CenLon'
]]

L = pd.DataFrame(columns = ['GlaThiDa_index', 'RGI_index'])
glac = pd.DataFrame()
for T_idx in tqdm(T.index):
    GlaThiDa_coords = (T['lat'].loc[T_idx],
                       T['lon'].loc[T_idx])
#     print(GlaThiDa_coords)
    for RGI_idx in RGI_coordinates.index:
#         print(RGI_idx)
        RGI_coords = (RGI_coordinates['CenLat'].loc[RGI_idx],
                      RGI_coordinates['CenLon'].loc[RGI_idx])
        
        distance = geopy.distance.geodesic(GlaThiDa_coords,RGI_coords).km
        if 0 <= distance < 1:
#             print(RGI_coords)
            f = pd.Series(distance, name='distance')
            L = L.copy()
            L = L.append(f, ignore_index=True)
            L['GlaThiDa_index'].iloc[-1] = T_idx
            L['RGI_index'].iloc[-1] = RGI_idx
#             L['distance'] = distance
            L.to_csv('l.csv')
#         elif 0 < distance < 1:
#             f = pd.Series(distance, name='distance')
#             L = L.copy().append(f, ignore_index=True)
#             L['GlaThiDa_index'].iloc[-1] = T_idx
#             L['RGI_index'].iloc[-1] = RGI_idx
#             L.to_csv('l.csv')
        

#             print('bing! go fuck yourself')
            
#             glac = glac.append(L, ignore_index=True)
#             glac.drop_duplicates(subset = ['GlaThiDa_index'], keep = 'first')
#             glac.to_csv('glac_combined.csv')
            
            
# glac.to_csv('GlaThiDa_RGI_matched_indexes.csv')