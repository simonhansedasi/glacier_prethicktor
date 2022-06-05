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
T = pd.read_csv(pth + 'T.csv', low_memory = False)
T = T.dropna(subset = ['MEAN_THICKNESS'])

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
RGI_coordinates

L = pd.DataFrame()
glac = pd.DataFrame()
for T_idx in tqdm(T.index):
    GlaThiDa_coords = (T['LAT'].loc[T_idx],
                       T['LON'].loc[T_idx])
#     print(GlaThiDa_coords)
    for RGI_idx in RGI_coordinates.index:
#         print(RGI_idx)
        RGI_coords = (RGI_coordinates['CenLat'].loc[RGI_idx],
                      RGI_coordinates['CenLon'].loc[RGI_idx])
        distance = geopy.distance.geodesic(GlaThiDa_coords,RGI_coords).km
        if distance == 0:
#             print('DING!')
#             print(T_idx)
#             print(RGI_idx)
#             print(RGI_coords)
            f = pd.Series(distance, name='distance')
            L = L.append(f, ignore_index=True)
            L['GlaThiDa_index'] = T_idx
            L['RGI_index'] = RGI_idx
            glac = glac.append(L, ignore_index=True)
#             glac_combined.to_csv('glac_combined.csv')
            break
            
glac.to_csv('GlaThiDa&RGI_matched_indexes_attempt2.csv')