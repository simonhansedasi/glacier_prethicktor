from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# import glacierml as gl

pth_2 = '/home/simonhans/data/prethicktor/RGI/rgi60-attribs/'
RGI_extra = pd.DataFrame(columns = ['Farinotti Mean Thickness'])
for file in tqdm(os.listdir(pth_2)):
    file_reader = pd.read_csv(pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = pd.concat([RGI_extra, file_reader], ignore_index = True)

    # select only RGI data that was used to train the model   
RGI = RGI_extra[[
    'RGIId',
    'Farinotti Mean Thickness'
]]
print(RGI)
# print(RGI['RGIId'])

pth_1 = '/home/simonhans/data/prethicktor/RGI/outlines/'
# rootdir = '~'
for region_number in range(1,20,1):
            
    if len(str(region_number)) == 1:
        N = 1
        region_number = str(region_number).zfill(N + len(str(region_number)))
    else:
        str(region_number) == str(region_number)
        
    region_folder = pth_1 + 'RGI60-' + str(region_number) + '/'
    for file in tqdm(os.listdir(region_folder)):
        im = Image.open(region_folder + file)
        imarray = np.array(im)
        df = pd.DataFrame(imarray)
        df = df.replace(0.0, np.nan)
        mean_glacier_thickness = np.nanmean(np.nanmean(df.to_numpy()))
        
        RGI['Farinotti Mean Thickness'].loc[RGI['RGIId'] == file[:14]] = mean_glacier_thickness
        
        
RGI.to_csv('Farinotti_mean_thickness_RGI_ID_2.csv')