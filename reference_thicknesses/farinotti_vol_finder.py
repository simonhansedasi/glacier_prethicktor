from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import scipy as sc
import cv2 as cv
import matplotlib.pyplot as plt
import rasterio as riot
from osgeo import gdal
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
Image.MAX_IMAGE_PIXELS = None



pth_1 = '/home/simonhans/data/prethicktor/RGI/outlines/Farinotti/'
# rootdir = '~'

df1 = pd.DataFrame()
df2 = pd.DataFrame()
for region_number in range(1,20,1):
            
    if len(str(region_number)) == 1:
        N = 1
        region_number = str(region_number).zfill(N + len(str(region_number)))
    else:
        str(region_number) == str(region_number)
        
    region_folder = pth_1 + 'RGI60-' + str(region_number) + '/'
    for file in tqdm(os.listdir(region_folder)):
        
        RGIId = file[:14]
        
        raster = gdal.Open(region_folder + file)
        got = raster.GetGeoTransform()
        # print(got)
        pixelSizeX = got[1]
        pixelSizeY = -got[5]
        pixel_area = (pixelSizeX * pixelSizeY)

        im = Image.open(region_folder + file)
        imarray = np.array(im)
        df = pd.DataFrame(imarray)
        df = df.replace(0.0, np.nan) 
        
        volume = df * pixel_area
        
        # print(volume_1)
        # print(volume_2)
        total_volume = np.nansum(volume)
        
        area = np.count_nonzero(~np.isnan(df)) * pixel_area

        # thickness_1 = total_volume_1 / area_1
        # thickness_2 = total_volume_2 / area_2
        
        
        # mean_volume = sum(volume) / area

        total_volume = pd.Series(total_volume)
        df1 = df1.append(total_volume, ignore_index = True)
        df1.loc[df1.index[-1], 'Area'] = area / 1e6
        df1.loc[df1.index[-1], 'RGIId'] = RGIId
    
df1.to_csv('Farinotti_vol.csv')

        
        
        
    #     break
    # break
        

#         break
#     break