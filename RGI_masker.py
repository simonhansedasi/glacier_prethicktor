import os
from rasterio.warp import transform
from os.path import join
from glob import glob
from tqdm import tqdm
import pandas as pd
import raster2xyz
from raster2xyz.raster2xyz import Raster2xyz
import warnings
warnings.filterwarnings('ignore')
#Supress default INFO logging

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)



pth = '/data/fast1/glacierml/RGI/'

dem_path = pth + 'extracted_dem_files/MAPZEN/'
mask_path = pth + 'extracted_mask_files/'

dem_csv_path = pth + 'extracted_dem_csv_files/'
mask_csv_path = pth + 'extracted_mask_csv_files/'

dem_path_full = join(dem_path,'*.tif')
for file in tqdm(glob(dem_path_full)):
    file_left = file[53:]
    RGI_file = file_left[:14]
    
    dem_file = RGI_file + '_dem.tif'
    dem = join(dem_path, dem_file)
    
    mask_file = RGI_file + '_glacier_mask.tif'
    mask = join(mask_path, mask_file)
    
#     dem_csv = join(dem_csv_path, RGI_file + '_dem_trans.csv')
#     mask_csv = join(mask_csv_path, RGI_file + '_mask_trans.csv')
    
#     print(RGI_file)

    # translate dem data to csv file
#     transmogrifyer(dem)
    dir_out_dem = pth + 'extracted_dem_csv_files/'
    dem_out = join(dir_out_dem,RGI_file + '_dem_trans.csv')
    input_raster = dem
    rtxyz = Raster2xyz()
    rtxyz.translate(input_raster, dem_out)
    
    
    
    # translate glacier mask to csv file
    dir_out_mask = pth + 'extracted_mask_csv_files/'
    mask_out = join(dir_out_mask,RGI_file + '_mask_trans.csv')
    input_raster = mask
    rtxyz = Raster2xyz()
    rtxyz.translate(input_raster, mask_out)
    