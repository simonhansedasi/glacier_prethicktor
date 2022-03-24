# import sys
# !{sys.executable} -m pip install raster2xyz
from osgeo import gdal
import fiona
import os
import urllib.request
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rasterio as rio
import xarray as xr
import os
import numpy as np
from rasterio.warp import transform
from os.path import join
from glob import glob
from tqdm import tqdm
import rioxarray as rxr
import rasterio.mask
import rasterstats as rst
import pandas as pd
from PIL import Image
import tifftools
import pyproj
import raster2xyz
import io
import sys
from raster2xyz.raster2xyz import Raster2xyz
import warnings
warnings.filterwarnings('ignore')
#Supress default INFO logging

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

def transmogrifyer(dem):
    dem_reader = xr.open_rasterio(dem)
    ny, nx = len(dem_reader['y']), len(dem_reader['x'])
    x, y = np.meshgrid(dem_reader['x'], dem_reader['y'])
    lon, lat = transform(dem_reader.crs, {'init': 'EPSG:4326'},
                     x.flatten(), y.flatten())
    
    dem_lon = np.asarray(lon).reshape((ny, nx))
    dem_lat = np.asarray(lat).reshape((ny, nx))
    dem_reader.coords['lon'] = (('y', 'x'), dem_lon)
    dem_reader.coords['lat'] = (('y', 'x'), dem_lat)
    dem_greyscale = dem_reader.mean(dim="band")
    
    ax = plt.subplot(projection=ccrs.PlateCarree())
    dem_plot = dem_greyscale.plot(ax=ax, x='lon', y='lat', transform=ccrs.PlateCarree(),
                                  cmap='Greys_r', add_colorbar=False, shading=None)
    

def plotter(mask_data):
    X = mask_data['x'].values.reshape(ny,nx).T
    Y = mask_data['y'].values.reshape(ny,nx).T
    Z = mask_data['z'].values.reshape(ny,nx).T

    plt.pcolormesh(X,Y,Z)
    plt.show()


dem_path = "/home/sa42/data/glac/RGI_TOPO/extracted_dem_files/MAPZEN"
mask_path = "/home/sa42/data/glac/RGI_TOPO/extracted_mask_files"

dem_csv_path = "/home/sa42/data/glac/RGI_TOPO/extracted_dem_csv_files/"
mask_csv_path = "/home/sa42/data/glac/RGI_TOPO/extracted_mask_csv_files/"

dem_path_full = join(dem_path,"*.tif")
for file in tqdm(glob(dem_path_full)):
    file_left = file[57:]
    RGI_file = file_left[:14]
    
    dem_file = RGI_file + "_dem.tif"
    dem = join(dem_path, dem_file)
    
    mask_file = RGI_file + "_glacier_mask.tif"
    mask = join(mask_path, mask_file)
    
    dem_csv = join(dem_csv_path, RGI_file + "_dem_trans.csv")
    mask_csv = join(mask_csv_path, RGI_file + "_mask_trans.csv")
    
    print(RGI_file)
    print(dem)
    print(dem_csv)
    print(mask)
    print(mask_csv)
#     transmogrifyer(mask)
#     dir_out_mask = "/home/sa42/data/glac/RGI_TOPO/extracted_mask_csv_files/"
#     mask_out = join(dir_out_mask,RGI_file + "_mask_trans.csv")

#     testy_out_path = "/home/sa42/data/glac/RGI_TOPO/testy_testy"
#     testy_out = join(testy_out_path, RGI_file + \.csv\)
#     input_raster = dem
#     rtxyz = Raster2xyz()
#     rtxyz.translate(input_raster, testy_out)
#     testy = pd.DataFrame(testy_out)
#     testy.to_csv(test.csv)
#     mask_data = pd.read_csv(test.csv)
    break

# dem_reader = xr.open_rasterio(dem)
# ny, nx = len(dem_reader['y']), len(dem_reader['x'])
# x, y = np.meshgrid(dem_reader['x'], dem_reader['y'])
# lon, lat = transform(dem_reader.crs, {'init': 'EPSG:4326'},
#                  x.flatten(), y.flatten())



mask_data = pd.read_csv(mask_csv)
mask_data.rename(columns = {"z":"z_mask"}, inplace=True)


dem_data = pd.read_csv(dem_csv)
mask_and_dem = pd.merge(mask_data, dem_data, how="inner")


# mask_and_dem = mask_and_dem[mask_and_dem["z_mask"] !=0]
# mask_and_dem = mask_and_dem.drop("z_mask",axis=1)

RGI = pd.read_csv("RGI.csv")
mask_and_dem = mask_and_dem.assign(RGIId = RGI_file)
# f = pd.merge(RGI,mask_and_dem, how = "inner" )
# f.to_csv("/home/sa42/data/glac/RGI_TOPO/compiled_csv_files/" + RGI_file + ".csv")
mask_and_dem.loc[mask_and_dem["z_mask"] == 0,"z"] = np.nan

plotter(mask_and_dem)