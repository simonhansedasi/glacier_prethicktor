from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
# import glacierml as gl
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from sklearn.cluster import KMeans
# import seaborn as sns
# pd.set_option('display.max_columns', None)



pth_2 = '/home/simonhans/data/prethicktor/RGI/rgi60-attribs/'
# pth_2 = '/data/fast1/glacierml/data/RGI/rgi60-attribs/'
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

pth_1 = '/home/simonhans/data/prethicktor/RGI/results_model_4/'



region_list = (7, 8, 11, 13, 14, 15, 18)
for region_number in region_list:
            
    if len(str(region_number)) == 1:
        N = 1
        region_number = str(region_number).zfill(N + len(str(region_number)))
    else:
        str(region_number) == str(region_number)

    region_folder = pth_1 + 'RGI60-' + str(region_number) + '/'
    for file in tqdm(os.listdir(region_folder)):
        im = Image.open(region_folder + file)
#         im.show()
        imarray = np.array(im)
        df = pd.DataFrame(imarray)
        df = df.replace(-9999, np.nan)
        df = df.replace(0.0, np.nan)
#         print(df)
        mean_glacier_thickness = df.mean().mean()
#         print(mean_glacier_thickness)
        RGI['Farinotti Mean Thickness'].loc[RGI['RGIId'] == file[:14]] = mean_glacier_thickness
RGI.to_csv('results_4_mean_thicknesses.csv')