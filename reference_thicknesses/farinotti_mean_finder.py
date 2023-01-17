from PIL import Image
import numpy as np
import pandas as pd
import os
from scipy.stats import shapiro
from scipy.stats import skew

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from tqdm import tqdm
# import glacierml as gl
def fxnUw():
    warnings.warn("UserWarning arose", UserWarning)

## For DeprecationWarning
def fxnDw():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxnUw()
    pth_2 = '/home/simonhans/data/glacierml/RGI/rgi60-attribs/'
    RGI_extra = pd.DataFrame(columns = ['Farinotti Mean Thickness','Shapiro-Wilk statistic', 
                                        'Shapiro-Wilk p_value', 'Farinotti Median Thickness', 'Farinotti Thickness STD',
                                        'Skew'])
    for file in tqdm(os.listdir(pth_2)):
        file_reader = pd.read_csv(pth_2 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
        RGI_extra = pd.concat([RGI_extra, file_reader], ignore_index = True)
        RGI_extra['region'] = RGI_extra['RGIId'].str[6:8]
        # select only RGI data that was used to train the model   
#     RGI = RGI_extra[[
#         'RGIId',
#         'Farinotti Mean Thickness',
#         'Shapiro-Wilk statistic', 'Shapiro-Wilk p_value',
#         'region', 'Farinotti Median Thickness', 'Farinotti Thickness STD'

#     ]]
#     print(RGI_extra)
    # print(RGI['RGIId'])

    pth_1 = '/home/simonhans/data/glacierml/RGI/outlines/'
    # rootdir = '~'
    for region_number in range(2,20,1):
        
        if len(str(region_number)) == 1:
            N = 1
            region_number = str(region_number).zfill(N + len(str(region_number)))
        else:
            str(region_number) == str(region_number)
        print(region_number)
        RGI = RGI_extra[RGI_extra['region'] == str(region_number)]
        region_folder = pth_1 + 'RGI60-' + str(region_number) + '/'
        
        
        dft = pd.DataFrame()
        for file in tqdm(os.listdir(region_folder)):
#             print(file)
#             print(file[:14])
            im = Image.open(region_folder + file)
            imarray = np.array(im)
            df = pd.DataFrame(imarray)
            df = df.replace(0.0, np.nan)
            median_glacier_thickness = df.median().median()
            mean_glacier_thickness = df.mean().mean()
            RGI['Farinotti Mean Thickness'].loc[RGI['RGIId'] == file[:14]] = mean_glacier_thickness
            
            glacier_std = df.std().std()
            
            shap_test = df.to_numpy()
    
            shap = shap_test[np.logical_not(np.isnan(shap_test))] #removing null values
            dft = pd.DataFrame()
            for i in list(df):
                dft = pd.concat([dft, df[i]])
                dft = dft.dropna()

            dft = dft.to_numpy()
            dfty = dft[np.logical_not(np.isnan(dft))]
            glacier_skew = skew(dfty)
#             if len(glacier_skew) != 1:
                
#                 print(len(glacier_skew))
            if len(shap) > 3:
                
                statistic, p_value = shapiro(shap)    
                RGI['Shapiro-Wilk statistic'].loc[RGI['RGIId'] == file[:14]] = statistic
                RGI['Shapiro-Wilk p_value'].loc[RGI['RGIId'] == file[:14]] = p_value
            RGI['Skew'].loc[RGI['RGIId'] == file[:14]] = glacier_skew

            RGI['Farinotti Median Thickness'].loc[RGI['RGIId'] == file[:14]] = median_glacier_thickness
            RGI['Farinotti Thickness STD'].loc[RGI['RGIId'] == file[:14]] = glacier_std
            
            dfrs = pd.DataFrame(shap)
            dfrs.to_csv(str(region_number) + '/' + str(file[:14]) + '_thickness_measurements.csv')
        RGI.to_csv('Farinotti_mean_thickness_RGI_ID' + str(region_number) + '.csv')