import pandas as pd
from tqdm import tqdm
import os as os
# import glacierml as gl
pd.set_option('mode.chained_assignment', None)

# load RGI
rootdir = '/data/fast1/glacierml/T_models/attribs/rgi60-attribs/'
RGI_extra = pd.DataFrame()
for file in tqdm(os.listdir(rootdir)):
    f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(f, ignore_index=True)

# read csv of matched indexes
indexes = pd.read_csv('GlaThiDa_RGI_matched_indexes.csv')
indexes =  indexes[[
    '0',
    'GlaThiDa_index',
    'RGI_index'
]]
indexes = indexes.rename(columns = {
    '0':'distance'
})
indexes = indexes.drop_duplicates(subset = 'GlaThiDa_index', keep = 'last')



# match RGI indexes with RGIIds
indexes['RGIId'] = 0
for i in tqdm(indexes.index):
    RGI_index = indexes['RGI_index'].loc[i]
    for ii in RGI_extra.index:
        if i == ii:
            indexes['RGIId'].loc[i] = RGI_extra['RGIId'].loc[RGI_index]

            
            
            
            
            
            
            
            
# go back through RGI folder and load each file individually. 
# create a temporary dataframe to store matched indexes by region
# Search both the opened RGI file and indexes table for a matching RGIId. When found, load data to df
# save df with regional name in raw folder for further cleaning
for file in tqdm(os.listdir(rootdir)):
    f = pd.read_csv(rootdir+file, encoding_errors = 'replace', on_bad_lines = 'skip')
    df = pd.DataFrame(columns = ['region','RGI_index', 'RGIId', 'GlaThiDa_index'])
    for RGI_index in f.index:
        r_id = f['RGIId'].loc[RGI_index]
        
        for g_index in indexes.index:
            g_id = indexes['RGIId'].loc[g_index]
            
            if r_id == g_id:
                print(r_id)
                print(g_id)
                region_1 = r_id[:8]
                region = region_1[-2:]
                s = pd.Series(region)
                df = df.append(s, ignore_index = True)
                df['region'].iloc[-1] = region
                df['RGI_index'].iloc[-1] = indexes['RGI_index'].loc[(g_index)]
                df['RGIId'].iloc[-1] = str(r_id)
                df['GlaThiDa_index'].iloc[-1] = indexes['GlaThiDa_index'].loc[g_index]
                
                df.to_csv(
                    '/data/fast1/glacierml/T_models/regional_data_1/raw/'+ region + '_' + file[9:])