import pandas as pd
from tqdm import tqdm
import os as os
import numpy as np
# import glacierml as gl
pd.set_option('mode.chained_assignment', None)



# load RGI
pth_1 = '/data/fast1/glacierml/T_models/RGI/rgi60-attribs/'
pth_2 = '/data/fast1/glacierml/T_models/matched_indexes/'

RGI_extra = pd.DataFrame()
for file in tqdm(os.listdir(pth_1)):
    file_reader = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(file_reader, ignore_index=True)

    
# read csv of matched indexes
indexes = pd.read_csv(pth_2 + 'GlaThiDa_RGI_matched_indexes.csv')
indexes =  indexes[[
#     '0',
    'GlaThiDa_index',
    'RGI_index'
]]
#     indexes = indexes.rename(columns = {
#         '0':'distance'
#     })
#     indexes = indexes.drop_duplicates(subset = 'GlaThiDa_index', keep = 'last')

indexes['GlaThiDa_index'] = indexes['GlaThiDa_index'].astype(int)
indexes['RGI_index'] = indexes['RGI_index'].astype(int)
# match RGI indexes with RGIIds
indexes['RGIId'] = 0
for i in tqdm(indexes.index):
    RGI_index = indexes['RGI_index'].loc[i]
    for ii in RGI_extra.index:
        if i == ii:
            indexes['RGIId'].loc[i] = RGI_extra['RGIId'].loc[RGI_index]


# go back through RGI folder and load each file individually. 
# create a temporary dataframe to store matched indexes by region
# Search both the opened RGI file and indexes table for a matching RGIId and append
# save df with regional name in raw folder for further cleaning
# open each RGI file in turn
for file in tqdm(os.listdir(pth_1)):
    file_reader = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
    df = pd.DataFrame(columns = ['region','RGI_index', 'RGIId', 'GlaThiDa_index'])
    r_id = file_reader['RGIId']

    # go through indexes and find any matches between GlaThiDa and RGI
    temp_df = pd.DataFrame(columns = ['region', 'RGI_index', 'RGIId', 'GlaThiDa_index'])
    for g_index in indexes.index:

        g_id = indexes['RGIId'].iloc[g_index]
#             print(g_index)

        # check if RGIId series contains RGIId string from GlaThiDa. Returns boolean
        match = r_id.str.contains(g_id, case = True, regex = True)

        # find index of True boolean
        s = match.where(lambda x: x).dropna().index

        # append the one line that matches
        df = file_reader.loc[s]
        if not df.empty:
            temp_df = temp_df.append(df)



            region_1 = g_id[:8]
            region = region_1[-2:]
#                 temp_df = temp_df.append(s, ignore_index = True)
            temp_df['region'].iloc[-1] = region
            temp_df['RGI_index'].iloc[-1] = indexes['RGI_index'].loc[(g_index)]
            temp_df['RGIId'].iloc[-1] = g_id
            temp_df['GlaThiDa_index'].iloc[-1] = indexes['GlaThiDa_index'].loc[g_index]

    if not temp_df.empty:
#             print(temp_df)

        sv_pth_raw = ('/data/fast1/glacierml/T_models/regional_data/raw/')
        isdir = os.path.isdir(sv_pth_raw)
        if isdir == False:
            os.makedirs(sv_pth_raw)

        temp_df.to_csv(
            sv_pth_raw +
            region + 
            '_' + 
            file[9:]
        )





# load dataset containing thicknesses
glacier = pd.read_csv('/data/fast1/glacierml/T_models/T_data/glacier.csv')
glacier = glacier.rename(columns = {
    'mean_thickness':'Thickness',
    'area':'area_g'
})


# assemble dataframe from raw data to include regional references 

dfs = pd.DataFrame()
pth_3 = '/data/fast1/glacierml/T_models/regional_data/raw/'
for file in tqdm(os.listdir(pth_3)):
    df = pd.read_csv(pth_3 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
    region_and_number = file[:-4]
    region_number = region_and_number[:2]
    region = region_and_number[3:]

    df['geographic region'] = region
    df['region'] = region_number
    dfs = dfs.append(df, ignore_index=True)


dfs = dfs.reset_index()

dfs = dfs[[
    'GlaThiDa_index',
    'RGI_index',
    'RGIId',
    'region',
    'geographic region'
]]

# load RGI data and compare to what is available
RGI_extra = pd.DataFrame()
for file in os.listdir(pth_1):
    f = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
    RGI_extra = RGI_extra.append(f, ignore_index = True)


#     region_and_number = file[:-4]
#     region_number = region_and_number[:2]
#     region = region_and_number[9:]
#     df = dfs[dfs['region'] == region_number]

#     percent_trainable = (len(df) / len(f)) * 100

#     print(
#         'region ' + str(region_number) + ' has ' + str(len(f)) + ' lines of data, ' +
#         str(len(df)) + ' or ' + str(percent_trainable) + '%'
#         ' of which are trainable with GlaThiDa thicknesses'    
#     )


# create temp df to house regional data, merge it with RGI attributes and save as training data
for region in dfs['region'].unique():
    df = dfs[dfs['region'] == region]

    df['Thickness'] = np.nan
    df['area_g'] = np.nan
    for df_idx in df.index:
        g_idx = df['GlaThiDa_index'].loc[df_idx]
        thickness = glacier['Thickness'].loc[g_idx]
        area = glacier['area_g'].loc[g_idx]
        df['Thickness'].loc[df_idx] = thickness
        df['area_g'].loc[df_idx] = area
    df = pd.merge(df, RGI_extra, on = 'RGIId')

    df = df.dropna(subset = ['Thickness'])

    sv_pth_td = ('/data/fast1/glacierml/T_models/regional_data/training_data/')
    isdir = os.path.isdir(sv_pth_td)
    if isdir == False:
        os.makedirs(sv_pth_td)
    df.to_csv(
        sv_pth_td + 
        df['region'].loc[0] + 
        '_' + 
        df['geographic region'].loc[0] 
        + '.csv'
    )