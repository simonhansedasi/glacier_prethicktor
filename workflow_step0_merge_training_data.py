import pandas as pd
from tqdm import tqdm
import os as os
import numpy as np
import glacierml as gl
# import glacierml as gl
pd.set_option('mode.chained_assignment', None)
print('Matching GlaThiDa and RGI by centroid distance')



gl.match_GlaThiDa_RGI_index(
#     pth_1 = '/data/fast1/glacierml/data/T_data/',
#     pth_2 = '/data/fast1/glacierml/data/RGI/rgi60-attribs/',
#     pth_3 = '/data/fast1/glacierml/data/matched_indexes/'
    version = 'v2',
    pth = '/home/prethicktor/data/'
#     pth_3 = '/home/prethicktor/data/matched_indexes/' + version + '/'
)



# load RGI


# pth_1 = '/home/prethicktor/data/RGI/rgi60-attribs/'
# pth_2 = '/home/prethicktor/data/matched_indexes/' + version + '/'


# RGI = pd.DataFrame()
# for file in tqdm(os.listdir(pth_1)):
#     file_reader = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
#     RGI = pd.concat([RGI,file_reader], ignore_index=True)
# print(RGI)
# print('Loading merged indexes')
# # read csv of matched indexes
# indexes = pd.read_csv(pth_2 + 'GlaThiDa_RGI_matched_indexes_' + version + '.csv')
# indexes =  indexes[[
# #     '0',
#     'GlaThiDa_index',
#     'RGI_index'
# ]]
# indexes['GlaThiDa_index'] = indexes['GlaThiDa_index'].astype(int)
# indexes['RGI_index'] = indexes['RGI_index'].astype(int)

# print('Matching RGI index with RGIId')
# # match RGI indexes with RGIIds
# indexes['RGIId'] = 0
# print(indexes)
# for i in tqdm(indexes.index):
#     RGI_index = indexes['RGI_index'].loc[i]
#     indexes['RGIId'].loc[i] = RGI['RGIId'].loc[RGI_index]

# print(indexes)
# go back through RGI folder and load each file individually. 
# create a temporary dataframe to store matched indexes by region
# Search both the opened RGI file and indexes table for a matching RGIId and append
# save df with regional name in raw folder for further cleaning
# open each RGI file in turn
      
# print('Matching GlaThiDa index with RGIId')

# for i in indexes['RGIId'].unique():
#     r_id = RGI['RGIId'].loc[i]
    
#     for j in indexes['GlaThiDa_index'].unique():
#         g_id = indexes['RGIId'].loc[j]

#         match = r_id.str.contains(g_id, case = True, regex = True)
#         print(match)

# for g_index in indexes.index:
    


# for file in tqdm(os.listdir(pth_1)):
#     file_reader = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
#     df = pd.DataFrame(columns = ['region','RGI_index', 'RGIId', 'GlaThiDa_index'])
#     r_id = file_reader['RGIId']

#     # go through indexes and find any matches between GlaThiDa and RGI
#     temp_df = df.copy()
# #     temp_df = pd.DataFrame(columns = ['region', 'RGI_index', 'RGIId', 'GlaThiDa_index'])
#     for g_index in indexes.index:

#         g_id = indexes['RGIId'].iloc[g_index]
# #             print(g_index)

#         # check if RGIId series contains RGIId string from GlaThiDa. Returns boolean
#         match = r_id.str.contains(g_id, case = True, regex = True)

        # find index of True boolean
#         s = match.where(lambda x: x).dropna().index
#         print(s)
#         # append the one line that matches
#         df = file_reader.loc[s]
#         df = RGI.loc[s]
# #         print(df)
#         if not df.empty:
#             temp_df = pd.concat([temp_df,df], ignore_index = True)



#             region_1 = g_id[:8]
#             region = region_1[-2:]
# #                 temp_df = temp_df.append(s, ignore_index = True)
#             temp_df['region'].iloc[-1] = region
#             temp_df['RGI_index'].iloc[-1] = indexes['RGI_index'].loc[(g_index)]
#             temp_df['RGIId'].iloc[-1] = g_id
#             temp_df['GlaThiDa_index'].iloc[-1] = indexes['GlaThiDa_index'].loc[g_index]

#     if not temp_df.empty:
# #             print(temp_df)

#         sv_pth_raw = ('/data/fast1/glacierml/T_models/regional_data/' + version + '/raw/')
#         isdir = os.path.isdir(sv_pth_raw)
#         if isdir == False:
#             os.makedirs(sv_pth_raw)

#         temp_df.to_csv(
#             sv_pth_raw +
#             region + 
#             '_' + 
#             file[9:]
#         )




# print('Loading thicknesses to match to RGI')
# # load dataset containing thicknesses
# if version == 'v1':
#     glacier = pd.read_csv('/home/prethicktor/data/T_data/glacier.csv')
#     glacier = glacier.rename(columns = {
#         'mean_thickness':'Thickness',
#         'area':'area_g'
#     })
# if version == 'v2':
#     glacier = pd.read_csv('/home/prethicktor/data/T_data/T.csv')
#     glacier = glacier.rename(columns = {
#         'MEAN_THICKNESS':'Thickness',
#         'AREA':'area_g'
#     })
# # assemble dataframe from raw data to include regional references 

# dfs = pd.DataFrame()
# pth_3 = '/home/prethicktor/data/regional_data/' + version + '/raw/'
# for file in tqdm(os.listdir(pth_3)):
#     df = pd.read_csv(pth_3 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
#     region_and_number = file[:-4]
#     region_number = region_and_number[:2]
#     region = region_and_number[3:]

#     df['geographic region'] = region
#     df['region'] = region_number
#     dfs = pd.concat([dfs,df], ignore_index=True)


# dfs = dfs.reset_index()

# dfs = dfs[[
#     'GlaThiDa_index',
#     'RGI_index',
#     'RGIId',
#     'region',
#     'geographic region'
# ]]

# print('GlaThiDa indexes matched with RGI indexes')
# # load RGI data and compare to what is available
# RGI_extra = pd.DataFrame()
# for file in os.listdir(pth_1):
#     f = pd.read_csv(pth_1 + file, encoding_errors = 'replace', on_bad_lines = 'skip')
#     RGI_extra = pd.concat([RGI_extra, f], ignore_index = True)


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

# print('Adding GlaThiDa thickness measurements to RGI')
# print(dfs)
# # create temp df to house regional data, merge it with RGI attributes and save as training data
# for region in dfs['region'].unique():
#     df = dfs[dfs['region'] == region]

#     df['Thickness'] = np.nan
#     df['area_g'] = np.nan
#     for df_idx in df.index:
#         g_idx = df['GlaThiDa_index'].loc[df_idx]
#         thickness = glacier['Thickness'].loc[g_idx]
#         area = glacier['area_g'].loc[g_idx]
#         df['Thickness'].loc[df_idx] = thickness
#         df['area_g'].loc[df_idx] = area
#     df = pd.merge(df, RGI_extra, on = 'RGIId')

#     df = df.dropna(subset = ['Thickness'])

#     sv_pth_td = ('/home/prethicktor/data/regional_data/training_data/' + version + '/')
#     isdir = os.path.isdir(sv_pth_td)
#     if isdir == False:
#         os.makedirs(sv_pth_td)
#     df.to_csv(
#         sv_pth_td + 
#         df['region'].loc[0] + 
#         '_' + 
#         df['geographic region'].loc[0] 
#         + '.csv'
#     )