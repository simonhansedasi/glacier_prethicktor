import pandas as pd
import numpy as np
import glacierml as gl
from tqdm import tqdm
import warnings
import os
import logging
# tf.get_logger().setLevel(logging.WARNING)
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)
import configparser
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# tf.random.set_seed(42)
for i in reversed(range(1,5,1)):
    parameterization = str(i)
# parameterization = '4'
    for loss in ['mae','mse']:

        df = gl.parameterize_data(parameterization)
        df = df.drop('RGIId', axis = 1)
#         # df = df.drop(['RGIId','CenLat','CenLon','Zmed','Zmax','Aspect'],axis = 1)
#         df = df.drop(df[df['Thickness'] >= 300].index)
#         df = df.reset_index()
#         df = df.drop('index',axis = 1)
#         # create a copy of df to draw K test sets to be kept in a vault
#         df_sampler = df.copy()
#         df_trainer = df.copy()

#         rs = 42

#         df1test = df_sampler.sample(frac = 0.333333,random_state = rs)
#         df_sampler = df_sampler.drop(df1test.index)

#         df2test = df_sampler.sample(frac = 0.5,random_state = rs)
#         df_sampler = df_sampler.drop(df2test.index)

#         df3test = df_sampler

#         df1 = df_trainer.drop(df1test.index)
#         df2 = df_trainer.drop(df2test.index)
#         df3 = df_trainer.drop(df3test.index)

#         df_list = [df1,df2,df3]
#         test_list = [df1test,df2test,df3test]
#         k_list = ['1','2','3']

#         for k,data,test_data in zip(k_list, df_list,test_list):


        gl.build_model_ensemble(df, parameterization,loss)

        gl.assess_model_performance(df, loss,parameterization )

        # # make glacier thickness estimates
#             l1_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#             l2_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

#             arch_list = []

#             for l2 in (l2_list):
#                 for l1 in (l1_list):
#                     if l1 <= l2:
#                         pass
#                     elif l1 > l2:
#                         arch = str(l1) + '-' + str(l2)
#                         arch_list.append(arch)
    #     parameterization = '4'
#     rootdir = 'temp/'
#     stats1 = pd.DataFrame()
#     stats2 = pd.DataFrame()
#     losses = ['mae','mse']

#     for k in range(1,4,1):
#         k = str(k)
#         for loss in losses:
#             stats = pd.read_pickle(
#                 rootdir + 'model_statistics_xval_' + k + '_' + 
#                 loss + '_' + parameterization + '.pkl'
#             )
#             stats['fold'] = k
#             if loss == 'mae':
#                 stats1 = pd.concat([stats1,stats])
#             if loss == 'mse':
#                 stats2 = pd.concat([stats2,stats])
#     stats1['parameter ratio'] = stats1['parameters'] / stats1['inputs']
#     stats2['parameter ratio'] = stats2['parameters'] / stats2['inputs']
#     stats = pd.merge(
#         stats1,stats2,how = 'inner',
#         on = ['layer architecture','parameter ratio','parameters','fold']
#     )

#     stats = stats.sort_values('parameter ratio')


#     k1 = stats[stats['fold'] == '1']
#     k2 = stats[stats['fold'] == '2']
#     k3 = stats[stats['fold'] == '3']
#     k_list = [k1,k2,k3]
#     n_list = ['1','2','3']

#     arch_list = pd.DataFrame()

#     for n, k in zip(n_list, k_list):



#         x1 = k['parameter ratio']
#         y1 = k['loss avg_x']
#         model1 = np.poly1d(np.polyfit(x1, y1, 2))

#         x2 = k['parameter ratio']
#         y2 = np.sqrt(k['loss avg_y'])
#         model2 = np.poly1d(np.polyfit(x2, y2, 2))

#         sts = k[
#             (k['loss avg_x'] <= model1(x1)) &
#     #         (k['loss avg_x'] <= model2(x2)) &
#             (np.sqrt(k['loss avg_y']) <= model2(x2)) 
#     #         (np.sqrt(k['loss avg_y']) <= model1(x1))
#         ]
#         std = k.drop(sts.index)

#         if n == '1':
#             arch_list_1 = sts
#         if n == '2':
#             arch_list_2 = sts     
#         if n == '3':
#             arch_list_3 = sts

#     arch_list = pd.merge(
#         arch_list_1['layer architecture'], 
#         arch_list_2['layer architecture'],
#         how = 'inner', on = 'layer architecture'
#     )
#     arch_list = pd.merge(
#         arch_list, 
#         arch_list_2['layer architecture'],
#         how = 'inner', on = 'layer architecture'
#     )
#     arch_list = arch_list.sort_values('layer architecture')

#     arch_list = arch_list.values.flatten()
    # print(arch_list.values.flatten())
    
    
    rootdir = 'temp/'
    stats1 = pd.DataFrame()
    stats2 = pd.DataFrame()
    
    for loss in ['mae','mse']:
        stats = pd.read_pickle(
            rootdir + 'model_statistics_' + 
            loss + '_' + parameterization + '.pkl'
        )
#         stats['fold'] = k
        if loss == 'mae':
            stats1 = pd.concat([stats1,stats])
        if loss == 'mse':
            stats2 = pd.concat([stats2,stats])
    stats1['parameter ratio'] = stats1['parameters'] / stats1['inputs']
    stats2['parameter ratio'] = stats2['parameters'] / stats2['inputs']
    stats = pd.merge(
        stats1,stats2,how = 'inner',
        on = ['layer architecture','parameter ratio','parameters']
    )

    stats = stats.sort_values('parameter ratio')
    
    x1 = stats['parameter ratio']
    y1 = stats['loss avg_x']
    model1 = np.poly1d(np.polyfit(x1, y1, 2))

    x2 = stats['parameter ratio']
    y2 = np.sqrt(stats['loss avg_y'])
    model2 = np.poly1d(np.polyfit(x2, y2, 2))

    sts = stats[
        (stats['loss avg_x'] <= model1(x1)) &
        (k['loss avg_x'] <= model2(x2)) &
        (np.sqrt(stats['loss avg_y']) <= model2(x2)) 
        (np.sqrt(k['loss avg_y']) <= model1(x1))
    ]
#     std = k.drop(sts.index)
    arch_list = sts['layer architecture'].values.flatten()
    
    
    
    
    gl.estimate_thickness(
        arch_list,parameterization, useMP = False, verbose = True,xval = False
    )

    # loss_functions = ['mse','mae']
    # for loss in loss_functions:

    #     loss = 'first'
    gl.compile_model_weighting_data(parameterization, arch_list,xval = False)

    architecture_weights, residual_model = gl.compute_model_weights(
        parameterization,
        #         pth = '/data/fast1/glacierml/data'
        pth = '/data/fast1/glacierml/data',
        xval = False
    )
    #     print(architecture_weights['layer architecture'].unique())
    gl.calculate_RGI_thickness_statistics(
            architecture_weights, residual_model, parameterization,xval = False
    )




