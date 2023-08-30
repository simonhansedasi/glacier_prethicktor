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

#     data = gl.parameterize_data(parameterization, 
#                                 pth = '/data/fast1/glacierml/data')
# #                                 pth = '/home/prethicktor/data/')
#     data = data.drop('RGIId', axis = 1)
    
    
    
    
#     gl.build_model_ensemble(data, parameterization)

#     gl.assess_model_performance(data, parameterization)

# # make glacier thickness estimates
    l1_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    l2_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    arch_list = []

    for l2 in (l2_list):
        for l1 in (l1_list):
            if l1 <= l2:
                pass
            elif l1 > l2:
                arch = str(l1) + '-' + str(l2)
                arch_list.append(arch)
                
#     gl.estimate_thickness(
#         arch_list, parameterization, useMP = False, verbose = True
#     )
    
    loss_functions = ['mse','mae']
    for loss in loss_functions:

#     loss = 'first'
        gl.compile_model_weighting_data(parameterization, arch_list,loss = loss)

        architecture_weights, residual_model = gl.compute_model_weights(
            parameterization, loss = loss,
            #         pth = '/data/fast1/glacierml/data'
            pth = '/data/fast1/glacierml/data'
        )
    #     print(architecture_weights['layer architecture'].unique())
        gl.calculate_RGI_thickness_statistics(
                architecture_weights, residual_model, parameterization, loss = loss
        )




