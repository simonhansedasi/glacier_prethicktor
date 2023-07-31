import pandas as pd
import numpy as np
import glacierml as gl
from tqdm import tqdm
import tensorflow as tf
import warnings
from tensorflow.python.util import deprecation
import os
import logging
tf.get_logger().setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)
import configparser
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

tf.random.set_seed(42)
    
# for i in range(7,9,1):
    
    # load training data information
    
# exclude_list = (False, True)

# for e in exclude_list:
# exclude = False
for i in reversed(range(1,5,1)):
    parameterization = str(i)

    data = gl.parameterize_data(parameterization, 
                                pth = '/data/fast1/glacierml/data')
#                                 pth = '/home/prethicktor/data/')
    data = data.drop('RGIId', axis = 1)
    
    
    
    
    gl.build_model_ensemble(data, parameterization)

    gl.assess_model_performance(data, parameterization)

# make glacier thickness estimates
    model_statistics = pd.read_pickle('zults/model_statistics_' + parameterization + '.pkl')
    model_statistics = model_statistics[[
    'layer architecture',
    ]]
    print(model_statistics)
    gl.estimate_thickness(
        model_statistics, parameterization, useMP = False, verbose = True
    )
    
    loss_functions = ['mse','mae']
    for loss in loss_functions:
#     loss = 'mae'
        gl.compile_model_weighting_data(parameterization, model_statistics,loss = loss)

        architecture_weights, residual_model = gl.compute_model_weights(
            parameterization, loss = loss,
            #         pth = '/data/fast1/glacierml/data'
            pth = '/data/fast1/glacierml/data'
        )

        gl.calculate_RGI_thickness_statistics(
                architecture_weights, residual_model, 
                model_statistics, parameterization, loss = loss
        )




