import pandas as pd
import numpy as np
import glacierml as gl
from tqdm import tqdm
import tensorflow as tf
import warnings
from tensorflow.python.util import deprecation
import os
import logging
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)
import configparser

tf.random.set_seed(42)
    
# for i in range(7,9,1):
    
    # load training data information
    
# exclude_list = (False, True)

# for e in exclude_list:
# exclude = False
for i in reversed(range(1,5,1)):
    parameterization = str('4')

    data = gl.parameterize_data(parameterization, 
    #                             pth = '/data/fast1/glacierml/data')
                                pth = '/home/prethicktor/data/')

    gl.build_model_ensemble(data, parameterization, useMP = False)

    gl.assess_model_performance(data, parameterization)

# make glacier thickness estimates
    model_statistics = pd.read_csv('zults/model_statistics_' + parameterization + '.csv')

    model_statistics = model_statistics[[
    'layer architecture',
    ]]

    gl.estimate_thickness(
        model_statistics, parameterization, useMP = False, verbose = True
    )


    