# import pandas as pd
# import numpy as np
import glacierml as gl
# from tqdm import tqdm
# import tensorflow as tf
# import warnings
# from tensorflow.python.util import deprecation
# import os
# import logging
# tf.get_logger().setLevel(logging.ERROR)
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)
# deprecation._PRINT_DEPRECATION_WARNINGS = False
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# pd.set_option('mode.chained_assignment', None)
# import configparser

tf.random.set_seed(42)
    
for i in range(1,4,1):
    
    # load training data information
    parameterization = str(i)
    
    gl.parameterize_data(parameterization)
    
    gl.build_model_ensemble(data, parameterization, useMP = False)

    gl.assess_model_performance(parameterization)

    # make glacier thickness estimates
    model_statistics = pd.read_csv('zults/model_statistics_' + parameterization + '.csv')

    model_statistics = model_statistics[[
    'layer architecture',
    ]]

    gl.estimate_thickness(
        model_statistics, parameterization, useMP = False, verbose = True
    )

    gl.calculate_RGI_thickness_statistics(
        model_statistics, parameterization
    )

