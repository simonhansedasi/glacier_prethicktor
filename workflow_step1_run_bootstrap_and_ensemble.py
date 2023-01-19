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

tf.random.set_seed(42)

def main():
    
    
    parameterization, dataset, dataset.name, res = gl.select_dataset_coregistration(
                                                        parameterization = 'sm2'
                                                    )
    
    
    RS = range(0,25,1)
    print('')
    print(dataset.name)
    print(dataset)  
#     print(len(dataset))

#     ep_input = '2000'
    layer_1_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    layer_2_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
#     lr_input = '0.01'
    print('Building models...')
    
    
    for layer_2_input in (layer_2_list):
        for layer_1_input in (layer_1_list):
            if layer_1_input <= layer_2_input:
                pass
            elif layer_1_input > layer_2_input:
                
                arch = str(layer_1_input) + '-' + str(layer_2_input)
                dropout = True
                print('')
                print(
                    'Running multi-variable DNN regression on ' + 
                    str(dataset.name) + 
                    ' dataset with layer architecture = ' +
                    arch
                    )

                for rs in tqdm(RS):
            #             for lr in LR:

                    gl.build_and_train_model(
                        dataset, 
                        random_state = rs, 
                        parameterization = parameterization, 
                        res = res,
                        layer_1 = layer_1_input,
                        layer_2 = layer_2_input,
                    )   




if __name__ == "__main__":
    main()