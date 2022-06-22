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


def main():
    # define range for learning rate and random state. 
    # learning rate no longer varied 
#     LR = 0.1, 0.01, 0.001
    RS = range(0,25,1)
    
    # select either to train on all available data, or break up training by regions
    print('select data type: global / regional')
    
    data_type = input()
    if data_type == 'global':
        print('please select module: sm1, sm2, sm3, sm4')
        module = input()
        
        # here we can select between databases
        # sm1 = original GlaThiDa information
        # sm2 = GlaThiDa matched with RGI using technique 1 defined in glacierml.py
        # sm3 = sm2 w/o lat and lon
        # sm4 = GlaThiDa matched with RGI using technique 2 defined in glacierml.py
        # res = variable to construct directory to save results
        if module == 'sm1':
            df1 = gl.data_loader(pth = '/home/prethicktor/data/')
            gl.thickness_renamer(df1)
            dataset = df1
            dataset.name = 'df1'
            res = 'sr1'
            print(module)
            print(dataset)

        if module == 'sm2':
            df2 = gl.data_loader_2(pth = '/home/prethicktor/data/')
            gl.thickness_renamer(df2)
            dataset = df2
            dataset.name = 'df2'
            res = 'sr2'
            print(module)
            print(dataset)

        if module == 'sm3':
            df2 = gl.data_loader_2(pth = '/home/prethicktor/data/')
            gl.thickness_renamer(df2)
            df3 = df2[[
                'Area',
                'thickness',
                'Slope',
                'Zmin',
                'Zmed',
                'Zmax',
                'Aspect',
                'Lmax'
            ]]
            dataset = df3
            dataset.name = 'df3'
            res = 'sr3'
            print(module)
            print(dataset)

        if module == 'sm4':
            df4 = gl.data_loader_4(pth = '/home/prethicktor/data/')
            gl.thickness_renamer(df4)
            dataset = df4
            dataset.name = 'df4'
            res = 'sr4'
            print(module)
            print(dataset)
            
    # here we can select between two regional datasets. 
    # sm5 uses matching technique 1, same to build sm2
    # sm6 uses matching technique 2, same to build sm4
    if data_type == 'regional':
        print('please select module: sm5, sm6')
        module = input()
        if module == 'sm5':
            df5 = gl.data_loader_5(pth = '/home/prethicktor/data/regional_data_1/training_data/')
            reg = df5['region'].iloc[-1]
            df5 = df5.drop('region', axis=1)
            dataset = df5
            dataset.name = str('df5_' + str(reg))
            res = 'sr5'
            print(module)
            print(dataset)            
            

        elif module == 'sm6':
            df6 = gl.data_loader_6(pth = '/home/prethicktor/data/regional_data_2/training_data/')
            reg = df6['region'].iloc[-1]
            df6 = df6.drop('region', axis=1)
            dataset = df6
            dataset.name = str('df6_' + str(reg))
            res = 'sr6'
            print(module)
            print(dataset)
            
    # define hyperparameters
    layer_1_input, layer_2_input, lr_input, ep_input = gl.prethicktor_inputs()
    
    # run the model
    for rs in RS:
#             for lr in LR:
        gl.build_and_train_model(
            dataset, 
            learning_rate = float(lr_input), 
            random_state = rs, 
            epochs = int(ep_input), 
            module = module, 
            res = res,
            layer_1 = layer_1_input,
            layer_2 = layer_2_input
        )

if __name__ == "__main__":
    main()