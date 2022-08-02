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
    # LR = 0.1, 0.01, 0.001
    RS = range(0,25,1)
    
    # select either to train on all available data, or break up training by regions
    print('please select module: sm1, sm2, sm3, sm4', 'sm5, sm6, sm7')
    module_list = ('sm1', 'sm2', 'sm3', 'sm4', 'sm5', 'sm6', 'sm7')
    module = input()
    
    while module not in module_list:
        print('please select valid module: sm1, sm2, sm3, sm4, sm5, sm6, sm7')
        module = input()
    # here we can select between databases
    # sm1 = original GlaThiDa information
    # sm2 = GlaThiDa matched with RGI at global scale
    # sm3 = sm2 with area scrubber on, no area anomalies greater than 1 km**2
    # sm4 = sm3 with area scrubber set to 5 km**2
    # sm5 = global scale, no scrubber, drop zmed
    # sm6 = GlaThiDa thicknesses with RGI attributes at regional scale
    # res = variable to construct directory to save results
    
    layer_1_input, layer_2_input, lr_input,  ep_input = gl.prethicktor_inputs()
    arch = str(layer_1_input) + '-' + str(layer_2_input)

    if module == 'sm1':
        df1 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'n'
        )
        dataset = df1
        dataset.name = 'df1'
        res = 'sr1'

    if module == 'sm2':
        df2 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
        )
        dataset = df2
        dataset.name = 'df2'
        res = 'sr2'

    if module == 'sm3':
        df3 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 1
        )
        dataset = df3
        dataset.name = 'df3'
        res = 'sr3'

    if module == 'sm4':
        df4 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 5
        )
        dataset = df4
        dataset.name = 'df4'
        res = 'sr4'
        
    # replicate module 2 and drop Zmed
    if module == 'sm5':
        df5 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
        )
        df5 = df5.drop('Zmed', axis = 1)
        res = 'sr5'
        dataset = df5
        dataset.name = 'df5'
    
    if module == 'sm6':
        for region_selection in range(1,20,1):
            
            if len(str(region_selection)) == 1:
                N = 1
                region_selection = str(region_selection).zfill(N + len(str(region_selection)))
            else:
                str(region_selection) == str(region_selection)
                
            # print(region_selection)

            df6 = gl.data_loader(
                root_dir = '/home/prethicktor/data/',
                RGI_input = 'y',
                scale = 'r',
                region_selection = int(region_selection),
                area_scrubber = 'off'
            )
            if df6.empty:
                pass            
            if len(df6) >= 3:
                df6 = df6.drop('region', axis=1)
                dataset = df6
                dataset.name = str('df6_' + str(region_selection))
                res = 'sr6'

                arch = str(layer_1_input) + '-' + str(layer_2_input)
                dropout_input_list = ('y', 'n')
                for dropout_input_iter in dropout_input_list:
                    dropout_input = dropout_input_iter
                    if dropout_input == 'y':
                        dropout = True
                    elif dropout_input == 'n':
                        dropout = False

                    print(
                        'Running multi-variable DNN regression on ' + 
                        str(dataset.name) + 
                        ' dataset with parameters: Learning Rate = ' + 
                        str(lr_input) + 
                        ', Layer Architechture = ' +
                        arch +
                        ', dropout = ' + 
                        str(dropout) +
                        ', Validation split = ' + 
                        str(0.2) + 
                        ', Epochs = ' + 
                        str(ep_input) 
                    )

                    for rs in tqdm(RS):
            #             for lr in LR:

                        gl.build_and_train_model(
                            dataset, 
                            learning_rate = float(lr_input), 
                            random_state = rs, 
                            epochs = int(ep_input), 
                            module = module, 
                            res = res,
                            layer_1 = layer_1_input,
                            layer_2 = layer_2_input,
                            dropout = dropout
                        )
            
    if module == 'sm6' and region_selection == '19':
        raise SystemExit
        
    if module == 'sm7':
        df7 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
        )
        dataset = df7
        dataset.name = 'df7'
        res = 'sr7'
        
        
        
    # print(dataset.name)
    # print(dataset)    
    arch = str(layer_1_input) + '-' + str(layer_2_input)
    dropout_input_list = ('y', 'n')
    for dropout_input_iter in dropout_input_list:
        dropout_input = dropout_input_iter
        if dropout_input == 'y':
            dropout = True
        elif dropout_input == 'n':
            dropout = False
        print('')
        print(
            'Running multi-variable DNN regression on ' + 
            str(dataset.name) + 
            ' dataset with parameters: Learning Rate = ' + 
            str(lr_input) + 
            ', Layer Architechture = ' +
            arch +
            ', dropout = ' + 
            str(dropout) +
            ', Validation split = ' + 
            str(0.2) + 
            ', Epochs = ' + 
            str(ep_input) 
        )

        for rs in tqdm(RS):
#             for lr in LR:

            gl.build_and_train_model(
                dataset, 
                learning_rate = float(lr_input), 
                random_state = rs, 
                epochs = int(ep_input), 
                module = module, 
                res = res,
                layer_1 = layer_1_input,
                layer_2 = layer_2_input,
                dropout = dropout
            )    
    

if __name__ == "__main__":
    main()