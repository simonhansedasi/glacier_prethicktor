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
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
pd.set_option('mode.chained_assignment', None)

tf.random.set_seed(42)

def main():
    # define range for learning rate and random state. 
    # learning rate no longer varied 
    # LR = 0.1, 0.01, 0.001
    RS = range(0,25,1)
    
    # select either to train on all available data, or break up training by regions
    
    
    print('please select data registration method: sm1, sm2, sm3, sm4, sm5, sm6, sm7')
#     module_list = ('sm1', 'sm2', 'sm3', 'sm4', 'sm5','sm6','sm7')
    module = input()
#     for module_item in module_list:
#         module = module_item
#     while module not in module_list:
#         print('please select valid module: sm1, sm2, sm3, sm4, sm5, sm6, sm7', 'sm8')
#         module = input()

#     layer_1_input, layer_2_input, lr_input,  ep_input = gl.prethicktor_inputs()


    if module == 'sm1':
        df1 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'n'
        )
        dataset = df1
        dataset.name = 'df1'
        res = 'sr1'
        layer_1_list = ['10','16', '24']
        layer_2_list = ['5', '8',  '12']

    if module == 'sm2':
        df2 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
        )
        df2 = df2.drop(['RGIId', 'region'], axis = 1)
        dataset = df2
        dataset.name = 'df2'
        res = 'sr2'
        layer_1_list = ['10','50', '64']
        layer_2_list = ['5', '28', '48']

    if module == 'sm3':
        df3 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 25
        )
        df3 = df3.drop(['RGIId', 'region'], axis = 1)
        dataset = df3
        dataset.name = 'df3'
        res = 'sr3'
        layer_1_list = ['10', '32', '45']
        layer_2_list = ['5',  '17', '28']

    if module == 'sm4':
        df4 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 75
        )
        df4 = df4.drop(['RGIId', 'region'], axis = 1)
        dataset = df4
        dataset.name = 'df4'
        res = 'sr4'
        layer_1_list = ['10', '47', '64']
        layer_2_list = ['5',  '21', '36']

    if module == 'sm5':
        df5 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
        )
        df5 = df5.drop(['RGIId', 'region'], axis = 1)
        df5['Zdelta'] = df5['Zmax'] - df5['Zmin']
        res = 'sr5'
        dataset = df5
        dataset.name = 'df5'
        layer_1_list = ['10','48', '64']
        layer_2_list = ['5', '32', '52']


    if module == 'sm6':
        df6 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 25
        )
        df6 = df6.drop(['RGIId', 'region'], axis = 1)
        df6['Zdelta'] = df6['Zmax'] - df6['Zmin']
        dataset = df6
        dataset.name = 'df6'
        res = 'sr6'
        layer_1_list = ['10', '32', '48']
        layer_2_list = ['5',  '18', '28']

    if module == 'sm7':
        df7 = gl.data_loader(
            root_dir = '/home/prethicktor/data/',
            RGI_input = 'y',
            scale = 'g',
            area_scrubber = 'on',
            anomaly_input = 75
        )
        df7 = df7.drop(['RGIId', 'region'], axis = 1)
        df7['Zdelta'] = df7['Zmax'] - df7['Zmin']
        dataset = df7
        dataset.name = 'df7'
        res = 'sr7'
        layer_1_list = ['10', '42', '64']
        layer_2_list = ['5',  '26', '40']


    print(dataset.name)
    print(dataset)  
    print(len(dataset))


#         arch = str(layer_1_input) + '-' + str(layer_2_input)
#                 print(arch)
#                 print(arch.type())
#             arch = str(layer_1_input) + '-' + str(layer_2_input)
#     lr_list = ('0.1', '0.01', '0.001')
    ep_input = '2000'
    layer_1_list = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    layer_2_list = [2,4,5,6,7,8,9,10,11,12,13,14,15]
    lr_input = '0.01'
#     dropout_input_list = ('y', 'n')
#     for lr in lr_list:
#         lr_input = lr
#         for layer_1, layer_2 in zip(layer_1_list, layer_2_list):
#             layer_1_input = layer_1
#             layer_2_input = layer_2
    for layer_2_input in layer_2_list:
        for layer_1_input in layer_1_list:
            if layer_1_input <= layer_2_input:
                pass
            elif layer_1_input > layer_2_input:
                arch = str(layer_1_input) + '-' + str(layer_2_input)
            #         for dropout_input_iter in dropout_input_list:
            #             dropout_input = dropout_input_iter
            #             if dropout_input == 'y':
            #                 dropout = True
            #             elif dropout_input == 'n':
            #                 dropout = False
                dropout = True
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