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
    
    LR = 0.1, 0.01, 0.001
#     VS = 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
    RS = range(0,25,1)
    
    print('please select module: sm, sm2, sm4, sm5, sm6, sm7')
     
    module = input()
    
    if module == 'sm':
        glacier = gl.data_loader(pth = '/home/prethicktor/data/')
        gl.thickness_renamer(glacier)
        dataset = glacier
        dataset.name = 'glacier'
        res = 'sr'
    if module == 'sm2':
        Glam = gl.data_loader_2(pth = '/home/prethicktor/data/')
        gl.thickness_renamer(Glam)
        dataset = Glam
        dataset.name = 'Glam'
        res = 'sr2'
    if module == 'sm4':
        Glam_phys = Glam[[
            'Area',
            'thickness',
            'Slope',
            'Zmin',
            'Zmed',
            'Zmax',
            'Aspect',
            'Lmax'
        ]]
        dataset = Glam_phys
        dataset.name = 'Glam_phys'
        res = 'sr4'
    if module == 'sm5':
        Glam_2 = gl.data_loader_3(pth = '/home/prethicktor/data/')
        gl.thickness_renamer(Glam_2)
        dataset = Glam_2
        dataset.name = 'Glam_2'
        res = 'sr5'
        
#     if module == 'sm6':
        # regional_data_1
        
    if module == 'sm7':
        regional_data = gl.data_loader_4()
        reg = regional_data['region'].iloc[-1]
        regional_data = regional_data.drop('region', axis=1)
        dataset = regional_data
        dataset.name = str('regional_data_' + str(reg))
        res = 'sr7'
        
    for rs in RS:
        for lr in LR:
            gl.build_and_train_model(
                dataset, 
                learning_rate = lr, 
                random_state = rs, 
                epochs = 300, 
                module = module, 
                res = res
            )

if __name__ == "__main__":
    main()