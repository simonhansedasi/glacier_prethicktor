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



def main():
    glacier = gl.data_loader(pth = '/home/prethicktor/data/')
    Glam = gl.data_loader_2(pth = '/home/prethicktor/data/')
    Glam_2 = gl.data_loader_3(pth = '/home/prethicktor/data/')
    gl.thickness_renamer(glacier)
    gl.thickness_renamer(Glam)
    gl.thickness_renamer(Glam_2)
    
    Glam_phys = Glam[[
#         'CenLon',
#         'CenLat',
        'Area',
        'thickness',
        'Slope',
        'Zmin',
        'Zmed',
        'Zmax',
        'Aspect',
        'Lmax'
    ]]
    
    Glam.name = 'Glam'
    Glam_2.name = 'Glam_2'
    Glam_phys.name = 'Glam_phys'
    glacier.name = 'glacier'
    LR = 0.1, 0.01, 0.001
#     VS = 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
    RS = range(0,25,1)

    for rs in RS:
        for lr in LR:
            gl.build_and_train_model(
                Glam, learning_rate = lr, random_state = rs, epochs = 300
            )
        
        
# def main():
#     #load and organize data
#     
#     gl.thickness_renamer(glacier)


#     glacier.name = 'glacier'
# #     LR = np.logspace(-3,2,6)
# #     VS = 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
#     RS = range(0,25,1)
#     LR = 0.1, 0.01, 0.001
    
#     for rs in RS:
#         for lr in LR:
#             gl.thickness_renamer(glacier)
#             gl.build_and_train_model(glacier, learning_rate = lr, random_state = rs)

        
if __name__ == "__main__":
    main()