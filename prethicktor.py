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
    #load and organize data
    glacier = gl.data_loader(pth = '/home/prethicktor/data/')
    gl.thickness_renamer(glacier)


    glacier.name = 'glacier'
#     LR = np.logspace(-3,2,6)
    VS = 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4
#     RS = range(0,25,1)

    for vs in VS:
        gl.thickness_renamer(glacier)
        gl.build_and_train_model(glacier,validation_split = vs)

        
if __name__ == "__main__":
    main()