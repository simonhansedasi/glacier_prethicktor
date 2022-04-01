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
    T,TT,TTT,TTTx,TTT_full = gl.data_loader()
    gl.thickness_renamer(T)
    gl.thickness_renamer(TT)
    T_t = T.head()
    glathida_list = T,TT,TTTx,TTT_full

    T.name = 'T'
    TT.name = 'TT'
    TTT.name = 'TTT'
    TTTx.name = 'TTTx'
    T_t.name = 'T_t'
    TTT_full.name = 'TTT_full'
    for i in glathida_list:
        
        gl.build_and_train_model(i)

if __name__ == "__main__":
    main()