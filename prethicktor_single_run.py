import pandas as pd
import numpy as np
import glacierml_old as gl
from tqdm import tqdm
import tensorflow as tf
import warnings
from tensorflow.python.util import deprecation
import os
import logging
from time import perf_counter
from multiprocessing import Pool
from functools import partial

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

                         
def main():
    #load and organize data
    T,TT,TTT = gl.data_loader(pth = '/home/prethicktor/data/')
    gl.thickness_renamer(T)
    gl.thickness_renamer(TT)
    T_t = T.head()
#     glathida_list = T,TT

    T.name = 'glacier'
    TT.name = 'band'
    TTT.name = 'point'
#     TTTx.name = 'TTTx'
    T_t.name = 'T_t'
#     TTT_full.name = 'TTT_full'
    LR = np.logspace(-3,2,6)
    VS = 0.1,0.15,0.2,0.25,0.3,0.35,0.4
    RS = range(0,25,1)
# #     for dataset in glathida_list:
#     wrapper = lambda x: gl.build_and_train_model(T,
#                              learning_rate=0.1, 
#                              validation_split=0.2, 
#                              epochs=300,
#                              random_state = x)

    batm = partial(gl.build_and_train_model,T)
    pool = Pool(4)
    output = pool.map(batm, RS)
    pool.close()
    print(output)
if __name__ == "__main__":
    main()