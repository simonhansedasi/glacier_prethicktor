import glacierml as gl
import numpy as np
from tqdm import tqdm

# loss = 'mae'
for i in tqdm(np.logspace(-5,3,10)):
    for j in np.logspace(-3,4,8):
        if i >= j:
            pass
        if i < j:
            (
                x,y,z,unc,
                x_new,y_new,z_new,unc_new ,
                far_ind, est_ind ,unc_ind
            ) = gl.assign_arrays(
                parameterization = '4',method = '1', loss = 'first',
#                 analysis = 'vol',
                size_thresh_1 = i, size_thresh_2 = j
            )
        
        
        
        
for i in tqdm(range(0, 900, 100)):
    for j in range(100,1000,100):
        if i >= j:
            pass
        if i < j:
    
            (
                x,y,z,unc,
                x_new,y_new,z_new,unc_new ,
                far_ind, est_ind ,unc_ind
            ) = gl.assign_arrays(
                parameterization = '4',method = '1',loss = 'first',
#                 analysis = 'thick',
                size_thresh_1 = i, size_thresh_2 = j
            )