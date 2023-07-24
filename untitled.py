import pandas as pd
from geopy.distance import geodesic
from itertools import combinations
import multiprocessing as mp
import numpy as np
parameterization = '4'

import glacierml as gl


df = pd.read_hdf(
    'predicted_thicknesses/compiled_raw_' + '4' + '.h5',
    key = 'compiled_raw', mode = 'a'
)
print('df loaded')
df_index_list = []
for i in range(0, len(df), 216501):
    df_index_list.append(i)

df = df.loc[
    df_index_list
]

weights = np.load(
    'model_weights/architecture_weights_' + parameterization +'.pkl', allow_pickle = True
)


df = pd.merge(df, weights, how = 'inner', on = 'layer architecture')
grp_lst_args = list(df.groupby('RGIId').groups.items())
print('df grouped')



model_list = []
for i in range(0,25,1):
    model_list.append(str(i))
    
def calc_dist2(arg):
    grp, lst = arg
    print('working on ' + grp)
    dft = df[model_list].loc[lst]

    predictions = df[model_list].loc[lst].to_numpy()   
    
    mean_thickness, mean_ci, var_ci = gl.calculate_confidence_intervals(predictions)
    
    
    t,tu = gl.mean_weighter(
        mean_thickness, mean_ci, parameterization
    )
    
    s1 = []
    s2 = []
    
    for i in range(1,5,1): 
        g = df['IQR_' + str(i)].loc[lst] / 1.34896
        print(g)
        print(t)
        print(g * t)
        s_mean = str(g * t)
        s1.append(s_mean)
        s_ci = str(g * tu)
        print(s_ci)
        s2.append(s_ci)
        
    print(s2)
    return pd.DataFrame(
               [ [grp,
                  t[0],tu[0][0],tu[0][1],
                  t[1],tu[1][0],tu[1][1],
                  t[2],tu[2][0],tu[2][1],
                  t[3],tu[3][0],tu[3][1],
                  s1[0],s2[0][0],s2[0][1],
                  s1[1],s2[1][0],s2[1][1],
                  s1[2],s2[2][0],s2[2][1],
                  s1[3],s2[3][0],s2[3][1]
                 ]
               ],
               columns=[
                   'RGIId','mean1','lower1','upper1',
                           'mean2','lower2','upper2',
                           'mean3','lower3','upper3',
                           'mean4','lower4','upper4',
                           'unc1','lowerunc1','upperunc1',
                           'unc2','lowerunc2','upperunc2',
                           'unc3','lowerunc3','upperunc3',
                           'unc4','lowerunc4','upperunc4',
               ])


pool = mp.Pool(processes = (48))
results = pool.map(calc_dist2, grp_lst_args)
pool.close()
pool.join()

results_df = pd.concat(results)

print(results_df)