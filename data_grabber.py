import pandas as pd

# def someting():
#     return nothing

def dg (pth='/home/sa42/data/glac/T_models/'):
    T = pd.read_csv('/home/sa42/data/glac/T_models/T.csv')
    T = T[[
        'LAT',
        'LON',
        'AREA',
        'MEAN_SLOPE',
    #     'MEAN_THICKNESS',
        'MAXIMUM_THICKNESS',
    ]]
    # drop null data
    T = T.dropna()

    TT = pd.read_csv('/home/sa42/data/glac/T_models/TT.csv')
    TT = TT[[
        'LOWER_BOUND',
        'UPPER_BOUND',
        'AREA',
        'MEAN_SLOPE',
        'MAXIMUM_THICKNESS'
    ]]
    TT = TT.dropna()


    TTT = pd.read_csv(pth + 'TTT.csv')
    TTT = TTT[[
        'ELEVATION',
        'THICKNESS',
        'POINT_LAT',
        'POINT_LON'
    ]]
    TTT = TTT.dropna()


    return T, TT, TTT


