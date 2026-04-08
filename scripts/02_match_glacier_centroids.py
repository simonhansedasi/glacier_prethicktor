"""
02_match_glacier_centroids.py — Match every GlaThiDa entry to its nearest RGI glacier.

Uses a scipy cKDTree on 3D unit-sphere coordinates — gives the exact same
nearest-neighbour result as a haversine brute-force search at ~1000× the speed.

Output:
  data_path/matched.pkl  — GlaThiDa rows with RGIId and RGI Centroid Distance (km) appended
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_manager as pm
import glacierml as gl

(
    home_path, data_path, RGI_path, glathida_path,
    ref_path, coregistration_testing_path, arch_test_path, LOO_path
) = pm.set_paths()

MATCHED_OUT = os.path.join(data_path, 'matched.pkl')


# ---------------------------------------------------------------------------
# Coordinate helper
# ---------------------------------------------------------------------------

def latlon_to_xyz(lat, lon):
    """Convert lat/lon arrays to 3D unit-sphere vectors."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    return np.column_stack([
        np.cos(lat_r) * np.cos(lon_r),
        np.cos(lat_r) * np.sin(lon_r),
        np.sin(lat_r),
    ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if os.path.exists(MATCHED_OUT):
        print(f'matched.pkl already exists at {MATCHED_OUT}, skipping.')
        sys.exit(0)

    print('Loading RGI...')
    RGI = gl.load_RGI(RGI_path)
    print(f'  {len(RGI):,} RGI glaciers loaded.')

    print('Loading GlaThiDa...')
    glathida = pd.read_csv(os.path.join(glathida_path, 'T.csv'))
    glathida = glathida.dropna(subset=['MEAN_THICKNESS']).reset_index(drop=True)
    print(f'  {len(glathida):,} GlaThiDa entries with mean thickness.')

    print('Building KD-tree and matching...')
    rgi_xyz = latlon_to_xyz(RGI['CenLat'].values, RGI['CenLon'].values)
    gl_xyz  = latlon_to_xyz(glathida['LAT'].values, glathida['LON'].values)

    tree = cKDTree(rgi_xyz)
    chord_dist, idx = tree.query(gl_xyz, k=1, workers=-1)

    # Convert chord distance (unit sphere) back to great-circle km
    dist_km = 2 * 6371 * np.arcsin(np.clip(chord_dist / 2, 0, 1))

    glathida = glathida.copy()
    glathida['RGIId']                = RGI['RGIId'].iloc[idx].values
    glathida['RGI Centroid Distance'] = dist_km

    glathida.to_pickle(MATCHED_OUT)
    print(f'Saved matched.pkl — {len(glathida):,} entries → {MATCHED_OUT}')
    print(glathida[['RGIId', 'RGI Centroid Distance']].describe().round(2))
