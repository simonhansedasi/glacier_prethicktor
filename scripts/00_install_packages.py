"""
00_install_packages.py — Install all dependencies for glacier_prethicktor.

Run once in your conda/venv environment before running any other script.
"""
import sys
import subprocess

PACKAGES = [
    'requests',
    'pandas',
    'tensorflow==2.11.0',
    'tqdm',
    'geopy',
    'matplotlib',
    'scikit-learn',
    'earthaccess',
    'geopandas',
    'dbfread',
    'scipy',
]

if __name__ == '__main__':
    print('Installing packages...')
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install',
        *PACKAGES,
        '--upgrade', 'numpy',
    ])
    print('\nVerifying key imports...')
    import numpy as np
    import tensorflow as tf
    print(f'  numpy:      {np.__version__}')
    print(f'  tensorflow: {tf.__version__}')
    print('\nAll packages installed.')
