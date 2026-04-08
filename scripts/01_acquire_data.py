"""
01_acquire_data.py — Download and prepare all datasets for glacier_prethicktor.

Downloads:
  - RGI 6.2     (OGGM mirror, ~421 MB, no auth)
  - GlaThiDa 3.1.0  (gtn-g.ch, ~5 MB)
  - Farinotti g2ti CSV  (OGGM mirror, ~8 MB, no auth)

Outputs:
  RGI_path/       *.csv — one per RGI region (20 total)
  glathida_path/  T.csv
  ref_path/       rgi60_itmix_df.csv, refs.pkl
"""
import os
import sys
import glob
import zipfile
import io
import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_manager as pm

(
    home_path, data_path, RGI_path, glathida_path,
    ref_path, coregistration_testing_path, arch_test_path, LOO_path
) = pm.set_paths()

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
RGI_OGGM_URL      = 'https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62.zip'
GLATHIDA_URL      = 'https://www.gtn-g.ch/database/glathida-3.1.0.zip'
FARINOTTI_CSV_URL = 'https://cluster.klima.uni-bremen.de/~oggm/g2ti/rgi60_itmix_df.csv'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_download(url, dest_path, label):
    """Download with progress bar. Skip if file already exists and is non-empty."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f'  {label}: already present, skipping.')
        return
    print(f'  Downloading {label}...')
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, desc=f'  {label}'
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
            bar.update(len(chunk))
    print(f'  {label} saved.')


def _valid_zip(path):
    try:
        with zipfile.ZipFile(path) as zf:
            zf.namelist()
        return True
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# 1. RGI 6.2
# ---------------------------------------------------------------------------

def acquire_rgi():
    print('\n[1/3] RGI 6.2')
    csv_files = glob.glob(os.path.join(RGI_path, '*.csv'))
    if csv_files:
        print(f'  RGI CSVs already present ({len(csv_files)} regions), skipping.')
        return

    rgi_zip = os.path.join(RGI_path, 'RGI.zip')
    if not _valid_zip(rgi_zip):
        if os.path.exists(rgi_zip):
            os.remove(rgi_zip)
        _stream_download(RGI_OGGM_URL, rgi_zip, 'RGI 6.2')

    from dbfread import DBF

    outer_dir = os.path.join(RGI_path, 'rgi62_extracted')
    os.makedirs(outer_dir, exist_ok=True)

    with zipfile.ZipFile(rgi_zip) as outer_zf:
        region_zips = [n for n in outer_zf.namelist()
                       if n.endswith('.zip') and 'rgi62' in n]
        print(f'  Extracting {len(region_zips)} regional zips...')
        for rz_name in tqdm(region_zips, desc='  Regions'):
            region_dir = os.path.join(outer_dir, rz_name.replace('.zip', ''))
            os.makedirs(region_dir, exist_ok=True)

            dbf_files = glob.glob(os.path.join(region_dir, '**', '*.dbf'), recursive=True)
            if not dbf_files:
                data = outer_zf.read(rz_name)
                with zipfile.ZipFile(io.BytesIO(data)) as inner_zf:
                    inner_zf.extractall(region_dir)
                dbf_files = glob.glob(os.path.join(region_dir, '**', '*.dbf'), recursive=True)
            if not dbf_files:
                continue

            table = DBF(dbf_files[0], encoding='latin-1', ignorecase=True)
            df = pd.DataFrame(iter(table))
            csv_name = rz_name[:2] + '_rgi62_attribs.csv'
            df.to_csv(os.path.join(RGI_path, csv_name), index=False)

    csv_files = glob.glob(os.path.join(RGI_path, '*.csv'))
    print(f'  Done — {len(csv_files)} CSV files written.')


# ---------------------------------------------------------------------------
# 2. GlaThiDa 3.1.0
# ---------------------------------------------------------------------------

def acquire_glathida():
    print('\n[2/3] GlaThiDa 3.1.0')
    t_csv = os.path.join(glathida_path, 'T.csv')
    if os.path.exists(t_csv):
        print('  T.csv already extracted, skipping.')
        return

    gz_path = os.path.join(glathida_path, 'glathida.zip')
    if not _valid_zip(gz_path):
        if os.path.exists(gz_path):
            os.remove(gz_path)
        _stream_download(GLATHIDA_URL, gz_path, 'GlaThiDa 3.1.0')

    target = 'glathida-3.1.0/data/T.csv'
    with zipfile.ZipFile(gz_path) as zf:
        file_size = zf.getinfo(target).file_size
        with zf.open(target) as src, open(t_csv, 'wb') as dst:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc='  Extracting T.csv') as bar:
                for chunk in iter(lambda: src.read(1024), b''):
                    dst.write(chunk)
                    bar.update(len(chunk))
    print(f'  T.csv saved to {t_csv}')


# ---------------------------------------------------------------------------
# 3. Farinotti et al. 2019 (OGGM g2ti CSV) → refs.pkl
# ---------------------------------------------------------------------------

def acquire_farinotti():
    print('\n[3/3] Farinotti et al. 2019 (g2ti)')
    refs_pkl = os.path.join(ref_path, 'refs.pkl')
    if os.path.exists(refs_pkl):
        print('  refs.pkl already exists, skipping.')
        return

    csv_path = os.path.join(ref_path, 'rgi60_itmix_df.csv')
    if not os.path.exists(csv_path):
        _stream_download(FARINOTTI_CSV_URL, csv_path, 'g2ti CSV')

    # FMT (mean thickness m) = volume (m³) / area (m²)
    # Area comes from RGI — same RGIId namespace (RGI60-)
    g2ti = pd.read_csv(csv_path)[['RGIId', 'vol_itmix_m3']]

    rgi_csvs = glob.glob(os.path.join(RGI_path, '*.csv'))
    if not rgi_csvs:
        raise RuntimeError('RGI CSVs not found — run acquire_rgi() first.')
    rgi_area = pd.concat(
        [pd.read_csv(f, usecols=['RGIId', 'Area']) for f in rgi_csvs],
        ignore_index=True,
    )

    merged = g2ti.merge(rgi_area, on='RGIId', how='inner')
    print(f'  Matched {len(merged):,} of {len(g2ti):,} g2ti glaciers to RGI area')

    merged['FMT'] = merged['vol_itmix_m3'] / (merged['Area'] * 1e6)
    refs = merged[['RGIId', 'FMT']].dropna()
    refs.to_pickle(refs_pkl)
    print(f'  refs.pkl saved — {len(refs):,} glaciers')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    acquire_rgi()
    acquire_glathida()
    acquire_farinotti()
    print('\nData acquisition complete.')
