"""
setup.py — Master setup script for glacier_prethicktor.

Runs the full data pipeline in order:
  00  Install Python dependencies
  01  Download RGI 6.2, GlaThiDa 3.1.0, Farinotti g2ti → refs.pkl
  02  Match GlaThiDa entries to nearest RGI glacier → matched.pkl

Usage:
    python setup.py

Each step is idempotent — already-completed work is skipped automatically.
After this script completes, the training notebooks and ML scripts are ready to run.
"""
import sys
import subprocess
import os

SCRIPTS = [
    ('00 — Install packages',          'scripts/00_install_packages.py'),
    ('01 — Acquire data',              'scripts/01_acquire_data.py'),
    ('02 — Match glacier centroids',   'scripts/02_match_glacier_centroids.py'),
]

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(label, script_path):
    full_path = os.path.join(REPO_ROOT, script_path)
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    result = subprocess.run([sys.executable, full_path], cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f'\nERROR: {script_path} exited with code {result.returncode}. Stopping.')
        sys.exit(result.returncode)


if __name__ == '__main__':
    print('glacier_prethicktor setup')
    print(f'Python: {sys.executable}')

    for label, script in SCRIPTS:
        run(label, script)

    print(f'\n{"="*60}')
    print('  Setup complete. You can now run the ML notebooks/scripts.')
    print(f'{"="*60}\n')
