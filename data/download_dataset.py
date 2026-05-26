"""
Download the Brain Tumor MRI Dataset from Kaggle.

Usage:
    python data/download_dataset.py

Requirements:
    - kaggle CLI configured (~/.kaggle/kaggle.json)
    - pip install kaggle
"""

import os
import subprocess
import sys

DATASET   = 'masoudnickparvar/brain-tumor-mri-dataset'
DATA_DIR  = os.path.join(os.path.dirname(__file__), 'brain_tumor_dataset')


def main():
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        print(f'Dataset already exists at {DATA_DIR}')
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'Downloading {DATASET}...')

    result = subprocess.run(
        ['kaggle', 'datasets', 'download', '-d', DATASET, '-p', DATA_DIR, '--unzip'],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print('Error downloading dataset:')
        print(result.stderr)
        sys.exit(1)

    print(f'Dataset downloaded to {DATA_DIR}')
    print('Directory structure:')
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, '').count(os.sep)
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        if level < 2:
            sub = '  ' * (level + 1)
            for f in files[:3]:
                print(f'{sub}{f}')
            if len(files) > 3:
                print(f'{sub}... and {len(files) - 3} more')


if __name__ == '__main__':
    main()
