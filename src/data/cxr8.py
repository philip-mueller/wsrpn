import ast
import logging
import os
import random
from functools import partial
from glob import glob
from multiprocessing import Pool
from shutil import rmtree

import kaggle
import pandas as pd
from tqdm import tqdm

from src import CXR8_DIR
from src.data import DatasetConfig

log = logging.getLogger(__name__)

"""
train: [
    'Atelectasis',         # 11559 ----
    'Cardiomegaly',        #  2776 ----
    'Consolidation',       #  4667
    'Edema',               #  2303
    'Effusion',            # 13317 ----
    'Emphysema',           #  2516
    'Fibrosis',            #  1686
    'Hernia',              #   227
    'Infiltration',        # 19894 ----
    'Mass',                #  5782 ----
    'Nodule',              #  6331 ----
    'Pleural_Thickening',  #  3385
    'Pneumonia',           #  1431 ----
    'Pneumothorax',        #  5302 ----
    'No Finding',          # 60361
]
total train_files: 112120
filtered train_files (only 8 classes and No Finding): 107010
"""

CLASSNAMES = [
    'Atelectasis',   # 102 | 78 | 180 in val | test | total
    'Cardiomegaly',  #  63 | 83 | 146
    'Effusion',      #  75 | 78 | 153
    'Infiltration',  #  63 | 60 | 123
    'Mass',          #  48 | 37 |  85
    'Nodule',        #  38 | 41 |  79
    'Pneumonia',     #  58 | 62 | 120
    'Pneumothorax',  #  45 | 53 |  98
    'No Finding',
]
CATEGORY_MAP = {c: index for index, c in enumerate(CLASSNAMES)}
OWN_BOXES_DIR = os.path.join(CXR8_DIR, 'own_bboxes')


def build_cxr8_dataset(config: DatasetConfig, mode: str):
    """
    Get all image-files, labels/bounding_boxes, and classnames for the CXR8
    dataset.
    """
    assert mode in ['train', 'val', 'test']

    if not os.path.exists(config.data_dir):
        log.info(f"Dataset not found, downloading to {config.data_dir}")
        download_cxr8(config.data_dir)

    prepare_cxr8(config.data_dir)

    df = pd.read_csv(os.path.join(config.data_dir,
                                  f'{mode}_list_detection.csv'))

    if mode == 'train':
        annotations = [ast.literal_eval(a) for a in df['labels'].values]
        annotations = [[CATEGORY_MAP[c] for c in a] for a in annotations]
    else:
        # (N, M_i, (x, y, w, h, c))
        # List of all samples. Each a list with M_i entries, each with a
        # bounding box of (x, y, w, h, c)
        annotations = [ast.literal_eval(a) for a in df['bboxes'].values]
        annotations = [[[*c[:-1], CATEGORY_MAP[c[-1]]] for c in a] for a in annotations]

    # Prepend absolute path
    with Pool(min(4, os.cpu_count())) as pool:
        files = pool.map(
            partial(prepend_path, path=config.data_dir),
            df['file_name'].values
        )

    return files, annotations, CLASSNAMES


def prepare_cxr8(path: str = CXR8_DIR):
    """
    Move all images in the images_* folders to a new folder called images
    Create train_list_detection.txt, val_list_detection.txt,
    and test_list_detection.txt
    """
    if not os.path.exists(os.path.join(path, 'images')):
        prepare_image_file_dirs_(path)

    if not os.path.exists(os.path.join(path, 'train_list_detection.csv')):
        prepare_image_file_lists_(path)


def prepare_image_file_dirs_(path: str = CXR8_DIR):
    log.info(f"Moving images to {os.path.join(path, 'images')}")
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)
    for f in glob(os.path.join(path, 'images_0*/images/*.png')):
        os.rename(f, os.path.join(path, 'images', os.path.basename(f)))

    log.info(f"Removing empty {os.path.join(path, 'images_0*')} folders")
    for d in glob(os.path.join(path, 'images_0*')):
        rmtree(d)


def prepare_image_file_lists_(path: str = CXR8_DIR):
    log.info("Creating train_list_detection.csv, val_list_detection.csv, "
             f"and test_list_detection.csv in {path}")
    all_files = [os.path.basename(f)
                 for f in glob(os.path.join(path, 'images/*.png'))]
    b = pd.read_csv(os.path.join(path, 'BBox_List_2017.csv'))
    bbox_files = list(b['Image Index'].values)

    # Train files are all files that are not in the bbox_files
    unique_patients = set([f[:-8] for f in bbox_files])
    train_files = [f for f in all_files if f[:-8] not in unique_patients]
    d = pd.read_csv(os.path.join(path, 'Data_Entry_2017.csv'))
    train_labels = []
    new_train_files = []
    valid_classes = CLASSNAMES + ['No Finding']
    for f in tqdm(train_files):
        labels = d[d['Image Index'] == f]['Finding Labels'].item().split('|')
        labels = [label for label in labels if label in valid_classes]
        if len(labels) > 0:
            new_train_files.append(f)
            train_labels.append(labels)
    train_files = new_train_files
    print(len(train_files), len(train_labels))
    train_df = pd.DataFrame({'file_name': train_files, 'labels': train_labels})
    train_df.to_csv(os.path.join(path, 'train_list_detection.csv'), index=False)

    # Split images in bbox_files equally between train and val. Watch for collisions
    random.Random(0).shuffle(bbox_files)
    n_files = len(bbox_files) // 2
    val_files = []
    test_files = []
    patient_names = set([box[:-8] for box in bbox_files])
    for p in patient_names:
        if len(val_files) < n_files:
            val_files.extend([f for f in bbox_files if f.startswith(p)])
        else:
            test_files.extend([f for f in bbox_files if f.startswith(p)])

    # Remove duplicate files
    val_files = list(set(val_files))
    test_files = list(set(test_files))

    for files, filename in [(val_files, 'val_list_detection.csv'),
                            (test_files, 'test_list_detection.csv')]:
        annotations = []
        for f in files:
            rows = b[b['Image Index'] == f]
            annot = []
            for i in range(len(rows)):
                row = rows.iloc[i]
                annot.append([
                    row['Bbox [x'].item(),
                    row['y'].item(),
                    row['w'].item(),
                    row['h]'].item(),
                    row['Finding Label'] if row['Finding Label'] != 'Infiltrate' else 'Infiltration',
                ])
            annotations.append(annot)
        df = pd.DataFrame({'file_name': files, 'bboxes': annotations})
        df.to_csv(os.path.join(path, filename), index=False)


def download_cxr8(path: str = CXR8_DIR):
    """Download the CXR8 dataset to path"""
    log.info(f"Downloading CXR8 dataset to {path}, this might take a while...")
    os.makedirs(path, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('nih-chest-xrays/data', path=path,
                                      unzip=True, force=False, quiet=False)

    # Move all image files to the images folder
    prepare_cxr8(path)


def prepend_path(fname, path):
    return os.path.join(path, 'images', fname)


if __name__ == '__main__':
    download_cxr8()
    prepare_cxr8()
