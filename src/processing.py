'''
Run this script to process raw data into processed data

Raw data:
Labels in .txt file
Images in  .jpg

Processed data:
Labels:
    - Regions
    - Surface
    - Layers
Images in .jpg

all sorted with the right naming
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import convert_labels, make_dir
import cv2
import time


def create_processed_dir():
    '''
    Creates the follow directories with saved images

    data
        - processed
            - labels
                - surfaces
                - regions
            - images
    all saved in .tif format
    '''

    # Getting paths
    # Raw data paths
    raw_data_path = os.path.join(os.getcwd(), 'data', 'raw')
    raw_img_path = os.path.join(raw_data_path, 'images')
    raw_label_path = os.path.join(raw_data_path, 'labels')

    # Prcoessed data paths
    processed_data_path = os.path.join(os.getcwd(), 'data', 'processed')
    processed_labels_path = os.path.join(processed_data_path, 'labels')
    processed_surfaces_path = os.path.join(processed_labels_path, 'surfaces')
    processed_regions_path = os.path.join(processed_labels_path, 'regions')
    processed_images_path = os.path.join(processed_data_path, 'images')

    # Making directory
    make_dir(processed_data_path)
    make_dir(processed_labels_path)
    make_dir(processed_surfaces_path)
    make_dir(processed_regions_path)
    make_dir(processed_images_path)

    # Reading in and Transforming
    # Getting names
    img_names = os.listdir(raw_img_path)
    label_names = os.listdir(raw_label_path)
    regions_names = [x for x in label_names if 'regions' in x]
    surfaces_names = [x for x in label_names if 'surfaces' in x]

    # Sort
    img_names.sort()
    regions_names.sort()
    surfaces_names.sort()

    # Saving
    common_names = [x.split('.')[0] for x in regions_names]

    for i in range(len(common_names)):
        img_temp = plt.imread(os.path.join(raw_img_path, img_names[i]))
        with open(os.path.join(raw_label_path, regions_names[i])) as f:
            reg_temp = convert_labels(f)
        with open(os.path.join(raw_label_path, surfaces_names[i])) as f:
            surf_temp = convert_labels(f)
        cv2.imwrite(os.path.join(processed_regions_path,
                                 common_names[i]) + '.tif', np.reshape(reg_temp, img_temp.shape[:-1]))
        cv2.imwrite(os.path.join(processed_surfaces_path,
                                 common_names[i]) + '.tif', np.reshape(surf_temp, img_temp.shape[:-1]))
        cv2.imwrite(os.path.join(processed_images_path,
                                 common_names[i]) + '.tif', img_temp[:, :, [2, 1, 0]])  # Changing from RGB2BGR
        print("writing..: {}".format(regions_names[i].split('.')[0]))


if __name__ == '__main__':
    # raw data
    tic = time.time()
    create_processed_dir()
    print("Process Finished. Time taken: {}".format(time.time() - tic))
