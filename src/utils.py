import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyarrow
import pyarrow.parquet as pq


def make_dir(directory):
    '''
    Creates a directory if there is no directory
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("Directory already exist: {}. No action taken".format(directory))


def convert_labels(file):
    '''
    function that takes in the .txt files from labels and convert them to
    usable numpy arrays
    '''
    numbers_we_care = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    temp = []
    for i in file.readlines():
        for j in i:
            if j in numbers_we_care:
                temp.append(int(j))
    temp = np.array(temp).astype('uint8')
    return temp


def resizing(image_path, mask_path, verbose=True, plot=False):
    '''
    resizes the images and masks to 240,320
    '''

    image_names = os.listdir(image_path)

    mask_names = os.listdir(mask_path)
    images = []
    masks = []
    for i in range(len(image_names)):
        img_temp = plt.imread(os.path.join(image_path, image_names[i]))
        if img_temp.shape == (240, 320, 3):
            images.append(plt.imread(os.path.join(image_path, image_names[i])))
            masks.append(plt.imread(os.path.join(mask_path, mask_names[i])))
        else:
            if verbose:
                print('Resizing image {} from {} -> (240,320)'.format(image_names[i],
                                                                      img_temp.shape))
            images.append(cv2.resize(img_temp, (320, 240)))
            masks.append(cv2.resize(plt.imread(os.path.join(mask_path, mask_names[i])), (320, 240)))
            if plot:
                plt.imshow(cv2.resize(img_temp, (320, 240)))
                plt.show()
    return images, masks


if __name__ == '__main__':
    # transformed_images, transformed_masks = resizing()
    pass
