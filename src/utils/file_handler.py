import numpy as np
import os

import cv2


def dir_img_list(dir_path, file_ext='jpg'):
    # return list of image file paths in a directory using os listdir
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext)]


def img_iter(dir_path, file_ext='jpg'):
    # iterator that yields image as a numpy array
    image_paths = dir_img_list(dir_path, file_ext)
    for path in image_paths:
        yield cv2.imread(path)


if __name__ == '__main__':
    path = r'C:\Users\omri_\OneDrive\Documents\neurogrametry_data\dogarden'
    img_files = dir_img_list(path)
    print(img_files)