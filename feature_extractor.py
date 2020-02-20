import os
import cv2
from skimage.feature import hog
from skimage.transform import rescale, downscale_local_mean

import numpy as np

def get_hog_features_by_image_path(path):
    # Given image path generate HOG features
    image = cv2.imread(path)
    image = rescale(image, 0.1)
    # BGR to RGB
    image = image[:, :, ::-1]
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd = np.ravel(fd).astype('float32')
    return fd.tolist()

def get_files_in_folder(folder):
    # Get list of files given a folder path
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def get_data_matrix(folder):
    features = []
    object_ids = []
    files = get_files_in_folder(folder)
    count = 0
    for file in files:
        if count % 100 == 0:
            print("processed %s files" % count)
        count += 1
        features_one = get_hog_features_by_image_path(file)
        object_ids.append(file.split('\\')[-1].split("/")[-1])
        features.append(features_one)
    data_matrix = np.asarray(features)
    print(data_matrix.shape)
    return data_matrix, object_ids