import argparse
import numpy as np
from feature_extractor import get_data_matrix
from pca import get_pca_decomposition, get_pca_transform
from utils import get_images_by_metadata
from kmeans import get_cluster_centers, Euclidean

parser = argparse.ArgumentParser(description='Label Images Dorsal/Palmar using latent semantics')
parser.add_argument('-c', type=int, default=5, help='The number of clusters to generate')
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-u', required=True, type=str, help='the folder where unlabeled images are stored')


def get_clusters(labeled_images_folder, unlabeled_images_folder, c):
    """
    Given a folder with unlabeled images, the system labels
them as dorsal or palmer using clustering
    :param folder: The folder containing images for training
    :param c: The number of clusters to generate
    :return: the labels for the unlabeled images
    """
    # TODO  write your code here
    k = 20
    # Extract features for the input folder
    features_data, object_ids = get_data_matrix(labeled_images_folder, method="cm")

    # PCA decomposition
    u, vt, pca_obj = get_pca_decomposition(features_data, k)

    # Extract features for the unlabeled folder
    ul_features_data, ul_object_ids = get_data_matrix(unlabeled_images_folder, method="cm")

    # Get dorsal images
    dorsal_data, dorsal_image_ids = get_images_by_metadata(object_ids, u, labeled_images_folder, dorsal=1)

    # Get palmar images
    palmar_data, palmar_image_ids = get_images_by_metadata(object_ids, u, labeled_images_folder, dorsal=0)

    # transform ul_features
    ul_features_u = get_pca_transform(pca_obj, ul_features_data)

    # # PCA decomposition
    # dorsal_u, dorsal_vt = get_pca_decomposition(dorsal_data, k)
    # palmar_u, palmar_vt = get_pca_decomposition(palmar_data, k)

    # Get clusters for dorsal
    dorsal_centroids = get_cluster_centers(dorsal_data, c)
    # Get clusters for palmar
    palmar_centroids = get_cluster_centers(palmar_data, c)

    # print ("Dorsal centroids are:")
    # for i in dorsal_centroids:
    #     print (i)
    # print ("Palmar centroids are:")
    # for i in palmar_centroids:
    #     print (i)

    # Only to measure accuracy
    _, ul_dorsal_image_ids = get_images_by_metadata(ul_object_ids, ul_features_u, unlabeled_images_folder, dorsal=1)

    _, ul_palmar_image_ids = get_images_by_metadata(ul_object_ids, ul_features_u, unlabeled_images_folder, dorsal=0)

    # Label images
    pred_true_count = 0
    for features, object_id in zip(ul_features_u, ul_object_ids):
        min_dorsal_dist = None
        min_palmar_dist = None
        for i in dorsal_centroids:
            dist = euclidean(i, features)
            if min_dorsal_dist is None or min_dorsal_dist > dist:
                min_dorsal_dist = dist
        for i in palmar_centroids:
            dist = euclidean(i, features)
            if min_palmar_dist is None or min_palmar_dist > dist:
                min_palmar_dist = dist
        if min_dorsal_dist < min_palmar_dist:
            if object_id in ul_dorsal_image_ids:
                pred_true_count += 1
            print ("Image %s is dorsal." % object_id)
        else:
            if object_id in ul_palmar_image_ids:
                pred_true_count += 1
            print ("Image %s is palmar." % object_id)

    print ("accuracy is %s" % (pred_true_count*100/len(ul_object_ids)))


if __name__ == "__main__":
    args = parser.parse_args()
    c = args.c
    labeled_folder = args.f
    unlabeled_folder = args.u

    get_clusters(labeled_folder, unlabeled_folder, c)
