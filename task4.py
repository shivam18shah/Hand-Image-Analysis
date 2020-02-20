# python task4.py -k 5 -f phase3_sample_data/Labelled/Set2 -u phase3_sample_data/Unlabelled/Set2 -c dtree/ppr

import argparse
from utils import get_images_by_metadata
import numpy as np
import ppr_classifier
from feature_extractor import get_hog_features_by_image_path
from task3 import pagerank
import dtree_temp
import feature_extractor
# from task3 import get_similar_images_for_each

parser = argparse.ArgumentParser(description='Label Images Dorsal/Palmar using latent semantics')
parser.add_argument('-k', type=int, default=3, help='The number of nearest neighbors')
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-u', required=True, type=str, help='the folder containing the unlabled images')
parser.add_argument('-c', required=True, type=str, help='the classifier to use')
excel_sheet = "HandInfo.xlsx"
vectors_file = 't4_vecs.csv'



def dtree(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids):
    dorsal_labels = np.array([[1]]*len(dorsal_image_ids))
    palmar_labels = np.array([[0]]*len(palmar_image_ids))
    dorsal_train = np.hstack((dorsal_data, dorsal_labels))
    palmar_train = np.hstack((palmar_data, palmar_labels))

    training_data = np.vstack((dorsal_train, palmar_train))
    # print(ul_data, ul_data.shape)
    print('########################   Training   ##############################')

    my_tree = dtree_temp.build_tree(training_data)
    print('########################   Trained   ##############################')
    predictions = list()
    for ul_vec, ul_id in zip(ul_data, ul_image_ids):
        # print(ul_id, ul_vec.shape)
        prediction = dtree_temp.classify(ul_vec, my_tree)
        for key, val in prediction.items():
            predictions.append([key, val])
    print('#####################   Finished   #################################')
    print('Predictions:',predictions)
    # ul_labels = get_label(ul_image_ids)
    return predictions


def get_label(labeled_images_folder, unlabeled_images_folder, k):
    """
    Given a folder with unlabeled images, the system labels
them as dorsal or palmer
    :param folder: The folder containing images for training
    :param k: The number of latent dimensions to compute
    :return: the labels for the unlabeled images
    """
    # Extract features for the input folder
    features_data, object_ids = feature_extractor.get_data_matrix(labeled_images_folder)

    # Extract features for the unlabeled folder
    ul_features_data, ul_object_ids = feature_extractor.get_data_matrix(unlabeled_images_folder)

    # Get dorsal images
    dorsal_data, dorsal_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=1)

    # Get palmar images
    palmar_data, palmar_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=0)

    unlabelled_data, unlabelled_image_ids = get_images_by_metadata(ul_object_ids, ul_features_data, unlabeled_images_folder)

    return dorsal_data, palmar_data, dorsal_image_ids, palmar_image_ids, unlabelled_data, unlabelled_image_ids


def classify(labeled_folder, unlabeled_folder, k, classifier):
    """

    :param labeled_folder:
    :param unlabeled_folder:
    :param k:
    :param classifier:
    :return:
    """
    # TODO  write your code here
    dorsal_data, palmar_data, dorsal_image_ids, palmar_image_ids, ul_data, ul_image_ids = get_label(labeled_folder, unlabeled_folder, k)
    # labeled_data,  = feature_extractor.get_data_matrix(labeled_folder, method='cm')
    # unlabeled_
    if classifier == 'dtree':
        # print('Unlabeled images, labels, confidence')
        preds = dtree(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids)
        # print('Idhar aa gaya')
        return preds
    if classifier == 'ppr':
        preds, confs = ppr_classifier.ppr(dorsal_data, dorsal_image_ids, palmar_data, palmar_image_ids, ul_data, ul_image_ids, unlabeled_folder)
        return preds[0][0]


if __name__ == "__main__":
    # args = parser.parse_args()
    # k = args.k
    # l_folder = args.f
    # u_folder = args.u
    # classifier = args.c
    # print(u_folder)
    # print(l_folder)
    # exit()
    l_folder = 'phase3_sample_data/Labelled/Set2'
    u_folder = 'phase3_sample_data/Unlabelled/Set2'
    k = 5
    classifier = 'ppr'
    preds = classify(l_folder, u_folder, k, classifier)
    accuracy = preds,
    #    classify(l_folder, u_folder, k, classifier)
