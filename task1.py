import argparse
from feature_extractor import get_data_matrix
from pca import get_pca_decomposition
from utils import get_images_by_metadata

parser = argparse.ArgumentParser(description='Label Images Dorsal/Palmar using latent semantics')
parser.add_argument('-k', type=int, default=5, help='The number of latent semantics to generate')
parser.add_argument('-f', required=True, type=str, help='the folder where labeled images are stored')
parser.add_argument('-u', required=True, type=str, help='the folder where unlabeled images are stored')


def get_label(labeled_images_folder, unlabeled_images_folder, k):
    """
    Given a folder with unlabeled images, the system labels
them as dorsal or palmer
    :param folder: The folder containing images for training
    :param k: The number of latent dimensions to compute
    :return: the labels for the unlabeled images
    """
    # Extract features for the input folder
    features_data, object_ids = get_data_matrix(labeled_images_folder)

    # Extract features for the unlabeled folder
    ul_features_data, ul_object_ids = get_data_matrix(unlabeled_images_folder)

    # Get dorsal images
    dorsal_data, dorsal_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=1)

    # Get palmar images
    palmar_data, palmar_image_ids = get_images_by_metadata(object_ids, features_data, labeled_images_folder, dorsal=0)

    # PCA decomposition
    dorsal_u, dorsal_vt = get_pca_decomposition(dorsal_data, k)
    palmar_u, palmar_vt = get_pca_decomposition(palmar_data, k)

    print (dorsal_data.shape, palmar_data.shape)
    print (dorsal_u.shape, dorsal_vt.shape)
    print (palmar_u.shape, palmar_vt.shape)

if __name__ == "__main__":
    args = parser.parse_args()
    k = args.k
    labeled_folder = args.f
    unlabeled_folder = args.u

    get_label(labeled_folder, unlabeled_folder, k)

