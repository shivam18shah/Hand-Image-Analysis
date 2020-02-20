from sklearn.decomposition import PCA


def get_pca_decomposition(data, k):
    pca_obj = PCA(n_components=min(min(data.shape), k))
    pca_obj.fit(data)
    X_pca = pca_obj.transform(data)
    vt = pca_obj.components_
    return X_pca, vt, pca_obj


def get_pca_transform(pca_obj, data):
    return pca_obj.transform(data)
