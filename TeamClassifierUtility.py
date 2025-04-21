#  This is a supporting class for TeamClassifier

from sklearn.cluster import KMeans
import numpy as np

# This method will take image and return Kmean clustered model.
def _initialize_kmeans(img_patch):
        pixel_data = img_patch.reshape(-1, 3)
        model = KMeans(n_clusters=2, init='k-means++', n_init=1)
        model.fit(pixel_data)
        return model

# This method will return the player color label based on the background label.   
def get_player_color_label(label_matrix):
    corners = [
            label_matrix[0, 0],
            label_matrix[0, -1],
            label_matrix[-1, 0],
            label_matrix[-1, -1]
        ]
    background_label = max(set(corners), key=corners.count)
    player_label = 1 - background_label
    return player_label

# This method will return k-mean clustering model.
def cluster_model(all_colors):
    clustering_model = KMeans(n_clusters=2, init='k-means++', n_init=10)
    clustering_model.fit(all_colors)
    return clustering_model