# This class will classify the player based on the jersey color into 2 different teams.
from sklearn.cluster import KMeans
import numpy as np
from TeamClassifierUtility import _initialize_kmeans, cluster_model, get_player_color_label

class TeamClassifier:
    def __init__(self):
        self.team_palette = {}
        self.player_team_map = {}
        self.kmeans_model = None

    def _extract_player_dominant_color(self, frame, box):
        x_min, y_min, x_max, y_max = map(int, box)
        player_crop = frame[y_min:y_max, x_min:x_max]
        tshirt_space = player_crop[:player_crop.shape[0] // 2, :]
        # From Utility
        kmeans_model = self._initialize_kmeans(tshirt_space)
        pixel_labels = kmeans_model.labels_
        label_matrix = pixel_labels.reshape(tshirt_space.shape[:2])
        # From Utility
        player_label = get_player_color_label(label_matrix)
        return kmeans_model.cluster_centers_[player_label]
    
    # Cluster the player of teams. 
    def cluster_players_by_team(self, image_frame, detections):
        # store all the colors from the team jersey.
        all_colors = []
        # Iterate over all the players
        for player in detections.values():
            player_box = player["bbox"]
            color_vector = self._extract_player_dominant_color(image_frame, player_box)
            all_colors.append(color_vector)

        self.kmeans_model = cluster_model(all_colors)

        self.team_palette[1] = self.kmeans_model.cluster_centers_[0]
        self.team_palette[2] = self.kmeans_model.cluster_centers_[1]

    def identify_player_team(self, image_frame, box, pid):
        if pid in self.player_team_map:
            return self.player_team_map[pid]

        dominant_color = self._extract_player_dominant_color(image_frame, box)
        predicted_cluster = self.kmeans_model.predict(dominant_color.reshape(1, -1))[0]
        team_number = predicted_cluster + 1

        # Edge case
        if pid == 91:
            team_number = 1

        self.player_team_map[pid] = team_number
        return team_number
