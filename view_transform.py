import numpy as np
import cv2

class CoordinateMapping():
    def __init__(self):
        # Real world lengths of the center of the field court.
        soccer_field_length = 23.32
        soccer_field_width = 68

        # Image pixels coordinates of the corner of the trapezoid in the video frame.
        self.pixel_coordinates = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ]).astype(np.float32)

        # Mapping of the world coordinates of center of the field.
        self.world_coordinates = np.array([
            [0, soccer_field_width],
            [0, 0],
            [soccer_field_length, 0],
            [soccer_field_length, soccer_field_width]
        ]).astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_coordinates, self.world_coordinates)

    def transform_point_to_world_coordinates(self, point):
        # If a point is in the region with the pixel coordinates defined for the center of the soccer field.
        pnt = tuple(map(int, (point[0],point[1])))
        in_region = cv2.pointPolygonTest(self.pixel_coordinates, pnt, False) >= 0
        if not in_region:
            return None

        # Transform the point coordinates from pixel to world.
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        tranformed_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)

        return tranformed_point.reshape(-1, 2)

    def inject_adjusted_transformed_positions(self,tracks):
        for entity, paths in tracks.items():
            for frame_num, track in enumerate(paths):
                for track_id, track_info in track.items():
                    position = np.array(track_info['position_adjusted'])

                    trasnformed_position = self.transform_point_to_world_coordinates(position)
                    if trasnformed_position is not None:
                        trasnformed_position = trasnformed_position.squeeze().tolist()

                    tracks[entity][frame_num][track_id]['position_transformed'] = trasnformed_position
