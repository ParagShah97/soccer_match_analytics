import numpy as np
import cv2

class CoordinateMapping():
    def __init__(self):
        # Real world lengths of the center of the field court.
        soccer_field_length = 23.32
        soccer_field_width = 68

        # Image pixels coordinates of the corner of the trapezoid in the video frame.
        self.pixel_coordinates = np.array([
            [115, 1040],
            [268, 279],
            [909, 265],
            [1637, 912]
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
        pnt = (int(point[0]), int(point[1]))
        region = cv2.pointPolygonTest(self.pixel_coordinates, pnt, False)

        if region >= 0:
            # Transform the point coordinates from pixel to world.
            reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
            tranformed_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)

            return tranformed_point.reshape(-1, 2)
        else:
            return None

    def inject_adjusted_transformed_positions(self,tracks):
        for entity, paths in tracks.items():
            for frame_num, fram_track in enumerate(paths):
                for t_id, track_info in fram_track.items():
                    # Get the adjusted position and transform coordinates.
                    position = np.array(track_info.get('position_adjusted'))
                    trasnformed_position = self.transform_point_to_world_coordinates(position)

                    if trasnformed_position is not None:
                        trasnformed_position = trasnformed_position.squeeze().tolist()

                    track_info['position_transformed'] = trasnformed_position
