import pickle
import cv2
import numpy as np
import os

from tracker_util import measure_distance, measure_xy_distance
# from tracker_util import measure_distance, measure_xy_distance

class CameraShiftAnalyzer:
    def __init__(self, first_frame):
        self.distance_threshold = 5

        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        gray_initial = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        restricted_mask = np.zeros_like(gray_initial)
        restricted_mask[:, :20] = 1
        restricted_mask[:, 900:1050] = 1

        self.feature_detection_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=restricted_mask
        )

    def inject_adjusted_positions(self, trajectory_data, shifts_per_frame):
        for entity, paths in trajectory_data.items():
            for frame_idx, path in enumerate(paths):
                for tid, info in path.items():
                    original_pos = info['position']
                    shift = shifts_per_frame[frame_idx]
                    adjusted_pos = (original_pos[0] - shift[0], original_pos[1] - shift[1])
                    trajectory_data[entity][frame_idx][tid]['position_adjusted'] = adjusted_pos

    # This method we will calculate the camera shift, and we can store the shift parameters to pkl file.
    def estimate_camera_shifts(self, frames_list, load_from_file=False, file_path=None):
        if load_from_file and file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        shifts = [[0, 0]] * len(frames_list)

        prev_gray = cv2.cvtColor(frames_list[0], cv2.COLOR_BGR2GRAY)
        prev_points = cv2.goodFeaturesToTrack(prev_gray, **self.feature_detection_params)

        for idx in range(1, len(frames_list)):
            curr_gray = cv2.cvtColor(frames_list[idx], cv2.COLOR_BGR2GRAY)
            new_points, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **self.optical_flow_params)

            biggest_movement = 0
            move_x, move_y = 0, 0

            for new_pt, old_pt in zip(new_points, prev_points):
                new_pt_flat = new_pt.ravel()
                old_pt_flat = old_pt.ravel()

                dist = measure_distance(new_pt_flat, old_pt_flat)
                if dist > biggest_movement:
                    biggest_movement = dist
                    move_x, move_y = measure_xy_distance(old_pt_flat, new_pt_flat)

            if biggest_movement > self.distance_threshold:
                shifts[idx] = [move_x, move_y]
                prev_points = cv2.goodFeaturesToTrack(curr_gray, **self.feature_detection_params)

            prev_gray = curr_gray.copy()

        if file_path:
            with open(file_path, 'wb') as f:
                pickle.dump(shifts, f)

        return shifts

    def visualize_camera_shifts(self, frames_list, shifts_list):
        annotated_frames = []

        for idx, frame in enumerate(frames_list):
            annotated_frame = frame.copy()

            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), thickness=-1)
            cv2.addWeighted(overlay, alpha=0.6, src2=annotated_frame, beta=0.4, gamma=0, dst=annotated_frame)

            shift_x, shift_y = shifts_list[idx]
            cv2.putText(annotated_frame, f"Shift X: {shift_x:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(annotated_frame, f"Shift Y: {shift_y:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            annotated_frames.append(annotated_frame)

        return annotated_frames