import pickle
import cv2
import numpy as np
import os

from tracker_util import measure_distance, measure_xy_distance

class CameraShiftAnalyzer:
    def __init__(self, first_frame):
        # Minimum movement to consider as camera shift
        self.distance_threshold = 5

        # Parameters for optical flow
        self.optical_flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Create restricted mask to not include side regions for tracking
        gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.restricted_mask = np.zeros_like(gray_init)
        self.restricted_mask[:, :20] = 1
        self.restricted_mask[:, 900:1050] = 1

        # Parameters for corner detection
        self.feature_detection_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=self.restricted_mask
        )

    def inject_adjusted_positions(self, trajectory_data, shifts_per_frame):
        # Subtract estimated camera shift from the each tracked position
        for entity, paths in trajectory_data.items():
            for frame_idx, frame_tracks in enumerate(paths):
                for tid, info in frame_tracks.items():
                    original_x, original_y = info['position']
                    shift_x, shift_y = shifts_per_frame[frame_idx]
                    adjusted_pos = (original_x - shift_x, original_y - shift_y)
                    info['position_adjusted'] = adjusted_pos

    # This method we will calculate the camera shift, and we can store the shift parameters to pkl file.
    def estimate_camera_shifts(self, frames, load_from_file=False, file_path=None):
        # Load the precomuted shift if it is available.
        if load_from_file and file_path and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        shifts = [[0, 0] for _ in range(len(frames))]
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_points = cv2.goodFeaturesToTrack(prev_gray, **self.feature_detection_params)

        for idx in range(1, len(frames)):
            curr_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)

            # Track points using optical flow
            new_points, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **self.optical_flow_params)

            max_movement = 0
            shift_x, shift_y = 0, 0

            # Find the point pair with maximum distance.
            for new, old in zip(new_points, prev_points):
                new_pt = new.ravel()
                old_pt = old.ravel()

                dist = measure_distance(new_pt, old_pt)
                if dist > max_movement:
                    max_movement = dist
                    shift_x, shift_y = measure_xy_distance(old_pt, new_pt)

            # If movement exceeds threshold then update shift.
            if max_movement > self.distance_threshold:
                shifts[idx] = [shift_x, shift_y]
                prev_points = cv2.goodFeaturesToTrack(curr_gray, **self.feature_detection_params)

            prev_gray = curr_gray.copy()

        # Save computed shifts.
        if file_path:
            with open(file_path, 'wb') as f:
                pickle.dump(shifts, f)

        return shifts

    def visualize_camera_shifts(self, frames, shifts):
        annotated_frames = []

        for idx, frame in enumerate(frames):
            overlay = frame.copy()

            # Overlay box for shift info
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), thickness=-1)
            blended = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Draw shift values on frame
            shift_x, shift_y = shifts[idx]
            cv2.putText(blended, f"Shift X: {shift_x:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(blended, f"Shift Y: {shift_y:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            annotated_frames.append(blended)

        return annotated_frames
