from collections import defaultdict
import cv2
from tracker_util import measure_distance, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.window_frame = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance = defaultdict(lambda: defaultdict(float))

        for entity, frames in tracks.items():
            # Skip this operation for ball and referees, as this is not relevant to them.
            if entity in {"ball", "referees"}:
                continue

            # Loop through a total number of frames with the frame rate.
            for frame_num in range(0, len(frames), self.window_frame):
                # Get the last frame for the calculations.
                last_frame = min(frame_num + self.window_frame, len(frames) - 1)

                # if track id not in frames then continue.
                for t_id in frames[frame_num].keys():
                    if t_id not in frames[last_frame]:
                        continue

                    # Mark Starting and ending position skip next steps if one of them is null.
                    start_pos = frames[frame_num][t_id].get('position_transformed')
                    end_pos = frames[last_frame][t_id].get('position_transformed')

                    if start_pos is not None and end_pos is not None:
                        # Calculate distance and Speed in MPH.
                        distance = measure_distance(start_pos, end_pos)
                        time = (last_frame - frame_num) / self.window_frame
                        speed_mile_per_hour = (distance / time) * 2.237 if time > 0 else 0

                        # Update and track distance in global store.
                        total_distance[entity][t_id] += distance

                        # Pop up speed and distance for each frame in the video.
                        for frame_batch in range(frame_num, last_frame):
                            if t_id in tracks[entity][frame_batch]:
                                tracks[entity][frame_batch][t_id]['speed'] = speed_mile_per_hour
                                tracks[entity][frame_batch][t_id]['distance'] = total_distance[entity][t_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = list()

        for frame_num, frame in enumerate(frames):
            for entity, paths in tracks.items():
                # Skip this operation for ball and referees, as this is not relevant to them.
                if entity in {"ball", "referees"}:
                    continue

                for track_info in paths[frame_num].values():
                    speed = track_info.get('speed')
                    distance = track_info.get('distance')

                    if speed is not None and distance is not None:
                        pos = list(get_foot_position(track_info['bbox']))
                        position = (int(pos[0]), int(pos[1]) + 40)

                        cv2.putText(frame, f"{speed:.2f} mile/h", position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame, f"{distance:.2f} m",(position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames
