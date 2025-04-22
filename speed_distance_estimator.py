import cv2
from tracker_util import measure_distance, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.window_frame = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance = {}

        for entity, paths in tracks.items():
            # Skip this operation for ball and referees, as this is not relevant to them.
            if entity == "ball" or entity == "referees":
                continue

            # Loop through a total number of frames with the frame rate.
            frames_count = len(paths)
            for frame_num in range(0, frames_count, self.window_frame):
                # Get the last frame for the calculations.
                last_frame = min(frame_num + self.window_frame, frames_count - 1)

                # if track id not in paths then continue.
                for t_id, _ in paths[frame_num].items():
                    if t_id not in paths[last_frame]:
                        continue

                    # Mark Starting and ending position skip next steps if one of them is null.
                    start_pos = paths[frame_num][t_id]['position_transformed']
                    end_pos = paths[last_frame][t_id]['position_transformed']

                    if start_pos is None or end_pos is None:
                        continue

                    # Calculate distance and Speed in MPH.
                    distance = measure_distance(start_pos, end_pos)
                    time = (last_frame - frame_num) / self.window_frame
                    speed_mile_per_hour = (distance / time) * 2.237

                    # Update and track distance in global store.
                    if entity not in total_distance:
                        total_distance[entity]= {}

                    if t_id not in total_distance[entity]:
                        total_distance[entity][t_id] = 0

                    total_distance[entity][t_id] += distance

                    # Pop up speed and distance for each frame in the video.
                    for frame_batch in range(frame_num, last_frame):
                        if t_id not in tracks[entity][frame_batch]:
                            continue

                        tracks[entity][frame_batch][t_id]['speed'] = speed_mile_per_hour
                        tracks[entity][frame_batch][t_id]['distance'] = total_distance[entity][t_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for entity, paths in tracks.items():
                # Skip this operation for ball and referees, as this is not relevant to them.
                if entity == "ball" or entity == "referees":
                    continue

                for _, track_info in paths[frame_num].items():
                   if "speed" in track_info:
                       speed = track_info.get('speed', None)
                       distance = track_info.get('distance', None)
                       if speed is None or distance is None:
                           continue

                       bbox = track_info['bbox']
                       position = get_foot_position(bbox)
                       position = list(position)
                       position[1]+=40

                       position = tuple(map(int,position))
                       cv2.putText(frame, f"{speed:.2f} mile/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                       cv2.putText(frame, f"{distance:.2f} m",(position[0], position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            output_frames.append(frame)

        return output_frames
