from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
# import sys
from tracker_util import get_ball_color, get_bbox_width, get_center_of_bbox, get_ellipse_for_player, get_foot_position, get_has_ball_color, get_player_color, get_rectangle, get_referee_color, make_rect_dim

# Todo: remove all the prints and commented code.

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections += self.model.predict(frames[i:i + batch_size], conf=0.1)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_supervision = sv.Detections.from_ultralytics(detection)

            for i, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[i] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for d in detection_with_tracks:
                bbox = d[0].tolist()
                cls_id, track_id = d[3], d[4]
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for d in detection_supervision:
                bbox, cls_id = d[0].tolist(), d[3]
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def add_position_to_tracks(self, tracks):
        for obj, frames in tracks.items():
            for frame_num, tracked_objs in enumerate(frames):
                for track_id, info in tracked_objs.items():
                    bbox = info['bbox']
                    position = get_foot_position(bbox) if obj != 'ball' else get_center_of_bbox(bbox)
                    tracks[obj][frame_num][track_id]['position'] = position

    def draw_player_mapper(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        get_ellipse_for_player(frame, x_center, y2, width, color)

        if track_id is not None:
            x1_rect, y1_rect, x2_rect, y2_rect = make_rect_dim(x_center, y2, 40, 20)
            get_rectangle(frame, x1_rect, y1_rect, x2_rect, y2_rect, color)
            x1_text = x1_rect + 12 if track_id <= 99 else x1_rect + 2
            cv2.putText(frame, f"{track_id}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def draw_triangle(self, frame, bbox, color):
        x, _ = get_center_of_bbox(bbox)
        y = int(bbox[1])
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame
    
    def render_ball_possession_overlay(self, image, current_index, possession_history):
        height, width = image.shape[:2]

        # Define relative positions
        x_start = int(width * 0.70)
        y_start = int(height * 0.90)
        x_end = int(width * 0.97)
        y_end = int(height * 0.98)

        shaded_layer = image.copy()
        cv2.rectangle(shaded_layer, (x_start, y_start), (x_end, y_end), (255, 255, 255), thickness=-1)
        cv2.addWeighted(shaded_layer, 0.4, image, 0.6, 0, image)

        recent_possession = possession_history[:current_index + 1]
        team_one_count = (recent_possession == 1).sum()
        team_two_count = (recent_possession == 2).sum()

        total = team_one_count + team_two_count
        team_one_pct = team_one_count / total if total else 0
        team_two_pct = team_two_count / total if total else 0

        text_one = f"Team 1 Ball Possession : {team_one_pct * 100:.2f}%"
        text_two = f"Team 2 Ball Possession: {team_two_pct * 100:.2f}%"

        text_x = x_start + int(width * 0.01)
        text_y_1 = y_start + int(height * 0.03)
        text_y_2 = y_start + int(height * 0.07)

        cv2.putText(image, text_one, (text_x, text_y_1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=1)
        cv2.putText(image, text_two, (text_x, text_y_2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), thickness=1)

        return image


    # This is the central method which will add all the annotatons for the match stream.
    # This will consist of:
    # 1) Player, refrees on the ground.
    # 2) Ball movement
    # 3) Mapping for the player who have the ball.
    def annotate_video(self, frames, tracks, team_ball_control):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            annotated = frame.copy()
            for pid, pdata in tracks['players'][frame_num].items():
                color = pdata.get('team_color', get_player_color())
                annotated = self.draw_player_mapper(annotated, pdata['bbox'], color, pid)
                # check if the current player have the ball or not.
                if pdata.get('has_ball', False):
                    annotated = self.draw_triangle(annotated, pdata['bbox'], get_has_ball_color())

            for _, ref in tracks['referees'][frame_num].items():
                annotated = self.draw_player_mapper(annotated, ref['bbox'], get_referee_color())

            for _, ball in tracks['ball'][frame_num].items():
                annotated = self.draw_triangle(annotated, ball['bbox'], get_ball_color())

            annotated = self.render_ball_possession_overlay(annotated, frame_num, team_ball_control)
            output_frames.append(annotated)

        return output_frames
