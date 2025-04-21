from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
from tracker_util import get_ball_color, get_bbox_width, get_center_of_bbox, get_foot_position, get_ellipse_for_player, get_has_ball_color, get_palyer_color, get_rectange, get_referee_color, make_rect_dim

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        
    def detect_frames(self, frames, batchSize=20):
        detections = [] 
        for i in range(0,len(frames),batchSize):
            detections_batch = self.model.predict(frames[i:i+batchSize],conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    """
    To draw the ellipse around the players and refree.
    """
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        get_ellipse_for_player(frame, x_center, y2, width, color)

        # cv2.ellipse(
        #     frame,
        #     center=(x_center,y2),
        #     axes=(int(width), int(0.35*width)),
        #     angle=0.0,
        #     startAngle=-45,
        #     endAngle=235,
        #     color = color,
        #     thickness=2,
        #     lineType=cv2.LINE_4
        # )

        # x1_rect = x_center - rectangle_width//2
        # x2_rect = x_center + rectangle_width//2
        # y1_rect = (y2- rectangle_height//2) +15
        # y2_rect = (y2+ rectangle_height//2) +15

        
        x1_rect_dim, y1_rect_dim ,x2_rect_dim, y2_rect_dim = make_rect_dim(x_center, y2, rectangle_width = 40, rectangle_height=20)
        
        if track_id is not None:
            get_rectange(frame, x1_rect_dim, y1_rect_dim ,x2_rect_dim, y2_rect_dim, color)
            # cv2.rectangle(frame,
            #               (int(x1_rect),int(y1_rect) ),
            #               (int(x2_rect),int(y2_rect)),
            #               color,
            #               cv2.FILLED)
            
            x1_text = x1_rect_dim+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect_dim+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    """
    To draw the triangle over the ball.
    """
    def draw_traingle(self,frame,bbox,color):        
        x,_ = get_center_of_bbox(bbox)
        y= int(bbox[1])

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    """
    To draw the circle in place of bounding boxes
    """
    def annotate_video(self, frames, tracking_data, ball_possession_data):
        annotated_frames = []

        for idx, original_frame in enumerate(frames):
            annotated_frame = original_frame.copy()

            players_info = tracking_data["players"][idx]
            ball_info = tracking_data["ball"][idx]
            referees_info = tracking_data["referees"][idx]

            # Annotate players
            for player_id, player_data in players_info.items():
                team_color = player_data.get("team_color", get_palyer_color())
                annotated_frame = self.draw_ellipse(annotated_frame, player_data["bbox"], team_color, player_id)

                if player_data.get("has_ball", False):
                    annotated_frame = self.draw_traingle(annotated_frame, player_data["bbox"], get_has_ball_color())

            # Annotate referees
            for _, referee_data in referees_info.items():
                annotated_frame = self.draw_ellipse(annotated_frame, referee_data["bbox"], get_referee_color())

            # Annotate ball
            for _, ball_data in ball_info.items():
                annotated_frame = self.draw_traingle(annotated_frame, ball_data["bbox"], get_ball_color())

            # Annotate team in possession
            annotated_frame = self.draw_team_ball_control(annotated_frame, idx, ball_possession_data)

            annotated_frames.append(annotated_frame)

        return annotated_frames
