import cv2
import pandas as pd

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

# Todo, need to add the function to draw the ellipse and triangle for the players.

def get_ellipse_for_player(frame, x_center, y2, width, color):
    cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.36*width)),
            angle=0.0,
            startAngle=-40,
            endAngle=230,
            color = color,
            thickness=1,
            lineType=cv2.LINE_4
        )
    
def get_rectange(frame, x1_rect, y1_rect, x2_rect, y2_rect, color):
    cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)

def make_rect_dim( x_center, y2, rect_width, rect_height):
    x1_rect = x_center - rect_width//2
    x2_rect = x_center + rect_width//2
    y1_rect = (y2- rect_height//2) +15
    y2_rect = (y2+ rect_height//2) +15
    
    return (x1_rect, y1_rect, x2_rect, y2_rect)

def get_has_ball_color():
    return (0, 0, 255)

def get_palyer_color():
    return (0, 0, 255)

def get_referee_color():
    return (0, 255, 255)

def get_ball_color():
    return (0, 255, 0)

# This method will generated the estimated ball position for the frames where model failed to detect the ball.
def get_ball_updated_position(raw_ball_data):
    # Extract bounding boxes from each frame's data
    extracted_boxes = []
    for frame_data in raw_ball_data:
        bbox = frame_data.get(1, {}).get('bbox', [])
        extracted_boxes.append(bbox)

    # Using pandas dataframe
    coords_df = pd.DataFrame(extracted_boxes, columns=['x1','y1','x2','y2'])
    # We will interpolate the missing values for the ball coordinates.
    coords_df = coords_df.interpolate().bfill()
    # Reconstruct the formatted data structure
    generated_positions = [{1: {'bbox': row}} for row in coords_df.to_numpy().tolist()]
    
    return generated_positions
