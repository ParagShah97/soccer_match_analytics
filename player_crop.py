# This file is for cropping a player from the frame for doing the segmentation.
import cv2

def get_crop_player(video_frames, player_tracks):
    
    # Get the player image from the frames
    for track_id, player_data in player_tracks['players'][0].items():
        bnd_box = player_data['bbox']
        frame = video_frames[0]

        # Crop bounding box from frame
        crop_img = frame[int(bnd_box[1]):int(bnd_box[3]), int(bnd_box[0]):int(bnd_box[2])]

        # Save the cropped image
        cv2.imwrite('player_segment/player_img.jpg', crop_img)
        break

    