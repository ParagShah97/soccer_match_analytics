import numpy as np
from TeamClassifier import TeamClassifier
from camera_shift_analyzer import CameraShiftAnalyzer
from player_crop import get_crop_player
from video_read import load_video, write_video
from tracker import Tracker
from tracker_util import get_ball_updated_position
from player_current_ball_assign import PlayerCurrentBallAssign
from view_transform import CoordinateMapping
from speed_distance_estimator import SpeedAndDistanceEstimator

def main():    
    frames = load_video("test_video.mp4")

    track_obj = Tracker("best.pt")
    tracks = track_obj.get_object_tracks(frames, read_from_stub=True, stub_path="/stubs/trackers.pkl")
    track_obj.add_position_to_tracks(tracks)

    # Get the image of player for KNN segmetation.
    # get_crop_player(frames, tracks)

    # Camera Movement analyzer
    camera_movement_estimator = CameraShiftAnalyzer(frames[0])
    camera_movement_per_frame = camera_movement_estimator.estimate_camera_shifts(frames,
                                                                                load_from_file=True,
                                                                                file_path='/stubs/camera_shift_stub.pkl')
    camera_movement_estimator.inject_adjusted_positions(tracks,camera_movement_per_frame)

    # View Transformer for image to world coordinate mapping.
    view_transformer = CoordinateMapping()
    view_transformer.inject_adjusted_transformed_positions(tracks)

    # Get Ball location prediction
    tracks["ball"] = get_ball_updated_position(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Team assignment here:
    team_assign = TeamClassifier()
    first_frame = frames[0]
    initial_player_data = tracks['players'][0]
    team_assign.cluster_players_by_team(first_frame, initial_player_data)

    for idx, player_data_by_frame in enumerate(tracks['players']):
        current_frame = frames[idx]

        for pid, player_info in player_data_by_frame.items():
            assigned_team = team_assign.identify_player_team(current_frame, player_info['bbox'], pid)

            player_info['team'] = assigned_team
            player_info['team_color'] = team_assign.team_palette[assigned_team]

    # Assign the player who have ball at the moment.
    assigned_player_obj = PlayerCurrentBallAssign()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = assigned_player_obj.find_nearest_player_to_ball(player_track, ball_bbox)

        if assigned_player != None:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    output_video_frames = track_obj.annotate_video(frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.visualize_camera_shifts(output_video_frames,camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    write_video(output_video_frames, "videos/op.avi")

if __name__ == "__main__":
    main()
