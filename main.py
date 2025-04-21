import numpy as np
from TeamClassifier import TeamClassifier
from camera_shift_analyzer import CameraShiftAnalyzer
from player_crop import get_crop_player
from video_read import load_video, write_video
from tracker import Tracker
from tracker_util import get_ball_updated_position
from player_current_ball_assign import PlayerCurrentBallAssign

def main():
    frames = load_video("test_video.mp4")
    
    track_obj = Tracker("best.pt")
    tracks = track_obj.get_object_tracks(frames, read_from_stub=True, stub_path="/stubs/trackers.pkl")
    
    # Get the image of player for KNN segmetation.
    # get_crop_player(frames, tracks)
    
    # Camera Movement analyzer
    camera_movement_estimator = CameraShiftAnalyzer(frames[0])
    camera_movement_per_frame = camera_movement_estimator.estimate_camera_shifts(frames,
                                                                                load_from_file=True,
                                                                                file_path='/stubs/camera_shift_stub.pkl')

    
    # Get Ball location prediction
    tracks["ball"] = get_ball_updated_position(tracks["ball"])    
    
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
    
    # for frame_num, player_track in enumerate(tracks['players']):
    #     for player_id, track in player_track.items():
    #         team = team_assign.get_player_team(frames[frame_num],   
    #                                              track['bbox'],
    #                                              player_id)
    #         tracks['players'][frame_num][player_id]['team'] = team 
    #         tracks['players'][frame_num][player_id]['team_color'] = team_assign.team_colors[team]
    
    # Assign the player who have ball at the moment.
    assigned_player_obj = PlayerCurrentBallAssign()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = assigned_player_obj.find_nearest_player_to_ball(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    
    output_video_frames = track_obj.annotate_video(frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.visualize_camera_shifts(output_video_frames,camera_movement_per_frame)
    
    write_video(output_video_frames, "videos/op.avi")

if __name__ == "__main__":
    main()
