from TeamClassifier import TeamClassifier
from player_crop import get_crop_player
from video_read import load_video, write_video
from tracker import Tracker
from tracker_util import get_ball_updated_position

def main():
    frames = load_video("test_video.mp4")
    
    track_obj = Tracker("best.pt")
    tracks = track_obj.get_object_tracks(frames, read_from_stub=True, stub_path="/stubs/trackers.pkl")
    
    # Get the image of player for KNN segmetation.
    # get_crop_player(frames, tracks)
    
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
    
    output_video_frames = track_obj.draw_annotations(frames, tracks=tracks)
    
    write_video(frames, "videos/op.avi")

if __name__ == "__main__":
    main()
