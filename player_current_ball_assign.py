from tracker_util import allowed_player_ball_distance, get_center_of_bbox, measure_distance

class PlayerCurrentBallAssign():
    def __init__(self):
        self.max_player_ball_distance = allowed_player_ball_distance()
            
    def find_nearest_player_to_ball(self, player_data, ball_box):
        ball_center = get_center_of_bbox(ball_box)
        
        closest_distance = float('inf')
        nearest_player_id = None

        for pid, pdata in player_data.items():
            player_bbox = pdata['bbox']
            left_foot = (player_bbox[0], player_bbox[-1])
            right_foot = (player_bbox[2], player_bbox[-1])

            dist_to_left = measure_distance(left_foot, ball_center)
            dist_to_right = measure_distance(right_foot, ball_center)
            player_dist = min(dist_to_left, dist_to_right)

            if player_dist < self.max_player_ball_distance and player_dist < closest_distance:
                closest_distance = player_dist
                nearest_player_id = pid

        return nearest_player_id
