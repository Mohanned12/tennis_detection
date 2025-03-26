import cv2
import numpy as np
import pandas as pd
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from playsound import playsound  # Install with: pip install playsound

# Import utility functions and classes
from utils import (read_video,
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import constants


def compute_shot_statistics(start_frame, end_frame, ball_mini_court_detections, player_mini_court_detections, mini_court):
    """
    Compute shot speed and player movement speed for a given shot interval.
    """
    ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # Assuming 24 FPS

    # Ball Speed Calculation
    distance_covered_by_ball_pixels = measure_distance(
        ball_mini_court_detections[start_frame][1],
        ball_mini_court_detections[end_frame][1]
    )
    distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
        distance_covered_by_ball_pixels,
        constants.DOUBLE_LINE_WIDTH,
        mini_court.get_width_of_mini_court()
    )
    speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

    # Player Movement Speed Calculation
    player_positions = player_mini_court_detections[start_frame]
    player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(
        player_positions[player_id],
        ball_mini_court_detections[start_frame][1]
    ))
    opponent_player_id = 1 if player_shot_ball == 2 else 2

    distance_covered_by_opponent_pixels = measure_distance(
        player_mini_court_detections[start_frame][opponent_player_id],
        player_mini_court_detections[end_frame][opponent_player_id]
    )
    distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(
        distance_covered_by_opponent_pixels,
        constants.DOUBLE_LINE_WIDTH,
        mini_court.get_width_of_mini_court()
    )
    speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

    return speed_of_ball_shot, speed_of_opponent, player_shot_ball, opponent_player_id


def main():
    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Initialize Trackers and Detectors
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    # Detect Players and Ball
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detector Model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Mini-Court
    mini_court = MiniCourt(video_frames[0])

    # Detect Ball Shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert Positions to Mini-Court Coordinates
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # Initialize Statistics
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    # Highlight Frames for Fast Shots
    highlight_frames = []

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]

        # Compute Shot Statistics
        speed_of_ball_shot, speed_of_opponent, player_shot_ball, opponent_player_id = compute_shot_statistics(
            start_frame, end_frame, ball_mini_court_detections, player_mini_court_detections, mini_court
        )

        # Update Statistics
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

        # Log Statistics to Console
        print(f"Frame {start_frame}: Player {player_shot_ball} Last Shot Speed = {speed_of_ball_shot:.2f} km/h")
        print(f"Frame {start_frame}: Player {opponent_player_id} Movement Speed = {speed_of_opponent:.2f} m/s")

        # Add Frame to Highlights if Shot Speed > Threshold
        if speed_of_ball_shot > 150:  # Example threshold: 150 km/h
            highlight_frames.append(video_frames[start_frame])
            playsound("sounds/fast_shot.mp3")  # Play sound effect

    # Save Statistics to CSV
    player_stats_data_df = pd.DataFrame(player_stats_data)
    player_stats_data_df.to_csv("output_statistics.csv", index=False)
    print("Statistics saved to output_statistics.csv")

    # Plot Trends
    plt.plot(player_stats_data_df['frame_num'], player_stats_data_df['player_1_last_shot_speed'], label="Player 1 Shot Speed")
    plt.plot(player_stats_data_df['frame_num'], player_stats_data_df['player_2_last_shot_speed'], label="Player 2 Shot Speed")
    plt.xlabel("Frame Number")
    plt.ylabel("Shot Speed (km/h)")
    plt.legend()
    plt.title("Shot Speed Trends Over Time")
    plt.savefig("shot_speed_trends.png")
    plt.show()

    # Draw Output
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    # Draw Player Stats
    for i, frame in enumerate(output_video_frames):
        stats = player_stats_data_df.iloc[min(i, len(player_stats_data_df) - 1)]
        cv2.putText(frame, f"Player 1 Avg Shot Speed: {stats['player_1_last_shot_speed']:.2f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Player 2 Avg Shot Speed: {stats['player_2_last_shot_speed']:.2f} km/h", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {i}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save Output Video
    save_video(output_video_frames, "output_videos/output_video.avi")

    # Save Highlight Reel
    highlight_video_writer = cv2.VideoWriter("output_videos/highlights.avi", cv2.VideoWriter_fourcc(*'XVID'), 24,
                                             (video_frames[0].shape[1], video_frames[0].shape[0]))
    for highlight_frame in highlight_frames:
        highlight_video_writer.write(highlight_frame)
    highlight_video_writer.release()


if __name__ == "__main__":
    main()