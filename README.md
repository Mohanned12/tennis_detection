# AI Tennis Analysis System using Computer Vision

This project is an AI-powered tennis analysis system that uses computer vision and deep learning techniques to track players, detect the tennis ball, analyze shots, and generate match statistics automatically from video footage.

The system combines object detection, player tracking, court detection, and performance analytics to create a smart sports analysis platform.

---

## Overview

The goal of this project is to analyze tennis matches automatically using artificial intelligence and computer vision.

The system is capable of:

* Detecting players and the tennis ball
* Tracking movement across video frames
* Detecting ball shots
* Measuring shot speed
* Measuring player movement speed
* Generating match statistics
* Creating highlight moments for fast shots

The project demonstrates how AI can be applied in sports analytics and performance tracking.

---

## Features

* Real-time player tracking
* Tennis ball detection and tracking
* Court line detection
* Shot speed calculation
* Player movement analysis
* Match statistics generation
* Highlight clip generation
* Mini-court visualization
* CSV statistics export
* Performance trend visualization

---

## Technologies Used

* Python
* OpenCV
* YOLO
* PyTorch
* NumPy
* Pandas
* Matplotlib

---

## Project Structure

```bash id="pq2g7x"
AI-Tennis-Analysis/
│
├── main.py                         # Main application
├── trackers/                       # Player and ball tracking modules
├── mini_court/                     # Mini-court visualization
├── court_line_detector/            # Court keypoint detection
├── models/                         # AI models
├── utils/                          # Utility functions
├── constants.py                    # Project constants
├── output_videos/                  # Generated videos
├── tracker_stubs/                  # Saved detections
└── README.md                       # Project documentation
```

---

## How the System Works

### 1. Video Processing

The system reads a tennis match video and extracts all frames.

```python id="2f1g5q"
video_frames = read_video(input_video_path)
```



---

### 2. Player and Ball Detection

YOLO-based trackers are used to detect:

* Tennis players
* Tennis ball

```python id="f4n8sj"
player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
```



---

### 3. Court Detection

The system detects court keypoints and creates a mini-court representation for accurate measurements.

```python id="9s3jla"
court_line_detector = CourtLineDetector(court_model_path)
```



---

### 4. Shot Analysis

The AI calculates:

* Ball speed
* Player movement speed
* Shot intervals

```python id="7v2mwd"
speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
```



---

### 5. Statistics Generation

The system stores player statistics and exports them to a CSV file.

```python id="h2r8ko"
player_stats_data_df.to_csv("output_statistics.csv", index=False)
```



---

### 6. Highlight Generation

Fast shots are automatically detected and added to a highlight reel.

```python id="q8u6ne"
if speed_of_ball_shot > 150:
```



---

## Output Features

The system generates:

* Annotated tennis match video
* Shot speed statistics
* Player movement statistics
* Match analytics CSV
* Shot speed graphs
* Highlight videos

---

## Installation

### Clone the repository

```bash id="c5d8zr"
git clone https://github.com/yourusername/ai-tennis-analysis.git
cd ai-tennis-analysis
```

---

## Install Dependencies

```bash id="y5w2lt"
pip install opencv-python torch ultralytics pandas matplotlib playsound
```

---

## Running the Project

Run the project using:

```bash id="l9x1oe"
python main.py
```

The system will process the tennis match video and generate analysis outputs automatically.

---

## Learning Outcomes

Through this project, I learned:

* Sports analytics using AI
* Object tracking with YOLO
* Video processing with OpenCV
* Player and ball tracking
* Motion analysis
* Computer vision pipelines
* Data visualization and analytics

---

## Future Improvements

Possible future improvements include:

* Real-time live match analysis
* Player identification system
* AI-generated match summaries
* More advanced shot classification
* Web dashboard for analytics
* Cloud deployment
* Multi-camera support

---

## Author

Haddad Mouhaned
Master's Student in Data Science

---

## License

This project was created for educational and research purposes.
