# BasketBallAnalysisSystem
BasketBallAnalysisSystem is a Python-based project for detecting basketball shots in videos and tracking the ball's trajectory for every successful basket. It leverages YOLO models for object detection and pose estimation, and provides tools for analyzing basketball gameplay footage.

## Features
- Shot Detection: Automatically detects when a basketball shot is made in a video.
- Ball & Rim Detection: Identifies and tracks the basketball and rim using pre-trained YOLO models.
- Trajectory Tracking: Tracks and visualizes the ball's trajectory for each shot attempt.
- Pose Analysis: (Optional) Analyzes player body pose using pose estimation models.
- Video Clipping: Clips and saves video segments of detected shots.
## Project Structure
```
main.py
requirements.txt
model_pt/
    ball_rimV8.pt
    shot_detection.pt
    yolov8n-pose.pt
testing-datasets/
    alan.mp4
    back.mp4
    frame.jpg
    gameplay.mp4
    output.mp4
    side.mp4
utils/
    ball_rim_detection.py
    ball_tracking.py
    BodyAnalyzer.py
    Clipper.py
    intersect.py
    shot_deteciton.py
    video_to_gif.py
    Tracking/
        basketballTracking.ipynb

```
## Getting Started
1. Install dependencies:
```
pip install -r [requirements.txt]
```


2. Prepare models:

- Place your YOLO model weights in the model_pt/ directory.

3. Run the main script:
```
python [main.py]
```
This will process the video at `side.mp4`, detect shots, and track the ball trajectory. The output video will be saved as `output.mp4`.

## Main Components

- `main.py`: Entry point for running the analysis.
- `utils/Clipper.py`: Core logic for video processing, shot detection, and trajectory tracking.
- `utils/ball_tracking.py`: Standalone script for ball tracking.
- `utils/BodyAnalyzer.py`: (Optional) Player pose and movement analysis.

## Requirements

- Python 3.8+
- OpenCV
- numpy
- ultralytics (YOLO)

See `requirements.txt` for the full list.

## Example

After running the main script, you will get an output video with detected shots and ball trajectories visualized.

---

**Note:** Make sure to provide your own video files and trained YOLO models for best results.

For more details, see the code in `main.py` and `utils/Clipper.py`.