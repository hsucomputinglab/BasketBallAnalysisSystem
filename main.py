import os
from utils.Clipper import Clipper

if __name__ == "__main__":
    video_path = os.path.join(os.getcwd(), "testing-datasets/side.mp4")
    ball_rim_model_path = os.path.join(os.getcwd(), "model_pt/ball_rimV8.pt")
    shot_model_path = os.path.join(os.getcwd(), "model_pt/shot_detection.pt")
    output_path = os.path.join(os.getcwd(), "output.mp4")

    clipper = Clipper(video_path, ball_rim_model_path, shot_model_path, output_path)
    clipper.run()