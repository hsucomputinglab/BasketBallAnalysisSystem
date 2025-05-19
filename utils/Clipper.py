from ultralytics import YOLO
import os
import cv2
import numpy as np
from utils.intersect import intersect

class Clipper:
    def __init__(self, video_path, ball_rim_model_path, shot_model_path, output_path):
        self.video_path = video_path
        self.ball_rim_model_path = ball_rim_model_path
        self.shot_model_path = shot_model_path
        self.output_path = output_path

        self.ball_rim_model = YOLO(self.ball_rim_model_path)
        self.shot_model = YOLO(self.shot_model_path)

        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.rim_bounding_box = None
        self.rim_location = None
        self.ball_location = None
        self.previous_ball_location = None
        self.ball_tracking = []
        self.ball_tracking_history = []
        self.shot_detected = False
        self.frame_to_count = self.fps * 2
        self.clipping_start = None
        self.clipping_end = None
        self.clipping_list = []

        self.standard_line = None
        self.color = tuple(np.random.randint(0, 255, size=(1, 3), dtype="uint8").squeeze().tolist())

    def detect_rim(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                rim = self.ball_rim_model.predict(frame, classes=[1], max_det=1)
                if rim[0].boxes.__len__() != 0:
                    self.rim_bounding_box = rim[0].boxes.data[0].cpu().numpy().astype(int)
                    rim_location = rim[0].boxes.xywh[0].cpu().numpy().astype(int)
                    self.rim_location = {"x": rim_location[0], "y": rim_location[1], "w": rim_location[2], "h": rim_location[3]}
                    print(f"get rim_location: {self.rim_location}")
                    break
            else:
                break
        self.standard_line = [
            (self.rim_location["x"] - self.rim_location["w"] // 2, self.rim_location["y"]),
            (self.rim_location["x"] + self.rim_location["w"] // 2, self.rim_location["y"])
        ]

    def process_frame(self, frame, frame_index):
        if self.shot_detected:
            self.frame_count += 1

        ball = self.ball_rim_model.predict(frame, classes=[0], max_det=1, conf=0.5)
        shot = self.shot_model.predict(frame, max_det=1, conf=0.5)

        if shot[0].boxes.__len__() != 0:
            self.shot_detected = True
            cv2.putText(frame, "Shot Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.clipping_start = frame_index

        if ball[0].boxes.__len__() != 0:
            self.ball_location = ball[0].boxes.xywh[0].cpu().numpy().astype(int)
            if self.shot_detected:
                self.ball_tracking.append(self.ball_location)

        self.draw_ball_tracking(frame)
        self.draw_standard_line(frame)
        self.check_ball_in_rim(frame)

        annotated_frame = ball[0].plot(conf=False, labels=False, boxes=False)
        return annotated_frame

    def draw_ball_tracking(self, frame):
        if len(self.ball_tracking_history) > 0:
            for history in self.ball_tracking_history:
                for i in range(len(history["ball_tracking"]) - 1):
                    cv2.line(frame, tuple(history["ball_tracking"][i][:2]), tuple(history["ball_tracking"][i + 1][:2]), history["color"], 2)

        if len(self.ball_tracking) > 1 and self.frame_count <= self.frame_to_count:
            for i in range(len(self.ball_tracking) - 1):
                cv2.line(frame, tuple(self.ball_tracking[i][:2]), tuple(self.ball_tracking[i + 1][:2]), self.color, 2)
        elif self.frame_count > self.frame_to_count:
            self.ball_tracking = []
            self.shot_detected = False
            self.frame_count = 0

    def draw_standard_line(self, frame):
        cv2.line(frame, self.standard_line[0], self.standard_line[1], (0, 255, 0), 2)
        cv2.rectangle(frame, (self.rim_bounding_box[0], self.rim_bounding_box[1]), (self.rim_bounding_box[2], self.rim_bounding_box[3]), (0, 255, 0), 2)

    def check_ball_in_rim(self, frame):
        trajectory = None
        if self.previous_ball_location is not None and self.ball_location is not None:
            trajectory = [tuple(self.previous_ball_location[:2]), tuple(self.ball_location[:2])]
            cv2.line(frame, tuple(self.previous_ball_location[:2]), tuple(self.ball_location[:2]), (255, 0, 0), 2)

        if trajectory is not None and intersect(trajectory, self.standard_line):
            cv2.putText(frame, "In", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            history = {
                "ball_tracking": self.ball_tracking,
                "color": self.color,
            }
            self.ball_tracking_history.append(history)
            self.frame_count = 0
            self.ball_tracking = []
            self.shot_detected = False
            self.color = tuple(np.random.randint(0, 255, size=(1, 3), dtype="uint8").squeeze().tolist())

    def run(self):
        self.detect_rim()
        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = 0

        while self.cap.isOpened():
            success, frame = self.cap.read()
            frame_index = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if success:
                annotated_frame = self.process_frame(frame, frame_index)
                cv2.imshow("Ball Tracking", annotated_frame)

                if self.ball_location is not None:
                    self.previous_ball_location = self.ball_location
                    self.ball_location = None

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     video_path = os.path.join(os.getcwd(), "testing-datasets/side.mp4")
#     ball_rim_model_path = os.path.join(os.getcwd(), "model_pt/ball_rimV8.pt")
#     shot_model_path = os.path.join(os.getcwd(), "model_pt/shot_detection.pt")
#     output_path = os.path.join(os.getcwd(), "output.mp4")

#     clipper = Clipper(video_path, ball_rim_model_path, shot_model_path, output_path)
#     clipper.run()