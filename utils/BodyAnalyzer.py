from ultralytics import YOLO
import cv2
import numpy as np

class BodyAnalyzer:
    def __init__(self, video_source, model_path):
        self.video_source = video_source
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.cap = cv2.VideoCapture(self.video_source)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.tracking_data = {'start_time': None, 'end_time': None, 'time_interval': None}
        self.current_frame = None
        self.keypoints = None
        self.elbow_angle = None
        self.knee_angle = None

    def process_frame(self, frame):
        results = self.model(frame, stream=True)
        for r in results:
            self.keypoints = r.keypoints.cpu().numpy()
            self.elbow_angle = self.get_elbow_angle(self.keypoints.xy[0])
            self.knee_angle = self.get_knee_angle(self.keypoints.xy[0])
            self.update_tracking_data()
            self.annotate_frame(frame)
        return frame

    def get_elbow_angle(self, keypoints):
        right_shoulder = keypoints[6][:2]
        right_elbow = keypoints[8][:2]
        right_wrist = keypoints[10][:2]
        return self.calculate_angle(right_shoulder, right_elbow, right_wrist)

    def get_knee_angle(self, keypoints):
        right_hip = keypoints[12][:2]
        right_knee = keypoints[14][:2]
        right_ankle = keypoints[16][:2]
        return self.calculate_angle(right_hip, right_knee, right_ankle)

    @staticmethod
    def calculate_angle(point1, point2, point3):
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        ab = a - b
        cb = c - b
        cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def is_arm_straight(self, keypoints, threshold=20):
        right_shoulder = keypoints[6][:2]
        right_elbow = keypoints[8][:2]
        right_wrist = keypoints[10][:2]
        right_arm_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
        return abs(right_arm_angle - 180) < threshold

    def is_leg_right_angle(self, keypoints, threshold=20):
        right_hip = keypoints[12][:2]
        right_knee = keypoints[14][:2]
        right_ankle = keypoints[16][:2]
        right_leg_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        return abs(right_leg_angle - 90) < threshold

    def draw_line_between_keypoints(self, frame, kp1, kp2, color):
        if int(kp1[0]) != 0 and int(kp1[1]) != 0 and int(kp2[0]) != 0 and int(kp2[1]) != 0:
            cv2.line(frame, (int(kp1[0]), int(kp1[1])), (int(kp2[0]), int(kp2[1])), color, 2)

    def update_tracking_data(self):
        if self.is_leg_right_angle(self.keypoints.xy[0]):
            if self.tracking_data['start_time'] is None:
                self.tracking_data['start_time'] = cv2.getTickCount() / cv2.getTickFrequency()
        if self.is_arm_straight(self.keypoints.xy[0]):
            if self.tracking_data['start_time'] is not None:
                self.tracking_data['end_time'] = cv2.getTickCount() / cv2.getTickFrequency()
                self.tracking_data['time_interval'] = self.tracking_data['end_time'] - self.tracking_data['start_time']
                self.tracking_data['start_time'] = None
                self.tracking_data['end_time'] = None

    def annotate_frame(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        elbow_color = (255, 0, 0)
        knee_color = (0, 255, 0)
        thickness = 2

        right_shoulder = self.keypoints.xy[0][6]
        right_elbow = self.keypoints.xy[0][8]
        right_wrist = self.keypoints.xy[0][10]
        right_hip = self.keypoints.xy[0][12]
        right_knee = self.keypoints.xy[0][14]
        right_ankle = self.keypoints.xy[0][16]

        self.draw_line_between_keypoints(frame, right_shoulder, right_elbow, elbow_color)
        self.draw_line_between_keypoints(frame, right_elbow, right_wrist, elbow_color)
        self.draw_line_between_keypoints(frame, right_hip, right_knee, knee_color)
        self.draw_line_between_keypoints(frame, right_knee, right_ankle, knee_color)

        if str(self.elbow_angle) != "nan":
            cv2.putText(frame, "Arm angle: " + str(self.elbow_angle), (50, 50), font, fontScale, elbow_color, thickness)
        if str(self.knee_angle) != "nan":
            cv2.putText(frame, "Knee angle: " + str(self.knee_angle), (50, 100), font, fontScale, knee_color, thickness)
        if self.tracking_data['time_interval'] is not None:
            cv2.putText(frame, f"Time interval: {self.tracking_data['time_interval']}", (50, 150), font, fontScale, (0, 0, 255), thickness)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                annotated_frame = self.process_frame(frame)
                cv2.imshow('Body Analyzer', annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_source =1  # Webcam or video file path
    model_path = "model_pt/yolov8n-pose.pt"
    analyzer = BodyAnalyzer(video_source, model_path)
    analyzer.run()
