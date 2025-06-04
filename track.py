from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("model_pt/player.pt")

# Open the video file
video_path = "testing-datasets/gameplay.mp4"
cap = cv2.VideoCapture(video_path)
fps   = int(cap.get(cv2.CAP_PROP_FPS))
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# prepare a VideoWriter to save the annotated output
# Change the codec to H.264 (avc1)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')# H.264 codec
out    = cv2.VideoWriter("runs/track_output.mp4", fourcc, fps, (w, h))

track_history = defaultdict(list)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    result = model.track(frame, persist=True,tracker="bytetrack.yaml")[0]

    # draw circles under tracked objects instead of default rectangles
    if result.boxes and result.boxes.is_track:
        boxes     = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        for (x, y, w, h), tid in zip(boxes, track_ids):
            # Draw a circle under the object (center bottom of bbox)
            center_x = int(x)
            center_y = int(y + h // 2)
            # Draw an ellipse slightly below the center bottom of bbox
            axes_length = (20, 10)  # (major axis, minor axis)
            angle = 0
            start_angle = 0
            end_angle = 360
            cv2.ellipse(
                frame,
                (center_x, center_y ),
                axes_length,
                angle,
                start_angle,
                end_angle,
                (0, 255, 0),
                thickness=3
            )
            # Draw the track ID
            cv2.putText(
                frame,
                str(tid),
                (center_x-15, center_y  + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (128, 128, 128),
                2
            )
            # draw history polylines
            hist = track_history[tid]
            hist.append((center_x, center_y))
            if len(hist) > 30:
                hist.pop(0)
            pts = np.array(hist, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, (0, 0, 0), 2)
    # write to disk
    # Change the codec to H.264 (avc1)
    out.write(frame)

    # if you do happen to have a GUI, you can still show it:
    # cv2.imshow("Track", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
out.release()
# cv2.destroyAllWindows()
print("👉 Saved tracking video as runs/track_output.mp4")