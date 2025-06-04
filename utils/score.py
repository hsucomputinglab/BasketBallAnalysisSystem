from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# load model
pt_path = os.path.join(os.getcwd(), "model_pt/ball_rimV8.pt")
model = YOLO(pt_path)

# show how many class in the model

# inference
video_path= os.path.join(os.getcwd(), "testing-datasets/back.mp4")
class_dict=model.names
#results = model(source, save=True, conf=0.3, show_labels=True, show_boxes=True, show_conf=False, stream=True)
# use class 1 to detect video_path = "testing-datasets/gameplay.mp4"
cap = cv2.VideoCapture(video_path)
fps   = int(cap.get(cv2.CAP_PROP_FPS))
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FPS, 6)
#set cap.fps =60

# prepare a VideoWriter to save the annotated output

#rim form Region of Interest (ROI)
# results = model.predict(source, save=True, conf=0.3, show_labels=True, show_boxes=True, show_conf=False, stream=True, classes=[1])
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')# H.264 codec
# out    = cv2.VideoWriter("runs/track_output.mp4", fourcc, fps, (w, h))
scored = 0  # Initialize scored status
prev_x, prev_y = (0, 0) 
# Initialize the position of the rim
rim_pos = (0, 0, 0, 0)
while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Resize the frame to the model's input size
        if np.all(rim_pos == (0, 0, 0, 0)):
            result = model.predict(frame,save=False, conf=0.3, show_labels=True, show_boxes=True, show_conf=False,classes=[1])
        # If rim position is not set, find the rim in the first frame
            if result[0].boxes is not None:
                rim_mask = (result[0].boxes.cls == 1)
                rim_boxes = result[0].boxes.xyxy[rim_mask].cpu().numpy()
                if len(rim_boxes) > 0:
                    rim_pos = rim_boxes[0]
                    rim_x1, rim_y1, rim_x2, rim_y2 = map(int, rim_pos)
                    rim_x1+= 5  # Adjust the rim position
                    rim_x2-= 15  # Adjust the rim position
                    cv2.rectangle(frame, (rim_x1, rim_y1), (rim_x2, rim_y2), (0, 255, 0), 2)
                    cv2.putText(frame, 'Rim', (rim_x1, rim_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    prev_x, prev_y = (0, 0)  # Reset previous position
                    scored = False  # Reset scored status
            cv2.putText(frame, "Finding Rim...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)   
        else:
            result = model.track(frame, persist=True,classes=[0] ,show=False)[0]
            # Draw the ball tracking results
            if result.boxes is not None:
                ball_mask = (result.boxes.cls == 0)
                ball_boxes = result.boxes.xyxy[ball_mask].cpu().numpy()
                for box in ball_boxes:
                    ball_x1, ball_y1, ball_x2, ball_y2 = map(int, box)
                    cv2.rectangle(frame, (ball_x1, ball_y1), (ball_x2, ball_y2), (255, 0, 0), 2)
                    cv2.putText(frame, 'Ball', (ball_x1, ball_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    # Calculate the center of the ball
                    center_x = (ball_x1 + ball_x2) // 2
                    center_y = (ball_y1 + ball_y2) // 2
                    # Draw a circle at the center of the ball
                    cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)
                    
                    # Calculate the distance from the previous position
                    if prev_x != 0 and prev_y != 0:
                        dx = center_x - prev_x
                        dy = center_y - prev_y
                    

                    # 判斷是否從上往下進入籃框區域

                in_roi_now = (rim_x1 <= center_x <= rim_x2) and (rim_y1 <= center_y <= rim_y2)
                # Define the rim's ROI as a rectangle
                in_roi_before = (rim_x1 <= prev_x <= rim_x2) and (rim_y1 <= prev_y <= rim_y2)
                if not in_roi_before and in_roi_now and dy > 0:
                    scored += 1
            cv2.rectangle(frame, (rim_x1, rim_y1), (rim_x2, rim_y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Rim', (rim_x1, rim_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Prev: ({prev_x}, {prev_y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Curr: ({center_x}, {center_y})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Rim: ({rim_x1}, {rim_y1}, {rim_x2}, {rim_y2})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            prev_x, prev_y = center_x, center_y
            cv2.putText(frame, f"Scored: {scored}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Original Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    else:
        print("End of video or cannot read the frame.")
        break
    # Write the annotated frame to the output video 
    # for result in results:
    #     if result.boxes is not None:
    #         rim_mask = (result.boxes.cls == 1)
    #         rim_boxes = result.boxes.xyxy[rim_mask].cpu().numpy()
    #         for box in rim_boxes:
    #             rim_x1, rim_y1, rim_x2, rim_y2 = map(int, box)
    #             cv2.rectangle(frame, (rim_x1, rim_y1), (rim_x2, rim_y2), (0, 255, 0), 2)
    #             cv2.putText(frame, 'Rim', (rim_x1, rim_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         ball_mask = (result.boxes.cls == 0)
    #         ball_boxes = result.boxes.xyxy[ball_mask].cpu().numpy()
    #         for box in ball_boxes:
    #             ball_x1, ball_y1, ball_x2, ball_y2 = map(int, box)
    #             cv2.rectangle(frame, (ball_x1, ball_y1), (ball_x2, ball_y2), (255, 0, 0), 2)
    #             cv2.putText(frame, 'Ball', (ball_x1, ball_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
    #             # Calculate the center of the ball
    #             center_x = (ball_x1 + ball_x2) // 2
    #             center_y = (ball_y1 + ball_y2) // 2
    #             # Draw a circle at the center of the ball
    #             cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)
                
    #             # Calculate the distance from the previous position
    #             if prev_x != 0 and prev_y != 0:
    #                 dx = center_x - prev_x
    #                 dy = center_y - prev_y
    #             else:
    #                 prev_x, prev_y = center_x, center_y
    #                 dx, dy = 0, 0

            
                    
    #             # 判斷是否從上往下進入籃框區域
    #             if not scored:
    #                 in_roi_now = (rim_x1 <= center_x <= rim_x2) and (rim_y1 <= center_y <= rim_y2)
    #                 # Define the rim's ROI as a rectangle
    #                 in_roi_before = (rim_x1 <= prev_x <= rim_x2) and (rim_y1 <= prev_y <= rim_y2)
    #                 if not in_roi_before and in_roi_now and dy > 0:
    #                     scored = True
    #                     cv2.putText(frame, "SCORED!", (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

cap.release()

cv2.destroyAllWindows()
#out.release()
    
    