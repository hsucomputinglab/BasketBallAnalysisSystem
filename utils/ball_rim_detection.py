from ultralytics import YOLO
import os

# load model
pt_path = os.path.join(os.getcwd(), "model_pt/ball_rimV8.pt")
model = YOLO(pt_path)

# show how many class in the model
print(model.names)

# inference
source = os.path.join(os.getcwd(), "testing-datasets/back.mp4")

#results = model(source, save=True, conf=0.3, show_labels=True, show_boxes=True, show_conf=False, stream=True)
model.predict(source,stream=True)  # generator of Results objects
# use class 1 to detect rim form Region of Interest (ROI)
# results = model.predict(source, save=True, conf=0.3, show_labels=True, show_boxes=True, show_conf=False, stream=True, classes=[1])
# for result in results:
#     if result.boxes.cls = = 1:
#         print(result.boxes.xyxy)  # bounding boxes for class 1 (rim)
#         print(result.boxes.xywhn)  # bounding boxes for class 1 (rim)
