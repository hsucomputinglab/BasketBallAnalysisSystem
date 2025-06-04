

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML


# Train the model
results = model.train(data="BasketBall_Game-7/data.yaml", epochs=500, imgsz=640)