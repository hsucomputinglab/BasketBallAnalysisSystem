import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile

# 預設使用 yolov8n.pt 模型（速度快，檔案小）
@st.cache_resource
def load_model():
    return YOLO("model_pt/ball_rimV8.pt")  # 你也可以改成 yolov8s.pt / yolov5s.pt

model = load_model()

st.title("🔍 YOLO 物件偵測 - 即時圖片分析")

uploaded_file = st.file_uploader("請上傳圖片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 顯示原始圖片
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="原始圖片", use_column_width=True)

    # 將圖片轉換為 OpenCV 格式
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.write("📦 偵測中...")
    results = model(image_bgr)[0]

    # 把框畫上
    annotated = results.plot()  # 繪製偵測結果（回傳的是 BGR 格式）

    # 顯示結果圖像
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="📍 偵測結果", use_column_width=True)

    # 顯示偵測到的類別
    detected = [model.model.names[int(cls)] for cls in results.boxes.cls]
    if detected:
        st.success(f"✅ 偵測到物件：{', '.join(detected)}")
    else:
        st.warning("⚠️ 沒有偵測到物件")
