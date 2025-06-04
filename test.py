import cv2

def print_video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"❌ Cannot open {path}")
        return

    # Query common properties
    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height      = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s  = frame_count / fps if fps > 0 else 0

    # Read FOURCC code as an int, then decode to 4‐char string
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"Path:        {path}")
    print(f"Resolution:  {int(width)}×{int(height)}")
    print(f"FPS:         {fps:.2f}")
    print(f"Frame count: {int(frame_count)}")
    print(f"Duration:    {duration_s:.2f} s")
    print(f"FOURCC:      {fourcc}")

    cap.release()

# Example
print_video_info("runs/track_output.mp4")
print_video_info("testing-datasets/gameplay.mp4")
