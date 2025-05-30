import cv2
import time
from onvif import ONVIFCamera

def ptz_control(ip, port=80, user='nckusport', pwd='Ncku1234', pan=0.0, tilt=0.0, zoom=0.0, timeout=0.5):
    """
    Pan/Tilt/Zoom control via ONVIF.
      ip, port, user, pwd: camera credentials
      pan, tilt, zoom: velocities in range [-1.0,1.0]
      timeout: seconds to move before stop
    """
    # connect to camera
    cam = ONVIFCamera(ip, port, user, pwd)
    media = cam.create_media_service()
    ptz = cam.create_ptz_service()

    # get profile token
    profile = media.GetProfiles()[0]
    token = profile.token

    # build continuous move request
    req = ptz.create_type('ContinuousMove')
    req.ProfileToken = token
    req.Velocity = {
        'PanTilt': {'x': pan, 'y': tilt},   # pan: left/right, tilt: up/down
        'Zoom':    {'x': zoom}
    }

    # send move command
    ptz.ContinuousMove(req)
    time.sleep(timeout)

    # stop motion
    stop_req = ptz.create_type('Stop')
    stop_req.ProfileToken = token
    stop_req.PanTilt = True
    stop_req.Zoom = True
    ptz.Stop(stop_req)
# adjust these values to control direction/speed
# pan: positive = right, negative = left
# tilt: positive = up,   negative = down
# ptz_control(
#         ip='10.30.3.28', port=80,
#         user='nckusport', pwd='Ncku1234',
#         pan=-0.5, tilt=-0.1, timeout=0.5
#     )
# print("Camera control command sent successfully.")


def build_rtsp_url(base_ip, stream_num):
    return f'rtsp://nckusport:Ncku1234@{base_ip}.{stream_num}/stream0'

def open_stream(base_ip, idx):
    url = build_rtsp_url(base_ip, idx)
    pzt_url= f'{base_ip}.{idx}'
    print(f"🔄 Switching to Camera {idx} → {url}")
    return cv2.VideoCapture(url),url ,pzt_url

def main():
    num_cameras = 34
    base_ip = '10.30.3'
    idx = 28  # 起始攝影機 index 1~34

    stream, selected_camera_url,ptz_url = open_stream(base_ip, idx)

    while True:
        ret, frame = stream.read()
        if not ret:
            print("❌ Failed to read from stream.")
            break

        frame = cv2.resize(frame, (640, 480))

        cv2.imshow('Video Stream', frame)

        key = cv2.waitKey(1) 
        print(f"Key pressed: {key} ({chr(key & 0xFF) if key & 0xFF < 256 else 'Non-ASCII'})")
        if (key & 0xFF) == ord('q'):
            break
        elif (key & 0xFF) == ord('s'):  # 上一個攝影機

            idx = (idx - 1) % num_cameras 
            if idx < 1:
                idx = num_cameras - 1
            stream.release()
            stream, selected_camera_url,ptz_url = open_stream(base_ip, idx)
        elif (key & 0xFF)  == ord('w'):  # 下一個攝影機
            idx = (idx + 1) % num_cameras
            if idx < 1:
                idx = 1
            stream.release()
            stream, selected_camera_url,ptz_url = open_stream(base_ip, idx)
        elif key ==82:#up
            # 控制攝影機角度
            print(f"Controlling Camera {idx} at {selected_camera_url} with pan and tilt adjustments.")
            ptz_control(ip=ptz_url, tilt=0.3)
        elif key== 84:  # down
            ptz_control(ip=ptz_url, tilt=-0.3)
        elif key== 81: # left
            ptz_control(ip=ptz_url,pan=-0.3)
        elif key== 83: # right
            ptz_control(ip=ptz_url,pan=0.3)
        elif key == 61:  # zoom in
            ptz_control(ip=ptz_url, zoom=0.1)
        elif key == 45:  # zoom out
            ptz_control(ip=ptz_url, zoom=-0.1)
    # Clean up
    stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
